"""
Bot de trading simplifié — version UTC corrigée
- Signaux à 9h UTC (fixe, peu importe le fuseau du serveur)
- Vérification automatique après chaque signal
"""

import os, json, asyncio
from datetime import datetime, timedelta, timezone
import requests
import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import create_engine, text
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from config import *
from utils import compute_indicators, rule_signal
from ml_predictor import MLSignalPredictor
from auto_verifier import AutoResultVerifier

# =====================================================
# Configuration UTC globale
# =====================================================

os.environ['TZ'] = 'UTC'  # Force le fuseau UTC
START_HOUR_UTC = 9
SIGNAL_INTERVAL_MIN = 5
DELAY_BEFORE_ENTRY_MIN = 3
NUM_SIGNALS_PER_DAY = 20

engine = create_engine(DB_URL, connect_args={'check_same_thread': False})
sched = AsyncIOScheduler(timezone=timezone.utc)
ml_predictor = MLSignalPredictor()
auto_verifier = None

BEST_PARAMS = {}
if os.path.exists(BEST_PARAMS_FILE):
    try:
        with open(BEST_PARAMS_FILE, 'r') as f:
            BEST_PARAMS = json.load(f)
    except:
        pass

TWELVE_TS_URL = 'https://api.twelvedata.com/time_series'
ohlc_cache = {}


# =====================================================
# Fonctions utilitaires
# =====================================================

def get_utc_now():
    return datetime.now(timezone.utc)

def fetch_ohlc_td(pair, interval, outputsize=300):
    params = {'symbol': pair, 'interval': interval, 'outputsize': outputsize,
              'apikey': TWELVEDATA_API_KEY, 'format':'JSON'}
    r = requests.get(TWELVE_TS_URL, params=params, timeout=10)
    r.raise_for_status()
    j = r.json()
    if 'values' not in j:
        raise RuntimeError(f"TwelveData error: {j}")
    df = pd.DataFrame(j['values'])[::-1].reset_index(drop=True)
    for col in ['open','high','low','close']:
        if col in df.columns:
            df[col] = df[col].astype(float)
    if 'volume' in df.columns:
        df['volume'] = df['volume'].astype(float)
    df.index = pd.to_datetime(df['datetime'])
    return df

def get_cached_ohlc(pair, interval, outputsize=300):
    cache_key = f"{pair}_{interval}"
    current_time = get_utc_now()
    if cache_key in ohlc_cache:
        cached_data, cached_time = ohlc_cache[cache_key]
        if (current_time - cached_time).total_seconds() < 60:
            return cached_data
    df = fetch_ohlc_td(pair, interval, outputsize)
    ohlc_cache[cache_key] = (df, current_time)
    return df

def persist_signal(payload):
    q = text("""INSERT INTO signals (pair,direction,reason,ts_enter,ts_send,confidence,payload_json)
                VALUES (:pair,:direction,:reason,:ts_enter,:ts_send,:confidence,:payload)""")
    with engine.begin() as conn:
        conn.execute(q, payload)


# =====================================================
# Commandes Telegram
# =====================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    try:
        with engine.begin() as conn:
            existing = conn.execute(text("SELECT user_id FROM subscribers WHERE user_id = :uid"),
                                    {"uid": user_id}).fetchone()
            if existing:
                await update.message.reply_text("✅ Vous êtes déjà abonné aux signaux !")
            else:
                conn.execute(text("INSERT INTO subscribers (user_id, username) VALUES (:uid, :uname)"),
                             {"uid": user_id, "uname": username})
                await update.message.reply_text(
                    f"✅ Bienvenue !\n\n"
                    f"📊 {NUM_SIGNALS_PER_DAY} signaux/jour à {START_HOUR_UTC}h UTC\n"
                    f"⏱️ Signal toutes les {SIGNAL_INTERVAL_MIN} min\n"
                    f"🔍 Vérification après chaque signal\n\n"
                    f"Commandes:\n"
                    f"/test - Tester un signal\n"
                    f"/stats - Voir les stats\n"
                    f"/verify - Vérifier les résultats"
                )
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        with engine.connect() as conn:
            total = conn.execute(text('SELECT COUNT(*) FROM signals')).scalar()
            wins = conn.execute(text("SELECT COUNT(*) FROM signals WHERE result='WIN'")).scalar()
            losses = conn.execute(text("SELECT COUNT(*) FROM signals WHERE result='LOSE'")).scalar()
            subs = conn.execute(text('SELECT COUNT(*) FROM subscribers')).scalar()
        
        verified = wins + losses
        winrate = (wins/verified*100) if verified > 0 else 0
        
        msg = f"📊 **Statistiques**\n\n"
        msg += f"Total signaux: {total}\n"
        msg += f"Vérifiés: {verified}\n"
        msg += f"✅ Réussis: {wins}\n"
        msg += f"❌ Échoués: {losses}\n"
        msg += f"📈 Win rate: {winrate:.1f}%\n"
        msg += f"👥 Abonnés: {subs}"
        
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_verify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    try:
        msg = await update.message.reply_text("🔍 Vérification en cours...")
        
        auto_verifier.add_admin(chat_id)
        if not auto_verifier.bot:
            auto_verifier.set_bot(context.application.bot)
        
        await auto_verifier.verify_pending_signals()
        
        try:
            await msg.delete()
        except:
            pass
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text("🔍 Test de signal...")
        pair = PAIRS[0]
        entry_time_utc = get_utc_now() + timedelta(minutes=DELAY_BEFORE_ENTRY_MIN)
        await send_pre_signal(pair, entry_time_utc, context.application)
        await update.message.reply_text("✅ Test terminé!")
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")


# =====================================================
# Envoi de signaux
# =====================================================

async def send_pre_signal(pair, entry_time_utc, app):
    now_utc = get_utc_now()
    print(f"\n📤 Signal {pair} - {now_utc.strftime('%H:%M:%S')} UTC")
    
    try:
        params = BEST_PARAMS.get(pair, {})
        df = get_cached_ohlc(pair, TIMEFRAME_M1, outputsize=400)
        df = compute_indicators(df, ema_fast=params.get('ema_fast',8),
                                ema_slow=params.get('ema_slow',21),
                                rsi_len=params.get('rsi',14),
                                bb_len=params.get('bb',20))
        base_signal = rule_signal(df)
        
        if not base_signal:
            print("⏭️ Pas de signal")
            return
        
        ml_signal, ml_conf = ml_predictor.predict_signal(df, base_signal)
        if ml_signal is None or ml_conf < 0.70:
            print(f"❌ Rejeté ({ml_conf:.1%})")
            return
        
        # Sauvegarder
        payload = {
            'pair': pair, 'direction': ml_signal, 'reason': f'ML {ml_conf:.1%}',
            'ts_enter': entry_time_utc.isoformat(), 
            'ts_send': now_utc.isoformat(),
            'confidence': ml_conf, 
            'payload': json.dumps({'pair': pair})
        }
        persist_signal(payload)
        
        # Envoyer
        with engine.connect() as conn:
            user_ids = [r[0] for r in conn.execute(text("SELECT user_id FROM subscribers")).fetchall()]
        
        direction_text = "BUY" if ml_signal == "CALL" else "SELL"
        gale1 = entry_time_utc + timedelta(minutes=5)
        gale2 = entry_time_utc + timedelta(minutes=10)
        
        msg = (
            f"📊 SIGNAL — {pair}\n\n"
            f"Entrée (UTC): {entry_time_utc.strftime('%H:%M')}\n\n"
            f"Direction: {direction_text}\n\n"
            f"     Gale 1: {gale1.strftime('%H:%M')}\n"
            f"     Gale 2: {gale2.strftime('%H:%M')}\n\n"
            f"Confiance: {int(ml_conf*100)}%"
        )
        
        for uid in user_ids:
            try:
                await app.bot.send_message(chat_id=uid, text=msg)
            except Exception as e:
                print(f"❌ Envoi à {uid}: {e}")
        
        print(f"✅ Signal envoyé ({ml_signal}, {ml_conf:.1%})")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")


# =====================================================
# Scheduler — UTC only
# =====================================================

def generate_schedule():
    print(f"\n📅 Planning UTC:")
    schedule = []
    active_pairs = PAIRS[:2]
    
    for i in range(NUM_SIGNALS_PER_DAY):
        utc_send_hour = START_HOUR_UTC + (i * SIGNAL_INTERVAL_MIN) / 60
        utc_entry = utc_send_hour + DELAY_BEFORE_ENTRY_MIN / 60
        
        schedule.append({
            'pair': active_pairs[i % len(active_pairs)],
            'hour': int(utc_send_hour),
            'minute': int((utc_send_hour % 1) * 60),
            'entry_utc_hour': int(utc_entry),
            'entry_utc_minute': int((utc_entry % 1) * 60)
        })
    
    return schedule


async def schedule_daily_signals(app, sched):
    now_utc = get_utc_now()
    if now_utc.weekday() > 4:
        print("🏖️ Weekend")
        return
    
    # Supprimer anciens jobs
    for job in sched.get_jobs():
        if job.id and job.id.startswith("signal_"):
            job.remove()
    
    schedule = generate_schedule()
    
    for i, item in enumerate(schedule):
        entry_time_utc = datetime(
            now_utc.year, now_utc.month, now_utc.day,
            item['entry_utc_hour'], item['entry_utc_minute'],
            tzinfo=timezone.utc
        )
        
        sched.add_job(
            send_pre_signal,
            'cron',
            hour=item['hour'],
            minute=item['minute'],
            args=[item['pair'], entry_time_utc, app],
            id=f"signal_{i}"
        )
    
    print(f"✅ {len(schedule)} signaux planifiés\n")


# =====================================================
# Base de données
# =====================================================

def ensure_db():
    sql = open('db_schema.sql').read()
    with engine.begin() as conn:
        for stmt in sql.split(';'):
            if stmt.strip():
                conn.execute(text(stmt.strip()))


# =====================================================
# Main
# =====================================================

async def main():
    global auto_verifier
    
    print("\n" + "="*60)
    print("🤖 BOT DE TRADING (UTC MODE)")
    print("="*60)
    print(f"🌍 Heure UTC: {get_utc_now().strftime('%H:%M:%S')}")
    print("="*60 + "\n")
    
    ensure_db()
    auto_verifier = AutoResultVerifier(engine, TWELVEDATA_API_KEY)
    
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler('start', cmd_start))
    app.add_handler(CommandHandler('stats', cmd_stats))
    app.add_handler(CommandHandler('verify', cmd_verify))
    app.add_handler(CommandHandler('test', cmd_test))

    sched.start()
    await schedule_daily_signals(app, sched)
    
    # Replanification quotidienne à 8h55 UTC
    sched.add_job(
        schedule_daily_signals,
        'cron',
        hour=8,
        minute=55,
        args=[app, sched],
        id='daily_schedule'
    )
    
    # Vérification auto toutes les 15 min
    sched.add_job(
        auto_verifier.verify_pending_signals,
        'interval',
        minutes=15,
        id='auto_verify'
    )

    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    
    print("✅ BOT DÉMARRÉ (en UTC)\n")
    
    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
        sched.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
