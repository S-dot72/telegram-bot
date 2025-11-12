"""
Production bot avec Machine Learning et vérification automatique des résultats.
- 20 signaux par jour à partir de 9h UTC
- Signal toutes les 5 minutes avec délai de 3 minutes avant entrée
- ML pour améliorer la confiance des signaux
- Vérification automatique WIN/LOSE
- Support multi-utilisateurs
- ✅ Correction du décalage horaire (force UTC réel, même sur Railway UTC-5)
"""

import os, json, asyncio
from datetime import datetime, timedelta, timezone, time as dtime
import pytz
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

# --- Configuration horaires (EN UTC) ---
START_HOUR_UTC = 9
SIGNAL_INTERVAL_MIN = 5
DELAY_BEFORE_ENTRY_MIN = 3
NUM_SIGNALS_PER_DAY = 20

# --- Database et scheduler EN UTC ---
engine = create_engine(DB_URL, connect_args={'check_same_thread': False})
sched = AsyncIOScheduler(timezone=pytz.UTC)  # Force UTC

# --- ML Predictor et Auto Verifier ---
ml_predictor = MLSignalPredictor()
auto_verifier = None

# --- Charger les meilleurs paramètres si présents ---
BEST_PARAMS = {}
if os.path.exists(BEST_PARAMS_FILE):
    try:
        with open(BEST_PARAMS_FILE, 'r') as f:
            BEST_PARAMS = json.load(f)
    except Exception:
        BEST_PARAMS = {}

TWELVE_TS_URL = 'https://api.twelvedata.com/time_series'

# Cache global pour les données OHLC
ohlc_cache = {}
CACHE_DURATION_SECONDS = 60

# --- Fonctions utilitaires ---

def get_utc_now():
    """Retourne l'heure actuelle en UTC"""
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
        if (current_time - cached_time).total_seconds() < CACHE_DURATION_SECONDS:
            print(f"💾 Utilisation du cache pour {pair}")
            return cached_data
    print(f"🌐 Appel API pour {pair}...")
    df = fetch_ohlc_td(pair, interval, outputsize)
    ohlc_cache[cache_key] = (df, current_time)
    return df

def persist_signal(payload):
    q = text("""INSERT INTO signals (pair,direction,reason,ts_enter,ts_send,confidence,payload_json)
                VALUES (:pair,:direction,:reason,:ts_enter,:ts_send,:confidence,:payload)""")
    with engine.begin() as conn:
        conn.execute(q, payload)

def generate_daily_schedule_for_today():
    """Génère le planning du jour strictement en UTC réel"""
    now_utc = get_utc_now()
    today_utc = now_utc.date()
    # Création correcte en UTC
    first_send_time_utc = datetime.combine(today_utc, dtime(START_HOUR_UTC, 0, 0, tzinfo=timezone.utc))

    # Si déjà passé 9h UTC aujourd'hui, planifier pour demain
    if now_utc >= first_send_time_utc + timedelta(hours=2):
        tomorrow_utc = today_utc + timedelta(days=1)
        first_send_time_utc = datetime.combine(tomorrow_utc, dtime(START_HOUR_UTC, 0, 0, tzinfo=timezone.utc))

    schedule = []
    active_pairs = PAIRS[:2]
    for i in range(NUM_SIGNALS_PER_DAY):
        send_time = first_send_time_utc + timedelta(minutes=i * SIGNAL_INTERVAL_MIN)
        entry_time = send_time + timedelta(minutes=DELAY_BEFORE_ENTRY_MIN)
        pair = active_pairs[i % len(active_pairs)]
        schedule.append({'pair': pair, 'send_time': send_time, 'entry_time': entry_time})

    print("📅 Planning (UTC):")
    print(f"   Date: {schedule[0]['send_time'].strftime('%Y-%m-%d')}")
    print(f"   Premier: {schedule[0]['send_time'].strftime('%H:%M')} UTC")
    print(f"   Dernier: {schedule[-1]['send_time'].strftime('%H:%M')} UTC")
    return schedule

def format_signal_message(pair, direction, entry_time, confidence, reason):
    direction_text = "BUY" if direction == "CALL" else "SELL"
    entry_time = entry_time.astimezone(timezone.utc)
    gale1 = entry_time + timedelta(minutes=5)
    gale2 = entry_time + timedelta(minutes=10)
    msg = (
        f"📊 SIGNAL — {pair} ({entry_time.strftime('%Y-%m-%d')})\n\n"
        f"Entrée (UTC): {entry_time.strftime('%H:%M')}\n\n"
        f"Direction: {direction_text}\n\n"
        f"     Gale 1: {gale1.strftime('%H:%M')}\n"
        f"     Gale 2: {gale2.strftime('%H:%M')}\n\n"
        f"Confiance: {int(confidence*100)}%"
    )
    return msg

# --- Commandes Telegram ---

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    try:
        with engine.begin() as conn:
            existing = conn.execute(text("SELECT user_id FROM subscribers WHERE user_id = :uid"),
                                    {"uid": user_id}).fetchone()
            if existing:
                await update.message.reply_text("✅ Vous êtes déjà abonné aux signaux.")
            else:
                conn.execute(text("INSERT INTO subscribers (user_id, username) VALUES (:uid, :uname)"),
                             {"uid": user_id, "uname": username})
                await update.message.reply_text(
                    f"✅ Bienvenue !\n\n📊 {NUM_SIGNALS_PER_DAY} signaux par jour à partir de {START_HOUR_UTC}h UTC\n"
                    f"⏱️ Un signal toutes les {SIGNAL_INTERVAL_MIN} minutes\n"
                    f"⏳ Entrée {DELAY_BEFORE_ENTRY_MIN} minutes après l’envoi"
                )
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_result(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 2:
        await update.message.reply_text('Usage: /result <ts_enter_iso> <WIN|LOSE>')
        return
    ts, res = args[0], args[1].upper()
    if res not in ('WIN','LOSE'):
        await update.message.reply_text('Result must be WIN or LOSE')
        return
    with engine.begin() as conn:
        conn.execute(text("UPDATE signals SET result=:r, ts_result=:t WHERE ts_enter=:ts"),
                     {'r':res,'t':get_utc_now().isoformat(),'ts':ts})
    await update.message.reply_text('✅ Résultat mis à jour')

async def cmd_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pair = PAIRS[0]
    entry_time = get_utc_now() + timedelta(minutes=DELAY_BEFORE_ENTRY_MIN)
    await update.message.reply_text("🔍 Génération test de signal...")
    await send_pre_signal(pair, entry_time, context.application)
    await update.message.reply_text(f"✅ Test terminé ({pair}) - Entrée {entry_time.strftime('%H:%M:%S')} UTC")

# --- Envoi de signaux ---

async def send_pre_signal(pair, entry_time, app):
    now = get_utc_now()
    entry_time = entry_time.astimezone(timezone.utc)
    print(f"\n🔄 SIGNAL {pair} - {now.strftime('%H:%M:%S')} UTC - Entrée {entry_time.strftime('%H:%M:%S')} UTC")
    try:
        params = BEST_PARAMS.get(pair, {})
        df = get_cached_ohlc(pair, TIMEFRAME_M1, outputsize=400)
        df = compute_indicators(df, ema_fast=params.get('ema_fast',8),
                                ema_slow=params.get('ema_slow',21),
                                rsi_len=params.get('rsi',14),
                                bb_len=params.get('bb',20))
        base_signal = rule_signal(df)
        if not base_signal:
            print("⏭️ Pas de signal base.")
            return
        ml_signal, ml_conf = ml_predictor.predict_signal(df, base_signal)
        if ml_signal is None or ml_conf < 0.70:
            print(f"❌ Signal rejeté (confiance {ml_conf:.1%})")
            return
        reason = f"Signal ML validé ({ml_conf:.1%})"
        payload = {
            'pair': pair, 'direction': ml_signal, 'reason': reason,
            'ts_enter': entry_time.isoformat(), 'ts_send': now.isoformat(),
            'confidence': ml_conf, 'payload': json.dumps({'pair': pair})
        }
        persist_signal(payload)
        with engine.connect() as conn:
            user_ids = [r[0] for r in conn.execute(text("SELECT user_id FROM subscribers")).fetchall()]
        msg = format_signal_message(pair, ml_signal, entry_time, ml_conf, reason)
        for uid in user_ids:
            try:
                await app.bot.send_message(chat_id=uid, text=msg)
            except Exception as e:
                print(f"❌ Envoi à {uid} échoué: {e}")
        print(f"✅ Signal {pair} envoyé ({ml_signal}, {ml_conf:.1%})")
    except Exception as e:
        print(f"❌ Erreur signal: {e}")

# --- Scheduler ---

async def schedule_today_signals(app, sched):
    now_utc = get_utc_now()
    if now_utc.weekday() > 4:
        print("🏖️ Weekend - pas de signaux")
        return
    for job in sched.get_jobs():
        if job.id and job.id.startswith("signal_"):
            job.remove()
    daily = generate_daily_schedule_for_today()
    for item in daily:
        if item['send_time'] > now_utc:
            sched.add_job(
                send_pre_signal, 'date', run_date=item['send_time'],
                args=[item['pair'], item['entry_time'], app],
                id=f"signal_{item['pair']}_{item['send_time'].strftime('%H%M')}"
            )
    print(f"✅ {len(daily)} signaux planifiés (UTC)")

# --- DB ---

def ensure_db():
    sql = open('db_schema.sql').read()
    with engine.begin() as conn:
        for stmt in sql.split(';'):
            if stmt.strip():
                conn.execute(text(stmt.strip()))

# --- Main ---

async def main():
    global auto_verifier
    print("\n🤖 DÉMARRAGE DU BOT (Heure UTC):", get_utc_now())
    ensure_db()
    auto_verifier = AutoResultVerifier(engine, TWELVEDATA_API_KEY)
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler('start', cmd_start))
    app.add_handler(CommandHandler('result', cmd_result))
    app.add_handler(CommandHandler('test', cmd_test))

    sched.start()
    await schedule_today_signals(app, sched)
    sched.add_job(schedule_today_signals, 'cron', hour=8, minute=55, args=[app, sched], id='daily_schedule')

    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    print("✅ BOT OPÉRATIONNEL - UTC vérifié")
    while True:
        await asyncio.sleep(1)

if __name__ == '__main__':
    asyncio.run(main())
