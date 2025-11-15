"""
Bot de trading - Signaux séquentiels après vérification
- Démarre à 9h AM heure d'Haïti (UTC-5)
- Envoie signal → attend vérification → envoie résultat → nouveau signal
- 20 signaux max par jour
"""

import os, json, asyncio
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
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

# Configuration
HAITI_TZ = ZoneInfo("America/Port-au-Prince")  # UTC-5
START_HOUR_HAITI = 9  # 9h AM heure d'Haïti
DELAY_BEFORE_ENTRY_MIN = 3
VERIFICATION_WAIT_MIN = 15  # Attendre 15 min après entrée avant vérification
NUM_SIGNALS_PER_DAY = 20

engine = create_engine(DB_URL, connect_args={'check_same_thread': False})
sched = AsyncIOScheduler(timezone=HAITI_TZ)  # Scheduler en heure d'Haïti
ml_predictor = MLSignalPredictor()
auto_verifier = None
signal_queue_running = False

BEST_PARAMS = {}
if os.path.exists(BEST_PARAMS_FILE):
    try:
        with open(BEST_PARAMS_FILE, 'r') as f:
            BEST_PARAMS = json.load(f)
    except:
        pass

TWELVE_TS_URL = 'https://api.twelvedata.com/time_series'
ohlc_cache = {}

def get_haiti_now():
    """Retourne l'heure actuelle en timezone Haïti"""
    return datetime.now(HAITI_TZ)

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
        if (current_time - cached_time).total_seconds() < 60:
            return cached_data
    df = fetch_ohlc_td(pair, interval, outputsize)
    ohlc_cache[cache_key] = (df, current_time)
    return df

def persist_signal(payload):
    q = text("""INSERT INTO signals (pair,direction,reason,ts_enter,ts_send,confidence,payload_json)
                VALUES (:pair,:direction,:reason,:ts_enter,:ts_send,:confidence,:payload)""")
    with engine.begin() as conn:
        result = conn.execute(q, payload)
        return result.lastrowid

# --- Commandes Telegram ---

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
                    f"📊 Jusqu'à {NUM_SIGNALS_PER_DAY} signaux/jour\n"
                    f"⏰ Début: {START_HOUR_HAITI}h00 AM (Haïti)\n"
                    f"🔄 Signal → Vérification → Résultat → Nouveau signal\n\n"
                    f"Commandes:\n"
                    f"/test - Tester un signal\n"
                    f"/stats - Voir les stats\n"
                    f"/verify - Vérifier manuellement"
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
        
        try:
            await auto_verifier.verify_pending_signals()
            await msg.edit_text("✅ Vérification terminée!")
        except Exception as e:
            print(f"❌ Erreur lors de la vérification: {e}")
            import traceback
            traceback.print_exc()
            await msg.edit_text(f"⚠️ Erreur de vérification: {str(e)[:100]}")
            
    except Exception as e:
        print(f"❌ Erreur cmd_verify: {e}")
        import traceback
        traceback.print_exc()
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text("🔍 Test de signal...")
        pair = PAIRS[0]
        entry_time_haiti = get_haiti_now() + timedelta(minutes=DELAY_BEFORE_ENTRY_MIN)
        await send_pre_signal(pair, entry_time_haiti, context.application)
        await update.message.reply_text("✅ Test terminé!")
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

# --- Envoi de signaux ---

async def send_pre_signal(pair, entry_time_haiti, app):
    """Envoie un signal avec horaire en heure d'Haïti"""
    now_haiti = get_haiti_now()
    print(f"\n📤 Tentative signal {pair} - {now_haiti.strftime('%H:%M:%S')} (Haïti)")
    
    try:
        params = BEST_PARAMS.get(pair, {})
        df = get_cached_ohlc(pair, TIMEFRAME_M1, outputsize=400)
        
        if df is None or len(df) < 50:
            print("❌ Pas assez de données")
            return None
            
        df = compute_indicators(df, ema_fast=params.get('ema_fast',8),
                                ema_slow=params.get('ema_slow',21),
                                rsi_len=params.get('rsi',14),
                                bb_len=params.get('bb',20))
        base_signal = rule_signal(df)
        
        if not base_signal:
            print("⏭️ Pas de signal de base (conditions techniques non remplies)")
            return None
        
        print(f"📊 Signal de base détecté: {base_signal}")
        
        ml_signal, ml_conf = ml_predictor.predict_signal(df, base_signal)
        if ml_signal is None or ml_conf < 0.70:
            print(f"❌ Rejeté par ML (confiance: {ml_conf:.1%} < 70%)")
            return None
        
        # Convertir en UTC pour la DB
        entry_time_utc = entry_time_haiti.astimezone(timezone.utc)
        
        # Sauvegarder
        payload = {
            'pair': pair, 
            'direction': ml_signal, 
            'reason': f'ML {ml_conf:.1%}',
            'ts_enter': entry_time_utc.isoformat(), 
            'ts_send': get_utc_now().isoformat(),
            'confidence': ml_conf, 
            'payload': json.dumps({'pair': pair})
        }
        signal_id = persist_signal(payload)
        
        # Récupérer les abonnés
        with engine.connect() as conn:
            user_ids = [r[0] for r in conn.execute(text("SELECT user_id FROM subscribers")).fetchall()]
        
        direction_text = "BUY" if ml_signal == "CALL" else "SELL"
        
        # Calculer les gales en heure d'Haïti
        gale1_haiti = entry_time_haiti + timedelta(minutes=5)
        gale2_haiti = entry_time_haiti + timedelta(minutes=10)
        
        msg = (
            f"📊 SIGNAL — {pair}\n\n"
            f"🕐 Entrée: {entry_time_haiti.strftime('%H:%M')} (Haïti)\n\n"
            f"📈 Direction: {direction_text}\n\n"
            f"🔄 Gale 1: {gale1_haiti.strftime('%H:%M')}\n"
            f"🔄 Gale 2: {gale2_haiti.strftime('%H:%M')}\n\n"
            f"💪 Confiance: {int(ml_conf*100)}%"
        )
        
        for uid in user_ids:
            try:
                await app.bot.send_message(chat_id=uid, text=msg)
            except Exception as e:
                print(f"❌ Envoi à {uid}: {e}")
        
        print(f"✅ Signal envoyé ({ml_signal}, {ml_conf:.1%})")
        print(f"   Entrée: {entry_time_haiti.strftime('%H:%M')} (Haïti)")
        
        return signal_id
        
    except Exception as e:
        print(f"❌ Erreur signal: {e}")
        import traceback
        traceback.print_exc()
        return None

async def verify_signal_manual(signal_id, app):
    """Vérification manuelle simplifiée d'un signal"""
    try:
        print(f"🔍 Vérification manuelle signal ID:{signal_id}")
        
        # Récupérer le signal
        with engine.connect() as conn:
            signal = conn.execute(
                text("SELECT pair, direction, ts_enter FROM signals WHERE id = :sid"),
                {"sid": signal_id}
            ).fetchone()
        
        if not signal:
            print(f"❌ Signal {signal_id} non trouvé")
            return False
        
        pair, direction, ts_enter_str = signal
        ts_enter = datetime.fromisoformat(ts_enter_str)
        
        print(f"📊 Vérification {pair} {direction} (entrée: {ts_enter})")
        
        # Récupérer les données OHLC
        df = get_cached_ohlc(pair, TIMEFRAME_M1, outputsize=100)
        
        if df is None or len(df) == 0:
            print("❌ Pas de données OHLC")
            return False
        
        # Trouver le prix d'entrée (à l'heure ts_enter)
        df_filtered = df[df.index >= ts_enter]
        
        if len(df_filtered) == 0:
            print("⏳ Pas encore de données après l'heure d'entrée")
            return False
        
        entry_price = df_filtered.iloc[0]['close']
        
        # Vérifier les 3 bougies suivantes (signal initial + 2 gales)
        max_candles = min(3, len(df_filtered))
        results = []
        
        for i in range(max_candles):
            if i >= len(df_filtered):
                break
                
            candle = df_filtered.iloc[i]
            open_price = entry_price if i == 0 else df_filtered.iloc[i]['open']
            close_price = candle['close']
            
            if direction == 'CALL':
                win = close_price > open_price
            else:  # PUT
                win = close_price < open_price
            
            results.append(win)
            print(f"  Bougie {i+1}: {'WIN' if win else 'LOSE'} (open={open_price:.5f}, close={close_price:.5f})")
            
            if win:
                # Gagné !
                gale_level = i  # 0=signal initial, 1=gale1, 2=gale2
                with engine.begin() as conn:
                    conn.execute(
                        text("UPDATE signals SET result='WIN', gale_level=:gale WHERE id=:sid"),
                        {"gale": gale_level, "sid": signal_id}
                    )
                print(f"✅ WIN au niveau {gale_level}")
                return True
        
        # Perdu après 3 tentatives
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE signals SET result='LOSE', gale_level=2 WHERE id=:sid"),
                {"sid": signal_id}
            )
        print(f"❌ LOSE après {max_candles} tentatives")
        return True
        
    except Exception as e:
        print(f"❌ Erreur verify_signal_manual: {e}")
        import traceback
        traceback.print_exc()
        return False
    """Envoie le résultat de vérification aux abonnés"""
    try:
        with engine.connect() as conn:
            signal = conn.execute(
                text("SELECT pair, direction, result FROM signals WHERE id = :sid"),
                {"sid": signal_id}
            ).fetchone()
            
            if not signal or not signal[2]:  # Pas de résultat
                return
            
            pair, direction, result = signal
            user_ids = [r[0] for r in conn.execute(text("SELECT user_id FROM subscribers")).fetchall()]
        
        # Message simple et clair
        if result == "WIN":
            emoji = "✅"
            status = "GAGNÉ"
        else:
            emoji = "❌"
            status = "PERDU"
        
        msg = f"{emoji} Résultat: {status}\n{pair} - {direction}"
        
        for uid in user_ids:
            try:
                await app.bot.send_message(chat_id=uid, text=msg)
            except Exception as e:
                print(f"❌ Envoi résultat à {uid}: {e}")
        
        print(f"📤 Résultat envoyé: {status}")
        
    except Exception as e:
        print(f"❌ Erreur envoi résultat: {e}")

# --- File de signaux séquentielle ---

async def process_signal_queue(app):
    """Traite les signaux séquentiellement: signal → vérification → résultat → nouveau signal"""
    global signal_queue_running
    
    if signal_queue_running:
        print("⚠️ File déjà en cours")
        return
    
    signal_queue_running = True
    
    try:
        now_haiti = get_haiti_now()
        
        print(f"\n{'='*60}")
        print(f"🚀 DÉBUT DE LA SESSION DE TRADING")
        print(f"{'='*60}")
        print(f"🕐 Heure actuelle (Haïti): {now_haiti.strftime('%H:%M:%S')}")
        print(f"🌍 Heure actuelle (UTC): {get_utc_now().strftime('%H:%M:%S')}")
        print(f"📊 Max {NUM_SIGNALS_PER_DAY} signaux aujourd'hui")
        print(f"{'='*60}\n")
        
        active_pairs = PAIRS[:2]  # EUR/USD et GBP/USD
        
        for i in range(NUM_SIGNALS_PER_DAY):
            pair = active_pairs[i % len(active_pairs)]
            
            print(f"\n{'─'*60}")
            print(f"📍 SIGNAL {i+1}/{NUM_SIGNALS_PER_DAY} - {pair}")
            print(f"{'─'*60}")
            
            # 1. Envoyer le signal
            now_haiti = get_haiti_now()
            entry_time_haiti = now_haiti + timedelta(minutes=DELAY_BEFORE_ENTRY_MIN)
            
            print(f"⏰ Tentative d'envoi du signal à {now_haiti.strftime('%H:%M:%S')}")
            
            # Réessayer jusqu'à 3 fois si pas de signal
            signal_id = None
            for attempt in range(3):
                signal_id = await send_pre_signal(pair, entry_time_haiti, app)
                if signal_id is not None:
                    break
                print(f"⚠️ Tentative {attempt + 1}/3 échouée, nouvelle tentative dans 30s...")
                await asyncio.sleep(30)
            
            if signal_id is None:
                print(f"❌ Aucun signal valide après 3 tentatives pour {pair}, passage à la paire suivante")
                continue
            
            # 2. Attendre le temps d'entrée + temps de vérification
            verification_time_haiti = entry_time_haiti + timedelta(minutes=VERIFICATION_WAIT_MIN)
            now_haiti = get_haiti_now()
            wait_seconds = (verification_time_haiti - now_haiti).total_seconds()
            
            if wait_seconds > 0:
                wait_minutes = wait_seconds / 60
                print(f"⏳ Attente de {wait_minutes:.1f} min jusqu'à {verification_time_haiti.strftime('%H:%M')}")
                await asyncio.sleep(wait_seconds)
            
            # 3. Vérifier le signal
            print(f"🔍 Vérification du signal ID:{signal_id}...")
            
            verification_success = False
            try:
                # Essayer d'abord avec auto_verifier
                await auto_verifier.verify_pending_signals()
                verification_success = True
                print(f"✅ Vérification auto réussie")
            except Exception as e:
                print(f"⚠️ Erreur auto_verifier: {e}")
                # Essayer la vérification manuelle
                try:
                    verification_success = await verify_signal_manual(signal_id, app)
                    print(f"✅ Vérification manuelle réussie")
                except Exception as e2:
                    print(f"❌ Erreur vérification manuelle: {e2}")
                    import traceback
                    traceback.print_exc()
            
            if not verification_success:
                # Notifier les utilisateurs de l'erreur
                with engine.connect() as conn:
                    user_ids = [r[0] for r in conn.execute(text("SELECT user_id FROM subscribers")).fetchall()]
                
                error_msg = f"⚠️ Impossible de vérifier le signal {pair}\nUtilisez /verify dans 5 minutes"
                for uid in user_ids:
                    try:
                        await app.bot.send_message(chat_id=uid, text=error_msg)
                    except:
                        pass
                continue
            
            # 4. Envoyer le résultat
            await send_verification_result(signal_id, app)
            
            print(f"✅ Cycle {i+1} terminé\n")
            
            # Petite pause avant le prochain signal
            await asyncio.sleep(30)
        
        print(f"\n{'='*60}")
        print(f"🏁 SESSION TERMINÉE")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"❌ Erreur dans la file: {e}")
        import traceback
        traceback.print_exc()
    finally:
        signal_queue_running = False

# --- Scheduler ---

async def start_daily_signals(app):
    """Démarre la session quotidienne à 9h AM Haïti"""
    now_haiti = get_haiti_now()
    
    # Vérifier si c'est un jour de semaine
    if now_haiti.weekday() > 4:  # Samedi=5, Dimanche=6
        print(f"🏖️ Weekend - Pas de trading")
        return
    
    print(f"\n📅 Démarrage session - {now_haiti.strftime('%A %d %B %Y, %H:%M:%S')}")
    asyncio.create_task(process_signal_queue(app))

def ensure_db():
    """Crée/met à jour la base de données"""
    try:
        sql = open('db_schema.sql').read()
        with engine.begin() as conn:
            for stmt in sql.split(';'):
                if stmt.strip():
                    conn.execute(text(stmt.strip()))
        
        # Ajouter la colonne gale_level si elle n'existe pas
        with engine.begin() as conn:
            try:
                conn.execute(text("ALTER TABLE signals ADD COLUMN gale_level INTEGER DEFAULT 0"))
                print("✅ Colonne gale_level ajoutée")
            except Exception as e:
                # La colonne existe déjà, c'est normal
                if "duplicate column" not in str(e).lower():
                    print(f"ℹ️ gale_level: {e}")
                    
    except Exception as e:
        print(f"⚠️ Erreur DB: {e}")

# --- Main ---

async def main():
    global auto_verifier
    
    now_haiti = get_haiti_now()
    now_utc = get_utc_now()
    
    print("\n" + "="*60)
    print("🤖 BOT DE TRADING - HAÏTI")
    print("="*60)
    print(f"🇭🇹 Heure Haïti: {now_haiti.strftime('%H:%M:%S %Z')}")
    print(f"🌍 Heure UTC: {now_utc.strftime('%H:%M:%S %Z')}")
    print(f"⏰ Début quotidien: {START_HOUR_HAITI}h00 AM (Haïti)")
    print(f"📊 Signaux: Séquentiels après vérification")
    print("="*60 + "\n")
    
    ensure_db()
    auto_verifier = AutoResultVerifier(engine, TWELVEDATA_API_KEY)
    
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler('start', cmd_start))
    app.add_handler(CommandHandler('stats', cmd_stats))
    app.add_handler(CommandHandler('verify', cmd_verify))
    app.add_handler(CommandHandler('test', cmd_test))

    sched.start()
    
    # Démarrer immédiatement si on est après 9h AM et avant 18h
    if (now_haiti.hour >= START_HOUR_HAITI and now_haiti.hour < 18 and 
        now_haiti.weekday() <= 4 and not signal_queue_running):
        print("🚀 Démarrage immédiat de la session")
        asyncio.create_task(process_signal_queue(app))
    
    # Job quotidien à 9h00 AM heure d'Haïti
    sched.add_job(
        start_daily_signals,
        'cron',
        hour=START_HOUR_HAITI,
        minute=0,
        timezone=HAITI_TZ,
        args=[app],
        id='daily_signals_haiti'
    )

    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    
    bot_info = await app.bot.get_me()
    print(f"✅ BOT ACTIF: @{bot_info.username}")
    print(f"📍 Prochaine session: Demain {START_HOUR_HAITI}h00 AM (Haïti)\n")
    
    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 Arrêt du bot...")
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
        sched.shutdown()

if __name__ == '__main__':
    asyncio.run(main())
