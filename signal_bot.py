"""
Bot de trading 
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
HAITI_TZ = ZoneInfo("America/Port-au-Prince")
START_HOUR_HAITI = 9
DELAY_BEFORE_ENTRY_MIN = 3
VERIFICATION_WAIT_MIN = 15
NUM_SIGNALS_PER_DAY = 20

engine = create_engine(DB_URL, connect_args={'check_same_thread': False})
sched = AsyncIOScheduler(timezone=HAITI_TZ)
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
    return datetime.now(HAITI_TZ)

def get_utc_now():
    return datetime.now(timezone.utc)

def is_forex_open():
    """Vérifie si le marché Forex est ouvert"""
    now_utc = get_utc_now()
    weekday = now_utc.weekday()
    hour = now_utc.hour
    
    print(f"[FOREX CHECK] UTC: {now_utc.strftime('%A %H:%M')} | Weekday: {weekday} | Hour: {hour}")
    
    # Samedi
    if weekday == 5:
        print(f"[FOREX CHECK] ❌ FERMÉ (Samedi)")
        return False
    
    # Dimanche avant 22h UTC
    if weekday == 6 and hour < 22:
        print(f"[FOREX CHECK] ❌ FERMÉ (Dimanche avant 22h)")
        return False
    
    # Vendredi après 22h UTC
    if weekday == 4 and hour >= 22:
        print(f"[FOREX CHECK] ❌ FERMÉ (Vendredi après 22h)")
        return False
    
    print(f"[FOREX CHECK] ✅ OUVERT")
    return True

def fetch_ohlc_td(pair, interval, outputsize=300):
    if not is_forex_open():
        raise RuntimeError("Marché Forex fermé")
    
    params = {'symbol': pair, 'interval': interval, 'outputsize': outputsize,
    'apikey': TWELVEDATA_API_KEY, 'format':'JSON'}
    r = requests.get(TWELVE_TS_URL, params=params, timeout=10)
    r.raise_for_status()
    j = r.json()
    
    if 'code' in j and j['code'] == 429:
        raise RuntimeError(f"Limite API atteinte: {j.get('message', 'Unknown')}")
    
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
    if not is_forex_open():
        return None
    
    cache_key = f"{pair}_{interval}"
    current_time = get_utc_now()
    
    if cache_key in ohlc_cache:
        cached_data, cached_time = ohlc_cache[cache_key]
        if (current_time - cached_time).total_seconds() < 60:
            return cached_data
    
    try:
        df = fetch_ohlc_td(pair, interval, outputsize)
        ohlc_cache[cache_key] = (df, current_time)
        return df
    except RuntimeError as e:
        print(f"⚠️ Cache OHLC: {e}")
        return None

def persist_signal(payload):
    q = text("""INSERT INTO signals (pair,direction,reason,ts_enter,ts_send,confidence,payload_json)
    VALUES (:pair,:direction,:reason,:ts_enter,:ts_send,:confidence,:payload)""")
    with engine.begin() as conn:
        result = conn.execute(q, payload)
    return result.lastrowid

def cleanup_weekend_signals():
    try:
        with engine.begin() as conn:
            result = conn.execute(text("""
                UPDATE signals 
                SET result = 'LOSE', 
                    reason = 'Signal créé pendant week-end (marché fermé)'
                WHERE result IS NULL 
                AND (
                    CAST(strftime('%w', ts_enter) AS INTEGER) = 0 OR
                    CAST(strftime('%w', ts_enter) AS INTEGER) = 6
                )
            """))
            
            count = result.rowcount
            if count > 0:
                print(f"🧹 {count} signaux du week-end nettoyés")
            return count
    except Exception as e:
        print(f"⚠️ Erreur cleanup: {e}")
        return 0

def force_cleanup_weekend():
    try:
        with engine.begin() as conn:
            result = conn.execute(text("""
                UPDATE signals 
                SET result = 'LOSE', 
                    reason = 'Signal créé/vérifié pendant marché fermé'
                WHERE (result IS NULL OR result != 'LOSE')
                AND (
                    CAST(strftime('%w', ts_enter) AS INTEGER) = 6 OR
                    CAST(strftime('%w', ts_enter) AS INTEGER) = 0
                )
            """))
            
            count = result.rowcount
            print(f"🧹 {count} signaux du week-end marqués comme LOSE")
            return count
    except Exception as e:
        print(f"❌ Erreur force cleanup: {e}")
        return 0

def ensure_db():
    try:
        sql = open('db_schema.sql').read()
        with engine.begin() as conn:
            for stmt in sql.split(';'):
                if stmt.strip():
                    conn.execute(text(stmt.strip()))

        with engine.begin() as conn:
            result = conn.execute(text("PRAGMA table_info(signals)")).fetchall()
            existing_cols = {row[1] for row in result}
            
            if 'gale_level' not in existing_cols:
                conn.execute(text("ALTER TABLE signals ADD COLUMN gale_level INTEGER DEFAULT 0"))
            
            if 'timeframe' not in existing_cols:
                conn.execute(text("ALTER TABLE signals ADD COLUMN timeframe INTEGER DEFAULT 5"))
            
            if 'max_gales' not in existing_cols:
                conn.execute(text("ALTER TABLE signals ADD COLUMN max_gales INTEGER DEFAULT 2"))
            
            if 'winning_attempt' not in existing_cols:
                conn.execute(text("ALTER TABLE signals ADD COLUMN winning_attempt TEXT"))
            
            if 'reason' not in existing_cols:
                conn.execute(text("ALTER TABLE signals ADD COLUMN reason TEXT"))
            
            print("✅ Base de données prête")
        
        cleanup_weekend_signals()

    except Exception as e:
        print(f"⚠️ Erreur DB: {e}")

# === COMMANDES TELEGRAM ===

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
                    f"🔄 Lundi-Vendredi (marché Forex)\n\n"
                    f"Commandes:\n"
                    f"/stats - Statistiques\n"
                    f"/status - État du bot\n"
                    f"/testsignal - Forcer un signal de test"
                )
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        with engine.connect() as conn:
            total = conn.execute(text('SELECT COUNT(*) FROM signals')).scalar()
            wins = conn.execute(text("SELECT COUNT(*) FROM signals WHERE result='WIN'")).scalar()
            losses = conn.execute(text("SELECT COUNT(*) FROM signals WHERE result='LOSE'")).scalar()
            pending = conn.execute(text("SELECT COUNT(*) FROM signals WHERE result IS NULL")).scalar()
            subs = conn.execute(text('SELECT COUNT(*) FROM subscribers')).scalar()

        verified = wins + losses
        winrate = (wins/verified*100) if verified > 0 else 0

        msg = f"📊 **Statistiques**\n\n"
        msg += f"Total signaux: {total}\n"
        msg += f"Vérifiés: {verified}\n"
        msg += f"✅ Réussis: {wins}\n"
        msg += f"❌ Échoués: {losses}\n"
        msg += f"⏳ En attente: {pending}\n"
        msg += f"📈 Win rate: {winrate:.1f}%\n"
        msg += f"👥 Abonnés: {subs}"
        
        await update.message.reply_text(msg)

    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        now_haiti = get_haiti_now()
        now_utc = get_utc_now()
        forex_open = is_forex_open()
        
        msg = f"🤖 **État du Bot**\n\n"
        msg += f"🇭🇹 Haïti: {now_haiti.strftime('%a %H:%M:%S')}\n"
        msg += f"🌍 UTC: {now_utc.strftime('%a %H:%M:%S')}\n"
        msg += f"📈 Forex: {'🟢 OUVERT' if forex_open else '🔴 FERMÉ'}\n"
        msg += f"🔄 Session: {'✅ Active' if signal_queue_running else '⏸️ Inactive'}\n\n"
        
        if not forex_open:
            if now_utc.weekday() == 6 and now_utc.hour < 22:
                msg += "⏰ Réouverture: Dimanche 22h UTC\n"
            elif now_utc.weekday() == 5:
                msg += "⏰ Réouverture: Dimanche 22h UTC\n"
            else:
                msg += "⏰ Réouverture: Lundi 00h UTC\n"
        
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_test_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Force le démarrage d'une session de test"""
    try:
        global signal_queue_running
        
        if signal_queue_running:
            await update.message.reply_text("⚠️ Une session est déjà en cours")
            return
        
        msg = await update.message.reply_text("🚀 Démarrage session de test...")
        
        # Force le démarrage
        app = context.application
        asyncio.create_task(process_signal_queue(app))
        
        await msg.edit_text("✅ Session de test lancée !")
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

# === ENVOI DE SIGNAUX ===

async def send_pre_signal(pair, entry_time_haiti, app):
    if not is_forex_open():
        print("[SIGNAL] 🏖️ Marché fermé - Pas de signal")
        return None
    
    now_haiti = get_haiti_now()
    print(f"\n[SIGNAL] 📤 Tentative {pair} - {now_haiti.strftime('%H:%M:%S')} (Haïti)")

    try:
        params = BEST_PARAMS.get(pair, {})
        df = get_cached_ohlc(pair, TIMEFRAME_M1, outputsize=400)

        if df is None or len(df) < 50:
            print("[SIGNAL] ❌ Pas de données")
            return None
        
        df = compute_indicators(df, ema_fast=params.get('ema_fast',8),
                                ema_slow=params.get('ema_slow',21),
                                rsi_len=params.get('rsi',14),
                                bb_len=params.get('bb',20))
        base_signal = rule_signal(df)
        
        if not base_signal:
            print("[SIGNAL] ⏭️ Pas de signal de base")
            return None
        
        ml_signal, ml_conf = ml_predictor.predict_signal(df, base_signal)
        if ml_signal is None or ml_conf < 0.70:
            print(f"[SIGNAL] ❌ Rejeté par ML ({ml_conf:.1%})")
            return None
        
        entry_time_utc = entry_time_haiti.astimezone(timezone.utc)
        
        payload = {
            'pair': pair, 'direction': ml_signal, 'reason': f'ML {ml_conf:.1%}',
            'ts_enter': entry_time_utc.isoformat(), 'ts_send': get_utc_now().isoformat(),
            'confidence': ml_conf, 'payload': json.dumps({'pair': pair})
        }
        signal_id = persist_signal(payload)
        
        with engine.connect() as conn:
            user_ids = [r[0] for r in conn.execute(text("SELECT user_id FROM subscribers")).fetchall()]
        
        direction_text = "BUY" if ml_signal == "CALL" else "SELL"
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
                print(f"[SIGNAL] ❌ Envoi à {uid}: {e}")
        
        print(f"[SIGNAL] ✅ Envoyé ({ml_signal}, {ml_conf:.1%})")
        return signal_id

    except Exception as e:
        print(f"[SIGNAL] ❌ Erreur: {e}")
        return None

async def send_verification_briefing(signal_id, app):
    """Envoie un briefing détaillé après vérification"""
    try:
        with engine.connect() as conn:
            signal = conn.execute(
                text("SELECT pair, direction, result, gale_level, confidence FROM signals WHERE id = :sid"),
                {"sid": signal_id}
            ).fetchone()

        if not signal or not signal[2]:
            print(f"[BRIEFING] ⚠️ Signal #{signal_id} non vérifié")
            return

        pair, direction, result, gale_level, confidence = signal
        
        with engine.connect() as conn:
            user_ids = [r[0] for r in conn.execute(text("SELECT user_id FROM subscribers")).fetchall()]
        
        if result == "WIN":
            emoji = "✅"
            status = "GAGNÉ"
            
            if gale_level == 0:
                attempt_text = "🎯 Signal initial"
            elif gale_level == 1:
                attempt_text = "🔄 Gale 1"
            elif gale_level == 2:
                attempt_text = "🔄 Gale 2"
            else:
                attempt_text = f"🔄 Gale {gale_level}"
        else:
            emoji = "❌"
            status = "PERDU"
            attempt_text = "Aucune des 3 tentatives"
        
        direction_emoji = "📈" if direction == "CALL" else "📉"
        
        briefing = (
            f"{emoji} **BRIEFING SIGNAL**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{direction_emoji} Paire: **{pair}**\n"
            f"📊 Direction: **{direction}**\n"
            f"💪 Confiance: {int(confidence*100)}%\n\n"
            f"🎲 Résultat: **{status}**\n"
            f"✨ Gagné par: {attempt_text}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
        
        sent_count = 0
        for uid in user_ids:
            try:
                await app.bot.send_message(chat_id=uid, text=briefing)
                sent_count += 1
            except Exception as e:
                print(f"[BRIEFING] ❌ Envoi à {uid}: {e}")
        
        print(f"[BRIEFING] ✅ Envoyé à {sent_count} abonnés: {status}")

    except Exception as e:
        print(f"[BRIEFING] ❌ Erreur: {e}")

async def send_daily_report(app):
    """Rapport final après tous les signaux"""
    try:
        print("\n[RAPPORT] 📊 Génération...")
        
        now_utc = datetime.now(timezone.utc)
        start_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        end_utc = start_utc + timedelta(days=1)
        
        with engine.connect() as conn:
            query = text("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN result = 'LOSE' THEN 1 ELSE 0 END) as losses,
                    SUM(CASE WHEN result = 'WIN' AND gale_level = 0 THEN 1 ELSE 0 END) as win_initial,
                    SUM(CASE WHEN result = 'WIN' AND gale_level = 1 THEN 1 ELSE 0 END) as win_gale1,
                    SUM(CASE WHEN result = 'WIN' AND gale_level = 2 THEN 1 ELSE 0 END) as win_gale2
                FROM signals
                WHERE ts_enter >= :start AND ts_enter < :end
            """)
            
            stats = conn.execute(query, {
                "start": start_utc.isoformat(),
                "end": end_utc.isoformat()
            }).fetchone()
            
            signals_query = text("""
                SELECT pair, direction, result, gale_level
                FROM signals
                WHERE ts_enter >= :start AND ts_enter < :end
                ORDER BY ts_enter ASC
            """)
            
            signals_list = conn.execute(signals_query, {
                "start": start_utc.isoformat(),
                "end": end_utc.isoformat()
            }).fetchall()
            
            user_ids = [r[0] for r in conn.execute(text("SELECT user_id FROM subscribers")).fetchall()]
        
        if not stats or stats[0] == 0:
            print("[RAPPORT] ⚠️ Aucun signal aujourd'hui")
            return
        
        total, wins, losses, win_initial, win_gale1, win_gale2 = stats
        verified = wins + losses
        winrate = (wins / verified * 100) if verified > 0 else 0
        
        now_haiti = get_haiti_now()
        
        report = (
            f"📊 **RAPPORT QUOTIDIEN**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📅 Date: {now_haiti.strftime('%d/%m/%Y')}\n"
            f"🕐 Heure: {now_haiti.strftime('%H:%M')} (Haïti)\n\n"
            f"📈 **PERFORMANCE GLOBALE**\n"
            f"• Total signaux: {total}\n"
            f"• ✅ Gagnés: {wins}\n"
            f"• ❌ Perdus: {losses}\n"
            f"• 📊 Win rate: **{winrate:.1f}%**\n\n"
        )
        
        if wins > 0:
            report += (
                f"🎯 **DÉTAIL DES GAINS**\n"
                f"• Signal initial: {win_initial}\n"
                f"• Gale 1: {win_gale1}\n"
                f"• Gale 2: {win_gale2}\n\n"
            )
        
        if len(signals_list) > 0:
            report += f"📋 **HISTORIQUE ({len(signals_list)} signaux)**\n\n"
            
            for i, sig in enumerate(signals_list, 1):
                pair, direction, result, gale_level = sig
                
                if result == "WIN":
                    emoji = "✅"
                    if gale_level == 0:
                        detail = "Signal"
                    elif gale_level == 1:
                        detail = "G1"
                    elif gale_level == 2:
                        detail = "G2"
                    else:
                        detail = f"G{gale_level}"
                else:
                    emoji = "❌"
                    detail = "Perdu"
                
                report += f"{i}. {emoji} {pair} {direction} • {detail}\n"
        
        report += (
            f"\n━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 Fin de la session quotidienne\n"
            f"📅 Prochaine session: Demain {START_HOUR_HAITI}h00 AM"
        )
        
        sent_count = 0
        for uid in user_ids:
            try:
                await app.bot.send_message(chat_id=uid, text=report)
                sent_count += 1
            except Exception as e:
                print(f"[RAPPORT] ❌ Envoi à {uid}: {e}")
        
        print(f"[RAPPORT] ✅ Envoyé à {sent_count} abonnés (Win rate: {winrate:.1f}%)")
        
    except Exception as e:
        print(f"[RAPPORT] ❌ Erreur: {e}")

# === FILE DE SIGNAUX ===

async def process_signal_queue(app):
    global signal_queue_running

    print("\n[SESSION] 🔍 Vérification des conditions...")
    print(f"[SESSION] - Marché ouvert: {is_forex_open()}")
    print(f"[SESSION] - Session running: {signal_queue_running}")
    
    if not is_forex_open():
        print("[SESSION] 🏖️ Marché fermé - Annulée")
        return

    if signal_queue_running:
        print("[SESSION] ⚠️ File déjà en cours")
        return

    signal_queue_running = True

    try:
        now_haiti = get_haiti_now()

        print(f"\n{'='*60}")
        print(f"[SESSION] 🚀 DÉBUT DE LA SESSION")
        print(f"{'='*60}")
        print(f"[SESSION] 🕐 Haïti: {now_haiti.strftime('%H:%M:%S')}")
        print(f"[SESSION] 🌍 UTC: {get_utc_now().strftime('%H:%M:%S')}")
        print(f"{'='*60}\n")
        
        active_pairs = PAIRS[:2]
        
        for i in range(NUM_SIGNALS_PER_DAY):
            if not is_forex_open():
                print("[SESSION] 🏖️ Marché fermé - Arrêt")
                break
            
            pair = active_pairs[i % len(active_pairs)]
            
            print(f"\n[SESSION] {'─'*60}")
            print(f"[SESSION] 📍 SIGNAL {i+1}/{NUM_SIGNALS_PER_DAY} - {pair}")
            print(f"[SESSION] {'─'*60}")
            
            now_haiti = get_haiti_now()
            entry_time_haiti = now_haiti + timedelta(minutes=DELAY_BEFORE_ENTRY_MIN)
            
            signal_id = None
            for attempt in range(3):
                signal_id = await send_pre_signal(pair, entry_time_haiti, app)
                if signal_id:
                    break
                await asyncio.sleep(30)
            
            if not signal_id:
                print(f"[SESSION] ❌ Aucun signal valide")
                continue
            
            verification_time_haiti = entry_time_haiti + timedelta(minutes=VERIFICATION_WAIT_MIN)
            now_haiti = get_haiti_now()
            wait_seconds = (verification_time_haiti - now_haiti).total_seconds()
            
            if wait_seconds > 0:
                print(f"[SESSION] ⏳ Attente de {wait_seconds/60:.1f} min")
                await asyncio.sleep(wait_seconds)
            
            print(f"[SESSION] 🔍 Vérification...")
            
            try:
                result = await auto_verifier.verify_single_signal(signal_id)
                
                if result:
                    print(f"[SESSION] ✅ Vérifié: {result}")
                else:
                    print(f"[SESSION] ⚠️ Vérification impossible")
            except Exception as e:
                print(f"[SESSION] ❌ Erreur vérification: {e}")
            
            await send_verification_briefing(signal_id, app)
            
            print(f"[SESSION] ✅ Cycle {i+1} terminé\n")
            await asyncio.sleep(30)
        
        print(f"\n{'='*60}")
        print(f"[SESSION] 🏁 SESSION TERMINÉE")
        print(f"{'='*60}\n")
        
        await send_daily_report(app)

    except Exception as e:
        print(f"[SESSION] ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
    finally:
        signal_queue_running = False
        print("[SESSION] 🔓 Session déverrouillée")

async def start_daily_signals(app):
    """Démarre la session quotidienne -
