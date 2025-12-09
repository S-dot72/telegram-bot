"""
Bot de trading M5 avec Sessions Planifiées
Evening Session: Signaux toutes les 10 minutes jusqu'à 02h00
+ Vérification automatique toutes les 15 minutes
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
from utils import compute_indicators, rule_signal_ultra_strict
from ml_predictor import MLSignalPredictor
from auto_verifier import AutoResultVerifier
from ml_continuous_learning import ContinuousLearning, scheduled_retraining
from backtester import BacktesterM5

# Configuration
HAITI_TZ = ZoneInfo("America/Port-au-Prince")

# SESSIONS PLANIFIÉES (en heure Haïti)
SCHEDULED_SESSIONS = [
    {
        'name': 'London Kill Zone',
        'start_hour': 2,
        'start_minute': 0,
        'end_hour': 5,
        'end_minute': 0,
        'signals_count': 3,
        'interval_minutes': 30,
        'priority': 3
    },
    {
        'name': 'London/NY Overlap',
        'start_hour': 9,
        'start_minute': 0,
        'end_hour': 11,
        'end_minute': 0,
        'signals_count': 4,
        'interval_minutes': 30,
        'priority': 5
    },
    {
        'name': 'NY Session',
        'start_hour': 14,
        'start_minute': 0,
        'end_hour': 17,
        'end_minute': 0,
        'signals_count': 4,
        'interval_minutes': 30,
        'priority': 3
    },
    {
        'name': 'Evening Session',
        'start_hour': 18,
        'start_minute': 0,
        'end_hour': 2,  # Jusqu'à 02h00 (prochaine session London)
        'end_minute': 0,
        'signals_count': -1,  # Illimité (jusqu'à fin session)
        'interval_minutes': 10,  # Toutes les 10 minutes
        'priority': 2,
        'continuous': True  # Flag pour session continue
    }
]

# Paramètres M5
TIMEFRAME_M5 = "5min"
DELAY_BEFORE_ENTRY_MIN = 5
VERIFICATION_WAIT_MIN = 5
CONFIDENCE_THRESHOLD = 0.65

engine = create_engine(DB_URL, connect_args={'check_same_thread': False})
sched = AsyncIOScheduler(timezone=HAITI_TZ)
ml_predictor = MLSignalPredictor()
auto_verifier = None
active_sessions = {}
signal_verification_queue = []

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
    
    if weekday == 5:
        return False
    if weekday == 6 and hour < 22:
        return False
    if weekday == 4 and hour >= 22:
        return False
    
    return True

def get_current_session():
    """Retourne la session active actuellement (basée sur heure Haïti)"""
    now_haiti = get_haiti_now()
    current_time = now_haiti.hour * 60 + now_haiti.minute
    
    for session in SCHEDULED_SESSIONS:
        start_time = session['start_hour'] * 60 + session['start_minute']
        end_time = session['end_hour'] * 60 + session['end_minute']
        
        # Gérer Evening Session qui traverse minuit
        if session.get('continuous') and session['end_hour'] < session['start_hour']:
            # Session traverse minuit (ex: 18h -> 02h)
            if current_time >= start_time or current_time < end_time:
                return session
        else:
            if start_time <= current_time < end_time:
                return session
    
    return None

def get_next_session():
    """Retourne la prochaine session à venir"""
    now_haiti = get_haiti_now()
    current_time = now_haiti.hour * 60 + now_haiti.minute
    
    for session in SCHEDULED_SESSIONS:
        start_time = session['start_hour'] * 60 + session['start_minute']
        
        if start_time > current_time:
            return session
    
    return SCHEDULED_SESSIONS[0]

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
    q = text("""INSERT INTO signals (pair,direction,reason,ts_enter,ts_send,confidence,payload_json,max_gales)
    VALUES (:pair,:direction,:reason,:ts_enter,:ts_send,:confidence,:payload,:max_gales)""")
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
                conn.execute(text("ALTER TABLE signals ADD COLUMN max_gales INTEGER DEFAULT 0"))
            if 'winning_attempt' not in existing_cols:
                conn.execute(text("ALTER TABLE signals ADD COLUMN winning_attempt TEXT"))
            if 'reason' not in existing_cols:
                conn.execute(text("ALTER TABLE signals ADD COLUMN reason TEXT"))
            if 'kill_zone' not in existing_cols:
                conn.execute(text("ALTER TABLE signals ADD COLUMN kill_zone TEXT"))
            
            print("✅ Base de données prête")
        
        cleanup_weekend_signals()

    except Exception as e:
        print(f"⚠️ Erreur DB: {e}")

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
                
                next_session = get_next_session()
                next_time = f"{next_session['start_hour']:02d}h{next_session['start_minute']:02d}"
                
                await update.message.reply_text(
                    f"✅ Bienvenue au Bot Trading M5 - Sessions Planifiées !\n\n"
                    f"📅 **SESSIONS QUOTIDIENNES:**\n\n"
                    f"🌅 **02h-05h** (London Kill Zone)\n"
                    f"   • 3 signaux à 30 min d'intervalle\n"
                    f"   • Priorité: 3/5\n\n"
                    f"🔥 **09h-11h** (London/NY Overlap)\n"
                    f"   • 4 signaux à 30 min d'intervalle\n"
                    f"   • Priorité: 5/5 ⭐\n\n"
                    f"📈 **14h-17h** (NY Session)\n"
                    f"   • 4 signaux à 30 min d'intervalle\n"
                    f"   • Priorité: 3/5\n\n"
                    f"🌆 **18h-02h** (Evening Session) 🔥\n"
                    f"   • Signaux toutes les 10 minutes\n"
                    f"   • Session intensive jusqu'à 02h\n"
                    f"   • Priorité: 2/5\n\n"
                    f"📍 Timeframe: M5 (5 minutes)\n"
                    f"💪 Total: 40-50 signaux par jour\n"
                    f"🔍 Vérification auto toutes les 15 min\n"
                    f"⏰ Prochaine session: {next_session['name']} à {next_time}\n\n"
                    f"📋 Tapez /menu pour toutes les commandes"
                )
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    menu_text = (
        "📋 **MENU DES COMMANDES**\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        "📊 **Statistiques & Info:**\n"
        "• /stats - Voir les statistiques générales\n"
        "• /status - État actuel du bot\n"
        "• /rapport - Rapport du jour en cours\n"
        "• /sessions - Planning des sessions\n\n"
        "🤖 **Machine Learning:**\n"
        "• /mlstats - Statistiques ML\n"
        "• /retrain - Réentraîner le modèle ML\n\n"
        "🔬 **Backtesting:**\n"
        "• /backtest - Lancer un backtest M5\n"
        "• /backtest <paire> - Backtest sur une paire\n\n"
        "🔧 **Contrôles:**\n"
        "• /testsignal - Forcer un signal de test\n"
        "• /verify - Forcer vérification signaux\n"
        "• /menu - Afficher ce menu\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 Timeframe: M5\n"
        f"💪 40-50 signaux/jour\n"
        f"🔍 Vérif auto: 15 min"
    )
    await update.message.reply_text(menu_text)

async def cmd_sessions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    now_haiti = get_haiti_now()
    current_session = get_current_session()
    next_session = get_next_session()
    
    msg = "📅 **PLANNING DES SESSIONS**\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += f"🕐 Heure actuelle: {now_haiti.strftime('%H:%M')} (Haïti)\n\n"
    
    if current_session:
        msg += f"✅ **Session active:** {current_session['name']}\n"
        msg += f"🔥 Priorité: {current_session['priority']}/5\n"
        if current_session.get('continuous'):
            msg += f"⚡ Mode intensif: Signal toutes les {current_session['interval_minutes']}min\n\n"
        else:
            msg += f"⚡ Signaux restants: À venir\n\n"
    else:
        msg += "⏸️ Aucune session active\n\n"
    
    msg += "📋 **Planning quotidien:**\n\n"
    
    for session in SCHEDULED_SESSIONS:
        start = f"{session['start_hour']:02d}h{session['start_minute']:02d}"
        end = f"{session['end_hour']:02d}h{session['end_minute']:02d}"
        priority_stars = "⭐" * session['priority']
        
        msg += f"**{session['name']}** ({start}-{end})\n"
        if session.get('continuous'):
            msg += f"   • Mode intensif: Signal toutes les {session['interval_minutes']}min\n"
            msg += f"   • Durée: 8h continues\n"
        else:
            msg += f"   • {session['signals_count']} signaux (intervalle: {session['interval_minutes']}min)\n"
        msg += f"   • Priorité: {priority_stars}\n\n"
    
    if next_session and not current_session:
        next_time = f"{next_session['start_hour']:02d}h{next_session['start_minute']:02d}"
        msg += f"⏰ Prochaine: {next_session['name']} à {next_time}\n\n"
    
    msg += "━━━━━━━━━━━━━━━━━━━━\n"
    msg += "💪 Total: 40-50 signaux par jour\n"
    msg += "🔍 Vérification auto: toutes les 15 min"
    
    await update.message.reply_text(msg)

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
        msg += f"👥 Abonnés: {subs}\n\n"
        msg += f"📍 Timeframe: M5\n"
        msg += f"💪 40-50 signaux/jour planifiés\n"
        msg += f"🔍 Vérif auto: 15 min"
        
        await update.message.reply_text(msg)

    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        now_haiti = get_haiti_now()
        now_utc = get_utc_now()
        forex_open = is_forex_open()
        current_session = get_current_session()
        next_session = get_next_session()
        
        msg = f"🤖 **État du Bot**\n\n"
        msg += f"🇭🇹 Haïti: {now_haiti.strftime('%a %H:%M:%S')}\n"
        msg += f"🌍 UTC: {now_utc.strftime('%a %H:%M:%S')}\n"
        msg += f"📈 Forex: {'🟢 OUVERT' if forex_open else '🔴 FERMÉ'}\n\n"
        
        if current_session:
            msg += f"✅ **Session Active:** {current_session['name']}\n"
            msg += f"🔥 Priorité: {current_session['priority']}/5\n"
            if current_session.get('continuous'):
                msg += f"⚡ Mode: Intensif (toutes les {current_session['interval_minutes']}min)\n"
            else:
                msg += f"⚡ Signaux prévus: {current_session['signals_count']}\n"
                msg += f"⏱️ Intervalle: {current_session['interval_minutes']} min\n"
            msg += "\n"
        else:
            msg += f"⏸️ Aucune session active\n\n"
            if next_session:
                next_time = f"{next_session['start_hour']:02d}h{next_session['start_minute']:02d}"
                msg += f"⏰ Prochaine: {next_session['name']} à {next_time}\n\n"
        
        msg += f"📍 Timeframe: M5\n"
        msg += f"💪 40-50 signaux/jour\n"
        msg += f"🔍 Vérif auto: 15 min"
        
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_verify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Commande pour forcer la vérification des signaux"""
    try:
        msg = await update.message.reply_text("🔍 Vérification forcée des signaux en attente...")
        
        await auto_verifier.verify_pending_signals()
        
        await msg.edit_text("✅ Vérification terminée ! Consultez /stats pour les résultats.")
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_retrain(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = await update.message.reply_text("🤖 Réentraînement en cours...")
        
        learner = ContinuousLearning(engine)
        result = learner.retrain_model(min_signals=30, min_accuracy_improvement=0.00)
        
        if result['success']:
            if result['accepted']:
                response = (
                    f"✅ **Modèle réentraîné avec succès**\n\n"
                    f"📊 Signaux: {result['signals_count']}\n"
                    f"🎯 Accuracy: {result['accuracy']*100:.2f}%\n"
                    f"📈 Amélioration: {result['improvement']*100:+.2f}%"
                )
            else:
                response = (
                    f"⚠️ **Modèle rejeté**\n\n"
                    f"📊 Signaux: {result['signals_count']}\n"
                    f"🎯 Accuracy: {result['accuracy']*100:.2f}%\n"
                    f"📉 Amélioration: {result['improvement']*100:+.2f}%\n"
                    f"ℹ️ Amélioration trop faible"
                )
        else:
            response = f"❌ Erreur: {result['reason']}"
        
        await msg.edit_text(response)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_mlstats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        learner = ContinuousLearning(engine)
        stats = learner.get_training_stats()
        
        msg = (
            f"🤖 **Statistiques ML**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📊 Entraînements: {stats['total_trainings']}\n"
            f"🎯 Meilleure accuracy: {stats['best_accuracy']*100:.2f}%\n"
            f"📈 Signaux entraînés: {stats['total_signals']}\n"
            f"📅 Dernier entraînement: {stats['last_training']}\n\n"
        )
        
        if stats['recent_trainings']:
            msg += "📋 **Derniers entraînements:**\n\n"
            for t in reversed(stats['recent_trainings']):
                date = datetime.fromisoformat(t['timestamp']).strftime('%d/%m %H:%M')
                emoji = "✅" if t.get('accepted', False) else "⚠️"
                msg += f"{emoji} {date} - {t['accuracy']*100:.1f}%\n"
        
        msg += "\n━━━━━━━━━━━━━━━━━━━━"
        
        await update.message.reply_text(msg)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_rapport(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = await update.message.reply_text("📊 Génération du rapport...")
        
        now_haiti = get_haiti_now()
        start_haiti = now_haiti.replace(hour=0, minute=0, second=0, microsecond=0)
        end_haiti = start_haiti + timedelta(days=1)
        
        start_utc = start_haiti.astimezone(timezone.utc)
        end_utc = end_haiti.astimezone(timezone.utc)
        
        with engine.connect() as conn:
            query = text("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN result = 'LOSE' THEN 1 ELSE 0 END) as losses
                FROM signals
                WHERE ts_send >= :start AND ts_send < :end
                AND result IS NOT NULL
            """)
            
            stats = conn.execute(query, {
                "start": start_utc.isoformat(),
                "end": end_utc.isoformat()
            }).fetchone()
        
        if not stats or stats[0] == 0:
            await msg.edit_text("ℹ️ Aucun signal aujourd'hui")
            return
        
        total, wins, losses = stats
        verified = wins + losses
        winrate = (wins / verified * 100) if verified > 0 else 0
        
        report = (
            f"📊 **RAPPORT DU JOUR**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📅 {now_haiti.strftime('%d/%m/%Y %H:%M')}\n\n"
            f"📈 **PERFORMANCE**\n"
            f"• Total: {total}\n"
            f"• ✅ Gagnés: {wins}\n"
            f"• ❌ Perdus: {losses}\n"
            f"• 📊 Win rate: **{winrate:.1f}%**\n\n"
            f"📍 40-50 signaux planifiés/jour\n\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
        
        await msg.edit_text(report)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_test_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = await update.message.reply_text("🚀 Test de signal...")
        
        current_session = get_current_session()
        if not current_session:
            next_session = get_next_session()
            next_time = f"{next_session['start_hour']:02d}h{next_session['start_minute']:02d}"
            await msg.edit_text(
                f"⏸️ Aucune session active\n\n"
                f"⏰ Prochaine: {next_session['name']} à {next_time}"
            )
            return
        
        app = context.application
        asyncio.create_task(send_single_signal(app, current_session))
        
        await msg.edit_text(f"✅ Signal de test lancé pour {current_session['name']} !")
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lance un backtest M5 et envoie les résultats"""
    try:
        msg = await update.message.reply_text(
            "🔬 **Lancement du backtest M5**\n\n"
            "⏳ Analyse en cours...\n"
            "Cela peut prendre 1-2 minutes."
        )
        
        pairs_to_test = PAIRS[:3]
        
        if context.args and len(context.args) > 0:
            requested_pair = context.args[0].upper().replace('-', '/')
            if requested_pair in PAIRS:
                pairs_to_test = [requested_pair]
            else:
                await msg.edit_text(
                    f"❌ Paire non reconnue: {requested_pair}\n\n"
                    f"Paires disponibles:\n" + "\n".join(f"• {p}" for p in PAIRS[:5])
                )
                return
        
        backtester = BacktesterM5(confidence_threshold=CONFIDENCE_THRESHOLD)
        
        print(f"\n[BACKTEST] Lancement pour {len(pairs_to_test)} paire(s)")
        
        results = backtester.run_full_backtest(
            pairs=pairs_to_test,
            outputsize=3000
        )
        
        result_msg = backtester.format_results_for_telegram(results)
        
        await msg.edit_text(result_msg)
        
        print(f"[BACKTEST] ✅ Résultats envoyés")
        
    except Exception as e:
        error_msg = f"❌ **Erreur backtest**\n\n{str(e)[:200]}"
        await update.message.reply_text(error_msg)
        print(f"[BACKTEST] ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

async def send_single_signal(app, session):
    """Envoie un signal unique pour une session"""
    try:
        if not is_forex_open():
            print("[SIGNAL] 🏖️ Marché fermé")
            return None
        
        now_haiti = get_haiti_now()
        print(f"\n[SIGNAL] 📤 Session: {session['name']} - {now_haiti.strftime('%H:%M:%S')}")
        
        # Rotation des paires
        active_pairs = PAIRS[:3]
        session_signals = active_sessions.get(session['name'], [])
        pair = active_pairs[len(session_signals) % len(active_pairs)]
        
        params = BEST_PARAMS.get(pair, {})
        df = get_cached_ohlc(pair, TIMEFRAME_M5, outputsize=400)
        
        if df is None or len(df) < 50:
            print("[SIGNAL] ❌ Pas de données")
            return None
        
        df = compute_indicators(df, ema_fast=params.get('ema_fast',8),
                                ema_slow=params.get('ema_slow',21),
                                rsi_len=params.get('rsi',14),
                                bb_len=params.get('bb',20))
        
        # Passer la priorité de session à la stratégie
        base_signal = rule_signal_ultra_strict(df, session_priority=session['priority'])
        
        if not base_signal:
            print("[SIGNAL] ⏭️ Pas de signal (stratégie)")
            return None
        
        ml_signal, ml_conf = ml_predictor.predict_signal(df, base_signal)
        if ml_signal is None or ml_conf < CONFIDENCE_THRESHOLD:
            print(f"[SIGNAL] ❌ Rejeté par ML ({ml_conf:.1%})")
            return None
        
        entry_time_haiti = now_haiti + timedelta(minutes=DELAY_BEFORE_ENTRY_MIN)
        entry_time_utc = entry_time_haiti.astimezone(timezone.utc)
        
        print(f"[SIGNAL] 📤 Signal trouvé ! Entrée: {entry_time_haiti.strftime('%H:%M')} (dans {DELAY_BEFORE_ENTRY_MIN} min)")
        
        payload = {
            'pair': pair, 'direction': ml_signal, 
            'reason': f'ML {ml_conf:.1%} - {session["name"]}',
            'ts_enter': entry_time_utc.isoformat(), 
            'ts_send': get_utc_now().isoformat(),
            'confidence': ml_conf, 
            'payload': json.dumps({'pair': pair, 'session': session['name']}),
            'max_gales': 0
        }
        signal_id = persist_signal(payload)
        
        try:
            with engine.begin() as conn:
                conn.execute(
                    text("UPDATE signals SET kill_zone = :kz WHERE id = :sid"),
                    {'kz': session['name'], 'sid': signal_id}
                )
        except:
            pass
        
        with engine.connect() as conn:
            user_ids = [r[0] for r in conn.execute(text("SELECT user_id FROM subscribers")).fetchall()]
        
        direction_text = "BUY" if ml_signal == "CALL" else "SELL"
        
        msg = (
            f"🎯 SIGNAL — {pair}\n\n"
            f"📅 Session: {session['name']}\n"
            f"🔥 Priorité: {session['priority']}/5\n"
            f"🕐 Entrée: {entry_time_haiti.strftime('%H:%M')} (Haïti)\n"
            f"📍 Timeframe: M5 (5 minutes)\n\n"
            f"📈 Direction: **{direction_text}**\n"
            f"💪 Confiance: **{int(ml_conf*100)}%**\n\n"
            f"🔍 Vérification: 5 min après entrée"
        )
        
        for uid in user_ids:
            try:
                await app.bot.send_message(chat_id=uid, text=msg)
            except Exception as e:
                print(f"[SIGNAL] ❌ Envoi à {uid}: {e}")
        
        print(f"[SIGNAL] ✅ Envoyé ({ml_signal}, {ml_conf:.1%})")
        
        # Tracking
        if session['name'] not in active_sessions:
            active_sessions[session['name']] = []
        active_sessions[session['name']].append(signal_id)
        
        # Ajouter à la queue de vérification (ne pas bloquer)
        verification_time = entry_time_utc + timedelta(minutes=VERIFICATION_WAIT_MIN)
        signal_verification_queue.append({
            'signal_id': signal_id,
            'verification_time': verification_time
        })
        
        return signal_id
        
    except Exception as e:
        print(f"[SIGNAL] ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

async def send_verification_briefing(signal_id, app):
    try:
        with engine.connect() as conn:
            signal = conn.execute(
                text("SELECT pair, direction, result, confidence, kill_zone FROM signals WHERE id = :sid"),
                {"sid": signal_id}
            ).fetchone()

        if not signal or not signal[2]:
            return

        pair, direction, result, confidence, kill_zone = signal
        
        with engine.connect() as conn:
            user_ids = [r[0] for r in conn.execute(text("SELECT user_id FROM subscribers")).fetchall()]
        
        emoji = "✅" if result == "WIN" else "❌"
        status = "GAGNÉ" if result == "WIN" else "PERDU"
        direction_emoji = "📈" if direction == "CALL" else "📉"
        
        briefing = (
            f"{emoji} **BRIEFING SIGNAL**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{direction_emoji} Paire: **{pair}**\n"
            f"📊 Direction: **{direction}**\n"
            f"💪 Confiance: {int(confidence*100)}%\n"
        )
        
        if kill_zone:
            briefing += f"📅 Session: {kill_zone}\n"
        
        briefing += f"\n🎲 Résultat: **{status}**\n\n"
        briefing += f"━━━━━━━━━━━━━━━━━━━━"
        
        for uid in user_ids:
            try:
                await app.bot.send_message(chat_id=uid, text=briefing)
            except:
                pass
        
        print(f"[BRIEFING] ✅ Envoyé: {status}")

    except Exception as e:
        print(f"[BRIEFING] ❌ Erreur: {e}")

async def run_scheduled_session(app, session):
    """Exécute une session planifiée"""
    if not is_forex_open():
        print(f"[SESSION] 🏖️ Marché fermé - {session['name']}")
        return
    
    print(f"\n[SESSION] 🚀 DÉBUT - {session['name']}")
    print(f"[SESSION] 🔥 Priorité: {session['priority']}/5")
    
    # Réinitialiser tracking
    active_sessions[session['name']] = []
    
    if session.get('continuous'):
        # Mode continu (Evening Session)
        print(f"[SESSION] ⚡ Mode INTENSIF - Signaux toutes les {session['interval_minutes']}min jusqu'à {session['end_hour']:02d}h")
        
        signal_count = 0
        while True:
            # Vérifier si on est toujours dans la session
            current_session = get_current_session()
            if not current_session or current_session['name'] != session['name']:
                print(f"[SESSION] ⏰ Fin de session atteinte")
                break
            
            if not is_forex_open():
                print(f"[SESSION] 🏖️ Marché fermé - Arrêt")
                break
            
            signal_count += 1
            print(f"\n[SESSION] 📍 Signal #{signal_count}")
            
            # Tenter d'envoyer signal (1 seule tentative en mode intensif)
            signal_id = await send_single_signal(app, session)
            
            if signal_id:
                print(f"[SESSION] ✅ Signal #{signal_count} envoyé")
            else:
                print(f"[SESSION] ⏭️ Signal #{signal_count} non généré (conditions non remplies)")
            
            # Attendre intervalle avant prochain signal
            print(f"[SESSION] ⏸️ Pause {session['interval_minutes']}min avant prochain signal...")
            await asyncio.sleep(session['interval_minutes'] * 60)
        
        signals_sent = len(active_sessions.get(session['name'], []))
        print(f"\n[SESSION] 🏁 FIN - {session['name']}")
        print(f"[SESSION] 📊 {signals_sent} signaux envoyés (mode intensif)")
        
    else:
        # Mode standard (sessions à nombre fixe)
        print(f"[SESSION] ⚡ {session['signals_count']} signaux à {session['interval_minutes']}min d'intervalle")
        
        for i in range(session['signals_count']):
            if not is_forex_open():
                print(f"[SESSION] 🏖️ Marché fermé - Arrêt session")
                break
            
            print(f"\n[SESSION] 📍 Signal {i+1}/{session['signals_count']}")
            
            # Tenter d'envoyer signal (max 3 tentatives)
            signal_sent = False
            for attempt in range(3):
                signal_id = await send_single_signal(app, session)
                if signal_id:
                    signal_sent = True
                    break
                
                if attempt < 2:
                    print(f"[SESSION] ⏳ Attente 20s avant nouvelle tentative...")
                    await asyncio.sleep(20)
            
            if not signal_sent:
                print(f"[SESSION] ⚠️ Signal {i+1} non envoyé (marché non favorable)")
            
            # Attendre intervalle avant prochain signal (sauf pour le dernier)
            if i < session['signals_count'] - 1:
                wait_time = session['interval_minutes'] * 60
                print(f"[SESSION] ⏸️ Pause {session['interval_minutes']}min avant prochain signal...")
                await asyncio.sleep(wait_time)
        
        signals_sent = len(active_sessions.get(session['name'], []))
        print(f"\n[SESSION] 🏁 FIN - {session['name']}")
        print(f"[SESSION] 📊 {signals_sent}/{session['signals_count']} signaux envoyés")

async def automated_verification_check(app):
    """Vérification automatique programmée toutes les 15 minutes"""
    try:
        print("\n[AUTO-VERIF] 🔍 Vérification automatique programmée...")
        
        # Appeler le vérificateur automatique
        await auto_verifier.verify_pending_signals()
        
        print("[AUTO-VERIF] ✅ Vérification terminée")
        
    except Exception as e:
        print(f"[AUTO-VERIF] ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

async def send_daily_report(app):
    """Rapport quotidien"""
    try:
        print("\n[RAPPORT] 📊 Génération rapport du jour...")
        
        now_haiti = get_haiti_now()
        start_haiti = now_haiti.replace(hour=0, minute=0, second=0, microsecond=0)
        end_haiti = start_haiti + timedelta(days=1)
        
        start_utc = start_haiti.astimezone(timezone.utc)
        end_utc = end_haiti.astimezone(timezone.utc)
        
        with engine.connect() as conn:
            query = text("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN result = 'LOSE' THEN 1 ELSE 0 END) as losses
                FROM signals
                WHERE ts_send >= :start AND ts_send < :end
                AND result IS NOT NULL
            """)
            
            stats = conn.execute(query, {
                "start": start_utc.isoformat(),
                "end": end_utc.isoformat()
            }).fetchone()
            
            user_ids = [r[0] for r in conn.execute(text("SELECT user_id FROM subscribers")).fetchall()]
        
        if not stats or stats[0] == 0:
            print("[RAPPORT] ⚠️ Aucun signal aujourd'hui")
            return
        
        total, wins, losses = stats
        verified = wins + losses
        winrate = (wins / verified * 100) if verified > 0 else 0
        
        report = (
            f"📊 **RAPPORT QUOTIDIEN**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📅 {now_haiti.strftime('%d/%m/%Y %H:%M')}\n\n"
            f"📈 **PERFORMANCE**\n"
            f"• Total: {total}\n"
            f"• ✅ Gagnés: {wins}\n"
            f"• ❌ Perdus: {losses}\n"
            f"• 📊 Win rate: **{winrate:.1f}%**\n\n"
            f"📍 40-50 signaux/jour planifiés\n"
            f"🌆 Evening Session: Intensive\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📅 Prochaine session: Demain 02h00"
        )
        
        for uid in user_ids:
            try:
                await app.bot.send_message(chat_id=uid, text=report)
            except:
                pass
        
        print(f"[RAPPORT] ✅ Envoyé (Win rate: {winrate:.1f}%)")
        
    except Exception as e:
        print(f"[RAPPORT] ❌ Erreur: {e}")

async def main():
    global auto_verifier

    now_haiti = get_haiti_now()
    now_utc = get_utc_now()

    print("\n" + "="*60)
    print("🤖 BOT DE TRADING M5 - EVENING SESSION INTENSIVE")
    print("="*60)
    print(f"🇭🇹 Haïti: {now_haiti.strftime('%H:%M:%S %Z')}")
    print(f"🌍 UTC: {now_utc.strftime('%H:%M:%S %Z')}")
    print(f"📈 Forex: {'🟢 OUVERT' if is_forex_open() else '🔴 FERMÉ'}")
    
    current_session = get_current_session()
    if current_session:
        print(f"✅ Session active: {current_session['name']} (Priorité: {current_session['priority']}/5)")
        if current_session.get('continuous'):
            print(f"🔥 Mode INTENSIF: Signal toutes les {current_session['interval_minutes']}min")
    else:
        next_session = get_next_session()
        next_time = f"{next_session['start_hour']:02d}h{next_session['start_minute']:02d}"
        print(f"⏸️ Aucune session active")
        print(f"⏰ Prochaine: {next_session['name']} à {next_time}")
    
    print(f"\n📅 SESSIONS PLANIFIÉES:")
    for session in SCHEDULED_SESSIONS:
        start = f"{session['start_hour']:02d}h{session['start_minute']:02d}"
        end = f"{session['end_hour']:02d}h{session['end_minute']:02d}"
        if session.get('continuous'):
            print(f"• 🔥 {session['name']}: {start}-{end} (INTENSIF - toutes les {session['interval_minutes']}min)")
        else:
            print(f"• {session['name']}: {start}-{end} ({session['signals_count']} signaux)")
    
    print(f"\n📍 Timeframe: M5 (5 minutes)")
    print(f"💪 Total: 40-50 signaux par jour")
    print(f"⏰ Intervalle Evening: 10 minutes")
    print(f"🔍 Vérification auto: 15 minutes")
    print("="*60 + "\n")

    ensure_db()
    auto_verifier = AutoResultVerifier(engine, TWELVEDATA_API_KEY)

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler('start', cmd_start))
    app.add_handler(CommandHandler('menu', cmd_menu))
    app.add_handler(CommandHandler('stats', cmd_stats))
    app.add_handler(CommandHandler('status', cmd_status))
    app.add_handler(CommandHandler('sessions', cmd_sessions))
    app.add_handler(CommandHandler('verify', cmd_verify))
    app.add_handler(CommandHandler('retrain', cmd_retrain))
    app.add_handler(CommandHandler('mlstats', cmd_mlstats))
    app.add_handler(CommandHandler('rapport', cmd_rapport))
    app.add_handler(CommandHandler('testsignal', cmd_test_signal))
    app.add_handler(CommandHandler('backtest', cmd_backtest))

    sched.start()

    # SESSIONS PLANIFIÉES
    for session in SCHEDULED_SESSIONS:
        job_id = f"session_{session['name'].lower().replace(' ', '_').replace('/', '_')}"
        sched.add_job(
            run_scheduled_session,
            'cron',
            hour=session['start_hour'],
            minute=session['start_minute'],
            timezone=HAITI_TZ,
            args=[app, session],
            id=job_id
        )
        print(f"✅ Session planifiée: {session['name']} à {session['start_hour']:02d}h{session['start_minute']:02d}")
    
    # VÉRIFICATION AUTOMATIQUE TOUTES LES 15 MINUTES
    sched.add_job(
        automated_verification_check,
        'cron',
        minute='*/15',  # Toutes les 15 minutes
        timezone=HAITI_TZ,
        args=[app],
        id='auto_verification'
    )
    print(f"✅ Vérification auto planifiée: Toutes les 15 minutes")
    
    # Rapport quotidien
    sched.add_job(
        send_daily_report,
        'cron',
        hour=22,
        minute=0,
        timezone=HAITI_TZ,
        args=[app],
        id='daily_report'
    )

    # Si session active au démarrage, la lancer
    if current_session and is_forex_open():
        print(f"\n🚀 Démarrage immédiat - Session: {current_session['name']}")
        asyncio.create_task(run_scheduled_session(app, current_session))

    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)

    bot_info = await app.bot.get_me()
    print(f"✅ BOT ACTIF: @{bot_info.username}\n")

    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 Arrêt...")
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
        sched.shutdown()

if __name__ == '__main__':
    asyncio.run(main())
