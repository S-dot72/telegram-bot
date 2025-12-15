"""
Bot de trading M5 - Version Finale
Evening Session Intensive + Vérification Auto + Briefings Auto
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
from aiohttp import web
from config import *
from utils import compute_indicators, rule_signal_ultra_strict
from ml_predictor import MLSignalPredictor
from auto_verifier import AutoResultVerifier
from ml_continuous_learning import ContinuousLearning
from backtester import BacktesterM5

# Configuration
HAITI_TZ = ZoneInfo("America/Port-au-Prince")

# SESSIONS
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
        'end_hour': 2,
        'end_minute': 0,
        'signals_count': -1,
        'interval_minutes': 10,
        'priority': 2,
        'continuous': True
    }
]

# Paramètres
TIMEFRAME_M5 = "5min"
DELAY_BEFORE_ENTRY_MIN = 5
VERIFICATION_WAIT_MIN = 5
CONFIDENCE_THRESHOLD = 0.65

engine = create_engine(DB_URL, connect_args={'check_same_thread': False})
sched = AsyncIOScheduler(timezone=HAITI_TZ)
ml_predictor = MLSignalPredictor()
auto_verifier = None
active_sessions = {}
session_running = {}
pending_verifications = []

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
    now_haiti = get_haiti_now()
    current_time = now_haiti.hour * 60 + now_haiti.minute
    
    for session in SCHEDULED_SESSIONS:
        start_time = session['start_hour'] * 60 + session['start_minute']
        end_time = session['end_hour'] * 60 + session['end_minute']
        
        if session.get('continuous') and session['end_hour'] < session['start_hour']:
            if current_time >= start_time or current_time < end_time:
                return session
        else:
            if start_time <= current_time < end_time:
                return session
    
    return None

def get_next_session():
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

# ===== COMMANDES TELEGRAM =====

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    try:
        with engine.begin() as conn:
            existing = conn.execute(text("SELECT user_id FROM subscribers WHERE user_id = :uid"),
            {"uid": user_id}).fetchone()
            if existing:
                await update.message.reply_text("✅ Vous êtes déjà abonné !")
            else:
                conn.execute(text("INSERT INTO subscribers (user_id, username) VALUES (:uid, :uname)"),
                {"uid": user_id, "uname": username})
                
                next_session = get_next_session()
                next_time = f"{next_session['start_hour']:02d}h{next_session['start_minute']:02d}"
                
                await update.message.reply_text(
                    f"✅ Bienvenue au Bot Trading M5 !\n\n"
                    f"📅 **SESSIONS:**\n\n"
                    f"🌅 02h-05h London Kill Zone\n"
                    f"🔥 09h-11h London/NY Overlap\n"
                    f"📈 14h-17h NY Session\n"
                    f"🌆 18h-02h Evening Intensive\n\n"
                    f"📍 M5 | 40-50 signaux/jour\n"
                    f"🔍 Vérif + Briefings auto\n\n"
                    f"⏰ Prochaine: {next_session['name']} à {next_time}\n\n"
                    f"📋 /menu pour commandes"
                )
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    menu_text = (
        "📋 **MENU**\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        "📊 **Info:**\n"
        "• /stats - Statistiques\n"
        "• /status - État bot\n"
        "• /rapport - Rapport jour\n"
        "• /sessions - Planning\n\n"
        "🤖 **ML:**\n"
        "• /mlstats - Stats ML\n"
        "• /retrain - Réentraîner\n\n"
        "🔬 **Tests:**\n"
        "• /backtest - Backtest M5\n"
        "• /testsignal - Test signal\n"
        "• /verify - Vérifier signaux\n"
        "• /forcesession - Force session\n"
        "• /cleanup - Nettoyer DB\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 M5 | Briefings auto"
    )
    await update.message.reply_text(menu_text)

async def cmd_sessions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    now_haiti = get_haiti_now()
    current_session = get_current_session()
    
    msg = "📅 **PLANNING**\n━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += f"🕐 {now_haiti.strftime('%H:%M')} (Haïti)\n\n"
    
    if current_session:
        is_running = session_running.get(current_session['name'], False)
        msg += f"✅ **Active:** {current_session['name']}\n"
        msg += f"🔥 Priorité: {current_session['priority']}/5\n"
        msg += f"⚙️ État: {'🟢 Running' if is_running else '⚠️ Stopped'}\n"
        if current_session.get('continuous'):
            msg += f"⚡ Intensif (10min)\n\n"
        else:
            msg += f"⚡ {current_session['signals_count']} signaux\n\n"
    else:
        msg += "⏸️ Aucune session\n\n"
    
    msg += "📋 **Planning:**\n\n"
    for session in SCHEDULED_SESSIONS:
        start = f"{session['start_hour']:02d}h{session['start_minute']:02d}"
        end = f"{session['end_hour']:02d}h{session['end_minute']:02d}"
        msg += f"**{session['name']}** ({start}-{end})\n"
        if session.get('continuous'):
            msg += f"   Intensif 10min\n\n"
        else:
            msg += f"   {session['signals_count']} signaux\n\n"
    
    msg += "━━━━━━━━━━━━━━━━━━━━\n💪 40-50 signaux/jour"
    
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
        msg += f"Total: {total}\n"
        msg += f"Vérifiés: {verified}\n"
        msg += f"✅ Wins: {wins}\n"
        msg += f"❌ Losses: {losses}\n"
        msg += f"⏳ En attente: {pending}\n"
        msg += f"📈 Win rate: {winrate:.1f}%\n"
        msg += f"👥 Abonnés: {subs}\n\n"
        msg += f"📍 M5 | Briefings auto"
        
        await update.message.reply_text(msg)

    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        now_haiti = get_haiti_now()
        now_utc = get_utc_now()
        forex_open = is_forex_open()
        current_session = get_current_session()
        
        msg = f"🤖 **État Bot**\n\n"
        msg += f"🇭🇹 {now_haiti.strftime('%a %H:%M:%S')}\n"
        msg += f"🌍 {now_utc.strftime('%a %H:%M:%S')}\n"
        msg += f"📈 Forex: {'🟢 OUVERT' if forex_open else '🔴 FERMÉ'}\n\n"
        
        if current_session:
            is_running = session_running.get(current_session['name'], False)
            msg += f"✅ **Session:** {current_session['name']}\n"
            msg += f"🔥 Priorité: {current_session['priority']}/5\n"
            msg += f"⚙️ État: {'🟢 Running' if is_running else '⚠️ Stopped'}\n"
            if current_session.get('continuous'):
                msg += f"⚡ Intensif (10min)\n\n"
            else:
                msg += f"⚡ {current_session['signals_count']} signaux\n\n"
            
            # Suggérer /forcesession si stopped
            if not is_running:
                msg += "💡 Utilisez /forcesession pour lancer\n\n"
        else:
            next_session = get_next_session()
            next_time = f"{next_session['start_hour']:02d}h{next_session['start_minute']:02d}"
            msg += f"⏸️ Aucune session\n"
            msg += f"⏰ Prochaine: {next_session['name']} à {next_time}\n\n"
        
        msg += f"📍 M5 | Briefings auto: 15min"
        
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_verify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = await update.message.reply_text("🔍 Vérification forcée...")
        
        # Récupérer IDs avant vérif
        with engine.connect() as conn:
            pending_ids = [r[0] for r in conn.execute(
                text("SELECT id FROM signals WHERE result IS NULL")
            ).fetchall()]
        
        if not pending_ids:
            await msg.edit_text("ℹ️ Aucun signal à vérifier")
            return
        
        # Vérifier
        await auto_verifier.verify_pending_signals()
        
        # Envoyer briefings
        app = context.application
        
        if not pending_ids:
            await msg.edit_text("ℹ️ Aucun signal à vérifier")
            return
        
        with engine.connect() as conn:
            placeholders = ','.join([f':id{i}' for i in range(len(pending_ids))])
            params = {f'id{i}': pid for i, pid in enumerate(pending_ids)}
            
            verified = conn.execute(
                text(f"SELECT id, result FROM signals WHERE id IN ({placeholders}) AND result IS NOT NULL"),
                params
            ).fetchall()
        
        for signal_id, result in verified:
            await send_verification_briefing(signal_id, app)
        
        await msg.edit_text(f"✅ {len(verified)} signaux vérifiés et briefings envoyés ! /stats pour détails.")
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_retrain(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = await update.message.reply_text("🤖 Réentraînement...")
        
        learner = ContinuousLearning(engine)
        result = learner.retrain_model(min_signals=30, min_accuracy_improvement=0.00)
        
        if result['success']:
            if result['accepted']:
                response = (
                    f"✅ **Modèle réentraîné**\n\n"
                    f"📊 Signaux: {result['signals_count']}\n"
                    f"🎯 Accuracy: {result['accuracy']*100:.2f}%\n"
                    f"📈 Amélioration: {result['improvement']*100:+.2f}%"
                )
            else:
                response = (
                    f"⚠️ **Modèle rejeté**\n\n"
                    f"📊 Signaux: {result['signals_count']}\n"
                    f"🎯 Accuracy: {result['accuracy']*100:.2f}%\n"
                    f"📉 Amélioration: {result['improvement']*100:+.2f}%"
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
            f"🤖 **Stats ML**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📊 Entraînements: {stats['total_trainings']}\n"
            f"🎯 Best accuracy: {stats['best_accuracy']*100:.2f}%\n"
            f"📈 Signaux: {stats['total_signals']}\n"
            f"📅 Dernier: {stats['last_training']}\n"
        )
        
        if stats['recent_trainings']:
            msg += "\n📋 **Derniers:**\n\n"
            for t in reversed(stats['recent_trainings'][-3:]):
                date = datetime.fromisoformat(t['timestamp']).strftime('%d/%m %H:%M')
                emoji = "✅" if t.get('accepted', False) else "⚠️"
                msg += f"{emoji} {date} - {t['accuracy']*100:.1f}%\n"
        
        await update.message.reply_text(msg)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_rapport(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = await update.message.reply_text("📊 Génération...")
        
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
            f"📊 **RAPPORT**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📅 {now_haiti.strftime('%d/%m/%Y')}\n\n"
            f"• Total: {total}\n"
            f"• ✅ Wins: {wins}\n"
            f"• ❌ Losses: {losses}\n"
            f"• 📊 WR: **{winrate:.1f}%**\n\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
        
        await msg.edit_text(report)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_test_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        current_session = get_current_session()
        if not current_session:
            next_session = get_next_session()
            next_time = f"{next_session['start_hour']:02d}h{next_session['start_minute']:02d}"
            await update.message.reply_text(
                f"⏸️ Aucune session\n⏰ Prochaine: {next_session['name']} à {next_time}"
            )
            return
        
        msg = await update.message.reply_text(f"🚀 Test {current_session['name']}...")
        
        app = context.application
        signal_id = await send_single_signal(app, current_session)
        
        if signal_id:
            await msg.edit_text(f"✅ Signal #{signal_id} envoyé !")
        else:
            await msg.edit_text("⚠️ Aucun signal (conditions non remplies)")
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_forcesession(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Force lancement session active"""
    try:
        current_session = get_current_session()
        if not current_session:
            await update.message.reply_text("⏸️ Aucune session à forcer")
            return
        
        if session_running.get(current_session['name'], False):
            await update.message.reply_text(f"⚠️ {current_session['name']} déjà en cours")
            return
        
        msg = await update.message.reply_text(f"🚀 Force {current_session['name']}...")
        
        app = context.application
        asyncio.create_task(run_scheduled_session(app, current_session))
        
        await msg.edit_text(f"✅ {current_session['name']} lancée !")
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = await update.message.reply_text("🔬 Backtest...\n⏳ 1-2 min...")
        
        pairs_to_test = PAIRS[:3]
        
        if context.args and len(context.args) > 0:
            requested_pair = context.args[0].upper().replace('-', '/')
            if requested_pair in PAIRS:
                pairs_to_test = [requested_pair]
            else:
                await msg.edit_text(f"❌ Paire inconnue: {requested_pair}")
                return
        
        backtester = BacktesterM5(confidence_threshold=CONFIDENCE_THRESHOLD)
        results = backtester.run_full_backtest(pairs=pairs_to_test, outputsize=3000)
        result_msg = backtester.format_results_for_telegram(results)
        
        await msg.edit_text(result_msg)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {str(e)[:200]}")

async def cmd_cleanup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Nettoie les anciens signaux en attente"""
    try:
        msg = await update.message.reply_text("🧹 Nettoyage en cours...")
        
        # Compter signaux en attente
        with engine.connect() as conn:
            pending_count = conn.execute(
                text("SELECT COUNT(*) FROM signals WHERE result IS NULL")
            ).scalar()
        
        if pending_count == 0:
            await msg.edit_text("✅ Aucun signal à nettoyer !")
            return
        
        # Marquer tous comme LOSE (anciens signaux invalides)
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE signals SET result = 'LOSE', reason = 'Nettoyage manuel' WHERE result IS NULL")
            )
        
        await msg.edit_text(
            f"✅ Nettoyage terminé !\n\n"
            f"🧹 {pending_count} anciens signaux marqués comme LOSE\n"
            f"📊 /stats pour voir résultat"
        )
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

# ===== FONCTIONS SIGNAL =====

async def send_single_signal(app, session):
    """Envoie signal avec debug"""
    try:
        if not is_forex_open():
            print("[SIGNAL] 🏖️ Marché fermé")
            return None
        
        now_haiti = get_haiti_now()
        print(f"\n[SIGNAL] 📤 {session['name']} - {now_haiti.strftime('%H:%M:%S')}")
        
        # ===== VÉRIFIER SIGNAUX PRÉCÉDENTS AVANT NOUVEAU SIGNAL =====
        print("[SIGNAL] 🔍 Vérification signaux précédents...")
        try:
            # Récupérer signaux en attente (plus vieux que 7 min pour données fraîches TwelveData)
            verification_cutoff = now_haiti - timedelta(minutes=7)
            verification_cutoff_utc = verification_cutoff.astimezone(timezone.utc)
            
            with engine.connect() as conn:
                pending_old = conn.execute(
                    text("""
                        SELECT id FROM signals 
                        WHERE result IS NULL 
                        AND ts_enter < :cutoff
                        ORDER BY id
                    """),
                    {"cutoff": verification_cutoff_utc.isoformat()}
                ).fetchall()
                pending_ids = [r[0] for r in pending_old]
            
            if pending_ids:
                print(f"[SIGNAL] 📊 {len(pending_ids)} signaux à vérifier")
                
                # Vérifier
                await auto_verifier.verify_pending_signals()
                
                # Envoyer briefings
                with engine.connect() as conn:
                    placeholders = ','.join([f':id{i}' for i in range(len(pending_ids))])
                    params = {f'id{i}': pid for i, pid in enumerate(pending_ids)}
                    
                    verified = conn.execute(
                        text(f"SELECT id, result FROM signals WHERE id IN ({placeholders}) AND result IS NOT NULL"),
                        params
                    ).fetchall()
                
                for signal_id, result in verified:
                    await send_verification_briefing(signal_id, app)
                    print(f"[SIGNAL] 📧 Briefing #{signal_id} envoyé ({result})")
                
                if verified:
                    print(f"[SIGNAL] ✅ {len(verified)} briefings envoyés")
                    # Petit délai pour que briefings arrivent avant nouveau signal
                    await asyncio.sleep(2)
            else:
                print("[SIGNAL] ℹ️ Aucun signal ancien à vérifier")
        
        except Exception as e:
            print(f"[SIGNAL] ⚠️ Erreur vérification: {e}")
        
        # ===== MAINTENANT GÉNÉRER NOUVEAU SIGNAL =====
        print(f"[SIGNAL] 🎯 Génération nouveau signal...")
        
        # Rotation paires
        active_pairs = PAIRS[:3]
        session_signals = active_sessions.get(session['name'], [])
        pair = active_pairs[len(session_signals) % len(active_pairs)]
        
        print(f"[SIGNAL] 🔍 {pair}...")
        
        # Données
        params = BEST_PARAMS.get(pair, {})
        df = get_cached_ohlc(pair, TIMEFRAME_M5, outputsize=400)
        
        if df is None or len(df) < 50:
            print("[SIGNAL] ❌ Pas de données")
            return None
        
        print(f"[SIGNAL] ✅ {len(df)} bougies")
        
        # Indicateurs
        df = compute_indicators(df, ema_fast=params.get('ema_fast',8),
                                ema_slow=params.get('ema_slow',21),
                                rsi_len=params.get('rsi',14),
                                bb_len=params.get('bb',20))
        
        # Stratégie
        base_signal = rule_signal_ultra_strict(df, session_priority=session['priority'])
        
        if not base_signal:
            print("[SIGNAL] ⏭️ Rejeté (stratégie)")
            last = df.iloc[-1]
            print(f"[DEBUG] ADX:{last.get('adx',0):.1f} RSI:{last.get('rsi',0):.1f}")
            return None
        
        print(f"[SIGNAL] ✅ Stratégie: {base_signal}")
        
        # ML
        ml_signal, ml_conf = ml_predictor.predict_signal(df, base_signal)
        if ml_signal is None or ml_conf < CONFIDENCE_THRESHOLD:
            print(f"[SIGNAL] ❌ ML ({ml_conf:.1%})")
            return None
        
        print(f"[SIGNAL] ✅ ML: {ml_signal} ({ml_conf:.1%})")
        
        # Persister
        entry_time_haiti = now_haiti + timedelta(minutes=DELAY_BEFORE_ENTRY_MIN)
        entry_time_utc = entry_time_haiti.astimezone(timezone.utc)
        
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
        
        # Envoyer abonnés
        with engine.connect() as conn:
            user_ids = [r[0] for r in conn.execute(text("SELECT user_id FROM subscribers")).fetchall()]
        
        direction_text = "BUY" if ml_signal == "CALL" else "SELL"
        
        msg = (
            f"🎯 SIGNAL — {pair}\n\n"
            f"📅 Session: {session['name']}\n"
            f"🔥 Priorité: {session['priority']}/5\n"
            f"🕐 Entrée: {entry_time_haiti.strftime('%H:%M')}\n"
            f"📍 Timeframe: M5\n\n"
            f"📈 Direction: **{direction_text}**\n"
            f"💪 Confiance: **{int(ml_conf*100)}%**\n\n"
            f"🔍 Briefing après 6-7 min"
        )
        
        sent = 0
        for uid in user_ids:
            try:
                await app.bot.send_message(chat_id=uid, text=msg)
                sent += 1
            except Exception as e:
                print(f"[SIGNAL] ❌ {uid}: {e}")
        
        print(f"[SIGNAL] ✅ Envoyé à {sent} abonnés")
        
        # Tracking
        if session['name'] not in active_sessions:
            active_sessions[session['name']] = []
        active_sessions[session['name']].append(signal_id)
        
        # Ajouter à queue vérif avec timestamp
        verification_time_utc = entry_time_utc + timedelta(minutes=VERIFICATION_WAIT_MIN)
        pending_verifications.append({
            'signal_id': signal_id,
            'verification_time': verification_time_utc,
            'app': app
        })
        
        return signal_id
        
    except Exception as e:
        print(f"[SIGNAL] ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

async def send_verification_briefing(signal_id, app):
    """Envoie briefing pour un signal vérifié"""
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
            f"{emoji} **BRIEFING**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{direction_emoji} {pair}\n"
            f"📊 {direction}\n"
            f"💪 {int(confidence*100)}%\n"
        )
        
        if kill_zone:
            briefing += f"📅 {kill_zone}\n"
        
        briefing += f"\n🎲 **{status}**\n\n━━━━━━━━━━━━━━━━━━━━"
        
        sent = 0
        for uid in user_ids:
            try:
                await app.bot.send_message(chat_id=uid, text=briefing)
                sent += 1
            except:
                pass
        
        print(f"[BRIEFING] ✅ #{signal_id} {status} → {sent} abonnés")

    except Exception as e:
        print(f"[BRIEFING] ❌ #{signal_id}: {e}")

async def run_scheduled_session(app, session):
    """Exécute session planifiée"""
    if not is_forex_open():
        print(f"[SESSION] 🏖️ Fermé - {session['name']}")
        return
    
    if session_running.get(session['name'], False):
        print(f"[SESSION] ⚠️ {session['name']} déjà running")
        return
    
    session_running[session['name']] = True
    
    print(f"\n[SESSION] 🚀 DÉBUT - {session['name']}")
    print(f"[SESSION] 🔥 Priorité: {session['priority']}/5")
    
    active_sessions[session['name']] = []
    
    try:
        if session.get('continuous'):
            # Continu
            print(f"[SESSION] ⚡ INTENSIF - 10min → {session['end_hour']:02d}h")
            
            signal_count = 0
            while True:
                current_session = get_current_session()
                if not current_session or current_session['name'] != session['name']:
                    print(f"[SESSION] ⏰ Fin")
                    break
                
                if not is_forex_open():
                    print(f"[SESSION] 🏖️ Fermé")
                    break
                
                signal_count += 1
                print(f"\n[SESSION] 📍 Signal #{signal_count}")
                
                signal_id = await send_single_signal(app, session)
                
                if signal_id:
                    print(f"[SESSION] ✅ #{signal_count} envoyé")
                else:
                    print(f"[SESSION] ⏭️ #{signal_count} non généré")
                
                print(f"[SESSION] ⏸️ Pause 10min...")
                await asyncio.sleep(600)
            
            sent = len(active_sessions.get(session['name'], []))
            print(f"\n[SESSION] 🏁 FIN - {sent} signaux")
            
        else:
            # Standard
            print(f"[SESSION] ⚡ {session['signals_count']} signaux")
            
            for i in range(session['signals_count']):
                if not is_forex_open():
                    break
                
                print(f"\n[SESSION] 📍 {i+1}/{session['signals_count']}")
                
                signal_sent = False
                for attempt in range(3):
                    signal_id = await send_single_signal(app, session)
                    if signal_id:
                        signal_sent = True
                        break
                    
                    if attempt < 2:
                        await asyncio.sleep(20)
                
                if not signal_sent:
                    print(f"[SESSION] ⚠️ #{i+1} non envoyé")
                
                if i < session['signals_count'] - 1:
                    await asyncio.sleep(session['interval_minutes'] * 60)
            
            sent = len(active_sessions.get(session['name'], []))
            print(f"\n[SESSION] 🏁 FIN - {sent}/{session['signals_count']}")
    
    finally:
        session_running[session['name']] = False

async def automated_verification_check(app):
    """Vérif auto + briefings"""
    try:
        print("\n[AUTO-VERIF] 🔍 Programmée...")
        
        # IDs avant vérif
        with engine.connect() as conn:
            pending_before = conn.execute(
                text("SELECT id FROM signals WHERE result IS NULL")
            ).fetchall()
            pending_ids = [row[0] for row in pending_before]
        
        if not pending_ids:
            print(f"[AUTO-VERIF] ℹ️ Aucun signal à vérifier")
            return
        
        print(f"[AUTO-VERIF] 📊 {len(pending_ids)} à vérifier")
        
        # Vérifier
        await auto_verifier.verify_pending_signals()
        
        # IDs vérifiés
        if not pending_ids:
            print(f"[AUTO-VERIF] ✅ Terminée")
            return
        
        with engine.connect() as conn:
            # SQLAlchemy nécessite dict ou bindparams explicites
            placeholders = ','.join([f':id{i}' for i in range(len(pending_ids))])
            params = {f'id{i}': pid for i, pid in enumerate(pending_ids)}
            
            verified = conn.execute(
                text(f"SELECT id, result FROM signals WHERE id IN ({placeholders}) AND result IS NOT NULL"),
                params
            ).fetchall()
        
        print(f"[AUTO-VERIF] ✅ {len(verified)} vérifiés")
        
        # Briefings
        for signal_id, result in verified:
            try:
                await send_verification_briefing(signal_id, app)
            except Exception as e:
                print(f"[AUTO-VERIF] ⚠️ Briefing #{signal_id}: {e}")
        
        print(f"[AUTO-VERIF] 📧 {len(verified)} briefings envoyés")
        print("[AUTO-VERIF] ✅ Terminée")
        
    except Exception as e:
        print(f"[AUTO-VERIF] ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

async def send_daily_report(app):
    """Rapport quotidien"""
    try:
        print("\n[RAPPORT] 📊 Génération...")
        
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
            print("[RAPPORT] ⚠️ Aucun signal")
            return
        
        total, wins, losses = stats
        verified = wins + losses
        winrate = (wins / verified * 100) if verified > 0 else 0
        
        report = (
            f"📊 **RAPPORT QUOTIDIEN**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📅 {now_haiti.strftime('%d/%m/%Y')}\n\n"
            f"• Total: {total}\n"
            f"• ✅ Wins: {wins}\n"
            f"• ❌ Losses: {losses}\n"
            f"• 📊 WR: **{winrate:.1f}%**\n\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
        
        for uid in user_ids:
            try:
                await app.bot.send_message(chat_id=uid, text=report)
            except:
                pass
        
        print(f"[RAPPORT] ✅ Envoyé ({winrate:.1f}% WR)")
        
    except Exception as e:
        print(f"[RAPPORT] ❌ Erreur: {e}")

# ===== SERVEUR HTTP POUR RENDER =====

async def health_check(request):
    """Health check endpoint pour Render"""
    now_haiti = get_haiti_now()
    return web.json_response({
        'status': 'ok',
        'timestamp': now_haiti.isoformat(),
        'forex_open': is_forex_open(),
        'current_session': get_current_session()['name'] if get_current_session() else None
    })

async def stats_endpoint(request):
    """Stats endpoint"""
    try:
        with engine.connect() as conn:
            total = conn.execute(text('SELECT COUNT(*) FROM signals')).scalar()
            wins = conn.execute(text("SELECT COUNT(*) FROM signals WHERE result='WIN'")).scalar()
            losses = conn.execute(text("SELECT COUNT(*) FROM signals WHERE result='LOSE'")).scalar()
        
        verified = wins + losses
        winrate = (wins/verified*100) if verified > 0 else 0
        
        return web.json_response({
            'total': total,
            'wins': wins,
            'losses': losses,
            'winrate': round(winrate, 2),
            'pending': total - verified
        })
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)

async def start_http_server():
    """Démarre serveur HTTP pour Render (port 10000)"""
    app = web.Application()
    app.router.add_get('/health', health_check)
    app.router.add_get('/stats', stats_endpoint)
    app.router.add_get('/', health_check)  # Root aussi
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    # Render expose sur port 10000 par défaut
    port = int(os.getenv('PORT', 10000))
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    
    print(f"✅ HTTP server running on :{port}")
    print(f"   Health: http://0.0.0.0:{port}/health")
    print(f"   Stats: http://0.0.0.0:{port}/stats")
    
    return runner

async def main():
    global auto_verifier

    now_haiti = get_haiti_now()
    now_utc = get_utc_now()

    print("\n" + "="*60)
    print("🤖 BOT M5 - VERSION FINALE")
    print("="*60)
    print(f"🇭🇹 {now_haiti.strftime('%H:%M:%S %Z')}")
    print(f"🌍 {now_utc.strftime('%H:%M:%S %Z')}")
    print(f"📈 Forex: {'🟢 OUVERT' if is_forex_open() else '🔴 FERMÉ'}")
    
    current_session = get_current_session()
    if current_session:
        print(f"✅ Session: {current_session['name']} (P:{current_session['priority']}/5)")
        if current_session.get('continuous'):
            print(f"🔥 INTENSIF - 10min")
    else:
        next_session = get_next_session()
        print(f"⏸️ Prochaine: {next_session['name']} à {next_session['start_hour']:02d}h")
    
    print(f"\n📅 SESSIONS:")
    for s in SCHEDULED_SESSIONS:
        mode = "INTENSIF" if s.get('continuous') else f"{s['signals_count']} sig"
        print(f"• {s['name']}: {s['start_hour']:02d}h ({mode})")
    
    print(f"\n📍 M5 | Briefings auto: 15min")
    print("="*60 + "\n")

    ensure_db()
    auto_verifier = AutoResultVerifier(engine, TWELVEDATA_API_KEY)

    # Démarrer serveur HTTP pour Render
    http_runner = await start_http_server()

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Commandes
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
    app.add_handler(CommandHandler('forcesession', cmd_forcesession))
    app.add_handler(CommandHandler('backtest', cmd_backtest))
    app.add_handler(CommandHandler('cleanup', cmd_cleanup))

    sched.start()

    # Sessions
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
        print(f"✅ Planifié: {session['name']} à {session['start_hour']:02d}h{session['start_minute']:02d}")
    
    # Vérif auto - TOUTES LES 7 MINUTES pour données fraîches TwelveData
    sched.add_job(
        automated_verification_check,
        'cron',
        minute='*/7',
        timezone=HAITI_TZ,
        args=[app],
        id='auto_verification'
    )
    print(f"✅ Vérif + Briefings auto: 7min")
    
    # Rapport
    sched.add_job(
        send_daily_report,
        'cron',
        hour=22,
        minute=0,
        timezone=HAITI_TZ,
        args=[app],
        id='daily_report'
    )

    # Lancement immédiat si session active
    if current_session and is_forex_open():
        print(f"\n🚀 LANCEMENT IMMÉDIAT - {current_session['name']}")
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
        await http_runner.cleanup()
        sched.shutdown()

if __name__ == '__main__':
    asyncio.run(main())
