"""
Bot M5 avec Vérification Synchronisée - VERSION COMPLÈTE
TOUTES LES COMMANDES PRÉSENTES
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
from utils import compute_indicators, rule_signal_ultra_strict, get_signal_quality_score
from ml_predictor import MLSignalPredictor
from auto_verifier import AutoResultVerifier
from ml_continuous_learning import ContinuousLearning
from backtester import BacktesterM5

# Configuration
HAITI_TZ = ZoneInfo("America/Port-au-Prince")

SCHEDULED_SESSIONS = [
    {
        'name': 'London Kill Zone',
        'start_hour': 2,
        'start_minute': 0,
        'end_hour': 5,
        'end_minute': 0,
        'signals_count': 3,
        'interval_minutes': 30,
        'priority': 3,
        'wait_verification': True
    },
    {
        'name': 'London/NY Overlap',
        'start_hour': 9,
        'start_minute': 0,
        'end_hour': 11,
        'end_minute': 0,
        'signals_count': 4,
        'interval_minutes': 30,
        'priority': 5,
        'wait_verification': True
    },
    {
        'name': 'NY Session',
        'start_hour': 14,
        'start_minute': 0,
        'end_hour': 17,
        'end_minute': 0,
        'signals_count': 4,
        'interval_minutes': 30,
        'priority': 3,
        'wait_verification': True
    },
    {
        'name': 'Evening Session',
        'start_hour': 18,
        'start_minute': 0,
        'end_hour': 2,
        'end_minute': 0,
        'signals_count': -1,
        'interval_minutes': 15,
        'priority': 2,
        'continuous': True,
        'wait_verification': True
    }
]

# Paramètres
TIMEFRAME_M5 = "5min"
DELAY_BEFORE_ENTRY_MIN = 5
VERIFICATION_WAIT_MIN = 5
CONFIDENCE_THRESHOLD = 0.70

engine = create_engine(DB_URL, connect_args={'check_same_thread': False})
sched = AsyncIOScheduler(timezone=HAITI_TZ)
ml_predictor = MLSignalPredictor()
auto_verifier = None
active_sessions = {}
session_running = {}
last_signal_pending_verification = {}

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

# ============================================
# COMMANDES TELEGRAM - TOUTES COMPLÈTES
# ============================================

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
                    f"✅ Bienvenue au Bot Trading M5 - ULTRA STRICT !\n\n"
                    f"📅 **SESSIONS QUOTIDIENNES:**\n\n"
                    f"🌅 **02h-05h** London Kill Zone (3 signaux)\n"
                    f"🔥 **09h-11h** London/NY Overlap (4 signaux)\n"
                    f"📈 **14h-17h** NY Session (4 signaux)\n"
                    f"🌆 **18h-02h** Evening Session (intensive)\n\n"
                    f"⚡ **NOUVELLE VERSION:**\n"
                    f"• Stratégie ultra-stricte (4/5 critères)\n"
                    f"• Anti contre-tendance\n"
                    f"• Vérif AVANT signal suivant\n"
                    f"• Score qualité min: 70/100\n"
                    f"• Seuil ML: 70%\n\n"
                    f"📍 Timeframe: M5\n"
                    f"🎯 Win rate attendu: 75-85%\n"
                    f"💪 8-15 signaux/jour\n\n"
                    f"⏰ Prochaine: {next_session['name']} à {next_time}\n\n"
                    f"📋 /menu pour toutes les commandes"
                )
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    menu_text = (
        "📋 **MENU DES COMMANDES**\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        "📊 **Statistiques:**\n"
        "• /stats - Statistiques générales\n"
        "• /status - État du bot\n"
        "• /rapport - Rapport du jour\n"
        "• /sessions - Planning sessions\n\n"
        "🤖 **Machine Learning:**\n"
        "• /mlstats - Stats ML\n"
        "• /retrain - Réentraîner ML\n\n"
        "🔬 **Backtesting:**\n"
        "• /backtest - Backtest M5\n"
        "• /backtest <paire> - Paire spécifique\n\n"
        "🔧 **Contrôles:**\n"
        "• /testsignal - Test signal\n"
        "• /verify - Vérifier signaux\n"
        "• /forcesession - Force lancement session\n"
        "• /menu - Ce menu\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 M5 | Ultra-Strict | 75-85% WR"
    )
    await update.message.reply_text(menu_text)

async def cmd_sessions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    now_haiti = get_haiti_now()
    current_session = get_current_session()
    next_session = get_next_session()
    
    msg = "📅 **PLANNING SESSIONS**\n━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += f"🕐 Actuelle: {now_haiti.strftime('%H:%M')} (Haïti)\n\n"
    
    if current_session:
        is_running = session_running.get(current_session['name'], False)
        msg += f"✅ **Active:** {current_session['name']}\n"
        msg += f"🔥 Priorité: {current_session['priority']}/5\n"
        msg += f"⚙️ État: {'🟢 Running' if is_running else '⚠️ Stopped'}\n"
        msg += f"🔍 Vérif synchro: {'✅ ON' if current_session.get('wait_verification') else '❌ OFF'}\n"
        if current_session.get('continuous'):
            msg += f"⚡ Mode intensif ({current_session['interval_minutes']}min)\n\n"
        else:
            msg += f"⚡ {current_session['signals_count']} signaux\n\n"
    else:
        msg += "⏸️ Aucune session active\n\n"
    
    msg += "📋 **Planning:**\n\n"
    for session in SCHEDULED_SESSIONS:
        start = f"{session['start_hour']:02d}h{session['start_minute']:02d}"
        end = f"{session['end_hour']:02d}h{session['end_minute']:02d}"
        msg += f"**{session['name']}** ({start}-{end})\n"
        if session.get('continuous'):
            msg += f"   Mode intensif {session['interval_minutes']}min\n"
        else:
            msg += f"   {session['signals_count']} signaux\n"
        msg += f"   Vérif synchro: {'✅' if session.get('wait_verification') else '❌'}\n\n"
    
    msg += "━━━━━━━━━━━━━━━━━━━━\n"
    msg += "🎯 8-15 signaux/jour | 75-85% WR"
    
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
        msg += f"🎯 Objectif: 75-85% WR\n"
        msg += f"📍 M5 Ultra-Strict"
        
        await update.message.reply_text(msg)

    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        now_haiti = get_haiti_now()
        now_utc = get_utc_now()
        forex_open = is_forex_open()
        current_session = get_current_session()
        
        msg = f"🤖 **État du Bot**\n\n"
        msg += f"🇭🇹 Haïti: {now_haiti.strftime('%a %H:%M:%S')}\n"
        msg += f"🌍 UTC: {now_utc.strftime('%a %H:%M:%S')}\n"
        msg += f"📈 Forex: {'🟢 OUVERT' if forex_open else '🔴 FERMÉ'}\n\n"
        
        msg += f"⚙️ **Configuration:**\n"
        msg += f"• Seuil ML: {CONFIDENCE_THRESHOLD:.0%}\n"
        msg += f"• Score qualité min: 70/100\n"
        msg += f"• Stratégie: Ultra-Stricte (4/5)\n"
        msg += f"• ADX min: 18\n\n"
        
        if current_session:
            is_running = session_running.get(current_session['name'], False)
            msg += f"✅ **Session Active:** {current_session['name']}\n"
            msg += f"🔥 Priorité: {current_session['priority']}/5\n"
            msg += f"⚙️ État: {'🟢 Running' if is_running else '⚠️ Stopped'}\n"
            msg += f"🔍 Vérif synchro: {'✅ ON' if current_session.get('wait_verification') else '❌ OFF'}\n"
            if current_session.get('continuous'):
                msg += f"⚡ Mode intensif ({current_session['interval_minutes']}min)\n"
            else:
                msg += f"⚡ {current_session['signals_count']} signaux\n"
        else:
            next_session = get_next_session()
            next_time = f"{next_session['start_hour']:02d}h{next_session['start_minute']:02d}"
            msg += f"⏸️ Aucune session active\n"
            msg += f"⏰ Prochaine: {next_session['name']} à {next_time}"
        
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_verify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = await update.message.reply_text("🔍 Vérification forcée des signaux en attente...")
        await auto_verifier.verify_pending_signals()
        await msg.edit_text("✅ Vérification terminée ! /stats pour résultats.")
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_retrain(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = await update.message.reply_text("🤖 Réentraînement ML en cours...")
        
        learner = ContinuousLearning(engine)
        result = learner.retrain_model(min_signals=30, min_accuracy_improvement=0.00)
        
        if result['success']:
            if result['accepted']:
                response = (
                    f"✅ **Modèle réentraîné et accepté**\n\n"
                    f"📊 Signaux utilisés: {result['signals_count']}\n"
                    f"🎯 Accuracy: {result['accuracy']*100:.2f}%\n"
                    f"📈 Amélioration: {result['improvement']*100:+.2f}%\n\n"
                    f"Le nouveau modèle est maintenant actif."
                )
            else:
                response = (
                    f"⚠️ **Modèle réentraîné mais rejeté**\n\n"
                    f"📊 Signaux utilisés: {result['signals_count']}\n"
                    f"🎯 Accuracy: {result['accuracy']*100:.2f}%\n"
                    f"📉 Amélioration: {result['improvement']*100:+.2f}%\n\n"
                    f"Le modèle actuel est conservé (meilleur)."
                )
        else:
            response = f"❌ **Échec réentraînement**\n\n{result['reason']}"
        
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
            f"📊 Total entraînements: {stats['total_trainings']}\n"
            f"🎯 Meilleure accuracy: {stats['best_accuracy']*100:.2f}%\n"
            f"📈 Signaux utilisés: {stats['total_signals']}\n"
            f"📅 Dernier entraînement: {stats['last_training']}\n"
        )
        
        if stats['recent_trainings']:
            msg += "\n📋 **Historique récent:**\n\n"
            for t in reversed(stats['recent_trainings'][-5:]):
                date = datetime.fromisoformat(t['timestamp']).strftime('%d/%m %H:%M')
                emoji = "✅" if t.get('accepted', False) else "⚠️"
                msg += f"{emoji} {date} - Acc: {t['accuracy']*100:.1f}%\n"
        
        msg += f"\n━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"💪 Entraînement min: 30 signaux"
        
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
                    SUM(CASE WHEN result = 'LOSE' THEN 1 ELSE 0 END) as losses,
                    AVG(CASE WHEN result IS NOT NULL THEN confidence ELSE NULL END) as avg_conf
                FROM signals
                WHERE ts_send >= :start AND ts_send < :end
            """)
            
            stats = conn.execute(query, {
                "start": start_utc.isoformat(),
                "end": end_utc.isoformat()
            }).fetchone()
        
        if not stats or stats[0] == 0:
            await msg.edit_text("ℹ️ Aucun signal aujourd'hui")
            return
        
        total, wins, losses, avg_conf = stats
        verified = wins + losses if wins and losses else 0
        winrate = (wins / verified * 100) if verified > 0 else 0
        pending = total - verified
        
        report = (
            f"📊 **RAPPORT DU JOUR**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📅 {now_haiti.strftime('%d/%m/%Y %H:%M')}\n\n"
            f"📈 **PERFORMANCE:**\n"
            f"• Total envoyés: {total}\n"
            f"• ✅ Gagnés: {wins or 0}\n"
            f"• ❌ Perdus: {losses or 0}\n"
            f"• ⏳ En attente: {pending}\n"
            f"• 📊 Win rate: **{winrate:.1f}%**\n"
        )
        
        if avg_conf:
            report += f"• 💪 Confiance moy: {avg_conf*100:.1f}%\n"
        
        report += f"\n━━━━━━━━━━━━━━━━━━━━\n"
        report += f"🎯 Objectif: 75-85% WR\n"
        report += f"📍 Stratégie Ultra-Stricte"
        
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
                f"⏸️ Aucune session active\n⏰ Prochaine: {next_session['name']} à {next_time}"
            )
            return
        
        msg = await update.message.reply_text(f"🚀 Test signal pour {current_session['name']}...")
        
        app = context.application
        signal_id = await send_single_signal(app, current_session)
        
        if signal_id:
            await msg.edit_text(f"✅ Signal #{signal_id} envoyé avec stratégie ultra-stricte !")
        else:
            await msg.edit_text(
                "⚠️ Aucun signal généré\n\n"
                "Raisons possibles:\n"
                "• Tendance pas assez forte (check_strong_trend)\n"
                "• Score qualité < 70\n"
                "• Confiance ML < 70%\n"
                "• ADX < 18\n"
                "• Moins de 4/5 critères validés\n\n"
                "Consultez les logs pour détails."
            )
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_forcesession(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Force le lancement de la session active"""
    try:
        current_session = get_current_session()
        if not current_session:
            await update.message.reply_text("⏸️ Aucune session active à forcer")
            return
        
        if session_running.get(current_session['name'], False):
            await update.message.reply_text(f"⚠️ {current_session['name']} déjà en cours")
            return
        
        msg = await update.message.reply_text(f"🚀 Force lancement {current_session['name']}...")
        
        app = context.application
        asyncio.create_task(run_scheduled_session(app, current_session))
        
        await msg.edit_text(
            f"✅ {current_session['name']} lancée !\n\n"
            f"Mode: {'Intensif' if current_session.get('continuous') else 'Standard'}\n"
            f"Vérif synchro: {'✅ ON' if current_session.get('wait_verification') else '❌ OFF'}"
        )
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = await update.message.reply_text("🔬 Backtest M5 Ultra-Strict...\n⏳ 1-2 minutes...")
        
        pairs_to_test = PAIRS[:3]
        
        if context.args and len(context.args) > 0:
            requested_pair = context.args[0].upper().replace('-', '/')
            if requested_pair in PAIRS:
                pairs_to_test = [requested_pair]
            else:
                await msg.edit_text(f"❌ Paire inconnue: {requested_pair}\n\nPaires dispo: {', '.join(PAIRS)}")
                return
        
        backtester = BacktesterM5(confidence_threshold=CONFIDENCE_THRESHOLD)
        results = backtester.run_full_backtest(pairs=pairs_to_test, outputsize=3000)
        result_msg = backtester.format_results_for_telegram(results)
        
        result_msg += f"\n\n🎯 Stratégie: Ultra-Stricte\n💪 Seuil ML: {CONFIDENCE_THRESHOLD:.0%}"
        
        await msg.edit_text(result_msg)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur backtest: {str(e)[:200]}")

# ============================================
# FONCTIONS SIGNAL ET SESSION
# ============================================

async def send_single_signal(app, session):
    """Envoie un signal avec stratégie ULTRA-STRICTE"""
    try:
        if not is_forex_open():
            print("[SIGNAL] 🏖️ Marché fermé")
            return None
        
        now_haiti = get_haiti_now()
        print(f"\n[SIGNAL] 📤 {session['name']} - {now_haiti.strftime('%H:%M:%S')}")
        
        # Rotation paires
        active_pairs = PAIRS[:3]
        session_signals = active_sessions.get(session['name'], [])
        pair = active_pairs[len(session_signals) % len(active_pairs)]
        
        print(f"[SIGNAL] 🔍 Analyse {pair}...")
        
        # Récupérer données
        params = BEST_PARAMS.get(pair, {})
        df = get_cached_ohlc(pair, TIMEFRAME_M5, outputsize=400)
        
        if df is None or len(df) < 50:
            print("[SIGNAL] ❌ Pas assez de données")
            return None
        
        print(f"[SIGNAL] ✅ {len(df)} bougies chargées")
        
        # Calculer indicateurs
        df = compute_indicators(df, ema_fast=params.get('ema_fast',8),
                                ema_slow=params.get('ema_slow',21),
                                rsi_len=params.get('rsi',14),
                                bb_len=params.get('bb',20))
        
        print(f"[SIGNAL] ✅ Indicateurs calculés")
        
        # Analyser avec stratégie ULTRA-STRICTE
        base_signal = rule_signal_ultra_strict(df, session_priority=session['priority'])
        
        if not base_signal:
            print("[SIGNAL] ⏭️ Rejeté par stratégie ULTRA-STRICTE")
            last = df.iloc[-1]
            print(f"[DEBUG] ADX: {last.get('adx', 0):.1f} (min: 18)")
            print(f"[DEBUG] RSI: {last.get('rsi', 0):.1f}")
            print(f"[DEBUG] Momentum 3: {last.get('momentum_3', 0):.4f}")
            print(f"[DEBUG] Momentum 5: {last.get('momentum_5', 0):.4f}")
            print(f"[DEBUG] Momentum 10: {last.get('momentum_10', 0):.4f}")
            return None
        
        print(f"[SIGNAL] ✅ Signal stratégie: {base_signal}")
        
        # Score qualité
        quality_score = get_signal_quality_score(df)
        print(f"[SIGNAL] 📊 Score qualité: {quality_score}/100")
        
        # Rejeter si score trop faible
        if quality_score < 70:
            print(f"[SIGNAL] ❌ Score insuffisant ({quality_score} < 70)")
            return None
        
        # ML prediction avec seuil augmenté
        ml_signal, ml_conf = ml_predictor.predict_signal(df, base_signal)
        if ml_signal is None or ml_conf < CONFIDENCE_THRESHOLD:
            print(f"[SIGNAL] ❌ Rejeté par ML ({ml_conf:.1%} < {CONFIDENCE_THRESHOLD:.0%})")
            return None
        
        print(f"[SIGNAL] ✅ ML approved: {ml_signal} ({ml_conf:.1%})")
        
        # Persister signal
        entry_time_haiti = now_haiti + timedelta(minutes=DELAY_BEFORE_ENTRY_MIN)
        entry_time_utc = entry_time_haiti.astimezone(timezone.utc)
        
        payload = {
            'pair': pair, 'direction': ml_signal, 
            'reason': f'ML {ml_conf:.1%} Q{quality_score} - {session["name"]}',
            'ts_enter': entry_time_utc.isoformat(), 
            'ts_send': get_utc_now().isoformat(),
            'confidence': ml_conf, 
            'payload': json.dumps({'pair': pair, 'session': session['name'], 'quality': quality_score}),
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
        
        # Envoyer aux abonnés
        with engine.connect() as conn:
            user_ids = [r[0] for r in conn.execute(text("SELECT user_id FROM subscribers")).fetchall()]
        
        direction_text = "BUY" if ml_signal == "CALL" else "SELL"
        
        msg = (
            f"🎯 SIGNAL — {pair}\n\n"
            f"📅 Session: {session['name']}\n"
            f"🔥 Priorité: {session['priority']}/5\n"
            f"🕐 Entrée: {entry_time_haiti.strftime('%H:%M')} (Haïti)\n"
            f"📍 Timeframe: M5\n\n"
            f"📈 Direction: **{direction_text}**\n"
            f"💪 Confiance: **{int(ml_conf*100)}%**\n"
            f"⭐ Qualité: **{quality_score}/100**\n\n"
            f"🛡️ Stratégie: Ultra-Stricte (4/5)\n"
            f"🔍 Vérif: 5 min après entrée"
        )
        
        sent_count = 0
        for uid in user_ids:
            try:
                await app.bot.send_message(chat_id=uid, text=msg)
                sent_count += 1
            except Exception as e:
                print(f"[SIGNAL] ❌ Envoi à {uid}: {e}")
        
        print(f"[SIGNAL] ✅ Envoyé à {sent_count} abonnés (ID: {signal_id})")
        
        # Tracking
        if session['name'] not in active_sessions:
            active_sessions[session['name']] = []
        active_sessions[session['name']].append(signal_id)
        
        # Marquer pour vérification
        last_signal_pending_verification[session['name']] = {
            'signal_id': signal_id,
            'entry_time': entry_time_utc,
            'verification_time': entry_time_utc + timedelta(minutes=VERIFICATION_WAIT_MIN)
        }
        
        return signal_id
        
    except Exception as e:
        print(f"[SIGNAL] ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

async def wait_and_verify_signal(app, signal_id, verification_time):
    """Attend puis vérifie un signal"""
    try:
        now = get_utc_now()
        wait_seconds = (verification_time - now).total_seconds()
        
        if wait_seconds > 0:
            print(f"[VERIF] ⏳ Attente {int(wait_seconds)}s avant vérification signal #{signal_id}")
            await asyncio.sleep(wait_seconds)
        
        print(f"[VERIF] 🔍 Vérification signal #{signal_id}")
        
        # Vérifier via auto_verifier
        verified = await auto_verifier.verify_single_signal(signal_id)
        
        if verified:
            # Récupérer résultat
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT pair, direction, result, confidence FROM signals WHERE id = :sid"),
                    {'sid': signal_id}
                ).fetchone()
            
            if result and result[2]:
                pair, direction, outcome, confidence = result
                emoji = "✅" if outcome == "WIN" else "❌"
                status = "GAGNÉ" if outcome == "WIN" else "PERDU"
                
                print(f"[VERIF] {emoji} Signal #{signal_id}: {status}")
                
                # Envoyer briefing
                with engine.connect() as conn:
                    user_ids = [r[0] for r in conn.execute(text("SELECT user_id FROM subscribers")).fetchall()]
                
                briefing = (
                    f"{emoji} **RÉSULTAT SIGNAL #{signal_id}**\n\n"
                    f"📊 {pair} {direction}\n"
                    f"💪 Confiance: {int(confidence*100)}%\n"
                    f"🎲 Résultat: **{status}**"
                )
                
                for uid in user_ids:
                    try:
                        await app.bot.send_message(chat_id=uid, text=briefing)
                    except:
                        pass
                
                return outcome == "WIN"
        
        return False
        
    except Exception as e:
        print(f"[VERIF] ❌ Erreur: {e}")
        return False

async def run_scheduled_session(app, session):
    """Exécute une session avec vérification synchronisée"""
    if not is_forex_open():
        print(f"[SESSION] 🏖️ Marché fermé - {session['name']}")
        return
    
    if session_running.get(session['name'], False):
        print(f"[SESSION] ⚠️ {session['name']} déjà en cours")
        return
    
    session_running[session['name']] = True
    
    print(f"\n[SESSION] 🚀 DÉBUT - {session['name']}")
    print(f"[SESSION] 🔥 Priorité: {session['priority']}/5")
    print(f"[SESSION] 🔍 Vérif synchro: {'✅ ACTIVÉE' if session.get('wait_verification') else '❌ DÉSACTIVÉE'}")
    print(f"[SESSION] 🛡️ Stratégie: Ultra-Stricte (4/5 critères)")
    
    active_sessions[session['name']] = []
    
    try:
        if session.get('continuous'):
            # Mode continu avec vérification
            print(f"[SESSION] ⚡ Mode INTENSIF - {session['interval_minutes']}min jusqu'à {session['end_hour']:02d}h")
            
            signal_count = 0
            while True:
                current_session = get_current_session()
                if not current_session or current_session['name'] != session['name']:
                    print(f"[SESSION] ⏰ Fin de session atteinte")
                    break
                
                if not is_forex_open():
                    print(f"[SESSION] 🏖️ Marché fermé - Arrêt")
                    break
                
                signal_count += 1
                print(f"\n[SESSION] 📍 Signal #{signal_count}")
                
                # Envoyer signal
                signal_id = await send_single_signal(app, session)
                
                if signal_id:
                    print(f"[SESSION] ✅ Signal #{signal_count} envoyé (ID: {signal_id})")
                    
                    # Attendre vérification si activée
                    if session.get('wait_verification'):
                        pending = last_signal_pending_verification.get(session['name'])
                        if pending:
                            print(f"[SESSION] ⏳ Attente vérification signal #{signal_id}...")
                            win = await wait_and_verify_signal(app, signal_id, pending['verification_time'])
                            print(f"[SESSION] {'✅ WIN' if win else '❌ LOSE'} - Vérification terminée")
                    
                else:
                    print(f"[SESSION] ⏭️ Signal #{signal_count} non généré (conditions strictes)")
                
                # Attendre intervalle
                print(f"[SESSION] ⏸️ Pause {session['interval_minutes']}min avant prochain signal...")
                await asyncio.sleep(session['interval_minutes'] * 60)
            
            signals_sent = len(active_sessions.get(session['name'], []))
            print(f"\n[SESSION] 🏁 FIN - {session['name']}")
            print(f"[SESSION] 📊 {signals_sent} signaux envoyés (mode intensif)")
            
        else:
            # Mode standard avec vérification
            print(f"[SESSION] ⚡ {session['signals_count']} signaux à {session['interval_minutes']}min d'intervalle")
            
            for i in range(session['signals_count']):
                if not is_forex_open():
                    print(f"[SESSION] 🏖️ Marché fermé - Arrêt session")
                    break
                
                print(f"\n[SESSION] 📍 Signal {i+1}/{session['signals_count']}")
                
                # 3 tentatives
                signal_sent = False
                signal_id = None
                for attempt in range(3):
                    signal_id = await send_single_signal(app, session)
                    if signal_id:
                        signal_sent = True
                        break
                    
                    if attempt < 2:
                        print(f"[SESSION] ⏳ Nouvelle tentative dans 20s...")
                        await asyncio.sleep(20)
                
                if signal_sent and signal_id:
                    # Attendre vérification si activée
                    if session.get('wait_verification'):
                        pending = last_signal_pending_verification.get(session['name'])
                        if pending:
                            print(f"[SESSION] ⏳ Attente vérification signal #{signal_id}...")
                            win = await wait_and_verify_signal(app, signal_id, pending['verification_time'])
                            print(f"[SESSION] {'✅ WIN' if win else '❌ LOSE'} - Vérification terminée")
                else:
                    print(f"[SESSION] ⚠️ Signal {i+1} non envoyé (critères non atteints)")
                
                # Attendre intervalle
                if i < session['signals_count'] - 1:
                    print(f"[SESSION] ⏸️ Pause {session['interval_minutes']}min avant prochain signal...")
                    await asyncio.sleep(session['interval_minutes'] * 60)
            
            signals_sent = len(active_sessions.get(session['name'], []))
            print(f"\n[SESSION] 🏁 FIN - {session['name']}")
            print(f"[SESSION] 📊 {signals_sent}/{session['signals_count']} signaux envoyés")
    
    finally:
        session_running[session['name']] = False

async def automated_verification_check(app):
    """Vérification auto toutes les 15min (backup si vérif synchro échoue)"""
    try:
        print("\n[AUTO-VERIF] 🔍 Vérification backup programmée...")
        await auto_verifier.verify_pending_signals()
        print("[AUTO-VERIF] ✅ Terminée")
    except Exception as e:
        print(f"[AUTO-VERIF] ❌ Erreur: {e}")

async def send_daily_report(app):
    """Rapport quotidien 22h"""
    try:
        print("\n[RAPPORT] 📊 Génération rapport quotidien...")
        
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
            f"📅 {now_haiti.strftime('%d/%m/%Y')}\n\n"
            f"• Total: {total}\n"
            f"• ✅ Wins: {wins}\n"
            f"• ❌ Losses: {losses}\n"
            f"• 📊 Win rate: **{winrate:.1f}%**\n\n"
            f"🎯 Objectif: 75-85%\n"
            f"🛡️ Stratégie: Ultra-Stricte\n\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
        
        for uid in user_ids:
            try:
                await app.bot.send_message(chat_id=uid, text=report)
            except:
                pass
        
        print(f"[RAPPORT] ✅ Envoyé (WR: {winrate:.1f}%)")
        
    except Exception as e:
        print(f"[RAPPORT] ❌ Erreur: {e}")

async def main():
    global auto_verifier

    now_haiti = get_haiti_now()
    now_utc = get_utc_now()

    print("\n" + "="*60)
    print("🤖 BOT M5 - ULTRA-STRICT + VÉRIF SYNCHRONISÉE")
    print("="*60)
    print(f"🇭🇹 Haïti: {now_haiti.strftime('%H:%M:%S %Z')}")
    print(f"🌍 UTC: {now_utc.strftime('%H:%M:%S %Z')}")
    print(f"📈 Forex: {'🟢 OUVERT' if is_forex_open() else '🔴 FERMÉ'}")
    
    current_session = get_current_session()
    if current_session:
        print(f"✅ Session: {current_session['name']} (P:{current_session['priority']}/5)")
        if current_session.get('continuous'):
            print(f"🔥 Mode INTENSIF - {current_session['interval_minutes']}min")
    else:
        next_session = get_next_session()
        print(f"⏸️ Prochaine: {next_session['name']} à {next_session['start_hour']:02d}h")
    
    print(f"\n⚙️ CONFIGURATION:")
    print(f"• Stratégie: Ultra-Stricte (4/5 critères)")
    print(f"• ADX min: 18")
    print(f"• Score qualité min: 70/100")
    print(f"• Seuil ML: {CONFIDENCE_THRESHOLD:.0%}")
    print(f"• Vérif synchronisée: ✅ ACTIVÉE")
    print(f"• Evening intervalle: 15min")
    
    print(f"\n📅 SESSIONS:")
    for s in SCHEDULED_SESSIONS:
        mode = "INTENSIF" if s.get('continuous') else f"{s['signals_count']} signaux"
        verif = "✅" if s.get('wait_verification') else "❌"
        print(f"• {s['name']}: {s['start_hour']:02d}h-{s['end_hour']:02d}h ({mode}) [Vérif: {verif}]")
    
    print(f"\n🎯 Win rate attendu: 75-85%")
    print(f"💪 Signaux/jour: 8-15")
    print("="*60 + "\n")

    ensure_db()
    auto_verifier = AutoResultVerifier(engine, TWELVEDATA_API_KEY)

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    # TOUTES LES COMMANDES
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

    sched.start()

    # Sessions planifiées
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
    
    # Vérification auto backup (15min)
    sched.add_job(
        automated_verification_check,
        'cron',
        minute='*/15',
        timezone=HAITI_TZ,
        args=[app],
        id='auto_verification_backup'
    )
    print(f"✅ Vérif backup: 15min")
    
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
    print(f"✅ Rapport quotidien: 22h00")

    # Lancement immédiat si session active
    if current_session and is_forex_open():
        print(f"\n🚀 LANCEMENT IMMÉDIAT - {current_session['name']}")
        asyncio.create_task(run_scheduled_session(app, current_session))

    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)

    bot_info = await app.bot.get_me()
    print(f"\n✅ BOT ACTIF: @{bot_info.username}")
    print("="*60 + "\n")

    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 Arrêt du bot...")
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
        sched.shutdown()
        print("👋 Bot arrêté proprement")

if __name__ == '__main__':
    asyncio.run(main())
