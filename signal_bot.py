"""
Bot de trading M1 - Version Interactive
8 signaux par session avec bouton Generate Signal
"""

import os, json, asyncio
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import requests
import pandas as pd
from sqlalchemy import create_engine, text
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
from aiohttp import web
from config import *
from utils import compute_indicators, rule_signal_ultra_strict
from ml_predictor import MLSignalPredictor
from auto_verifier import AutoResultVerifier

# Configuration
HAITI_TZ = ZoneInfo("America/Port-au-Prince")
TIMEFRAME_M1 = "1min"
SIGNALS_PER_SESSION = 8
VERIFICATION_WAIT_MIN = 2  # M1: vérifier après 2 minutes
CONFIDENCE_THRESHOLD = 0.65

engine = create_engine(DB_URL, connect_args={'check_same_thread': False})
ml_predictor = MLSignalPredictor()
auto_verifier = None

# Sessions actives (user_id -> session_data)
active_sessions = {}

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

def fetch_ohlc_td(pair, interval, outputsize=300):
    if not is_forex_open():
        raise RuntimeError("Marché Forex fermé")
    
    params = {'symbol': pair, 'interval': interval, 'outputsize': outputsize,
    'apikey': TWELVEDATA_API_KEY, 'format':'JSON'}
    r = requests.get(TWELVE_TS_URL, params=params, timeout=10)
    r.raise_for_status()
    j = r.json()
    
    if 'code' in j and j['code'] == 429:
        raise RuntimeError(f"Limite API atteinte")
    
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
        if (current_time - cached_time).total_seconds() < 30:
            return cached_data
    
    try:
        df = fetch_ohlc_td(pair, interval, outputsize)
        ohlc_cache[cache_key] = (df, current_time)
        return df
    except RuntimeError as e:
        print(f"⚠️ Cache OHLC: {e}")
        return None

def persist_signal(payload):
    q = text("""INSERT INTO signals (pair,direction,reason,ts_enter,ts_send,confidence,payload_json,max_gales,timeframe)
    VALUES (:pair,:direction,:reason,:ts_enter,:ts_send,:confidence,:payload,:max_gales,:timeframe)""")
    with engine.begin() as conn:
        result = conn.execute(q, payload)
    return result.lastrowid

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
            
            if 'timeframe' not in existing_cols:
                conn.execute(text("ALTER TABLE signals ADD COLUMN timeframe INTEGER DEFAULT 1"))
            
            print("✅ Base de données prête")

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
            if not existing:
                conn.execute(text("INSERT INTO subscribers (user_id, username) VALUES (:uid, :uname)"),
                {"uid": user_id, "uname": username})
        
        await update.message.reply_text(
            "✅ **Bienvenue au Bot Trading M1 !**\n\n"
            "🎯 Mode: **Interactive Session**\n"
            "📊 8 signaux M1 par session\n"
            "⚡ Vérification auto après 2 min\n\n"
            "**Commandes:**\n"
            "• /startsession - Démarrer session\n"
            "• /stats - Statistiques\n"
            "• /menu - Menu complet\n\n"
            "━━━━━━━━━━━━━━━━━━━━"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    menu_text = (
        "📋 **MENU**\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        "**Session:**\n"
        "• /startsession - Nouvelle session\n"
        "• /sessionstatus - État session\n\n"
        "**Info:**\n"
        "• /stats - Statistiques\n"
        "• /rapport - Rapport jour\n\n"
        "**ML:**\n"
        "• /mlstats - Stats ML\n"
        "• /retrain - Réentraîner\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "🎯 M1 | 8 signaux/session"
    )
    await update.message.reply_text(menu_text)

async def cmd_start_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Démarre une nouvelle session de 8 signaux"""
    user_id = update.effective_user.id
    
    # Vérifier si session active
    if user_id in active_sessions:
        session = active_sessions[user_id]
        await update.message.reply_text(
            f"⚠️ Session déjà active !\n\n"
            f"📊 Progression: {session['signal_count']}/{SIGNALS_PER_SESSION}\n"
            f"✅ Wins: {session['wins']}\n"
            f"❌ Losses: {session['losses']}\n\n"
            f"Utilisez /endsession pour terminer"
        )
        return
    
    if not is_forex_open():
        await update.message.reply_text("🏖️ Marché fermé")
        return
    
    # Créer nouvelle session
    now_haiti = get_haiti_now()
    active_sessions[user_id] = {
        'start_time': now_haiti,
        'signal_count': 0,
        'wins': 0,
        'losses': 0,
        'pending': 0,
        'signals': []
    }
    
    # Bouton pour générer premier signal
    keyboard = [[InlineKeyboardButton("🎯 Generate Signal #1", callback_data=f"gen_signal_{user_id}")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🚀 **SESSION DÉMARRÉE**\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📅 {now_haiti.strftime('%H:%M:%S')}\n"
        f"🎯 Objectif: {SIGNALS_PER_SESSION} signaux M1\n"
        f"⚡ Vérification: 2 min auto\n\n"
        f"Cliquez pour générer signal #1 ⬇️",
        reply_markup=reply_markup
    )

async def cmd_session_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche l'état de la session"""
    user_id = update.effective_user.id
    
    if user_id not in active_sessions:
        await update.message.reply_text("ℹ️ Aucune session active\n\nUtilisez /startsession")
        return
    
    session = active_sessions[user_id]
    duration = (get_haiti_now() - session['start_time']).total_seconds() / 60
    winrate = (session['wins'] / session['signal_count'] * 100) if session['signal_count'] > 0 else 0
    
    msg = (
        "📊 **ÉTAT SESSION**\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        f"⏱️ Durée: {duration:.1f} min\n"
        f"📈 Progression: {session['signal_count']}/{SIGNALS_PER_SESSION}\n\n"
        f"✅ Wins: {session['wins']}\n"
        f"❌ Losses: {session['losses']}\n"
        f"⏳ En attente: {session['pending']}\n\n"
        f"📊 Win Rate: {winrate:.1f}%\n"
        "━━━━━━━━━━━━━━━━━━━━"
    )
    
    await update.message.reply_text(msg)

async def callback_generate_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Callback pour générer un signal"""
    query = update.callback_query
    await query.answer()
    
    user_id = int(query.data.split('_')[2])
    
    # Vérifier session
    if user_id not in active_sessions:
        await query.edit_message_text("❌ Session expirée\n\nUtilisez /startsession")
        return
    
    session = active_sessions[user_id]
    
    # Vérifier limite
    if session['signal_count'] >= SIGNALS_PER_SESSION:
        await end_session_summary(user_id, context.application, query.message)
        return
    
    # Générer signal
    await query.edit_message_text("⏳ Génération signal M1...")
    
    signal_id = await generate_m1_signal(user_id, context.application)
    
    if signal_id:
        session['signal_count'] += 1
        session['pending'] += 1
        session['signals'].append(signal_id)
        
        # Programmer vérification auto
        asyncio.create_task(auto_verify_signal(signal_id, user_id, context.application))
        
        await query.edit_message_text(
            f"✅ **Signal #{session['signal_count']} généré**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🔍 Vérification dans 2 min...\n"
            f"📊 Progression: {session['signal_count']}/{SIGNALS_PER_SESSION}"
        )
    else:
        await query.edit_message_text(
            "⚠️ Aucun signal (conditions non remplies)\n\n"
            "Réessayez dans quelques secondes"
        )
        
        # Proposer de réessayer
        keyboard = [[InlineKeyboardButton("🔄 Réessayer", callback_data=f"gen_signal_{user_id}")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.reply_text("Voulez-vous réessayer ?", reply_markup=reply_markup)

async def generate_m1_signal(user_id, app):
    """Génère un signal M1"""
    try:
        print(f"\n[SIGNAL] 📤 M1 pour user {user_id}")
        
        # Rotation paires
        active_pairs = PAIRS[:3]
        session = active_sessions.get(user_id)
        pair = active_pairs[session['signal_count'] % len(active_pairs)]
        
        print(f"[SIGNAL] 🔍 {pair}...")
        
        # Données M1
        df = get_cached_ohlc(pair, TIMEFRAME_M1, outputsize=400)
        
        if df is None or len(df) < 50:
            print("[SIGNAL] ❌ Pas de données")
            return None
        
        print(f"[SIGNAL] ✅ {len(df)} bougies M1")
        
        # Indicateurs
        df = compute_indicators(df)
        
        # Stratégie
        base_signal = rule_signal_ultra_strict(df, session_priority=5)
        
        if not base_signal:
            print("[SIGNAL] ⏭️ Rejeté (stratégie)")
            return None
        
        print(f"[SIGNAL] ✅ Stratégie: {base_signal}")
        
        # ML
        ml_signal, ml_conf = ml_predictor.predict_signal(df, base_signal)
        if ml_signal is None or ml_conf < CONFIDENCE_THRESHOLD:
            print(f"[SIGNAL] ❌ ML ({ml_conf:.1%})")
            return None
        
        print(f"[SIGNAL] ✅ ML: {ml_signal} ({ml_conf:.1%})")
        
        # Persister
        now_haiti = get_haiti_now()
        entry_time_haiti = now_haiti + timedelta(minutes=1)
        entry_time_utc = entry_time_haiti.astimezone(timezone.utc)
        
        payload = {
            'pair': pair, 
            'direction': ml_signal, 
            'reason': f'M1 Session - ML {ml_conf:.1%}',
            'ts_enter': entry_time_utc.isoformat(), 
            'ts_send': get_utc_now().isoformat(),
            'confidence': ml_conf, 
            'payload': json.dumps({'pair': pair, 'user_id': user_id}),
            'max_gales': 0,
            'timeframe': 1
        }
        signal_id = persist_signal(payload)
        
        # Envoyer à l'utilisateur
        direction_text = "BUY ↗️" if ml_signal == "CALL" else "SELL ↘️"
        
        msg = (
            f"🎯 **SIGNAL M1 #{session['signal_count'] + 1}**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"💱 {pair}\n"
            f"📈 Direction: **{direction_text}**\n"
            f"💪 Confiance: **{int(ml_conf*100)}%**\n"
            f"🕐 Entrée: {entry_time_haiti.strftime('%H:%M')}\n\n"
            f"🔍 Vérification auto dans 2 min..."
        )
        
        try:
            await app.bot.send_message(chat_id=user_id, text=msg)
        except Exception as e:
            print(f"[SIGNAL] ❌ Envoi: {e}")
        
        return signal_id
        
    except Exception as e:
        print(f"[SIGNAL] ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

async def auto_verify_signal(signal_id, user_id, app):
    """Vérifie automatiquement un signal après 2 minutes"""
    try:
        # Attendre 2 minutes
        await asyncio.sleep(120)
        
        print(f"\n[VERIF] 🔍 Signal #{signal_id}")
        
        # Vérifier
        result = await auto_verifier.verify_single_signal(signal_id)
        
        if not result:
            print(f"[VERIF] ⚠️ Impossible de vérifier #{signal_id}")
            return
        
        # Mettre à jour session
        if user_id in active_sessions:
            session = active_sessions[user_id]
            session['pending'] -= 1
            
            if result == 'WIN':
                session['wins'] += 1
            else:
                session['losses'] += 1
        
        # Récupérer détails
        with engine.connect() as conn:
            signal = conn.execute(
                text("SELECT pair, direction, confidence FROM signals WHERE id = :sid"),
                {"sid": signal_id}
            ).fetchone()
        
        if not signal:
            return
        
        pair, direction, confidence = signal
        
        # Envoyer résultat
        emoji = "✅" if result == "WIN" else "❌"
        status = "GAGNÉ" if result == "WIN" else "PERDU"
        direction_emoji = "📈" if direction == "CALL" else "📉"
        
        briefing = (
            f"{emoji} **RÉSULTAT**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{direction_emoji} {pair} - {direction}\n"
            f"💪 {int(confidence*100)}%\n\n"
            f"🎲 **{status}**\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
        
        # Vérifier si session toujours active
        if user_id in active_sessions:
            session = active_sessions[user_id]
            
            # Ajouter bouton si pas terminé
            if session['signal_count'] < SIGNALS_PER_SESSION:
                next_num = session['signal_count'] + 1
                keyboard = [[InlineKeyboardButton(f"🎯 Generate Signal #{next_num}", callback_data=f"gen_signal_{user_id}")]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                briefing += f"\n\n📊 {session['signal_count']}/{SIGNALS_PER_SESSION} signaux"
                
                await app.bot.send_message(chat_id=user_id, text=briefing, reply_markup=reply_markup)
            else:
                # Session terminée
                await app.bot.send_message(chat_id=user_id, text=briefing)
                await end_session_summary(user_id, app)
        else:
            await app.bot.send_message(chat_id=user_id, text=briefing)
        
        print(f"[VERIF] ✅ Briefing #{signal_id} envoyé ({result})")
        
    except Exception as e:
        print(f"[VERIF] ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

async def end_session_summary(user_id, app, message=None):
    """Envoie le résumé de fin de session"""
    if user_id not in active_sessions:
        return
    
    session = active_sessions[user_id]
    duration = (get_haiti_now() - session['start_time']).total_seconds() / 60
    winrate = (session['wins'] / session['signal_count'] * 100) if session['signal_count'] > 0 else 0
    
    summary = (
        "🏁 **SESSION TERMINÉE**\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        f"⏱️ Durée: {duration:.1f} min\n"
        f"📊 Signaux: {session['signal_count']}/{SIGNALS_PER_SESSION}\n\n"
        f"✅ Wins: {session['wins']}\n"
        f"❌ Losses: {session['losses']}\n"
        f"📈 Win Rate: **{winrate:.1f}%**\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "Utilisez /startsession pour nouvelle session"
    )
    
    # Bouton nouvelle session
    keyboard = [[InlineKeyboardButton("🚀 Nouvelle Session", callback_data="new_session")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if message:
        await message.reply_text(summary, reply_markup=reply_markup)
    else:
        await app.bot.send_message(chat_id=user_id, text=summary, reply_markup=reply_markup)
    
    # Supprimer session
    del active_sessions[user_id]

async def callback_new_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Callback pour démarrer nouvelle session"""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    
    # Simuler commande /startsession
    await query.message.delete()
    
    # Créer update simulé
    from telegram import Message, Chat, User
    fake_message = query.message
    fake_update = Update(update_id=0, message=fake_message)
    fake_update.effective_user = query.from_user
    
    await cmd_start_session(fake_update, context)

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        with engine.connect() as conn:
            total = conn.execute(text('SELECT COUNT(*) FROM signals WHERE timeframe = 1')).scalar()
            wins = conn.execute(text("SELECT COUNT(*) FROM signals WHERE result='WIN' AND timeframe = 1")).scalar()
            losses = conn.execute(text("SELECT COUNT(*) FROM signals WHERE result='LOSE' AND timeframe = 1")).scalar()

        verified = wins + losses
        winrate = (wins/verified*100) if verified > 0 else 0

        msg = (
            f"📊 **Statistiques M1**\n\n"
            f"Total: {total}\n"
            f"✅ Wins: {wins}\n"
            f"❌ Losses: {losses}\n"
            f"📈 Win rate: {winrate:.1f}%\n\n"
            f"🎯 8 signaux/session"
        )
        
        await update.message.reply_text(msg)

    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_rapport(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Rapport quotidien M1"""
    try:
        msg = await update.message.reply_text("📊 Génération rapport...")
        
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
                AND timeframe = 1
                AND result IS NOT NULL
            """)
            
            stats = conn.execute(query, {
                "start": start_utc.isoformat(),
                "end": end_utc.isoformat()
            }).fetchone()
        
        if not stats or stats[0] == 0:
            await msg.edit_text("ℹ️ Aucun signal M1 aujourd'hui")
            return
        
        total, wins, losses = stats
        verified = wins + losses
        winrate = (wins / verified * 100) if verified > 0 else 0
        
        report = (
            f"📊 **RAPPORT M1**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📅 {now_haiti.strftime('%d/%m/%Y')}\n\n"
            f"• Total: {total}\n"
            f"• ✅ Wins: {wins}\n"
            f"• ❌ Losses: {losses}\n"
            f"• 📊 Win Rate: **{winrate:.1f}%**\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 Timeframe: M1"
        )
        
        await msg.edit_text(report)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_mlstats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Statistiques ML"""
    try:
        from ml_continuous_learning import ContinuousLearning
        
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

async def cmd_retrain(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Réentraîner le modèle ML"""
    try:
        from ml_continuous_learning import ContinuousLearning
        
        msg = await update.message.reply_text("🤖 Réentraînement ML...\n⏳ Cela peut prendre 1-2 minutes...")
        
        learner = ContinuousLearning(engine)
        result = learner.retrain_model(min_signals=30, min_accuracy_improvement=0.00)
        
        if result['success']:
            if result['accepted']:
                response = (
                    f"✅ **Modèle réentraîné**\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"📊 Signaux: {result['signals_count']}\n"
                    f"🎯 Accuracy: {result['accuracy']*100:.2f}%\n"
                    f"📈 Amélioration: {result['improvement']*100:+.2f}%\n\n"
                    f"✨ {result['reason']}"
                )
            else:
                response = (
                    f"⚠️ **Modèle rejeté**\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"📊 Signaux: {result['signals_count']}\n"
                    f"🎯 Accuracy: {result['accuracy']*100:.2f}%\n"
                    f"📉 Amélioration: {result['improvement']*100:+.2f}%\n\n"
                    f"ℹ️ {result['reason']}"
                )
        else:
            response = f"❌ **Échec réentraînement**\n\n{result['reason']}"
        
        await msg.edit_text(response)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

# ===== SERVEUR HTTP =====

async def health_check(request):
    return web.json_response({
        'status': 'ok',
        'timestamp': get_haiti_now().isoformat(),
        'forex_open': is_forex_open(),
        'active_sessions': len(active_sessions)
    })

async def start_http_server():
    app = web.Application()
    app.router.add_get('/health', health_check)
    app.router.add_get('/', health_check)
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    port = int(os.getenv('PORT', 10000))
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    
    print(f"✅ HTTP server running on :{port}")
    return runner

async def main():
    global auto_verifier

    print("\n" + "="*60)
    print("🤖 BOT M1 - VERSION INTERACTIVE")
    print("="*60)
    print(f"🎯 8 signaux/session")
    print(f"⚡ Vérification: 2 min auto")
    print("="*60 + "\n")

    ensure_db()
    auto_verifier = AutoResultVerifier(engine, TWELVEDATA_API_KEY)

    http_runner = await start_http_server()

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Commandes
    app.add_handler(CommandHandler('start', cmd_start))
    app.add_handler(CommandHandler('menu', cmd_menu))
    app.add_handler(CommandHandler('startsession', cmd_start_session))
    app.add_handler(CommandHandler('sessionstatus', cmd_session_status))
    app.add_handler(CommandHandler('stats', cmd_stats))
    app.add_handler(CommandHandler('rapport', cmd_rapport))
    app.add_handler(CommandHandler('mlstats', cmd_mlstats))
    app.add_handler(CommandHandler('retrain', cmd_retrain))
    
    # Callbacks
    app.add_handler(CallbackQueryHandler(callback_generate_signal, pattern=r'^gen_signal_'))
    app.add_handler(CallbackQueryHandler(callback_new_session, pattern=r'^new_session$'))

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

if __name__ == '__main__':
    asyncio.run(main())
