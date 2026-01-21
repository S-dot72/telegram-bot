"""
Bot Pocket Option - Signaux M5 sur Commande
Fournit jusqu'à 8 signaux de haute qualité par session
"""

import os, json, asyncio
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import requests
import pandas as pd
from sqlalchemy import create_engine, text
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from aiohttp import web
from config import *
from utils import compute_indicators, rule_signal_ultra_strict
from ml_predictor import MLSignalPredictor
from auto_verifier import AutoResultVerifier

# Configuration
HAITI_TZ = ZoneInfo("America/Port-au-Prince")
TIMEFRAME_M1 = "1min"  # Données M1 pour analyse
EXPIRATION_MINUTES = 1  # Expiration 1 minute
CONFIDENCE_THRESHOLD = 0.75  # Seuil plus élevé pour qualité
MAX_SIGNALS_PER_SESSION = 8  # Maximum 8 signaux par session
SIGNAL_INTERVAL_MINUTES = 5  # 5 minutes entre chaque signal

engine = create_engine(DB_URL, connect_args={'check_same_thread': False})
ml_predictor = MLSignalPredictor()
auto_verifier = None

# État des sessions actives
active_sessions = {}  # user_id -> session_info

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
    
    # Samedi fermé
    if weekday == 5:
        return False
    # Dimanche fermé avant 22h UTC
    if weekday == 6 and hour < 22:
        return False
    # Vendredi fermé après 22h UTC
    if weekday == 4 and hour >= 22:
        return False
    
    return True

def fetch_ohlc_td(pair, interval, outputsize=300):
    """Récupère les données OHLC depuis TwelveData"""
    if not is_forex_open():
        raise RuntimeError("Marché Forex fermé")
    
    params = {
        'symbol': pair,
        'interval': interval,
        'outputsize': outputsize,
        'apikey': TWELVEDATA_API_KEY,
        'format': 'JSON'
    }
    
    r = requests.get(TWELVE_TS_URL, params=params, timeout=10)
    r.raise_for_status()
    j = r.json()
    
    if 'code' in j and j['code'] == 429:
        raise RuntimeError(f"Limite API atteinte: {j.get('message', 'Unknown')}")
    
    if 'values' not in j:
        raise RuntimeError(f"TwelveData error: {j}")
    
    df = pd.DataFrame(j['values'])[::-1].reset_index(drop=True)
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            df[col] = df[col].astype(float)
    if 'volume' in df.columns:
        df['volume'] = df['volume'].astype(float)
    df.index = pd.to_datetime(df['datetime'])
    return df

def get_cached_ohlc(pair, interval, outputsize=300):
    """Récupère les données avec cache"""
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
    """Sauvegarde signal en base"""
    q = text("""INSERT INTO signals (pair,direction,reason,ts_enter,ts_send,confidence,payload_json,max_gales)
    VALUES (:pair,:direction,:reason,:ts_enter,:ts_send,:confidence,:payload,:max_gales)""")
    with engine.begin() as conn:
        result = conn.execute(q, payload)
    return result.lastrowid

def ensure_db():
    """Initialise la base de données"""
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
    except Exception as e:
        print(f"⚠️ Erreur DB: {e}")

def analyze_single_pair(pair, priority=5):
    """Analyse une paire et retourne un signal si trouvé"""
    try:
        # Récupérer données M1 pour analyse fine
        df = get_cached_ohlc(pair, TIMEFRAME_M1, outputsize=500)
        
        if df is None or len(df) < 100:
            return None
        
        # Calculer indicateurs
        df = compute_indicators(df)
        
        # Stratégie avec priorité maximale
        base_signal = rule_signal_ultra_strict(df, session_priority=priority)
        
        if not base_signal:
            return None
        
        # ML avec seuil strict
        ml_signal, ml_conf = ml_predictor.predict_signal(df, base_signal)
        
        if ml_signal is None or ml_conf < CONFIDENCE_THRESHOLD:
            return None
        
        # Score de qualité
        last = df.iloc[-1]
        quality_score = 0
        
        # ADX (0-30 points)
        adx = last.get('adx', 0)
        if adx > 30:
            quality_score += 30
        elif adx > 25:
            quality_score += 25
        elif adx > 20:
            quality_score += 20
        elif adx > 15:
            quality_score += 15
        
        # RSI (0-25 points)
        rsi = last.get('rsi', 50)
        if 45 < rsi < 55:
            quality_score += 25
        elif 40 < rsi < 60:
            quality_score += 20
        elif 35 < rsi < 65:
            quality_score += 15
        
        # MACD alignement (0-20 points)
        macd = last.get('MACD_12_26_9', 0)
        macd_signal = last.get('MACDs_12_26_9', 0)
        if (ml_signal == 'CALL' and macd > macd_signal) or (ml_signal == 'PUT' and macd < macd_signal):
            quality_score += 20
        
        # Confiance ML (0-25 points)
        quality_score += int(ml_conf * 25)
        
        return {
            'pair': pair,
            'signal': ml_signal,
            'confidence': ml_conf,
            'quality_score': quality_score,
            'adx': adx,
            'rsi': rsi,
            'df': df
        }
        
    except Exception as e:
        print(f"⚠️ Erreur analyse {pair}: {e}")
        return None

# ===== COMMANDES TELEGRAM =====

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Commande /start"""
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    
    try:
        with engine.begin() as conn:
            existing = conn.execute(
                text("SELECT user_id FROM subscribers WHERE user_id = :uid"),
                {"uid": user_id}
            ).fetchone()
            
            if existing:
                await update.message.reply_text("✅ Vous êtes déjà inscrit !")
            else:
                conn.execute(
                    text("INSERT INTO subscribers (user_id, username) VALUES (:uid, :uname)"),
                    {"uid": user_id, "uname": username}
                )
                
                await update.message.reply_text(
                    "✅ **Bienvenue sur Pocket Option Bot !**\n\n"
                    "🎯 **Caractéristiques:**\n"
                    "• Signaux M1 haute qualité\n"
                    "• Expiration: 1 minute\n"
                    "• Maximum 8 signaux/session\n"
                    "• Intervalle: 5 minutes\n"
                    "• Confiance minimum 75%\n"
                    "• Vérification automatique\n\n"
                    "📋 **Commandes:**\n"
                    "• /signaux - Lancer session (max 8)\n"
                    "• /stats - Voir statistiques\n"
                    "• /verify - Vérifier résultats\n"
                    "• /help - Aide détaillée\n\n"
                    "🔥 **Utilisez /signaux pour commencer !**"
                )
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Commande /help"""
    help_text = (
        "📖 **GUIDE D'UTILISATION**\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        "**🎯 /signaux** - Session de signaux\n"
        "Lance une session qui analyse toutes les paires\n"
        "et envoie jusqu'à 8 signaux de haute qualité.\n"
        "Intervalle: 5 minutes entre signaux.\n\n"
        "**📊 /stats** - Statistiques\n"
        "Affiche vos performances:\n"
        "• Total signaux\n"
        "• Win rate\n"
        "• Gains/Pertes\n\n"
        "**🔍 /verify** - Vérification\n"
        "Force la vérification des signaux\n"
        "en attente et envoie les résultats.\n\n"
        "**⚙️ CRITÈRES DE QUALITÉ:**\n"
        "• Confiance ML ≥ 75%\n"
        "• ADX ≥ 15 (tendance)\n"
        "• RSI entre 30-70\n"
        "• MACD aligné\n"
        "• Score qualité ≥ 60/100\n\n"
        "**⏰ TIMEFRAME:**\n"
        "• Analyse: M1 (données 1 min)\n"
        "• Expiration: 1 minute\n"
        "• Intervalle signaux: 5 minutes\n\n"
        "**💡 AVANTAGES M1:**\n"
        "• Analyse plus précise\n"
        "• Détection rapide des tendances\n"
        "• Résultats immédiats\n"
        "• 5 min entre signaux = meilleure qualité\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "💡 Utilisez /signaux quand le marché est ouvert"
    )
    await update.message.reply_text(help_text)

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Commande /stats"""
    try:
        with engine.connect() as conn:
            total = conn.execute(text('SELECT COUNT(*) FROM signals')).scalar()
            wins = conn.execute(text("SELECT COUNT(*) FROM signals WHERE result='WIN'")).scalar()
            losses = conn.execute(text("SELECT COUNT(*) FROM signals WHERE result='LOSE'")).scalar()
            pending = conn.execute(text("SELECT COUNT(*) FROM signals WHERE result IS NULL")).scalar()

        verified = wins + losses
        winrate = (wins/verified*100) if verified > 0 else 0

        msg = (
            f"📊 **VOS STATISTIQUES**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📈 Total signaux: {total}\n"
            f"✅ Gagnés: {wins}\n"
            f"❌ Perdus: {losses}\n"
            f"⏳ En attente: {pending}\n\n"
            f"🎯 **Win Rate: {winrate:.1f}%**\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 Expiration: M1 | Pocket Option"
        )
        
        await update.message.reply_text(msg)

    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_signaux(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Commande /signaux - Lance une session de 8 signaux max"""
    user_id = update.effective_user.id
    
    # Vérifier si session active
    if user_id in active_sessions:
        await update.message.reply_text(
            "⚠️ Vous avez déjà une session active !\n"
            "Attendez qu'elle se termine."
        )
        return
    
    # Vérifier marché ouvert
    if not is_forex_open():
        await update.message.reply_text(
            "🏖️ **Marché Forex fermé**\n\n"
            "Le marché est ouvert:\n"
            "• Dimanche 22h - Vendredi 22h (UTC)\n\n"
            "Réessayez pendant les heures d'ouverture."
        )
        return
    
    # Créer session
    active_sessions[user_id] = {
        'start_time': get_utc_now(),
        'signals_sent': 0,
        'max_signals': MAX_SIGNALS_PER_SESSION,
        'chat_id': update.effective_chat.id
    }
    
    await update.message.reply_text(
        "🚀 **SESSION LANCÉE**\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📊 Maximum: {MAX_SIGNALS_PER_SESSION} signaux\n"
        f"⏱️ Intervalle: {SIGNAL_INTERVAL_MINUTES} minutes\n"
        f"⏰ Expiration: {EXPIRATION_MINUTES} minute\n"
        f"🎯 Confiance min: {CONFIDENCE_THRESHOLD*100:.0f}%\n\n"
        "⏳ Analyse en cours...\n\n"
        "💡 Analyse M1 pour plus de précision"
    )
    
    # Lancer génération signaux en arrière-plan
    asyncio.create_task(generate_signal_session(user_id, context.application))

async def generate_signal_session(user_id, app):
    """Génère jusqu'à 8 signaux de haute qualité avec intervalle de 5 minutes"""
    session = active_sessions.get(user_id)
    if not session:
        return
    
    chat_id = session['chat_id']
    signals_found = []
    
    try:
        # Liste des paires à analyser
        pairs_to_analyze = PAIRS[:6]  # Top 6 paires
        
        # Boucle jusqu'à avoir 8 signaux
        while session['signals_sent'] < MAX_SIGNALS_PER_SESSION:
            
            # Vérifier si marché toujours ouvert
            if not is_forex_open():
                await app.bot.send_message(
                    chat_id=chat_id,
                    text="⚠️ Session interrompue: marché fermé"
                )
                break
            
            # Analyser toutes les paires et trouver le meilleur signal
            best_signal = None
            best_score = 0
            
            await app.bot.send_message(
                chat_id=chat_id,
                text=f"🔍 Analyse signal {session['signals_sent'] + 1}/{MAX_SIGNALS_PER_SESSION}..."
            )
            
            for pair in pairs_to_analyze:
                result = analyze_single_pair(pair, priority=5)
                
                if result and result['quality_score'] > best_score:
                    # Vérifier qu'on n'a pas déjà envoyé cette paire récemment
                    if not any(s['pair'] == pair for s in signals_found[-2:]):
                        best_signal = result
                        best_score = result['quality_score']
            
            # Si signal trouvé avec score suffisant
            if best_signal and best_score >= 60:
                # Envoyer le signal
                await send_signal_to_user(chat_id, best_signal, session['signals_sent'] + 1, app)
                
                # Sauvegarder
                signals_found.append(best_signal)
                session['signals_sent'] += 1
                
                # Attendre 5 minutes avant prochain signal (sauf si dernier)
                if session['signals_sent'] < MAX_SIGNALS_PER_SESSION:
                    remaining = MAX_SIGNALS_PER_SESSION - session['signals_sent']
                    await app.bot.send_message(
                        chat_id=chat_id,
                        text=f"⏰ Prochain signal dans {SIGNAL_INTERVAL_MINUTES} minutes\n"
                             f"📊 Restant: {remaining} signal{'s' if remaining > 1 else ''}"
                    )
                    await asyncio.sleep(SIGNAL_INTERVAL_MINUTES * 60)
            else:
                # Pas de signal de qualité, attendre 2 minutes et réessayer
                await app.bot.send_message(
                    chat_id=chat_id,
                    text="⏳ Conditions non optimales, nouvelle analyse dans 2 min..."
                )
                await asyncio.sleep(120)
        
        # Fin de session
        duration_minutes = (get_utc_now() - session['start_time']).seconds // 60
        avg_quality = sum(s['quality_score'] for s in signals_found) / len(signals_found) if signals_found else 0
        
        await app.bot.send_message(
            chat_id=chat_id,
            text=(
                f"✅ **SESSION TERMINÉE**\n"
                f"━━━━━━━━━━━━━━━━━━━━\n\n"
                f"📊 Signaux envoyés: {session['signals_sent']}/{MAX_SIGNALS_PER_SESSION}\n"
                f"⏱️ Durée totale: {duration_minutes} min\n"
                f"📈 Qualité moyenne: {avg_quality:.0f}/100\n\n"
                f"🔍 Vérification auto dans 2-3 minutes\n\n"
                f"💡 Utilisez /verify pour forcer la vérification"
            )
        )
        
    except Exception as e:
        await app.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Erreur session: {e}"
        )
    
    finally:
        # Nettoyer session
        if user_id in active_sessions:
            del active_sessions[user_id]

async def send_signal_to_user(chat_id, signal_data, signal_num, app):
    """Envoie un signal formaté à l'utilisateur"""
    try:
        pair = signal_data['pair']
        signal = signal_data['signal']
        confidence = signal_data['confidence']
        quality_score = signal_data['quality_score']
        adx = signal_data['adx']
        rsi = signal_data['rsi']
        
        # Calculer temps d'entrée (prochaine minute pleine)
        now_haiti = get_haiti_now()
        entry_time = now_haiti + timedelta(minutes=1)
        entry_time = entry_time.replace(second=0, microsecond=0)
        
        # Direction
        direction_text = "📈 BUY (CALL)" if signal == "CALL" else "📉 SELL (PUT)"
        direction_emoji = "🟢" if signal == "CALL" else "🔴"
        
        # Qualité
        if quality_score >= 80:
            quality_text = "🔥 EXCELLENT"
        elif quality_score >= 70:
            quality_text = "✨ TRÈS BON"
        else:
            quality_text = "✅ BON"
        
        msg = (
            f"{direction_emoji} **SIGNAL #{signal_num}/8**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"💱 Paire: **{pair}**\n"
            f"📊 Direction: {direction_text}\n"
            f"⏰ Entrée: **{entry_time.strftime('%H:%M')}** (Haïti)\n"
            f"⏱️ Expiration: **M1** (1 minute)\n\n"
            f"🎯 Confiance: **{int(confidence*100)}%**\n"
            f"📈 Qualité: {quality_text} ({quality_score}/100)\n\n"
            f"📊 **Indicateurs M1:**\n"
            f"• ADX: {adx:.1f} (tendance)\n"
            f"• RSI: {rsi:.1f}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🔍 Résultat auto dans 2-3 min"
        )
        
        await app.bot.send_message(chat_id=chat_id, text=msg)
        
        # Sauvegarder en base
        entry_time_utc = entry_time.astimezone(timezone.utc)
        
        payload = {
            'pair': pair,
            'direction': signal,
            'reason': f'M1 ML {confidence:.1%} - Q{quality_score}',
            'ts_enter': entry_time_utc.isoformat(),
            'ts_send': get_utc_now().isoformat(),
            'confidence': confidence,
            'payload': json.dumps({'quality_score': quality_score, 'timeframe': 'M1'}),
            'max_gales': 0
        }
        
        signal_id = persist_signal(payload)
        print(f"✅ Signal #{signal_num} envoyé: {pair} {signal} M1 (ID: {signal_id})")
        
    except Exception as e:
        print(f"❌ Erreur envoi signal: {e}")

async def cmd_verify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Commande /verify - Vérifie les signaux en attente"""
    try:
        msg = await update.message.reply_text("🔍 Vérification en cours...")
        
        # Vérifier
        await auto_verifier.verify_pending_signals()
        
        # Compter résultats
        with engine.connect() as conn:
            verified = conn.execute(
                text("SELECT COUNT(*) FROM signals WHERE result IS NOT NULL")
            ).scalar()
            wins = conn.execute(
                text("SELECT COUNT(*) FROM signals WHERE result='WIN'")
            ).scalar()
            losses = conn.execute(
                text("SELECT COUNT(*) FROM signals WHERE result='LOSE'")
            ).scalar()
        
        winrate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
        
        await msg.edit_text(
            f"✅ **Vérification terminée**\n\n"
            f"📊 Vérifiés: {verified}\n"
            f"✅ Gagnés: {wins}\n"
            f"❌ Perdus: {losses}\n"
            f"📈 Win Rate: {winrate:.1f}%\n\n"
            f"Utilisez /stats pour plus de détails"
        )
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

# ===== SERVEUR HTTP =====

async def health_check(request):
    """Health check pour Render"""
    return web.json_response({
        'status': 'ok',
        'timestamp': get_haiti_now().isoformat(),
        'forex_open': is_forex_open()
    })

async def start_http_server():
    """Démarre serveur HTTP"""
    app = web.Application()
    app.router.add_get('/health', health_check)
    app.router.add_get('/', health_check)
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    port = int(os.getenv('PORT', 10000))
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    
    print(f"✅ HTTP server: http://0.0.0.0:{port}/health")
    return runner

async def main():
    global auto_verifier

    print("\n" + "="*60)
    print("🎯 POCKET OPTION BOT - SIGNAUX SUR COMMANDE")
    print("="*60)
    print(f"🇭🇹 {get_haiti_now().strftime('%H:%M:%S %Z')}")
    print(f"📈 Forex: {'🟢 OUVERT' if is_forex_open() else '🔴 FERMÉ'}")
    print(f"⏰ Timeframe: M1 (1 minute)")
    print(f"⏱️ Expiration: {EXPIRATION_MINUTES} minute")
    print(f"🎯 Max signaux/session: {MAX_SIGNALS_PER_SESSION}")
    print(f"⏰ Intervalle: {SIGNAL_INTERVAL_MINUTES} minutes")
    print(f"💪 Confiance minimum: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print("="*60 + "\n")

    ensure_db()
    auto_verifier = AutoResultVerifier(engine, TWELVEDATA_API_KEY)

    # Serveur HTTP
    http_runner = await start_http_server()

    # Bot Telegram
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    app.add_handler(CommandHandler('start', cmd_start))
    app.add_handler(CommandHandler('help', cmd_help))
    app.add_handler(CommandHandler('stats', cmd_stats))
    app.add_handler(CommandHandler('signaux', cmd_signaux))
    app.add_handler(CommandHandler('verify', cmd_verify))

    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)

    bot_info = await app.bot.get_me()
    print(f"✅ BOT ACTIF: @{bot_info.username}")
    print(f"💡 Utilisez /signaux pour lancer une session\n")

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
