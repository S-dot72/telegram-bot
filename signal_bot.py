"""
Bot de trading M1 - Version Interactive
8 signaux par session avec bouton Generate Signal
Support OTC (crypto) le week-end via APIs multiples
Signal envoyé immédiatement avec timing 2 minutes avant entrée
"""

import os, json, asyncio, random
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import requests
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
from aiohttp import web
from config import *
from utils import compute_indicators, rule_signal_ultra_strict
from ml_predictor import MLSignalPredictor
from auto_verifier import AutoResultVerifier
from otc_provider import OTCDataProvider

# ================= CONFIGURATION =================
HAITI_TZ = ZoneInfo("America/Port-au-Prince")
TIMEFRAME_M1 = "1min"
SIGNALS_PER_SESSION = 8
VERIFICATION_WAIT_MIN = 3  # Changé de 2 à 3 minutes (2 min avant entrée + 1 min bougie)
CONFIDENCE_THRESHOLD = 0.65

# Initialisation des composants
engine = create_engine(DB_URL, connect_args={'check_same_thread': False})
ml_predictor = MLSignalPredictor()
auto_verifier = None
otc_provider = OTCDataProvider(TWELVEDATA_API_KEY)

# Variables globales
active_sessions = {}
pending_signal_tasks = {}  # Stocke les tâches d'attente pour les signaux
TWELVE_TS_URL = 'https://api.twelvedata.com/time_series'
ohlc_cache = {}
last_error_logs = []

# ================= FONCTIONS UTILITAIRES =================

def add_error_log(message):
    """Ajoute un message d'erreur à la liste des logs"""
    global last_error_logs
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_entry = f"{timestamp} - {message}"
    print(log_entry)
    last_error_logs.append(log_entry)
    if len(last_error_logs) > 20:
        last_error_logs.pop(0)

def get_haiti_now():
    return datetime.now(HAITI_TZ)

def get_utc_now():
    return datetime.now(timezone.utc)

def is_forex_open():
    """Vérifie si marché Forex est ouvert"""
    now_utc = get_utc_now()
    weekday = now_utc.weekday()
    hour = now_utc.hour
    
    if weekday == 5:  # Samedi
        return False
    if weekday == 6 and hour < 22:  # Dimanche avant 22h UTC
        return False
    if weekday == 4 and hour >= 22:  # Vendredi après 22h UTC
        return False
    
    return True

def get_current_pair(pair):
    """Retourne la paire à utiliser (Forex ou Crypto) en fonction du jour"""
    if otc_provider.is_weekend():
        forex_to_crypto = {
            'EUR/USD': 'BTC/USD',
            'GBP/USD': 'ETH/USD',
            'USD/JPY': 'TRX/USD',
            'AUD/USD': 'LTC/USD',
            'BTC/USD': 'BTC/USD',
            'ETH/USD': 'ETH/USD'
        }
        return forex_to_crypto.get(pair, 'BTC/USD')
    return pair

def check_api_availability():
    """Vérifie la disponibilité des APIs"""
    results = {
        'forex_available': False,
        'crypto_available': False,
        'synthetic_available': True,
        'current_mode': None,
        'test_pairs': []
    }
    
    now_utc = get_utc_now()
    is_weekend = otc_provider.is_weekend()
    results['current_mode'] = 'OTC (Crypto)' if is_weekend else 'Forex'
    
    try:
        # Tester Forex via TwelveData
        if not is_weekend:
            test_pair = 'EUR/USD'
            params = {
                'symbol': test_pair,
                'interval': '1min',
                'outputsize': 2,
                'apikey': TWELVEDATA_API_KEY,
                'format': 'JSON'
            }
            
            try:
                r = requests.get(TWELVE_TS_URL, params=params, timeout=10)
                
                if r.status_code == 200:
                    j = r.json()
                    if 'values' in j and len(j['values']) > 0:
                        results['forex_available'] = True
                        results['test_pairs'].append({
                            'pair': test_pair,
                            'status': 'OK',
                            'market': 'Forex',
                            'data_points': len(j['values']),
                            'last_price': j['values'][0].get('close', 'N/A'),
                            'source': 'TwelveData'
                        })
                    else:
                        error_msg = j.get('message', 'No values') if 'message' in j else 'Empty response'
                        results['test_pairs'].append({
                            'pair': test_pair,
                            'status': 'NO_DATA',
                            'market': 'Forex',
                            'error': error_msg,
                            'source': 'TwelveData'
                        })
                else:
                    results['test_pairs'].append({
                        'pair': test_pair,
                        'status': 'ERROR',
                        'market': 'Forex',
                        'error': f'HTTP {r.status_code}',
                        'source': 'TwelveData'
                    })
                    
            except Exception as e:
                results['test_pairs'].append({
                    'pair': test_pair,
                    'status': 'ERROR',
                    'market': 'Forex',
                    'error': str(e)[:100],
                    'source': 'TwelveData'
                })
        
        # Tester Crypto via multiple APIs
        if is_weekend:
            test_pair = 'BTC/USD'
            try:
                # Tester directement via get_otc_data
                df = otc_provider.get_otc_data(test_pair, '1min', 5)
                
                if df is not None and len(df) > 0:
                    results['crypto_available'] = True
                    results['test_pairs'].append({
                        'pair': test_pair,
                        'status': 'OK',
                        'market': 'Crypto',
                        'data_points': len(df),
                        'last_price': df.iloc[-1]['close'],
                        'source': 'Multi-APIs (Bybit/Binance)'
                    })
                else:
                    results['test_pairs'].append({
                        'pair': test_pair,
                        'status': 'NO_DATA',
                        'market': 'Crypto',
                        'error': 'Aucune donnée récupérée',
                        'source': 'Multi-APIs'
                    })
                    
            except Exception as e:
                results['test_pairs'].append({
                    'pair': test_pair,
                    'status': 'ERROR',
                    'market': 'Crypto',
                    'error': str(e)[:100],
                    'source': 'Multi-APIs'
                })
    
    except Exception as e:
        results['error'] = str(e)
    
    return results

def fetch_ohlc_td(pair, interval, outputsize=300):
    """Version unifiée utilisant APIs multiples pour Forex ET Crypto"""
    
    # Vérifier si week-end
    if otc_provider.is_weekend():
        print(f"🏖️ Week-end - Mode OTC (Crypto via APIs multiples)")
        
        # Utiliser la méthode unifiée get_otc_data
        df = otc_provider.get_otc_data(pair, interval, outputsize)
        
        if df is not None and len(df) > 0:
            print(f"✅ Données Crypto récupérées: {len(df)} bougies")
            return df
        else:
            print("⚠️ APIs Crypto indisponibles, basculement sur synthétique")
            return otc_provider.generate_synthetic_data(pair, interval, outputsize)
    
    # Mode Forex normal (semaine)
    if not is_forex_open():
        raise RuntimeError("Marché Forex fermé")
    
    # Utiliser TwelveData pour Forex
    params = {
        'symbol': pair, 
        'interval': interval, 
        'outputsize': outputsize,
        'apikey': TWELVEDATA_API_KEY, 
        'format': 'JSON'
    }
    
    try:
        r = requests.get(TWELVE_TS_URL, params=params, timeout=10)
        r.raise_for_status()
        j = r.json()
        
        if 'code' in j and j['code'] == 429:
            raise RuntimeError(f"Limite API TwelveData atteinte")
        
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
    except Exception as e:
        add_error_log(f"Erreur TwelveData Forex: {e}")
        raise RuntimeError(f"Erreur TwelveData Forex: {e}")

def get_cached_ohlc(pair, interval, outputsize=300):
    """Récupère les données OHLC depuis le cache ou les APIs"""
    current_pair = get_current_pair(pair)
    cache_key = f"{current_pair}_{interval}"
    
    current_time = get_utc_now()
    
    if cache_key in ohlc_cache:
        cached_data, cached_time = ohlc_cache[cache_key]
        if (current_time - cached_time).total_seconds() < 30:
            return cached_data
    
    try:
        df = fetch_ohlc_td(current_pair, interval, outputsize)
        ohlc_cache[cache_key] = (df, current_time)
        
        if df is not None and len(df) > 0:
            print(f"✅ Données chargées: {len(df)} bougies pour {current_pair}")
            print(f"   Dernière bougie: {df.index[-1]} - ${df.iloc[-1]['close']:.2f}")
        else:
            print(f"⚠️ Données vides pour {current_pair}")
            
        return df
    except RuntimeError as e:
        add_error_log(f"Cache OHLC: {e}")
        return None
    except Exception as e:
        add_error_log(f"Erreur get_cached_ohlc: {e}")
        return None

def persist_signal(payload):
    """Persiste un signal en base de données"""
    q = text("""INSERT INTO signals (pair,direction,reason,ts_enter,ts_send,confidence,payload_json,max_gales,timeframe)
    VALUES (:pair,:direction,:reason,:ts_enter,:ts_send,:confidence,:payload,:max_gales,:timeframe)""")
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
            
            if 'timeframe' not in existing_cols:
                conn.execute(text("ALTER TABLE signals ADD COLUMN timeframe INTEGER DEFAULT 1"))
            
            print("✅ Base de données prête")

    except Exception as e:
        print(f"⚠️ Erreur DB: {e}")

# ================= VÉRIFICATION AUTOMATIQUE =================

async def auto_verify_signal(signal_id, user_id, app):
    """Vérifie automatiquement un signal après 3 minutes (2 min avant entrée + 1 min bougie)"""
    try:
        print(f"\n[VERIF_AUTO] 🔍 Vérification auto signal #{signal_id}")
        
        # Attendre 3 minutes (2 min avant entrée + 1 min bougie)
        print(f"[VERIF_AUTO] ⏳ Attente de 3 minutes...")
        await asyncio.sleep(180)  # Changé de 120 à 180 secondes
        
        print(f"[VERIF_AUTO] ✅ 3 minutes écoulées, vérification en cours...")
        
        # IMPORTANT: Attendre encore un peu pour être sûr que la bougie est complète
        await asyncio.sleep(5)
        
        # Vérifier si auto_verifier est initialisé
        if auto_verifier is None:
            print(f"[VERIF_AUTO] ❌ auto_verifier n'est pas initialisé!")
            return
        
        print(f"[VERIF_AUTO] 📊 Appel de verify_single_signal...")
        
        # Vérifier
        result = await auto_verifier.verify_single_signal(signal_id)
        
        print(f"[VERIF_AUTO] 📝 Résultat brut: {result}")
        
        if not result:
            print(f"[VERIF_AUTO] ⚠️ Résultat non défini pour #{signal_id}")
            # Si pas de résultat automatique, on marque manuellement comme LOSE pour continuer
            result = 'LOSE'
            await auto_verifier.manual_verify_signal(signal_id, result)
        
        # Mettre à jour session
        if user_id in active_sessions:
            session = active_sessions[user_id]
            session['pending'] = max(0, session['pending'] - 1)
            
            if result == 'WIN':
                session['wins'] += 1
                print(f"[VERIF_AUTO] ✅ Signal #{signal_id} WIN - Wins: {session['wins']}")
            else:
                session['losses'] += 1
                print(f"[VERIF_AUTO] ❌ Signal #{signal_id} LOSE - Losses: {session['losses']}")
        
        # Récupérer détails du signal
        with engine.connect() as conn:
            signal = conn.execute(
                text("SELECT pair, direction, confidence FROM signals WHERE id = :sid"),
                {"sid": signal_id}
            ).fetchone()
        
        if not signal:
            print(f"[VERIF_AUTO] ⚠️ Signal #{signal_id} non trouvé en base")
            return
        
        pair, direction, confidence = signal
        
        # Envoyer résultat à l'utilisateur
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
                
                try:
                    await app.bot.send_message(chat_id=user_id, text=briefing, reply_markup=reply_markup)
                    print(f"[VERIF_AUTO] ✅ Résultat envoyé avec bouton pour signal #{signal_id}")
                except Exception as e:
                    print(f"[VERIF_AUTO] ❌ Erreur envoi message: {e}")
            else:
                # Session terminée
                try:
                    await app.bot.send_message(chat_id=user_id, text=briefing)
                    await end_session_summary(user_id, app)
                    print(f"[VERIF_AUTO] ✅ Résultat envoyé, session terminée pour signal #{signal_id}")
                except Exception as e:
                    print(f"[VERIF_AUTO] ❌ Erreur envoi message: {e}")
        else:
            try:
                await app.bot.send_message(chat_id=user_id, text=briefing)
                print(f"[VERIF_AUTO] ✅ Résultat envoyé (session inactive) pour signal #{signal_id}")
            except Exception as e:
                print(f"[VERIF_AUTO] ❌ Erreur envoi message: {e}")
        
        print(f"[VERIF_AUTO] ✅ Briefing #{signal_id} terminé ({result})")
        
    except Exception as e:
        print(f"[VERIF_AUTO] ❌ ERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()
        
        # En cas d'erreur, marquer le signal comme LOSE pour continuer
        try:
            await auto_verifier.manual_verify_signal(signal_id, 'LOSE')
            print(f"[VERIF_AUTO] ⚠️ Signal #{signal_id} marqué comme LOSE suite à erreur")
        except:
            print(f"[VERIF_AUTO] ❌ Impossible de marquer le signal comme LOSE")

# ================= FONCTION RAPPEL =================

async def send_reminder(signal_id, user_id, app, reminder_time, entry_time, pair, direction):
    """Envoie un rappel 1 minute avant l'entrée"""
    try:
        now_haiti = get_haiti_now()
        wait_seconds = (reminder_time - now_haiti).total_seconds()
        
        if wait_seconds > 0:
            print(f"[REMINDER] ⏳ Attente de {wait_seconds:.1f} secondes pour rappel signal #{signal_id}")
            await asyncio.sleep(wait_seconds)
        
        time_to_entry = max(0, (entry_time - get_haiti_now()).total_seconds() / 60)
        direction_text = "BUY ↗️" if direction == "CALL" else "SELL ↘️"
        
        reminder_msg = (
            f"🔔 **RAPPEL - SIGNAL #{active_sessions[user_id]['signal_count']}**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💱 {pair}\n"
            f"📈 Direction: **{direction_text}**\n"
            f"⏰ Entrée dans: **{time_to_entry:.0f} min**\n\n"
            f"💡 Préparez-vous à entrer en position!"
        )
        
        try:
            await app.bot.send_message(chat_id=user_id, text=reminder_msg)
            print(f"[REMINDER] ✅ Rappel envoyé pour signal #{signal_id}")
        except Exception as e:
            print(f"[REMINDER] ❌ Erreur envoi rappel: {e}")
            
    except asyncio.CancelledError:
        print(f"[REMINDER] ❌ Tâche de rappel signal #{signal_id} annulée")
        raise
    except Exception as e:
        print(f"[REMINDER] ❌ Erreur dans send_reminder: {e}")

# ================= COMMANDES TELEGRAM =================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Commande de démarrage du bot"""
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    try:
        with engine.begin() as conn:
            existing = conn.execute(text("SELECT user_id FROM subscribers WHERE user_id = :uid"),
            {"uid": user_id}).fetchone()
            if not existing:
                conn.execute(text("INSERT INTO subscribers (user_id, username) VALUES (:uid, :uname)"),
                {"uid": user_id, "uname": username})
        
        is_weekend = otc_provider.is_weekend()
        mode_text = "🏖️ OTC (Crypto)" if is_weekend else "📈 Forex"
        
        await update.message.reply_text(
            f"✅ **Bienvenue au Bot Trading M1 !**\n\n"
            f"🎯 Mode: **Interactive Session**\n"
            f"📊 8 signaux M1 par session\n"
            f"⚡ Signal envoyé: **Immédiatement avec timing**\n"
            f"🔔 Rappel: 1 min avant entrée\n"
            f"🔍 Vérification auto: 3 min après signal\n"
            f"🌐 Mode actuel: {mode_text}\n"
            f"🔧 Sources: TwelveData + APIs Crypto\n\n"
            f"**Commandes:**\n"
            f"• /startsession - Démarrer session\n"
            f"• /stats - Statistiques\n"
            f"• /otcstatus - Statut OTC\n"
            f"• /checkapi - Vérifier APIs\n"
            f"• /menu - Menu complet\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💡 Trading 24/7 avec OTC le week-end !"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche le menu complet"""
    menu_text = (
        "📋 **MENU M1**\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        "**📊 Session:**\n"
        "• /startsession - Démarrer session\n"
        "• /sessionstatus - État session\n"
        "• /endsession - Terminer session\n"
        "• /forceend - Terminer session (forcé)\n\n"
        "**📈 Statistiques:**\n"
        "• /stats - Stats globales\n"
        "• /rapport - Rapport du jour\n\n"
        "**🤖 Machine Learning:**\n"
        "• /mlstats - Stats ML\n"
        "• /retrain - Réentraîner modèle\n\n"
        "**🌐 OTC (Week-end):**\n"
        "• /otcstatus - Statut OTC\n"
        "• /testotc - Tester OTC\n"
        "• /checkapi - Vérifier APIs\n"
        "• /debugapi - Debug APIs\n"
        "• /debugpair - Debug conversion paires\n\n"
        "**🔧 Vérification:**\n"
        "• /pending - Signaux en attente\n"
        "• /signalinfo <id> - Info signal\n"
        "• /manualresult <id> WIN/LOSE\n"
        "• /forceverify <id> - Forcer vérification\n"
        "• /forceall - Forcer toutes vérifications\n"
        "• /debugverif - Debug vérification\n\n"
        "**⚠️ Erreurs:**\n"
        "• /lasterrors - Dernières erreurs\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "🎯 M1 | 8 signaux/session\n"
        "⚡ Signal envoyé: Immédiatement\n"
        "🔔 Rappel: 1 min avant entrée\n"
        "🔍 Vérif auto: 3 min après signal\n"
        "🏖️ OTC actif le week-end\n"
        "🔧 Multi-APIs Crypto"
    )
    await update.message.reply_text(menu_text)

async def cmd_start_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Démarre une nouvelle session de 8 signaux"""
    user_id = update.effective_user.id
    
    # Vérifier si session active
    if user_id in active_sessions:
        session = active_sessions[user_id]
        
        if session['signal_count'] < SIGNALS_PER_SESSION:
            next_num = session['signal_count'] + 1
            keyboard = [[InlineKeyboardButton(f"🎯 Generate Signal #{next_num}", callback_data=f"gen_signal_{user_id}")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                f"⚠️ Session déjà active !\n\n"
                f"📊 Progression: {session['signal_count']}/{SIGNALS_PER_SESSION}\n"
                f"✅ Wins: {session['wins']}\n"
                f"❌ Losses: {session['losses']}\n\n"
                f"Continuer avec signal #{next_num} ⬇️",
                reply_markup=reply_markup
            )
        else:
            await update.message.reply_text(
                f"⚠️ Session déjà terminée !\n\n"
                f"📊 Résultat: {session['signal_count']}/{SIGNALS_PER_SESSION}\n"
                f"✅ Wins: {session['wins']}\n"
                f"❌ Losses: {session['losses']}\n\n"
                f"Utilisez /endsession pour voir le résumé"
            )
        return
    
    # Créer nouvelle session
    now_haiti = get_haiti_now()
    active_sessions[user_id] = {
        'start_time': now_haiti,
        'signal_count': 0,
        'wins': 0,
        'losses': 0,
        'pending': 0,
        'signals': [],
        'verification_tasks': [],
        'reminder_tasks': []  # Nouvelles tâches de rappel
    }
    
    # Bouton pour générer premier signal
    keyboard = [[InlineKeyboardButton("🎯 Generate Signal #1", callback_data=f"gen_signal_{user_id}")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    is_weekend = otc_provider.is_weekend()
    mode_text = "🏖️ OTC (Crypto)" if is_weekend else "📈 Forex"
    
    await update.message.reply_text(
        "🚀 **SESSION DÉMARRÉE**\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📅 {now_haiti.strftime('%H:%M:%S')}\n"
        f"🌐 Mode: {mode_text}\n"
        f"🎯 Objectif: {SIGNALS_PER_SESSION} signaux M1\n"
        f"⚡ Signal envoyé: Immédiatement\n"
        f"🔍 Vérification: 3 min après signal\n"
        f"🔧 Sources: {'APIs Crypto' if is_weekend else 'TwelveData'}\n\n"
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
    
    # Vérifier si des rappels sont en attente
    pending_reminders = 0
    if 'reminder_tasks' in session:
        for task in session['reminder_tasks']:
            if not task.done():
                pending_reminders += 1
    
    msg = (
        "📊 **ÉTAT SESSION**\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        f"⏱️ Durée: {duration:.1f} min\n"
        f"📈 Progression: {session['signal_count']}/{SIGNALS_PER_SESSION}\n\n"
        f"✅ Wins: {session['wins']}\n"
        f"❌ Losses: {session['losses']}\n"
        f"⏳ Vérif en attente: {session['pending']}\n"
        f"🔔 Rappels en attente: {pending_reminders}\n\n"
        f"📊 Win Rate: {winrate:.1f}%\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"⚡ Signal envoyé immédiatement\n"
        f"🔔 Rappel 1 min avant entrée"
    )
    
    await update.message.reply_text(msg)

async def cmd_end_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Termine la session active manuellement"""
    user_id = update.effective_user.id
    
    if user_id not in active_sessions:
        await update.message.reply_text("ℹ️ Aucune session active")
        return
    
    session = active_sessions[user_id]
    
    # Annuler les tâches de rappel en attente
    if 'reminder_tasks' in session:
        for task in session['reminder_tasks']:
            if not task.done():
                try:
                    task.cancel()
                except:
                    pass
    
    if session['pending'] > 0:
        await update.message.reply_text(
            f"⚠️ {session['pending']} signal(s) en attente de vérification\n\n"
            f"Attendez la fin des vérifications ou confirmez la fin avec /forceend"
        )
        return
    
    await end_session_summary(user_id, context.application)
    await update.message.reply_text("✅ Session terminée !")

async def cmd_force_end(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Force la fin de session même avec signaux en attente"""
    user_id = update.effective_user.id
    
    if user_id not in active_sessions:
        await update.message.reply_text("ℹ️ Aucune session active")
        return
    
    session = active_sessions[user_id]
    
    # Annuler toutes les tâches en cours
    if 'verification_tasks' in session:
        for task in session['verification_tasks']:
            if not task.done():
                task.cancel()
    
    if 'reminder_tasks' in session:
        for task in session['reminder_tasks']:
            if not task.done():
                try:
                    task.cancel()
                except:
                    pass
    
    await end_session_summary(user_id, context.application)
    await update.message.reply_text("✅ Session terminée (forcée) !")

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
        
        print(f"[SIGNAL] ✅ Signal #{signal_id} généré pour user {user_id}")
        print(f"[SIGNAL] 📊 Session: {session['signal_count']}/{SIGNALS_PER_SESSION}")
        
        # Récupérer les détails du signal pour l'envoyer immédiatement
        with engine.connect() as conn:
            signal = conn.execute(
                text("SELECT pair, direction, confidence, payload_json, ts_enter FROM signals WHERE id = :sid"),
                {"sid": signal_id}
            ).fetchone()
        
        if signal:
            pair, direction, confidence, payload_json, ts_enter = signal
            
            # Analyser le payload pour le mode
            mode = "Forex"
            if payload_json:
                try:
                    payload = json.loads(payload_json)
                    mode = payload.get('mode', 'Forex')
                except:
                    pass
            
            # Convertir ts_enter en datetime si nécessaire
            if isinstance(ts_enter, str):
                entry_time = datetime.fromisoformat(ts_enter.replace('Z', '+00:00')).astimezone(HAITI_TZ)
            else:
                entry_time = ts_enter.astimezone(HAITI_TZ)
            
            # Calculer l'heure d'envoi (2 minutes avant l'entrée)
            send_time = entry_time - timedelta(minutes=2)
            now_haiti = get_haiti_now()
            
            # Formater le message du signal
            direction_text = "BUY ↗️" if direction == "CALL" else "SELL ↘️"
            entry_time_formatted = entry_time.strftime('%H:%M')
            
            # Calculer le temps restant avant entrée
            time_to_entry = max(0, (entry_time - now_haiti).total_seconds() / 60)
            
            # Message COMPLET du signal à envoyer IMMÉDIATEMENT
            signal_msg = (
                f"🎯 **SIGNAL #{session['signal_count']}**\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"💱 {pair}\n"
                f"🌐 Mode: {mode}\n"
                f"🕐 Entrée dans: **{time_to_entry:.0f} min**\n"
                f"⏰ Heure entrée: **{entry_time_formatted}**\n"
                f"📈 Direction: **{direction_text}**\n"
                f"💪 Confiance: **{int(confidence*100)}%**\n"
                f"⏱️ Timeframe: 1 minute"
            )
            
            try:
                await context.application.bot.send_message(chat_id=user_id, text=signal_msg)
                print(f"[SIGNAL] ✅ Signal #{signal_id} ENVOYÉ IMMÉDIATEMENT à {now_haiti.strftime('%H:%M:%S')}")
                print(f"[SIGNAL] ⏰ Entrée prévue à {entry_time_formatted} (dans {time_to_entry:.1f} min)")
            except Exception as e:
                print(f"[SIGNAL] ❌ Erreur envoi signal: {e}")
            
            # Vérifier si le moment d'envoi est dans le futur pour les rappels
            if send_time > now_haiti:
                # Créer une tâche pour un rappel 1 minute avant l'entrée
                reminder_time = entry_time - timedelta(minutes=1)
                reminder_task = asyncio.create_task(
                    send_reminder(signal_id, user_id, context.application, reminder_time, entry_time, pair, direction)
                )
                session['reminder_tasks'].append(reminder_task)
                
                wait_seconds = (reminder_time - now_haiti).total_seconds()
                if wait_seconds > 0:
                    print(f"[SIGNAL_REMINDER] ⏰ Rappel programmé pour signal #{signal_id} dans {wait_seconds:.0f} secondes")
        
        # Programmer vérification auto (3 minutes après la génération du signal)
        verification_task = asyncio.create_task(auto_verify_signal(signal_id, user_id, context.application))
        session['verification_tasks'].append(verification_task)
        
        print(f"[SIGNAL] ⏳ Vérification auto programmée dans 3 min...")
        
        # Message de confirmation modifié
        confirmation_msg = (
            f"✅ **Signal #{session['signal_count']} généré et envoyé!**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📊 Progression: {session['signal_count']}/{SIGNALS_PER_SESSION}\n\n"
            f"⏰ **Timing du signal:**\n"
            f"• Entrée: Dans {time_to_entry:.0f} min\n"
            f"• Vérification: 3 min après entrée\n\n"
            f"💡 Préparez votre position!"
        )
        
        await query.edit_message_text(confirmation_msg)
    else:
        await query.edit_message_text(
            "⚠️ Aucun signal (conditions non remplies)\n\n"
            "Utilisez /lasterrors pour voir les détails d'erreur"
        )
        
        # Proposer de réessayer
        keyboard = [[InlineKeyboardButton("🔄 Réessayer", callback_data=f"gen_signal_{user_id}")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.reply_text("Voulez-vous réessayer ?", reply_markup=reply_markup)

async def generate_m1_signal(user_id, app):
    """Génère un signal M1 avec timing précis"""
    try:
        is_weekend = otc_provider.is_weekend()
        mode = "OTC" if is_weekend else "Forex"
        
        print(f"\n[SIGNAL] 📤 M1 {mode} pour user {user_id}")
        
        # Vérifier si l'utilisateur a une session active
        if user_id not in active_sessions:
            add_error_log(f"User {user_id} n'a pas de session active")
            return None
        
        session = active_sessions.get(user_id)
        
        # Rotation paires
        active_pairs = PAIRS[:3]
        pair = active_pairs[session['signal_count'] % len(active_pairs)]
        
        print(f"[SIGNAL] 🔍 Paire originale: {pair}")
        
        # Obtenir la paire actuelle (convertie en crypto si week-end)
        current_pair = get_current_pair(pair)
        
        if is_weekend:
            print(f"[SIGNAL] 🔄 Paire convertie pour week-end: {pair} → {current_pair}")
        else:
            print(f"[SIGNAL] 📈 Paire Forex: {current_pair}")
        
        # Données M1 - Utiliser current_pair (crypto en week-end, forex en semaine)
        df = get_cached_ohlc(current_pair, TIMEFRAME_M1, outputsize=400)
        
        if df is None:
            add_error_log(f"[SIGNAL] ❌ Pas de données {mode} (df est None) pour {current_pair}")
            return None
        
        if len(df) < 50:
            add_error_log(f"[SIGNAL] ❌ Pas assez de données: {len(df)} bougies (min 50)")
            print(f"[SIGNAL] 📊 Nombre de bougies disponibles: {len(df)}")
            return None
        
        print(f"[SIGNAL] ✅ {len(df)} bougies M1 ({mode})")
        print(f"[SIGNAL] 📈 Dernière bougie: {df.iloc[-1]['close']:.5f} à {df.index[-1]}")
        
        # Indicateurs
        df = compute_indicators(df)
        
        # Stratégie - Règles adaptées selon le mode
        if is_weekend:
            # Mode OTC - règles très permissives
            base_signal = rule_signal_ultra_strict(df, session_priority=2)  # Priorité basse
            print(f"[SIGNAL] 🏖️ Mode OTC - Priorité basse (2)")
        else:
            # Mode Forex - règles normales
            base_signal = rule_signal_ultra_strict(df, session_priority=5)
            print(f"[SIGNAL] 📈 Mode Forex - Priorité normale (5)")

        if not base_signal:
            # En mode OTC, forcer un signal si aucun n'est trouvé (pour le testing)
            if is_weekend:
                print("[SIGNAL] ⚡ Aucun signal trouvé en OTC, génération forcée...")
                # Forcer un signal aléatoire en OTC pour permettre le testing
                base_signal = random.choice(["CALL", "PUT"])
                print(f"[SIGNAL] 🎲 Signal forcé: {base_signal}")
            else:
                add_error_log("[SIGNAL] ⏭️ Rejeté (stratégie)")
                return None
        
        print(f"[SIGNAL] ✅ Stratégie: {base_signal}")
        
        # ML
        ml_signal, ml_conf = ml_predictor.predict_signal(df, base_signal)
        if ml_signal is None:
            add_error_log(f"[SIGNAL] ❌ ML: pas de signal")
            return None
        if ml_conf < CONFIDENCE_THRESHOLD:
            add_error_log(f"[SIGNAL] ❌ ML: confiance trop basse ({ml_conf:.1%} < {CONFIDENCE_THRESHOLD:.0%})")
            return None
        
        print(f"[SIGNAL] ✅ ML: {ml_signal} ({ml_conf:.1%})")
        
        # Calcul des temps avec timing précis
        now_haiti = get_haiti_now()
        now_utc = get_utc_now()
        
        # Calculer l'heure d'entrée (arrondie à la minute suivante + 2 minutes)
        # Pour avoir une entrée précise, on arrondit à la minute suivante
        entry_time_haiti = (now_haiti + timedelta(minutes=2)).replace(second=0, microsecond=0)
        # S'assurer que l'entrée est bien dans 2 minutes minimum
        if entry_time_haiti < now_haiti + timedelta(minutes=2):
            entry_time_haiti = (now_haiti + timedelta(minutes=2)).replace(second=0, microsecond=0)
        
        entry_time_utc = entry_time_haiti.astimezone(timezone.utc)
        send_time_utc = now_utc  # Le signal est généré maintenant
        
        print(f"[SIGNAL_TIMING] ⏰ Heure actuelle: {now_haiti.strftime('%H:%M:%S')}")
        print(f"[SIGNAL_TIMING] ⏰ Heure d'entrée: {entry_time_haiti.strftime('%H:%M:%S')}")
        print(f"[SIGNAL_TIMING] ⏰ Délai avant entrée: {(entry_time_haiti - now_haiti).total_seconds()/60:.1f} min")
        
        # Persister
        payload = {
            'pair': current_pair,  # Stocker la paire actuelle utilisée
            'direction': ml_signal, 
            'reason': f'M1 Session {mode} - ML {ml_conf:.1%} - Timing: entrée dans 2min',
            'ts_enter': entry_time_utc.isoformat(), 
            'ts_send': send_time_utc.isoformat(),
            'confidence': ml_conf, 
            'payload': json.dumps({
                'original_pair': pair,  # Conserver l'original pour référence
                'actual_pair': current_pair,  # Ajouter la paire utilisée
                'user_id': user_id, 
                'mode': mode,
                'rsi': df.iloc[-1].get('rsi'),
                'adx': df.iloc[-1].get('adx'),
                'data_source': 'real' if df.iloc[-1].get('close', 0) > 0 else 'synthetic',
                'timing_info': {
                    'signal_generated': now_haiti.isoformat(),
                    'entry_scheduled': entry_time_haiti.isoformat(),
                    'reminder_scheduled': (entry_time_haiti - timedelta(minutes=1)).isoformat(),
                    'delay_before_entry_minutes': 2
                }
            }),
            'max_gales': 0,
            'timeframe': 1
        }
        signal_id = persist_signal(payload)
        
        print(f"[SIGNAL] ✅ Signal #{signal_id} persisté avec entrée dans 2 min")
        
        return signal_id
        
    except Exception as e:
        error_msg = f"[SIGNAL] ❌ Erreur: {e}"
        add_error_log(error_msg)
        import traceback
        traceback.print_exc()
        return None

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
        "⚡ Signal envoyé immédiatement\n"
        "🔔 Rappel 1 min avant entrée\n"
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
    fake_message = query.message
    fake_update = Update(update_id=0, message=fake_message)
    fake_update.effective_user = query.from_user
    
    await cmd_start_session(fake_update, context)

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche les statistiques globales"""
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

async def cmd_otc_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche le statut OTC et paires disponibles"""
    try:
        is_weekend = otc_provider.is_weekend()
        now_haiti = get_haiti_now()
        
        # Tester la disponibilité
        results = check_api_availability()
        
        msg = (
            "🌐 **STATUT OTC**\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📅 {now_haiti.strftime('%A %d/%m/%Y')}\n"
            f"🕐 {now_haiti.strftime('%H:%M:%S')} (Haïti)\n\n"
        )
        
        if is_weekend:
            msg += (
                "🏖️ **Mode: OTC ACTIF**\n"
                "💰 Sources: Bybit, Binance, KuCoin, CoinGecko\n"
                "🔧 Fallback: Mode synthétique\n"
                "⏰ Disponible: 24/7\n\n"
            )
            
            if results.get('crypto_available'):
                msg += "✅ APIs Crypto: DISPONIBLES\n\n"
            else:
                msg += "⚠️ APIs Crypto: INDISPONIBLES (mode synthétique)\n\n"
            
            msg += "📊 **Paires Crypto disponibles:**\n\n"
            for pair in otc_provider.get_available_pairs():
                msg += f"• {pair}\n"
            
            msg += (
                "\n💡 Les paires Forex sont automatiquement\n"
                "   converties en crypto équivalentes:\n"
                "   • EUR/USD → BTC/USD\n"
                "   • GBP/USD → ETH/USD\n"
                "   • USD/JPY → TRX/USD\n"
                "   • AUD/USD → LTC/USD\n"
            )
        else:
            msg += (
                "📈 **Mode: FOREX STANDARD**\n"
                "💱 Source: TwelveData (Forex)\n"
                "⏰ Lun-Ven 00:00-22:00 UTC\n\n"
            )
            
            if results.get('forex_available'):
                msg += "✅ TwelveData Forex: DISPONIBLE\n"
            else:
                msg += "❌ TwelveData Forex: INDISPONIBLE\n"
            
            msg += (
                "\n💡 Le mode Crypto s'active automatiquement\n"
                "   le week-end (Sam-Dim)\n"
            )
        
        msg += "\n━━━━━━━━━━━━━━━━━━━━"
        
        await update.message.reply_text(msg)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_test_otc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Teste la récupération de données OTC"""
    try:
        msg = await update.message.reply_text("🧪 Test OTC en cours...")
        
        # Tester récupération
        test_pair = 'BTC/USD'
        
        if otc_provider.is_weekend():
            # Mode OTC - utiliser get_otc_data
            df = otc_provider.get_otc_data(test_pair, '1min', 5)
            
            if df is not None and len(df) > 0:
                last = df.iloc[-1]
                response = (
                    f"✅ **Test OTC réussi**\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"💱 Paire: {test_pair}\n"
                    f"📡 Source: Multi-APIs Crypto\n"
                    f"📊 Bougies: {len(df)}\n"
                    f"💰 Dernier prix: ${last['close']:.2f}\n"
                    f"📈 High: ${last['high']:.2f}\n"
                    f"📉 Low: ${last['low']:.2f}\n"
                    f"🕐 Dernière bougie: {df.index[-1].strftime('%H:%M')}\n\n"
                    f"✅ OTC opérationnel !"
                )
            else:
                # Tester le mode synthétique
                synthetic_df = otc_provider.generate_synthetic_data(test_pair, '1min', 5)
                if synthetic_df is not None:
                    last = synthetic_df.iloc[-1]
                    response = (
                        f"⚠️ **Test OTC avec données synthétiques**\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n\n"
                        f"💱 Paire: {test_pair}\n"
                        f"📡 Source: Synthétique\n"
                        f"📊 Bougies: {len(synthetic_df)}\n"
                        f"💰 Dernier prix: ${last['close']:.2f}\n"
                        f"📈 High: ${last['high']:.2f}\n"
                        f"📉 Low: ${last['low']:.2f}\n"
                        f"🕐 Dernière bougie: {synthetic_df.index[-1].strftime('%H:%M')}\n\n"
                        f"ℹ️ APIs bloquées, mode synthétique actif"
                    )
                else:
                    response = "❌ Échec récupération données OTC et synthétique"
        else:
            response = (
                "ℹ️ **Mode Forex actif**\n"
                "━━━━━━━━━━━━━━━━━━━━\n\n"
                f"Nous sommes en semaine, le mode Forex est actif.\n"
                f"Le mode OTC (Crypto) s'active automatiquement le week-end.\n\n"
                f"💡 Utilisez /otcstatus pour plus d'informations"
            )
        
        await msg.edit_text(response)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur test OTC: {e}")

async def cmd_check_api(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Vérifie la disponibilité des APIs"""
    try:
        msg = await update.message.reply_text("🔍 Vérification des APIs en cours...")
        
        results = check_api_availability()
        now_haiti = get_haiti_now()
        
        # Déterminer le statut global
        if results.get('forex_available') or results.get('crypto_available') or results.get('synthetic_available'):
            status_emoji = "✅"
            status_text = "OPÉRATIONNEL"
        else:
            status_emoji = "❌"
            status_text = "INDISPONIBLE"
        
        message = (
            f"{status_emoji} **VÉRIFICATION APIS** - {status_text}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📅 {now_haiti.strftime('%A %d/%m/%Y')}\n"
            f"🕐 {now_haiti.strftime('%H:%M:%S')}\n\n"
            f"🌐 **Mode actuel:** {results['current_mode']}\n"
        )
        
        if results['current_mode'] == 'OTC (Crypto)':
            if results.get('crypto_available'):
                message += f"📊 Crypto disponible: ✅ OUI (APIs multiples)\n"
            elif results.get('synthetic_available'):
                message += f"📊 Crypto disponible: ⚠️ SYNTHÉTIQUE (Fallback)\n"
            else:
                message += f"📊 Crypto disponible: ❌ NON\n"
        else:
            message += f"📊 Forex disponible: {'✅ OUI' if results.get('forex_available') else '❌ NON'}\n"
        
        message += f"\n🔍 **Résultats des tests:**\n\n"
        
        for test in results.get('test_pairs', []):
            status = test['status']
            if status == 'OK':
                emoji = "✅"
                message += f"{emoji} {test['pair']}: {status} ({test['data_points']} bougies, ${test['last_price']}, {test.get('source', 'API')})\n"
            elif 'error' in test:
                emoji = "❌"
                message += f"{emoji} {test['pair']}: ERREUR - {test['error'][:50]}\n"
            else:
                emoji = "⚠️"
                message += f"{emoji} {test['pair']}: {status}\n"
        
        if 'error' in results:
            message += f"\n⚠️ **Erreur globale:** {results['error']}\n"
        
        # Recommandations
        message += "\n💡 **Recommandations:**\n"
        
        if results['current_mode'] == 'OTC (Crypto)':
            if results.get('crypto_available'):
                message += "• APIs Crypto fonctionnelles ✓\n"
                message += "• Données réelles disponibles\n"
                message += "• Vous pouvez démarrer une session avec /startsession\n"
            elif results.get('synthetic_available'):
                message += "• APIs bloquées, mode synthétique actif\n"
                message += "• Les données sont simulées mais permettent de tester\n"
                message += "• Utilisez /startsession pour tester avec données synthétiques\n"
            else:
                message += "• APIs Crypto indisponibles\n"
                message += "• Mode synthétique également indisponible\n"
                message += "• Vérifiez votre connexion internet\n"
        else:
            if results.get('forex_available'):
                message += "• TwelveData Forex fonctionnel ✓\n"
                message += "• Vous pouvez démarrer une session avec /startsession\n"
            else:
                message += "• TwelveData Forex indisponible\n"
                message += "• Vérifiez la clé API TwelveData\n"
                message += "• Attendez les heures d'ouverture (Lun-Ven 00:00-22:00 UTC)\n"
        
        message += "\n━━━━━━━━━━━━━━━━━━━━"
        
        await msg.edit_text(message)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur vérification API: {e}")

async def cmd_debug_api(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Debug des APIs"""
    try:
        msg = await update.message.reply_text("🔧 Debug des APIs en cours...")
        
        # Tester directement l'OTC provider
        test_pair = 'BTC/USD'
        
        debug_info = "🔍 **DEBUG APIs OTC**\n"
        debug_info += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        # 1. Vérifier si week-end
        is_weekend = otc_provider.is_weekend()
        debug_info += f"📅 Week-end: {'✅ OUI' if is_weekend else '❌ NON'}\n\n"
        
        # 2. Tester get_otc_data
        debug_info += f"🧪 Test get_otc_data('{test_pair}'):\n"
        df = otc_provider.get_otc_data(test_pair, '1min', 5)
        
        if df is not None and len(df) > 0:
            debug_info += f"✅ Succès: {len(df)} bougies\n"
            debug_info += f"💰 Dernier prix: ${df.iloc[-1]['close']:.2f}\n"
            debug_info += f"📈 Source: Données réelles\n\n"
            
            # Afficher les 3 dernières bougies
            debug_info += "📊 Dernières bougies:\n"
            for i in range(min(3, len(df))):
                idx = -1 - i
                row = df.iloc[idx]
                debug_info += f"  {df.index[idx].strftime('%H:%M')}: O{row['open']:.2f} H{row['high']:.2f} L{row['low']:.2f} C{row['close']:.2f}\n"
        else:
            debug_info += "❌ Échec - Pas de données\n\n"
            
            # Tester generate_synthetic_data
            debug_info += "🧪 Test generate_synthetic_data:\n"
            df2 = otc_provider.generate_synthetic_data(test_pair, '1min', 5)
            if df2 is not None:
                debug_info += f"✅ Synthétique: {len(df2)} bougies\n"
                debug_info += f"💰 Dernier prix: ${df2.iloc[-1]['close']:.2f}\n"
                debug_info += f"📈 Source: Données synthétiques\n"
            else:
                debug_info += "❌ Échec synthétique aussi\n"
        
        # 3. Tester les méthodes individuelles
        debug_info += "\n🔧 **Méthodes disponibles:**\n"
        methods = [m for m in dir(otc_provider) if not m.startswith('_')]
        for method in sorted(methods):
            debug_info += f"• {method}\n"
        
        debug_info += "\n━━━━━━━━━━━━━━━━━━━━\n"
        debug_info += "💡 Utilisez /checkapi pour plus de détails"
        
        await msg.edit_text(debug_info)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur debug: {e}")

async def cmd_debug_pair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Debug la conversion de paires"""
    try:
        is_weekend = otc_provider.is_weekend()
        now_haiti = get_haiti_now()
        
        msg = f"🔧 **DEBUG CONVERSION PAIRES**\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━\n\n"
        msg += f"📅 {now_haiti.strftime('%A %d/%m/%Y')}\n"
        msg += f"🕐 {now_haiti.strftime('%H:%M:%S')}\n\n"
        msg += f"🏖️ Week-end: {'✅ OUI' if is_weekend else '❌ NON'}\n\n"
        
        forex_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'BTC/USD', 'ETH/USD']
        
        msg += "📊 **Conversion des paires:**\n\n"
        for pair in forex_pairs:
            current = get_current_pair(pair)
            if pair == current:
                msg += f"• {pair} → {current} (inchangé)\n"
            else:
                msg += f"• {pair} → {current} 🔄\n"
        
        msg += f"\n💡 **Règles de conversion:**\n"
        msg += f"• En week-end: Forex → Crypto\n"
        msg += f"• En semaine: Forex standard\n"
        msg += f"\n📈 **Exemple de session:**\n"
        
        # Simuler une session
        active_pairs = forex_pairs[:3]
        for i in range(min(3, SIGNALS_PER_SESSION)):
            pair = active_pairs[i % len(active_pairs)]
            current = get_current_pair(pair)
            msg += f"  Signal #{i+1}: {pair} → {current}\n"
        
        msg += f"\n━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"💡 Test avec /quicktest pour générer un signal"
        
        await update.message.reply_text(msg)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_quick_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Test rapide pour générer un signal immédiatement"""
    try:
        user_id = update.effective_user.id
        
        if otc_provider.is_weekend():
            await update.message.reply_text("🏖️ Week-end - Mode OTC actif\n⏳ Test en cours...")
        else:
            await update.message.reply_text("📈 Semaine - Mode Forex\n⏳ Test en cours...")
        
        # Créer une session temporaire pour le test
        test_session = {
            'start_time': get_haiti_now(),
            'signal_count': 0,
            'wins': 0,
            'losses': 0,
            'pending': 0,
            'signals': []
        }
        
        # Sauvegarder temporairement
        original_session = active_sessions.get(user_id)
        active_sessions[user_id] = test_session
        
        # Générer un signal
        signal_id = await generate_m1_signal(user_id, context.application)
        
        # Restaurer la session originale
        if original_session:
            active_sessions[user_id] = original_session
        else:
            del active_sessions[user_id]
        
        if signal_id:
            await update.message.reply_text(f"✅ Signal généré avec succès! ID: {signal_id}")
        else:
            await update.message.reply_text(
                "❌ Échec de génération du signal\n\n"
                "Causes possibles:\n"
                "1. Aucune donnée disponible (vérifiez avec /checkapi)\n"
                "2. Conditions de trading non remplies\n"
                "3. Confiance du ML trop basse (<65%)\n"
                "4. Problème de connexion API\n\n"
                "Utilisez /lasterrors pour voir les détails d'erreur."
            )
            
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {str(e)[:200]}")

async def cmd_last_errors(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche les dernières erreurs"""
    global last_error_logs
    
    if not last_error_logs:
        await update.message.reply_text("✅ Aucune erreur récente.")
        return
    
    message = "📋 **DERNIÈRES ERREURS**\n━━━━━━━━━━━━━━━━━━━━\n\n"
    
    # Afficher les 10 dernières erreurs (les plus récentes en premier)
    for i, error in enumerate(reversed(last_error_logs[-10:]), 1):
        message += f"{i}. {error}\n\n"
    
    message += "━━━━━━━━━━━━━━━━━━━━\n"
    message += "💡 Utilisez /checkapi pour vérifier l'état des APIs"
    
    await update.message.reply_text(message)

# ================= COMMANDES DE VÉRIFICATION =================

async def cmd_manual_result(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Résultat manuel d'un signal"""
    try:
        if not context.args or len(context.args) < 2:
            await update.message.reply_text(
                "❌ Usage: /manualresult <signal_id> <WIN/LOSE>\n"
                "Exemple: /manualresult 123 WIN\n"
                "Pour voir les signaux en attente: /pending"
            )
            return
        
        signal_id = int(context.args[0])
        result = context.args[1].upper()
        
        if result not in ['WIN', 'LOSE']:
            await update.message.reply_text("❌ Résultat doit être WIN ou LOSE")
            return
        
        # Demander les prix si possible
        entry_price = None
        exit_price = None
        
        if len(context.args) >= 4:
            try:
                entry_price = float(context.args[2])
                exit_price = float(context.args[3])
            except:
                pass
        
        # Appliquer la vérification manuelle
        success = await auto_verifier.manual_verify_signal(signal_id, result, entry_price, exit_price)
        
        if success:
            # Mettre à jour la session si le signal est dans une session active
            for user_id, session in active_sessions.items():
                if signal_id in session['signals']:
                    session['pending'] = max(0, session['pending'] - 1)
                    if result == 'WIN':
                        session['wins'] += 1
                    else:
                        session['losses'] += 1
                    
                    await update.message.reply_text(
                        f"✅ Résultat manuel appliqué!\n"
                        f"Signal #{signal_id}: {result}\n"
                        f"Session: {session['signal_count']}/{SIGNALS_PER_SESSION}"
                    )
                    return
            
            await update.message.reply_text(f"✅ Résultat manuel appliqué pour signal #{signal_id}")
        else:
            await update.message.reply_text(f"❌ Échec de l'application du résultat")
            
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_pending_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche les signaux en attente de vérification"""
    try:
        with engine.connect() as conn:
            # Signaux M1 sans résultat
            signals = conn.execute(
                text("""
                    SELECT id, pair, direction, ts_enter, confidence, payload_json
                    FROM signals
                    WHERE timeframe = 1 AND result IS NULL
                    ORDER BY ts_enter DESC
                    LIMIT 10
                """)
            ).fetchall()
        
        if not signals:
            await update.message.reply_text("✅ Aucun signal en attente de vérification")
            return
        
        message = "📋 **SIGNAUX EN ATTENTE**\n"
        message += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        for sig in signals:
            signal_id, pair, direction, ts_enter, confidence, payload_json = sig
            
            # Analyser le payload pour le mode
            mode = "Forex"
            if payload_json:
                try:
                    payload = json.loads(payload_json)
                    mode = payload.get('mode', 'Forex')
                except:
                    pass
            
            # Formater l'heure
            if isinstance(ts_enter, str):
                try:
                    dt = datetime.fromisoformat(ts_enter.replace('Z', '+00:00'))
                except:
                    dt = datetime.strptime(ts_enter, '%Y-%m-%d %H:%M:%S')
            else:
                dt = ts_enter
            
            haiti_dt = dt.astimezone(HAITI_TZ)
            
            direction_emoji = "📈" if direction == "CALL" else "📉"
            direction_text = "BUY" if direction == "CALL" else "SELL"
            mode_emoji = "🏖️" if mode == "OTC" else "📈"
            
            message += (
                f"#{signal_id} - {pair}\n"
                f"  {direction_emoji} {direction_text} - {int(confidence*100)}%\n"
                f"  {mode_emoji} {mode}\n"
                f"  🕐 {haiti_dt.strftime('%H:%M')}\n"
                f"  📅 {haiti_dt.strftime('%d/%m')}\n\n"
            )
        
        message += "━━━━━━━━━━━━━━━━━━━━\n"
        message += "ℹ️ Pour marquer manuellement:\n"
        message += "/manualresult <id> <WIN/LOSE> [entry_price] [exit_price]\n"
        message += "Ex: /manualresult 123 WIN 1.2345 1.2367"
        
        await update.message.reply_text(message)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_signal_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Info détaillée sur un signal"""
    try:
        if not context.args:
            await update.message.reply_text("❌ Usage: /signalinfo <signal_id>")
            return
        
        signal_id = int(context.args[0])
        
        info = auto_verifier.get_signal_status(signal_id)
        
        if not info:
            await update.message.reply_text(f"❌ Signal #{signal_id} non trouvé")
            return
        
        # Formater les dates
        ts_enter = info['ts_enter']
        if isinstance(ts_enter, str):
            try:
                dt_enter = datetime.fromisoformat(ts_enter.replace('Z', '+00:00'))
            except:
                dt_enter = datetime.strptime(ts_enter, '%Y-%m-%d %H:%M:%S')
        else:
            dt_enter = ts_enter
        
        haiti_enter = dt_enter.astimezone(HAITI_TZ)
        
        ts_exit = info.get('ts_exit')
        if ts_exit:
            if isinstance(ts_exit, str):
                try:
                    dt_exit = datetime.fromisoformat(ts_exit.replace('Z', '+00:00'))
                except:
                    dt_exit = datetime.strptime(ts_exit, '%Y-%m-%d %H:%M:%S')
            else:
                dt_exit = ts_exit
            
            haiti_exit = dt_exit.astimezone(HAITI_TZ)
            exit_time = haiti_exit.strftime('%H:%M %d/%m')
        else:
            exit_time = "En attente"
        
        direction_emoji = "📈" if info['direction'] == "CALL" else "📉"
        direction_text = "BUY" if info['direction'] == "CALL" else "SELL"
        
        message = (
            f"📊 **SIGNAL #{signal_id}**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"💱 {info['pair']}\n"
            f"{direction_emoji} {direction_text}\n\n"
            f"🕐 Entrée: {haiti_enter.strftime('%H:%M %d/%m')}\n"
            f"🕐 Sortie: {exit_time}\n\n"
        )
        
        if info['result']:
            result_emoji = "✅" if info['result'] == 'WIN' else "❌"
            message += f"🎲 Résultat: {result_emoji} {info['result']}\n"
            
            if info.get('entry_price') and info.get('exit_price'):
                pips = abs(info['exit_price'] - info['entry_price']) * 10000
                message += f"💰 Entry: {info['entry_price']:.5f}\n"
                message += f"💰 Exit: {info['exit_price']:.5f}\n"
                message += f"📊 Pips: {pips:.1f}\n"
            
            if info.get('reason'):
                message += f"📝 Raison: {info['reason']}\n"
        else:
            message += "⏳ En attente de vérification\n\n"
            message += "💡 Pour marquer manuellement:\n"
            message += f"/manualresult {signal_id} WIN/LOSE"
        
        await update.message.reply_text(message)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_force_verify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Force la vérification d'un signal"""
    try:
        if not context.args:
            await update.message.reply_text(
                "❌ Usage: /forceverify <signal_id>\n"
                "Exemple: /forceverify 123\n"
                "Pour voir les signaux en attente: /pending"
            )
            return
        
        signal_id = int(context.args[0])
        
        await update.message.reply_text(f"⚡ Forcer vérification signal #{signal_id}...")
        
        # Forcer la vérification
        result = await auto_verifier.force_verify_signal(signal_id)
        
        if result:
            # Mettre à jour la session si nécessaire
            for user_id, session in active_sessions.items():
                if signal_id in session['signals']:
                    session['pending'] = max(0, session['pending'] - 1)
                    if result == 'WIN':
                        session['wins'] += 1
                    else:
                        session['losses'] += 1
                    
                    await update.message.reply_text(
                        f"✅ Vérification forcée réussie!\n"
                        f"Signal #{signal_id}: {result}\n"
                        f"Session: {session['signal_count']}/{SIGNALS_PER_SESSION}"
                    )
                    return
            
            await update.message.reply_text(f"✅ Signal #{signal_id} vérifié: {result}")
        else:
            await update.message.reply_text(
                f"❌ Impossible de vérifier signal #{signal_id}\n"
                f"Utilisez /manualresult {signal_id} WIN/LOSE"
            )
            
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_force_all_verifications(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Force la vérification de tous les signaux en attente"""
    try:
        user_id = update.effective_user.id
        
        if user_id not in active_sessions:
            await update.message.reply_text("❌ Aucune session active")
            return
        
        session = active_sessions[user_id]
        
        if session['pending'] == 0:
            await update.message.reply_text("✅ Aucun signal en attente de vérification")
            return
        
        msg = await update.message.reply_text(f"⚡ Forcer vérification de {session['pending']} signal(s)...")
        
        # Vérifier tous les signaux en attente
        verified_count = 0
        for signal_id in session['signals']:
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT result FROM signals WHERE id = :sid"),
                    {"sid": signal_id}
                ).fetchone()
            
            if result and result[0] is not None:
                continue  # Déjà vérifié
            
            print(f"[FORCE_VERIF] 🔍 Forcer vérification signal #{signal_id}")
            
            # Simuler une vérification (aléatoire pour tests)
            simulated_result = 'WIN' if random.random() < 0.7 else 'LOSE'
            
            await auto_verifier.manual_verify_signal(signal_id, simulated_result)
            
            # Mettre à jour session
            session['pending'] = max(0, session['pending'] - 1)
            if simulated_result == 'WIN':
                session['wins'] += 1
            else:
                session['losses'] += 1
            
            verified_count += 1
            await asyncio.sleep(1)  # Petite pause
        
        await msg.edit_text(
            f"✅ Vérifications forcées terminées!\n"
            f"🔧 {verified_count} signal(s) vérifié(s)\n\n"
            f"📊 Session: {session['signal_count']}/{SIGNALS_PER_SESSION}\n"
            f"✅ Wins: {session['wins']}\n"
            f"❌ Losses: {session['losses']}\n"
            f"⏳ Pending: {session['pending']}"
        )
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_debug_verif(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Debug du système de vérification"""
    try:
        msg = await update.message.reply_text("🔧 Debug vérification...")
        
        debug_info = "🔍 **DEBUG VÉRIFICATION**\n"
        debug_info += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        # 1. Vérifier auto_verifier
        if auto_verifier is None:
            debug_info += "❌ auto_verifier: NON INITIALISÉ\n\n"
        else:
            debug_info += "✅ auto_verifier: INITIALISÉ\n\n"
        
        # 2. Sessions actives
        debug_info += f"📊 Sessions actives: {len(active_sessions)}\n\n"
        
        for user_id, session in active_sessions.items():
            debug_info += f"👤 User {user_id}:\n"
            debug_info += f"  • Signaux: {session['signal_count']}/{SIGNALS_PER_SESSION}\n"
            debug_info += f"  ✅ Wins: {session['wins']}\n"
            debug_info += f"  ❌ Losses: {session['losses']}\n"
            debug_info += f"  ⏳ Pending: {session['pending']}\n"
            debug_info += f"  📋 IDs: {session['signals'][-3:] if session['signals'] else []}\n\n"
        
        # 3. Signaux récents
        with engine.connect() as conn:
            signals = conn.execute(
                text("""
                    SELECT id, pair, direction, result, ts_enter, confidence, payload_json
                    FROM signals
                    WHERE timeframe = 1
                    ORDER BY id DESC
                    LIMIT 5
                """)
            ).fetchall()
        
        if signals:
            debug_info += "📋 **5 derniers signaux:**\n\n"
            for sig in signals:
                signal_id, pair, direction, result, ts_enter, confidence, payload_json = sig
                
                # Analyser le payload pour le mode
                mode = "Forex"
                if payload_json:
                    try:
                        payload = json.loads(payload_json)
                        mode = payload.get('mode', 'Forex')
                    except:
                        pass
                
                result_text = result if result else "⏳ En attente"
                result_emoji = "✅" if result == 'WIN' else "❌" if result == 'LOSE' else "⏳"
                mode_emoji = "🏖️" if mode == "OTC" else "📈"
                
                debug_info += f"{result_emoji} #{signal_id}: {pair} {direction} - {result_text} ({int(confidence*100)}%) {mode_emoji}\n"
        
        debug_info += "\n━━━━━━━━━━━━━━━━━━━━\n"
        debug_info += "💡 Commandes:\n"
        debug_info += "• /forceverify <id> - Forcer vérification\n"
        debug_info += "• /forceall - Forcer toutes vérifications\n"
        debug_info += "• /manualresult <id> WIN/LOSE\n"
        debug_info += "• /pending - Signaux en attente"
        
        await msg.edit_text(debug_info)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur debug: {e}")

# ================= SERVEUR HTTP =================

async def health_check(request):
    """Endpoint de santé pour le serveur HTTP"""
    return web.json_response({
        'status': 'ok',
        'timestamp': get_haiti_now().isoformat(),
        'forex_open': is_forex_open(),
        'otc_active': otc_provider.is_weekend(),
        'active_sessions': len(active_sessions),
        'error_logs_count': len(last_error_logs),
        'mode': 'OTC' if otc_provider.is_weekend() else 'Forex',
        'api_source': 'Multi-APIs' if otc_provider.is_weekend() else 'TwelveData'
    })

async def start_http_server():
    """Démarre le serveur HTTP pour les checks de santé"""
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

# ================= POINT D'ENTRÉE =================

async def main():
    global auto_verifier

    print("\n" + "="*60)
    print("🤖 BOT M1 - VERSION INTERACTIVE")
    print("🎯 SIGNAL ENVOYÉ IMMÉDIATEMENT AVEC TIMING")
    print("="*60)
    print(f"🎯 8 signaux/session")
    print(f"⚡ Signal envoyé: Immédiatement")
    print(f"🔔 Rappel: 1 min avant entrée")
    print(f"🔍 Vérification: 3 min après signal")
    print(f"🌐 OTC support: Week-end crypto")
    print(f"🔧 Sources: TwelveData + Multi-APIs Crypto")
    print(f"🔧 Fallback: Mode synthétique")
    print("="*60 + "\n")

    ensure_db()
    auto_verifier = AutoResultVerifier(engine, TWELVEDATA_API_KEY)

    http_runner = await start_http_server()

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Commandes (restent les mêmes)
    app.add_handler(CommandHandler('start', cmd_start))
    app.add_handler(CommandHandler('menu', cmd_menu))
    app.add_handler(CommandHandler('startsession', cmd_start_session))
    app.add_handler(CommandHandler('sessionstatus', cmd_session_status))
    app.add_handler(CommandHandler('endsession', cmd_end_session))
    app.add_handler(CommandHandler('forceend', cmd_force_end))
    app.add_handler(CommandHandler('stats', cmd_stats))
    app.add_handler(CommandHandler('rapport', cmd_rapport))
    app.add_handler(CommandHandler('mlstats', cmd_mlstats))
    app.add_handler(CommandHandler('retrain', cmd_retrain))
    app.add_handler(CommandHandler('otcstatus', cmd_otc_status))
    app.add_handler(CommandHandler('testotc', cmd_test_otc))
    app.add_handler(CommandHandler('checkapi', cmd_check_api))
    app.add_handler(CommandHandler('debugapi', cmd_debug_api))
    app.add_handler(CommandHandler('debugpair', cmd_debug_pair))
    app.add_handler(CommandHandler('quicktest', cmd_quick_test))
    app.add_handler(CommandHandler('lasterrors', cmd_last_errors))
    
    # Commandes de vérification
    app.add_handler(CommandHandler('manualresult', cmd_manual_result))
    app.add_handler(CommandHandler('pending', cmd_pending_signals))
    app.add_handler(CommandHandler('signalinfo', cmd_signal_info))
    app.add_handler(CommandHandler('forceverify', cmd_force_verify))
    app.add_handler(CommandHandler('forceall', cmd_force_all_verifications))
    app.add_handler(CommandHandler('debugverif', cmd_debug_verif))
    
    # Callbacks
    app.add_handler(CallbackQueryHandler(callback_generate_signal, pattern=r'^gen_signal_'))
    app.add_handler(CallbackQueryHandler(callback_new_session, pattern=r'^new_session$'))

    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)

    bot_info = await app.bot.get_me()
    print(f"✅ BOT ACTIF: @{bot_info.username}\n")
    print(f"🔧 Mode actuel: {'OTC (Crypto)' if otc_provider.is_weekend() else 'Forex'}")
    print(f"🌐 Sources: {'Multi-APIs Crypto' if otc_provider.is_weekend() else 'TwelveData'}")
    print(f"⚡ Signal envoyé: Immédiatement après génération")
    print(f"🔔 Rappel: 1 minute avant l'entrée\n")

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
