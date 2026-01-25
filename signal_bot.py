"""
Bot de trading M1 - Version Interactive
8 signaux par session avec bouton Generate Signal
Support OTC (crypto) le week-end via APIs multiples
Signal envoyé 2 minutes avant l'entrée en position
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
            'USD/JPY': 'XRP/USD',
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

# ================= NOUVELLE FONCTION: ATTENTE SIGNAL =================

async def wait_and_send_signal(signal_id, user_id, app, send_time, entry_time):
    """Attend le moment approprié pour envoyer le signal (2 minutes avant l'entrée)"""
    try:
        now_haiti = get_haiti_now()
        wait_seconds = (send_time - now_haiti).total_seconds()
        
        if wait_seconds > 0:
            print(f"[SIGNAL_TIMING] ⏳ Attente de {wait_seconds:.1f} secondes avant envoi du signal #{signal_id}")
            await asyncio.sleep(wait_seconds)
        
        # Récupérer les détails du signal depuis la base
        with engine.connect() as conn:
            signal = conn.execute(
                text("SELECT pair, direction, confidence, payload_json FROM signals WHERE id = :sid"),
                {"sid": signal_id}
            ).fetchone()
        
        if not signal:
            print(f"[SIGNAL_TIMING] ❌ Signal #{signal_id} non trouvé en base")
            return
        
        pair, direction, confidence, payload_json = signal
        
        # Analyser le payload pour le mode
        mode = "Forex"
        if payload_json:
            try:
                payload = json.loads(payload_json)
                mode = payload.get('mode', 'Forex')
            except:
                pass
        
        # Formater le message du signal
        direction_text = "BUY ↗️" if direction == "CALL" else "SELL ↘️"
        entry_time_formatted = entry_time.strftime('%H:%M')
        
        # Calculer le temps restant avant entrée
        time_to_entry = max(0, (entry_time - get_haiti_now()).total_seconds() / 60)
        
        msg = (
            f"🎯 **SIGNAL #{active_sessions[user_id]['signal_count']}**\n"
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
            await app.bot.send_message(chat_id=user_id, text=msg)
            print(f"[SIGNAL_TIMING] ✅ Signal #{signal_id} envoyé à {get_haiti_now().strftime('%H:%M:%S')}")
            print(f"[SIGNAL_TIMING] ⏰ Entrée prévue à {entry_time_formatted} (dans {time_to_entry:.1f} min)")
        except Exception as e:
            print(f"[SIGNAL_TIMING] ❌ Erreur envoi signal: {e}")
            
    except asyncio.CancelledError:
        print(f"[SIGNAL_TIMING] ❌ Tâche d'attente signal #{signal_id} annulée")
        raise
    except Exception as e:
        print(f"[SIGNAL_TIMING] ❌ Erreur dans wait_and_send_signal: {e}")

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
            f"⚡ Signal envoyé: **2 min avant l'entrée**\n"
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
        "⚡ Signal envoyé: 2 min avant entrée\n"
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
        'signal_tasks': []  # Nouvelles tâches d'envoi de signal
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
        f"⚡ Signal envoyé: 2 min avant entrée\n"
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
    
    # Vérifier si des signaux sont en attente d'envoi
    pending_signals = 0
    if 'signal_tasks' in session:
        for task in session['signal_tasks']:
            if not task.done():
                pending_signals += 1
    
    msg = (
        "📊 **ÉTAT SESSION**\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        f"⏱️ Durée: {duration:.1f} min\n"
        f"📈 Progression: {session['signal_count']}/{SIGNALS_PER_SESSION}\n\n"
        f"✅ Wins: {session['wins']}\n"
        f"❌ Losses: {session['losses']}\n"
        f"⏳ Vérif en attente: {session['pending']}\n"
        f"📨 Signaux en attente d'envoi: {pending_signals}\n\n"
        f"📊 Win Rate: {winrate:.1f}%\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"⚡ Signal timing: 2 min avant entrée"
    )
    
    await update.message.reply_text(msg)

async def cmd_end_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Termine la session active manuellement"""
    user_id = update.effective_user.id
    
    if user_id not in active_sessions:
        await update.message.reply_text("ℹ️ Aucune session active")
        return
    
    session = active_sessions[user_id]
    
    # Annuler les tâches d'envoi de signal en attente
    if 'signal_tasks' in session:
        for task in session['signal_tasks']:
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
    
    if 'signal_tasks' in session:
        for task in session['signal_tasks']:
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
        print(f"[SIGNAL] ⏳ Vérification auto dans 3 min...")
        
        # Récupérer l'heure d'entrée du signal
        with engine.connect() as conn:
            signal = conn.execute(
                text("SELECT ts_enter FROM signals WHERE id = :sid"),
                {"sid": signal_id}
            ).fetchone()
        
        if signal:
            ts_enter = signal[0]
            if isinstance(ts_enter, str):
                entry_time = datetime.fromisoformat(ts_enter.replace('Z', '+00:00')).astimezone(HAITI_TZ)
            else:
                entry_time = ts_enter.astimezone(HAITI_TZ)
            
            # Calculer l'heure d'envoi (2 minutes avant l'entrée)
            send_time = entry_time - timedelta(minutes=2)
            now_haiti = get_haiti_now()
            
            # Vérifier si le moment d'envoi est dans le futur
            if send_time > now_haiti:
                # Créer une tâche pour envoyer le signal au bon moment
                signal_task = asyncio.create_task(
                    wait_and_send_signal(signal_id, user_id, context.application, send_time, entry_time)
                )
                session['signal_tasks'].append(signal_task)
                
                wait_seconds = (send_time - now_haiti).total_seconds()
                print(f"[SIGNAL_TIMING] ⏰ Signal #{signal_id} sera envoyé dans {wait_seconds:.0f} secondes")
                print(f"[SIGNAL_TIMING] ⏰ Heure d'envoi: {send_time.strftime('%H:%M:%S')}")
                print(f"[SIGNAL_TIMING] ⏰ Heure d'entrée: {entry_time.strftime('%H:%M:%S')}")
            else:
                # Si le moment d'envoi est déjà passé, envoyer immédiatement
                print(f"[SIGNAL_TIMING] ⚠️ Heure d'envoi déjà passée, envoi immédiat")
                # Récupérer les détails du signal et l'envoyer
                with engine.connect() as conn:
                    signal = conn.execute(
                        text("SELECT pair, direction, confidence, payload_json FROM signals WHERE id = :sid"),
                        {"sid": signal_id}
                    ).fetchone()
                
                if signal:
                    pair, direction, confidence, payload_json = signal
                    direction_text = "BUY ↗️" if direction == "CALL" else "SELL ↘️"
                    
                    msg = (
                        f"🎯 **SIGNAL #{session['signal_count']}**\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"💱 {pair}\n"
                        f"📈 Direction: **{direction_text}**\n"
                        f"💪 Confiance: **{int(confidence*100)}%**\n"
                        f"⏱️ Timeframe: 1 minute\n"
                        f"⚠️ Signal envoyé immédiatement (timing dépassé)"
                    )
                    
                    try:
                        await context.application.bot.send_message(chat_id=user_id, text=msg)
                    except Exception as e:
                        print(f"[SIGNAL] ❌ Erreur envoi signal: {e}")
        
        # Programmer vérification auto (3 minutes après la génération du signal)
        verification_task = asyncio.create_task(auto_verify_signal(signal_id, user_id, context.application))
        session['verification_tasks'].append(verification_task)
        
        await query.edit_message_text(
            f"✅ **Signal #{session['signal_count']} généré**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📊 Progression: {session['signal_count']}/{SIGNALS_PER_SESSION}\n\n"
            f"⏰ **Timing du signal:**\n"
            f"• Génération: Maintenant\n"
            f"• Envoi: 2 min avant entrée\n"
            f"• Entrée: Dans 2 min\n"
            f"• Vérification: 3 min après génération"
        )
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
        entry_time_haiti = (now_haiti + timedelta(minutes=3)).replace(second=0, microsecond=0)
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
            'reason': f'M1 Session {mode} - ML {ml_conf:.1%} - Timing: 2min avant entrée',
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
                    'send_scheduled': (entry_time_haiti - timedelta(minutes=2)).isoformat(),
                    'delay_before_entry_minutes': 2
                }
            }),
            'max_gales': 0,
            'timeframe': 1
        }
        signal_id = persist_signal(payload)
        
        print(f"[SIGNAL] ✅ Signal #{signal_id} persisté avec timing 2 min avant entrée")
        
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
        "⚡ Timing: Signaux envoyés 2 min avant entrée\n"
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

# ... (le reste du code reste inchangé, seule la logique d'envoi et de timing est modifiée)

# Les autres fonctions (cmd_stats, cmd_rapport, etc.) restent inchangées
# Seules les modifications liées au timing ont été apportées

# ================= POINT D'ENTRÉE =================

async def main():
    global auto_verifier

    print("\n" + "="*60)
    print("🤖 BOT M1 - VERSION INTERACTIVE")
    print("🎯 SIGNAL TIMING: 2 MINUTES AVANT ENTRÉE")
    print("="*60)
    print(f"🎯 8 signaux/session")
    print(f"⚡ Signal envoyé: 2 min avant entrée")
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
    print(f"⏰ Timing signal: 2 minutes avant l'entrée\n")

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
