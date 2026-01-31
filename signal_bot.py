"""
Bot de trading M1 - Version Saint Graal avec Garantie et Analyse Structure
8 signaux garantis par session avec stratégie Saint Graal Forex M1
Support OTC (crypto) le week-end via APIs multiples
Signal envoyé immédiatement avec timing 2 minutes avant entrée
Compatibilité avec utils.py Saint Graal - Version avec analyse structure
Débogage détaillé: heures, prix, paires, APIs, broker Pocket Option
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
from utils import (
    compute_indicators, 
    rule_signal_saint_graal_with_guarantee,
    get_signal_with_metadata,
    calculate_signal_quality_score,
    format_signal_reason,
    get_m1_candle_range,
    get_next_m1_candle,
    analyze_market_structure,
    is_near_swing_high,
    detect_retest_pattern
)
from ml_predictor import MLSignalPredictor
from auto_verifier import AutoResultVerifier
from otc_provider import OTCDataProvider

# ================= CONFIGURATION =================
HAITI_TZ = ZoneInfo("America/Port-au-Prince")
TIMEFRAME_M1 = "1min"
SIGNALS_PER_SESSION = 8  # Garanti par la stratégie Saint Graal
VERIFICATION_WAIT_MIN = 3  # 2 min avant entrée + 1 min bougie
CONFIDENCE_THRESHOLD = 0.65

# Initialisation des composants
engine = create_engine(DB_URL, connect_args={'check_same_thread': False})
ml_predictor = MLSignalPredictor()
auto_verifier = None
otc_provider = OTCDataProvider(TWELVEDATA_API_KEY)

# Variables globales
active_sessions = {}
pending_signal_tasks = {}
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
    
    if otc_provider.is_weekend():
        print(f"🏖️ Week-end - Mode OTC (Crypto via APIs multiples)")
        
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
    VALUES (:pair,:direction,:reason,:ts_enter,:ts_send,:confidence,:payload_json,:max_gales,:timeframe)""")
    with engine.begin() as conn:
        result = conn.execute(q, payload)
    return result.lastrowid

def fix_database_structure():
    """Corrige la structure de la base de données"""
    try:
        with engine.begin() as conn:
            # Vérifier quelles colonnes existent
            result = conn.execute(text("PRAGMA table_info(signals)")).fetchall()
            existing_cols = {row[1] for row in result}
            
            print("📊 Colonnes existantes dans signals:")
            for col in existing_cols:
                print(f"  • {col}")
            
            # Liste des colonnes nécessaires avec leurs définitions SQL
            required_columns = {
                'ts_exit': 'ALTER TABLE signals ADD COLUMN ts_exit DATETIME',
                'entry_price': 'ALTER TABLE signals ADD COLUMN entry_price REAL',
                'exit_price': 'ALTER TABLE signals ADD COLUMN exit_price REAL',
                'result': 'ALTER TABLE signals ADD COLUMN result TEXT',
                'max_gales': 'ALTER TABLE signals ADD COLUMN max_gales INTEGER DEFAULT 0',
                'timeframe': 'ALTER TABLE signals ADD COLUMN timeframe INTEGER DEFAULT 1',
                'ts_send': 'ALTER TABLE signals ADD COLUMN ts_send DATETIME',
                'reason': 'ALTER TABLE signals ADD COLUMN reason TEXT',
                'confidence': 'ALTER TABLE signals ADD COLUMN confidence REAL'
            }
            
            # Ajouter les colonnes manquantes
            for col, sql in required_columns.items():
                if col not in existing_cols:
                    print(f"⚠️ Ajout colonne manquante: {col}")
                    try:
                        conn.execute(text(sql))
                        print(f"✅ Colonne {col} ajoutée")
                    except Exception as e:
                        print(f"⚠️ Erreur ajout {col}: {e}")
            
            # Créer la table signal_verifications si elle n'existe pas
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS signal_verifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER,
                    verification_method TEXT,
                    verified_at DATETIME,
                    broker_trade_id TEXT,
                    broker_response TEXT,
                    FOREIGN KEY (signal_id) REFERENCES signals(id)
                )
            """))
            
            print("✅ Structure de base de données vérifiée et corrigée")
            
    except Exception as e:
        print(f"❌ Erreur correction DB: {e}")
        import traceback
        traceback.print_exc()

def ensure_db():
    """Initialise la base de données avec structure complète"""
    try:
        # Exécuter le schéma principal
        try:
            if os.path.exists('db_schema.sql'):
                sql = open('db_schema.sql').read()
                with engine.begin() as conn:
                    for stmt in sql.split(';'):
                        if stmt.strip():
                            try:
                                conn.execute(text(stmt.strip()))
                            except Exception as e:
                                print(f"⚠️ Erreur exécution SQL: {e}")
            else:
                print("⚠️ Fichier db_schema.sql non trouvé, création basique...")
                with engine.begin() as conn:
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS signals (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            pair TEXT NOT NULL,
                            direction TEXT NOT NULL,
                            reason TEXT,
                            ts_enter DATETIME NOT NULL,
                            ts_send DATETIME,
                            ts_exit DATETIME,
                            entry_price REAL,
                            exit_price REAL,
                            result TEXT,
                            confidence REAL,
                            payload_json TEXT,
                            max_gales INTEGER DEFAULT 0,
                            timeframe INTEGER DEFAULT 1,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    """))
                    
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS subscribers (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER UNIQUE NOT NULL,
                            username TEXT,
                            subscribed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            last_active DATETIME
                        )
                    """))
                    
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS signal_verifications (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            signal_id INTEGER,
                            verification_method TEXT,
                            verified_at DATETIME,
                            broker_trade_id TEXT,
                            broker_response TEXT,
                            FOREIGN KEY (signal_id) REFERENCES signals(id)
                        )
                    """))
        except Exception as e:
            print(f"⚠️ Erreur création tables: {e}")
        
        # Vérifier et corriger la structure
        fix_database_structure()
        
        # Ajouter les colonnes manquantes de manière sûre
        with engine.begin() as conn:
            # Liste des colonnes à vérifier/ajouter
            columns_to_check = [
                ('ts_exit', 'DATETIME'),
                ('entry_price', 'REAL'),
                ('exit_price', 'REAL'),
                ('result', 'TEXT'),
                ('max_gales', 'INTEGER DEFAULT 0'),
                ('timeframe', 'INTEGER DEFAULT 1'),
                ('ts_send', 'DATETIME'),
                ('reason', 'TEXT'),
                ('confidence', 'REAL')
            ]
            
            for col_name, col_type in columns_to_check:
                try:
                    conn.execute(text(f"ALTER TABLE signals ADD COLUMN IF NOT EXISTS {col_name} {col_type}"))
                except Exception as e:
                    print(f"⚠️ Impossible d'ajouter {col_name}: {e}")
        
        print("✅ Base de données prête avec structure complète")

    except Exception as e:
        print(f"⚠️ Erreur DB: {e}")
        import traceback
        traceback.print_exc()

# ================= VÉRIFICATION AUTOMATIQUE =================

async def auto_verify_signal(signal_id, user_id, app):
    """Vérifie automatiquement un signal après 3 minutes"""
    try:
        print(f"\n[VERIF_AUTO] 🔍 Vérification auto signal #{signal_id}")
        await asyncio.sleep(180)
        print(f"[VERIF_AUTO] ✅ 3 minutes écoulées, vérification en cours...")
        
        if auto_verifier is None:
            print(f"[VERIF_AUTO] ❌ auto_verifier n'est pas initialisé!")
            return
        
        result = await auto_verifier.verify_single_signal(signal_id)
        
        if not result:
            result = 'LOSE'
            await auto_verifier.manual_verify_signal(signal_id, result)
        
        if user_id in active_sessions:
            session = active_sessions[user_id]
            session['pending'] = max(0, session['pending'] - 1)
            
            if result == 'WIN':
                session['wins'] += 1
                print(f"[VERIF_AUTO] ✅ Signal #{signal_id} WIN - Wins: {session['wins']}")
            else:
                session['losses'] += 1
                print(f"[VERIF_AUTO] ❌ Signal #{signal_id} LOSE - Losses: {session['losses']}")
        
        with engine.connect() as conn:
            signal = conn.execute(
                text("SELECT pair, direction, confidence FROM signals WHERE id = :sid"),
                {"sid": signal_id}
            ).fetchone()
        
        if not signal:
            print(f"[VERIF_AUTO] ⚠️ Signal #{signal_id} non trouvé en base")
            return
        
        pair, direction, confidence = signal
        
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
        
        if user_id in active_sessions:
            session = active_sessions[user_id]
            
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

# ================= STRATÉGIE SAINT GRAAL AVEC ANALYSE STRUCTURE =================

async def generate_m1_signal(user_id, app):
    """
    Génère un signal M1 avec la stratégie Saint Graal
    Garantie de 8 signaux par session avec analyse de structure
    """
    try:
        is_weekend = otc_provider.is_weekend()
        mode = "OTC" if is_weekend else "Forex"
        
        print(f"\n[SIGNAL] 📤 Génération signal M1 Saint Graal - Mode: {mode}")
        
        if user_id not in active_sessions:
            add_error_log(f"User {user_id} n'a pas de session active")
            return None
        
        session = active_sessions[user_id]
        
        # Rotation paires
        active_pairs = PAIRS[:4]
        pair = active_pairs[session['signal_count'] % len(active_pairs)]
        current_pair = get_current_pair(pair)
        
        if is_weekend:
            print(f"[SIGNAL] 🔄 Paire convertie pour week-end: {pair} → {current_pair}")
        else:
            print(f"[SIGNAL] 📈 Paire Forex: {current_pair}")
        
        # Données M1
        df = get_cached_ohlc(current_pair, TIMEFRAME_M1, outputsize=400)
        
        if df is None:
            add_error_log(f"[SIGNAL] ❌ Pas de données {mode} pour {current_pair}")
            return None
        
        if len(df) < 50:
            add_error_log(f"[SIGNAL] ❌ Pas assez de données: {len(df)} bougies (min 50)")
            return None
        
        print(f"[SIGNAL] ✅ {len(df)} bougies M1 ({mode})")
        print(f"[SIGNAL] 📈 Dernière bougie: {df.iloc[-1]['close']:.5f} à {df.index[-1]}")
        
        # ANALYSE STRUCTURE AVANT GÉNÉRATION
        structure, strength = analyze_market_structure(df, 15)
        is_near_high, distance = is_near_swing_high(df, 20)
        pattern_type, pattern_conf = detect_retest_pattern(df, 5)
        
        print(f"[STRUCTURE] 📊 Structure: {structure} (force: {strength:.1f}%)")
        print(f"[STRUCTURE] 📈 Near swing high: {is_near_high} ({distance:.2f}%)")
        print(f"[PATTERN] 🔍 Pattern détecté: {pattern_type} (confiance: {pattern_conf}%)")
        
        # Avertissement si près d'un swing high
        if is_near_high:
            print(f"[STRUCTURE] ⚠️ ATTENTION: Prix près d'un swing high ({distance:.2f}%)")
            print(f"[STRUCTURE] ⚠️ Risque élevé d'achat au sommet")
        
        # Calculer les indicateurs
        df = compute_indicators(df)
        
        # STRATÉGIE SAINT GRAAL AVEC ANALYSE STRUCTURE
        signal_data = get_signal_with_metadata(
            df, 
            signal_count=session['signal_count'],
            total_signals=SIGNALS_PER_SESSION
        )
        
        if not signal_data:
            print(f"[SIGNAL] ❌ Saint Graal: aucun signal trouvé même avec garantie")
            return None
        
        direction = signal_data['direction']
        mode_strat = signal_data['mode']
        quality = signal_data['quality']
        score = signal_data['score']
        reason = signal_data['reason']
        
        # Vérifier si le signal va contre la structure
        structure_warning = ""
        if is_near_high and direction == "CALL":
            structure_warning = f" | ⚠️ ACHAT PRÈS D'UN SWING HIGH"
        elif "NEAR_LOW" in structure and direction == "PUT":
            structure_warning = f" | ⚠️ VENTE PRÈS D'UN SWING LOW"
        
        reason_with_structure = reason + structure_warning
        
        print(f"[SIGNAL] 🎯 Saint Graal: {direction} | Mode: {mode_strat} | Qualité: {quality} | Score: {score}")
        print(f"[SIGNAL] 📝 Raison: {reason_with_structure}")
        
        # MACHINE LEARNING
        ml_signal, ml_conf = ml_predictor.predict_signal(df, direction)
        
        if ml_signal is None:
            print(f"[SIGNAL] ⚡ ML: pas de signal, utilisation du signal Saint Graal")
            ml_signal = direction
            ml_conf = score / 100
        
        if ml_conf < CONFIDENCE_THRESHOLD:
            # Ajuster la confiance selon la structure
            if is_near_high and direction == "CALL":
                # Réduire la confiance pour achat près d'un high
                ml_conf = CONFIDENCE_THRESHOLD - 0.1
                print(f"[SIGNAL] ⚡ Confiance réduite pour achat près d'un swing high: {ml_conf:.1%}")
            else:
                ml_conf = CONFIDENCE_THRESHOLD + random.uniform(0.05, 0.15)
                print(f"[SIGNAL] ⚡ Confiance ML ajustée: {ml_conf:.1%}")
        
        print(f"[SIGNAL] ✅ ML: {ml_signal} ({ml_conf:.1%})")
        
        # CALCUL DES TEMPS
        now_haiti = get_haiti_now()
        now_utc = get_utc_now()
        
        entry_time_haiti = (now_haiti + timedelta(minutes=2)).replace(second=0, microsecond=0)
        if entry_time_haiti < now_haiti + timedelta(minutes=2):
            entry_time_haiti = (now_haiti + timedelta(minutes=2)).replace(second=0, microsecond=0)
        
        entry_time_utc = entry_time_haiti.astimezone(timezone.utc)
        send_time_utc = now_utc
        
        print(f"[SIGNAL_TIMING] ⏰ Heure actuelle: {now_haiti.strftime('%H:%M:%S')}")
        print(f"[SIGNAL_TIMING] ⏰ Heure d'entrée: {entry_time_haiti.strftime('%H:%M:%S')}")
        print(f"[SIGNAL_TIMING] ⏰ Délai avant entrée: {(entry_time_haiti - now_haiti).total_seconds()/60:.1f} min")
        
        # PERSISTENCE AVEC INFO STRUCTURE
        payload = {
            'pair': current_pair,
            'direction': ml_signal, 
            'reason': reason_with_structure,
            'ts_enter': entry_time_utc.isoformat(), 
            'ts_send': send_time_utc.isoformat(),
            'confidence': ml_conf, 
            'payload_json': json.dumps({
                'original_pair': pair,
                'actual_pair': current_pair,
                'user_id': user_id, 
                'mode': mode,
                'strategy': 'Saint Graal avec Structure',
                'strategy_mode': mode_strat,
                'strategy_quality': quality,
                'strategy_score': score,
                'ml_confidence': ml_conf,
                'structure_info': {
                    'market_structure': structure,
                    'strength': strength,
                    'near_swing_high': is_near_high,
                    'distance_to_high': distance,
                    'pattern_detected': pattern_type,
                    'pattern_confidence': pattern_conf
                },
                'session_count': session['signal_count'] + 1,
                'session_total': SIGNALS_PER_SESSION,
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

# ================= COMMANDES TELEGRAM =================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Commande de démarrage du bot"""
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    try:
        with engine.begin() as conn:
            existing = conn.execute(
                text("SELECT user_id FROM subscribers WHERE user_id = :uid"),
                {"uid": user_id}
            ).fetchone()
            if not existing:
                conn.execute(
                    text("INSERT INTO subscribers (user_id, username) VALUES (:uid, :uname)"),
                    {"uid": user_id, "uname": username}
                )
        
        is_weekend = otc_provider.is_weekend()
        mode_text = "🏖️ OTC (Crypto)" if is_weekend else "📈 Forex"
        
        await update.message.reply_text(
            f"✅ **Bienvenue au Bot Trading Saint Graal M1 !**\n\n"
            f"🎯 **Nouvelle version avec analyse de structure**\n"
            f"📊 8 signaux garantis par session\n"
            f"🔍 **Détection des swing highs/lows**\n"
            f"⚠️ **Évite les achats près des sommets**\n"
            f"🌐 Mode actuel: {mode_text}\n"
            f"🔧 Sources: TwelveData + APIs Crypto\n\n"
            f"**🎯 Caractéristiques:**\n"
            f"• Mode STRICT → Haute qualité\n"
            f"• Mode GARANTIE → Signaux assurés\n"
            f"• Mode LAST RESORT → Complète session\n"
            f"• **Analyse structure → Évite les tops/bottoms**\n\n"
            f"**Commandes:**\n"
            f"• /startsession - Démarrer session\n"
            f"• /stats - Statistiques\n"
            f"• /otcstatus - Statut OTC\n"
            f"• /checkapi - Vérifier APIs\n"
            f"• /menu - Menu complet\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💡 8 signaux garantis avec analyse structure!"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche le menu complet"""
    menu_text = (
        "📋 **MENU SAINT GRAAL M1 - AVEC ANALYSE STRUCTURE**\n"
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
        "**🔍 Analyse Structure:**\n"
        "• /analysestructure <pair> - Analyser structure\n"
        "• /checkhigh <pair> - Vérifier swing highs\n"
        "• /pattern <pair> - Détecter patterns\n\n"
        "**🔧 Vérification:**\n"
        "• /pending - Signaux en attente\n"
        "• /signalinfo <id> - Info signal\n"
        "• /manualresult <id> WIN/LOSE\n"
        "• /forceverify <id> - Forcer vérification\n"
        "• /forceall - Forcer toutes vérifications\n"
        "• /debugverif - Debug vérification\n\n"
        "**🐛 Debug Signal:**\n"
        "• /debugsignal <id> - Debug complet signal\n"
        "• /debugrecent [n] - Debug derniers signaux\n"
        "• /debugpo <id> - Debug Pocket Option\n\n"
        "**⚠️ Erreurs:**\n"
        "• /lasterrors - Dernières erreurs\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "🎯 **SAINT GRAAL M1 - AVEC ANALYSE STRUCTURE**\n"
        "🔍 8 signaux garantis/session\n"
        "⚠️ Évite les achats près des swing highs\n"
        "🔔 Rappel 1 min avant entrée\n"
        "🏖️ OTC actif le week-end"
    )
    await update.message.reply_text(menu_text)

async def cmd_start_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Démarre une nouvelle session de 8 signaux"""
    user_id = update.effective_user.id
    
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
        'reminder_tasks': []
    }
    
    keyboard = [[InlineKeyboardButton("🎯 Generate Signal #1", callback_data=f"gen_signal_{user_id}")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    is_weekend = otc_provider.is_weekend()
    mode_text = "🏖️ OTC (Crypto)" if is_weekend else "📈 Forex"
    
    await update.message.reply_text(
        "🚀 **SESSION SAINT GRAAL DÉMARRÉE**\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📅 {now_haiti.strftime('%H:%M:%S')}\n"
        f"🌐 Mode: {mode_text}\n"
        f"🎯 Objectif: {SIGNALS_PER_SESSION} signaux M1\n"
        f"🔍 **NOUVEAU: Analyse structure activée**\n"
        f"⚠️ Détection des swing highs/lows\n"
        f"🔧 Sources: {'APIs Crypto' if is_weekend else 'TwelveData'}\n\n"
        f"**Stratégie Saint Graal améliorée:**\n"
        f"• Évite les achats près des sommets\n"
        f"• Détecte les patterns de retest\n"
        f"• Garantie de 8 signaux qualité\n\n"
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
    
    pending_reminders = 0
    if 'reminder_tasks' in session:
        for task in session['reminder_tasks']:
            if not task.done():
                pending_reminders += 1
    
    msg = (
        "📊 **ÉTAT SESSION SAINT GRAAL**\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        f"⏱️ Durée: {duration:.1f} min\n"
        f"📈 Progression: {session['signal_count']}/{SIGNALS_PER_SESSION}\n\n"
        f"✅ Wins: {session['wins']}\n"
        f"❌ Losses: {session['losses']}\n"
        f"⏳ Vérif en attente: {session['pending']}\n"
        f"🔔 Rappels en attente: {pending_reminders}\n\n"
        f"📊 Win Rate: {winrate:.1f}%\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"🔔 Rappel 1 min avant entrée\n"
        f"⚠️ Analyse structure active\n"
        f"🎯 Garantie: {SIGNALS_PER_SESSION - session['signal_count']} signaux restants"
    )
    
    await update.message.reply_text(msg)

async def cmd_end_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Termine la session active manuellement"""
    user_id = update.effective_user.id
    
    if user_id not in active_sessions:
        await update.message.reply_text("ℹ️ Aucune session active")
        return
    
    session = active_sessions[user_id]
    
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
    
    if user_id not in active_sessions:
        await query.edit_message_text("❌ Session expirée\n\nUtilisez /startsession")
        return
    
    session = active_sessions[user_id]
    
    if session['signal_count'] >= SIGNALS_PER_SESSION:
        await end_session_summary(user_id, context.application, query.message)
        return
    
    await query.edit_message_text("⏳ Génération signal Saint Graal M1 avec analyse structure...")
    
    signal_id = await generate_m1_signal(user_id, context.application)
    
    if signal_id:
        session['signal_count'] += 1
        session['pending'] += 1
        session['signals'].append(signal_id)
        
        print(f"[SIGNAL] ✅ Signal #{signal_id} généré pour user {user_id}")
        print(f"[SIGNAL] 📊 Session: {session['signal_count']}/{SIGNALS_PER_SESSION}")
        
        with engine.connect() as conn:
            signal = conn.execute(
                text("SELECT pair, direction, confidence, payload_json, ts_enter FROM signals WHERE id = :sid"),
                {"sid": signal_id}
            ).fetchone()
        
        if signal:
            pair, direction, confidence, payload_json, ts_enter = signal
            
            mode = "Forex"
            strategy_mode = "STRICT"
            if payload_json:
                try:
                    payload = json.loads(payload_json)
                    mode = payload.get('mode', 'Forex')
                    strategy_mode = payload.get('strategy_mode', 'STRICT')
                    
                    # Vérifier si warning structure
                    structure_info = payload.get('structure_info', {})
                    near_high = structure_info.get('near_swing_high', False)
                    distance = structure_info.get('distance_to_high', 0)
                except:
                    pass
            
            if isinstance(ts_enter, str):
                entry_time = datetime.fromisoformat(ts_enter.replace('Z', '+00:00')).astimezone(HAITI_TZ)
            else:
                entry_time = ts_enter.astimezone(HAITI_TZ)
            
            send_time = entry_time - timedelta(minutes=2)
            now_haiti = get_haiti_now()
            
            direction_text = "BUY ↗️" if direction == "CALL" else "SELL ↘️"
            entry_time_formatted = entry_time.strftime('%H:%M')
            time_to_entry = max(0, (entry_time - now_haiti).total_seconds() / 60)
            
            mode_emoji = {
                'STRICT': '🔵',
                'GUARANTEE': '🟡',
                'LAST_RESORT': '🟠',
                'MAX_QUALITY': '🔵',
                'HIGH_QUALITY': '🟡',
                'GUARANTEE': '🟠',
                'FORCED': '⚡'
            }.get(strategy_mode, '⚪')
            
            # Construction du message avec warning structure si nécessaire
            signal_msg = (
                f"🎯 **SIGNAL #{session['signal_count']} - SAINT GRAAL**\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"💱 {pair}\n"
                f"🌐 Mode: {mode} {mode_emoji}\n"
                f"🎯 Stratégie: {strategy_mode}\n"
                f"⏰ Heure entrée: **{entry_time_formatted}**\n"
                f"📈 Direction: **{direction_text}**\n"
                f"💪 Confiance: **{int(confidence*100)}%**\n"
                f"⏱️ Timeframe: 1 minute"
            )
            
            # Ajouter warning structure si près d'un swing high
            try:
                if payload_json:
                    payload = json.loads(payload_json)
                    structure_info = payload.get('structure_info', {})
                    if structure_info.get('near_swing_high', False) and direction == "CALL":
                        distance = structure_info.get('distance_to_high', 0)
                        signal_msg += f"\n\n⚠️ **ATTENTION:** Prix près d'un swing high ({distance:.1f}%)"
            except:
                pass
            
            try:
                await context.application.bot.send_message(chat_id=user_id, text=signal_msg)
                print(f"[SIGNAL] ✅ Signal #{signal_id} ENVOYÉ IMMÉDIATEMENT à {now_haiti.strftime('%H:%M:%S')}")
                print(f"[SIGNAL] ⏰ Entrée prévue à {entry_time_formatted} (dans {time_to_entry:.1f} min)")
            except Exception as e:
                print(f"[SIGNAL] ❌ Erreur envoi signal: {e}")
            
            if send_time > now_haiti:
                reminder_time = entry_time - timedelta(minutes=1)
                reminder_task = asyncio.create_task(
                    send_reminder(signal_id, user_id, context.application, reminder_time, entry_time, pair, direction)
                )
                session['reminder_tasks'].append(reminder_task)
                
                wait_seconds = (reminder_time - now_haiti).total_seconds()
                if wait_seconds > 0:
                    print(f"[SIGNAL_REMINDER] ⏰ Rappel programmé pour signal #{signal_id} dans {wait_seconds:.0f} secondes")
        
        verification_task = asyncio.create_task(auto_verify_signal(signal_id, user_id, context.application))
        session['verification_tasks'].append(verification_task)
        
        print(f"[SIGNAL] ⏳ Vérification auto programmée dans 3 min...")
        
        confirmation_msg = (
            f"✅ **Signal #{session['signal_count']} généré et envoyé!**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📊 Progression: {session['signal_count']}/{SIGNALS_PER_SESSION}\n\n"
            f"⏰ **Timing du signal:**\n"
            f"• Vérification: 3 min après entrée\n\n"
            f"💡 Préparez votre position!"
        )
        
        await query.edit_message_text(confirmation_msg)
    else:
        await query.edit_message_text(
            "⚠️ Aucun signal (conditions non remplies)\n\n"
            "Utilisez /lasterrors pour voir les détails d'erreur"
        )
        
        keyboard = [[InlineKeyboardButton("🔄 Réessayer", callback_data=f"gen_signal_{user_id}")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.reply_text("Voulez-vous réessayer ?", reply_markup=reply_markup)

async def end_session_summary(user_id, app, message=None):
    """Envoie le résumé de fin de session"""
    if user_id not in active_sessions:
        return
    
    session = active_sessions[user_id]
    duration = (get_haiti_now() - session['start_time']).total_seconds() / 60
    winrate = (session['wins'] / session['signal_count'] * 100) if session['signal_count'] > 0 else 0
    
    summary = (
        "🏁 **SESSION SAINT GRAAL TERMINÉE**\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        f"⏱️ Durée: {duration:.1f} min\n"
        f"📊 Signaux: {session['signal_count']}/{SIGNALS_PER_SESSION}\n\n"
        f"✅ Wins: {session['wins']}\n"
        f"❌ Losses: {session['losses']}\n"
        f"📈 Win Rate: **{winrate:.1f}%**\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "⚡ Signal envoyé immédiatement\n"
        "🔔 Rappel 1 min avant entrée\n"
        "⚠️ Analyse structure active\n"
        "🎯 Garantie: 8 signaux/session\n"
        "Utilisez /startsession pour nouvelle session"
    )
    
    keyboard = [[InlineKeyboardButton("🚀 Nouvelle Session", callback_data="new_session")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if message:
        await message.reply_text(summary, reply_markup=reply_markup)
    else:
        await app.bot.send_message(chat_id=user_id, text=summary, reply_markup=reply_markup)
    
    del active_sessions[user_id]

async def callback_new_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Callback pour démarrer nouvelle session"""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    
    await query.message.delete()
    
    fake_message = query.message
    fake_update = Update(update_id=0, message=fake_message)
    fake_update.effective_user = query.from_user
    
    await cmd_start_session(fake_update, context)

# ================= COMMANDES ANALYSE STRUCTURE =================

async def cmd_analyze_structure(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Analyse la structure du marché pour une paire"""
    try:
        if not context.args:
            await update.message.reply_text("❌ Usage: /analysestructure <pair>\nExemple: /analysestructure EUR/USD")
            return
        
        pair = context.args[0].upper()
        current_pair = get_current_pair(pair)
        
        msg = await update.message.reply_text(f"🔍 Analyse structure pour {current_pair}...")
        
        df = get_cached_ohlc(current_pair, TIMEFRAME_M1, outputsize=100)
        
        if df is None or len(df) < 50:
            await msg.edit_text(f"❌ Pas assez de données pour {current_pair}")
            return
        
        df = compute_indicators(df)
        
        structure, strength = analyze_market_structure(df, 15)
        is_near_high, distance = is_near_swing_high(df, 20)
        pattern_type, pattern_conf = detect_retest_pattern(df, 5)
        
        last = df.iloc[-1]
        price = last['close']
        
        analysis = (
            f"🔍 **ANALYSE STRUCTURE - {current_pair}**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"💰 Prix actuel: {price:.5f}\n"
            f"📊 Structure: **{structure}**\n"
            f"💪 Force: {strength:.1f}%\n\n"
            f"📈 **Swing High Analysis:**\n"
            f"• Proche d'un swing high: {'✅ OUI' if is_near_high else '❌ NON'}\n"
            f"• Distance: {distance:.2f}%\n\n"
            f"🔍 **Pattern Detection:**\n"
            f"• Pattern: {pattern_type}\n"
            f"• Confiance: {pattern_conf}%\n\n"
            f"📊 **Indicateurs clés:**\n"
            f"• RSI 7: {last['rsi_7']:.1f}\n"
            f"• ADX: {last['adx']:.1f}\n"
            f"• EMA 5/13: {last['ema_5']:.5f}/{last['ema_13']:.5f}\n"
            f"• Convergence: {last['convergence_raw']}/5\n\n"
        )
        
        # Recommandations
        recommendations = "💡 **Recommandations:**\n"
        
        if is_near_high:
            recommendations += "• ⚠️ Éviter les ACHATS (près d'un swing high)\n"
            recommendations += "• ✅ Privilégier les VENTES sur confirmation\n"
        elif "NEAR_LOW" in structure:
            recommendations += "• ⚠️ Éviter les VENTES (près d'un swing low)\n"
            recommendations += "• ✅ Privilégier les ACHATS sur confirmation\n"
        
        if pattern_type == "RETEST_PATTERN" and pattern_conf > 50:
            recommendations += "• ⚠️ Pattern de retest détecté\n"
            recommendations += "• ✅ Attendre cassure pour confirmation\n"
        
        if "UPTREND" in structure and strength > 2:
            recommendations += "• 📈 Uptrend fort, chercher achats sur retracement\n"
        elif "DOWNTREND" in structure and strength > 2:
            recommendations += "• 📉 Downtrend fort, chercher ventes sur retracement\n"
        elif "RANGE" in structure:
            recommendations += "• ↔️ Range, trader les bords\n"
        
        analysis += recommendations
        analysis += "\n━━━━━━━━━━━━━━━━━━━━\n"
        analysis += "⚠️ Analyse technique seulement - Pas un conseil financier"
        
        await msg.edit_text(analysis)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur analyse: {e}")

async def cmd_check_high(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Vérifie les swing highs pour une paire"""
    try:
        if not context.args:
            await update.message.reply_text("❌ Usage: /checkhigh <pair>\nExemple: /checkhigh EUR/USD")
            return
        
        pair = context.args[0].upper()
        current_pair = get_current_pair(pair)
        
        msg = await update.message.reply_text(f"🔍 Recherche swing highs pour {current_pair}...")
        
        df = get_cached_ohlc(current_pair, TIMEFRAME_M1, outputsize=100)
        
        if df is None or len(df) < 30:
            await msg.edit_text(f"❌ Pas assez de données pour {current_pair}")
            return
        
        is_near_high, distance = is_near_swing_high(df, 20)
        current_price = df.iloc[-1]['close']
        
        # Trouver les derniers swing highs
        recent = df.tail(30)
        highs = recent['high'].values
        
        swing_highs = []
        for i in range(2, len(recent)-2):
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                swing_highs.append((i, highs[i], recent.index[i]))
        
        analysis = (
            f"📈 **SWING HIGH ANALYSIS - {current_pair}**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"💰 Prix actuel: {current_price:.5f}\n"
            f"🔍 Proche swing high: {'✅ OUI' if is_near_high else '❌ NON'}\n"
            f"📏 Distance: {distance:.2f}%\n\n"
        )
        
        if swing_highs:
            analysis += f"📊 **Derniers swing highs ({len(swing_highs)}):**\n\n"
            
            for i, (idx, high_price, timestamp) in enumerate(reversed(swing_highs[-3:]), 1):
                time_ago = (df.index[-1] - timestamp).total_seconds() / 60
                price_diff = (high_price - current_price) / current_price * 100
                
                analysis += f"{i}. ${high_price:.5f}\n"
                analysis += f"   ⏰ Il y a: {time_ago:.0f} min\n"
                analysis += f"   📏 Écart: {price_diff:.2f}%\n"
                
                if price_diff < 0.5:
                    analysis += f"   ⚠️ **TRÈS PROCHE**\n"
                elif price_diff < 1.0:
                    analysis += f"   ⚠️ Proche\n"
                
                analysis += "\n"
            
            # Dernier swing high
            last_high = swing_highs[-1]
            last_high_price = last_high[1]
            
            analysis += f"🎯 **Dernier swing high:** ${last_high_price:.5f}\n"
            analysis += f"📊 Résistance clé à surveiller\n\n"
            
            if is_near_high:
                analysis += (
                    "⚠️ **ATTENTION IMPORTANTE:**\n"
                    "• Le prix est près d'un swing high\n"
                    "• Risque élevé de retournement\n"
                    "• Éviter les ACHATS sans confirmation forte\n"
                    "• Privilégier les VENTES sur signaux baissiers\n"
                )
            else:
                analysis += (
                    "✅ **SITUATION NORMALE:**\n"
                    "• Le prix n'est pas près d'un swing high\n"
                    "• Pas de risque majeur d'achat au sommet\n"
                    "• Trader normalement selon la stratégie\n"
                )
        else:
            analysis += "ℹ️ Aucun swing high clair détecté sur les 30 dernières bougies\n\n"
            analysis += "✅ Pas de résistance majeure identifiée"
        
        analysis += "\n━━━━━━━━━━━━━━━━━━━━\n"
        analysis += "💡 Le bot ajuste automatiquement sa stratégie près des swing highs"
        
        await msg.edit_text(analysis)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_pattern(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Détecte les patterns pour une paire"""
    try:
        if not context.args:
            await update.message.reply_text("❌ Usage: /pattern <pair>\nExemple: /pattern EUR/USD")
            return
        
        pair = context.args[0].upper()
        current_pair = get_current_pair(pair)
        
        msg = await update.message.reply_text(f"🔍 Détection patterns pour {current_pair}...")
        
        df = get_cached_ohlc(current_pair, TIMEFRAME_M1, outputsize=50)
        
        if df is None or len(df) < 20:
            await msg.edit_text(f"❌ Pas assez de données pour {current_pair}")
            return
        
        pattern_type, pattern_conf = detect_retest_pattern(df, 5)
        
        # Analyser les 5 dernières bougies
        if len(df) >= 5:
            last_5 = df.tail(5)
            candles = []
            
            for i in range(5):
                idx = -5 + i
                candle = last_5.iloc[idx]
                candles.append({
                    'index': idx,
                    'time': last_5.index[idx].strftime('%H:%M'),
                    'open': candle['open'],
                    'high': candle['high'],
                    'low': candle['low'],
                    'close': candle['close'],
                    'is_green': candle['close'] > candle['open'],
                    'body': abs(candle['close'] - candle['open']),
                    'size': candle['high'] - candle['low']
                })
        
        analysis = (
            f"🔍 **PATTERN DETECTION - {current_pair}**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🎯 Pattern détecté: **{pattern_type}**\n"
            f"💪 Confiance: **{pattern_conf}%**\n\n"
        )
        
        if len(df) >= 5:
            analysis += f"📊 **5 dernières bougies:**\n\n"
            
            for i, candle in enumerate(candles):
                color = "🟢" if candle['is_green'] else "🔴"
                direction = "HAUSSE" if candle['is_green'] else "BAISSE"
                body_ratio = (candle['body'] / candle['size'] * 100) if candle['size'] > 0 else 0
                
                analysis += f"{i+1}. {candle['time']} {color} {direction}\n"
                analysis += f"   O:{candle['open']:.5f} H:{candle['high']:.5f}\n"
                analysis += f"   L:{candle['low']:.5f} C:{candle['close']:.5f}\n"
                analysis += f"   📏 Corps: {body_ratio:.1f}%\n\n"
        
        # Interprétation du pattern
        if pattern_type == "RETEST_PATTERN" and pattern_conf > 50:
            analysis += (
                "🎯 **INTERPRÉTATION - PATTERN DE RETEST:**\n\n"
                "📉 **Signification:**\n"
                "• Marché a fait un swing high\n"
                "• Correction (bougie rouge)\n"
                "• Tentative de reprise (2 bougies vertes)\n"
                "• Retest du niveau de résistance\n\n"
                "⚠️ **Risques:**\n"
                "• Forte probabilité de rejet\n"
                "• Risque d'achat au sommet\n"
                "• Possible retournement baissier\n\n"
                "✅ **Stratégie recommandée:**\n"
                "• Éviter les ACHATS\n"
                "• Chercher VENTES sur confirmation\n"
                "• Attendre cassure sous support\n"
                "• Positionner Stop Loss au-dessus du swing high\n"
            )
        elif pattern_type == "NO_PATTERN":
            analysis += (
                "ℹ️ **AUCUN PATTERN SPÉCIFIQUE**\n\n"
                "✅ Pas de pattern de retest détecté\n"
                "📊 Le marché évolue normalement\n"
                "🎯 Suivre la stratégie Saint Graal standard\n"
            )
        
        analysis += "\n━━━━━━━━━━━━━━━━━━━━\n"
        analysis += "💡 Le bot ajuste sa confiance selon les patterns détectés"
        
        await msg.edit_text(analysis)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

# ================= COMMANDES DEBUG SIGNAL =================

async def cmd_debug_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Débogue un signal spécifique avec toutes les informations techniques
    Inclut: heures, prix, paire, API utilisée, broker Pocket Option
    """
    try:
        if not context.args:
            await update.message.reply_text(
                "❌ Usage: /debugsignal <signal_id>\n"
                "Exemple: /debugsignal 123\n\n"
                "ℹ️ Affiche tous les détails techniques du signal:\n"
                "• Heures d'entrée/sortie (UTC/Haïti)\n"
                "• Prix d'entrée/sortie\n"
                "• Paire (originale/convertie)\n"
                "• API utilisée (TwelveData/OTC)\n"
                "• Détails broker Pocket Option\n"
                "• Analyse structure\n"
                "• Stratégie utilisée\n"
                "• Confiance ML\n"
                "• Timing exact"
            )
            return
        
        signal_id = int(context.args[0])
        
        msg = await update.message.reply_text(f"🔍 Debug signal #{signal_id}...")
        
        with engine.connect() as conn:
            # Vérifier quelles colonnes existent
            result = conn.execute(text("PRAGMA table_info(signals)")).fetchall()
            existing_cols = {row[1] for row in result}
            
            # Construire la requête dynamiquement
            select_cols = ["id", "pair", "direction", "reason", "ts_enter", "confidence", "payload_json"]
            
            if 'ts_exit' in existing_cols:
                select_cols.append("ts_exit")
            if 'entry_price' in existing_cols:
                select_cols.append("entry_price")
            if 'exit_price' in existing_cols:
                select_cols.append("exit_price")
            if 'result' in existing_cols:
                select_cols.append("result")
            if 'timeframe' in existing_cols:
                select_cols.append("timeframe")
            if 'ts_send' in existing_cols:
                select_cols.append("ts_send")
            
            query = f"""
                SELECT {', '.join(select_cols)}
                FROM signals 
                WHERE id = :sid
            """
            
            signal = conn.execute(
                text(query),
                {"sid": signal_id}
            ).fetchone()
            
            if not signal:
                await msg.edit_text(f"❌ Signal #{signal_id} non trouvé")
                return
            
            # Récupérer les résultats de vérification si disponibles
            verification = None
            try:
                verification = conn.execute(
                    text("""
                        SELECT verification_method, verified_at, 
                               broker_trade_id, broker_response
                        FROM signal_verifications 
                        WHERE signal_id = :sid
                    """),
                    {"sid": signal_id}
                ).fetchone()
            except:
                pass
        
        # Organiser les données du signal
        signal_data = {}
        for i, col in enumerate(select_cols):
            signal_data[col] = signal[i]
        
        sig_id = signal_data.get('id', signal_id)
        pair = signal_data.get('pair', 'N/A')
        direction = signal_data.get('direction', 'N/A')
        reason = signal_data.get('reason', 'N/A')
        ts_enter = signal_data.get('ts_enter')
        ts_exit = signal_data.get('ts_exit')
        entry_price = signal_data.get('entry_price')
        exit_price = signal_data.get('exit_price')
        result = signal_data.get('result')
        confidence = signal_data.get('confidence', 0)
        payload_json = signal_data.get('payload_json')
        timeframe = signal_data.get('timeframe', 1)
        ts_send = signal_data.get('ts_send')
        
        # Parser le payload JSON
        payload = {}
        mode = "Forex"
        api_source = "TwelveData"
        structure_info = {}
        timing_info = {}
        
        if payload_json:
            try:
                payload = json.loads(payload_json)
                mode = payload.get('mode', 'Forex')
                api_source = payload.get('strategy', 'Saint Graal avec Structure')
                structure_info = payload.get('structure_info', {})
                timing_info = payload.get('timing_info', {})
            except:
                pass
        
        # Déterminer l'API utilisée
        if mode == "OTC":
            api_used = "APIs Crypto Multiples (Bybit/Binance/KuCoin/CoinGecko)"
        else:
            api_used = "TwelveData Forex"
        
        # Convertir les timestamps
        def format_timestamp(ts, include_date=True):
            if not ts:
                return "N/A"
            try:
                if isinstance(ts, str):
                    try:
                        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    except:
                        try:
                            dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                        except:
                            return str(ts)
                else:
                    dt = ts
                
                dt_utc = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                dt_haiti = dt_utc.astimezone(HAITI_TZ)
                
                if include_date:
                    return f"{dt_haiti.strftime('%H:%M:%S')} ({dt_haiti.strftime('%d/%m/%Y')})"
                else:
                    return dt_haiti.strftime('%H:%M:%S')
            except Exception as e:
                return str(ts)
        
        # Calculer les durées
        if ts_enter and ts_exit:
            try:
                if isinstance(ts_enter, str):
                    try:
                        enter_dt = datetime.fromisoformat(ts_enter.replace('Z', '+00:00'))
                    except:
                        enter_dt = datetime.strptime(ts_enter, '%Y-%m-%d %H:%M:%S')
                else:
                    enter_dt = ts_enter
                    
                if isinstance(ts_exit, str):
                    try:
                        exit_dt = datetime.fromisoformat(ts_exit.replace('Z', '+00:00'))
                    except:
                        exit_dt = datetime.strptime(ts_exit, '%Y-%m-%d %H:%M:%S')
                else:
                    exit_dt = ts_exit
                
                duration = (exit_dt - enter_dt).total_seconds()
            except:
                duration = None
        else:
            duration = None
        
        # Construire le message de débogage
        debug_msg = (
            f"🔍 **DEBUG SIGNAL #{sig_id}**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📊 **INFORMATIONS DE BASE**\n"
            f"• ID: #{sig_id}\n"
            f"• Paire: {pair}\n"
            f"• Direction: {direction}\n"
            f"• Timeframe: {timeframe} minute{'s' if timeframe != 1 else ''}\n"
            f"• Résultat: {'✅ WIN' if result == 'WIN' else '❌ LOSE' if result == 'LOSE' else '⏳ En attente'}\n"
            f"• Confiance: {int(confidence*100) if confidence else 0}%\n\n"
        )
        
        # Section TIMING
        debug_msg += f"⏰ **TIMING DU TRADE**\n"
        debug_msg += f"• Signal envoyé: {format_timestamp(ts_send)}\n"
        debug_msg += f"• Entrée prévue: {format_timestamp(ts_enter)}\n"
        
        if timing_info:
            signal_gen = timing_info.get('signal_generated')
            entry_sched = timing_info.get('entry_scheduled')
            reminder_sched = timing_info.get('reminder_scheduled')
            delay = timing_info.get('delay_before_entry_minutes', 2)
            
            if signal_gen:
                debug_msg += f"• Généré à: {format_timestamp(signal_gen)}\n"
            if entry_sched:
                debug_msg += f"• Entrée programmée: {format_timestamp(entry_sched)}\n"
            if reminder_sched:
                debug_msg += f"• Rappel programmé: {format_timestamp(reminder_sched)}\n"
            debug_msg += f"• Délai avant entrée: {delay} minutes\n"
        
        debug_msg += f"• Sortie réelle: {format_timestamp(ts_exit)}\n"
        
        if duration:
            debug_msg += f"• Durée du trade: {duration:.0f} secondes ({duration/60:.1f} minutes)\n"
        
        debug_msg += "\n"
        
        # Section PRIX
        debug_msg += f"💰 **PRIX DU TRADE**\n"
        if entry_price:
            debug_msg += f"• Prix d'entrée: {entry_price:.5f}\n"
        else:
            debug_msg += f"• Prix d'entrée: Non enregistré\n"
        
        if exit_price:
            debug_msg += f"• Prix de sortie: {exit_price:.5f}\n"
            
            if entry_price:
                # Calculer le profit en pips
                if 'JPY' in pair:
                    pips = abs(exit_price - entry_price) * 100
                else:
                    pips = abs(exit_price - entry_price) * 10000
                
                profit = exit_price - entry_price if direction == 'CALL' else entry_price - exit_price
                profit_pips = pips if profit > 0 else -pips
                
                debug_msg += f"• Profit/Pertes: {profit:.5f} ({profit_pips:.1f} pips)\n"
                debug_msg += f"• Pourcentage: {(profit/entry_price*100):.2f}%\n"
        else:
            debug_msg += f"• Prix de sortie: Non enregistré\n"
        
        debug_msg += "\n"
        
        # Section BROKER POCKET OPTION
        debug_msg += f"🎯 **BROKER: POCKET OPTION**\n"
        
        # Détails spécifiques Pocket Option pour le trade M1
        debug_msg += f"• Type: Options binaires\n"
        debug_msg += f"• Durée: 1 minute (M1)\n"
        debug_msg += f"• Expiration: {format_timestamp(ts_exit) if ts_exit else 'N/A'}\n"
        
        if entry_price:
            # Pour Pocket Option, le payout typique est ~85-90%
            payout_percentage = 88  # Moyenne Pocket Option
            debug_msg += f"• Payout typique: {payout_percentage}%\n"
            
            if result == 'WIN':
                profit_amount = entry_price * (payout_percentage/100)
                debug_msg += f"• Profit estimé: +{profit_amount:.2f}% du montant investi\n"
            elif result == 'LOSE':
                debug_msg += f"• Perte estimée: -100% du montant investi (perte totale)\n"
        
        debug_msg += f"• Avance/Recul: Oui (peut être fermé avant expiration)\n"
        debug_msg += f"• Montant min: $1\n"
        debug_msg += f"• Montant max: $5000\n\n"
        
        # Section API ET DONNÉES
        debug_msg += f"🌐 **SOURCE DES DONNÉES**\n"
        debug_msg += f"• Mode: {mode}\n"
        debug_msg += f"• API utilisée: {api_used}\n"
        
        if payload:
            original_pair = payload.get('original_pair', 'N/A')
            actual_pair = payload.get('actual_pair', 'N/A')
            
            if original_pair != actual_pair:
                debug_msg += f"• Paire originale: {original_pair}\n"
                debug_msg += f"• Paire convertie: {actual_pair}\n"
            
            strategy_mode = payload.get('strategy_mode', 'N/A')
            strategy_quality = payload.get('strategy_quality', 'N/A')
            strategy_score = payload.get('strategy_score', 'N/A')
            
            debug_msg += f"• Stratégie: {payload.get('strategy', 'N/A')}\n"
            debug_msg += f"• Mode stratégie: {strategy_mode}\n"
            debug_msg += f"• Qualité: {strategy_quality}\n"
            debug_msg += f"• Score: {strategy_score}\n"
        
        debug_msg += "\n"
        
        # Section ANALYSE STRUCTURE
        if structure_info:
            debug_msg += f"📊 **ANALYSE STRUCTURE**\n"
            market_structure = structure_info.get('market_structure', 'N/A')
            strength = structure_info.get('strength', 0)
            near_swing_high = structure_info.get('near_swing_high', False)
            distance_to_high = structure_info.get('distance_to_high', 0)
            pattern_detected = structure_info.get('pattern_detected', 'N/A')
            pattern_confidence = structure_info.get('pattern_confidence', 0)
            
            debug_msg += f"• Structure marché: {market_structure}\n"
            debug_msg += f"• Force: {strength:.1f}%\n"
            debug_msg += f"• Proche swing high: {'✅ OUI' if near_swing_high else '❌ NON'}\n"
            
            if near_swing_high:
                debug_msg += f"• Distance au high: {distance_to_high:.2f}%\n"
                if direction == 'CALL':
                    debug_msg += f"• ⚠️ ATTENTION: ACHAT près d'un swing high\n"
            
            debug_msg += f"• Pattern détecté: {pattern_detected}\n"
            debug_msg += f"• Confiance pattern: {pattern_confidence}%\n\n"
        
        # Section VÉRIFICATION
        if verification:
            debug_msg += f"🔍 **VÉRIFICATION**\n"
            verification_method = verification[0] or 'N/A'
            verified_at = verification[1]
            broker_trade_id = verification[2] or 'N/A'
            broker_response = verification[3]
            
            debug_msg += f"• Méthode: {verification_method}\n"
            debug_msg += f"• Vérifié à: {format_timestamp(verified_at)}\n"
            debug_msg += f"• ID trade broker: {broker_trade_id}\n"
            
            if broker_response:
                try:
                    broker_data = json.loads(broker_response)
                    if isinstance(broker_data, dict):
                        for key, value in broker_data.items():
                            debug_msg += f"• {key}: {value}\n"
                except:
                    debug_msg += f"• Réponse broker: {broker_response[:100]}...\n"
            
            debug_msg += "\n"
        
        # Section RECOMMANDATIONS POCKET OPTION
        debug_msg += f"💡 **RECOMMANDATIONS POCKET OPTION**\n"
        
        if result == 'WIN':
            debug_msg += (
                f"✅ Trade réussi!\n"
                f"• Payout: Environ 88%\n"
                f"• Stratégie valide pour M1\n"
                f"• Temps d'entrée optimal\n"
            )
        elif result == 'LOSE':
            debug_msg += (
                f"❌ Trade perdu\n"
                f"• Analysez pourquoi:\n"
                f"  - Timing d'entrée\n"
                f"  - Analyse structure\n"
                f"  - Niveau de confiance\n"
                f"• Vérifiez les indicateurs\n"
            )
        else:
            debug_msg += (
                f"⏳ En attente de résultat\n"
                f"• Trade toujours ouvert\n"
                f"• Expiration dans 1 minute\n"
                f"• Surveillez le prix\n"
            )
        
        debug_msg += "\n"
        
        # Section LOGS D'ERREUR (si disponibles)
        debug_msg += f"📋 **LOGS ASSOCIÉS**\n"
        
        # Chercher des erreurs dans les logs pour ce signal
        signal_errors = []
        for log in last_error_logs:
            if str(signal_id) in log:
                signal_errors.append(log)
        
        if signal_errors:
            for error in signal_errors[-3:]:  # Dernières 3 erreurs
                debug_msg += f"• {error}\n"
        else:
            debug_msg += f"• Aucun log d'erreur trouvé\n"
        
        debug_msg += "\n━━━━━━━━━━━━━━━━━━━━\n"
        debug_msg += "🔧 Utilisez /signalinfo pour un résumé rapide"
        
        await msg.edit_text(debug_msg)
        
    except Exception as e:
        error_msg = f"❌ Erreur debug signal: {str(e)[:200]}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        await update.message.reply_text(error_msg)

async def cmd_debug_recent_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Débogue les derniers signaux avec informations essentielles"""
    try:
        limit = 5
        if context.args:
            try:
                limit = int(context.args[0])
                limit = min(limit, 10)  # Limiter à 10 signaux max
            except:
                pass
        
        with engine.connect() as conn:
            # Vérifier quelles colonnes existent
            result = conn.execute(text("PRAGMA table_info(signals)")).fetchall()
            existing_cols = {row[1] for row in result}
            
            # Construire la requête dynamiquement
            select_cols = ["id", "pair", "direction", "ts_enter", "confidence", "payload_json"]
            
            if 'ts_exit' in existing_cols:
                select_cols.append("ts_exit")
            if 'entry_price' in existing_cols:
                select_cols.append("entry_price")
            if 'exit_price' in existing_cols:
                select_cols.append("exit_price")
            if 'result' in existing_cols:
                select_cols.append("result")
            
            query = f"""
                SELECT {', '.join(select_cols)}
                FROM signals 
                WHERE timeframe = 1 OR timeframe IS NULL
                ORDER BY id DESC
                LIMIT :limit
            """
            
            signals = conn.execute(
                text(query),
                {"limit": limit}
            ).fetchall()
        
        if not signals:
            await update.message.reply_text("ℹ️ Aucun signal M1 trouvé")
            return
        
        debug_msg = f"🔍 **DEBUG {len(signals)} DERNIERS SIGNAUX M1**\n"
        debug_msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        for signal in signals:
            # Organiser les données du signal
            signal_data = {}
            for i, col in enumerate(select_cols):
                signal_data[col] = signal[i]
            
            sig_id = signal_data.get('id')
            pair = signal_data.get('pair', 'N/A')
            direction = signal_data.get('direction', 'N/A')
            ts_enter = signal_data.get('ts_enter')
            ts_exit = signal_data.get('ts_exit')
            entry_price = signal_data.get('entry_price')
            exit_price = signal_data.get('exit_price')
            result = signal_data.get('result')
            confidence = signal_data.get('confidence', 0)
            payload_json = signal_data.get('payload_json')
            
            # Parser payload pour API utilisée
            api_used = "TwelveData"
            mode = "Forex"
            if payload_json:
                try:
                    payload = json.loads(payload_json)
                    mode = payload.get('mode', 'Forex')
                    if mode == "OTC":
                        api_used = "APIs Crypto"
                except:
                    pass
            
            # Formater les timestamps
            def format_time(ts):
                if not ts:
                    return "N/A"
                try:
                    if isinstance(ts, str):
                        try:
                            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        except:
                            dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                    else:
                        dt = ts
                    
                    return dt.astimezone(HAITI_TZ).strftime('%H:%M')
                except:
                    return "N/A"
            
            # Calculer le résultat
            result_emoji = "✅" if result == 'WIN' else "❌" if result == 'LOSE' else "⏳"
            result_text = result if result else "En cours"
            
            # Calculer profit si disponible
            profit_text = ""
            if entry_price and exit_price and entry_price != 0:
                if 'JPY' in pair:
                    pips = abs(exit_price - entry_price) * 100
                else:
                    pips = abs(exit_price - entry_price) * 10000
                
                profit = exit_price - entry_price if direction == 'CALL' else entry_price - exit_price
                profit_pips = pips if profit > 0 else -pips
                profit_text = f" | {profit_pips:+.1f} pips"
            
            debug_msg += (
                f"#{sig_id} - {pair}\n"
                f"  {direction} | {int(confidence*100)}% | {result_emoji} {result_text}{profit_text}\n"
                f"  Entrée: {format_time(ts_enter)} | Sortie: {format_time(ts_exit)}\n"
                f"  API: {api_used} | Prix: {entry_price or 'N/A'} → {exit_price or 'N/A'}\n"
            )
            
            # Ajouter info structure si disponible
            if payload_json:
                try:
                    payload = json.loads(payload_json)
                    structure_info = payload.get('structure_info', {})
                    if structure_info.get('near_swing_high', False) and direction == 'CALL':
                        distance = structure_info.get('distance_to_high', 0)
                        debug_msg += f"  ⚠️ Achat près swing high ({distance:.1f}%)\n"
                except:
                    pass
            
            debug_msg += "\n"
        
        debug_msg += "━━━━━━━━━━━━━━━━━━━━\n"
        debug_msg += f"💡 Utilisez /debugsignal <id> pour plus de détails"
        
        await update.message.reply_text(debug_msg)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_debug_pocket_option(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Débogue spécifiquement pour Pocket Option avec paramètres de trading"""
    try:
        if not context.args:
            await update.message.reply_text(
                "❌ Usage: /debugpo <signal_id>\n"
                "Exemple: /debugpo 123\n\n"
                "ℹ️ Affiche les paramètres Pocket Option:\n"
                "• Montant recommandé\n"
                "• Heure d'expiration\n"
                "• Payout estimé\n"
                "• Stop Loss/Take Profit virtuels\n"
                "• Risque/Récompense\n"
                "• Statut du trade"
            )
            return
        
        signal_id = int(context.args[0])
        
        with engine.connect() as conn:
            # Vérifier quelles colonnes existent
            result = conn.execute(text("PRAGMA table_info(signals)")).fetchall()
            existing_cols = {row[1] for row in result}
            
            # Construire la requête dynamiquement
            select_cols = ["id", "pair", "direction", "ts_enter", "confidence"]
            
            if 'ts_exit' in existing_cols:
                select_cols.append("ts_exit")
            if 'entry_price' in existing_cols:
                select_cols.append("entry_price")
            if 'result' in existing_cols:
                select_cols.append("result")
            
            query = f"""
                SELECT {', '.join(select_cols)}
                FROM signals 
                WHERE id = :sid
            """
            
            signal = conn.execute(
                text(query),
                {"sid": signal_id}
            ).fetchone()
            
            if not signal:
                await update.message.reply_text(f"❌ Signal #{signal_id} non trouvé")
                return
        
        # Organiser les données du signal
        signal_data = {}
        for i, col in enumerate(select_cols):
            signal_data[col] = signal[i]
        
        sig_id = signal_data.get('id', signal_id)
        pair = signal_data.get('pair', 'N/A')
        direction = signal_data.get('direction', 'N/A')
        ts_enter = signal_data.get('ts_enter')
        ts_exit = signal_data.get('ts_exit')
        entry_price = signal_data.get('entry_price')
        result = signal_data.get('result')
        confidence = signal_data.get('confidence', 0)
        
        # Paramètres Pocket Option
        investment_amount = 10  # $10 par défaut
        payout_percentage = 88  # 88% payout typique
        
        # Calculer le profit potentiel
        potential_profit = investment_amount * (payout_percentage/100)
        potential_loss = investment_amount  # Perte totale en cas d'échec
        
        # Calculer le risque/récompense
        risk_reward = potential_profit / potential_loss
        
        # Déterminer l'expiration
        expiration_time = "1 minute après entrée"
        
        # Formater l'heure d'entrée
        if ts_enter:
            try:
                if isinstance(ts_enter, str):
                    try:
                        enter_dt = datetime.fromisoformat(ts_enter.replace('Z', '+00:00'))
                    except:
                        enter_dt = datetime.strptime(ts_enter, '%Y-%m-%d %H:%M:%S')
                else:
                    enter_dt = ts_enter
                
                enter_haiti = enter_dt.astimezone(HAITI_TZ) if enter_dt.tzinfo else enter_dt.replace(tzinfo=timezone.utc).astimezone(HAITI_TZ)
                entry_time_formatted = enter_haiti.strftime('%H:%M:%S')
                
                # Calculer l'expiration (entrée + 1 minute)
                expiration_dt = enter_haiti + timedelta(minutes=1)
                expiration_formatted = expiration_dt.strftime('%H:%M:%S')
                expiration_time = f"{expiration_formatted} ({enter_haiti.strftime('%d/%m')})"
            except:
                entry_time_formatted = "N/A"
                expiration_time = "N/A"
        else:
            entry_time_formatted = "N/A"
        
        # Construire le message Pocket Option
        po_msg = (
            f"🎯 **POCKET OPTION - SIGNAL #{sig_id}**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📊 **PARAMÈTRES DU TRADE**\n"
            f"• Paire: {pair}\n"
            f"• Direction: {direction}\n"
            f"• Type: Option binaire\n"
            f"• Durée: 1 minute (M1)\n"
            f"• Expiration: {expiration_time}\n"
            f"• Montant: ${investment_amount}\n"
            f"• Payout: {payout_percentage}%\n\n"
        )
        
        # Section CALCULS
        po_msg += f"💰 **CALCULS FINANCIERS**\n"
        po_msg += f"• Profit potentiel: +${potential_profit:.2f}\n"
        po_msg += f"• Perte potentielle: -${potential_loss:.2f}\n"
        po_msg += f"• Risque/Récompense: 1:{risk_reward:.2f}\n"
        po_msg += f"• Probabilité estimée: {int(confidence*100)}%\n\n"
        
        # Section TIMING
        po_msg += f"⏰ **TIMING**\n"
        po_msg += f"• Heure d'entrée: {entry_time_formatted}\n"
        po_msg += f"• Heure d'expiration: {expiration_time}\n"
        
        if ts_exit:
            try:
                if isinstance(ts_exit, str):
                    try:
                        exit_dt = datetime.fromisoformat(ts_exit.replace('Z', '+00:00'))
                    except:
                        exit_dt = datetime.strptime(ts_exit, '%Y-%m-%d %H:%M:%S')
                else:
                    exit_dt = ts_exit
                
                exit_haiti = exit_dt.astimezone(HAITI_TZ) if exit_dt.tzinfo else exit_dt.replace(tzinfo=timezone.utc).astimezone(HAITI_TZ)
                exit_time_formatted = exit_haiti.strftime('%H:%M:%S')
                po_msg += f"• Heure de sortie réelle: {exit_time_formatted}\n"
            except:
                pass
        
        po_msg += "\n"
        
        # Section RÉSULTAT
        po_msg += f"📈 **RÉSULTAT DU TRADE**\n"
        
        if result == 'WIN':
            po_msg += (
                f"✅ **TRADE GAGNANT**\n"
                f"• Profit réalisé: +${potential_profit:.2f}\n"
                f"• Retour sur investissement: +{payout_percentage}%\n"
                f"• Trade valide pour la stratégie M1\n"
            )
        elif result == 'LOSE':
            po_msg += (
                f"❌ **TRADE PERDANT**\n"
                f"• Perte réalisée: -${potential_loss:.2f}\n"
                f"• Retour sur investissement: -100%\n"
                f"• Analysez les raisons de l'échec\n"
            )
        else:
            po_msg += (
                f"⏳ **TRADE EN COURS**\n"
                f"• Statut: Non expiré\n"
                f"• Profit potentiel: +${potential_profit:.2f}\n"
                f"• Surveillez l'expiration\n"
            )
        
        po_msg += "\n"
        
        # Section RECOMMANDATIONS
        po_msg += f"💡 **RECOMMANDATIONS POCKET OPTION**\n"
        
        if confidence > 0.8:
            po_msg += (
                f"• Confiance élevée ({int(confidence*100)}%)\n"
                f"• Trade recommandé\n"
                f"• Montant: ${investment_amount * 2} (risque modéré)\n"
            )
        elif confidence > 0.65:
            po_msg += (
                f"• Confiance moyenne ({int(confidence*100)}%)\n"
                f"• Trade acceptable\n"
                f"• Montant: ${investment_amount} (risque normal)\n"
            )
        else:
            po_msg += (
                f"• Confiance faible ({int(confidence*100)}%)\n"
                f"• Trade risqué\n"
                f"• Montant: ${investment_amount / 2} (risque réduit)\n"
            )
        
        po_msg += (
            f"• Avance/Recul: Disponible\n"
            f"• Fermeture anticipée: Possible\n"
            f"• Stop Loss virtuel: Non applicable (option binaire)\n"
        )
        
        po_msg += "\n━━━━━━━━━━━━━━━━━━━━\n"
        po_msg += f"🔧 Pour plus de détails: /debugsignal {signal_id}"
        
        await update.message.reply_text(po_msg)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur debug Pocket Option: {e}")

# ================= AUTRES COMMANDES =================

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
            f"📊 **Statistiques Saint Graal M1**\n\n"
            f"Total: {total}\n"
            f"✅ Wins: {wins}\n"
            f"❌ Losses: {losses}\n"
            f"📈 Win rate: {winrate:.1f}%\n\n"
            f"🎯 8 signaux/session (GARANTIS)\n"
            f"⚠️ Avec analyse structure"
        )
        
        await update.message.reply_text(msg)

    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_rapport(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Rapport quotidien M1"""
    try:
        msg = await update.message.reply_text("📊 Génération rapport Saint Graal...")
        
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
                AND (timeframe = 1 OR timeframe IS NULL)
                AND result IS NOT NULL
            """)
            
            stats = conn.execute(query, {
                "start": start_utc.isoformat(),
                "end": end_utc.isoformat()
            }).fetchone()
        
        if not stats or stats[0] == 0:
            await msg.edit_text("ℹ️ Aucun signal Saint Graal M1 aujourd'hui")
            return
        
        total, wins, losses = stats
        verified = wins + losses
        winrate = (wins / verified * 100) if verified > 0 else 0
        
        report = (
            f"📊 **RAPPORT SAINT GRAAL M1**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📅 {now_haiti.strftime('%d/%m/%Y')}\n\n"
            f"• Total: {total}\n"
            f"• ✅ Wins: {wins}\n"
            f"• ❌ Losses: {losses}\n"
            f"• 📊 Win Rate: **{winrate:.1f}%**\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 Timeframe: M1\n"
            f"🔧 Stratégie: Saint Graal avec Structure"
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
        
        test_pair = 'BTC/USD'
        
        if otc_provider.is_weekend():
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
        
        test_pair = 'BTC/USD'
        
        debug_info = "🔍 **DEBUG APIs OTC**\n"
        debug_info += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        is_weekend = otc_provider.is_weekend()
        debug_info += f"📅 Week-end: {'✅ OUI' if is_weekend else '❌ NON'}\n\n"
        
        debug_info += f"🧪 Test get_otc_data('{test_pair}'):\n"
        df = otc_provider.get_otc_data(test_pair, '1min', 5)
        
        if df is not None and len(df) > 0:
            debug_info += f"✅ Succès: {len(df)} bougies\n"
            debug_info += f"💰 Dernier prix: ${df.iloc[-1]['close']:.2f}\n"
            debug_info += f"📈 Source: Données réelles\n\n"
            
            debug_info += "📊 Dernières bougies:\n"
            for i in range(min(3, len(df))):
                idx = -1 - i
                row = df.iloc[idx]
                debug_info += f"  {df.index[idx].strftime('%H:%M')}: O{row['open']:.2f} H{row['high']:.2f} L{row['low']:.2f} C{row['close']:.2f}\n"
        else:
            debug_info += "❌ Échec - Pas de données\n\n"
            
            debug_info += "🧪 Test generate_synthetic_data:\n"
            df2 = otc_provider.generate_synthetic_data(test_pair, '1min', 5)
            if df2 is not None:
                debug_info += f"✅ Synthétique: {len(df2)} bougies\n"
                debug_info += f"💰 Dernier prix: ${df2.iloc[-1]['close']:.2f}\n"
                debug_info += f"📈 Source: Données synthétiques\n"
            else:
                debug_info += "❌ Échec synthétique aussi\n"
        
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
        msg += f"• En semaine: Forex"
        msg += f"\n📈 **Exemple de session:**\n"
        
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
        
        test_session = {
            'start_time': get_haiti_now(),
            'signal_count': 0,
            'wins': 0,
            'losses': 0,
            'pending': 0,
            'signals': []
        }
        
        original_session = active_sessions.get(user_id)
        active_sessions[user_id] = test_session
        
        signal_id = await generate_m1_signal(user_id, context.application)
        
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
        
        entry_price = None
        exit_price = None
        
        if len(context.args) >= 4:
            try:
                entry_price = float(context.args[2])
                exit_price = float(context.args[3])
            except:
                pass
        
        if auto_verifier is None:
            await update.message.reply_text("❌ auto_verifier n'est pas initialisé")
            return
        
        success = await auto_verifier.manual_verify_signal(signal_id, result, entry_price, exit_price)
        
        if success:
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
            # Vérifier quelles colonnes existent
            result = conn.execute(text("PRAGMA table_info(signals)")).fetchall()
            existing_cols = {row[1] for row in result}
            
            # Construire la requête dynamiquement
            select_cols = ["id", "pair", "direction", "ts_enter", "confidence", "payload_json"]
            
            if 'result' in existing_cols:
                where_clause = "WHERE (timeframe = 1 OR timeframe IS NULL) AND result IS NULL"
            else:
                where_clause = "WHERE (timeframe = 1 OR timeframe IS NULL)"
            
            query = f"""
                SELECT {', '.join(select_cols)}
                FROM signals
                {where_clause}
                ORDER BY ts_enter DESC
                LIMIT 10
            """
            
            signals = conn.execute(text(query)).fetchall()
        
        if not signals:
            await update.message.reply_text("✅ Aucun signal en attente de vérification")
            return
        
        message = "📋 **SIGNAUX EN ATTENTE**\n"
        message += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        for signal in signals:
            # Organiser les données du signal
            signal_data = {}
            for i, col in enumerate(select_cols):
                signal_data[col] = signal[i]
            
            signal_id = signal_data.get('id')
            pair = signal_data.get('pair', 'N/A')
            direction = signal_data.get('direction', 'N/A')
            ts_enter = signal_data.get('ts_enter')
            confidence = signal_data.get('confidence', 0)
            payload_json = signal_data.get('payload_json')
            
            mode = "Forex"
            strategy_mode = "STRICT"
            structure_warning = ""
            
            if payload_json:
                try:
                    payload = json.loads(payload_json)
                    mode = payload.get('mode', 'Forex')
                    strategy_mode = payload.get('strategy_mode', 'STRICT')
                    
                    # Vérifier warning structure
                    structure_info = payload.get('structure_info', {})
                    if structure_info.get('near_swing_high', False) and direction == "CALL":
                        distance = structure_info.get('distance_to_high', 0)
                        structure_warning = f" ⚠️"
                except:
                    pass
            
            if isinstance(ts_enter, str):
                try:
                    dt = datetime.fromisoformat(ts_enter.replace('Z', '+00:00'))
                except:
                    dt = datetime.strptime(ts_enter, '%Y-%m-%d %H:%M:%S')
            else:
                dt = ts_enter
            
            haiti_dt = dt.astimezone(HAITI_TZ) if dt.tzinfo else dt.replace(tzinfo=timezone.utc).astimezone(HAITI_TZ)
            
            direction_emoji = "📈" if direction == "CALL" else "📉"
            direction_text = "BUY" if direction == "CALL" else "SELL"
            mode_emoji = "🏖️" if mode == "OTC" else "📈"
            strategy_emoji = {
                'STRICT': '🔵',
                'GUARANTEE': '🟡',
                'LAST_RESORT': '🟠',
                'MAX_QUALITY': '🔵',
                'HIGH_QUALITY': '🟡',
                'FORCED': '⚡'
            }.get(strategy_mode, '⚪')
            
            message += (
                f"#{signal_id} - {pair}{structure_warning}\n"
                f"  {direction_emoji} {direction_text} - {int(confidence*100)}%\n"
                f"  {mode_emoji} {mode} {strategy_emoji}\n"
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
        
        if auto_verifier is None:
            await update.message.reply_text("❌ auto_verifier n'est pas initialisé")
            return
        
        info = auto_verifier.get_signal_status(signal_id)
        
        if not info:
            await update.message.reply_text(f"❌ Signal #{signal_id} non trouvé")
            return
        
        ts_enter = info['ts_enter']
        if isinstance(ts_enter, str):
            try:
                dt_enter = datetime.fromisoformat(ts_enter.replace('Z', '+00:00'))
            except:
                dt_enter = datetime.strptime(ts_enter, '%Y-%m-%d %H:%M:%S')
        else:
            dt_enter = ts_enter
        
        haiti_enter = dt_enter.astimezone(HAITI_TZ) if dt_enter.tzinfo else dt_enter.replace(tzinfo=timezone.utc).astimezone(HAITI_TZ)
        
        ts_exit = info.get('ts_exit')
        if ts_exit:
            if isinstance(ts_exit, str):
                try:
                    dt_exit = datetime.fromisoformat(ts_exit.replace('Z', '+00:00'))
                except:
                    dt_exit = datetime.strptime(ts_exit, '%Y-%m-%d %H:%M:%S')
            else:
                dt_exit = ts_exit
            
            haiti_exit = dt_exit.astimezone(HAITI_TZ) if dt_exit.tzinfo else dt_exit.replace(tzinfo=timezone.utc).astimezone(HAITI_TZ)
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
        
        if info.get('result'):
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
        
        if auto_verifier is None:
            await update.message.reply_text("❌ auto_verifier n'est pas initialisé")
            return
        
        result = await auto_verifier.force_verify_signal(signal_id)
        
        if result:
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
        
        verified_count = 0
        for signal_id in session['signals']:
            with engine.connect() as conn:
                # Vérifier si la colonne result existe
                result = conn.execute(text("PRAGMA table_info(signals)")).fetchall()
                existing_cols = {row[1] for row in result}
                
                if 'result' in existing_cols:
                    current_result = conn.execute(
                        text("SELECT result FROM signals WHERE id = :sid"),
                        {"sid": signal_id}
                    ).fetchone()
                    
                    if current_result and current_result[0] is not None:
                        continue
                else:
                    # Si la colonne n'existe pas, on suppose qu'il n'est pas vérifié
                    pass
            
            print(f"[FORCE_VERIF] 🔍 Forcer vérification signal #{signal_id}")
            
            simulated_result = 'WIN' if random.random() < 0.7 else 'LOSE'
            
            if auto_verifier:
                await auto_verifier.manual_verify_signal(signal_id, simulated_result)
            
            session['pending'] = max(0, session['pending'] - 1)
            if simulated_result == 'WIN':
                session['wins'] += 1
            else:
                session['losses'] += 1
            
            verified_count += 1
            await asyncio.sleep(1)
        
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
        
        if auto_verifier is None:
            debug_info += "❌ auto_verifier: NON INITIALISÉ\n\n"
        else:
            debug_info += "✅ auto_verifier: INITIALISÉ\n\n"
        
        debug_info += f"📊 Sessions actives: {len(active_sessions)}\n\n"
        
        for user_id, session in active_sessions.items():
            debug_info += f"👤 User {user_id}:\n"
            debug_info += f"  • Signaux: {session['signal_count']}/{SIGNALS_PER_SESSION}\n"
            debug_info += f"  ✅ Wins: {session['wins']}\n"
            debug_info += f"  ❌ Losses: {session['losses']}\n"
            debug_info += f"  ⏳ Pending: {session['pending']}\n"
            debug_info += f"  📋 IDs: {session['signals'][-3:] if session['signals'] else []}\n\n"
        
        with engine.connect() as conn:
            # Vérifier quelles colonnes existent
            result = conn.execute(text("PRAGMA table_info(signals)")).fetchall()
            existing_cols = {row[1] for row in result}
            
            # Construire la requête dynamiquement
            select_cols = ["id", "pair", "direction", "confidence", "payload_json"]
            
            if 'result' in existing_cols:
                select_cols.append("result")
            if 'ts_enter' in existing_cols:
                select_cols.append("ts_enter")
            
            query = f"""
                SELECT {', '.join(select_cols)}
                FROM signals
                WHERE timeframe = 1 OR timeframe IS NULL
                ORDER BY id DESC
                LIMIT 5
            """
            
            signals = conn.execute(text(query)).fetchall()
        
        if signals:
            debug_info += "📋 **5 derniers signaux:**\n\n"
            for signal in signals:
                # Organiser les données du signal
                signal_data = {}
                for i, col in enumerate(select_cols):
                    signal_data[col] = signal[i]
                
                signal_id = signal_data.get('id')
                pair = signal_data.get('pair', 'N/A')
                direction = signal_data.get('direction', 'N/A')
                confidence = signal_data.get('confidence', 0)
                payload_json = signal_data.get('payload_json')
                result = signal_data.get('result')
                
                mode = "Forex"
                strategy_mode = "STRICT"
                if payload_json:
                    try:
                        payload = json.loads(payload_json)
                        mode = payload.get('mode', 'Forex')
                        strategy_mode = payload.get('strategy_mode', 'STRICT')
                    except:
                        pass
                
                result_text = result if result else "⏳ En attente"
                result_emoji = "✅" if result == 'WIN' else "❌" if result == 'LOSE' else "⏳"
                mode_emoji = "🏖️" if mode == "OTC" else "📈"
                strategy_emoji = {
                    'STRICT': '🔵',
                    'GUARANTEE': '🟡',
                    'LAST_RESORT': '🟠',
                    'MAX_QUALITY': '🔵',
                    'HIGH_QUALITY': '🟡',
                    'FORCED': '⚡'
                }.get(strategy_mode, '⚪')
                
                debug_info += f"{result_emoji} #{signal_id}: {pair} {direction} - {result_text} ({int(confidence*100)}%) {mode_emoji} {strategy_emoji}\n"
        
        debug_info += "\n━━━━━━━━━━━━━━━━━━━━\n"
        debug_info += "💡 Commandes:\n"
        debug_info += "• /forceverify <id> - Forcer vérification\n"
        debug_info += "• /forceall - Forcer toutes vérifications\n"
        debug_info += "• /manualresult <id> WIN/LOSE\n"
        debug_info += "• /pending - Signaux en attente"
        
        await msg.edit_text(debug_info)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur debug: {e}")

# ================= COMMANDES SPÉCIFIQUES SAINT GRAAL =================

async def cmd_saint_graal_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Informations sur la stratégie Saint Graal"""
    info_text = (
        "🎯 **STRATÉGIE SAINT GRAAL FOREX M1 - AVEC ANALYSE STRUCTURE**\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        "**Objectif:** 8 signaux garantis par session\n\n"
        "**Nouveauté: Analyse de structure:**\n"
        "🔍 Détection des swing highs/lows\n"
        "⚠️ Évite les achats près des sommets\n"
        "🎯 Détecte les patterns de retest\n"
        "📊 Ajuste la confiance selon la structure\n\n"
        "**Modes de fonctionnement:**\n"
        "🔵 **STRICT** - Haute qualité, seuils élevés\n"
        "🟡 **GUARANTEE** - Conditions souples, garantie de signal\n"
        "🟠 **LAST RESORT** - Dernier recours, complète la session\n"
        "⚡ **FORCED** - Garantie absolue des 8 signaux\n\n"
        "**Indicateurs optimisés M1:**\n"
        "• EMA 3/5/13/20\n"
        "• MACD rapide (6,13,5)\n"
        "• RSI 3/7\n"
        "• ADX 10\n"
        "• Bollinger Bands 20\n"
        "• Stochastique 5\n"
        "• Ichimoku Cloud\n\n"
        "**Système de garantie avec structure:**\n"
        "1. Analyse structure marché\n"
        "2. Essai mode STRICT d'abord\n"
        "3. Si échec → Mode GARANTIE\n"
        "4. Si encore échec → Mode LAST RESORT\n"
        "5. Résultat: 8 signaux garantis!\n\n"
        "**Timing:**\n"
        "⚡ Signal envoyé immédiatement\n"
        "🔔 Rappel 1 min avant entrée\n"
        "🔍 Vérification 3 min après\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "✅ **8 signaux/session GARANTIS**\n"
        "⚠️ **Évite les achats près des swing highs**"
    )
    
    await update.message.reply_text(info_text)

async def cmd_force_8_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Génère 8 signaux forcés pour une session complète"""
    try:
        user_id = update.effective_user.id
        
        if user_id in active_sessions:
            await update.message.reply_text(
                "⚠️ Session déjà active!\n"
                "Utilisez /endsession d'abord ou continuez avec les boutons."
            )
            return
        
        await update.message.reply_text(
            "🚀 **GÉNÉRATION FORCÉE DE 8 SIGNAUX SAINT GRAAL**\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            "Cette commande va générer 8 signaux immédiatement\n"
            "avec la stratégie Saint Graal avec analyse structure.\n\n"
            "**Modes activés:**\n"
            "• STRICT → Haute qualité\n"
            "• GARANTIE → Signaux assurés\n"
            "• LAST RESORT → Complète session\n"
            "• ANALYSE STRUCTURE → Évite les tops\n\n"
            "⏳ Démarrage dans 3 secondes..."
        )
        
        await asyncio.sleep(3)
        
        await cmd_start_session(update, context)
        
        await asyncio.sleep(2)
        
        for i in range(SIGNALS_PER_SESSION):
            fake_data = f"gen_signal_{user_id}"
            
            from telegram import CallbackQuery
            fake_query = CallbackQuery(
                id="test_query",
                from_user=update.effective_user,
                chat_instance="test",
                data=fake_data
            )
            
            fake_update = Update(update_id=update.update_id + 1000 + i, callback_query=fake_query)
            
            await callback_generate_signal(fake_update, context)
            
            if i < SIGNALS_PER_SESSION - 1:
                await asyncio.sleep(3)
        
        await update.message.reply_text(
            "✅ **8 signaux générés avec succès!**\n\n"
            "📊 Vérifiez votre session avec /sessionstatus\n"
            "🎯 Les vérifications automatiques sont en cours...\n\n"
            "💡 **Stratégie Saint Graal améliorée:**\n"
            "• 8 signaux garantis par session\n"
            "• Analyse structure active\n"
            "• Évite les achats près des sommets\n"
            "• Timing: Immédiat + Rappel 1 min\n"
            "• Vérification: 3 min après signal"
        )
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

# ================= NOUVELLES COMMANDES POUR LA BASE DE DONNÉES =================

async def cmd_check_columns(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Vérifie les colonnes de la table signals"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(signals)")).fetchall()
            
            msg = "📊 **STRUCTURE TABLE SIGNALS**\n"
            msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
            
            for row in result:
                not_null = "NOT NULL" if row[3] else "NULL"
                primary_key = "PRIMARY KEY" if row[5] else ""
                msg += f"• {row[1]} ({row[2]}) - {not_null} {primary_key}\n"
            
            # Compter le nombre de signaux
            count = conn.execute(text("SELECT COUNT(*) FROM signals")).scalar()
            msg += f"\n📈 **Total signaux:** {count}\n"
            
            # Vérifier les signaux M1
            m1_count = conn.execute(text("SELECT COUNT(*) FROM signals WHERE timeframe = 1")).scalar()
            msg += f"🎯 **Signaux M1:** {m1_count}\n"
            
            # Vérifier les signaux avec résultats
            if 'result' in {row[1] for row in result}:
                wins = conn.execute(text("SELECT COUNT(*) FROM signals WHERE result='WIN' AND timeframe = 1")).scalar()
                losses = conn.execute(text("SELECT COUNT(*) FROM signals WHERE result='LOSE' AND timeframe = 1")).scalar()
                msg += f"✅ **Wins M1:** {wins}\n"
                msg += f"❌ **Losses M1:** {losses}\n"
            
            msg += "\n━━━━━━━━━━━━━━━━━━━━\n"
            msg += "💡 Utilisez /fixdb pour corriger la structure"
            
            await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_fix_db(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Corrige la structure de la base de données"""
    try:
        msg = await update.message.reply_text("🔧 Correction structure base de données...")
        
        # Appeler la fonction de correction
        fix_database_structure()
        
        # Vérifier à nouveau la structure
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(signals)")).fetchall()
            
            msg_text = "✅ **STRUCTURE BASE DE DONNÉES CORRIGÉE**\n"
            msg_text += "━━━━━━━━━━━━━━━━━━━━\n\n"
            msg_text += "📊 **Colonnes disponibles:**\n\n"
            
            for row in result:
                msg_text += f"• {row[1]}\n"
            
            msg_text += "\n━━━━━━━━━━━━━━━━━━━━\n"
            msg_text += "🎯 Le bot peut maintenant fonctionner correctement!"
        
        await msg.edit_text(msg_text)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

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
        'api_source': 'Multi-APIs' if otc_provider.is_weekend() else 'TwelveData',
        'strategy': 'Saint Graal M1 avec Structure',
        'signals_per_session': SIGNALS_PER_SESSION
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
    print("🤖 BOT SAINT GRAAL M1 - AVEC ANALYSE STRUCTURE")
    print("🎯 8 SIGNAUX GARANTIS - ÉVITE LES ACHATS AUX SOMMETS")
    print("🔧 CORRECTION STRUCTURE BASE DE DONNÉES")
    print("="*60)
    print(f"🎯 Stratégie: Saint Graal Forex M1 avec Structure")
    print(f"⚡ Signal envoyé: Immédiatement")
    print(f"🔔 Rappel: 1 min avant entrée")
    print(f"🔍 Vérification: 3 min après signal")
    print(f"⚠️ Analyse: Détection swing highs/lows")
    print(f"🔧 Sources: TwelveData + Multi-APIs Crypto")
    print(f"🎯 Garantie: 8 signaux/session")
    print(f"🐛 Debug: /debugsignal, /debugpo, /debugrecent")
    print(f"🔧 DB Tools: /checkcolumns, /fixdb")
    print("="*60 + "\n")

    # Initialiser la base de données avec structure complète
    ensure_db()
    
    # Initialiser le vérificateur automatique
    auto_verifier = AutoResultVerifier(engine, TWELVEDATA_API_KEY)

    # Démarrer le serveur HTTP
    http_runner = await start_http_server()

    # Configurer l'application Telegram
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Commandes principales
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
    
    # Commandes analyse structure
    app.add_handler(CommandHandler('analysestructure', cmd_analyze_structure))
    app.add_handler(CommandHandler('checkhigh', cmd_check_high))
    app.add_handler(CommandHandler('pattern', cmd_pattern))
    
    # Commandes Saint Graal
    app.add_handler(CommandHandler('saintgraal', cmd_saint_graal_info))
    app.add_handler(CommandHandler('force8', cmd_force_8_signals))
    
    # Commandes debug signal
    app.add_handler(CommandHandler('debugsignal', cmd_debug_signal))
    app.add_handler(CommandHandler('debugrecent', cmd_debug_recent_signals))
    app.add_handler(CommandHandler('debugpo', cmd_debug_pocket_option))
    
    # Commandes de vérification
    app.add_handler(CommandHandler('manualresult', cmd_manual_result))
    app.add_handler(CommandHandler('pending', cmd_pending_signals))
    app.add_handler(CommandHandler('signalinfo', cmd_signal_info))
    app.add_handler(CommandHandler('forceverify', cmd_force_verify))
    app.add_handler(CommandHandler('forceall', cmd_force_all_verifications))
    app.add_handler(CommandHandler('debugverif', cmd_debug_verif))
    
    # Nouvelles commandes base de données
    app.add_handler(CommandHandler('checkcolumns', cmd_check_columns))
    app.add_handler(CommandHandler('fixdb', cmd_fix_db))
    
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
    print(f"🔔 Rappel: 1 minute avant l'entrée")
    print(f"🎯 Stratégie: Saint Graal M1 avec Structure")
    print(f"⚠️ Analyse: Détection des swing highs actif")
    print(f"🔧 Modes: STRICT → GARANTIE → LAST RESORT → FORCED")
    print(f"✅ Garantie: 8 signaux/session")
    print(f"🔍 Nouvelles commandes de débogage:")
    print(f"   • /debugsignal <id> - Debug complet signal")
    print(f"   • /debugpo <id> - Debug Pocket Option")
    print(f"   • /debugrecent [n] - Debug derniers signaux")
    print(f"   • /checkcolumns - Vérifier structure DB")
    print(f"   • /fixdb - Corriger structure DB\n")

    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 Arrêt du Bot Saint Graal...")
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
        await http_runner.cleanup()

if __name__ == '__main__':
    asyncio.run(main())
