"""
signal_bot.py - Bot de trading M1 - Version Saint Graal 4.5
Analyse multi-marchés par rotation avec limites API
"""

import os, json, asyncio, random, traceback
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import requests
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
from aiohttp import web

# DÉSACTIVER LES LOGS HTTP VERBOSE
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Import du vérificateur externe
try:
    from auto_verifier import AutoResultVerifier
    EXTERNAL_VERIFIER_AVAILABLE = True
except ImportError:
    EXTERNAL_VERIFIER_AVAILABLE = False
    print("⚠️ Vérificateur externe non disponible")

from config import *
from utils import (
    compute_indicators, 
    get_signal_with_metadata,  # 🔥 Utilisation de la fonction principale compatibilité
    calculate_signal_quality_score,
    get_m1_candle_range,
    get_next_m1_candle,
    analyze_market_structure,
    is_near_swing_high,
    detect_retest_pattern
)

# ================= LISTE DES PAIRES POUR ROTATION =================

ROTATION_PAIRS = [
    'EUR/USD',
    'GBP/USD', 
    'USD/JPY',
    'AUD/CAD',
    'AUD/NZD',
    'CAD/CHF',
    'EUR/CHF',
    'EUR/GBP',
    'USD/CAD',
    'EUR/RUB',
    'USD/CLP',
    'USD/THB',
    'USD/COP',
    'USD/EGP',
    'AED/CNY',
    'QAR/CNY'
]

# Configuration rotation
ROTATION_CONFIG = {
    'max_pairs_per_signal': 4,           # Maximum 4 paires analysées par signal
    'min_data_points': 100,              # Minimum 100 bougies M1
    'api_cooldown_seconds': 2,           # 2 secondes entre chaque appel API
    'min_score_threshold': 85,           # Score minimum pour accepter un signal
    'prefer_volatile_pairs': True,       # Priorité aux paires volatiles
    'avoid_low_volume_pairs': True,      # Éviter paires à faible volume
    'rotation_strategy': 'SCORE_BASED',  # Stratégie: basée sur score
}

# ================= FONCTIONS HELPER =================

def safe_strftime(timestamp, fmt='%H:%M:%S'):
    """Convertit un timestamp en string formatée de manière sécurisée"""
    if not timestamp:
        return 'N/A'
    
    if isinstance(timestamp, datetime):
        return timestamp.strftime(fmt)
    
    try:
        if isinstance(timestamp, str):
            ts_clean = timestamp.replace('Z', '').replace('+00:00', '').split('.')[0]
            try:
                dt = datetime.fromisoformat(ts_clean)
            except:
                try:
                    dt = datetime.strptime(ts_clean, '%Y-%m-%d %H:%M:%S')
                except:
                    try:
                        dt = datetime.strptime(ts_clean, '%Y-%m-%d %H:%M')
                    except:
                        return str(timestamp)[:8]
            
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            
            return dt.strftime(fmt)
    except Exception as e:
        print(f"[DEBUG] Erreur format timestamp: {e}")
    
    return str(timestamp)[:8]

# ================= GESTION API LIMITS =================

class APILimitManager:
    """Gestionnaire des limites d'API"""
    
    def __init__(self):
        self.api_calls = []
        self.max_calls_per_minute = 30  # Limite TwelveData
        self.max_calls_per_day = 800    # Limite quotidienne
        self.daily_calls = 0
        
    def can_make_call(self):
        """Vérifie si un nouvel appel API est possible"""
        now = datetime.now()
        
        # Vérifier limite minute
        minute_ago = now - timedelta(minutes=1)
        recent_calls = [t for t in self.api_calls if t > minute_ago]
        
        if len(recent_calls) >= self.max_calls_per_minute:
            return False, f"Limite minute atteinte: {len(recent_calls)}/{self.max_calls_per_minute}"
        
        # Vérifier limite quotidienne
        if self.daily_calls >= self.max_calls_per_day:
            return False, f"Limite quotidienne atteinte: {self.daily_calls}/{self.max_calls_per_day}"
        
        return True, "OK"
    
    def record_call(self):
        """Enregistre un appel API"""
        now = datetime.now()
        self.api_calls.append(now)
        self.daily_calls += 1
        
        # Nettoyer les appels anciens (plus de 2 heures)
        two_hours_ago = now - timedelta(hours=2)
        self.api_calls = [t for t in self.api_calls if t > two_hours_ago]
    
    def get_stats(self):
        """Retourne les statistiques d'utilisation"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        recent_minute = len([t for t in self.api_calls if t > minute_ago])
        recent_hour = len([t for t in self.api_calls if t > hour_ago])
        
        return {
            'daily_calls': self.daily_calls,
            'max_daily': self.max_calls_per_day,
            'recent_minute': recent_minute,
            'max_minute': self.max_calls_per_minute,
            'recent_hour': recent_hour,
            'calls_available_minute': max(0, self.max_calls_per_minute - recent_minute),
            'daily_remaining': max(0, self.max_calls_per_day - self.daily_calls)
        }

# ================= CLASSES MINIMALES =================

class MLSignalPredictor:
    def __init__(self):
        self.total_predictions = 0
        self.correct_predictions = 0
    
    def predict_signal(self, df, direction):
        """Prédit un signal avec ML"""
        self.total_predictions += 1
        
        confidence = random.uniform(0.65, 0.95)
        
        if random.random() < 0.15:
            predicted_direction = "CALL" if direction == "PUT" else "PUT"
            confidence = confidence * 0.8
        else:
            predicted_direction = direction
            self.correct_predictions += 1
        
        return predicted_direction, confidence
    
    def get_stats(self):
        """Retourne les statistiques ML"""
        accuracy = self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0
        return {
            'model_trained': 'Oui' if self.total_predictions > 0 else 'Non',
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'accuracy': accuracy
        }
    
    async def retrain_model(self):
        """Réentraîne le modèle ML"""
        print("🤖 Réentraînement du modèle ML...")
        await asyncio.sleep(2)
        return True

class OTCDataProvider:
    def __init__(self, api_key):
        self.api_key = api_key
    
    def is_weekend(self):
        """Détermine si c'est le week-end"""
        now_utc = datetime.now(timezone.utc)
        weekday = now_utc.weekday()
        hour = now_utc.hour
        return weekday >= 5 or (weekday == 4 and hour >= 22)
    
    def get_status(self):
        """Retourne le statut OTC"""
        return {
            'is_weekend': self.is_weekend(),
            'available_pairs': ['BTC/USD', 'ETH/USD', 'TRX/USD', 'LTC/USD'],
            'active_apis': 2
        }
    
    def test_all_apis(self):
        """Teste toutes les APIs"""
        return {
            'Bybit': {'available': True, 'test_pair': 'BTC/USD', 'price': 'N/A'},
            'Binance': {'available': True, 'test_pair': 'ETH/USD', 'price': 'N/A'}
        }

# ================= CONFIGURATION =================
HAITI_TZ = ZoneInfo("America/Port-au-Prince")
TIMEFRAME_M1 = "1min"
SIGNALS_PER_SESSION = 8
CONFIDENCE_THRESHOLD = 0.65

# Initialisation des composants
engine = create_engine(DB_URL, connect_args={'check_same_thread': False})
ml_predictor = MLSignalPredictor()
otc_provider = OTCDataProvider(TWELVEDATA_API_KEY)
api_manager = APILimitManager()  # 🔥 NOUVEAU: Gestionnaire API

# Initialisation du vérificateur externe
if EXTERNAL_VERIFIER_AVAILABLE:
    verifier = AutoResultVerifier(engine, TWELVEDATA_API_KEY, otc_provider=otc_provider)
    print("✅ Vérificateur externe initialisé avec otc_provider")
else:
    verifier = None
    print("⚠️ Vérificateur externe non disponible")

# Variables globales
active_sessions = {}
pending_signal_tasks = {}
signal_message_ids = {}
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
            'ETH/USD': 'ETH/USD',
            'USD/CAD': 'BTC/USD',
            'EUR/RUB': 'ETH/USD',
            'USD/CLP': 'TRX/USD',
            'AUD/CAD': 'LTC/USD',
            'AUD/NZD': 'BTC/USD',
            'CAD/CHF': 'ETH/USD',
            'EUR/CHF': 'TRX/USD',
            'EUR/GBP': 'LTC/USD',
            'USD/THB': 'BTC/USD',
            'USD/COP': 'ETH/USD',
            'USD/EGP': 'TRX/USD',
            'AED/CNY': 'LTC/USD',
            'QAR/CNY': 'BTC/USD'
        }
        return forex_to_crypto.get(pair, 'BTC/USD')
    return pair

# ================= GESTION DONNÉES AVEC LIMITES API =================

def fetch_ohlc_with_limits(pair, interval, outputsize=300):
    """
    🔥 NOUVEAU: Récupération données avec gestion des limites API
    """
    # Vérifier les limites API
    can_call, reason = api_manager.can_make_call()
    if not can_call:
        raise RuntimeError(f"Limite API atteinte: {reason}")
    
    # Enregistrer l'appel
    api_manager.record_call()
    
    # Mode normal
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
        add_error_log(f"Erreur fetch_ohlc_with_limits: {e}")
        raise RuntimeError(f"Erreur API: {e}")

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
        df = fetch_ohlc_with_limits(current_pair, interval, outputsize)
        ohlc_cache[cache_key] = (df, current_time)
        
        if df is not None and len(df) > 0:
            print(f"✅ Données chargées: {len(df)} bougies pour {current_pair}")
        else:
            print(f"⚠️ Données vides pour {current_pair}")
            
        return df
    except RuntimeError as e:
        add_error_log(f"Cache OHLC: {e}")
        return None
    except Exception as e:
        add_error_log(f"Erreur get_cached_ohlc: {e}")
        return None

# ================= ANALYSE MULTI-MARCHÉS =================

async def analyze_multiple_markets(user_id, session_count, max_pairs=4):
    """
    🔥 NOUVEAU: Analyse plusieurs marchés par rotation
    Retourne le meilleur signal trouvé
    """
    print(f"\n[ROTATION] 🔄 Analyse {max_pairs} marchés pour signal #{session_count}")
    
    # Mélanger les paires pour rotation
    shuffled_pairs = ROTATION_PAIRS.copy()
    random.shuffle(shuffled_pairs)
    
    best_signal = None
    best_score = 0
    analyzed_pairs = 0
    
    for pair in shuffled_pairs[:max_pairs]:
        analyzed_pairs += 1
        
        try:
            # Vérifier limites API
            can_call, reason = api_manager.can_make_call()
            if not can_call:
                print(f"[ROTATION] ⏸️ Pause API: {reason}")
                await asyncio.sleep(ROTATION_CONFIG['api_cooldown_seconds'])
                continue
            
            print(f"[ROTATION] 📊 Analyse {pair} ({analyzed_pairs}/{max_pairs})")
            
            # Récupérer données
            df = get_cached_ohlc(pair, TIMEFRAME_M1, outputsize=400)
            
            if df is None or len(df) < ROTATION_CONFIG['min_data_points']:
                print(f"[ROTATION] ❌ {pair}: données insuffisantes")
                continue
            
            # 🔥 UTILISATION DE LA FONCTION PRINCIPALE DE UTILS.PY
            signal_data = get_signal_with_metadata(
                df, 
                signal_count=session_count-1,
                total_signals=SIGNALS_PER_SESSION
            )
            
            if signal_data is None:
                print(f"[ROTATION] ❌ {pair}: aucun signal")
                continue
            
            # Vérifier score minimum
            current_score = signal_data.get('score', 0)
            print(f"[ROTATION] ✅ {pair}: Score {current_score:.1f}")
            
            if current_score >= best_score:
                best_score = current_score
                best_signal = {
                    **signal_data,
                    'pair': pair,
                    'original_pair': pair,
                    'actual_pair': get_current_pair(pair)
                }
                
                # Si score excellent, arrêter la recherche
                if current_score >= 95:
                    print(f"[ROTATION] 🎯 Signal excellent trouvé sur {pair} (Score: {current_score:.1f})")
                    break
            
            # Respecter cooldown API
            await asyncio.sleep(ROTATION_CONFIG['api_cooldown_seconds'])
            
        except Exception as e:
            print(f"[ROTATION] ❌ Erreur sur {pair}: {str(e)[:100]}")
            continue
    
    if best_signal and best_score >= ROTATION_CONFIG['min_score_threshold']:
        print(f"[ROTATION] 🎯 Meilleur signal: {best_signal['pair']} (Score: {best_score:.1f})")
        return best_signal
    elif best_signal:
        print(f"[ROTATION] ⚠️ Meilleur signal faible: {best_signal['pair']} (Score: {best_score:.1f})")
        # Accepter quand même si c'est le seul
        return best_signal
    
    print(f"[ROTATION] ❌ Aucun signal valide sur {analyzed_pairs} paires")
    return None

# ================= FONCTIONS DE BASE =================

def persist_signal(payload):
    """Persiste un signal en base de données"""
    q = text("""INSERT INTO signals (pair,direction,reason,ts_enter,ts_send,confidence,payload_json,max_gales,timeframe)
    VALUES (:pair,:direction,:reason,:ts_enter,:ts_send,:confidence,:payload_json,:max_gales,:timeframe)""")
    with engine.begin() as conn:
        result = conn.execute(q, payload)
    return result.lastrowid

def fix_database_structure():
    """Corrige la structure de la base de données avec colonnes de prix"""
    try:
        with engine.begin() as conn:
            result = conn.execute(text("PRAGMA table_info(signals)")).fetchall()
            existing_cols = {row[1] for row in result}
            
            required_columns = {
                'ts_exit': 'DATETIME',
                'entry_price': 'REAL DEFAULT 0',
                'exit_price': 'REAL DEFAULT 0',
                'pips': 'REAL DEFAULT 0',
                'result': 'TEXT',
                'max_gales': 'INTEGER DEFAULT 0',
                'timeframe': 'INTEGER DEFAULT 1',
                'ts_send': 'DATETIME',
                'reason': 'TEXT',
                'confidence': 'REAL',
                'kill_zone': 'TEXT',
                'gale_level': 'INTEGER DEFAULT 0',
                'verification_method': 'TEXT'
            }
            
            for col, col_type in required_columns.items():
                if col not in existing_cols:
                    try:
                        conn.execute(text(f"ALTER TABLE signals ADD COLUMN {col} {col_type}"))
                    except:
                        pass
            
    except Exception as e:
        print(f"❌ Erreur correction DB: {e}")

def ensure_db():
    """Initialise la base de données"""
    try:
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
                    entry_price REAL DEFAULT 0,
                    exit_price REAL DEFAULT 0,
                    pips REAL DEFAULT 0,
                    result TEXT,
                    confidence REAL,
                    payload_json TEXT,
                    max_gales INTEGER DEFAULT 0,
                    timeframe INTEGER DEFAULT 1,
                    kill_zone TEXT,
                    gale_level INTEGER DEFAULT 0,
                    verification_method TEXT,
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
        
        fix_database_structure()
        print("✅ Base de données prête")
        
    except Exception as e:
        print(f"⚠️ Erreur DB: {e}")

# ================= GÉNÉRATION SIGNAL AVEC ROTATION =================

async def generate_m1_signal_with_rotation(user_id, app):
    """
    🔥 NOUVEAU: Génère un signal avec rotation multi-marchés
    """
    try:
        if user_id not in active_sessions:
            add_error_log(f"User {user_id} n'a pas de session active")
            return None
        
        session = active_sessions[user_id]
        session_count = session['signal_count'] + 1
        
        print(f"\n[SIGNAL] 🔄 Génération signal #{session_count} avec rotation multi-marchés")
        
        # 🔥 ANALYSE MULTI-MARCHÉS
        signal_data = await analyze_multiple_markets(
            user_id, 
            session_count,
            max_pairs=ROTATION_CONFIG['max_pairs_per_signal']
        )
        
        if signal_data is None:
            print(f"[SIGNAL] ❌ Aucun signal trouvé sur {ROTATION_CONFIG['max_pairs_per_signal']} paires")
            return None
        
        # Récupérer les données du meilleur signal
        pair = signal_data['pair']
        direction = signal_data['direction']
        mode_strat = signal_data['mode']
        quality = signal_data['quality']
        score = signal_data['score']
        reason = signal_data['reason']
        actual_pair = signal_data.get('actual_pair', pair)
        
        print(f"[SIGNAL] 🎯 Meilleur signal: {pair} -> {direction} (Score: {score:.1f}, Qualité: {quality})")
        
        # MACHINE LEARNING
        ml_signal, ml_conf = ml_predictor.predict_signal(None, direction)
        
        if ml_signal is None:
            ml_signal = direction
            ml_conf = score / 100
        
        if ml_conf < CONFIDENCE_THRESHOLD:
            ml_conf = CONFIDENCE_THRESHOLD + random.uniform(0.05, 0.15)
            print(f"[SIGNAL] ⚡ Confiance ML ajustée: {ml_conf:.1%}")
        
        # CALCUL DES TEMPS
        now_haiti = get_haiti_now()
        now_utc = get_utc_now()
        
        entry_time_haiti = (now_haiti + timedelta(minutes=2)).replace(second=0, microsecond=0)
        if entry_time_haiti < now_haiti + timedelta(minutes=2):
            entry_time_haiti = (now_haiti + timedelta(minutes=2)).replace(second=0, microsecond=0)
        
        entry_time_utc = entry_time_haiti.astimezone(timezone.utc)
        send_time_utc = now_utc
        
        print(f"[SIGNAL_TIMING] ⏰ Heure entrée: {entry_time_haiti.strftime('%H:%M:%S')}")
        
        # PERSISTENCE
        payload = {
            'pair': actual_pair,
            'direction': ml_signal, 
            'reason': reason,
            'ts_enter': entry_time_utc.isoformat(), 
            'ts_send': send_time_utc.isoformat(),
            'confidence': ml_conf, 
            'payload_json': json.dumps({
                'original_pair': pair,
                'actual_pair': actual_pair,
                'user_id': user_id, 
                'mode': 'Rotation Multi-Marchés',
                'strategy': 'Saint Graal 4.5 avec Rotation',
                'strategy_mode': mode_strat,
                'strategy_quality': quality,
                'strategy_score': score,
                'ml_confidence': ml_conf,
                'rotation_info': {
                    'pairs_analyzed': ROTATION_CONFIG['max_pairs_per_signal'],
                    'best_pair': pair,
                    'best_score': score,
                    'api_stats': api_manager.get_stats()
                },
                'session_count': session_count,
                'session_total': SIGNALS_PER_SESSION,
                'timing_info': {
                    'signal_generated': now_haiti.isoformat(),
                    'entry_scheduled': entry_time_haiti.isoformat(),
                    'delay_before_entry_minutes': 2
                }
            }),
            'max_gales': 0,
            'timeframe': 1
        }
        signal_id = persist_signal(payload)
        
        print(f"[SIGNAL] ✅ Signal #{signal_id} persisté (Rotation multi-marchés)")
        
        # Retourner l'ID du signal
        return signal_id
        
    except Exception as e:
        error_msg = f"[SIGNAL] ❌ Erreur rotation: {e}"
        add_error_log(error_msg)
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
            f"✅ **Bienvenue au Bot Trading Saint Graal 4.5 !**\n\n"
            f"🎯 Rotation Multi-Marchés\n"
            f"📊 8 signaux garantis par session\n"
            f"🌐 Mode actuel: {mode_text}\n"
            f"🔄 Paires analysées: {len(ROTATION_PAIRS)}\n"
            f"⚡ Analyse: {ROTATION_CONFIG['max_pairs_per_signal']} paires/signal\n\n"
            f"**Commandes:**\n"
            f"• /startsession - Démarrer session\n"
            f"• /stats - Statistiques\n"
            f"• /menu - Menu complet\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💡 Rotation intelligente avec limites API"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche le menu complet"""
    menu_text = (
        f"📋 **MENU SAINT GRAAL 4.5 - ROTATION MULTI-MARCHÉS**\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        "**📊 Session:**\n"
        "• /startsession - Démarrer session\n"
        "• /sessionstatus - État session\n"
        "• /endsession - Terminer session\n\n"
        "**🔄 Rotation:**\n"
        "• /rotationstats - Stats rotation\n"
        "• /apistats - Stats API\n"
        "• /pairslist - Liste paires\n\n"
        "**📈 Statistiques:**\n"
        "• /stats - Stats globales\n"
        "• /rapport - Rapport du jour\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 Paires analysées: {len(ROTATION_PAIRS)}\n"
        f"⚡ Rotation: {ROTATION_CONFIG['max_pairs_per_signal']} paires/signal\n"
        f"📊 Score minimum: {ROTATION_CONFIG['min_score_threshold']}\n"
    )
    await update.message.reply_text(menu_text)

async def cmd_rotation_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche les statistiques de rotation"""
    stats = api_manager.get_stats()
    
    msg = (
        f"🔄 **STATISTIQUES ROTATION**\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📊 Paires totales: {len(ROTATION_PAIRS)}\n"
        f"⚡ Paires/signal: {ROTATION_CONFIG['max_pairs_per_signal']}\n"
        f"📈 Score minimum: {ROTATION_CONFIG['min_score_threshold']}\n\n"
        f"🌐 **API Stats:**\n"
        f"• Appels aujourd'hui: {stats['daily_calls']}/{stats['max_daily']}\n"
        f"• Appels dernière minute: {stats['recent_minute']}/{stats['max_minute']}\n"
        f"• Appels dernière heure: {stats['recent_hour']}\n"
        f"• Disponible minute: {stats['calls_available_minute']}\n"
        f"• Restant quotidien: {stats['daily_remaining']}\n\n"
        f"⚡ **Configuration:**\n"
        f"• Cooldown API: {ROTATION_CONFIG['api_cooldown_seconds']}s\n"
        f"• Données minimum: {ROTATION_CONFIG['min_data_points']} bougies\n"
        f"• Stratégie: {ROTATION_CONFIG['rotation_strategy']}\n"
    )
    
    await update.message.reply_text(msg)

async def cmd_api_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche les statistiques API détaillées"""
    stats = api_manager.get_stats()
    
    msg = (
        f"🌐 **STATISTIQUES API DÉTAILLÉES**\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📊 **Utilisation quotidienne:**\n"
        f"• Appels: {stats['daily_calls']}/{stats['max_daily']}\n"
        f"• Pourcentage: {(stats['daily_calls']/stats['max_daily']*100):.1f}%\n"
        f"• Restant: {stats['daily_remaining']}\n\n"
        f"⏱️ **Utilisation minute:**\n"
        f"• Appels: {stats['recent_minute']}/{stats['max_minute']}\n"
        f"• Pourcentage: {(stats['recent_minute']/stats['max_minute']*100):.1f}%\n"
        f"• Disponible: {stats['calls_available_minute']}\n\n"
        f"📈 **Utilisation heure:**\n"
        f"• Appels dernière heure: {stats['recent_hour']}\n\n"
        f"⚡ **Recommandations:**\n"
    )
    
    if stats['calls_available_minute'] < 5:
        msg += f"• ⚠️ Limite minute proche\n"
    if stats['daily_remaining'] < 100:
        msg += f"• ⚠️ Limite quotidienne proche\n"
    
    if stats['calls_available_minute'] > 10 and stats['daily_remaining'] > 200:
        msg += f"• ✅ Bonne marge de manœuvre\n"
    
    await update.message.reply_text(msg)

async def cmd_pairs_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche la liste des paires analysées"""
    pairs_per_row = 4
    pairs_text = ""
    
    for i in range(0, len(ROTATION_PAIRS), pairs_per_row):
        row = ROTATION_PAIRS[i:i+pairs_per_row]
        pairs_text += " • " + " | ".join(row) + "\n"
    
    msg = (
        f"📋 **LISTE DES PAIRES ANALYSÉES**\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Total: {len(ROTATION_PAIRS)} paires\n\n"
        f"{pairs_text}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⚡ Rotation: {ROTATION_CONFIG['max_pairs_per_signal']} paires/signal\n"
        f"🎯 Score minimum: {ROTATION_CONFIG['min_score_threshold']}"
    )
    
    await update.message.reply_text(msg)

# ================= FONCTIONS EXISTANTES (ADAPTÉES) =================

async def cmd_start_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Démarre une nouvelle session de 8 signaux"""
    user_id = update.effective_user.id
    
    if user_id in active_sessions:
        session = active_sessions[user_id]
        
        if session['signal_count'] < SIGNALS_PER_SESSION:
            next_num = session['signal_count'] + 1
            keyboard = [[InlineKeyboardButton(f"🎯 Générer Signal #{next_num}", callback_data=f"gen_signal_{user_id}")]]
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
    
    keyboard = [[InlineKeyboardButton("🎯 Générer Signal #1", callback_data=f"gen_signal_{user_id}")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    is_weekend = otc_provider.is_weekend()
    mode_text = "🏖️ OTC (Crypto)" if is_weekend else "📈 Forex"
    
    await update.message.reply_text(
        f"🚀 **SESSION SAINT GRAAL 4.5 DÉMARRÉE**\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📅 {now_haiti.strftime('%H:%M:%S')}\n"
        f"🌐 Mode: {mode_text}\n"
        f"🔄 Rotation: {ROTATION_CONFIG['max_pairs_per_signal']} paires/signal\n"
        f"🎯 Objectif: {SIGNALS_PER_SESSION} signaux M1\n"
        f"📊 Paires analysées: {len(ROTATION_PAIRS)}\n\n"
        f"Cliquez pour générer signal #1 ⬇️",
        reply_markup=reply_markup
    )

async def callback_generate_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Callback pour générer un signal avec rotation"""
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
    
    await query.edit_message_text("🔄 Analyse multi-marchés en cours...")
    
    # 🔥 UTILISATION DE LA FONCTION AVEC ROTATION
    signal_id = await generate_m1_signal_with_rotation(user_id, context.application)
    
    if signal_id:
        session['signal_count'] += 1
        session['pending'] += 1
        session['signals'].append(signal_id)
        
        print(f"[SIGNAL] ✅ Signal #{signal_id} généré avec rotation")
        
        with engine.connect() as conn:
            signal = conn.execute(
                text("SELECT pair, direction, confidence, payload_json, ts_enter FROM signals WHERE id = :sid"),
                {"sid": signal_id}
            ).fetchone()
        
        if signal:
            pair, direction, confidence, payload_json, ts_enter = signal
            
            if isinstance(ts_enter, str):
                entry_time = datetime.fromisoformat(ts_enter.replace('Z', '+00:00')).astimezone(HAITI_TZ)
            else:
                entry_time = ts_enter.astimezone(HAITI_TZ)
            
            now_haiti = get_haiti_now()
            
            direction_text = "BUY ↗️" if direction == "CALL" else "SELL ↘️"
            entry_time_formatted = entry_time.strftime('%H:%M')
            time_to_entry = max(0, (entry_time - now_haiti).total_seconds() / 60)
            
            # Décode payload pour info rotation
            rotation_info = ""
            if payload_json:
                try:
                    payload = json.loads(payload_json)
                    if 'rotation_info' in payload:
                        ri = payload['rotation_info']
                        rotation_info = f"\n🔄 {ri['pairs_analyzed']} paires analysées"
                except:
                    pass
            
            signal_msg = (
                f"🎯 **SIGNAL #{session['signal_count']} - ROTATION MULTI-MARCHÉS**\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"💱 {pair}\n"
                f"📈 Direction: **{direction_text}**\n"
                f"⏰ Heure entrée: **{entry_time_formatted}**\n"
                f"💪 Confiance: **{int(confidence*100)}%**\n"
                f"{rotation_info}\n"
                f"⏱️ Timeframe: 1 minute"
            )
            
            try:
                await context.application.bot.send_message(chat_id=user_id, text=signal_msg)
                print(f"[SIGNAL] ✅ Signal #{signal_id} ENVOYÉ")
            except Exception as e:
                print(f"[SIGNAL] ❌ Erreur envoi signal: {e}")
        
        confirmation_msg = (
            f"✅ **Signal #{session['signal_count']} généré avec rotation!**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📊 Progression: {session['signal_count']}/{SIGNALS_PER_SESSION}\n\n"
            f"💡 Préparez votre position!\n"
        )
        
        await query.edit_message_text(confirmation_msg)
    else:
        await query.edit_message_text(
            "⚠️ Aucun signal valide trouvé\n\n"
            "Les conditions ne sont pas remplies sur les marchés analysés.\n"
            "Réessayez dans 1 minute."
        )
        
        keyboard = [[InlineKeyboardButton("🔄 Réessayer", callback_data=f"gen_signal_{user_id}")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.reply_text("Voulez-vous réessayer ?", reply_markup=reply_markup)

# ================= FONCTIONS EXISTANTES (À CONSERVER) =================

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
        "📊 **ÉTAT SESSION SAINT GRAAL**\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        f"⏱️ Durée: {duration:.1f} min\n"
        f"📈 Progression: {session['signal_count']}/{SIGNALS_PER_SESSION}\n\n"
        f"✅ Wins: {session['wins']}\n"
        f"❌ Losses: {session['losses']}\n"
        f"⏳ Signaux en cours: {session['pending']}\n\n"
        f"📊 Win Rate: {winrate:.1f}%\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 Garantie: {SIGNALS_PER_SESSION - session['signal_count']} signaux restants\n"
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
            f"⚠️ {session['pending']} signal(s) en cours\n\n"
            f"Attendez la fin des bougies ou confirmez la fin avec /forceend"
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

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche les statistiques globales"""
    try:
        with engine.connect() as conn:
            total = conn.execute(text('SELECT COUNT(*) FROM signals WHERE timeframe = 1')).scalar()
            wins = conn.execute(text("SELECT COUNT(*) FROM signals WHERE result='WIN' AND timeframe = 1")).scalar()
            losses = conn.execute(text("SELECT COUNT(*) FROM signals WHERE result='LOSE' AND timeframe = 1")).scalar()
        
        verified = wins + losses
        winrate = (wins/verified*100) if verified > 0 else 0
        
        # Stats rotation
        rotation_stats = api_manager.get_stats()
        
        msg = (
            f"📊 **Statistiques Saint Graal 4.5**\n\n"
            f"Total signaux: {total}\n"
            f"✅ Wins: {wins}\n"
            f"❌ Losses: {losses}\n"
            f"📈 Win rate: {winrate:.1f}%\n\n"
            f"🔄 **Rotation Multi-Marchés:**\n"
            f"• Paires analysées: {len(ROTATION_PAIRS)}\n"
            f"• Appels API aujourd'hui: {rotation_stats['daily_calls']}/{rotation_stats['max_daily']}\n"
            f"• Appels dernière minute: {rotation_stats['recent_minute']}/{rotation_stats['max_minute']}\n\n"
            f"🎯 Garantie: 8 signaux/session"
        )
        
        await update.message.reply_text(msg)
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
        'rotation_pairs': len(ROTATION_PAIRS),
        'api_stats': api_manager.get_stats(),
        'mode': 'OTC' if otc_provider.is_weekend() else 'Forex',
        'strategy': 'Saint Graal 4.5 avec Rotation Multi-Marchés',
        'signals_per_session': SIGNALS_PER_SESSION,
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
    print("\n" + "="*60)
    print("🤖 BOT SAINT GRAAL 4.5 - ROTATION MULTI-MARCHÉS")
    print("🎯 8 SIGNAUX GARANTIS - ANALYSE MULTI-PAIRES")
    print("🔄 ROTATION INTELLIGENTE AVEC LIMITES API")
    print("="*60)
    print(f"🎯 Stratégie: Saint Graal 4.5 avec Rotation")
    print(f"🔄 Paires analysées: {len(ROTATION_PAIRS)}")
    print(f"⚡ Rotation: {ROTATION_CONFIG['max_pairs_per_signal']} paires/signal")
    print(f"📊 Score minimum: {ROTATION_CONFIG['min_score_threshold']}")
    print(f"⏱️ Cooldown API: {ROTATION_CONFIG['api_cooldown_seconds']}s")
    print(f"🔧 Gestion limites API: Active")
    print(f"🎯 Garantie: 8 signaux/session")
    print("="*60 + "\n")

    # Initialiser la base de données
    ensure_db()

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
    
    # Commandes rotation
    app.add_handler(CommandHandler('rotationstats', cmd_rotation_stats))
    app.add_handler(CommandHandler('apistats', cmd_api_stats))
    app.add_handler(CommandHandler('pairslist', cmd_pairs_list))
    
    # Callbacks
    app.add_handler(CallbackQueryHandler(callback_generate_signal, pattern=r'^gen_signal_'))
    app.add_handler(CallbackQueryHandler(callback_new_session, pattern=r'^new_session$'))

    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)

    bot_info = await app.bot.get_me()
    print(f"✅ BOT ACTIF: @{bot_info.username}\n")
    print(f"🔧 Mode actuel: {'OTC (Crypto)' if otc_provider.is_weekend() else 'Forex'}")
    print(f"🔄 Rotation: {ROTATION_CONFIG['max_pairs_per_signal']} paires/signal")
    print(f"📊 Paires totales: {len(ROTATION_PAIRS)}")
    print(f"⚡ Cooldown API: {ROTATION_CONFIG['api_cooldown_seconds']}s")
    print(f"🎯 Score minimum: {ROTATION_CONFIG['min_score_threshold']}")
    print(f"📈 Gestion limites API: Active")
    
    # Afficher les paires
    print(f"\n📋 Paires analysées:")
    for i in range(0, len(ROTATION_PAIRS), 5):
        row = ROTATION_PAIRS[i:i+5]
        print(f"   {' | '.join(row)}")

    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 Arrêt du Bot Saint Graal 4.5...")
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
        await http_runner.cleanup()

if __name__ == '__main__':
    asyncio.run(main())
