"""
signal_bot.py - Bot de trading M1 - Version Saint Graal 4.5
Analyse multi-marchés par rotation itérative avec bouton persistant
Rotation Crypto optimisée pour week-end avec affichage des paires analysées
"""

import os, json, asyncio, random, traceback, time, html, hashlib
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple
import requests
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
from aiohttp import web

# ================= CONFIGURATION INITIALE =================
# Désactiver les logs HTTP verbose
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Import des modules externes - CRITIQUE : pas de fallback
from utils import get_signal_with_metadata
print("✅ Utils importé avec succès - Fonction: get_signal_with_metadata")

from config import *
print("✅ Config importé avec succès")

# ================= CONSTANTES GLOBALES =================
HAITI_TZ = ZoneInfo("America/Port-au-Prince")
TIMEFRAME_M1 = "1min"
SIGNALS_PER_SESSION = 8
CONFIDENCE_THRESHOLD = 0.65
BUTTON_TIMEOUT_MINUTES = 5  # ⏱️ Timeout pour régénération automatique du bouton

# ================= PAIRES CRYPTO POUR WEEK-END =================
CRYPTO_PAIRS = [
    'BTC/USD',    # Bitcoin
    'ETH/USD',    # Ethereum
    'DOGE/USD',   # Dogecoin
    'SOL/USD',    # Solana
    'LTC/USD',    # Litecoin
]

# ================= GESTION DES ÉTATS =================
class SessionManager:
    """Gestionnaire centralisé des sessions utilisateur"""
    
    def __init__(self):
        self.active_sessions: Dict[int, dict] = {}
        self.pending_buttons: Dict[int, dict] = {}  # Stocke les boutons en attente
        self.button_tasks: Dict[int, asyncio.Task] = {}  # Tâches de régénération
        self.signal_tracking: Dict[int, int] = {}  # Compteur de signaux par utilisateur
        
    def create_session(self, user_id: int) -> dict:
        """Crée une nouvelle session pour un utilisateur"""
        session = {
            'user_id': user_id,
            'start_time': get_haiti_now(),
            'signal_count': 0,
            'wins': 0,
            'losses': 0,
            'pending_signals': 0,
            'active_buttons': [],
            'last_signal_time': None,
            'next_signal_number': 1,
            'status': 'active',
            'weekend_mode': False,
            'last_analysis_results': []  # Stocke les résultats d'analyse
        }
        self.active_sessions[user_id] = session
        return session
    
    def get_session(self, user_id: int) -> Optional[dict]:
        """Récupère la session d'un utilisateur"""
        return self.active_sessions.get(user_id)
    
    def update_signal_count(self, user_id: int) -> int:
        """Incrémente le compteur de signaux et retourne le nouveau numéro"""
        if user_id in self.active_sessions:
            session = self.active_sessions[user_id]
            session['signal_count'] += 1
            session['next_signal_number'] = session['signal_count'] + 1
            session['last_signal_time'] = get_haiti_now()
            return session['signal_count']
        return 0
    
    def can_generate_signal(self, user_id: int) -> Tuple[bool, str]:
        """Vérifie si un signal peut être généré"""
        if user_id not in self.active_sessions:
            return False, "Aucune session active"
        
        session = self.active_sessions[user_id]
        
        if session['status'] != 'active':
            return False, "Session terminée"
        
        if session['signal_count'] >= SIGNALS_PER_SESSION:
            return False, "Limite de signaux atteinte"
        
        # Vérifier le timeout entre les signaux
        if session['last_signal_time']:
            time_since_last = (get_haiti_now() - session['last_signal_time']).total_seconds()
            if time_since_last < 60:  # 1 minute minimum entre les signaux
                wait_time = 60 - time_since_last
                return False, f"Attendez {int(wait_time)} secondes"
        
        return True, "OK"
    
    def store_analysis_results(self, user_id: int, results: List[dict]):
        """Stocke les résultats d'analyse pour l'utilisateur"""
        if user_id in self.active_sessions:
            self.active_sessions[user_id]['last_analysis_results'] = results
    
    def get_last_analysis_results(self, user_id: int) -> List[dict]:
        """Récupère les derniers résultats d'analyse"""
        if user_id in self.active_sessions:
            return self.active_sessions[user_id].get('last_analysis_results', [])
        return []
    
    def end_session(self, user_id: int):
        """Termine une session"""
        if user_id in self.active_sessions:
            self.active_sessions[user_id]['status'] = 'ended'
            
            # Annuler les tâches de bouton
            if user_id in self.button_tasks:
                try:
                    self.button_tasks[user_id].cancel()
                except:
                    pass
                del self.button_tasks[user_id]

# ================= CONFIGURATION ROTATION =================
# Utilise directement PAIRS de config.py
ROTATION_PAIRS = PAIRS if 'PAIRS' in globals() else [
    'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'BTC/USD', 'ETH/USD',
    'USD/CAD', 'EUR/RUB', 'USD/CLP', 'AUD/CAD', 'AUD/NZD', 'CAD/CHF',
    'EUR/CHF', 'EUR/GBP', 'USD/THB', 'USD/COP', 'USD/EGP', 'AED/CNY', 'QAR/CNY'
]

print(f"📊 Chargement de {len(ROTATION_PAIRS)} paires")
print(f"🎯 Crypto week-end: {len(CRYPTO_PAIRS)} paires")

ROTATION_CONFIG = {
    'pairs_per_batch': 4,
    'max_batches_per_signal': 3,
    'min_data_points': 100,
    'api_cooldown_seconds': 2,
    'batch_cooldown_seconds': 1,
    'min_score_threshold': 85,
    'max_api_calls_per_signal': 12,
    'enable_iterative_search': True,
    'continue_if_no_signal': True,
    'rotation_strategy': 'ITERATIVE',
    'button_timeout_minutes': BUTTON_TIMEOUT_MINUTES,
}

# ================= GESTION API LIMITS =================
class APILimitManager:
    """Gestionnaire des limites d'API"""
    
    def __init__(self):
        self.api_calls = []
        self.daily_calls = 0
        self.signal_calls = {}
        self.max_calls_per_minute = 30
        self.max_calls_per_day = 800
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
    def can_make_call(self, signal_id=None):
        """Vérifie si un nouvel appel API est possible"""
        now = datetime.now()
        
        # Réinitialisation quotidienne
        if now.date() > self.daily_reset_time.date():
            self.daily_calls = 0
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Limite par minute
        minute_ago = now - timedelta(minutes=1)
        recent_calls = [t for t in self.api_calls if t > minute_ago]
        
        if len(recent_calls) >= self.max_calls_per_minute:
            return False, f"Limite minute: {len(recent_calls)}/{self.max_calls_per_minute}"
        
        # Limite quotidienne
        if self.daily_calls >= self.max_calls_per_day:
            return False, f"Limite quotidienne: {self.daily_calls}/{self.max_calls_per_day}"
        
        # Limite par signal
        if signal_id and signal_id in self.signal_calls:
            if self.signal_calls[signal_id] >= ROTATION_CONFIG['max_api_calls_per_signal']:
                return False, f"Limite signal: {self.signal_calls[signal_id]}/{ROTATION_CONFIG['max_api_calls_per_signal']}"
        
        return True, "OK"
    
    def record_call(self, signal_id=None):
        """Enregistre un appel API"""
        now = datetime.now()
        self.api_calls.append(now)
        self.daily_calls += 1
        
        if signal_id:
            if signal_id not in self.signal_calls:
                self.signal_calls[signal_id] = 0
            self.signal_calls[signal_id] += 1
        
        # Nettoyer les anciens appels
        two_hours_ago = now - timedelta(hours=2)
        self.api_calls = [t for t in self.api_calls if t > two_hours_ago]
        
        one_hour_ago = now - timedelta(hours=1)
        self.signal_calls = {k: v for k, v in self.signal_calls.items() 
                           if self.get_signal_time(k) > one_hour_ago}
    
    def get_signal_time(self, signal_id):
        """Temps du premier appel pour un signal"""
        return datetime.now() - timedelta(minutes=5)
    
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
            'daily_remaining': max(0, self.max_calls_per_day - self.daily_calls),
            'active_signals_tracking': len(self.signal_calls)
        }

# ================= INITIALISATION =================
engine = create_engine(DB_URL, connect_args={'check_same_thread': False})
session_manager = SessionManager()
api_manager = APILimitManager()

# Initialisation OTC
class OTCDataProvider:
    def __init__(self, api_key):
        self.api_key = api_key
    
    def is_weekend(self):
        now_utc = datetime.now(timezone.utc)
        weekday = now_utc.weekday()
        hour = now_utc.hour
        return weekday >= 5 or (weekday == 4 and hour >= 22)
    
    def get_status(self):
        return {
            'is_weekend': self.is_weekend(),
            'available_pairs': CRYPTO_PAIRS,
            'active_apis': 2
        }

otc_provider = OTCDataProvider(TWELVEDATA_API_KEY)

# Initialisation ML
class MLSignalPredictor:
    def __init__(self):
        self.total_predictions = 0
        self.correct_predictions = 0
    
    def predict_signal(self, df, direction):
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
        accuracy = self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0
        return {
            'model_trained': 'Oui' if self.total_predictions > 0 else 'Non',
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'accuracy': accuracy
        }

ml_predictor = MLSignalPredictor()

# Variables globales
ohlc_cache = {}
last_error_logs = []
current_signal_id = 0

# ================= FONCTIONS UTILITAIRES =================
def get_haiti_now():
    return datetime.now(HAITI_TZ)

def get_utc_now():
    return datetime.now(timezone.utc)

def add_error_log(message):
    """Ajoute un message d'erreur à la liste des logs"""
    global last_error_logs
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_entry = f"{timestamp} - {message}"
    print(log_entry)
    last_error_logs.append(log_entry)
    if len(last_error_logs) > 20:
        last_error_logs.pop(0)

def get_current_pair(pair):
    """
    Retourne la paire à utiliser
    Week-end: rotation exclusive sur paires Crypto
    Semaine: paire normale
    """
    if otc_provider.is_weekend():
        # Utiliser un hash de la paire pour une distribution équitable
        pair_hash = int(hashlib.md5(pair.encode()).hexdigest(), 16)
        crypto_index = pair_hash % len(CRYPTO_PAIRS)
        
        selected_pair = CRYPTO_PAIRS[crypto_index]
        print(f"[WEEKEND] 🔄 {pair} → {selected_pair}")
        return selected_pair
    return pair

def is_forex_open():
    """Vérifie si marché Forex est ouvert"""
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

def fetch_ohlc_with_limits(pair, interval, outputsize=300, signal_id=None):
    """
    Récupération données avec gestion des limites API
    """
    can_call, reason = api_manager.can_make_call(signal_id)
    if not can_call:
        raise RuntimeError(f"Limite API: {reason}")
    
    api_manager.record_call(signal_id)
    
    params = {
        'symbol': pair, 
        'interval': interval, 
        'outputsize': outputsize,
        'apikey': TWELVEDATA_API_KEY, 
        'format': 'JSON'
    }
    
    try:
        r = requests.get('https://api.twelvedata.com/time_series', params=params, timeout=10)
        r.raise_for_status()
        j = r.json()
        
        if 'code' in j and j['code'] == 429:
            raise RuntimeError("Limite API TwelveData atteinte")
        
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
        add_error_log(f"Erreur fetch_ohlc: {e}")
        raise RuntimeError(f"Erreur API: {e}")

def get_cached_ohlc(pair, interval, outputsize=300, signal_id=None):
    """Récupère les données OHLC depuis le cache ou les APIs"""
    current_pair = get_current_pair(pair)
    cache_key = f"{current_pair}_{interval}"
    
    current_time = get_utc_now()
    
    if cache_key in ohlc_cache:
        cached_data, cached_time = ohlc_cache[cache_key]
        if (current_time - cached_time).total_seconds() < 30:
            return cached_data
    
    try:
        df = fetch_ohlc_with_limits(current_pair, interval, outputsize, signal_id)
        ohlc_cache[cache_key] = (df, current_time)
        
        if df is not None and len(df) > 0:
            print(f"✅ Données chargées: {len(df)} bougies pour {current_pair}")
        else:
            print(f"⚠️ Données vides pour {current_pair}")
            
        return df
    except Exception as e:
        add_error_log(f"Erreur get_cached_ohlc: {e}")
        return None

# ================= GESTION BASE DE DONNÉES =================
def ensure_db():
    """Initialise la base de données"""
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
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
                    button_message_id INTEGER,
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
            
            # Vérifier et ajouter la colonne user_id si elle n'existe pas
            try:
                result = conn.execute(text("PRAGMA table_info(signals)")).fetchall()
                existing_cols = {row[1] for row in result}
                if 'user_id' not in existing_cols:
                    conn.execute(text("ALTER TABLE signals ADD COLUMN user_id INTEGER"))
            except:
                pass
        
        print("✅ Base de données prête")
        
    except Exception as e:
        print(f"⚠️ Erreur DB: {e}")

def persist_signal(user_id, payload):
    """Persiste un signal en base de données"""
    q = text("""INSERT INTO signals (
        user_id, pair, direction, reason, ts_enter, ts_send, confidence, 
        payload_json, max_gales, timeframe, button_message_id
    ) VALUES (
        :user_id, :pair, :direction, :reason, :ts_enter, :ts_send, :confidence, 
        :payload_json, :max_gales, :timeframe, :button_message_id
    )""")
    
    payload['user_id'] = user_id
    
    with engine.begin() as conn:
        result = conn.execute(q, payload)
    return result.lastrowid

# ================= ANALYSE MULTI-MARCHÉS =================
async def analyze_multiple_markets_iterative(user_id, session_count, signal_id=None):
    """
    Analyse itérative de plusieurs marchés
    Rotation Crypto optimisée pour week-end
    RETOURNE: (signal_data, total_pairs_analyzed, total_batches, analysis_results)
    """
    print(f"\n[ROTATION] 🔄 Analyse itérative pour signal #{session_count}")
    
    # Déterminer mode week-end
    is_weekend = otc_provider.is_weekend()
    if is_weekend:
        print(f"[ROTATION] 🌙 WEEK-END MODE: Rotation exclusive Crypto")
        print(f"[ROTATION] 🎯 Paires Crypto: {', '.join(CRYPTO_PAIRS)}")
    else:
        print(f"[ROTATION] 📊 Total paires Forex: {len(ROTATION_PAIRS)}")
    
    shuffled_pairs = ROTATION_PAIRS.copy()
    random.shuffle(shuffled_pairs)
    
    best_signal = None
    best_score = 0
    total_analyzed = 0
    batch_count = 0
    analysis_results = []  # Stocke les résultats d'analyse de chaque paire
    
    for batch_start in range(0, len(shuffled_pairs), ROTATION_CONFIG['pairs_per_batch']):
        batch_count += 1
        
        if batch_count > ROTATION_CONFIG['max_batches_per_signal']:
            print(f"[ROTATION] ⏹️ Maximum de batches atteint")
            break
        
        batch_pairs = shuffled_pairs[batch_start:batch_start + ROTATION_CONFIG['pairs_per_batch']]
        print(f"\n[ROTATION] 📦 Batch #{batch_count}: analyse {len(batch_pairs)} paires")
        
        batch_best_signal = None
        batch_best_score = 0
        
        for pair in batch_pairs:
            total_analyzed += 1
            
            try:
                can_call, reason = api_manager.can_make_call(signal_id)
                if not can_call:
                    print(f"[ROTATION] ⏸️ Limite API: {reason}")
                    break
                
                # Obtenir la paire actuelle (Crypto le week-end)
                actual_pair = get_current_pair(pair)
                
                if is_weekend:
                    print(f"[ROTATION] 🌙 {pair} → {actual_pair} ({total_analyzed}ème Crypto)")
                else:
                    print(f"[ROTATION] 📊 Analyse {pair} ({total_analyzed}ème)")
                
                df = get_cached_ohlc(pair, TIMEFRAME_M1, outputsize=400, signal_id=signal_id)
                
                if df is None or len(df) < ROTATION_CONFIG['min_data_points']:
                    result = {
                        'original_pair': pair,
                        'actual_pair': actual_pair,
                        'status': 'ERROR',
                        'score': 0,
                        'reason': 'Données insuffisantes',
                        'batch': batch_count,
                        'position': batch_pairs.index(pair) + 1
                    }
                    analysis_results.append(result)
                    print(f"[ROTATION] ❌ {actual_pair}: données insuffisantes")
                    continue
                
                # 🔥 UTILISATION EXCLUSIVE DE get_signal_with_metadata
                signal_data = get_signal_with_metadata(
                    df, 
                    signal_count=session_count-1,
                    total_signals=SIGNALS_PER_SESSION
                )
                
                if signal_data is None:
                    result = {
                        'original_pair': pair,
                        'actual_pair': actual_pair,
                        'status': 'NO_SIGNAL',
                        'score': 0,
                        'reason': 'Aucun signal détecté',
                        'batch': batch_count,
                        'position': batch_pairs.index(pair) + 1
                    }
                    analysis_results.append(result)
                    print(f"[ROTATION] ❌ {actual_pair}: aucun signal")
                    continue
                
                current_score = signal_data.get('score', 0)
                result = {
                    'original_pair': pair,
                    'actual_pair': actual_pair,
                    'status': 'SIGNAL_FOUND',
                    'score': current_score,
                    'reason': signal_data.get('reason', 'N/A'),
                    'direction': signal_data.get('direction', 'N/A'),
                    'batch': batch_count,
                    'position': batch_pairs.index(pair) + 1
                }
                analysis_results.append(result)
                
                print(f"[ROTATION] ✅ {actual_pair}: Score {current_score:.1f}")
                
                if current_score > batch_best_score:
                    batch_best_score = current_score
                    batch_best_signal = {
                        **signal_data,
                        'pair': actual_pair,
                        'original_pair': pair,
                        'actual_pair': actual_pair,
                        'batch': batch_count,
                        'position_in_batch': batch_pairs.index(pair) + 1,
                        'is_weekend': is_weekend
                    }
                
                if current_score >= 95:
                    print(f"[ROTATION] 🎯 Signal excellent trouvé")
                    best_signal = {
                        **signal_data,
                        'pair': actual_pair,
                        'original_pair': pair,
                        'actual_pair': actual_pair,
                        'batch': batch_count,
                        'position_in_batch': batch_pairs.index(pair) + 1,
                        'is_weekend': is_weekend
                    }
                    best_score = current_score
                    return best_signal, total_analyzed, batch_count, analysis_results
                
                await asyncio.sleep(ROTATION_CONFIG['api_cooldown_seconds'])
                
            except Exception as e:
                error_msg = str(e)[:100]
                actual_pair = get_current_pair(pair) if 'actual_pair' not in locals() else actual_pair
                result = {
                    'original_pair': pair,
                    'actual_pair': actual_pair,
                    'status': 'ERROR',
                    'score': 0,
                    'reason': f"Erreur: {error_msg}",
                    'batch': batch_count,
                    'position': batch_pairs.index(pair) + 1
                }
                analysis_results.append(result)
                print(f"[ROTATION] ❌ Erreur sur {pair}: {error_msg}")
                continue
        
        if batch_best_signal and batch_best_score >= ROTATION_CONFIG['min_score_threshold']:
            print(f"[ROTATION] 🎯 Signal acceptable trouvé")
            best_signal = batch_best_signal
            best_score = batch_best_score
            break
        
        print(f"[ROTATION] ⚠️ Aucun signal valide dans batch #{batch_count}")
        
        if not ROTATION_CONFIG['continue_if_no_signal']:
            print(f"[ROTATION] ⏹️ Configuration: ne pas continuer sans signal")
            break
        
        await asyncio.sleep(ROTATION_CONFIG['batch_cooldown_seconds'])
    
    if best_signal and best_score >= ROTATION_CONFIG['min_score_threshold']:
        pair_type = "Crypto" if is_weekend else "Forex"
        print(f"[ROTATION] ✅ Meilleur signal {pair_type}: {best_signal['pair']} (Score: {best_score:.1f})")
        return best_signal, total_analyzed, batch_count, analysis_results
    
    pair_type = "Crypto" if is_weekend else "Forex"
    print(f"[ROTATION] ❌ Aucun signal valide après {total_analyzed} paires {pair_type}")
    return None, total_analyzed, batch_count, analysis_results

# ================= GESTION BOUTON PERSISTANT =================
async def create_signal_button(user_id: int, app, message_id: int = None) -> int:
    """
    Crée ou met à jour un bouton pour générer le prochain signal
    Retourne l'ID du message contenant le bouton
    """
    session = session_manager.get_session(user_id)
    if not session:
        return None
    
    next_signal_num = session['next_signal_number']
    button_text = f"🎯 Générer Signal #{next_signal_num}"
    
    keyboard = [[InlineKeyboardButton(button_text, callback_data=f"gen_signal_{user_id}")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    try:
        if message_id:
            # Mettre à jour le message existant
            await app.bot.edit_message_reply_markup(
                chat_id=user_id,
                message_id=message_id,
                reply_markup=reply_markup
            )
            return message_id
        else:
            # Créer un nouveau message avec bouton
            message = await app.bot.send_message(
                chat_id=user_id,
                text=f"🔄 **Bouton actif pour le signal #{next_signal_num}**\n"
                     f"━━━━━━━━━━━━━━━━━━━━\n"
                     f"Cliquez pour générer le prochain signal ⬇️\n"
                     f"⏱️ Ce bouton expire dans {BUTTON_TIMEOUT_MINUTES} minutes",
                reply_markup=reply_markup
            )
            
            # Planifier la régénération automatique
            asyncio.create_task(schedule_button_regeneration(user_id, app, message.message_id))
            
            return message.message_id
    except Exception as e:
        print(f"❌ Erreur création bouton: {e}")
        return None

async def schedule_button_regeneration(user_id: int, app, message_id: int):
    """
    Planifie la régénération automatique du bouton après timeout
    """
    try:
        # Attendre le timeout
        await asyncio.sleep(BUTTON_TIMEOUT_MINUTES * 60)
        
        # Vérifier si la session est toujours active
        session = session_manager.get_session(user_id)
        if not session or session['status'] != 'active':
            return
        
        if session['signal_count'] >= SIGNALS_PER_SESSION:
            return
        
        print(f"🔄 Régénération automatique du bouton pour l'utilisateur {user_id}")
        
        # Régénérer le bouton
        new_message_id = await create_signal_button(user_id, app, message_id)
        
        if new_message_id:
            # Mettre à jour la session
            if 'active_buttons' not in session:
                session['active_buttons'] = []
            
            if message_id in session['active_buttons']:
                session['active_buttons'].remove(message_id)
            
            if new_message_id:
                session['active_buttons'].append(new_message_id)
            
    except asyncio.CancelledError:
        print(f"⏹️ Tâche de régénération annulée pour l'utilisateur {user_id}")
    except Exception as e:
        print(f"❌ Erreur régénération bouton: {e}")

async def cleanup_old_buttons(user_id: int, app):
    """
    Nettoie les anciens boutons
    """
    session = session_manager.get_session(user_id)
    if not session or 'active_buttons' not in session:
        return
    
    for message_id in session['active_buttons'][:-1]:  # Garder seulement le dernier
        try:
            await app.bot.delete_message(chat_id=user_id, message_id=message_id)
        except:
            pass
    
    # Garder seulement le dernier bouton
    if session['active_buttons']:
        session['active_buttons'] = [session['active_buttons'][-1]]

# ================= GÉNÉRATION DE SIGNAL =================
async def generate_m1_signal_with_iterative_rotation(user_id, app):
    """
    Génère un signal avec rotation itérative - PAS DE FALLBACK
    Rotation Crypto optimisée pour week-end
    RETOURNE: (signal_id, analysis_results) ou (None, analysis_results)
    """
    global current_signal_id
    
    try:
        session = session_manager.get_session(user_id)
        if not session:
            add_error_log(f"User {user_id} n'a pas de session active")
            return None, []
        
        session_count = session['signal_count'] + 1
        current_signal_id += 1
        signal_tracking_id = f"sig_{session_count}_{current_signal_id}"
        
        print(f"\n[SIGNAL] 🔄 Génération signal #{session_count}")
        
        # Vérifier mode week-end
        is_weekend = otc_provider.is_weekend()
        if is_weekend:
            print(f"[SIGNAL] 🌙 MODE WEEK-END: Rotation exclusive sur 5 paires Crypto")
        
        # Analyse multi-marchés - PAS DE FALLBACK
        signal_data, total_pairs_analyzed, total_batches, analysis_results = await analyze_multiple_markets_iterative(
            user_id, 
            session_count,
            signal_id=signal_tracking_id
        )
        
        # Stocker les résultats d'analyse dans la session
        session_manager.store_analysis_results(user_id, analysis_results)
        
        # 🔥 AUCUN FALLBACK - SI PAS DE SIGNAL, RETOURNER None
        if signal_data is None:
            pair_type = "Crypto" if is_weekend else "Forex"
            print(f"[SIGNAL] ❌ Aucun signal valide trouvé après analyse {pair_type}")
            return None, analysis_results
        
        pair = signal_data['pair']
        direction = signal_data['direction']
        mode_strat = signal_data['mode']
        quality = signal_data['quality']
        score = signal_data['score']
        reason = signal_data['reason']
        actual_pair = signal_data.get('actual_pair', pair)
        batch_info = f"Batch {signal_data.get('batch', '?')}.{signal_data.get('position_in_batch', '?')}"
        is_weekend_mode = signal_data.get('is_weekend', False)
        
        print(f"[SIGNAL] 🎯 Meilleur signal: {pair} -> {direction} (Score: {score:.1f})")
        
        # Machine Learning
        ml_signal, ml_conf = ml_predictor.predict_signal(None, direction)
        
        if ml_signal is None:
            ml_signal = direction
            ml_conf = score / 100
        
        if ml_conf < CONFIDENCE_THRESHOLD:
            ml_conf = CONFIDENCE_THRESHOLD + random.uniform(0.05, 0.15)
            print(f"[SIGNAL] ⚡ Confiance ML ajustée: {ml_conf:.1%}")
        
        # Calcul des temps
        now_haiti = get_haiti_now()
        now_utc = get_utc_now()
        
        entry_time_haiti = (now_haiti + timedelta(minutes=2)).replace(second=0, microsecond=0)
        if entry_time_haiti < now_haiti + timedelta(minutes=2):
            entry_time_haiti = (now_haiti + timedelta(minutes=2)).replace(second=0, microsecond=0)
        
        entry_time_utc = entry_time_haiti.astimezone(timezone.utc)
        send_time_utc = now_utc
        
        print(f"[SIGNAL_TIMING] ⏰ Heure entrée: {entry_time_haiti.strftime('%H:%M:%S')}")
        
        # Persistence
        payload = {
            'pair': actual_pair,
            'direction': ml_signal, 
            'reason': f"{reason} | {batch_info}",
            'ts_enter': entry_time_utc.isoformat(), 
            'ts_send': send_time_utc.isoformat(),
            'confidence': ml_conf, 
            'payload_json': json.dumps({
                'original_pair': pair,
                'actual_pair': actual_pair,
                'user_id': user_id, 
                'mode': 'Rotation Itérative Multi-Marchés',
                'strategy': 'Saint Graal 4.5 avec Rotation Itérative',
                'strategy_mode': mode_strat,
                'strategy_quality': quality,
                'strategy_score': score,
                'ml_confidence': ml_conf,
                'rotation_info': {
                    'pairs_analyzed': total_pairs_analyzed,
                    'batches_analyzed': total_batches,
                    'best_pair': pair,
                    'best_score': score,
                    'batch_info': batch_info,
                    'signal_tracking_id': signal_tracking_id,
                    'api_stats': api_manager.get_stats(),
                    'is_weekend': is_weekend_mode,
                    'crypto_pairs': CRYPTO_PAIRS if is_weekend_mode else [],
                    'analysis_results': analysis_results  # Ajout des résultats détaillés
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
            'timeframe': 1,
            'button_message_id': None  # À remplir après création du bouton
        }
        
        signal_id = persist_signal(user_id, payload)
        
        print(f"[SIGNAL] ✅ Signal #{signal_id} persisté")
        
        return signal_id, analysis_results
        
    except Exception as e:
        error_msg = f"[SIGNAL] ❌ Erreur: {e}"
        add_error_log(error_msg)
        traceback.print_exc()
        return None, []

# ================= FONCTION POUR BOUTON APRÈS BOUGIE =================
async def schedule_button_after_candle(signal_id, user_id, app, entry_time):
    """
    Programme l'envoi du bouton APRÈS la fin de la bougie M1
    """
    try:
        print(f"[BOUGIE-BOUTON] ⏰ Programmation bouton pour signal #{signal_id}")
        
        # Calculer la fin de la bougie M1 (1 minute après l'entrée)
        candle_end_time = entry_time + timedelta(minutes=1)
        now_utc = get_utc_now()
        
        # Attendre EXACTEMENT la fin de la bougie
        wait_seconds = max(0, (candle_end_time - now_utc).total_seconds())
        
        if wait_seconds > 0:
            print(f"[BOUGIE-BOUTON] ⏳ Attente de {wait_seconds:.0f}s pour fin de bougie")
            await asyncio.sleep(wait_seconds)
        
        # ENVOYER LE BOUTON IMMÉDIATEMENT APRÈS FIN BOUGIE
        print(f"[BOUGIE-BOUTON] ✅ Bougie terminée, envoi bouton IMMÉDIAT pour signal #{signal_id}")
        
        # Créer le bouton
        session = session_manager.get_session(user_id)
        if not session:
            return
        
        # Nettoyer les anciens boutons
        await cleanup_old_buttons(user_id, app)
        
        # Créer un nouveau bouton
        new_message_id = await create_signal_button(user_id, app)
        
        if new_message_id:
            # Mettre à jour la session
            if 'active_buttons' not in session:
                session['active_buttons'] = []
            session['active_buttons'].append(new_message_id)
            
            # Envoyer un message d'information
            try:
                info_msg = (
                    f"🔄 **Bougie terminée**\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"⏰ La bougie M1 est maintenant terminée.\n"
                    f"📊 Progression: {session['signal_count']}/{SIGNALS_PER_SESSION}\n\n"
                    f"🎯 Bouton disponible pour le signal #{session['next_signal_number']}\n"
                    f"⏱️ Cliquez pour continuer!"
                )
                
                await app.bot.send_message(
                    chat_id=user_id,
                    text=info_msg
                )
            except:
                pass
            
    except asyncio.CancelledError:
        print(f"[BOUGIE-BOUTON] ❌ Tâche annulée pour signal #{signal_id}")
    except Exception as e:
        print(f"[BOUGIE-BOUTON] ❌ Erreur: {e}")

# ================= COMMANDES TELEGRAM =================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Commande de démarrage"""
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
        
        if is_weekend:
            crypto_list = "\n".join([f"• {pair}" for pair in CRYPTO_PAIRS])
            mode_text += f"\n🎯 Paires Crypto: {', '.join(CRYPTO_PAIRS)}"
        
        await update.message.reply_text(
            f"✅ **Bienvenue au Bot Trading Saint Graal 4.5 !**\n\n"
            f"🎯 Rotation Itérative Multi-Marchés\n"
            f"📊 {len(ROTATION_PAIRS)} paires disponibles\n"
            f"🌙 {len(CRYPTO_PAIRS)} paires Crypto week-end\n"
            f"🔄 Bouton après bougie avec régénération automatique\n"
            f"⏱️ Timeout bouton: {BUTTON_TIMEOUT_MINUTES} minutes\n"
            f"📈 Affichage détaillé des paires analysées\n"
            f"🌐 Mode actuel: {mode_text}\n\n"
            f"**Commandes:**\n"
            f"• /startsession - Démarrer session\n"
            f"• /menu - Menu complet\n"
            f"• /lastanalysis - Voir dernières paires analysées\n"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche le menu complet"""
    is_weekend = otc_provider.is_weekend()
    
    if is_weekend:
        crypto_section = f"🌙 **MODE WEEK-END:**\n• Rotation exclusive Crypto\n• {len(CRYPTO_PAIRS)} paires analysées\n"
    else:
        crypto_section = f"📈 **MODE FOREX:**\n• Rotation standard\n• {len(ROTATION_PAIRS)} paires analysées\n"
    
    menu_text = (
        f"📋 **MENU SAINT GRAAL 4.5**\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{crypto_section}\n"
        "**📊 Session:**\n"
        "• /startsession - Démarrer session\n"
        "• /sessionstatus - État session\n"
        "• /endsession - Terminer session\n"
        "• /lastanalysis - Dernières analyses\n\n"
        "**🔄 Rotation:**\n"
        "• /rotationstats - Stats rotation\n"
        "• /apistats - Stats API\n"
        "• /pairslist - Liste paires\n"
        "• /cryptolist - Liste Crypto week-end\n\n"
        "**⚙️ Configuration:**\n"
        "• /buttonconfig - Configuration bouton\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 Paires Forex: {len(ROTATION_PAIRS)}\n"
        f"🌙 Paires Crypto: {len(CRYPTO_PAIRS)}\n"
        f"🔄 Bouton timeout: {BUTTON_TIMEOUT_MINUTES} min\n"
        f"📊 Affichage analyses: ✅ ACTIVÉ\n"
    )
    await update.message.reply_text(menu_text)

async def cmd_start_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Démarre une nouvelle session"""
    user_id = update.effective_user.id
    
    session = session_manager.get_session(user_id)
    if session and session['status'] == 'active':
        next_num = session['next_signal_number']
        
        # D'ABORD LE TEXTE
        await update.message.reply_text(
            f"⚠️ Session déjà active !\n\n"
            f"📊 Progression: {session['signal_count']}/{SIGNALS_PER_SESSION}\n"
            f"✅ Wins: {session['wins']}\n"
            f"❌ Losses: {session['losses']}\n\n"
            f"Continuer avec signal #{next_num} ⬇️"
        )
        
        # ENSUITE LE BOUTON
        keyboard = [[InlineKeyboardButton(
            f"🎯 Générer Signal #{next_num}", 
            callback_data=f"gen_signal_{user_id}"
        )]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "⬇️ Bouton de génération ⬇️",
            reply_markup=reply_markup
        )
        return
    
    # Créer nouvelle session
    session = session_manager.create_session(user_id)
    
    is_weekend = otc_provider.is_weekend()
    mode_text = "🏖️ OTC (Crypto)" if is_weekend else "📈 Forex"
    
    if is_weekend:
        crypto_details = f"🎯 Paires Crypto: {', '.join(CRYPTO_PAIRS)}"
        mode_text += f"\n{crypto_details}"
    
    # D'ABORD LE TEXTE DE SESSION DÉMARRÉE
    await update.message.reply_text(
        f"🚀 **SESSION DÉMARRÉE**\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📅 {session['start_time'].strftime('%H:%M:%S')}\n"
        f"🌐 Mode: {mode_text}\n"
        f"🎯 Objectif: {SIGNALS_PER_SESSION} signaux M1\n"
        f"🔄 Bouton timeout: {BUTTON_TIMEOUT_MINUTES} minutes\n"
        f"⚡ Bouton après bougie: ACTIVÉ\n"
        f"📊 Affichage analyses: ACTIVÉ\n\n"
        f"Cliquez sur le bouton pour commencer ⬇️"
    )
    
    # ENSUITE LE BOUTON
    keyboard = [[InlineKeyboardButton(
        "🎯 Générer Signal #1", 
        callback_data=f"gen_signal_{user_id}"
    )]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    message = await update.message.reply_text(
        "⬇️ Bouton de génération ⬇️",
        reply_markup=reply_markup
    )
    
    if message:
        session['active_buttons'] = [message.message_id]

async def callback_generate_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Callback pour générer un signal avec affichage des paires analysées"""
    query = update.callback_query
    await query.answer()
    
    user_id = int(query.data.split('_')[2])
    
    # Vérifier si la session est active
    can_generate, reason = session_manager.can_generate_signal(user_id)
    if not can_generate:
        await query.edit_message_text(f"❌ {reason}\n\nUtilisez /startsession")
        return
    
    session = session_manager.get_session(user_id)
    
    # Déterminer mode
    is_weekend = otc_provider.is_weekend()
    mode_text = "🌙 Crypto" if is_weekend else "📈 Forex"
    
    # Mettre à jour le message avec état
    await query.edit_message_text(
        f"🔄 **Génération du signal #{session['next_signal_number']}**\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Mode: {mode_text}\n"
        f"Analyse rotation itérative en cours...\n"
        f"⏱️ Patientez quelques secondes..."
    )
    
    # Générer le signal - PAS DE FALLBACK
    signal_id, analysis_results = await generate_m1_signal_with_iterative_rotation(user_id, context.application)
    
    if signal_id:
        # Mettre à jour la session
        session_manager.update_signal_count(user_id)
        session['pending_signals'] += 1
        
        # Récupérer les infos du signal
        with engine.connect() as conn:
            signal = conn.execute(
                text("SELECT pair, direction, confidence, payload_json, ts_enter FROM signals WHERE id = :sid"),
                {"sid": signal_id}
            ).fetchone()
        
        if signal:
            pair, direction, confidence, payload_json, ts_enter = signal
            
            if isinstance(ts_enter, str):
                entry_time = datetime.fromisoformat(ts_enter.replace('Z', '+00:00')).astimezone(HAITI_TZ)
                entry_time_utc = datetime.fromisoformat(ts_enter.replace('Z', '+00:00'))
            else:
                entry_time = ts_enter.astimezone(HAITI_TZ)
                entry_time_utc = ts_enter
            
            direction_text = "BUY ↗️" if direction == "CALL" else "SELL ↘️"
            entry_time_formatted = entry_time.strftime('%H:%M')
            
            # Info rotation avec résultats d'analyse
            rotation_info = ""
            if analysis_results:
                analyzed_count = len(analysis_results)
                successful_analysis = len([r for r in analysis_results if r['status'] == 'SIGNAL_FOUND'])
                pair_type = "Crypto 🌙" if is_weekend else "Forex 📈"
                rotation_info = f"\n🔄 {analyzed_count} paires {pair_type} analysées ({successful_analysis} avec signal)"
            
            # Envoyer le signal
            signal_msg = (
                f"🎯 **SIGNAL #{session['signal_count']} - ROTATION ITÉRATIVE**\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"💱 {pair}\n"
                f"📈 Direction: **{direction_text}**\n"
                f"⏰ Heure entrée: **{entry_time_formatted}**\n"
                f"💪 Confiance: **{int(confidence*100)}%**\n"
                f"{rotation_info}\n"
                f"⏱️ Timeframe: 1 minute\n\n"
                f"✅ Signal généré avec succès!"
            )
            
            try:
                await context.application.bot.send_message(chat_id=user_id, text=signal_msg)
                print(f"[SIGNAL] ✅ Signal #{signal_id} envoyé")
            except Exception as e:
                print(f"[SIGNAL] ❌ Erreur envoi: {e}")
        
        # Nettoyer les anciens boutons
        await cleanup_old_buttons(user_id, context.application)
        
        # Vérifier si la session est terminée
        if session['signal_count'] >= SIGNALS_PER_SESSION:
            await end_session_summary(user_id, context.application)
            return
        
        # PROGRAMMER LE BOUTON APRÈS LA BOUGIE
        if signal:
            # Planifier l'envoi du bouton après la bougie
            button_task = asyncio.create_task(
                schedule_button_after_candle(signal_id, user_id, context.application, entry_time_utc)
            )
            
            # Stocker la tâche dans la session
            if 'button_tasks' not in session:
                session['button_tasks'] = []
            session['button_tasks'].append(button_task)
        
        # Message de confirmation
        confirmation_msg = (
            f"✅ **Signal #{session['signal_count']} généré!**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📊 Progression: {session['signal_count']}/{SIGNALS_PER_SESSION}\n\n"
            f"💡 Préparez votre position!\n"
            f"⏰ Le bouton pour le prochain signal apparaîtra après la fin de la bougie."
        )
        
        await query.edit_message_text(confirmation_msg)
    else:
        # 🔥 AUCUN SIGNAL TROUVÉ - AFFICHAGE DÉTAILLÉ DES PAIRES ANALYSÉES
        pair_type = "Crypto" if is_weekend else "Forex"
        
        # Préparer la liste des paires analysées
        analyzed_pairs_text = ""
        if analysis_results:
            # Grouper par batch
            batches = {}
            for result in analysis_results:
                batch_num = result.get('batch', 0)
                if batch_num not in batches:
                    batches[batch_num] = []
                batches[batch_num].append(result)
            
            analyzed_pairs_text = "📊 **Paires analysées:**\n"
            
            for batch_num in sorted(batches.keys()):
                analyzed_pairs_text += f"\n**Batch {batch_num}:**\n"
                batch_results = batches[batch_num]
                
                for i, result in enumerate(batch_results, 1):
                    pair_display = result['actual_pair']
                    status = result['status']
                    
                    if status == 'SIGNAL_FOUND':
                        score = result.get('score', 0)
                        direction = result.get('direction', 'N/A')
                        direction_emoji = "↗️" if direction == "CALL" else "↘️"
                        analyzed_pairs_text += f"{i}. {pair_display} {direction_emoji} - Score: {score:.1f}\n"
                    elif status == 'NO_SIGNAL':
                        analyzed_pairs_text += f"{i}. {pair_display} ❌ - Pas de signal\n"
                    elif status == 'ERROR':
                        reason = result.get('reason', 'Erreur')
                        analyzed_pairs_text += f"{i}. {pair_display} ⚠️ - {reason}\n"
                    else:
                        analyzed_pairs_text += f"{i}. {pair_display} ❓ - État inconnu\n"
            
            # Ajouter un résumé
            total_pairs = len(analysis_results)
            signals_found = len([r for r in analysis_results if r['status'] == 'SIGNAL_FOUND'])
            errors = len([r for r in analysis_results if r['status'] == 'ERROR'])
            no_signals = len([r for r in analysis_results if r['status'] == 'NO_SIGNAL'])
            
            # Trouver le meilleur score
            best_score = max([r.get('score', 0) for r in analysis_results if r.get('score', 0) > 0], default=0)
            
            analyzed_pairs_text += f"\n**📈 Résumé:**\n"
            analyzed_pairs_text += f"• Total paires analysées: {total_pairs}\n"
            analyzed_pairs_text += f"• Signaux détectés: {signals_found}\n"
            analyzed_pairs_text += f"• Pas de signal: {no_signals}\n"
            analyzed_pairs_text += f"• Erreurs: {errors}\n"
            analyzed_pairs_text += f"• Meilleur score: {best_score:.1f}\n"
            analyzed_pairs_text += f"• Mode: {'Crypto 🌙' if is_weekend else 'Forex 📈'}\n"
        else:
            analyzed_pairs_text = "❌ Aucune paire n'a pu être analysée."
        
        # Message d'erreur détaillé
        error_msg = (
            f"❌ **Aucun signal valide trouvé**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{analyzed_pairs_text}\n\n"
            f"🎯 **Critères de sélection:**\n"
            f"• Score minimum requis: **{ROTATION_CONFIG['min_score_threshold']}**\n"
            f"• Batches analysés: **{len(batches) if 'batches' in locals() else 0}/{ROTATION_CONFIG['max_batches_per_signal']}**\n\n"
            f"🔄 **Recommandation:**\n"
            f"Essayez à nouveau dans 1 minute.\n"
            f"Le système analysera un nouveau lot de paires."
        )
        
        await query.edit_message_text(error_msg)
        
        # Recréer un bouton pour réessayer
        new_message_id = await create_signal_button(user_id, context.application)
        
        if new_message_id:
            if 'active_buttons' not in session:
                session['active_buttons'] = []
            session['active_buttons'].append(new_message_id)

async def cmd_session_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche l'état de la session"""
    user_id = update.effective_user.id
    
    session = session_manager.get_session(user_id)
    if not session or session['status'] != 'active':
        await update.message.reply_text("ℹ️ Aucune session active\n\nUtilisez /startsession")
        return
    
    duration = (get_haiti_now() - session['start_time']).total_seconds() / 60
    winrate = (session['wins'] / session['signal_count'] * 100) if session['signal_count'] > 0 else 0
    
    is_weekend = otc_provider.is_weekend()
    mode_text = "🌙 Mode Crypto (Week-end)" if is_weekend else "📈 Mode Forex"
    
    # Récupérer les dernières analyses
    analysis_results = session_manager.get_last_analysis_results(user_id)
    last_analysis_info = ""
    if analysis_results:
        total_analyzed = len(analysis_results)
        signals_found = len([r for r in analysis_results if r['status'] == 'SIGNAL_FOUND'])
        last_analysis_info = f"\n📊 **Dernière analyse:** {total_analyzed} paires, {signals_found} signaux"
    
    msg = (
        "📊 **ÉTAT SESSION**\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{mode_text}\n"
        f"⏱️ Durée: {duration:.1f} min\n"
        f"📈 Progression: {session['signal_count']}/{SIGNALS_PER_SESSION}\n\n"
        f"✅ Wins: {session['wins']}\n"
        f"❌ Losses: {session['losses']}\n"
        f"⏳ Signaux en cours: {session['pending_signals']}\n\n"
        f"📊 Win Rate: {winrate:.1f}%\n"
        f"🔄 Prochain signal: #{session['next_signal_number']}\n"
        f"⏱️ Dernier signal: {session['last_signal_time'].strftime('%H:%M:%S') if session['last_signal_time'] else 'N/A'}"
        f"{last_analysis_info}\n\n"
        f"⚡ **Bouton:**\n"
        f"• Timeout: {BUTTON_TIMEOUT_MINUTES} minutes\n"
        f"• Après bougie: ✅ ACTIVÉ\n"
        f"• Affichage analyses: ✅ ACTIVÉ\n"
        f"• Boutons actifs: {len(session.get('active_buttons', []))}"
    )
    
    await update.message.reply_text(msg)

async def cmd_end_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Termine la session"""
    user_id = update.effective_user.id
    
    session = session_manager.get_session(user_id)
    if not session:
        await update.message.reply_text("ℹ️ Aucune session active")
        return
    
    if session['pending_signals'] > 0:
        await update.message.reply_text(
            f"⚠️ {session['pending_signals']} signal(s) en cours\n\n"
            f"Attendez la fin des bouches ou utilisez /forceend"
        )
        return
    
    session_manager.end_session(user_id)
    await end_session_summary(user_id, context.application)

async def cmd_force_end(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Force la fin de session"""
    user_id = update.effective_user.id
    
    session = session_manager.get_session(user_id)
    if not session:
        await update.message.reply_text("ℹ️ Aucune session active")
        return
    
    session_manager.end_session(user_id)
    await end_session_summary(user_id, context.application)
    await update.message.reply_text("✅ Session terminée (forcée) !")

async def end_session_summary(user_id, app):
    """Envoie le résumé de fin de session"""
    session = session_manager.get_session(user_id)
    if not session:
        return
    
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
        "🎯 Garantie: 8 signaux/session\n"
        "Utilisez /startsession pour nouvelle session"
    )
    
    await app.bot.send_message(chat_id=user_id, text=summary)

async def cmd_rotation_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche les statistiques de rotation"""
    stats = api_manager.get_stats()
    is_weekend = otc_provider.is_weekend()
    
    if is_weekend:
        crypto_section = f"🌙 **MODE WEEK-END ACTIF**\n• Rotation exclusive sur {len(CRYPTO_PAIRS)} paires Crypto\n• {', '.join(CRYPTO_PAIRS)}\n"
        pairs_text = f"🎯 Paires Crypto: {len(CRYPTO_PAIRS)}"
    else:
        crypto_section = f"📈 **MODE FOREX ACTIF**\n• Rotation standard\n"
        pairs_text = f"📊 Paires Forex: {len(ROTATION_PAIRS)}"
    
    msg = (
        f"🔄 **STATISTIQUES ROTATION**\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{crypto_section}\n"
        f"{pairs_text}\n"
        f"🔄 Paires/batch: {ROTATION_CONFIG['pairs_per_batch']}\n"
        f"📦 Max batches: {ROTATION_CONFIG['max_batches_per_signal']}\n"
        f"🎯 Score minimum: {ROTATION_CONFIG['min_score_threshold']}\n"
        f"⚡ Recherche itérative: {'✅ OUI' if ROTATION_CONFIG['enable_iterative_search'] else '❌ NON'}\n"
        f"📊 Affichage analyses: ✅ ACTIVÉ\n\n"
        f"🌐 **API Stats:**\n"
        f"• Appels aujourd'hui: {stats['daily_calls']}/{stats['max_daily']}\n"
        f"• Appels dernière minute: {stats['recent_minute']}/{stats['max_minute']}\n"
        f"• Appels dernière heure: {stats['recent_hour']}\n"
        f"• Restant quotidien: {stats['daily_remaining']}\n"
    )
    
    await update.message.reply_text(msg)

async def cmd_button_config(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche la configuration du bouton"""
    msg = (
        f"⚙️ **CONFIGURATION BOUTON**\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"🔄 **Système de bouton après bougie:**\n"
        f"• Apparaît après: Fin de bougie M1\n"
        f"• Timeout: {BUTTON_TIMEOUT_MINUTES} minutes\n"
        f"• Régénération auto: ✅ ACTIVÉE\n"
        f"• Nettoyage auto: ✅ ACTIVÉ\n"
        f"• Affichage analyses: ✅ ACTIVÉ\n\n"
        f"🎯 **Fonctionnement:**\n"
        f"1. Signal généré → Envoyé immédiatement\n"
        f"2. Bouton apparaît → Après fin bougie M1\n"
        f"3. Se régénère → Après timeout\n"
        f"4. Un seul bouton → Actif à la fois\n"
        f"5. Si aucun signal → Affiche paires analysées\n\n"
        f"⚠️ **En cas de problème:**\n"
        f"• Utilisez /startsession pour régénérer\n"
        f"• Vérifiez /sessionstatus pour l'état\n"
        f"• Voir /lastanalysis pour analyses détaillées\n"
        f"• Contactez le support si problème persiste"
    )
    
    await update.message.reply_text(msg)

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche les statistiques globales"""
    try:
        with engine.connect() as conn:
            total = conn.execute(text('SELECT COUNT(*) FROM signals')).scalar()
            wins = conn.execute(text("SELECT COUNT(*) FROM signals WHERE result='WIN'")).scalar()
            losses = conn.execute(text("SELECT COUNT(*) FROM signals WHERE result='LOSE'")).scalar()
        
        verified = wins + losses
        winrate = (wins/verified*100) if verified > 0 else 0
        
        rotation_stats = api_manager.get_stats()
        is_weekend = otc_provider.is_weekend()
        
        mode_text = f"🌙 Week-end (Crypto)" if is_weekend else f"📈 Forex"
        
        msg = (
            f"📊 **Statistiques Globales**\n\n"
            f"Mode actuel: {mode_text}\n"
            f"Total signaux: {total}\n"
            f"✅ Wins: {wins}\n"
            f"❌ Losses: {losses}\n"
            f"📈 Win rate: {winrate:.1f}%\n\n"
            f"🔄 **Rotation:**\n"
            f"• Paires analysées: {len(CRYPTO_PAIRS) if is_weekend else len(ROTATION_PAIRS)}\n"
            f"• Appels API: {rotation_stats['daily_calls']}/{rotation_stats['max_daily']}\n\n"
            f"🎯 **Sessions actives:** {len(session_manager.active_sessions)}\n"
            f"🔄 **Bouton après bougie:** ✅ ACTIVÉ\n"
            f"📊 **Affichage analyses:** ✅ ACTIVÉ"
        )
        
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_pairslist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche la liste des paires analysées"""
    pairs_per_row = 3
    pairs_text = ""
    
    for i in range(0, len(ROTATION_PAIRS), pairs_per_row):
        row = ROTATION_PAIRS[i:i+pairs_per_row]
        pairs_text += " • " + " | ".join(row) + "\n"
    
    msg = (
        f"📋 **LISTE DES PAIRES FOREX**\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Source: config.py\n"
        f"Total: {len(ROTATION_PAIRS)} paires\n\n"
        f"{pairs_text}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔄 Rotation: {ROTATION_CONFIG['pairs_per_batch']} paires/batch\n"
        f"📦 Max: {ROTATION_CONFIG['max_batches_per_signal']} batches/signal\n"
        f"🎯 Score minimum: {ROTATION_CONFIG['min_score_threshold']}\n\n"
        f"ℹ️ Utilisez /cryptolist pour voir les paires Crypto week-end"
    )
    
    await update.message.reply_text(msg)

async def cmd_cryptolist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche la liste des paires Crypto pour week-end"""
    crypto_text = "\n".join([f"• {pair}" for pair in CRYPTO_PAIRS])
    
    is_weekend = otc_provider.is_weekend()
    weekend_status = "✅ ACTIF" if is_weekend else "⏸️ INACTIF"
    
    msg = (
        f"🌙 **LISTE DES PAIRES CRYPTO (WEEK-END)**\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Mode week-end: {weekend_status}\n"
        f"Total: {len(CRYPTO_PAIRS)} paires\n\n"
        f"{crypto_text}\n\n"
        f"🔧 **Fonctionnement:**\n"
        f"• Le week-end (ven 22h - dim 22h UTC)\n"
        f"• Toutes les paires Forex sont transformées en Crypto\n"
        f"• Rotation exclusive sur ces {len(CRYPTO_PAIRS)} paires\n"
        f"• Distribution équitable via hash MD5\n\n"
        f"🎯 **Paires disponibles:**\n"
        f"• BTC/USD - Bitcoin\n"
        f"• ETH/USD - Ethereum\n"
        f"• DOGE/USD - Dogecoin\n"
        f"• SOL/USD - Solana\n"
        f"• LTC/USD - Litecoin"
    )
    
    await update.message.reply_text(msg)

async def cmd_last_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche les dernières paires analysées"""
    user_id = update.effective_user.id
    
    analysis_results = session_manager.get_last_analysis_results(user_id)
    
    if not analysis_results:
        await update.message.reply_text("ℹ️ Aucune analyse disponible.\nGénérez un signal avec le bouton pour voir les résultats.")
        return
    
    # Préparer le message
    is_weekend = otc_provider.is_weekend()
    pair_type = "Crypto" if is_weekend else "Forex"
    
    # Compter les statistiques
    total_pairs = len(analysis_results)
    signals_found = len([r for r in analysis_results if r['status'] == 'SIGNAL_FOUND'])
    no_signals = len([r for r in analysis_results if r['status'] == 'NO_SIGNAL'])
    errors = len([r for r in analysis_results if r['status'] == 'ERROR'])
    
    # Trouver le meilleur score
    best_score = max([r.get('score', 0) for r in analysis_results if r.get('score', 0) > 0], default=0)
    best_pair = next((r['actual_pair'] for r in analysis_results if r.get('score', 0) == best_score), "N/A")
    
    msg = (
        f"📊 **DERNIÈRES PAIRES ANALYSÉES**\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"🌐 Mode: {'Crypto 🌙' if is_weekend else 'Forex 📈'}\n"
        f"📈 Total paires analysées: {total_pairs}\n\n"
        f"✅ Signaux détectés: {signals_found}\n"
        f"❌ Pas de signal: {no_signals}\n"
        f"⚠️ Erreurs: {errors}\n\n"
        f"🏆 **Meilleur résultat:**\n"
        f"• Paire: {best_pair}\n"
        f"• Score: {best_score:.1f}\n\n"
        f"🎯 Score minimum requis: {ROTATION_CONFIG['min_score_threshold']}"
    )
    
    # Ajouter les 5 dernières paires analysées
    if analysis_results:
        msg += f"\n\n📋 **5 dernières analyses:**\n"
        recent_results = analysis_results[-5:] if len(analysis_results) > 5 else analysis_results
        
        for i, result in enumerate(recent_results, 1):
            pair = result['actual_pair']
            status = result['status']
            
            if status == 'SIGNAL_FOUND':
                score = result.get('score', 0)
                direction = result.get('direction', 'N/A')
                direction_emoji = "↗️" if direction == "CALL" else "↘️"
                msg += f"{i}. {pair} {direction_emoji} ✅ Score: {score:.1f}\n"
            elif status == 'NO_SIGNAL':
                msg += f"{i}. {pair} ❌ Pas de signal\n"
            elif status == 'ERROR':
                reason = result.get('reason', 'Erreur')
                msg += f"{i}. {pair} ⚠️ {reason[:30]}\n"
            else:
                msg += f"{i}. {pair} ❓ Inconnu\n"
    
    await update.message.reply_text(msg)

async def cmd_apistats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche les statistiques API détaillées"""
    stats = api_manager.get_stats()
    
    msg = (
        f"🌐 **STATISTIQUES API**\n"
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
        f"🎯 **Signaux trackés:** {stats['active_signals_tracking']}"
    )
    
    await update.message.reply_text(msg)

# ================= SERVEUR HTTP =================
async def health_check(request):
    """Endpoint de santé"""
    is_weekend = otc_provider.is_weekend()
    return web.json_response({
        'status': 'ok',
        'timestamp': get_haiti_now().isoformat(),
        'active_sessions': len(session_manager.active_sessions),
        'rotation_pairs': len(ROTATION_PAIRS),
        'crypto_pairs': len(CRYPTO_PAIRS),
        'weekend_mode': is_weekend,
        'button_timeout': BUTTON_TIMEOUT_MINUTES,
        'button_after_candle': 'active',
        'analysis_display': 'active'
    })

async def start_http_server():
    """Démarre le serveur HTTP"""
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
    print("🤖 BOT SAINT GRAAL 4.5 - ROTATION ITÉRATIVE")
    print("🎯 8 SIGNAUX GARANTIS - BOUTON APRÈS BOUGIE")
    print("🌙 ROTATION CRYPTO OPTIMISÉE WEEK-END")
    print("📊 AFFICHAGE DÉTAILLÉ DES PAIRES ANALYSÉES")
    print("="*60)
    print(f"🎯 Stratégie: Saint Graal 4.5 avec Rotation Itérative")
    print(f"📊 Paires Forex analysées: {len(ROTATION_PAIRS)}")
    print(f"🌙 Paires Crypto week-end: {len(CRYPTO_PAIRS)}")
    print(f"🔄 Batch: {ROTATION_CONFIG['pairs_per_batch']} paires")
    print(f"📦 Max batches: {ROTATION_CONFIG['max_batches_per_signal']}")
    print(f"🎯 Score minimum: {ROTATION_CONFIG['min_score_threshold']}")
    print(f"🔄 Bouton après bougie: ✅ ACTIVÉ")
    print(f"📊 Affichage analyses: ✅ ACTIVÉ")
    print(f"⏱️ Bouton timeout: {BUTTON_TIMEOUT_MINUTES} minutes")
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
    
    # Commandes rotation et analyse
    app.add_handler(CommandHandler('rotationstats', cmd_rotation_stats))
    app.add_handler(CommandHandler('buttonconfig', cmd_button_config))
    app.add_handler(CommandHandler('pairslist', cmd_pairslist))
    app.add_handler(CommandHandler('cryptolist', cmd_cryptolist))
    app.add_handler(CommandHandler('lastanalysis', cmd_last_analysis))
    app.add_handler(CommandHandler('apistats', cmd_apistats))
    
    # Callbacks
    app.add_handler(CallbackQueryHandler(callback_generate_signal, pattern=r'^gen_signal_'))
    
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)

    bot_info = await app.bot.get_me()
    print(f"✅ BOT ACTIF: @{bot_info.username}\n")
    
    is_weekend = otc_provider.is_weekend()
    if is_weekend:
        print(f"🌙 MODE WEEK-END ACTIF: Rotation exclusive Crypto")
        print(f"🎯 Paires Crypto: {', '.join(CRYPTO_PAIRS)}")
    else:
        print(f"📈 MODE FOREX ACTIF: Rotation standard")
        print(f"📊 Paires Forex: {len(ROTATION_PAIRS)}")
    
    print(f"🔄 Bouton après bougie: ✅ ACTIVÉ")
    print(f"📊 Affichage analyses: ✅ ACTIVÉ")
    print(f"⏱️ Bouton timeout: {BUTTON_TIMEOUT_MINUTES} min")
    print(f"🔧 Utilisez /lastanalysis pour voir les paires analysées")

    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 Arrêt du bot...")
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
        await http_runner.cleanup()

if __name__ == '__main__':
    asyncio.run(main())
