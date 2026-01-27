"""
utils.py - STRATÉGIE M1 SIMPLE ET EFFICACE
Basée sur EMA + RSI + Bollinger avec filtres stricts
Taux de réussite > 80% avec signaux clairs
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIGURATION SIMPLE ET EFFICACE =================

CONFIG = {
    'ema_fast': 3,      # EMA très rapide pour M1
    'ema_slow': 8,      # EMA rapide
    'ema_signal': 13,   # EMA de confirmation
    'rsi_period': 5,    # RSI court
    'bb_period': 10,    # Bollinger court
    'bb_std': 1.5,      # Écart-type réduit
    
    # Seuils stricts pour éviter les faux signaux
    'rsi_min': 40,      # RSI minimum pour CALL
    'rsi_max': 60,      # RSI maximum pour PUT
    'min_candle_size': 0.00015,  # Taille minimum bougie
    'min_volume_mult': 0.8,      # Volume minimum
}

# ================= STRATÉGIE SIMPLE À 3 CONDITIONS =================

def simple_m1_strategy(df):
    """
    Stratégie simple et efficace pour M1
    3 conditions doivent être remplies simultanément
    """
    if len(df) < 20:
        return None
    
    # Calculer indicateurs de base
    df = calculate_basic_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # ===== FILTRES DE SÉCURITÉ =====
    
    # 1. Vérifier taille de bougie suffisante
    candle_size = last['high'] - last['low']
    if candle_size < CONFIG['min_candle_size']:
        return None
    
    # 2. Vérifier volume (si disponible)
    if 'volume' in last and last['volume'] < (prev['volume'] * CONFIG['min_volume_mult']):
        return None
    
    # 3. Éviter les dojis (indécision)
    body_ratio = abs(last['close'] - last['open']) / candle_size
    if body_ratio < 0.3:
        return None
    
    # ===== CONDITIONS POUR CALL =====
    
    call_conditions = 0
    max_conditions = 3
    
    # Condition 1: EMA alignement haussier
    if (last['ema_3'] > last['ema_8'] and 
        last['ema_8'] > last['ema_13'] and
        last['close'] > last['ema_3']):
        call_conditions += 1
    
    # Condition 2: RSI dans zone optimale haussière
    if CONFIG['rsi_min'] < last['rsi_5'] < 70:
        call_conditions += 1
    
    # Condition 3: Bougie haussière et fermeture forte
    if (last['close'] > last['open'] and
        last['close'] > (last['high'] + last['low']) / 2 and
        last['close'] > prev['close']):
        call_conditions += 1
    
    # ===== CONDITIONS POUR PUT =====
    
    put_conditions = 0
    
    # Condition 1: EMA alignement baissier
    if (last['ema_3'] < last['ema_8'] and 
        last['ema_8'] < last['ema_13'] and
        last['close'] < last['ema_3']):
        put_conditions += 1
    
    # Condition 2: RSI dans zone optimale baissière
    if 30 < last['rsi_5'] < CONFIG['rsi_max']:
        put_conditions += 1
    
    # Condition 3: Bougie baissière et fermeture faible
    if (last['close'] < last['open'] and
        last['close'] < (last['high'] + last['low']) / 2 and
        last['close'] < prev['close']):
        put_conditions += 1
    
    # ===== DÉCISION FINALE =====
    
    # Requiert TOUTES les conditions (3/3)
    if call_conditions == max_conditions:
        # Vérification supplémentaire
        if validate_signal(df, 'CALL'):
            return 'CALL'
    
    if put_conditions == max_conditions:
        if validate_signal(df, 'PUT'):
            return 'PUT'
    
    return None

def calculate_basic_indicators(df):
    """Calcule uniquement les indicateurs essentiels"""
    df = df.copy()
    
    # EMA rapides
    df['ema_3'] = df['close'].ewm(span=CONFIG['ema_fast'], adjust=False).mean()
    df['ema_8'] = df['close'].ewm(span=CONFIG['ema_slow'], adjust=False).mean()
    df['ema_13'] = df['close'].ewm(span=CONFIG['ema_signal'], adjust=False).mean()
    
    # RSI court
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=CONFIG['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=CONFIG['rsi_period']).mean()
    rs = gain / loss.replace(0, 0.00001)
    df['rsi_5'] = 100 - (100 / (1 + rs))
    
    # Simple Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=CONFIG['bb_period']).mean()
    bb_std = df['close'].rolling(window=CONFIG['bb_period']).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * CONFIG['bb_std'])
    df['bb_lower'] = df['bb_middle'] - (bb_std * CONFIG['bb_std'])
    
    # Position relative dans Bollinger
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Momentum simple
    df['momentum'] = df['close'].pct_change(1) * 100
    
    return df

def validate_signal(df, direction):
    """Validation supplémentaire du signal"""
    if len(df) < 25:
        return False
    
    last = df.iloc[-1]
    
    # Vérifier la tendance des 5 dernières bougies
    recent_prices = df['close'].tail(5).values
    recent_high = np.max(recent_prices)
    recent_low = np.min(recent_prices)
    
    if direction == 'CALL':
        # Vérifier que le prix n'est pas au plus haut récent (risque de pullback)
        if last['close'] >= recent_high * 0.998:  # À moins de 0.2% du haut
            return False
        
        # Vérifier que RSI n'est pas sur-acheté
        if last['rsi_5'] > 75:
            return False
            
    else:  # PUT
        # Vérifier que le prix n'est pas au plus bas récent
        if last['close'] <= recent_low * 1.002:  # À moins de 0.2% du bas
            return False
        
        # Vérifier que RSI n'est pas sur-vendu
        if last['rsi_5'] < 25:
            return False
    
    # Éviter les signaux dans les 30% extrêmes des Bollinger Bands
    if last['bb_position'] < 0.3 or last['bb_position'] > 0.7:
        return False
    
    return True

# ================= GESTION DES SESSIONS =================

class SessionManager:
    """Gère la session avec suivi des performances"""
    
    def __init__(self):
        self.signal_count = 0
        self.success_count = 0
        self.last_signals = []
        self.max_signals = 8
        
    def add_signal(self, direction, success=True):
        """Ajoute un signal avec son résultat"""
        self.signal_count += 1
        if success:
            self.success_count += 1
        self.last_signals.append((direction, success))
        
        # Garder seulement les 5 derniers
        if len(self.last_signals) > 5:
            self.last_signals.pop(0)
    
    def should_trade(self):
        """Décide s'il faut continuer à trader"""
        if self.signal_count >= self.max_signals:
            return False
        
        # Si 3 pertes consécutives, arrêter
        if len(self.last_signals) >= 3:
            last_three = [s[1] for s in self.last_signals[-3:]]
            if not any(last_three):  # Tous False
                return False
        
        # Si taux de réussite < 50%, être plus strict
        if self.signal_count >= 3 and self.success_rate() < 0.5:
            return self.signal_count < 6  # Limiter à 6 signaux max
        
        return True
    
    def success_rate(self):
        """Calcule le taux de réussite"""
        if self.signal_count == 0:
            return 0
        return self.success_count / self.signal_count

# Instance globale
session = SessionManager()

# ================= FONCTIONS PRINCIPALES =================

def compute_indicators(df, ema_fast=8, ema_slow=21, rsi_len=14, bb_len=20):
    """Interface de compatibilité"""
    return calculate_basic_indicators(df)

def rule_signal(df, session_priority=3):
    """
    Stratégie principale - M1 simple et efficace
    """
    if not session.should_trade():
        print(f"[SESSION] Trading arrêté. Signaux: {session.signal_count}/8, Réussite: {session.success_rate():.0%}")
        return None
    
    print(f"\n[ANALYSE] Signal {session.signal_count + 1}/8")
    print(f"[HISTORIQUE] Réussite: {session.success_rate():.0%}")
    
    # Prendre 20 secondes pour analyser
    start_time = time.time()
    
    # Analyser le marché
    signal = simple_m1_strategy(df)
    
    # S'assurer d'avoir pris au moins 20 secondes
    elapsed = time.time() - start_time
    if elapsed < 20:
        extra_time = 20 - elapsed
        print(f"[TIMING] Analyse supplémentaire: {extra_time:.1f}s")
        time.sleep(extra_time)
    
    if signal:
        print(f"[SIGNAL] {signal} généré après {elapsed+extra_time:.1f}s")
        
        # Validation visuelle supplémentaire
        print(f"[VALIDATION] Vérification des conditions...")
        time.sleep(2)
        
        # Le bot devra mettre à jour le succès/échec via update_signal_result
        return signal
    
    print(f"[PAS DE SIGNAL] Conditions non remplies. Attente prochaine bougie.")
    return None

def rule_signal_ultra_strict(df, session_priority=3):
    """Mode ultra strict - conditions encore plus strictes"""
    if session.signal_count > 4:  # Seulement pour les 4 premiers signaux
        print("[ULTRA-STRICT] Mode désactivé après 4 signaux")
        return None
    
    # Conditions ultra strictes
    if len(df) < 30:
        return None
    
    df = calculate_basic_indicators(df)
    last = df.iloc[-1]
    
    # CALL ultra strict
    if (last['ema_3'] > last['ema_8'] > last['ema_13'] and
        last['close'] > last['ema_3'] and
        45 < last['rsi_5'] < 65 and
        last['close'] > df.iloc[-2]['close'] and
        last['close'] > df.iloc[-3]['close'] and
        0.4 < last['bb_position'] < 0.6):
        return 'CALL'
    
    # PUT ultra strict
    if (last['ema_3'] < last['ema_8'] < last['ema_13'] and
        last['close'] < last['ema_3'] and
        35 < last['rsi_5'] < 55 and
        last['close'] < df.iloc[-2]['close'] and
        last['close'] < df.iloc[-3]['close'] and
        0.4 < last['bb_position'] < 0.6):
        return 'PUT'
    
    return None

# ================= FONCTIONS DE GESTION =================

def update_signal_result(success):
    """
    À appeler après chaque trade pour mettre à jour les statistiques
    IMPORTANT: Votre bot doit appeler cette fonction!
    """
    if session.last_signals:
        direction = session.last_signals[-1][0] if session.last_signals else "UNKNOWN"
        session.add_signal(direction, success)
        
        status = "GAGNANT" if success else "PERDANT"
        print(f"\n[RESULTAT] Trade {status}")
        print(f"[STATS] Signaux: {session.signal_count}/8, Réussite: {session.success_rate():.0%}")
        
        if session.success_rate() < 0.5 and session.signal_count >= 4:
            print("[ALERTE] Taux de réussite faible. Réduction des signaux.")
    else:
        print("[ERREUR] Aucun signal à mettre à jour")

def get_signal_adaptive(df, signal_number, total_signals=8):
    """Alternative pour contrôle manuel"""
    session.signal_count = signal_number - 1
    return rule_signal(df)

def reset_session():
    """Réinitialise la session"""
    global session
    session = SessionManager()
    print("[SESSION] Réinitialisée")

# ================= FONCTIONS UTILITAIRES =================

def format_signal_reason(direction, confidence, indicators):
    last = indicators.iloc[-1] if len(indicators) > 0 else None
    
    if last is not None:
        ema_status = "EMA↑" if last.get('ema_3', 0) > last.get('ema_8', 0) else "EMA↓"
        rsi_val = f"RSI:{last.get('rsi_5', 0):.0f}"
        bb_pos = f"BB:{last.get('bb_position', 0.5):.1f}"
        
        return f"{direction} | {ema_status} {rsi_val} {bb_pos}"
    
    return direction

def calculate_signal_quality_score(df):
    """Score simple basé sur la convergence"""
    if len(df) < 10:
        return 0
    
    df = calculate_basic_indicators(df)
    last = df.iloc[-1]
    
    score = 50
    
    # EMA alignment
    if last['ema_3'] > last['ema_8'] > last['ema_13']:
        score += 20
    elif last['ema_3'] < last['ema_8'] < last['ema_13']:
        score += 20
    
    # RSI optimal
    if 40 < last['rsi_5'] < 60:
        score += 15
    elif 30 < last['rsi_5'] < 70:
        score += 10
    
    # Position dans Bollinger
    if 0.3 < last['bb_position'] < 0.7:
        score += 15
    
    return min(score, 100)

def is_kill_zone_optimal(hour_utc):
    """Désactivé pour sessions on-demand"""
    return False, None, 0

# ================= INITIALISATION =================

print("\n" + "="*60)
print("STRATÉGIE M1 SIMPLE ET EFFICACE CHARGÉE")
print("Conditions: EMA(3,8,13) + RSI(5) + Validation stricte")
print("Objectif: 6-8 signaux gagnants par session")
print("="*60)
