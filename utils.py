"""
utils.py - STRATÉGIE SAINT GRAAL FOREX M1 ANALYSE PROFONDE
Version ultra-accélérée avec analyse en temps réel des 100 dernières bougies
Garantie de signaux QUALITÉ avec validation multi-couches
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from ta.trend import EMAIndicator, MACD, ADXIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange, DonchianChannel, KeltnerChannel
from ta.volume import VolumeWeightedAveragePrice, MFIIndicator, OnBalanceVolumeIndicator
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIGURATION ULTRA-ACCÉLÉRÉE =================

SAINT_GRAAL_CONFIG = {
    # Timeframes optimisés POUR VITESSE
    'rsi_period': 5,        # RSI plus rapide
    'ema_fast': 3,          # EMA ultra-rapide
    'ema_slow': 8,          # EMA rapide
    'stoch_period': 5,
    'macd_fast': 3,         # MACD ultra-rapide
    'macd_slow': 7,
    'macd_signal': 2,
    'bb_period': 10,        # Bollinger plus réactif
    'adx_period': 7,
    
    # Seuils STRICT (qualité max)
    'strict': {
        'rsi_overbought': 65,
        'rsi_oversold': 35,
        'adx_min': 20,
        'min_indicators_confirm': 7,  # 7/9 indicateurs doivent confirmer
        'min_convergence': 0.75,      # 75% de convergence
    },
    
    # Seuils NORMAL (équilibre qualité/vitesse)
    'normal': {
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'adx_min': 15,
        'min_indicators_confirm': 5,  # 5/9 indicateurs
        'min_convergence': 0.60,
    },
    
    # Seuils GARANTIE (pour compléter session)
    'guarantee': {
        'rsi_overbought': 75,
        'rsi_oversold': 25,
        'adx_min': 12,
        'min_indicators_confirm': 4,  # 4/9 indicateurs
        'min_convergence': 0.50,
    },
}

# ================= FONCTIONS ULTRA-RAPIDES =================

def compute_ultra_fast_indicators(df):
    """
    Calcule les indicateurs en temps réel - OPTIMISÉ POUR VITESSE
    Ne calcule que l'essentiel pour M1
    """
    if len(df) < 5:
        return df
    
    df = df.copy()
    
    # Assurer les types
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remplir NaN rapidement
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # ===== INDICATEURS ESSENTIELS (optimisés) =====
    
    # EMA ultra-rapides
    df['ema_3'] = df['close'].ewm(span=3, adjust=False).mean()
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema_13'] = df['close'].ewm(span=13, adjust=False).mean()
    
    # MACD ultra-rapide (3,7,2)
    exp1 = df['close'].ewm(span=3, adjust=False).mean()
    exp2 = df['close'].ewm(span=7, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=2, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # RSI rapide
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=5).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=5).mean()
    rs = gain / loss.replace(0, 0.00001)
    df['rsi_5'] = 100 - (100 / (1 + rs))
    
    # RSI très rapide
    gain_3 = (delta.where(delta > 0, 0)).rolling(window=3).mean()
    loss_3 = (-delta.where(delta < 0, 0)).rolling(window=3).mean()
    rs_3 = gain_3 / loss_3.replace(0, 0.00001)
    df['rsi_3'] = 100 - (100 / (1 + rs_3))
    
    # Bollinger Bands rapides
    df['bb_middle'] = df['close'].rolling(window=10).mean()
    bb_std = df['close'].rolling(window=10).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 1.5)  # 1.5 au lieu de 2 pour plus de sensibilité
    df['bb_lower'] = df['bb_middle'] - (bb_std * 1.5)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Stochastique rapide
    low_5 = df['low'].rolling(window=5).min()
    high_5 = df['high'].rolling(window=5).max()
    df['stoch_k'] = 100 * ((df['close'] - low_5) / (high_5 - low_5))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Price action
    df['candle_body'] = df['close'] - df['open']
    df['candle_size'] = df['high'] - df['low']
    df['body_ratio'] = abs(df['candle_body']) / df['candle_size'].replace(0, 0.00001)
    
    # Momentum instantané
    df['momentum_1'] = df['close'].pct_change(1) * 100
    df['momentum_3'] = df['close'].pct_change(3) * 100
    
    # Volume Weighted Price (si volume disponible)
    if 'volume' in df.columns:
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['price_vs_vwap'] = (df['close'] / df['vwap']) - 1
    
    # Force du mouvement
    df['trend_strength'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 100
    
    # Volatilité instantanée
    df['volatility'] = df['close'].rolling(window=10).std() / df['close'].rolling(window=10).mean()
    
    # Dernier remplissage
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

# ================= ANALYSE EN TEMPS RÉEL =================

def analyze_realtime_signal(df):
    """
    Analyse en temps réel avec validation multi-niveaux
    Retourne un signal seulement si confirmé par plusieurs indicateurs
    """
    if len(df) < 10:
        return None
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # ===== NIVEAU 1: ANALYSE TECHNIQUE RAPIDE =====
    
    # Score CALL
    call_signals = 0
    total_signals = 9
    
    # 1. EMA alignment
    if last['ema_3'] > last['ema_5'] > last['ema_8']:
        call_signals += 1
    
    # 2. MACD positif
    if last['macd'] > last['macd_signal'] and last['macd_diff'] > 0:
        call_signals += 1
    
    # 3. RSI haussier
    if 55 < last['rsi_5'] < 70:
        call_signals += 1
    
    # 4. Position dans Bollinger
    if last['bb_position'] > 0.5:
        call_signals += 1
    
    # 5. Stochastique haussier
    if last['stoch_k'] > last['stoch_d'] and last['stoch_k'] > 50:
        call_signals += 1
    
    # 6. Candle haussière
    if last['candle_body'] > 0 and last['body_ratio'] > 0.3:
        call_signals += 1
    
    # 7. Momentum positif
    if last['momentum_1'] > 0:
        call_signals += 1
    
    # 8. Tendance générale
    if last['trend_strength'] > 0:
        call_signals += 1
    
    # 9. Volatilité contrôlée
    if last['volatility'] < 0.02:
        call_signals += 1
    
    call_ratio = call_signals / total_signals
    
    # Score PUT
    put_signals = 0
    
    # 1. EMA alignment
    if last['ema_3'] < last['ema_5'] < last['ema_8']:
        put_signals += 1
    
    # 2. MACD négatif
    if last['macd'] < last['macd_signal'] and last['macd_diff'] < 0:
        put_signals += 1
    
    # 3. RSI baissier
    if 30 < last['rsi_5'] < 45:
        put_signals += 1
    
    # 4. Position dans Bollinger
    if last['bb_position'] < 0.5:
        put_signals += 1
    
    # 5. Stochastique baissier
    if last['stoch_k'] < last['stoch_d'] and last['stoch_k'] < 50:
        put_signals += 1
    
    # 6. Candle baissière
    if last['candle_body'] < 0 and last['body_ratio'] > 0.3:
        put_signals += 1
    
    # 7. Momentum négatif
    if last['momentum_1'] < 0:
        put_signals += 1
    
    # 8. Tendance générale
    if last['trend_strength'] < 0:
        put_signals += 1
    
    # 9. Volatilité contrôlée
    if last['volatility'] < 0.02:
        put_signals += 1
    
    put_ratio = put_signals / total_signals
    
    # ===== NIVEAU 2: VALIDATION PAR CONVERGENCE =====
    
    convergence_threshold = 0.65  # 65% de convergence minimum
    
    if call_ratio >= convergence_threshold and call_ratio > put_ratio:
        # Validation supplémentaire
        if validate_signal_quality(df, 'CALL'):
            return {
                'signal': 'CALL',
                'confidence': call_ratio,
                'indicators': call_signals,
                'reason': f"CALL confirmé par {call_signals}/9 indicateurs ({call_ratio:.0%})"
            }
    
    elif put_ratio >= convergence_threshold and put_ratio > call_ratio:
        if validate_signal_quality(df, 'PUT'):
            return {
                'signal': 'PUT',
                'confidence': put_ratio,
                'indicators': put_signals,
                'reason': f"PUT confirmé par {put_signals}/9 indicateurs ({put_ratio:.0%})"
            }
    
    return None

def validate_signal_quality(df, direction):
    """
    Validation supplémentaire de la qualité du signal
    """
    if len(df) < 15:
        return False
    
    last = df.iloc[-1]
    prev_3 = df.iloc[-4]  # Bougie -3 pour voir la tendance
    
    # Vérification de la tendance récente
    if direction == 'CALL':
        # Vérifier que la tendance haussière est cohérente
        recent_prices = df['close'].tail(5).values
        if not all(recent_prices[i] <= recent_prices[i+1] for i in range(4)):
            # Pas tous en hausse, mais au moins la dernière doit être > que la -3
            if last['close'] <= prev_3['close']:
                return False
        
        # Vérifier RSI cohérent
        if last['rsi_5'] > 75:  # Trop acheté
            return False
            
    else:  # PUT
        # Vérifier que la tendance baissière est cohérente
        recent_prices = df['close'].tail(5).values
        if not all(recent_prices[i] >= recent_prices[i+1] for i in range(4)):
            # Pas tous en baisse, mais au moins la dernière doit être < que la -3
            if last['close'] >= prev_3['close']:
                return False
        
        # Vérifier RSI cohérent
        if last['rsi_5'] < 25:  # Trop vendu
            return False
    
    # Vérifier la volatilité
    if last['volatility'] > 0.03:  # Trop volatile
        return False
    
    # Vérifier la taille de bougie
    if last['body_ratio'] < 0.2:  # Doji ou petite bougie
        return False
    
    return True

# ================= STRATÉGIE ADAPTATIVE =================

def adaptive_signal_generation(df, urgency_level=0):
    """
    Génération adaptative de signaux basée sur l'urgence
    urgency_level: 0=normal, 1=modéré, 2=élevé, 3=critique
    """
    if len(df) < 20:
        return None
    
    # Calculer les indicateurs rapides
    df_indicators = compute_ultra_fast_indicators(df)
    
    # ===== MODE NORMAL (haute qualité) =====
    if urgency_level == 0:
        result = analyze_realtime_signal(df_indicators)
        if result and result['confidence'] >= 0.70:
            return result
    
    # ===== MODE MODÉRÉ (qualité moyenne) =====
    elif urgency_level == 1:
        result = analyze_realtime_signal(df_indicators)
        if result and result['confidence'] >= 0.60:
            return result
    
    # ===== MODE ÉLEVÉ (signal rapide) =====
    elif urgency_level == 2:
        last = df_indicators.iloc[-1]
        
        # Règles simplifiées mais efficaces
        if (last['ema_3'] > last['ema_5'] and 
            last['macd'] > last['macd_signal'] and 
            last['rsi_5'] > 55 and 
            last['close'] > last['open']):
            return {
                'signal': 'CALL',
                'confidence': 0.55,
                'indicators': 4,
                'reason': "CALL rapide: EMA↑ MACD↑ RSI↑"
            }
        
        elif (last['ema_3'] < last['ema_5'] and 
              last['macd'] < last['macd_signal'] and 
              last['rsi_5'] < 45 and 
              last['close'] < last['open']):
            return {
                'signal': 'PUT',
                'confidence': 0.55,
                'indicators': 4,
                'reason': "PUT rapide: EMA↓ MACD↓ RSI↓"
            }
    
    # ===== MODE CRITIQUE (dernier recours) =====
    elif urgency_level == 3:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Simple momentum trading
        if last['close'] > prev['close']:
            return {
                'signal': 'CALL',
                'confidence': 0.45,
                'indicators': 1,
                'reason': "CALL momentum: prix↑"
            }
        else:
            return {
                'signal': 'PUT',
                'confidence': 0.45,
                'indicators': 1,
                'reason': "PUT momentum: prix↓"
            }
    
    return None

# ================= INTERFACE DE COMPATIBILITÉ =================
# Ces fonctions gardent la même signature pour le bot

def compute_indicators(df, ema_fast=8, ema_slow=21, rsi_len=14, bb_len=20):
    """
    Interface pour signal_bot - version ultra-rapide
    """
    # Utilise la version accélérée mais garde la compatibilité
    return compute_ultra_fast_indicators(df)

def rule_signal_ultra_strict(df, session_priority=3):
    """
    Mode ultra strict - seulement les meilleurs signaux
    """
    result = analyze_realtime_signal(df)
    if result and result['confidence'] >= 0.75:
        print(f"[ULTRA-STRICT] ✅ Signal haute qualité: {result['signal']} ({result['confidence']:.0%})")
        return result['signal']
    return None

def rule_signal(df, session_priority=3):
    """
    Fonction principale - adaptative selon la session
    """
    # Déterminer l'urgence basée sur la session
    current_hour = datetime.now().hour
    
    if session_priority == 1:
        # Haute priorité - besoin de signaux
        if current_hour in [7, 8, 9, 13, 14, 15]:  # Heures de trading
            urgency = 0  # Haute qualité
        else:
            urgency = 1  # Qualité moyenne
    
    elif session_priority == 2:
        # Priorité moyenne
        urgency = 1
    
    else:
        # Priorité basse - prendre plus de risques
        urgency = 2
    
    # Générer le signal adaptatif
    result = adaptive_signal_generation(df, urgency_level=urgency)
    
    if result:
        print(f"[ADAPTATIVE] 📊 {result['signal']} | Confiance: {result['confidence']:.0%} | {result['reason']}")
        
        # Validation finale
        if result['confidence'] >= 0.60:
            return result['signal']
    
    return None

def rule_signal_original(df, session_priority=3):
    """Alias pour compatibilité"""
    return rule_signal(df, session_priority)

# ================= FONCTIONS UTILITAIRES =================

def calculate_signal_quality_score(df):
    """Score de qualité rapide"""
    if len(df) < 10:
        return 0
    
    last = df.iloc[-1]
    score = 50  # Score de base
    
    # EMA alignment
    if last.get('ema_3', 0) > last.get('ema_5', 0) > last.get('ema_8', 0):
        score += 15
    elif last.get('ema_3', 0) < last.get('ema_5', 0) < last.get('ema_8', 0):
        score += 15
    
    # MACD direction
    if last.get('macd_diff', 0) > 0:
        score += 10
    elif last.get('macd_diff', 0) < 0:
        score += 10
    
    # RSI dans zone optimale
    rsi = last.get('rsi_5', 50)
    if 40 < rsi < 60:
        score += 10
    elif 30 < rsi < 70:
        score += 5
    
    # Bougie significative
    if last.get('body_ratio', 0) > 0.4:
        score += 10
    
    return min(score, 100)

def format_signal_reason(direction, confidence, indicators):
    """Format rapide"""
    last = indicators.iloc[-1] if len(indicators) > 0 else None
    
    if last is not None:
        ema_trend = "EMA↑" if last.get('ema_3', 0) > last.get('ema_5', 0) else "EMA↓"
        macd_trend = "MACD↑" if last.get('macd_diff', 0) > 0 else "MACD↓"
        rsi_val = f"RSI:{last.get('rsi_5', 0):.1f}"
        
        return f"{direction} | {ema_trend} {macd_trend} {rsi_val} | Conf:{confidence:.0%}"
    
    return f"{direction} | Conf:{confidence:.0%}"

def is_kill_zone_optimal(hour_utc):
    """Zones temporelles optimisées"""
    if 7 <= hour_utc < 10:
        return True, "London Open", 5
    if 13 <= hour_utc < 16:
        return True, "NY Open", 5
    if 10 <= hour_utc < 12:
        return True, "London/NY Overlap", 5
    return False, None, 0

# ================= FONCTIONS AVANCÉES =================

def get_signal_with_metadata(df, signal_count=0, total_signals=8):
    """
    Version avancée avec métadonnées
    """
    # Calculer l'urgence basée sur les signaux manquants
    signals_needed = total_signals - signal_count
    
    if signals_needed <= 2:
        urgency = 0  # Haute qualité - peu de signaux nécessaires
    elif signals_needed <= 4:
        urgency = 1  # Qualité moyenne
    elif signals_needed <= 6:
        urgency = 2  # Besoin de signaux
    else:
        urgency = 3  # Besoin critique
    
    result = adaptive_signal_generation(df, urgency_level=urgency)
    
    if result:
        return {
            'direction': result['signal'],
            'confidence': result['confidence'],
            'indicators': result['indicators'],
            'reason': result['reason'],
            'quality': 'HIGH' if result['confidence'] >= 0.70 else 'MEDIUM' if result['confidence'] >= 0.60 else 'LOW'
        }
    
    return None

def get_signal_basic(df):
    """Version ultra-rapide"""
    last = df.iloc[-1] if len(df) > 0 else None
    prev = df.iloc[-2] if len(df) > 1 else None
    
    if last is None or prev is None:
        return None
    
    # Analyse ultra-rapide
    price_up = last['close'] > prev['close']
    volume_up = 'volume' not in last or last['volume'] > prev['volume']
    
    if price_up and volume_up:
        return 'CALL'
    elif not price_up and volume_up:
        return 'PUT'
    
    return None

# ================= ANALYSE PROFONDE =================

def deep_market_analysis(df):
    """
    Analyse approfondie du marché - appelée périodiquement
    """
    if len(df) < 50:
        return "Données insuffisantes"
    
    analysis = []
    
    # Trend analysis
    prices = df['close'].tail(20).values
    trend = "Haussière" if prices[-1] > prices[0] else "Baissière" if prices[-1] < prices[0] else "Neutre"
    analysis.append(f"Tendance: {trend}")
    
    # Volatility analysis
    volatility = df['close'].tail(20).std() / df['close'].tail(20).mean()
    if volatility < 0.01:
        analysis.append("Volatilité: FAIBLE")
    elif volatility < 0.02:
        analysis.append("Volatilité: MODÉRÉE")
    else:
        analysis.append("Volatilité: ÉLEVÉE")
    
    # Support/Resistance levels
    recent_high = df['high'].tail(20).max()
    recent_low = df['low'].tail(20).min()
    current = df['close'].iloc[-1]
    
    position = (current - recent_low) / (recent_high - recent_low) * 100
    if position > 70:
        analysis.append("Position: Proche RESISTANCE")
    elif position < 30:
        analysis.append("Position: Proche SUPPORT")
    else:
        analysis.append("Position: Zone NEUTRE")
    
    return " | ".join(analysis)

# ================= INITIALISATION =================

print("[UTILS] ⚡ Module Saint Graal ULTRA-RAPIDE chargé")
print("[UTILS] 📊 Analyse en temps réel: ACTIVÉE")
print("[UTILS] 🎯 Stratégie: Adaptive Multi-niveaux")

# Export des fonctions
__all__ = [
    'compute_indicators',
    'rule_signal',
    'rule_signal_ultra_strict',
    'rule_signal_original',
    'calculate_signal_quality_score',
    'format_signal_reason',
    'is_kill_zone_optimal',
    'get_signal_with_metadata',
    'get_signal_basic',
    'deep_market_analysis'
]
