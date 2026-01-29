"""
utils.py - STRATÉGIE FOREX M1 - 8 SIGNAUX QUALITÉ MAXIMALE AVEC GARANTIE
Version avec analyse de structure pour éviter d'acheter près des swing highs
Correction bug JSON serialization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIGURATION 8 SIGNAUX QUALITÉ MAX =================

SAINT_GRAAL_CONFIG = {
    # Indicateurs M1 optimisés
    'rsi_period': 7,
    'ema_fast': 5,
    'ema_slow': 13,
    'stoch_period': 5,
    
    # Seuils QUALITÉ MAXIMALE POUR 8 SIGNAUX
    'max_quality': {
        'rsi_overbought': 68,
        'rsi_oversold': 32,
        'adx_min': 25,
        'stoch_overbought': 74,
        'stoch_oversold': 26,
        'min_confluence_points': 7,
        'min_body_ratio': 0.4,
        'max_wick_ratio': 0.4,
        'min_quality_score': 80,
    },
    
    # Seuils QUALITÉ ÉLEVÉE
    'high_quality': {
        'rsi_overbought': 72,
        'rsi_oversold': 28,
        'adx_min': 22,
        'stoch_overbought': 78,
        'stoch_oversold': 22,
        'min_confluence_points': 6,
        'min_body_ratio': 0.35,
        'max_wick_ratio': 0.5,
        'min_quality_score': 70,
    },
    
    # Mode GARANTIE
    'guarantee_mode': {
        'rsi_overbought': 75,
        'rsi_oversold': 25,
        'adx_min': 20,
        'stoch_overbought': 80,
        'stoch_oversold': 20,
        'min_confluence_points': 5,
        'min_body_ratio': 0.3,
        'max_wick_ratio': 0.6,
        'min_quality_score': 60,
    },
    
    # Filtres anti-manipulation
    'anti_manip': {
        'max_wick_ratio': 0.65,
        'max_candle_size_ratio': 2.5,
        'min_ema_spread': 0.0006,
        'max_volatility': 0.04,
        'min_data_quality': 0.8,
    },
    
    # Paramètres structure marché
    'structure': {
        'swing_lookback': 15,
        'near_high_threshold': 0.5,
        'min_trend_strength': 1.5,
        'pattern_lookback': 5,
    },
    
    # Paramètres généraux
    'target_signals': 8,
    'max_signals': 8,
}

# ================= FONCTIONS DE BASE =================

def round_to_m1_candle(dt):
    """Arrondit un datetime à la bougie M1 (minute)"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.replace(second=0, microsecond=0)

def get_next_m1_candle(dt):
    """Retourne le début de la PROCHAINE bougie M1"""
    current_candle = round_to_m1_candle(dt)
    return current_candle + timedelta(minutes=1)

def get_m1_candle_range(dt):
    """Retourne le range de la bougie M1 actuelle"""
    current_candle = round_to_m1_candle(dt)
    start_time = current_candle
    end_time = current_candle + timedelta(minutes=1)
    return start_time, end_time

# ================= ANALYSE STRUCTURE MARCHÉ =================

def analyze_market_structure(df, lookback=15):
    """
    Analyse la structure du marché pour éviter d'acheter au sommet
    Retourne: (structure, strength%)
    """
    if len(df) < lookback + 5:
        return "INSUFFICIENT_DATA", 0
    
    recent = df.tail(lookback).copy()
    highs = recent['high'].values
    lows = recent['low'].values
    
    # Détection des pivots (swing highs/lows)
    swing_highs = []
    swing_lows = []
    
    for i in range(2, len(recent)-2):
        # Swing High
        if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
            highs[i] > highs[i+1] and highs[i] > highs[i+2]):
            swing_highs.append((i, float(highs[i])))
        
        # Swing Low
        if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
            lows[i] < lows[i+1] and lows[i] < lows[i+2]):
            swing_lows.append((i, float(lows[i])))
    
    # Analyser la tendance
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        last_high = swing_highs[-1][1]
        prev_high = swing_highs[-2][1] if len(swing_highs) >= 2 else last_high
        
        last_low = swing_lows[-1][1]
        prev_low = swing_lows[-2][1] if len(swing_lows) >= 2 else last_low
        
        # Uptrend: HH + HL
        if last_high > prev_high and last_low > prev_low:
            structure = "UPTREND"
            strength = float((last_high - prev_high) / prev_high * 100)
        
        # Downtrend: LH + LL
        elif last_high < prev_high and last_low < prev_low:
            structure = "DOWNTREND"
            strength = float((prev_low - last_low) / last_low * 100)
        
        # Range
        else:
            structure = "RANGE"
            strength = 0.0
    else:
        structure = "NO_CLEAR_STRUCTURE"
        strength = 0.0
    
    # Vérifier proximité des swings
    current_price = float(recent.iloc[-1]['close'])
    
    if swing_highs:
        nearest_high = min(swing_highs, key=lambda x: abs(x[0] - len(recent)))
        distance_to_high = float((nearest_high[1] - current_price) / current_price * 100)
        
        if distance_to_high < SAINT_GRAAL_CONFIG['structure']['near_high_threshold']:
            structure += "_NEAR_HIGH"
    
    if swing_lows:
        nearest_low = min(swing_lows, key=lambda x: abs(x[0] - len(recent)))
        distance_to_low = float((current_price - nearest_low[1]) / current_price * 100)
        
        if distance_to_low < SAINT_GRAAL_CONFIG['structure']['near_high_threshold']:
            structure += "_NEAR_LOW"
    
    return structure, strength

def is_near_swing_high(df, lookback=20):
    """
    Vérifie si le prix est proche d'un swing high récent
    Retourne: (is_near, distance%)
    """
    if len(df) < lookback:
        return False, 0.0
    
    recent = df.tail(lookback)
    highs = recent['high'].values
    
    # Trouver le swing high récent
    swing_high_idx = np.argmax(highs)
    swing_high = float(highs[swing_high_idx])
    
    current_price = float(df.iloc[-1]['close'])
    distance = float((swing_high - current_price) / swing_high * 100)
    
    threshold = SAINT_GRAAL_CONFIG['structure']['near_high_threshold']
    is_near = distance < threshold
    
    return bool(is_near), distance  # Convertir en bool pour JSON

def detect_retest_pattern(df, lookback=5):
    """
    Détecte les patterns de retest (rouge → vert → vert)
    Retourne: pattern_type, confidence
    """
    if len(df) < lookback + 1:
        return "NO_PATTERN", 0
    
    confidence = 0
    pattern_type = "NO_PATTERN"
    
    # Bougies nécessaires pour le pattern
    if len(df) >= 4:
        # Indices: -4 = bougie rouge, -3 et -2 = vertes, -1 = actuelle
        idx_red = -4
        idx_green1 = -3
        idx_green2 = -2
        
        # Vérifier le pattern rouge → vert → vert
        red_candle = df.iloc[idx_red]
        green1_candle = df.iloc[idx_green1]
        green2_candle = df.iloc[idx_green2]
        
        is_red = bool(red_candle['close'] < red_candle['open'])
        is_green1 = bool(green1_candle['close'] > green1_candle['open'])
        is_green2 = bool(green2_candle['close'] > green2_candle['open'])
        
        if is_red and is_green1 and is_green2:
            # Pattern détecté
            pattern_type = "RETEST_PATTERN"
            
            # Calculer la confiance
            # 1. Les vertes doivent être plus petites que la rouge
            red_size = float(abs(red_candle['close'] - red_candle['open']))
            green1_size = float(abs(green1_candle['close'] - green1_candle['open']))
            green2_size = float(abs(green2_candle['close'] - green2_candle['open']))
            
            if green1_size < red_size and green2_size < red_size:
                confidence += 30
            
            # 2. La bougie actuelle doit être sous le haut de la rouge
            current_candle = df.iloc[-1]
            if float(current_candle['high']) < float(red_candle['high']):
                confidence += 30
            
            # 3. Les vertes doivent fermer dans la moitié supérieure de la rouge
            red_body_mid = float((red_candle['open'] + red_candle['close']) / 2)
            if float(green1_candle['close']) > red_body_mid:
                confidence += 20
            if float(green2_candle['close']) > red_body_mid:
                confidence += 20
    
    return pattern_type, confidence

# ================= INDICATEURS QUALITÉ MAX =================

def compute_saint_graal_indicators(df):
    """
    Calcule les indicateurs pour qualité maximale
    """
    df = df.copy()
    
    # Assurer les types numériques
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remplir les NaN
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    config = SAINT_GRAAL_CONFIG
    
    # ===== 1. INDICATEURS PRINCIPAUX =====
    
    # EMA 5 & 13
    df['ema_5'] = EMAIndicator(close=df['close'], window=config['ema_fast']).ema_indicator()
    df['ema_13'] = EMAIndicator(close=df['close'], window=config['ema_slow']).ema_indicator()
    df['ema_spread'] = abs(df['ema_5'] - df['ema_13']) / df['close']
    df['ema_trend'] = (df['ema_5'] > df['ema_13']).astype(int)
    
    # RSI 7
    df['rsi_7'] = RSIIndicator(close=df['close'], window=config['rsi_period']).rsi()
    df['rsi_trend'] = (df['rsi_7'] > 50).astype(int)
    
    # Stochastique 5,3,3
    stoch = StochasticOscillator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=config['stoch_period'],
        smooth_window=3
    )
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df['stoch_trend'] = (df['stoch_k'] > df['stoch_d']).astype(int)
    
    # ADX 10
    adx = ADXIndicator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=10
    )
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()
    df['adx_trend'] = (df['adx_pos'] > df['adx_neg']).astype(int)
    
    # ===== 2. VOLATILITÉ ET RISQUE =====
    
    # ATR 10
    df['atr_10'] = AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=10
    ).average_true_range()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ===== 3. PRICE ACTION DÉTAILLÉE =====
    
    # Bougie actuelle
    df['candle_body'] = df['close'] - df['open']
    df['candle_size'] = df['high'] - df['low']
    df['body_ratio'] = abs(df['candle_body']) / df['candle_size'].replace(0, 0.00001)
    
    # Mèches
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['total_wick'] = df['upper_wick'] + df['lower_wick']
    df['wick_ratio'] = df['total_wick'] / abs(df['candle_body']).replace(0, 0.00001)
    
    # Tendance prix
    df['price_trend'] = (df['close'] > df['open']).astype(int)
    df['momentum_1'] = df['close'].pct_change(1) * 100
    
    # ===== 4. QUALITÉ ET CONVERGENCE =====
    
    # Score de convergence (5 indicateurs alignés)
    df['convergence_raw'] = (
        df['ema_trend'] + 
        df['rsi_trend'] + 
        df['stoch_trend'] + 
        df['adx_trend'] + 
        df['price_trend']
    )
    df['convergence_score'] = df['convergence_raw'] / 5.0
    
    # Qualité globale
    df['data_quality'] = (
        (df['close'].notna()).astype(int) +
        (df['bb_width'] < 0.04).astype(int) +
        (df['wick_ratio'] < 0.6).astype(int) +
        (df['body_ratio'] > 0.2).astype(int)
    ) / 4.0
    
    # Remplir les derniers NaN
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    # Convertir tous les types numériques pour éviter les problèmes
    for col in df.columns:
        if df[col].dtype in ['float32', 'float64']:
            df[col] = df[col].astype('float64')
        elif df[col].dtype in ['int32', 'int64']:
            df[col] = df[col].astype('int64')
    
    return df

def calculate_signal_quality_score(df):
    """
    Calcule un score de qualité global du signal (0-100)
    """
    if len(df) < 20:
        return 0
    
    last = df.iloc[-1]
    score = 0
    
    # Convergence (30 points max)
    convergence = last.get('convergence_score', 0.5)
    score += float(convergence) * 30
    
    # Force de la tendance (25 points)
    adx = last.get('adx', 0)
    if adx > 30:
        score += 25
    elif adx > 25:
        score += 20
    elif adx > 20:
        score += 15
    elif adx > 15:
        score += 10
    
    # Alignement des indicateurs (20 points)
    aligned_indicators = 0
    if last.get('ema_trend', 0) == 1:
        aligned_indicators += 1
    if last.get('rsi_trend', 0) == 1:
        aligned_indicators += 1
    if last.get('stoch_trend', 0) == 1:
        aligned_indicators += 1
    if last.get('adx_trend', 0) == 1:
        aligned_indicators += 1
    
    score += (aligned_indicators / 4) * 20
    
    # Volatilité contrôlée (15 points)
    bb_width = last.get('bb_width', 0)
    if 0.01 < bb_width < 0.03:
        score += 15
    elif 0.005 < bb_width < 0.04:
        score += 10
    elif 0.002 < bb_width < 0.05:
        score += 5
    
    # Qualité de la bougie (10 points)
    body_ratio = last.get('body_ratio', 0)
    wick_ratio = last.get('wick_ratio', 0)
    
    if body_ratio > 0.4 and wick_ratio < 0.3:
        score += 10
    elif body_ratio > 0.3 and wick_ratio < 0.4:
        score += 5
    
    return min(score, 100)

# ================= FILTRES ANTI-MANIPULATION =================

def check_anti_manipulation(df, strict_mode=True):
    """Vérifie les conditions anti-manipulation"""
    if len(df) < 15:
        return False, "Données insuffisantes"
    
    last = df.iloc[-1]
    anti = SAINT_GRAAL_CONFIG['anti_manip']
    
    # 1. Qualité des données minimale
    min_quality = 0.85 if strict_mode else anti['min_data_quality']
    if last['data_quality'] < min_quality:
        return False, f"Qualité données faible: {last['data_quality']:.2f}"
    
    # 2. Mèches suspectes
    max_wick = anti['max_wick_ratio'] * (0.9 if strict_mode else 1.0)
    if last['wick_ratio'] > max_wick:
        return False, f"Mèche suspecte: {last['wick_ratio']:.1%}"
    
    # 3. Bougie anormale
    if last['candle_size'] > last['atr_10'] * 3:
        return False, f"Bougie trop grande: {last['candle_size']:.5f} > ATRx3"
    
    # 4. Volatilité excessive
    if last['bb_width'] > anti['max_volatility']:
        return False, f"Volatilité excessive: {last['bb_width']:.3%}"
    
    # 5. EMA trop plates
    if last['ema_spread'] < anti['min_ema_spread']:
        return False, f"EMA plates: écart {last['ema_spread']:.5%}"
    
    # 6. ADX trop faible (sauf mode non-strict)
    if strict_mode and last['adx'] < 15:
        return False, f"ADX faible: {last['adx']:.1f}"
    
    return True, "OK"

# ================= STRATÉGIE 8 SIGNAUX AVEC ANALYSE STRUCTURE =================

def rule_signal_saint_graal_with_guarantee(df, signal_count=0, total_signals_needed=8):
    """
    STRATÉGIE - 8 signaux qualité maximale avec analyse de structure
    """
    
    if len(df) < 30:
        print("[STRATEGIE] ⚠️ Données insuffisantes")
        return None
    
    target_signals = SAINT_GRAAL_CONFIG['target_signals']
    
    print(f"\n[STRATEGIE] 🎯 Signal #{signal_count+1}/{target_signals}")
    
    # ===== ANALYSE STRUCTURE MARCHÉ =====
    structure, strength = analyze_market_structure(df, 15)
    is_near_high, distance_to_high = is_near_swing_high(df, 20)
    pattern_type, pattern_confidence = detect_retest_pattern(df, 5)
    
    print(f"[STRUCTURE] {structure} (force: {strength:.1f}%)")
    print(f"[STRUCTURE] Near high: {is_near_high} ({distance_to_high:.2f}%)")
    print(f"[PATTERN] {pattern_type} (confiance: {pattern_confidence}%)")
    
    # ===== 1. QUALITÉ MAXIMALE (signaux 1-6) =====
    if signal_count < 6:
        print(f"[STRATEGIE] 🔵 Mode QUALITÉ MAXIMALE")
        
        manip_ok, manip_reason = check_anti_manipulation(df, strict_mode=True)
        if not manip_ok:
            print(f"[STRATEGIE] ⚠️ Anti-manip: {manip_reason}")
        
        max_quality_signal = rule_signal_max_quality(df, structure, is_near_high, pattern_type, pattern_confidence)
        if max_quality_signal:
            quality_score = calculate_signal_quality(df, max_quality_signal)
            min_score = SAINT_GRAAL_CONFIG['max_quality']['min_quality_score']
            
            if quality_score >= min_score:
                print(f"[STRATEGIE] ✅ Signal QUALITÉ MAX #{signal_count+1} | Score: {quality_score}/100")
                return {
                    'signal': max_quality_signal,
                    'mode': 'MAX_QUALITY',
                    'quality': 'EXCELLENT',
                    'score': quality_score,
                    'structure_info': {
                        'market_structure': structure,
                        'near_high': is_near_high,
                        'distance_to_high': float(distance_to_high),
                        'pattern': pattern_type,
                        'pattern_confidence': pattern_confidence,
                        'trend_strength': float(strength)
                    }
                }
    
    # ===== 2. QUALITÉ ÉLEVÉE =====
    print(f"[STRATEGIE] 🟡 Mode QUALITÉ ÉLEVÉE")
    
    manip_ok, manip_reason = check_anti_manipulation(df, strict_mode=False)
    if not manip_ok:
        print(f"[STRATEGIE] ⚠️ Anti-manip (toléré): {manip_reason}")
    
    high_quality_signal = rule_signal_high_quality(df, structure, is_near_high)
    if high_quality_signal:
        quality_score = calculate_signal_quality_for_mode(df, high_quality_signal, 'HIGH_QUALITY')
        min_score = SAINT_GRAAL_CONFIG['high_quality']['min_quality_score']
        
        if quality_score >= min_score:
            print(f"[STRATEGIE] ✅ Signal QUALITÉ ÉLEVÉE #{signal_count+1} | Score: {quality_score}/100")
            return {
                'signal': high_quality_signal,
                'mode': 'HIGH_QUALITY',
                'quality': 'HIGH',
                'score': quality_score,
                'structure_info': {
                    'market_structure': structure,
                    'near_high': is_near_high,
                    'distance_to_high': float(distance_to_high),
                    'pattern': pattern_type,
                    'pattern_confidence': pattern_confidence,
                    'trend_strength': float(strength)
                }
            }
    
    # ===== 3. MODE GARANTIE =====
    print(f"[STRATEGIE] 🟠 Mode GARANTIE")
    
    guarantee_signal = rule_signal_guarantee_mode(df, structure, is_near_high)
    if guarantee_signal:
        quality_score = calculate_signal_quality_for_mode(df, guarantee_signal, 'GUARANTEE')
        min_score = SAINT_GRAAL_CONFIG['guarantee_mode']['min_quality_score']
        
        if quality_score >= min_score:
            print(f"[STRATEGIE] ✅ Signal GARANTIE #{signal_count+1} | Score: {quality_score}/100")
            return {
                'signal': guarantee_signal,
                'mode': 'GUARANTEE',
                'quality': 'ACCEPTABLE',
                'score': quality_score,
                'structure_info': {
                    'market_structure': structure,
                    'near_high': is_near_high,
                    'distance_to_high': float(distance_to_high),
                    'pattern': pattern_type,
                    'pattern_confidence': pattern_confidence,
                    'trend_strength': float(strength)
                }
            }
    
    # ===== 4. FORCÉ =====
    print(f"[STRATEGIE] ⚡ Mode FORCÉ pour garantir le signal #{signal_count+1}")
    
    forced_signal = force_signal_with_structure(df, structure, is_near_high)
    if forced_signal:
        print(f"[STRATEGIE] ✅ Signal FORCÉ #{signal_count+1}")
        return {
            'signal': forced_signal,
            'mode': 'FORCED',
            'quality': 'MINIMUM',
            'score': 50,
            'structure_info': {
                'market_structure': structure,
                'near_high': is_near_high,
                'distance_to_high': float(distance_to_high),
                'pattern': pattern_type,
                'pattern_confidence': pattern_confidence,
                'trend_strength': float(strength)
            }
        }
    
    print(f"[STRATEGIE] ❌ ERREUR: Impossible de générer un signal")
    return None

def rule_signal_max_quality(df, structure, is_near_high, pattern_type, pattern_confidence):
    """QUALITÉ MAXIMALE avec analyse de structure"""
    if len(df) < 25:
        return None
    
    config = SAINT_GRAAL_CONFIG['max_quality']
    last = df.iloc[-1]
    
    # === FILTRES ABSOLUS ===
    if last['data_quality'] < 0.85:
        return None
    if last['adx'] < config['adx_min']:
        return None
    if last['convergence_raw'] < 3:
        return None
    
    # === ANALYSE STRUCTURE ===
    call_penalty = 0.0
    put_bonus = 0.0
    
    # 1. Si près d'un swing high, pénalité pour CALL
    if is_near_high:
        call_penalty -= 2.0
        put_bonus += 1.5
    
    # 2. Si pattern de retest, forte pénalité CALL
    if pattern_type == "RETEST_PATTERN" and pattern_confidence > 50:
        call_penalty -= 3.0
        put_bonus += 2.0
    
    # 3. Si UPTREND mature, prudence
    if "UPTREND" in structure and "NEAR_HIGH" in structure:
        call_penalty -= 1.5
        put_bonus += 1.0
    
    # 4. Si DOWNTREND et près d'un low, bonus CALL
    if "DOWNTREND" in structure and "NEAR_LOW" in structure:
        call_penalty += 1.5
    
    # === ANALYSE CALL ===
    call_points = 0.0
    
    # 1. EMA alignées
    if last['ema_5'] > last['ema_13'] and last['ema_spread'] > 0.001:
        call_points += 3.0
    elif last['ema_5'] > last['ema_13']:
        call_points += 2.0
    
    # 2. RSI optimal
    if 58 < last['rsi_7'] < config['rsi_overbought']:
        call_points += 2.0
    elif 55 < last['rsi_7'] < 70:
        call_points += 1.0
    
    # 3. Stochastique
    if last['stoch_k'] > last['stoch_d'] and last['stoch_k'] < config['stoch_overbought']:
        call_points += 2.0
    elif last['stoch_k'] > last['stoch_d']:
        call_points += 1.0
    
    # 4. ADX
    if last['adx_pos'] > last['adx_neg'] and last['adx_pos'] > 25:
        call_points += 2.0
    elif last['adx_pos'] > last['adx_neg']:
        call_points += 1.0
    
    # 5. Bougie haussière
    if last['price_trend'] == 1:
        call_points += 1.0
    
    # 6. Qualité bougie
    if last['body_ratio'] > config['min_body_ratio']:
        call_points += 0.5
    if last['wick_ratio'] < config['max_wick_ratio']:
        call_points += 0.5
    
    # Appliquer pénalités structure
    call_points += call_penalty
    
    # === ANALYSE PUT ===
    put_points = 0.0
    
    # 1. EMA alignées
    if last['ema_5'] < last['ema_13'] and last['ema_spread'] > 0.001:
        put_points += 3.0
    elif last['ema_5'] < last['ema_13']:
        put_points += 2.0
    
    # 2. RSI optimal
    if config['rsi_oversold'] < last['rsi_7'] < 42:
        put_points += 2.0
    elif 30 < last['rsi_7'] < 45:
        put_points += 1.0
    
    # 3. Stochastique
    if last['stoch_k'] < last['stoch_d'] and last['stoch_k'] > config['stoch_oversold']:
        put_points += 2.0
    elif last['stoch_k'] < last['stoch_d']:
        put_points += 1.0
    
    # 4. ADX
    if last['adx_neg'] > last['adx_pos'] and last['adx_neg'] > 25:
        put_points += 2.0
    elif last['adx_neg'] > last['adx_pos']:
        put_points += 1.0
    
    # 5. Bougie baissière
    if last['price_trend'] == 0:
        put_points += 1.0
    
    # 6. Qualité bougie
    if last['body_ratio'] > config['min_body_ratio']:
        put_points += 0.5
    if last['wick_ratio'] < config['max_wick_ratio']:
        put_points += 0.5
    
    # Appliquer bonus structure
    put_points += put_bonus
    
    # === DÉCISION ===
    min_points = config['min_confluence_points']
    
    print(f"[STRUCTURE] CALL: {call_points:.1f}, PUT: {put_points:.1f}, Min: {min_points}")
    print(f"[STRUCTURE] Call penalty: {call_penalty:.1f}, Put bonus: {put_bonus:.1f}")
    
    # Augmenter seuil pour CALL si près d'un high
    min_points_call = min_points + (1 if is_near_high else 0)
    
    if call_points >= min_points_call and call_points > put_points:
        return 'CALL'
    if put_points >= min_points and put_points > call_points:
        return 'PUT'
    
    # Si égalité, décider par structure
    if call_points == put_points and call_points >= min_points - 1:
        if is_near_high:
            return 'PUT'
        elif "DOWNTREND" in structure and "NEAR_LOW" in structure:
            return 'CALL'
        else:
            if last['rsi_7'] > 50:
                return 'CALL'
            else:
                return 'PUT'
    
    return None

def rule_signal_high_quality(df, structure, is_near_high):
    """QUALITÉ ÉLEVÉE avec analyse structure"""
    if len(df) < 20:
        return None
    
    config = SAINT_GRAAL_CONFIG['high_quality']
    last = df.iloc[-1]
    
    if last['data_quality'] < 0.75:
        return None
    if last['adx'] < config['adx_min']:
        return None
    
    # Ajustements structure
    call_adj = 0
    put_adj = 0
    
    if is_near_high:
        call_adj -= 1
        put_adj += 1
    
    # Conditions CALL
    call_conditions = [
        bool(last['ema_5'] > last['ema_13']),
        bool(last['rsi_7'] > 52),
        bool(last['stoch_k'] > last['stoch_d']),
        bool(last['adx_pos'] > last['adx_neg']),
        bool(last['price_trend'] == 1),
        bool(last['body_ratio'] > config['min_body_ratio']),
        bool(last['wick_ratio'] < config['max_wick_ratio']),
    ]
    
    # Conditions PUT
    put_conditions = [
        bool(last['ema_5'] < last['ema_13']),
        bool(last['rsi_7'] < 48),
        bool(last['stoch_k'] < last['stoch_d']),
        bool(last['adx_neg'] > last['adx_pos']),
        bool(last['price_trend'] == 0),
        bool(last['body_ratio'] > config['min_body_ratio']),
        bool(last['wick_ratio'] < config['max_wick_ratio']),
    ]
    
    call_score = sum(call_conditions) + call_adj
    put_score = sum(put_conditions) + put_adj
    
    min_conditions = config['min_confluence_points']
    
    if call_score >= min_conditions and call_score > put_score:
        return 'CALL'
    if put_score >= min_conditions and put_score > call_score:
        return 'PUT'
    
    return None

def rule_signal_guarantee_mode(df, structure, is_near_high):
    """MODE GARANTIE avec analyse structure"""
    if len(df) < 15:
        return None
    
    config = SAINT_GRAAL_CONFIG['guarantee_mode']
    last = df.iloc[-1]
    
    if last['data_quality'] < 0.65:
        return None
    
    # Ajustements structure forts
    call_adj = 0
    put_adj = 0
    
    if is_near_high:
        call_adj -= 2
        put_adj += 2
    
    # Conditions simples
    call_conditions = [
        bool(last['ema_5'] > last['ema_13']),
        bool(last['rsi_7'] > 50),
        bool(last['stoch_k'] > last['stoch_d']),
        bool(last['price_trend'] == 1),
    ]
    
    put_conditions = [
        bool(last['ema_5'] < last['ema_13']),
        bool(last['rsi_7'] < 50),
        bool(last['stoch_k'] < last['stoch_d']),
        bool(last['price_trend'] == 0),
    ]
    
    call_score = sum(call_conditions) + call_adj
    put_score = sum(put_conditions) + put_adj
    
    min_conditions = config['min_confluence_points']
    
    if call_score >= min_conditions:
        return 'CALL'
    if put_score >= min_conditions:
        return 'PUT'
    
    # Dernier recours
    if is_near_high:
        return 'PUT'
    else:
        if last['rsi_7'] > 50:
            return 'CALL'
        else:
            return 'PUT'

def force_signal_with_structure(df, structure, is_near_high):
    """Force un signal en considérant la structure"""
    if len(df) < 10:
        return None
    
    last = df.iloc[-1]
    
    # Si près d'un swing high, forcer PUT
    if is_near_high:
        return 'PUT'
    
    # Si DOWNTREND et près d'un low, forcer CALL
    if "DOWNTREND" in structure and "NEAR_LOW" in structure:
        return 'CALL'
    
    # Sinon basé sur RSI + EMA
    if last['rsi_7'] > 50 and last['ema_5'] > last['ema_13']:
        return 'CALL'
    elif last['rsi_7'] < 50 and last['ema_5'] < last['ema_13']:
        return 'PUT'
    
    # Default
    if last['rsi_7'] > 50:
        return 'CALL'
    else:
        return 'PUT'

# ================= FONCTIONS DE QUALITÉ =================

def calculate_signal_quality(df, direction):
    """Calcule un score de qualité (0-100)"""
    if len(df) < 20:
        return 0
    
    last = df.iloc[-1]
    score = 0
    
    # Convergence (40 points)
    convergence = last['convergence_raw']
    if convergence >= 4:
        score += 40
    elif convergence >= 3:
        score += 30
    elif convergence >= 2:
        score += 20
    
    # Force tendance (30 points)
    if last['adx'] > 30:
        score += 30
    elif last['adx'] > 25:
        score += 25
    elif last['adx'] > 20:
        score += 20
    elif last['adx'] > 15:
        score += 15
    
    # Qualité bougie (20 points)
    if last['body_ratio'] > 0.5:
        score += 15
    elif last['body_ratio'] > 0.4:
        score += 10
    elif last['body_ratio'] > 0.3:
        score += 5
    
    if last['wick_ratio'] < 0.3:
        score += 5
    
    # Alignement (10 points)
    aligned = 0
    if direction == 'CALL':
        if last['ema_5'] > last['ema_13']:
            aligned += 1
        if last['rsi_trend'] == 1:
            aligned += 1
        if last['stoch_trend'] == 1:
            aligned += 1
        if last['adx_trend'] == 1:
            aligned += 1
    else:
        if last['ema_5'] < last['ema_13']:
            aligned += 1
        if last['rsi_trend'] == 0:
            aligned += 1
        if last['stoch_trend'] == 0:
            aligned += 1
        if last['adx_trend'] == 0:
            aligned += 1
    
    score += (aligned / 4) * 10
    
    return min(score, 100)

def calculate_signal_quality_for_mode(df, direction, mode):
    """Calcule le score de qualité adapté au mode"""
    last = df.iloc[-1]
    score = 0
    
    base_scores = {
        'MAX_QUALITY': 80,
        'HIGH_QUALITY': 70,
        'GUARANTEE': 60
    }
    
    score = base_scores.get(mode, 60)
    
    # Bonus indicateurs
    if direction == 'CALL':
        if last['ema_5'] > last['ema_13']:
            score += 5
        if last['rsi_7'] > 55:
            score += 5
        if last['stoch_k'] > last['stoch_d']:
            score += 5
        if last['adx_pos'] > last['adx_neg']:
            score += 5
    else:
        if last['ema_5'] < last['ema_13']:
            score += 5
        if last['rsi_7'] < 45:
            score += 5
        if last['stoch_k'] < last['stoch_d']:
            score += 5
        if last['adx_neg'] > last['adx_pos']:
            score += 5
    
    # Bonus convergence
    if last['convergence_raw'] >= 3:
        score += 5
    
    return min(score, 100)

def format_signal_reason(direction, confidence, indicators):
    """Formate la raison du signal"""
    last = indicators.iloc[-1]
    quality_score = calculate_signal_quality_score(indicators)
    
    reason_parts = [f"🎯 {direction}"]
    
    # Évaluation qualité
    if quality_score >= 90:
        reason_parts.append("⭐⭐⭐⭐⭐")
    elif quality_score >= 85:
        reason_parts.append("⭐⭐⭐⭐")
    elif quality_score >= 80:
        reason_parts.append("⭐⭐⭐")
    elif quality_score >= 75:
        reason_parts.append("⭐⭐")
    else:
        reason_parts.append("⭐")
    
    # Indicateurs clés
    reason_parts.append(f"RSI7: {last['rsi_7']:.1f}")
    reason_parts.append(f"ADX: {last['adx']:.1f}")
    
    # Convergence
    if last['convergence_raw'] >= 4:
        reason_parts.append("CONV: EXCELLENTE")
    elif last['convergence_raw'] >= 3:
        reason_parts.append("CONV: BONNE")
    
    # Confiance
    reason_parts.append(f"CONF: {int(confidence)}%")
    
    return " | ".join(reason_parts)

# ================= FONCTIONS DE COMPATIBILITÉ =================

def compute_indicators(df, ema_fast=8, ema_slow=21, rsi_len=14, bb_len=20):
    """Wrapper pour compatibilité"""
    return compute_saint_graal_indicators(df)

def rule_signal(df):
    """Version par défaut"""
    result = rule_signal_saint_graal_with_guarantee(df, signal_count=0, total_signals_needed=8)
    return result['signal'] if result else None

def get_signal_with_metadata(df, signal_count=0, total_signals=8):
    """Fonction principale pour le bot - CORRIGÉ POUR JSON"""
    result = rule_signal_saint_graal_with_guarantee(df, signal_count, total_signals)
    
    if result:
        # Analyser structure pour rapport
        structure, strength = analyze_market_structure(df, 15)
        is_near_high, distance = is_near_swing_high(df, 20)
        pattern_type, pattern_conf = detect_retest_pattern(df, 5)
        
        # Construire raison avec info structure
        base_reason = format_signal_reason(result['signal'], result['score'], df)
        
        # Ajouter warnings structure
        warnings = []
        if is_near_high and result['signal'] == 'CALL':
            warnings.append(f"⚠️ NEAR HIGH ({distance:.1f}%)")
        if pattern_type == "RETEST_PATTERN" and pattern_conf > 50:
            warnings.append(f"⚠️ RETEST PATTERN")
        
        if warnings:
            reason = base_reason + " | " + " | ".join(warnings)
        else:
            reason = base_reason
        
        # Préparer les données structure pour JSON
        structure_info = {
            'market_structure': str(structure),
            'strength': float(strength),
            'near_swing_high': bool(is_near_high),
            'distance_to_high': float(distance),
            'pattern_detected': str(pattern_type),
            'pattern_confidence': int(pattern_conf)
        }
        
        mode_display = {
            'MAX_QUALITY': '🔵 STRICT',
            'HIGH_QUALITY': '🟡 HIGH',
            'GUARANTEE': '🟠 GARANTIE',
            'FORCED': '⚡ FORCED'
        }.get(result['mode'], '⚪ STANDARD')
        
        quality_text = {
            'EXCELLENT': 'EXCELLENTE',
            'HIGH': 'ÉLEVÉE',
            'ACCEPTABLE': 'ACCEPTABLE',
            'MINIMUM': 'MINIMUM'
        }.get(result['quality'], 'STANDARD')
        
        return {
            'direction': result['signal'],
            'mode': result['mode'],
            'quality': result['quality'],
            'score': float(result['score']),  # Convertir en float pour JSON
            'reason': reason,
            'structure_info': structure_info,
            'session_info': {
                'current_signal': signal_count + 1,
                'total_signals': total_signals,
                'mode_used': result['mode'],
                'quality_level': result['quality']
            }
        }
    
    # Fallback absolu
    last = df.iloc[-1] if len(df) > 0 else None
    if last is not None:
        forced_direction = 'CALL' if last.get('rsi_7', 50) > 50 else 'PUT'
    else:
        forced_direction = 'CALL'
    
    return {
        'direction': forced_direction,
        'mode': 'FALLBACK',
        'quality': 'MINIMUM',
        'score': 50.0,
        'reason': f"FALLBACK {forced_direction} - Aucun signal trouvé",
        'structure_info': {
            'market_structure': 'UNKNOWN',
            'strength': 0.0,
            'near_swing_high': False,
            'distance_to_high': 0.0,
            'pattern_detected': 'NO_PATTERN',
            'pattern_confidence': 0
        },
        'session_info': {
            'current_signal': signal_count + 1,
            'total_signals': total_signals,
            'mode_used': 'FALLBACK',
            'quality_level': 'MINIMUM'
        }
    }

# ================= FONCTIONS DE TIMING =================

def is_kill_zone_optimal(hour_utc):
    """Heures de trading optimales"""
    if 7 <= hour_utc < 10:
        return True, "London Open", 10
    if 13 <= hour_utc < 16:
        return True, "NY Open", 9
    if 10 <= hour_utc < 12:
        return True, "London/NY Overlap", 8
    if 1 <= hour_utc < 4:
        return True, "Asia Close", 6
    
    return False, "Heure non optimale", 3
