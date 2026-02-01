"""
utils.py - STRATÉGIE FOREX M1 PRO - ARCHITECTURE TRADER ULTIME
Version 4.0 : Approche hiérarchique + Compatibilité maintenue
Objectif : 8 signaux avec qualité dégressive contrôlée
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIGURATION ULTIME =================

SAINT_GRAAL_CONFIG = {
    # Indicateurs M1
    'rsi_period': 7,
    'ema_fast': 5,
    'ema_slow': 13,
    'stoch_period': 5,
    
    # Seuils qualité
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
    
    # CORRECTION 1: Règles RANGE strictes
    'range_rules': {
        'max_score_range': 15,
        'require_zone_in_range': True,
        'require_candle_in_range': True,
        'min_body_range': 0.35,
    },
    
    # CORRECTION 2: Stochastic turning strict
    'stoch_turning': {
        'min_consecutive_bars': 2,
        'min_stoch_move': 3.0,
        'confirmation_bars': 1,
    },
    
    # CORRECTION 3: MACD comme anti-signal
    'macd_filter': {
        'use_as_veto': True,
        'veto_penalty': 15,
        'veto_threshold': 0.0002,
    },
    
    # CORRECTION 4: Fallback pro
    'fallback_rules': {
        'min_aligned_indicators': 2,
        'no_trade_middle_range': True,
        'require_stoch_aligned': True,
        'max_rsi_extreme': 70,
        'min_rsi_extreme': 30,
    },
    
    # Détection structure corrigée
    'structure': {
        'swing_lookback': 20,
        'min_swings': 3,
        'min_swing_height_pips': 2,
        'trend_strength_min': 0.5,
        'near_high_threshold': 0.5,
    },
    
    # Zones S/R améliorées
    'zone_config': {
        'lookback_bars': 100,
        'min_touches': 2,
        'cluster_tolerance_atr_multiplier': 0.5,
        'max_zone_width_pips': 15,
        'recent_zone_age_bars': 30,
    },
    
    # Paramètres généraux
    'target_signals': 8,
    'max_signals': 8,
    'quality_thresholds': {
        'PRO_MAX': 85,
        'PRO_HIGH': 75,
        'PRO_STANDARD': 65,
        'PRO_MINIMUM': 55,
    }
}

# ================= FONCTIONS DE BASE =================

def round_to_m1_candle(dt):
    """Arrondit à la bougie M1"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.replace(second=0, microsecond=0)

def get_next_m1_candle(dt):
    """Début de la prochaine bougie M1"""
    current_candle = round_to_m1_candle(dt)
    return current_candle + timedelta(minutes=1)

def get_m1_candle_range(dt):
    """Range de la bougie M1 actuelle"""
    current_candle = round_to_m1_candle(dt)
    start_time = current_candle
    end_time = current_candle + timedelta(minutes=1)
    return start_time, end_time

# ================= DÉTECTION STRUCTURE =================

def analyze_market_structure(df, lookback=15):
    """
    Détection VRAIE de la structure : HH+HL pour uptrend, LH+LL pour downtrend
    Version corrigée : ne confond plus range/chaos avec downtrend
    """
    if len(df) < lookback + 5:
        return "INSUFFICIENT_DATA", 0.0, [], []
    
    recent = df.tail(lookback).copy()
    highs = recent['high'].values
    lows = recent['low'].values
    
    swing_highs = []
    swing_lows = []
    
    # Détection swings robuste
    for i in range(3, len(recent)-3):
        # Swing High
        if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
            highs[i] > highs[i-3] and
            highs[i] > highs[i+1] and highs[i] > highs[i+2] and
            highs[i] > highs[i+3]):
            
            min_height = recent['close'].mean() * 0.0002  # 2 pips min
            if highs[i] - max(highs[i-3], highs[i-2], highs[i-1],
                             highs[i+1], highs[i+2], highs[i+3]) > min_height:
                swing_highs.append({
                    'index': i,
                    'price': float(highs[i]),
                })
        
        # Swing Low
        if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
            lows[i] < lows[i-3] and
            lows[i] < lows[i+1] and lows[i] < lows[i+2] and
            lows[i] < lows[i+3]):
            
            min_depth = recent['close'].mean() * 0.0002
            if min(lows[i-3], lows[i-2], lows[i-1],
                  lows[i+1], lows[i+2], lows[i+3]) - lows[i] > min_depth:
                swing_lows.append({
                    'index': i,
                    'price': float(lows[i]),
                })
    
    # === LOGIQUE CORRIGÉE : DÉTECTION VRAIE TREND ===
    structure = "RANGE"
    trend_strength = 0.0
    
    if len(swing_highs) >= 3 and len(swing_lows) >= 3:
        # Prendre les 3 derniers swings
        recent_highs = sorted(swing_highs, key=lambda x: x['index'])[-3:]
        recent_lows = sorted(swing_lows, key=lambda x: x['index'])[-3:]
        
        high_prices = [h['price'] for h in recent_highs]
        low_prices = [l['price'] for l in recent_lows]
        
        # 1. Vérifier UPTREND : Higher Highs AND Higher Lows
        is_hh = all(high_prices[i] > high_prices[i-1] for i in range(1, len(high_prices)))
        is_hl = all(low_prices[i] > low_prices[i-1] for i in range(1, len(low_prices)))
        
        # 2. Vérifier DOWNTREND : Lower Highs AND Lower Lows
        is_lh = all(high_prices[i] < high_prices[i-1] for i in range(1, len(high_prices)))
        is_ll = all(low_prices[i] < low_prices[i-1] for i in range(1, len(low_prices)))
        
        # Décision avec priorités
        if is_hh and is_hl:
            structure = "UPTREND"
            trend_strength = float((high_prices[-1] - high_prices[0]) / high_prices[0] * 100)
        
        elif is_lh and is_ll:
            structure = "DOWNTREND"
            trend_strength = float((low_prices[0] - low_prices[-1]) / low_prices[-1] * 100)
        
        # 3. Patterns spéciaux
        elif is_hh and not is_hl and not is_ll:
            structure = "UPTREND_FAIBLE"
            trend_strength = float((high_prices[-1] - high_prices[0]) / high_prices[0] * 100) / 2
        
        elif is_lh and not is_hh and not is_hl:
            structure = "DOWNTREND_FAIBLE"
            trend_strength = float((low_prices[0] - low_prices[-1]) / low_prices[-1] * 100) / 2
        
        elif is_hh and is_ll:
            structure = "EXPANDING_WEDGE"
            trend_strength = 0.0
        
        elif is_lh and is_hl:
            structure = "CONTRACTING_WEDGE"
            trend_strength = 0.0
        
        else:
            # Range normal
            price_range = max(high_prices) - min(low_prices)
            avg_candle = recent['high'].mean() - recent['low'].mean()
            
            if price_range < avg_candle * 3:
                structure = "TIGHT_RANGE"
            else:
                structure = "RANGE"
    
    # Détection chaos/transition
    elif len(swing_highs) < 3 or len(swing_lows) < 3:
        volatility = recent['high'].std() / recent['close'].mean()
        
        if volatility > 0.0005:
            structure = "CHAOTIC"
        else:
            structure = "NO_CLEAR_STRUCTURE"
    
    # Identifier phase de mouvement
    if structure in ["UPTREND", "DOWNTREND"] and len(swing_highs) >= 4 and len(swing_lows) >= 4:
        if structure == "UPTREND":
            recent_move = high_prices[-1] - high_prices[-2] if len(high_prices) >= 2 else 0
            older_highs = sorted(swing_highs, key=lambda x: x['index'])[-4:-1]
            if len(older_highs) >= 2:
                older_move = older_highs[-1]['price'] - older_highs[-2]['price']
                if recent_move > older_move * 1.5:
                    structure += "_ACCELERATING"
                elif recent_move < older_move * 0.5:
                    structure += "_DECELERATING"
        
        elif structure == "DOWNTREND":
            recent_move = low_prices[-2] - low_prices[-1] if len(low_prices) >= 2 else 0
            older_lows = sorted(swing_lows, key=lambda x: x['index'])[-4:-1]
            if len(older_lows) >= 2:
                older_move = older_lows[-2]['price'] - older_lows[-1]['price']
                if recent_move > older_move * 1.5:
                    structure += "_ACCELERATING"
                elif recent_move < older_move * 0.5:
                    structure += "_DECELERATING"
    
    return structure, trend_strength, swing_highs, swing_lows

def is_near_swing_high(df, lookback=20):
    """Vérifie proximité swing high"""
    if len(df) < lookback:
        return False, 0.0
    
    recent = df.tail(lookback)
    swing_high = float(recent['high'].max())
    current = float(df.iloc[-1]['close'])
    
    distance = (swing_high - current) / swing_high * 100
    is_near = distance < SAINT_GRAAL_CONFIG['structure']['near_high_threshold']
    
    return bool(is_near), float(distance)

def detect_retest_pattern(df, lookback=5):
    """Détecte patterns de retest"""
    if len(df) < lookback + 1:
        return "NO_PATTERN", 0
    
    confidence = 0
    pattern_type = "NO_PATTERN"
    
    if len(df) >= 4:
        idx_red = -4
        idx_green1 = -3
        idx_green2 = -2
        
        red_candle = df.iloc[idx_red]
        green1_candle = df.iloc[idx_green1]
        green2_candle = df.iloc[idx_green2]
        
        is_red = bool(red_candle['close'] < red_candle['open'])
        is_green1 = bool(green1_candle['close'] > green1_candle['open'])
        is_green2 = bool(green2_candle['close'] > green2_candle['open'])
        
        if is_red and is_green1 and is_green2:
            pattern_type = "RETEST_PATTERN"
            
            red_size = float(abs(red_candle['close'] - red_candle['open']))
            green1_size = float(abs(green1_candle['close'] - green1_candle['open']))
            green2_size = float(abs(green2_candle['close'] - green2_candle['open']))
            
            if green1_size < red_size and green2_size < red_size:
                confidence += 30
            
            current_candle = df.iloc[-1]
            if float(current_candle['high']) < float(red_candle['high']):
                confidence += 30
            
            red_body_mid = float((red_candle['open'] + red_candle['close']) / 2)
            if float(green1_candle['close']) > red_body_mid:
                confidence += 20
            if float(green2_candle['close']) > red_body_mid:
                confidence += 20
    
    return pattern_type, confidence

# ================= ZONES S/R AMÉLIORÉES =================

def detect_key_zones(df, lookback=100, min_touches=2, tolerance_pips=None):
    """Détection améliorée des zones support/résistance avec clustering"""
    if len(df) < lookback:
        return [], []
    
    recent = df.tail(lookback).copy()
    
    # Calcul ATR pour tolérance dynamique
    atr = AverageTrueRange(
        high=recent['high'],
        low=recent['low'],
        close=recent['close'],
        window=14
    ).average_true_range()
    
    current_atr = atr.iloc[-1] if not atr.empty else 0.0005
    
    # Tolérance dynamique basée sur ATR (0.5 ATR par défaut)
    if tolerance_pips is None:
        tolerance = current_atr * 0.5
    else:
        tolerance = tolerance_pips * 0.0001
    
    # Détection des swings pour résistances
    resistances = []
    for i in range(3, len(recent)-3):
        current_high = float(recent.iloc[i]['high'])
        is_swing_high = (
            current_high == recent['high'].iloc[i-3:i+4].max() and
            current_high > recent['high'].iloc[i-1] and
            current_high > recent['high'].iloc[i+1]
        )
        
        if is_swing_high:
            # Vérifier clustering
            found_cluster = False
            for res in resistances:
                if abs(current_high - res['price']) <= tolerance:
                    # Ajouter au cluster existant
                    res['prices'].append(current_high)
                    res['touches'] += 1
                    res['last_touch'] = recent.index[i]
                    res['min_price'] = min(res['min_price'], current_high)
                    res['max_price'] = max(res['max_price'], current_high)
                    res['width_pips'] = (res['max_price'] - res['min_price']) / 0.0001
                    found_cluster = True
                    break
            
            if not found_cluster:
                resistances.append({
                    'price': current_high,
                    'prices': [current_high],
                    'touches': 1,
                    'last_touch': recent.index[i],
                    'type': 'RESISTANCE',
                    'min_price': current_high,
                    'max_price': current_high,
                    'width_pips': 0,
                    'strength_score': 0
                })
    
    # Détection des swings pour supports
    supports = []
    for i in range(3, len(recent)-3):
        current_low = float(recent.iloc[i]['low'])
        is_swing_low = (
            current_low == recent['low'].iloc[i-3:i+4].min() and
            current_low < recent['low'].iloc[i-1] and
            current_low < recent['low'].iloc[i+1]
        )
        
        if is_swing_low:
            found_cluster = False
            for sup in supports:
                if abs(current_low - sup['price']) <= tolerance:
                    sup['prices'].append(current_low)
                    sup['touches'] += 1
                    sup['last_touch'] = recent.index[i]
                    sup['min_price'] = min(sup['min_price'], current_low)
                    sup['max_price'] = max(sup['max_price'], current_low)
                    sup['width_pips'] = (sup['max_price'] - sup['min_price']) / 0.0001
                    found_cluster = True
                    break
            
            if not found_cluster:
                supports.append({
                    'price': current_low,
                    'prices': [current_low],
                    'touches': 1,
                    'last_touch': recent.index[i],
                    'type': 'SUPPORT',
                    'min_price': current_low,
                    'max_price': current_low,
                    'width_pips': 0,
                    'strength_score': 0
                })
    
    # Calcul du prix moyen pour chaque cluster
    for zone in supports + resistances:
        if len(zone['prices']) > 1:
            zone['price'] = float(np.mean(zone['prices']))
    
    # Calcul du score de force
    current_price = float(recent.iloc[-1]['close'])
    
    for zone in supports + resistances:
        # Score de base basé sur touches
        touches_score = min(zone['touches'] * 15, 40)
        
        # Pénalité pour largeur
        width_penalty = min(zone['width_pips'] * 1.5, 20)
        
        # Bonus récence
        age_bars = len(recent) - recent.index.get_loc(zone['last_touch'])
        recency_bonus = max(0, (100 - age_bars) * 0.5)
        
        # Score brut
        raw_score = touches_score - width_penalty + recency_bonus
        
        # Normalisation 0-50
        zone['strength_score'] = min(50, max(0, raw_score))
        
        # Marquer zones cassées
        if zone['type'] == 'RESISTANCE':
            closes_above = recent[recent['close'] > zone['price'] + tolerance]
            zone['broken'] = len(closes_above) > 2
            if zone['broken'] and not closes_above.empty:
                zone['broken_since'] = closes_above.index[0]
        else:  # SUPPORT
            closes_below = recent[recent['close'] < zone['price'] - tolerance]
            zone['broken'] = len(closes_below) > 2
            if zone['broken'] and not closes_below.empty:
                zone['broken_since'] = closes_below.index[0]
    
    # Filtrer zones
    valid_supports = [
        s for s in supports 
        if s['touches'] >= min_touches 
        and not s.get('broken', False)
        and s['width_pips'] <= 15
    ]
    
    valid_resistances = [
        r for r in resistances 
        if r['touches'] >= min_touches 
        and not r.get('broken', False)
        and r['width_pips'] <= 15
    ]
    
    # Trier par score
    valid_supports.sort(key=lambda x: x['strength_score'], reverse=True)
    valid_resistances.sort(key=lambda x: x['strength_score'], reverse=True)
    
    return valid_supports[:3], valid_resistances[:3]

def is_price_near_zone_pro(current_price, zones, max_distance_pips=10):
    """Vérifie proximité zone (version pro)"""
    max_distance = max_distance_pips * 0.0001
    
    if not zones:
        return False, None, float('inf')
    
    nearest_zone = None
    min_distance = float('inf')
    
    for zone in zones:
        distance = abs(current_price - zone['price'])
        if distance < min_distance:
            min_distance = distance
            nearest_zone = zone
    
    is_near = min_distance <= max_distance
    distance_pips = min_distance / 0.0001
    
    return is_near, nearest_zone, distance_pips

def calculate_zone_strength(zone):
    """Calcule un bonus de trading basé sur la force de la zone"""
    if not zone:
        return 0, "Aucune zone"
    
    strength = zone.get('strength_score', 0)
    
    if strength > 45:
        bonus = 25
        reason = f"Zone {zone['type']} TRÈS FORTE ({strength:.0f}/50)"
    elif strength > 35:
        bonus = 20
        reason = f"Zone {zone['type']} FORTE ({strength:.0f}/50)"
    elif strength > 25:
        bonus = 15
        reason = f"Zone {zone['type']} MOYENNE ({strength:.0f}/50)"
    elif strength > 15:
        bonus = 10
        reason = f"Zone {zone['type']} FAIBLE ({strength:.0f}/50)"
    else:
        bonus = 5
        reason = f"Zone {zone['type']} TRÈS FAIBLE ({strength:.0f}/50)"
    
    # Réduction pour zones larges
    if zone.get('width_pips', 0) > 10:
        bonus = int(bonus * 0.7)
        reason += " (zone large)"
    
    return bonus, reason

def get_zone_quality(zones):
    """Retourne un diagnostic de qualité des zones détectées"""
    if not zones:
        return "AUCUNE", 0, "Pas de zones détectées"
    
    qualities = []
    total_score = 0
    
    for zone in zones:
        score = zone.get('strength_score', 0)
        total_score += score
        
        if score > 40:
            qualities.append(f"{zone['type']}: EXCELLENT ({score:.0f})")
        elif score > 30:
            qualities.append(f"{zone['type']}: BON ({score:.0f})")
        elif score > 20:
            qualities.append(f"{zone['type']}: MOYEN ({score:.0f})")
        else:
            qualities.append(f"{zone['type']}: FAIBLE ({score:.0f})")
    
    avg_score = total_score / len(zones)
    
    if avg_score > 35:
        overall = "EXCELLENT"
    elif avg_score > 25:
        overall = "BON"
    elif avg_score > 15:
        overall = "ACCEPTABLE"
    else:
        overall = "MAUVAIS"
    
    return overall, avg_score, " | ".join(qualities)

# ================= MOMENTUM AVEC CORRECTIONS =================

def analyze_momentum_pro(df):
    """
    Momentum avec CORRECTION 2: Stochastic turning strict
    """
    if len(df) < 20:
        return {"status": "INSUFFICIENT_DATA"}
    
    # RSI
    rsi = RSIIndicator(close=df['close'], window=SAINT_GRAAL_CONFIG['rsi_period']).rsi()
    current_rsi = rsi.iloc[-1]
    
    # Stochastic avec turning strict
    stoch = StochasticOscillator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=SAINT_GRAAL_CONFIG['stoch_period'],
        smooth_window=3
    )
    stoch_k = stoch.stoch()
    stoch_d = stoch.stoch_signal()
    
    # CORRECTION 2: Turning strict (2 bougies consécutives)
    stoch_values = stoch_k.tail(4).values
    
    bearish_turning = False
    if len(stoch_values) >= 4:
        bearish_turning = (
            stoch_values[-1] < stoch_values[-2] < stoch_values[-3] and
            stoch_k.iloc[-1] < stoch_d.iloc[-1] and
            (stoch_values[-2] - stoch_values[-1]) > SAINT_GRAAL_CONFIG['stoch_turning']['min_stoch_move']
        )
    
    bullish_turning = False
    if len(stoch_values) >= 4:
        bullish_turning = (
            stoch_values[-1] > stoch_values[-2] > stoch_values[-3] and
            stoch_k.iloc[-1] > stoch_d.iloc[-1] and
            (stoch_values[-1] - stoch_values[-2]) > SAINT_GRAAL_CONFIG['stoch_turning']['min_stoch_move']
        )
    
    # MACD (CORRECTION 3: comme anti-signal)
    macd_ind = MACD(
        close=df['close'],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )
    macd_line = macd_ind.macd().iloc[-1]
    macd_signal = macd_ind.macd_signal().iloc[-1]
    
    # Calcul scores
    sell_score = 0
    buy_score = 0
    
    # Conditions SELL
    if current_rsi >= 60:
        sell_score += 20
    elif current_rsi >= 55:
        sell_score += 15
    
    if stoch_k.iloc[-1] >= 70:
        sell_score += 20
    elif stoch_k.iloc[-1] >= 50:
        sell_score += 15
    
    if bearish_turning:
        sell_score += 25
    
    # Conditions BUY
    if current_rsi <= 40:
        buy_score += 20
    elif current_rsi <= 45:
        buy_score += 15
    
    if stoch_k.iloc[-1] <= 30:
        buy_score += 20
    elif stoch_k.iloc[-1] <= 50:
        buy_score += 15
    
    if bullish_turning:
        buy_score += 25
    
    return {
        'rsi': float(current_rsi),
        'stoch_k': float(stoch_k.iloc[-1]),
        'stoch_d': float(stoch_d.iloc[-1]),
        'stoch_turning_bearish': bearish_turning,
        'stoch_turning_bullish': bullish_turning,
        'macd_line': float(macd_line),
        'macd_signal': float(macd_signal),
        'sell_score': sell_score,
        'buy_score': buy_score,
        'dominant': "SELL" if sell_score > buy_score else "BUY" if buy_score > sell_score else "NEUTRAL",
        'strength': abs(sell_score - buy_score),
    }

def apply_macd_veto(macd_info, direction):
    """
    CORRECTION 3: MACD comme veto (anti-signal)
    """
    if not SAINT_GRAAL_CONFIG['macd_filter']['use_as_veto']:
        return 0, ""
    
    macd_line = macd_info['macd_line']
    macd_signal = macd_info['macd_signal']
    threshold = SAINT_GRAAL_CONFIG['macd_filter']['veto_threshold']
    
    if direction == "SELL":
        if macd_line > macd_signal and (macd_line - macd_signal) > threshold:
            return -SAINT_GRAAL_CONFIG['macd_filter']['veto_penalty'], "Veto MACD: haussier"
    
    else:  # BUY
        if macd_line < macd_signal and (macd_signal - macd_line) > threshold:
            return -SAINT_GRAAL_CONFIG['macd_filter']['veto_penalty'], "Veto MACD: baissier"
    
    return 0, ""

# ================= CANDLE VALIDATION =================

def validate_candle_pattern(df, direction):
    """
    Validation bougie avec règles strictes
    """
    if len(df) < 5:
        return False, "INSUFFICIENT_DATA", 0, ""
    
    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]
    
    candle_body = abs(last_candle['close'] - last_candle['open'])
    candle_size = last_candle['high'] - last_candle['low']
    
    if candle_size == 0:
        return False, "NO_BODY", 0, ""
    
    upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
    lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
    
    is_bullish = last_candle['close'] > last_candle['open']
    is_bearish = last_candle['close'] < last_candle['open']
    
    patterns = []
    
    if direction == "SELL":
        # En range, exiger meilleure qualité
        structure, _, _, _ = analyze_market_structure(df.tail(20))
        in_range = "RANGE" in structure or "TIGHT" in structure
        min_wick_ratio = 2.5 if in_range else 2.0
        
        # Pin bar baissier
        if upper_wick > candle_body * min_wick_ratio and is_bearish:
            quality = 85 if upper_wick > candle_body * 3.0 else 75
            patterns.append(("PIN_BAR_BEARISH", quality))
        
        # Engulfing baissier
        if (is_bearish and prev_candle['close'] > prev_candle['open'] and
            last_candle['open'] >= prev_candle['close'] and
            last_candle['close'] <= prev_candle['open']):
            
            size_ratio = candle_body / abs(prev_candle['close'] - prev_candle['open'])
            quality = 90 if size_ratio > 1.8 else 80 if size_ratio > 1.5 else 70
            patterns.append(("ENGULFING_BEARISH", quality))
        
        # Clôture sous le low précédent
        if last_candle['close'] < prev_candle['low']:
            patterns.append(("CLOSE_BELOW_PREV_LOW", 75))
        
        # Rejection haute
        if upper_wick > candle_body * min_wick_ratio:
            patterns.append(("UPPER_REJECTION", 70))
    
    else:  # BUY
        structure, _, _, _ = analyze_market_structure(df.tail(20))
        in_range = "RANGE" in structure or "TIGHT" in structure
        min_wick_ratio = 2.5 if in_range else 2.0
        
        # Pin bar haussier
        if lower_wick > candle_body * min_wick_ratio and is_bullish:
            quality = 85 if lower_wick > candle_body * 3.0 else 75
            patterns.append(("PIN_BAR_BULLISH", quality))
        
        # Engulfing haussier
        if (is_bullish and prev_candle['close'] < prev_candle['open'] and
            last_candle['open'] <= prev_candle['close'] and
            last_candle['close'] >= prev_candle['open']):
            
            size_ratio = candle_body / abs(prev_candle['close'] - prev_candle['open'])
            quality = 90 if size_ratio > 1.8 else 80 if size_ratio > 1.5 else 70
            patterns.append(("ENGULFING_BULLISH", quality))
        
        # Clôture au-dessus du high précédent
        if last_candle['close'] > prev_candle['high']:
            patterns.append(("CLOSE_ABOVE_PREV_HIGH", 75))
        
        # Rejection basse
        if lower_wick > candle_body * min_wick_ratio:
            patterns.append(("LOWER_REJECTION", 70))
    
    if patterns:
        patterns.sort(key=lambda x: x[1], reverse=True)
        return True, patterns[0][0], patterns[0][1], "Bougie validée"
    
    return False, "NO_PATTERN", 0, "Aucun pattern bougie valide"

# ================= CALCUL STRUCTURE SCORE =================

def calculate_structure_score_pro(structure, direction, near_support, near_resistance):
    """
    CORRECTION 1: Calcul score structure avec règles RANGE strictes
    """
    if direction == "SELL":
        if structure == "DOWNTREND":
            return 25, "DOWNTREND fort (LH+LL)"
        
        elif structure == "DOWNTREND_ACCELERATING":
            return 30, "DOWNTREND accélérant"
        
        elif structure == "DOWNTREND_DECELERATING":
            return 20, "DOWNTREND ralentissant"
        
        elif structure == "UPTREND_DECELERATING" and near_resistance:
            return 15, "UPTREND ralentissant + résistance"
        
        elif structure == "EXPANDING_WEDGE" and near_resistance:
            return 18, "Wedge expansif + résistance"
        
        elif structure == "UPTREND" and near_resistance:
            return 10, "Vente risquée en UPTREND (zone uniquement)"
        
        elif structure in ["RANGE", "TIGHT_RANGE"]:
            if near_resistance:
                return 15, "Range + résistance"
            else:
                return 0, "Range sans zone (ignoré)"
        
        elif structure in ["UPTREND_FAIBLE", "DOWNTREND_FAIBLE"]:
            if near_resistance:
                return 12, "Tendance faible + zone"
            else:
                return 0, "Tendance faible sans zone"
        
        elif structure == "CHAOTIC":
            return -10, "Marché chaotique (éviter)"
        
        elif structure == "NO_CLEAR_STRUCTURE":
            return 0, "Structure non claire"
        
        else:
            return 10, f"Structure: {structure}"
    
    else:  # BUY
        if structure == "UPTREND":
            return 25, "UPTREND fort (HH+HL)"
        
        elif structure == "UPTREND_ACCELERATING":
            return 30, "UPTREND accélérant"
        
        elif structure == "UPTREND_DECELERATING":
            return 20, "UPTREND ralentissant"
        
        elif structure == "DOWNTREND_DECELERATING" and near_support:
            return 15, "DOWNTREND ralentissant + support"
        
        elif structure == "CONTRACTING_WEDGE" and near_support:
            return 18, "Wedge contractant + support"
        
        elif structure == "DOWNTREND" and near_support:
            return 10, "Achat risqué en DOWNTREND (zone uniquement)"
        
        elif structure in ["RANGE", "TIGHT_RANGE"]:
            if near_support:
                return 15, "Range + support"
            else:
                return 0, "Range sans zone (ignoré)"
        
        elif structure in ["UPTREND_FAIBLE", "DOWNTREND_FAIBLE"]:
            if near_support:
                return 12, "Tendance faible + zone"
            else:
                return 0, "Tendance faible sans zone"
        
        elif structure == "CHAOTIC":
            return -10, "Marché chaotique (éviter)"
        
        elif structure == "NO_CLEAR_STRUCTURE":
            return 0, "Structure non claire"
        
        else:
            return 10, f"Structure: {structure}"

# ================= FALLBACK PRO (CORRECTION 4) =================

def pro_fallback_intelligent(df, signal_count, total_signals):
    """
    CORRECTION 4: Fallback pro intelligent
    """
    if len(df) < 20:
        return create_minimal_fallback("Données insuffisantes", signal_count, total_signals)
    
    # Calcul indicateurs
    df_indicators = compute_saint_graal_indicators(df)
    last = df_indicators.iloc[-1]
    
    # Conditions minimales
    config = SAINT_GRAAL_CONFIG['fallback_rules']
    rsi = last.get('rsi_7', 50)
    
    # Éviter extrêmes RSI
    if rsi > config['max_rsi_extreme'] or rsi < config['min_rsi_extreme']:
        return create_minimal_fallback(f"RSI extrême: {rsi:.1f}", signal_count, total_signals)
    
    # Stochastic aligné
    stoch_k = last.get('stoch_k', 50)
    stoch_d = last.get('stoch_d', 50)
    
    if config['require_stoch_aligned']:
        stoch_aligned = (stoch_k > stoch_d and rsi > 50) or (stoch_k < stoch_d and rsi < 50)
        if not stoch_aligned:
            return create_minimal_fallback("Stochastic non aligné", signal_count, total_signals)
    
    # Éviter milieu de range
    if config['no_trade_middle_range'] and len(df) >= 15:
        recent_high = df['high'].tail(15).max()
        recent_low = df['low'].tail(15).min()
        current = df['close'].iloc[-1]
        
        range_height = recent_high - recent_low
        if range_height > 0:
            position = (current - recent_low) / range_height
            if 0.3 < position < 0.7:
                return create_minimal_fallback("Milieu de range", signal_count, total_signals)
    
    # Décision
    ema_5 = last.get('ema_5', 0)
    ema_13 = last.get('ema_13', 0)
    
    buy_conditions = 0
    sell_conditions = 0
    
    if rsi < 45:
        buy_conditions += 1
    if stoch_k < stoch_d:
        buy_conditions += 1
    if ema_5 > ema_13:
        buy_conditions += 1
    
    if rsi > 55:
        sell_conditions += 1
    if stoch_k > stoch_d:
        sell_conditions += 1
    if ema_5 < ema_13:
        sell_conditions += 1
    
    if buy_conditions >= 2 and buy_conditions > sell_conditions:
        direction = "CALL"
        score = 55.0
        reason = f"Fallback BUY (RSI:{rsi:.1f}, Stoch:{stoch_k:.1f}/{stoch_d:.1f})"
    
    elif sell_conditions >= 2 and sell_conditions > buy_conditions:
        direction = "PUT"
        score = 55.0
        reason = f"Fallback SELL (RSI:{rsi:.1f}, Stoch:{stoch_k:.1f}/{stoch_d:.1f})"
    
    else:
        if rsi > 50:
            direction = "CALL"
            score = 50.0
            reason = f"Fallback DEFAULT CALL (RSI:{rsi:.1f})"
        else:
            direction = "PUT"
            score = 50.0
            reason = f"Fallback DEFAULT PUT (RSI:{rsi:.1f})"
    
    return {
        'signal': direction,
        'mode': 'FALLBACK_PRO',
        'quality': 'MINIMUM',
        'score': float(score),
        'reason': reason,
        'structure_info': {
            'market_structure': 'FALLBACK',
            'trend_strength': 0.0,
            'near_swing_high': False,
            'distance_to_high': 0.0,
            'pattern_detected': 'FALLBACK_PRO',
            'pattern_confidence': int(score),
        }
    }

def create_minimal_fallback(reason, signal_count, total_signals):
    """Fallback absolu minimal"""
    return {
        'signal': 'CALL',
        'mode': 'FALLBACK_MINIMAL',
        'quality': 'CRITICAL',
        'score': 45.0,
        'reason': f"Fallback minimal: {reason}",
        'structure_info': {
            'market_structure': 'UNKNOWN',
            'trend_strength': 0.0,
            'near_swing_high': False,
            'distance_to_high': 0.0,
            'pattern_detected': 'FALLBACK_MIN',
            'pattern_confidence': 0,
        }
    }

# ================= HIÉRARCHIE DE QUALITÉ (NOUVEAU) =================

def get_hierarchy_level(signal_count):
    """Détermine le niveau hiérarchique basé sur le numéro du signal"""
    if signal_count < 4:  # Signaux 1-4
        return 'TOP', {
            'min_score': 85,
            'require_timing_break': True,
            'allow_counter_trend': False,
            'require_zone': True,
            'timing_type': 'STRICT',
            'quality_label': 'EXCELLENT'
        }
    elif signal_count < 6:  # Signaux 5-6
        return 'HIGH', {
            'min_score': 75,
            'require_timing_break': True,
            'allow_counter_trend': True,
            'counter_trend_penalty': 15,
            'require_zone': True,
            'timing_type': 'MODERATE',
            'quality_label': 'HIGH'
        }
    elif signal_count < 7:  # Signal 7
        return 'STANDARD', {
            'min_score': 65,
            'require_timing_break': False,
            'allow_counter_trend': True,
            'counter_trend_penalty': 10,
            'require_zone': False,
            'timing_type': 'LENIENT',
            'quality_label': 'ACCEPTABLE'
        }
    else:  # Signal 8
        return 'FALLBACK', {
            'min_score': 55,
            'require_timing_break': False,
            'allow_counter_trend': True,
            'require_zone': False,
            'timing_type': 'NONE',
            'quality_label': 'MINIMUM'
        }

def validate_timing_hierarchical(df, direction, timing_type):
    """Validation timing selon niveau hiérarchique"""
    if timing_type == 'NONE' or len(df) < 2:
        return True, "Timing non requis"
    
    prev_candle = df.iloc[-2]
    current_candle = df.iloc[-1]
    
    if timing_type == 'STRICT':
        # Break strict du high/low précédent
        if direction == "SELL":
            condition = current_candle['close'] < prev_candle['low']
            reason = f"Close {current_candle['close']:.5f} < Prev Low {prev_candle['low']:.5f}"
        else:  # BUY
            condition = current_candle['close'] > prev_candle['high']
            reason = f"Close {current_candle['close']:.5f} > Prev High {prev_candle['high']:.5f}"
        
        return condition, reason if condition else f"❌ {reason}"
    
    elif timing_type == 'MODERATE':
        # Test du high/low précédent
        if direction == "SELL":
            condition = current_candle['low'] <= prev_candle['low']
            reason = f"Test du low précédent"
        else:  # BUY
            condition = current_candle['high'] >= prev_candle['high']
            reason = f"Test du high précédent"
        
        return condition, reason
    
    else:  # LENIENT
        # Simple confirmation de direction
        if direction == "SELL":
            condition = current_candle['close'] < current_candle['open']
            reason = f"Bougie baissière"
        else:  # BUY
            condition = current_candle['close'] > current_candle['open']
            reason = f"Bougie haussière"
        
        return condition, reason

def calculate_structure_score_hierarchical(structure, direction, near_support, near_resistance, level_config):
    """Calcul de score structure avec règles hiérarchiques"""
    base_score, base_reason = calculate_structure_score_pro(structure, direction, near_support, near_resistance)
    
    # Vérifier contre-tendance
    if direction == "SELL" and structure in ["UPTREND", "UPTREND_ACCELERATING"]:
        if not level_config['allow_counter_trend']:
            return -25, "🚫 Vente interdite en uptrend (niveau TOP)"
        else:
            penalty = level_config.get('counter_trend_penalty', 10)
            return base_score - penalty, f"⚠️ Vente risquée en uptrend (-{penalty})"
    
    if direction == "BUY" and structure in ["DOWNTREND", "DOWNTREND_ACCELERATING"]:
        if not level_config['allow_counter_trend']:
            return -25, "🚫 Achat interdit en downtrend (niveau TOP)"
        else:
            penalty = level_config.get('counter_trend_penalty', 10)
            return base_score - penalty, f"⚠️ Achat risqué en downtrend (-{penalty})"
    
    return base_score, base_reason

# ================= FONCTION PRINCIPALE AVEC HIÉRARCHIE =================

def rule_signal_saint_graal_with_guarantee(df, signal_count=0, total_signals_needed=8):
    """
    STRATÉGIE - 8 signaux qualité maximale avec approche hiérarchique
    """
    print(f"\n{'='*70}")
    print(f"🚀 STRATÉGIE PRO HIÉRARCHIQUE - Signal #{signal_count+1}/{total_signals_needed}")
    print(f"{'='*70}")
    
    if len(df) < 50:
        print("[PRO HIÉRARCHIQUE] ⚠️ Données insuffisantes")
        return pro_fallback_intelligent(df, signal_count, total_signals_needed)
    
    # ===== DÉTERMINER NIVEAU HIÉRARCHIQUE =====
    level_name, level_config = get_hierarchy_level(signal_count)
    
    print(f"\n[PRO HIÉRARCHIQUE] 📊 NIVEAU {level_name}")
    print(f"   Score min: {level_config['min_score']} | Timing: {level_config['timing_type']}")
    print(f"   Contre-tendance: {'Autorisé' if level_config['allow_counter_trend'] else 'Interdit'}")
    print(f"   Zone requise: {'Oui' if level_config['require_zone'] else 'Non'}")
    
    # ===== 1. ANALYSE STRUCTURE =====
    structure, trend_strength, swing_highs, swing_lows = analyze_market_structure(df)
    print(f"\n[PRO HIÉRARCHIQUE] 🏗️  STRUCTURE: {structure} (force: {trend_strength:.1f}%)")
    
    # ===== 2. ZONES CLÉS =====
    supports, resistances = detect_key_zones(df)
    current_price = float(df.iloc[-1]['close'])
    
    near_support, nearest_support, dist_support = is_price_near_zone_pro(
        current_price, supports, max_distance_pips=10
    )
    near_resistance, nearest_resistance, dist_resistance = is_price_near_zone_pro(
        current_price, resistances, max_distance_pips=10
    )
    
    print(f"[PRO HIÉRARCHIQUE] 📍 ZONES: Support {near_support} ({dist_support:.1f}p) | Résistance {near_resistance} ({dist_resistance:.1f}p)")
    
    # Vérifier condition zone requise
    if level_config['require_zone'] and not (near_support or near_resistance):
        print(f"   ❌ Zone requise manquante pour niveau {level_name}")
        return pro_fallback_intelligent(df, signal_count, total_signals_needed)
    
    # ===== 3. MOMENTUM =====
    momentum = analyze_momentum_pro(df)
    print(f"\n[PRO HIÉRARCHIQUE] ⚡ MOMENTUM: RSI {momentum['rsi']:.1f} | Stoch {momentum['stoch_k']:.1f}/{momentum['stoch_d']:.1f}")
    print(f"   Dominant: {momentum['dominant']}")
    
    # ===== 4. SCORING HIÉRARCHIQUE =====
    print(f"\n[PRO HIÉRARCHIQUE] 🎯 SCORING (niveau {level_name})")
    
    # Score SELL
    sell_score = 0
    sell_details = []
    
    # Structure avec règles hiérarchiques
    structure_score_sell, structure_reason = calculate_structure_score_hierarchical(
        structure, "SELL", near_support, near_resistance, level_config
    )
    sell_score += structure_score_sell
    if structure_score_sell != 0:
        sell_details.append(structure_reason)
    
    # Momentum
    sell_score += momentum['sell_score']
    if momentum['sell_score'] > 0:
        sell_details.append(f"Momentum: {momentum['sell_score']}pts")
    
    # MACD veto
    macd_veto_penalty, macd_reason = apply_macd_veto(momentum, "SELL")
    sell_score += macd_veto_penalty
    if macd_veto_penalty < 0:
        sell_details.append(macd_reason)
    
    # Zone bonus
    if near_resistance:
        zone_bonus, zone_reason = calculate_zone_strength(nearest_resistance)
        sell_score += zone_bonus
        sell_details.append(zone_reason)
    
    # Score BUY
    buy_score = 0
    buy_details = []
    
    structure_score_buy, structure_reason = calculate_structure_score_hierarchical(
        structure, "BUY", near_support, near_resistance, level_config
    )
    buy_score += structure_score_buy
    if structure_score_buy != 0:
        buy_details.append(structure_reason)
    
    buy_score += momentum['buy_score']
    if momentum['buy_score'] > 0:
        buy_details.append(f"Momentum: {momentum['buy_score']}pts")
    
    macd_veto_penalty, macd_reason = apply_macd_veto(momentum, "BUY")
    buy_score += macd_veto_penalty
    if macd_veto_penalty < 0:
        buy_details.append(macd_reason)
    
    if near_support:
        zone_bonus, zone_reason = calculate_zone_strength(nearest_support)
        buy_score += zone_bonus
        buy_details.append(zone_reason)
    
    print(f"   SELL: {sell_score}/100 - {', '.join(sell_details[:3])}")
    print(f"   BUY: {buy_score}/100 - {', '.join(buy_details[:3])}")
    
    # ===== 5. VALIDATION TIMING =====
    direction = None
    final_score = 0
    validation_issues = []
    
    # Vérifier SELL
    if sell_score >= level_config['min_score'] and sell_score > buy_score:
        timing_ok, timing_reason = validate_timing_hierarchical(df, "SELL", level_config['timing_type'])
        
        if level_config['require_timing_break'] and not timing_ok:
            validation_issues.append(f"Timing: {timing_reason}")
        else:
            candle_valid, pattern, pattern_conf, candle_reason = validate_candle_pattern(df, "SELL")
            if candle_valid:
                direction = "SELL"
                final_score = sell_score + (pattern_conf / 10)
                if timing_ok:
                    final_score *= 1.1  # Bonus timing
                print(f"   ✅ Validation: {pattern} ({pattern_conf}%) | Timing: {timing_reason}")
            else:
                validation_issues.append(f"Bougie: {candle_reason}")
    
    # Vérifier BUY
    elif buy_score >= level_config['min_score'] and buy_score > sell_score:
        timing_ok, timing_reason = validate_timing_hierarchical(df, "BUY", level_config['timing_type'])
        
        if level_config['require_timing_break'] and not timing_ok:
            validation_issues.append(f"Timing: {timing_reason}")
        else:
            candle_valid, pattern, pattern_conf, candle_reason = validate_candle_pattern(df, "BUY")
            if candle_valid:
                direction = "BUY"
                final_score = buy_score + (pattern_conf / 10)
                if timing_ok:
                    final_score *= 1.1  # Bonus timing
                print(f"   ✅ Validation: {pattern} ({pattern_conf}%) | Timing: {timing_reason}")
            else:
                validation_issues.append(f"Bougie: {candle_reason}")
    
    # Si validation échoue
    if not direction and validation_issues:
        print(f"   ❌ Validation échouée: {validation_issues[0]}")
    
    # ===== 6. DÉCISION FINALE =====
    if direction:
        quality_map = {
            'TOP': 'EXCELLENT',
            'HIGH': 'HIGH',
            'STANDARD': 'ACCEPTABLE',
            'FALLBACK': 'MINIMUM'
        }
        
        quality = quality_map.get(level_name, 'MINIMUM')
        
        print(f"\n[PRO HIÉRARCHIQUE] 🎉 DÉCISION: {direction}")
        print(f"   Score: {final_score:.1f}/100 | Niveau: {level_name} | Qualité: {quality}")
        
        if direction == "SELL" and nearest_resistance:
            print(f"   📍 Cible: Résistance à {nearest_resistance['price']:.5f} ({nearest_resistance.get('touches', 1)} touches)")
        elif direction == "BUY" and nearest_support:
            print(f"   📍 Cible: Support à {nearest_support['price']:.5f} ({nearest_support.get('touches', 1)} touches)")
        
        print(f"{'='*70}")
        
        direction_display = "CALL" if direction == "BUY" else "PUT"
        
        return {
            'signal': direction_display,
            'mode': f'HIERARCHICAL_{level_name}',
            'quality': quality,
            'score': float(final_score),
            'reason': f"{direction} | Niveau {level_name} | Score {final_score:.1f}",
            'structure_info': {
                'market_structure': structure,
                'trend_strength': float(trend_strength),
                'near_swing_high': near_resistance,
                'distance_to_high': float(dist_resistance),
                'near_swing_low': near_support,
                'distance_to_low': float(dist_support),
                'pattern_detected': 'PRO_HIERARCHICAL',
                'pattern_confidence': int(final_score),
                'level': level_name,
                'timing_validated': True
            }
        }
    
    # ===== 7. FALLBACK =====
    print(f"\n[PRO HIÉRARCHIQUE] ⚡ FALLBACK ACTIVÉ")
    return pro_fallback_intelligent(df, signal_count, total_signals_needed)

# ================= FONCTIONS DE COMPATIBILITÉ =================

def compute_saint_graal_indicators(df):
    """Calcule les indicateurs pour qualité maximale"""
    df = df.copy()
    
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    config = SAINT_GRAAL_CONFIG
    
    # EMA
    df['ema_5'] = EMAIndicator(close=df['close'], window=config['ema_fast']).ema_indicator()
    df['ema_13'] = EMAIndicator(close=df['close'], window=config['ema_slow']).ema_indicator()
    df['ema_spread'] = abs(df['ema_5'] - df['ema_13']) / df['close']
    df['ema_trend'] = (df['ema_5'] > df['ema_13']).astype(int)
    
    # RSI
    df['rsi_7'] = RSIIndicator(close=df['close'], window=config['rsi_period']).rsi()
    df['rsi_trend'] = (df['rsi_7'] > 50).astype(int)
    
    # Stochastic
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
    
    # ADX
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
    
    # Price action
    df['candle_body'] = df['close'] - df['open']
    df['candle_size'] = df['high'] - df['low']
    df['body_ratio'] = abs(df['candle_body']) / df['candle_size'].replace(0, 0.00001)
    
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['total_wick'] = df['upper_wick'] + df['lower_wick']
    df['wick_ratio'] = df['total_wick'] / abs(df['candle_body']).replace(0, 0.00001)
    
    df['price_trend'] = (df['close'] > df['open']).astype(int)
    
    # Convergence
    df['convergence_raw'] = (
        df['ema_trend'] + 
        df['rsi_trend'] + 
        df['stoch_trend'] + 
        df['adx_trend'] + 
        df['price_trend']
    )
    df['convergence_score'] = df['convergence_raw'] / 5.0
    
    # Qualité
    df['data_quality'] = (
        (df['close'].notna()).astype(int) +
        (df['wick_ratio'] < 0.6).astype(int) +
        (df['body_ratio'] > 0.2).astype(int)
    ) / 3.0
    
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    for col in df.columns:
        if df[col].dtype in ['float32', 'float64']:
            df[col] = df[col].astype('float64')
        elif df[col].dtype in ['int32', 'int64']:
            df[col] = df[col].astype('int64')
    
    return df

def calculate_signal_quality_score(df):
    """Calcule un score de qualité global du signal (0-100)"""
    if len(df) < 20:
        return 0
    
    last = df.iloc[-1]
    score = 0
    
    # Convergence (30 points)
    convergence = last.get('convergence_score', 0.5)
    score += float(convergence) * 30
    
    # Force tendance (25 points)
    adx = last.get('adx', 0)
    if adx > 30:
        score += 25
    elif adx > 25:
        score += 20
    elif adx > 20:
        score += 15
    elif adx > 15:
        score += 10
    
    # Alignement indicateurs (20 points)
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
    
    # Volatilité (15 points)
    bb_width = last.get('bb_width', 0) if 'bb_width' in last else 0.02
    if 0.01 < bb_width < 0.03:
        score += 15
    elif 0.005 < bb_width < 0.04:
        score += 10
    elif 0.002 < bb_width < 0.05:
        score += 5
    
    # Qualité bougie (10 points)
    body_ratio = last.get('body_ratio', 0)
    wick_ratio = last.get('wick_ratio', 0)
    
    if body_ratio > 0.4 and wick_ratio < 0.3:
        score += 10
    elif body_ratio > 0.3 and wick_ratio < 0.4:
        score += 5
    
    return min(score, 100)

def check_anti_manipulation(df, strict_mode=True):
    """Vérifie les conditions anti-manipulation"""
    if len(df) < 15:
        return False, "Données insuffisantes"
    
    last = df.iloc[-1]
    
    # Vérifications de base
    if 'data_quality' in last and last['data_quality'] < 0.85:
        return False, f"Qualité données faible: {last['data_quality']:.2f}"
    
    if 'wick_ratio' in last and last['wick_ratio'] > 0.6:
        return False, f"Mèche suspecte: {last['wick_ratio']:.1%}"
    
    if 'body_ratio' in last and last['body_ratio'] < 0.1:
        return False, f"Bougie trop plate: {last['body_ratio']:.1%}"
    
    # Vérification volatilité
    if 'candle_size' in last and 'atr_10' in last:
        atr_10 = last['atr_10'] if 'atr_10' in last else 0.0005
        if last['candle_size'] > atr_10 * 3:
            return False, f"Bougie trop grande: {last['candle_size']:.5f} > ATRx3"
    
    return True, "OK"

def get_signal_with_metadata(df, signal_count=0, total_signals=8):
    """Fonction principale pour le bot - CORRIGÉ POUR JSON"""
    try:
        if df is None or len(df) < 30:
            level_name, _ = get_hierarchy_level(signal_count)
            return create_minimal_fallback("Données insuffisantes", signal_count, total_signals)
        
        df_with_indicators = compute_saint_graal_indicators(df)
        
        if df_with_indicators.empty:
            level_name, _ = get_hierarchy_level(signal_count)
            return create_minimal_fallback("Erreur indicateurs", signal_count, total_signals)
        
        result = rule_signal_saint_graal_with_guarantee(df_with_indicators, signal_count, total_signals)
        
        if result:
            direction_display = result['signal']
            quality_display = {
                'EXCELLENT': '⭐⭐⭐⭐⭐',
                'HIGH': '⭐⭐⭐⭐',
                'ACCEPTABLE': '⭐⭐⭐',
                'MINIMUM': '⭐⭐',
                'CRITICAL': '⭐'
            }.get(result['quality'], '⭐')
            
            reason = f"{quality_display} {direction_display} | Score: {result['score']:.0f}/100 | {result['mode']}"
            
            return {
                'direction': direction_display,
                'mode': result['mode'],
                'quality': result['quality'],
                'score': float(result['score']),
                'reason': reason,
                'structure_info': result['structure_info'],
                'session_info': {
                    'current_signal': signal_count + 1,
                    'total_signals': total_signals,
                    'mode_used': result['mode'],
                    'quality_level': result['quality']
                }
            }
        
    except Exception as e:
        print(f"❌ Erreur dans get_signal_with_metadata: {str(e)}")
    
    # Fallback absolu
    return {
        'direction': 'CALL',
        'mode': 'ERROR',
        'quality': 'CRITICAL',
        'score': 45.0,
        'reason': 'Erreur système',
        'structure_info': {
            'market_structure': 'ERROR',
            'strength': 0.0,
            'near_swing_high': False,
            'distance_to_high': 0.0,
            'pattern_detected': 'ERROR',
            'pattern_confidence': 0
        },
        'session_info': {
            'current_signal': signal_count + 1,
            'total_signals': total_signals,
            'mode_used': 'ERROR',
            'quality_level': 'CRITICAL'
        }
    }

def compute_indicators(df, ema_fast=8, ema_slow=21, rsi_len=14, bb_len=20):
    """Wrapper pour compatibilité"""
    return compute_saint_graal_indicators(df)

def rule_signal(df):
    """Version par défaut"""
    result = rule_signal_saint_graal_with_guarantee(df, signal_count=0, total_signals_needed=8)
    return result['signal'] if result else 'CALL'

def format_signal_reason(direction, confidence, indicators):
    """Formate la raison du signal"""
    last = indicators.iloc[-1]
    quality_score = calculate_signal_quality_score(indicators)
    
    reason_parts = [f"🎯 {direction}"]
    
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
    
    reason_parts.append(f"RSI7: {last['rsi_7']:.1f}")
    reason_parts.append(f"ADX: {last['adx']:.1f}")
    
    if last['convergence_raw'] >= 4:
        reason_parts.append("CONV: EXCELLENTE")
    elif last['convergence_raw'] >= 3:
        reason_parts.append("CONV: BONNE")
    
    reason_parts.append(f"CONF: {int(confidence)}%")
    
    return " | ".join(reason_parts)

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

# ================= INITIALISATION =================

if __name__ == "__main__":
    print("🚀 STRATÉGIE FOREX M1 PRO - APPROCHE HIÉRARCHIQUE")
    print("📊 Version: 4.0 avec zones S/R améliorées et hiérarchie")
    print("🎯 Objectif: 8 signaux avec qualité contrôlée")
    print("\n📋 HIÉRARCHIE DES SIGNAUX:")
    print("1. TOP (1-4): 85+ score, timing strict, pas de contre-tendance")
    print("2. HIGH (5-6): 75+ score, timing modéré, contre-tendance pénalisé")
    print("3. STANDARD (7): 65+ score, timing lenient, contre-tendance accepté")
    print("4. FALLBACK (8): Règles minimales")
    print("\n💡 Architecture: Structure → Zones → Momentum → Timing → Validation hiérarchique")
    print("✅ Prêt pour le trading live")
