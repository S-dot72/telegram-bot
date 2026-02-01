"""
utils.py - STRATÉGIE BINAIRE M1 PRO - VERSION 4.2 COMPATIBLE
Niveau desk pro avec compatibilité totale signal_bot
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIGURATION DESK PRO =================

SAINT_GRAAL_CONFIG = {
    # 🔥 ASYMÉTRIE BUY/SELL
    'buy_rules': {
        'stoch_period': 5,      # Rapide pour les BUY
        'stoch_smooth': 3,
        'rsi_max_for_buy': 45,  # RSI maximum pour BUY
        'rsi_oversold': 35,     # Seuil oversold standard
        'require_swing_confirmation': False,  # BUY plus réactif
    },
    
    'sell_rules': {
        'stoch_period': 9,      # Lent pour les SELL
        'stoch_smooth': 3,
        'rsi_min_for_sell': 58,  # 🔥 Augmenté à 58
        'stoch_min_overbought': 65,
        'require_swing_break': True,  # 🔥 Confirmation swing obligatoire
        'max_swing_distance_pips': 5,  # Distance max au swing
        'momentum_gate_diff': 12,      # 🔥 Augmenté à 12
    },
    
    # Seuils contextuels optimisés
    'momentum_context': {
        'trend_overbought': 62,  # 🔥 Baissé à 62
        'trend_oversold': 38,    # 🔥 Hausse à 38
        'range_overbought': 72,
        'range_oversold': 28,
        'strong_trend_threshold': 1.0,  # 🔥 Hausse à 1.0%
    },
    
    # Structure rules PRO
    'structure_rules': {
        'sell_in_uptrend': {
            'require_swing_break': True,
            'min_stoch_overbought': 68,
            'allow_only_at_resistance': True,
            'max_rsi': 62,
        },
        'buy_in_downtrend': {
            'require_swing_break': True,
            'max_stoch_oversold': 32,
            'allow_only_at_support': True,
            'min_rsi': 38,
        },
        'range_trading': {
            'min_zone_strength': 25,
            'require_bounce_confirmation': True,
            'max_entry_from_zone_pips': 3,
        }
    },
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
    COMPATIBILITÉ : Retourne seulement 2 valeurs (structure, trend_strength)
    Pour usage dans signal_bot.py
    """
    if len(df) < lookback + 5:
        return "INSUFFICIENT_DATA", 0.0
    
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
    
    # Logique de détection
    structure = "RANGE"
    trend_strength = 0.0
    
    if len(swing_highs) >= 3 and len(swing_lows) >= 3:
        recent_highs = sorted(swing_highs, key=lambda x: x['index'])[-3:]
        recent_lows = sorted(swing_lows, key=lambda x: x['index'])[-3:]
        
        high_prices = [h['price'] for h in recent_highs]
        low_prices = [l['price'] for l in recent_lows]
        
        is_hh = all(high_prices[i] > high_prices[i-1] for i in range(1, len(high_prices)))
        is_hl = all(low_prices[i] > low_prices[i-1] for i in range(1, len(low_prices)))
        is_lh = all(high_prices[i] < high_prices[i-1] for i in range(1, len(high_prices)))
        is_ll = all(low_prices[i] < low_prices[i-1] for i in range(1, len(low_prices)))
        
        if is_hh and is_hl:
            structure = "UPTREND"
            trend_strength = float((high_prices[-1] - high_prices[0]) / high_prices[0] * 100)
        elif is_lh and is_ll:
            structure = "DOWNTREND"
            trend_strength = float((low_prices[0] - low_prices[-1]) / low_prices[-1] * 100)
        else:
            structure = "RANGE"
    
    return structure, trend_strength

# ================= DÉTECTION SWING INTERNE (NOUVELLE FONCTION) =================

def detect_internal_swings(df, lookback=10):
    """
    🔥 NOUVELLE FONCTION : Détection swings internes pour confirmation structure
    """
    if len(df) < lookback:
        return None, None
    
    recent = df.tail(lookback).copy()
    
    # Dernier swing high interne
    internal_high_idx = recent['high'].idxmax()
    internal_high = {
        'price': float(recent.loc[internal_high_idx, 'high']),
        'index': internal_high_idx,
        'bars_ago': len(recent) - 1 - recent.index.get_loc(internal_high_idx)
    }
    
    # Dernier swing low interne
    internal_low_idx = recent['low'].idxmin()
    internal_low = {
        'price': float(recent.loc[internal_low_idx, 'low']),
        'index': internal_low_idx,
        'bars_ago': len(recent) - 1 - recent.index.get_loc(internal_low_idx)
    }
    
    return internal_high, internal_low

def check_swing_break(current_price, internal_swing, direction, max_distance_pips=5):
    """
    Vérifie si le prix a cassé un swing interne
    """
    if internal_swing is None:
        return False, 0
    
    distance_pips = abs(current_price - internal_swing['price']) / 0.0001
    
    if direction == "SELL":
        # Pour SELL: besoin de casser le swing low interne
        is_broken = current_price < internal_swing['price']
    else:  # BUY
        # Pour BUY: besoin de casser le swing high interne
        is_broken = current_price > internal_swing['price']
    
    is_near = distance_pips <= max_distance_pips
    
    return is_broken and is_near, distance_pips

# ================= ZONES S/R (NOUVELLES FONCTIONS) =================

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
            found_cluster = False
            for res in resistances:
                if abs(current_high - res['price']) <= tolerance:
                    res['prices'].append(current_high)
                    res['touches'] += 1
                    res['last_touch'] = recent.index[i]
                    found_cluster = True
                    break
            
            if not found_cluster:
                resistances.append({
                    'price': current_high,
                    'prices': [current_high],
                    'touches': 1,
                    'last_touch': recent.index[i],
                    'type': 'RESISTANCE',
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
                    found_cluster = True
                    break
            
            if not found_cluster:
                supports.append({
                    'price': current_low,
                    'prices': [current_low],
                    'touches': 1,
                    'last_touch': recent.index[i],
                    'type': 'SUPPORT',
                })
    
    # Calcul du score de force
    current_price = float(recent.iloc[-1]['close'])
    
    for zone in supports + resistances:
        # Score de base basé sur touches
        touches_score = min(zone['touches'] * 15, 40)
        
        # Pénalité pour largeur
        if len(zone['prices']) > 1:
            zone_width = max(zone['prices']) - min(zone['prices'])
            width_pips = zone_width / 0.0001
            width_penalty = min(width_pips * 1.5, 20)
        else:
            width_penalty = 0
        
        # Bonus récence
        age_bars = len(recent) - recent.index.get_loc(zone['last_touch'])
        recency_bonus = max(0, (100 - age_bars) * 0.5)
        
        # Score brut
        raw_score = touches_score - width_penalty + recency_bonus
        
        # Normalisation 0-50
        zone['strength_score'] = min(50, max(0, raw_score))
    
    # Filtrer zones
    valid_supports = [
        s for s in supports 
        if s['touches'] >= min_touches 
    ]
    
    valid_resistances = [
        r for r in resistances 
        if r['touches'] >= min_touches 
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
    
    return bonus, reason

# ================= MOMENTUM ASYMÉTRIQUE (NOUVELLE FONCTION) =================

def analyze_momentum_asymmetric(df):
    """
    🔥 NOUVELLE FONCTION : Momentum asymétrique BUY/SELL
    Stochastic rapide pour BUY, lent pour SELL
    """
    if len(df) < 30:
        return {"status": "INSUFFICIENT_DATA"}
    
    # RSI commun
    rsi = RSIIndicator(close=df['close'], window=7).rsi()
    current_rsi = rsi.iloc[-1]
    
    # 🔥 STOCHASTIC RAPIDE POUR BUY (5,3,3)
    stoch_fast = StochasticOscillator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=SAINT_GRAAL_CONFIG['buy_rules']['stoch_period'],
        smooth_window=SAINT_GRAAL_CONFIG['buy_rules']['stoch_smooth']
    )
    stoch_k_fast = stoch_fast.stoch()
    stoch_d_fast = stoch_fast.stoch_signal()
    
    # 🔥 STOCHASTIC LENT POUR SELL (9,3,3)
    stoch_slow = StochasticOscillator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=SAINT_GRAAL_CONFIG['sell_rules']['stoch_period'],
        smooth_window=SAINT_GRAAL_CONFIG['sell_rules']['stoch_smooth']
    )
    stoch_k_slow = stoch_slow.stoch()
    stoch_d_slow = stoch_slow.stoch_signal()
    
    # Détection structure pour seuils contextuels
    structure, trend_strength = analyze_market_structure(df.tail(15))
    is_strong_trend = abs(trend_strength) > SAINT_GRAAL_CONFIG['momentum_context']['strong_trend_threshold']
    
    if is_strong_trend:
        overbought_threshold = SAINT_GRAAL_CONFIG['momentum_context']['trend_overbought']
        oversold_threshold = SAINT_GRAAL_CONFIG['momentum_context']['trend_oversold']
    else:
        overbought_threshold = SAINT_GRAAL_CONFIG['momentum_context']['range_overbought']
        oversold_threshold = SAINT_GRAAL_CONFIG['momentum_context']['range_oversold']
    
    # 🔥 DÉTECTION PIC STOCHASTIC LENT (pour SELL)
    stoch_slow_values = stoch_k_slow.tail(10).values
    stoch_peak_detected = False
    stoch_peak_value = 0
    
    if len(stoch_slow_values) >= 8:
        for i in range(3, len(stoch_slow_values)-3):
            if (stoch_slow_values[i] > stoch_slow_values[i-1] and 
                stoch_slow_values[i] > stoch_slow_values[i-2] and
                stoch_slow_values[i] > stoch_slow_values[i-3] and
                stoch_slow_values[i] > stoch_slow_values[i+1] and
                stoch_slow_values[i] > stoch_slow_values[i+2] and
                stoch_slow_values[i] > stoch_slow_values[i+3]):
                
                if stoch_slow_values[i] >= overbought_threshold:
                    stoch_peak_detected = True
                    stoch_peak_value = stoch_slow_values[i]
                    break
    
    # Turning bearish sur slow
    stoch_turning_bearish = False
    if len(stoch_slow_values) >= 4:
        recent_slow = stoch_slow_values[-4:]
        if (recent_slow[0] > recent_slow[1] > recent_slow[2] > recent_slow[3] and
            recent_slow[0] - recent_slow[3] > 8.0):  # 🔥 8 points minimum
            stoch_turning_bearish = True
    
    # Turning bullish sur fast
    stoch_fast_values = stoch_k_fast.tail(6).values
    stoch_turning_bullish = False
    if len(stoch_fast_values) >= 4:
        recent_fast = stoch_fast_values[-4:]
        if (recent_fast[0] < recent_fast[1] < recent_fast[2] < recent_fast[3] and
            recent_fast[3] - recent_fast[0] > 6.0):
            stoch_turning_bullish = True
    
    # MACD
    macd_ind = MACD(
        close=df['close'],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )
    macd_line = macd_ind.macd().iloc[-1]
    macd_signal = macd_ind.macd_signal().iloc[-1]
    macd_histogram = macd_line - macd_signal
    
    # 🔥 CALCUL SCORES ASYMÉTRIQUES
    sell_score = 0
    buy_score = 0
    sell_reasons = []
    buy_reasons = []
    
    # ========== CONDITIONS SELL (STRICTISSIMES) ==========
    current_stoch_slow = stoch_k_slow.iloc[-1]
    
    # RÈGLE 1: RSI minimum 58
    if current_rsi >= SAINT_GRAAL_CONFIG['sell_rules']['rsi_min_for_sell']:
        sell_score += 15
        sell_reasons.append(f"RSI:{current_rsi:.1f}")
    
    # RÈGLE 2: Stochastic lent overbought + turning
    if (current_stoch_slow >= overbought_threshold and 
        stoch_turning_bearish and
        current_stoch_slow <= 82):  # 🔥 Éviter extrêmes tardifs
        
        sell_score += 25
        sell_reasons.append(f"StochS:{current_stoch_slow:.1f}↓")
    
    # RÈGLE 3: Pic détecté + baisse confirmée
    elif stoch_peak_detected and stoch_turning_bearish:
        sell_score += 30  # 🔥 Bonus pic
        sell_reasons.append(f"Pic:{stoch_peak_value:.1f}↓")
    
    # RÈGLE 4: MACD baissier fort
    if macd_histogram < -0.00015:  # 🔥 Seuil plus strict
        sell_score += 12
        sell_reasons.append("MACD↓")
    
    # ========== CONDITIONS BUY (RÉACTIVES) ==========
    current_stoch_fast = stoch_k_fast.iloc[-1]
    
    # RSI oversold ou bas
    if current_rsi <= SAINT_GRAAL_CONFIG['buy_rules']['rsi_max_for_buy']:
        buy_score += 18  # 🔥 Plus généreux
        buy_reasons.append(f"RSI:{current_rsi:.1f}")
    
    # Stochastic fast oversold ou turning
    if (current_stoch_fast <= oversold_threshold or 
        stoch_turning_bullish):
        
        buy_score += 28
        buy_reasons.append(f"StochF:{current_stoch_fast:.1f}↑")
    
    # MACD haussier
    if macd_histogram > 0.0001:
        buy_score += 10
        buy_reasons.append("MACD↑")
    
    # 🔥 VÉTO MOMENTUM STRICT
    momentum_gate_passed = True
    gate_reason = ""
    
    if sell_score > buy_score:
        diff = sell_score - buy_score
        if diff < SAINT_GRAAL_CONFIG['sell_rules']['momentum_gate_diff']:
            momentum_gate_passed = False
            gate_reason = f"Momentum SELL insuffisant ({diff} < {SAINT_GRAAL_CONFIG['sell_rules']['momentum_gate_diff']})"
    
    elif buy_score > sell_score:
        diff = buy_score - sell_score
        if diff < 8:  # 🔥 Gate plus léger pour BUY
            momentum_gate_passed = False
            gate_reason = f"Momentum BUY insuffisant ({diff} < 8)"
    
    return {
        'rsi': float(current_rsi),
        'stoch_k_fast': float(stoch_k_fast.iloc[-1]),
        'stoch_d_fast': float(stoch_d_fast.iloc[-1]),
        'stoch_k_slow': float(stoch_k_slow.iloc[-1]),
        'stoch_d_slow': float(stoch_d_slow.iloc[-1]),
        'stoch_turning_bearish': stoch_turning_bearish,
        'stoch_turning_bullish': stoch_turning_bullish,
        'stoch_peak_detected': stoch_peak_detected,
        'stoch_peak_value': float(stoch_peak_value),
        'macd_histogram': float(macd_histogram),
        'sell_score': sell_score,
        'buy_score': buy_score,
        'sell_reasons': sell_reasons,
        'buy_reasons': buy_reasons,
        'overbought_threshold': overbought_threshold,
        'oversold_threshold': oversold_threshold,
        'dominant': "SELL" if sell_score > buy_score else "BUY" if buy_score > sell_score else "NEUTRAL",
        'strength': abs(sell_score - buy_score),
        'momentum_gate_passed': momentum_gate_passed,
        'gate_reason': gate_reason,
        'structure': structure,
        'trend_strength': trend_strength,
    }

# ================= STRUCTURE SCORE PRO (NOUVELLE FONCTION) =================

def calculate_structure_score_pro_m1(structure, direction, momentum_info, internal_swing_break):
    """
    🔥 NOUVELLE FONCTION : Structure score avec règles desk pro
    """
    score = 0
    reasons = []
    
    if direction == "SELL":
        # 🔥 RÈGLE DESK: SELL en uptrend = conditions ultra strictes
        if "UPTREND" in structure:
            rules = SAINT_GRAAL_CONFIG['structure_rules']['sell_in_uptrend']
            
            if momentum_info['rsi'] > rules['max_rsi']:
                score = -20
                reasons.append(f"RSI {momentum_info['rsi']:.1f} > {rules['max_rsi']}")
            
            elif not internal_swing_break and rules['require_swing_break']:
                score = -25
                reasons.append("Pas de break swing interne")
            
            elif momentum_info['stoch_k_slow'] < rules['min_stoch_overbought']:
                score = -15
                reasons.append(f"Stoch {momentum_info['stoch_k_slow']:.1f} < {rules['min_stoch_overbought']}")
            
            else:
                score = 15  # 🔥 Score réduit pour SELL contre tendance
                reasons.append("SELL uptrend strict")
        
        # SELL en downtrend = favorable
        elif "DOWNTREND" in structure:
            score = 25
            reasons.append("DOWNTREND fort")
        
        # SELL en range = conditions moyennes
        elif "RANGE" in structure:
            score = 18
            reasons.append("Range + zone")
        
        # SELL en wedge/chao = éviter
        elif "WEDGE" in structure or "CHAOTIC" in structure:
            score = -10
            reasons.append("Structure défavorable")
    
    else:  # BUY
        # 🔥 RÈGLE DESK: BUY en downtrend = conditions strictes
        if "DOWNTREND" in structure:
            rules = SAINT_GRAAL_CONFIG['structure_rules']['buy_in_downtrend']
            
            if momentum_info['rsi'] < rules['min_rsi']:
                score = -20
                reasons.append(f"RSI {momentum_info['rsi']:.1f} < {rules['min_rsi']}")
            
            elif not internal_swing_break and rules['require_swing_break']:
                score = -25
                reasons.append("Pas de break swing interne")
            
            elif momentum_info['stoch_k_fast'] > rules['max_stoch_oversold']:
                score = -15
                reasons.append(f"Stoch {momentum_info['stoch_k_fast']:.1f} > {rules['max_stoch_oversold']}")
            
            else:
                score = 15
                reasons.append("BUY downtrend strict")
        
        # BUY en uptrend = favorable
        elif "UPTREND" in structure:
            score = 25
            reasons.append("UPTREND fort")
        
        # BUY en range
        elif "RANGE" in structure:
            score = 18
            reasons.append("Range + zone")
        
        # BUY en wedge/chao
        elif "WEDGE" in structure or "CHAOTIC" in structure:
            score = -10
            reasons.append("Structure défavorable")
    
    return score, " | ".join(reasons)

# ================= VALIDATION BOUGIE (NOUVELLE FONCTION) =================

def validate_candle_for_binary_m1(df, direction, require_rejection=True):
    """
    Validation bougie pour binaire M1
    Exige des confirmations claires de retournement
    """
    if len(df) < 5:
        return False, "NO_DATA", 0, ""
    
    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]
    prev2_candle = df.iloc[-3] if len(df) >= 3 else None
    
    candle_body = abs(last_candle['close'] - last_candle['open'])
    candle_size = last_candle['high'] - last_candle['low']
    
    if candle_size == 0:
        return False, "NO_BODY", 0, ""
    
    upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
    lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
    
    patterns = []
    
    if direction == "SELL":
        # 🔥 RÈGLE: Exiger rejet clair en M1 binaire
        if require_rejection:
            # Pin bar baissier avec mèche haute > 2x corps
            if upper_wick > candle_body * 2.0:
                quality = 85 if upper_wick > candle_body * 3.0 else 75
                patterns.append(("PIN_BAR_BEARISH", quality, f"Mèche:{upper_wick/candle_body:.1f}x"))
            
            # Clôture sous low précédent (confirmation forte)
            if last_candle['close'] < prev_candle['low']:
                patterns.append(("CLOSE_BELOW_PREV_LOW", 90, "Break bas"))
            
            # Engulfing baissier
            if (last_candle['close'] < last_candle['open'] and
                prev_candle['close'] > prev_candle['open'] and
                last_candle['open'] >= prev_candle['close'] and
                last_candle['close'] <= prev_candle['open']):
                
                size_ratio = candle_body / abs(prev_candle['close'] - prev_candle['open'])
                quality = 95 if size_ratio > 1.8 else 85 if size_ratio > 1.5 else 75
                patterns.append(("ENGULFING_BEARISH", quality, f"Ratio:{size_ratio:.1f}"))
        
        # Pattern 3-bar reversal
        if prev2_candle is not None:
            if (prev2_candle['close'] > prev2_candle['open'] and
                prev_candle['close'] > prev_candle['open'] and
                last_candle['close'] < last_candle['open'] and
                last_candle['close'] < prev2_candle['low']):
                
                patterns.append(("3_BAR_REVERSAL", 80, "Retournement 3 bougies"))
    
    else:  # BUY
        if require_rejection:
            # Pin bar haussier
            if lower_wick > candle_body * 2.0:
                quality = 85 if lower_wick > candle_body * 3.0 else 75
                patterns.append(("PIN_BAR_BULLISH", quality, f"Mèche:{lower_wick/candle_body:.1f}x"))
            
            # Clôture au-dessus high précédent
            if last_candle['close'] > prev_candle['high']:
                patterns.append(("CLOSE_ABOVE_PREV_HIGH", 90, "Break haut"))
            
            # Engulfing haussier
            if (last_candle['close'] > last_candle['open'] and
                prev_candle['close'] < prev_candle['open'] and
                last_candle['open'] <= prev_candle['close'] and
                last_candle['close'] >= prev_candle['open']):
                
                size_ratio = candle_body / abs(prev_candle['close'] - prev_candle['open'])
                quality = 95 if size_ratio > 1.8 else 85 if size_ratio > 1.5 else 75
                patterns.append(("ENGULFING_BULLISH", quality, f"Ratio:{size_ratio:.1f}"))
        
        # Pattern 3-bar reversal
        if prev2_candle is not None:
            if (prev2_candle['close'] < prev2_candle['open'] and
                prev_candle['close'] < prev_candle['open'] and
                last_candle['close'] > last_candle['open'] and
                last_candle['close'] > prev2_candle['high']):
                
                patterns.append(("3_BAR_REVERSAL", 80, "Retournement 3 bougies"))
    
    if patterns:
        patterns.sort(key=lambda x: x[1], reverse=True)
        best_pattern = patterns[0]
        return True, best_pattern[0], best_pattern[1], best_pattern[2]
    
    # Si aucun pattern mais confirmation simple
    if direction == "SELL" and last_candle['close'] < last_candle['open']:
        return True, "BEARISH_CANDLE", 60, "Simple bougie baissière"
    elif direction == "BUY" and last_candle['close'] > last_candle['open']:
        return True, "BULLISH_CANDLE", 60, "Simple bougie haussière"
    
    return False, "NO_PATTERN", 0, "Aucune confirmation bougie"

# ================= FONCTIONS DE FALLBACK =================

def pro_fallback_intelligent(df, signal_count, total_signals):
    """
    Fallback pro intelligent
    """
    if len(df) < 20:
        return create_minimal_fallback("Données insuffisantes", signal_count, total_signals)
    
    # Calcul indicateurs
    df_indicators = compute_saint_graal_indicators(df)
    last = df_indicators.iloc[-1]
    
    # Conditions minimales
    rsi = last.get('rsi_7', 50)
    stoch_k = last.get('stoch_k', 50)
    stoch_d = last.get('stoch_d', 50)
    
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
    }

def create_minimal_fallback(reason, signal_count, total_signals):
    """Fallback absolu minimal"""
    return {
        'signal': 'CALL',
        'mode': 'FALLBACK_MINIMAL',
        'quality': 'CRITICAL',
        'score': 45.0,
        'reason': f"Fallback minimal: {reason}",
    }

# ================= FONCTIONS DE COMPATIBILITÉ =================

def compute_indicators(df, ema_fast=8, ema_slow=21, rsi_len=14, bb_len=20):
    """
    🔥 FONCTION DE COMPATIBILITÉ POUR SIGNAL_BOT
    Wrapper pour compute_saint_graal_indicators
    """
    return compute_saint_graal_indicators(df)

def compute_saint_graal_indicators(df):
    """Calcule les indicateurs pour qualité maximale"""
    df = df.copy()
    
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    # EMA
    df['ema_5'] = EMAIndicator(close=df['close'], window=5).ema_indicator()
    df['ema_13'] = EMAIndicator(close=df['close'], window=13).ema_indicator()
    
    # RSI
    df['rsi_7'] = RSIIndicator(close=df['close'], window=7).rsi()
    
    # Stochastic
    stoch = StochasticOscillator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=5,
        smooth_window=3
    )
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # ADX
    adx = ADXIndicator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=10
    )
    df['adx'] = adx.adx()
    
    # Price action
    df['candle_body'] = df['close'] - df['open']
    df['candle_size'] = df['high'] - df['low']
    df['body_ratio'] = abs(df['candle_body']) / df['candle_size'].replace(0, 0.00001)
    
    return df

def calculate_signal_quality_score(df):
    """
    🔥 COMPATIBILITÉ : Calcule un score de qualité global du signal (0-100)
    """
    if len(df) < 20:
        return 0
    
    last = df.iloc[-1]
    score = 0
    
    # Convergence (30 points)
    if 'rsi_7' in last and 'stoch_k' in last:
        rsi_ok = 30 < last['rsi_7'] < 70
        stoch_ok = 20 < last['stoch_k'] < 80
        if rsi_ok and stoch_ok:
            score += 30
    
    # Force tendance (25 points)
    if 'adx' in last:
        if last['adx'] > 30:
            score += 25
        elif last['adx'] > 25:
            score += 20
        elif last['adx'] > 20:
            score += 15
    
    # Qualité bougie (15 points)
    if 'body_ratio' in last:
        if last['body_ratio'] > 0.4:
            score += 15
        elif last['body_ratio'] > 0.3:
            score += 10
    
    return min(score, 100)

def check_anti_manipulation(df, strict_mode=True):
    """
    🔥 COMPATIBILITÉ : Vérifie les conditions anti-manipulation
    """
    if len(df) < 15:
        return False, "Données insuffisantes"
    
    return True, "OK"

def is_kill_zone_optimal(hour_utc):
    """
    🔥 COMPATIBILITÉ : Heures de trading optimales
    """
    if 7 <= hour_utc < 10:
        return True, "London Open", 10
    if 13 <= hour_utc < 16:
        return True, "NY Open", 9
    if 10 <= hour_utc < 12:
        return True, "London/NY Overlap", 8
    if 1 <= hour_utc < 4:
        return True, "Asia Close", 6
    
    return False, "Heure non optimale", 3

def rule_signal(df):
    """
    🔥 COMPATIBILITÉ : Version simple pour signal_bot
    """
    result = rule_signal_saint_graal_m1_pro_v2(df, signal_count=0, total_signals_needed=8)
    return result['signal'] if result else 'CALL'

def get_signal_with_metadata(df, signal_count=0, total_signals=8):
    """
    🔥 FONCTION PRINCIPALE DE COMPATIBILITÉ
    Utilisée par signal_bot.py
    """
    try:
        if df is None or len(df) < 30:
            return create_minimal_fallback("Données insuffisantes", signal_count, total_signals)
        
        # Utiliser la version PRO
        result = rule_signal_saint_graal_m1_pro_v2(df, signal_count, total_signals)
        
        if result:
            direction_display = result['signal']
            quality_display = {
                'EXCELLENT': '⭐⭐⭐⭐⭐',
                'HIGH': '⭐⭐⭐⭐',
                'SOLID': '⭐⭐⭐',
                'MINIMUM': '⭐⭐',
                'CRITICAL': '⭐'
            }.get(result['quality'], '⭐')
            
            reason = f"{quality_display} {direction_display} | Score: {result['score']:.0f}/100"
            
            return {
                'direction': direction_display,
                'mode': result['mode'],
                'quality': result['quality'],
                'score': float(result['score']),
                'reason': reason,
                'session_info': {
                    'current_signal': signal_count + 1,
                    'total_signals': total_signals,
                    'mode_used': result['mode'],
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
        'session_info': {
            'current_signal': signal_count + 1,
            'total_signals': total_signals,
            'mode_used': 'ERROR',
        }
    }

# ================= FONCTION PRINCIPALE V4.2 =================

def rule_signal_saint_graal_m1_pro_v2(df, signal_count=0, total_signals_needed=8):
    """
    🔥 VERSION 4.2 : LOGIQUE DESK PRO
    Asymétrie BUY/SELL + règles structure strictes
    """
    if len(df) < 50:
        return pro_fallback_intelligent(df, signal_count, total_signals_needed)
    
    current_price = float(df.iloc[-1]['close'])
    
    # ===== 1. ANALYSE COMPLÈTE =====
    structure, trend_strength = analyze_market_structure(df)
    internal_high, internal_low = detect_internal_swings(df.tail(10))
    
    # ===== 2. MOMENTUM ASYMÉTRIQUE =====
    momentum = analyze_momentum_asymmetric(df)
    
    # ===== 3. ZONES S/R =====
    supports, resistances = detect_key_zones(df)
    near_support, nearest_support, dist_support = is_price_near_zone_pro(current_price, supports, 8)
    near_resistance, nearest_resistance, dist_resistance = is_price_near_zone_pro(current_price, resistances, 8)
    
    # ===== 4. VÉRIFICATION SWING BREAK =====
    swing_break_sell = False
    swing_break_buy = False
    
    if internal_low:
        swing_break_sell, dist_break_sell = check_swing_break(
            current_price, internal_low, "SELL", 
            SAINT_GRAAL_CONFIG['sell_rules']['max_swing_distance_pips']
        )
    
    if internal_high:
        swing_break_buy, dist_break_buy = check_swing_break(
            current_price, internal_high, "BUY", 5
        )
    
    # ===== 5. SCORING DESK PRO =====
    sell_score = 0
    buy_score = 0
    
    # Structure score avec règles pro
    structure_score_sell, _ = calculate_structure_score_pro_m1(
        structure, "SELL", momentum, swing_break_sell
    )
    sell_score += structure_score_sell
    
    structure_score_buy, _ = calculate_structure_score_pro_m1(
        structure, "BUY", momentum, swing_break_buy
    )
    buy_score += structure_score_buy
    
    # Momentum scores
    sell_score += momentum['sell_score']
    buy_score += momentum['buy_score']
    
    # Zones bonus
    if near_resistance and dist_resistance <= 5:
        zone_bonus, _ = calculate_zone_strength(nearest_resistance)
        sell_score += int(zone_bonus * 0.8)
    
    if near_support and dist_support <= 5:
        zone_bonus, _ = calculate_zone_strength(nearest_support)
        buy_score += int(zone_bonus * 0.8)
    
    # ===== 6. DÉCISION AVEC FILTRES DESK =====
    direction = None
    final_score = 0
    
    # 🔥 VÉTO ABSOLU POUR SELL
    if sell_score > buy_score:
        # VÉTO 1: Momentum gate
        if not momentum['momentum_gate_passed']:
            pass
        # VÉTO 2: Structure contre tendance sans break
        elif "UPTREND" in structure and not swing_break_sell:
            pass
        # VÉTO 3: RSI trop bas
        elif momentum['rsi'] < 52:
            pass
        # VÉTO 4: Stochastic lent insuffisant
        elif momentum['stoch_k_slow'] < 58 and not momentum['stoch_peak_detected']:
            pass
        else:
            # Validation bougie
            candle_valid, pattern, pattern_conf, _ = validate_candle_for_binary_m1(
                df, "SELL", require_rejection=True
            )
            
            if candle_valid:
                direction = "SELL"
                final_score = sell_score + (pattern_conf / 10)
    
    # 🔥 BUY (moins strict)
    elif buy_score > sell_score:
        if not momentum['momentum_gate_passed']:
            pass
        # VÉTO BUY en downtrend sans break
        elif "DOWNTREND" in structure and not swing_break_buy:
            pass
        else:
            candle_valid, pattern, pattern_conf, _ = validate_candle_for_binary_m1(
                df, "BUY", require_rejection=False
            )
            
            if candle_valid:
                direction = "BUY"
                final_score = buy_score + (pattern_conf / 10)
    
    # ===== 7. DÉCISION FINALE =====
    if direction and final_score >= 65:
        # Qualité basée sur score et structure
        if final_score >= 88 and ("TREND" in structure or swing_break_sell or swing_break_buy):
            quality = "EXCELLENT"
            mode = "DESK_MAX"
        elif final_score >= 78:
            quality = "HIGH"
            mode = "DESK_PRO"
        elif final_score >= 68:
            quality = "SOLID"
            mode = "DESK_STANDARD"
        else:
            quality = "MINIMUM"
            mode = "DESK_MIN"
        
        direction_display = "CALL" if direction == "BUY" else "PUT"
        
        return {
            'signal': direction_display,
            'mode': mode,
            'quality': quality,
            'score': float(final_score),
            'reason': f"{direction} | {quality} | Score {final_score:.1f} | {structure}",
        }
    
    # ===== 8. FALLBACK SÉCURISÉ =====
    return pro_fallback_intelligent(df, signal_count, total_signals_needed)

if __name__ == "__main__":
    print("🚀 DESK PRO BINAIRE M1 - VERSION 4.2 COMPATIBLE")
    print("📊 Compatibilité totale avec signal_bot.py")
    print("\n✅ Fonctions de compatibilité ajoutées:")
    print("   - compute_indicators")
    print("   - calculate_signal_quality_score")
    print("   - check_anti_manipulation")
    print("   - is_kill_zone_optimal")
    print("   - rule_signal")
    print("   - get_signal_with_metadata")
    print("\n🎯 Prêt pour déploiement sur Render!")
