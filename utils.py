"""
utils.py - STRATÉGIE FOREX M1 PRO - ARCHITECTURE TRADER HIÉRARCHIQUE
Version 4.0 : Approche hiérarchique à 3 niveaux
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

# ================= CONFIGURATION HIÉRARCHIQUE =================

SAINT_GRAAL_CONFIG = {
    # Niveaux hiérarchiques
    'hierarchy': {
        'TOP': {
            'signal_range': (0, 3),      # Signaux 1-4
            'min_score': 85,
            'require_timing_break': True,
            'allow_counter_trend': False,
            'require_zone': True,
            'timing_type': 'STRICT',
            'max_fallbacks': 0,
            'quality_label': 'EXCELLENT',
        },
        'HIGH': {
            'signal_range': (4, 5),      # Signaux 5-6
            'min_score': 75,
            'require_timing_break': True,
            'allow_counter_trend': True,
            'counter_trend_penalty': 15,
            'require_zone': True,
            'timing_type': 'MODERATE',
            'max_fallbacks': 0,
            'quality_label': 'HIGH',
        },
        'STANDARD': {
            'signal_range': (6, 7),      # Signaux 7-8
            'min_score': 65,
            'require_timing_break': False,
            'allow_counter_trend': True,
            'counter_trend_penalty': 10,
            'require_zone': False,
            'timing_type': 'LENIENT',
            'max_fallbacks': 1,
            'quality_label': 'ACCEPTABLE',
        },
        'FALLBACK': {
            'signal_range': (8, 8),      # Signal 8 uniquement (fallback)
            'min_score': 55,
            'require_timing_break': False,
            'allow_counter_trend': True,
            'require_zone': False,
            'timing_type': 'NONE',
            'max_fallbacks': 2,
            'quality_label': 'MINIMUM',
        }
    },
    
    # Indicateurs M1
    'rsi_period': 7,
    'ema_fast': 5,
    'ema_slow': 13,
    'stoch_period': 5,
    
    # Seuils qualité généraux
    'max_quality': {
        'rsi_overbought': 68,
        'rsi_oversold': 32,
        'adx_min': 25,
        'stoch_overbought': 74,
        'stoch_oversold': 26,
        'min_confluence_points': 7,
        'min_body_ratio': 0.4,
        'max_wick_ratio': 0.4,
    },
    
    # Détection structure
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

# ================= DÉTECTION STRUCTURE =================

def analyze_market_structure(df, lookback=15):
    """Détection de structure avec classification hiérarchique"""
    if len(df) < lookback + 5:
        return "INSUFFICIENT_DATA", 0.0, [], []
    
    recent = df.tail(lookback).copy()
    highs = recent['high'].values
    lows = recent['low'].values
    
    swing_highs = []
    swing_lows = []
    
    for i in range(3, len(recent)-3):
        # Swing High
        if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
            highs[i] > highs[i-3] and
            highs[i] > highs[i+1] and highs[i] > highs[i+2] and
            highs[i] > highs[i+3]):
            
            min_height = recent['close'].mean() * 0.0002
            if highs[i] - max(highs[i-3], highs[i-2], highs[i-1],
                             highs[i+1], highs[i+2], highs[i+3]) > min_height:
                swing_highs.append({'index': i, 'price': float(highs[i])})
        
        # Swing Low
        if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
            lows[i] < lows[i-3] and
            lows[i] < lows[i+1] and lows[i] < lows[i+2] and
            lows[i] < lows[i+3]):
            
            min_depth = recent['close'].mean() * 0.0002
            if min(lows[i-3], lows[i-2], lows[i-1],
                  lows[i+1], lows[i+2], lows[i+3]) - lows[i] > min_depth:
                swing_lows.append({'index': i, 'price': float(lows[i])})
    
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
        elif is_hh and not is_hl and not is_ll:
            structure = "UPTREND_FAIBLE"
            trend_strength = float((high_prices[-1] - high_prices[0]) / high_prices[0] * 100) / 2
        elif is_lh and not is_hh and not is_hl:
            structure = "DOWNTREND_FAIBLE"
            trend_strength = float((low_prices[0] - low_prices[-1]) / low_prices[-1] * 100) / 2
        elif is_hh and is_ll:
            structure = "EXPANDING_WEDGE"
        elif is_lh and is_hl:
            structure = "CONTRACTING_WEDGE"
        else:
            price_range = max(high_prices) - min(low_prices)
            avg_candle = recent['high'].mean() - recent['low'].mean()
            structure = "TIGHT_RANGE" if price_range < avg_candle * 3 else "RANGE"
    
    return structure, trend_strength, swing_highs, swing_lows

# ================= ZONES S/R =================

def detect_key_zones(df, lookback=None, min_touches=None):
    """Détection des zones S/R avec clustering"""
    config = SAINT_GRAAL_CONFIG['zone_config']
    lookback = lookback or config['lookback_bars']
    min_touches = min_touches or config['min_touches']
    
    if len(df) < lookback:
        return [], []
    
    recent = df.tail(lookback).copy()
    
    # ATR pour tolérance dynamique
    atr = AverageTrueRange(
        high=recent['high'],
        low=recent['low'],
        close=recent['close'],
        window=14
    ).average_true_range()
    current_atr = atr.iloc[-1] if not atr.empty else 0.0005
    tolerance = current_atr * config['cluster_tolerance_atr_multiplier']
    
    resistances = []
    supports = []
    
    # Détection résistances
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
                })
    
    # Détection supports
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
                })
    
    # Calcul prix moyen pour clusters
    for zone in supports + resistances:
        if len(zone['prices']) > 1:
            zone['price'] = float(np.mean(zone['prices']))
    
    # Filtrer et trier
    valid_supports = [s for s in supports if s['touches'] >= min_touches]
    valid_resistances = [r for r in resistances if r['touches'] >= min_touches]
    
    valid_supports.sort(key=lambda x: x['touches'], reverse=True)
    valid_resistances.sort(key=lambda x: x['touches'], reverse=True)
    
    return valid_supports[:3], valid_resistances[:3]

def is_price_near_zone_pro(current_price, zones, max_distance_pips=10):
    """Vérification proximité zone"""
    if not zones:
        return False, None, float('inf')
    
    nearest_zone = None
    min_distance = float('inf')
    
    for zone in zones:
        distance = abs(current_price - zone['price'])
        if distance < min_distance:
            min_distance = distance
            nearest_zone = zone
    
    max_distance = max_distance_pips * 0.0001
    is_near = min_distance <= max_distance
    distance_pips = min_distance / 0.0001
    
    return is_near, nearest_zone, distance_pips

# ================= HIÉRARCHIE DE QUALITÉ =================

def get_hierarchy_level(signal_count):
    """Détermine le niveau hiérarchique basé sur le numéro du signal"""
    hierarchy = SAINT_GRAAL_CONFIG['hierarchy']
    
    for level_name, level_config in hierarchy.items():
        start, end = level_config['signal_range']
        if start <= signal_count <= end:
            return level_name, level_config
    
    # Par défaut, STANDARD
    return 'STANDARD', hierarchy['STANDARD']

def calculate_structure_score_hierarchical(structure, direction, near_support, near_resistance, level_config):
    """Calcul de score structure avec règles hiérarchiques"""
    
    if direction == "SELL":
        # Contre-tendance check
        if not level_config['allow_counter_trend']:
            if structure in ["UPTREND", "UPTREND_ACCELERATING"]:
                return -25, "🚫 Vente interdite en uptrend (niveau TOP)"
        
        # Calcul base
        if structure == "DOWNTREND":
            return 30, "DOWNTREND fort"
        elif structure == "DOWNTREND_ACCELERATING":
            return 35, "DOWNTREND accélérant"
        elif structure == "DOWNTREND_DECELERATING":
            return 25, "DOWNTREND ralentissant"
        elif structure == "UPTREND_DECELERATING" and near_resistance:
            return 20, "UPTREND ralentissant + résistance"
        elif structure == "EXPANDING_WEDGE" and near_resistance:
            return 22, "Wedge expansif + résistance"
        elif structure == "UPTREND" and near_resistance and level_config['allow_counter_trend']:
            penalty = level_config.get('counter_trend_penalty', 10)
            return 25 - penalty, f"Vente risquée en UPTREND (-{penalty})"
        elif structure in ["RANGE", "TIGHT_RANGE"]:
            if near_resistance:
                return 20, "Range + résistance"
            else:
                return 5 if level_config['require_zone'] else 15, "Range sans zone"
        elif structure in ["UPTREND_FAIBLE", "DOWNTREND_FAIBLE"]:
            if near_resistance:
                return 18, "Tendance faible + zone"
            else:
                return 3 if level_config['require_zone'] else 12, "Tendance faible sans zone"
        elif structure == "CHAOTIC":
            return -15, "Marché chaotique"
        else:
            return 10, f"Structure: {structure}"
    
    else:  # BUY
        # Contre-tendance check
        if not level_config['allow_counter_trend']:
            if structure in ["DOWNTREND", "DOWNTREND_ACCELERATING"]:
                return -25, "🚫 Achat interdit en downtrend (niveau TOP)"
        
        # Calcul base
        if structure == "UPTREND":
            return 30, "UPTREND fort"
        elif structure == "UPTREND_ACCELERATING":
            return 35, "UPTREND accélérant"
        elif structure == "UPTREND_DECELERATING":
            return 25, "UPTREND ralentissant"
        elif structure == "DOWNTREND_DECELERATING" and near_support:
            return 20, "DOWNTREND ralentissant + support"
        elif structure == "CONTRACTING_WEDGE" and near_support:
            return 22, "Wedge contractant + support"
        elif structure == "DOWNTREND" and near_support and level_config['allow_counter_trend']:
            penalty = level_config.get('counter_trend_penalty', 10)
            return 25 - penalty, f"Achat risqué en DOWNTREND (-{penalty})"
        elif structure in ["RANGE", "TIGHT_RANGE"]:
            if near_support:
                return 20, "Range + support"
            else:
                return 5 if level_config['require_zone'] else 15, "Range sans zone"
        elif structure in ["UPTREND_FAIBLE", "DOWNTREND_FAIBLE"]:
            if near_support:
                return 18, "Tendance faible + zone"
            else:
                return 3 if level_config['require_zone'] else 12, "Tendance faible sans zone"
        elif structure == "CHAOTIC":
            return -15, "Marché chaotique"
        else:
            return 10, f"Structure: {structure}"

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
            condition = current_candle['low'] <= prev_candle['low'] * 1.00005
            reason = f"Test du low précédent"
        else:  # BUY
            condition = current_candle['high'] >= prev_candle['high'] * 0.99995
            reason = f"Test du high précédent"
        
        return condition, reason if condition else f"❌ {reason}"
    
    else:  # LENIENT
        # Simple confirmation de direction
        if direction == "SELL":
            condition = current_candle['close'] < current_candle['open']
            reason = f"Bougie baissière"
        else:  # BUY
            condition = current_candle['close'] > current_candle['open']
            reason = f"Bougie haussière"
        
        return condition, reason if condition else f"❌ {reason}"

def calculate_zone_bonus_hierarchical(zone, level_config):
    """Bonus de zone selon niveau hiérarchique"""
    if not zone:
        return 0, "Aucune zone"
    
    touches = zone.get('touches', 1)
    
    # Base bonus selon niveau
    if level_config['require_zone']:
        base_bonus = 25
    else:
        base_bonus = 15
    
    # Ajustement selon touches
    if touches >= 4:
        bonus = base_bonus + 10
        strength = "TRÈS FORTE"
    elif touches >= 3:
        bonus = base_bonus + 5
        strength = "FORTE"
    elif touches >= 2:
        bonus = base_bonus
        strength = "MOYENNE"
    else:
        bonus = base_bonus - 5
        strength = "FAIBLE"
    
    # Réduction pour zones larges
    width_pips = zone.get('width_pips', 0)
    if width_pips > 10:
        bonus = int(bonus * 0.7)
        strength += " (large)"
    elif width_pips > 5:
        bonus = int(bonus * 0.85)
        strength += " (moyenne)"
    
    return bonus, f"{zone['type']} {strength}"

# ================= MOMENTUM =================

def analyze_momentum_pro(df):
    """Analyse momentum"""
    if len(df) < 20:
        return {"status": "INSUFFICIENT_DATA"}
    
    # RSI
    rsi = RSIIndicator(close=df['close'], window=SAINT_GRAAL_CONFIG['rsi_period']).rsi()
    current_rsi = rsi.iloc[-1]
    
    # Stochastic
    stoch = StochasticOscillator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=SAINT_GRAAL_CONFIG['stoch_period'],
        smooth_window=3
    )
    stoch_k = stoch.stoch()
    stoch_d = stoch.stoch_signal()
    
    # Calcul scores
    sell_score = 0
    buy_score = 0
    
    if current_rsi >= 60:
        sell_score += 20
    elif current_rsi >= 55:
        sell_score += 15
    
    if stoch_k.iloc[-1] >= 70:
        sell_score += 20
    elif stoch_k.iloc[-1] >= 50:
        sell_score += 15
    
    if current_rsi <= 40:
        buy_score += 20
    elif current_rsi <= 45:
        buy_score += 15
    
    if stoch_k.iloc[-1] <= 30:
        buy_score += 20
    elif stoch_k.iloc[-1] <= 50:
        buy_score += 15
    
    return {
        'rsi': float(current_rsi),
        'stoch_k': float(stoch_k.iloc[-1]),
        'stoch_d': float(stoch_d.iloc[-1]),
        'sell_score': sell_score,
        'buy_score': buy_score,
        'dominant': "SELL" if sell_score > buy_score else "BUY" if buy_score > sell_score else "NEUTRAL",
    }

# ================= VALIDATION BOUGIE =================

def validate_candle_pattern(df, direction):
    """Validation pattern bougie"""
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
        if upper_wick > candle_body * 2.0 and is_bearish:
            quality = 85 if upper_wick > candle_body * 3.0 else 75
            patterns.append(("PIN_BAR_BEARISH", quality))
        
        if (is_bearish and prev_candle['close'] > prev_candle['open'] and
            last_candle['open'] >= prev_candle['close'] and
            last_candle['close'] <= prev_candle['open']):
            patterns.append(("ENGULFING_BEARISH", 80))
        
        if last_candle['close'] < prev_candle['low']:
            patterns.append(("CLOSE_BELOW_PREV_LOW", 75))
    
    else:  # BUY
        if lower_wick > candle_body * 2.0 and is_bullish:
            quality = 85 if lower_wick > candle_body * 3.0 else 75
            patterns.append(("PIN_BAR_BULLISH", quality))
        
        if (is_bullish and prev_candle['close'] < prev_candle['open'] and
            last_candle['open'] <= prev_candle['close'] and
            last_candle['close'] >= prev_candle['open']):
            patterns.append(("ENGULFING_BULLISH", 80))
        
        if last_candle['close'] > prev_candle['high']:
            patterns.append(("CLOSE_ABOVE_PREV_HIGH", 75))
    
    if patterns:
        patterns.sort(key=lambda x: x[1], reverse=True)
        return True, patterns[0][0], patterns[0][1], "Bougie validée"
    
    return False, "NO_PATTERN", 0, "Aucun pattern bougie valide"

# ================= FALLBACK HIÉRARCHIQUE =================

def hierarchical_fallback(df, signal_count, total_signals, level_name, level_config, fallback_count=0):
    """Fallback adapté au niveau hiérarchique"""
    
    if fallback_count >= level_config['max_fallbacks']:
        return {
            'signal': 'NO_SIGNAL',
            'mode': 'WAIT',
            'quality': 'WAIT',
            'score': 0.0,
            'reason': f'Fallback max atteint ({fallback_count}/{level_config["max_fallbacks"]})',
            'hierarchy_info': {
                'level': level_name,
                'fallback_count': fallback_count,
                'max_fallbacks': level_config['max_fallbacks']
            }
        }
    
    # Calcul indicateurs simples pour fallback
    if len(df) < 20:
        return create_minimal_fallback(signal_count, total_signals, level_name)
    
    current_price = float(df.iloc[-1]['close'])
    ema_5 = df['close'].rolling(5).mean().iloc[-1]
    ema_13 = df['close'].rolling(13).mean().iloc[-1]
    
    rsi = RSIIndicator(close=df['close'], window=7).rsi().iloc[-1]
    stoch = StochasticOscillator(
        high=df['high'], low=df['low'], close=df['close'], window=5
    ).stoch().iloc[-1]
    
    # Logique fallback selon niveau
    if level_name == "HIGH":
        # Fallback restrictif
        if rsi < 35 and stoch < 25 and current_price > ema_5:
            return {
                'signal': 'CALL',
                'mode': 'FALLBACK_HIGH',
                'quality': 'MINIMUM',
                'score': 60.0,
                'reason': f'Fallback HIGH (RSI:{rsi:.1f}, Stoch:{stoch:.1f}, Price>EMA5)',
                'hierarchy_info': {'level': level_name, 'fallback_count': fallback_count + 1}
            }
        elif rsi > 65 and stoch > 75 and current_price < ema_5:
            return {
                'signal': 'PUT',
                'mode': 'FALLBACK_HIGH',
                'quality': 'MINIMUM',
                'score': 60.0,
                'reason': f'Fallback HIGH (RSI:{rsi:.1f}, Stoch:{stoch:.1f}, Price<EMA5)',
                'hierarchy_info': {'level': level_name, 'fallback_count': fallback_count + 1}
            }
    
    elif level_name == "STANDARD":
        # Fallback modéré
        if rsi < 40 and current_price > ema_13:
            return {
                'signal': 'CALL',
                'mode': 'FALLBACK_STANDARD',
                'quality': 'MINIMUM',
                'score': 55.0,
                'reason': f'Fallback STANDARD (RSI:{rsi:.1f}, Price>EMA13)',
                'hierarchy_info': {'level': level_name, 'fallback_count': fallback_count + 1}
            }
        elif rsi > 60 and current_price < ema_13:
            return {
                'signal': 'PUT',
                'mode': 'FALLBACK_STANDARD',
                'quality': 'MINIMUM',
                'score': 55.0,
                'reason': f'Fallback STANDARD (RSI:{rsi:.1f}, Price<EMA13)',
                'hierarchy_info': {'level': level_name, 'fallback_count': fallback_count + 1}
            }
    
    else:  # FALLBACK level
        # Fallback permissif
        if rsi < 50:
            return {
                'signal': 'CALL',
                'mode': 'FALLBACK_PERMISSIVE',
                'quality': 'CRITICAL',
                'score': 50.0,
                'reason': f'Fallback permissif (RSI:{rsi:.1f})',
                'hierarchy_info': {'level': level_name, 'fallback_count': fallback_count + 1}
            }
        else:
            return {
                'signal': 'PUT',
                'mode': 'FALLBACK_PERMISSIVE',
                'quality': 'CRITICAL',
                'score': 50.0,
                'reason': f'Fallback permissif (RSI:{rsi:.1f})',
                'hierarchy_info': {'level': level_name, 'fallback_count': fallback_count + 1}
            }
    
    # Si aucune condition remplie
    return {
        'signal': 'NO_SIGNAL',
        'mode': 'WAIT',
        'quality': 'WAIT',
        'score': 0.0,
        'reason': 'Aucune condition fallback remplie',
        'hierarchy_info': {'level': level_name, 'fallback_count': fallback_count}
    }

def create_minimal_fallback(signal_count, total_signals, level_name):
    """Fallback minimal"""
    return {
        'signal': 'CALL',
        'mode': 'FALLBACK_MINIMAL',
        'quality': 'CRITICAL',
        'score': 45.0,
        'reason': f'Fallback minimal (level:{level_name})',
        'hierarchy_info': {
            'level': level_name,
            'fallback_count': 1,
            'max_fallbacks': SAINT_GRAAL_CONFIG['hierarchy'][level_name]['max_fallbacks']
        }
    }

# ================= FONCTION PRINCIPALE HIÉRARCHIQUE =================

def rule_signal_saint_graal_hierarchical(df, signal_count=0, total_signals_needed=8, fallback_history=None):
    """
    STRATÉGIE - 8 signaux avec approche hiérarchique
    """
    if fallback_history is None:
        fallback_history = {'TOP': 0, 'HIGH': 0, 'STANDARD': 0, 'FALLBACK': 0}
    
    print(f"\n{'='*70}")
    print(f"🎯 HIÉRARCHIE PRO - Signal #{signal_count+1}/{total_signals_needed}")
    print(f"{'='*70}")
    
    if len(df) < 50:
        level_name, level_config = get_hierarchy_level(signal_count)
        return hierarchical_fallback(df, signal_count, total_signals_needed, level_name, level_config)
    
    # ===== 1. DÉTERMINER NIVEAU HIÉRARCHIQUE =====
    level_name, level_config = get_hierarchy_level(signal_count)
    fallback_count = fallback_history.get(level_name, 0)
    
    print(f"\n📊 NIVEAU: {level_name}")
    print(f"   Score min: {level_config['min_score']} | Timing: {level_config['timing_type']}")
    print(f"   Contre-tendance: {'Autorisé' if level_config['allow_counter_trend'] else 'Interdit'}")
    print(f"   Zone requise: {'Oui' if level_config['require_zone'] else 'Non'}")
    print(f"   Fallbacks utilisés: {fallback_count}/{level_config['max_fallbacks']}")
    
    # ===== 2. ANALYSE STRUCTURE =====
    structure, trend_strength, _, _ = analyze_market_structure(df)
    print(f"\n🏗️  STRUCTURE: {structure} (force: {trend_strength:.1f}%)")
    
    # ===== 3. ZONES S/R =====
    supports, resistances = detect_key_zones(df)
    current_price = float(df.iloc[-1]['close'])
    
    near_support, nearest_support, dist_support = is_price_near_zone_pro(
        current_price, supports, max_distance_pips=12
    )
    near_resistance, nearest_resistance, dist_resistance = is_price_near_zone_pro(
        current_price, resistances, max_distance_pips=12
    )
    
    print(f"📍 ZONES: Support {near_support} ({dist_support:.1f}p) | Résistance {near_resistance} ({dist_resistance:.1f}p)")
    
    # Vérifier condition zone requise
    if level_config['require_zone']:
        if not (near_support or near_resistance):
            print(f"   ❌ Zone requise manquante pour niveau {level_name}")
            return hierarchical_fallback(df, signal_count, total_signals_needed, level_name, level_config, fallback_count)
    
    # ===== 4. MOMENTUM =====
    momentum = analyze_momentum_pro(df)
    print(f"\n⚡ MOMENTUM: RSI {momentum['rsi']:.1f} | Stoch {momentum['stoch_k']:.1f}/{momentum['stoch_d']:.1f}")
    print(f"   Dominant: {momentum['dominant']}")
    
    # ===== 5. SCORING HIÉRARCHIQUE =====
    print(f"\n🎯 SCORING HIÉRARCHIQUE (niveau {level_name})")
    
    # Score SELL
    sell_score = 0
    sell_details = []
    
    # Structure
    structure_score_sell, structure_reason = calculate_structure_score_hierarchical(
        structure, "SELL", near_support, near_resistance, level_config
    )
    sell_score += structure_score_sell
    sell_details.append(structure_reason)
    
    # Momentum
    sell_score += momentum['sell_score']
    if momentum['sell_score'] > 0:
        sell_details.append(f"Momentum: {momentum['sell_score']}pts")
    
    # Zone bonus
    if near_resistance:
        zone_bonus, zone_reason = calculate_zone_bonus_hierarchical(nearest_resistance, level_config)
        sell_score += zone_bonus
        sell_details.append(zone_reason)
    
    # Score BUY
    buy_score = 0
    buy_details = []
    
    # Structure
    structure_score_buy, structure_reason = calculate_structure_score_hierarchical(
        structure, "BUY", near_support, near_resistance, level_config
    )
    buy_score += structure_score_buy
    buy_details.append(structure_reason)
    
    # Momentum
    buy_score += momentum['buy_score']
    if momentum['buy_score'] > 0:
        buy_details.append(f"Momentum: {momentum['buy_score']}pts")
    
    # Zone bonus
    if near_support:
        zone_bonus, zone_reason = calculate_zone_bonus_hierarchical(nearest_support, level_config)
        buy_score += zone_bonus
        buy_details.append(zone_reason)
    
    print(f"   SELL: {sell_score} - {', '.join(sell_details[:2])}")
    print(f"   BUY: {buy_score} - {', '.join(buy_details[:2])}")
    
    # ===== 6. DÉCISION ET VALIDATION =====
    direction = None
    final_score = 0
    validation_issues = []
    
    if sell_score >= level_config['min_score'] and sell_score > buy_score:
        # Validation timing
        timing_ok, timing_reason = validate_timing_hierarchical(df, "SELL", level_config['timing_type'])
        
        if level_config['require_timing_break'] and not timing_ok:
            validation_issues.append(f"Timing: {timing_reason}")
        else:
            # Validation bougie
            candle_valid, pattern, pattern_conf, candle_reason = validate_candle_pattern(df, "SELL")
            if candle_valid:
                direction = "SELL"
                final_score = sell_score + (pattern_conf / 10)
                if timing_ok:
                    final_score *= 1.1  # Bonus timing
                print(f"   ✅ Validation: {pattern} ({pattern_conf}%) | Timing: {timing_reason}")
            else:
                validation_issues.append(f"Bougie: {candle_reason}")
    
    elif buy_score >= level_config['min_score'] and buy_score > sell_score:
        # Validation timing
        timing_ok, timing_reason = validate_timing_hierarchical(df, "BUY", level_config['timing_type'])
        
        if level_config['require_timing_break'] and not timing_ok:
            validation_issues.append(f"Timing: {timing_reason}")
        else:
            # Validation bougie
            candle_valid, pattern, pattern_conf, candle_reason = validate_candle_pattern(df, "BUY")
            if candle_valid:
                direction = "BUY"
                final_score = buy_score + (pattern_conf / 10)
                if timing_ok:
                    final_score *= 1.1  # Bonus timing
                print(f"   ✅ Validation: {pattern} ({pattern_conf}%) | Timing: {timing_reason}")
            else:
                validation_issues.append(f"Bougie: {candle_reason}")
    
    # ===== 7. RÉSULTAT FINAL =====
    if direction:
        print(f"\n🎉 DÉCISION FINALE: {direction}")
        print(f"   Score: {final_score:.1f} | Niveau: {level_name} | Qualité: {level_config['quality_label']}")
        
        if direction == "SELL" and nearest_resistance:
            print(f"   📍 Cible: Résistance à {nearest_resistance['price']:.5f} ({nearest_resistance.get('touches', 1)} touches)")
        elif direction == "BUY" and nearest_support:
            print(f"   📍 Cible: Support à {nearest_support['price']:.5f} ({nearest_support.get('touches', 1)} touches)")
        
        print(f"{'='*70}")
        
        return {
            'signal': "PUT" if direction == "SELL" else "CALL",
            'mode': f'HIERARCHICAL_{level_name}',
            'quality': level_config['quality_label'],
            'score': float(final_score),
            'reason': f"{direction} | Niveau {level_name} | Score {final_score:.1f}",
            'hierarchy_info': {
                'level': level_name,
                'min_score': level_config['min_score'],
                'timing_type': level_config['timing_type'],
                'require_zone': level_config['require_zone'],
                'allow_counter_trend': level_config['allow_counter_trend'],
                'structure': structure,
                'trend_strength': float(trend_strength),
                'near_zone': near_resistance if direction == "SELL" else near_support,
                'zone_distance': dist_resistance if direction == "SELL" else dist_support,
            }
        }
    
    # ===== 8. FALLBACK =====
    print(f"\n⚡ AUCUN SIGNAL VALIDE - FALLBACK {level_name} ACTIVÉ")
    if fallback_count < level_config['max_fallbacks']:
        fallback_history[level_name] = fallback_count + 1
        return hierarchical_fallback(df, signal_count, total_signals_needed, level_name, level_config, fallback_count)
    else:
        print(f"   ❌ Maximum de fallbacks atteint pour le niveau {level_name}")
        return {
            'signal': 'NO_SIGNAL',
            'mode': 'WAIT',
            'quality': 'WAIT',
            'score': 0.0,
            'reason': f'Maximum de fallbacks atteint ({fallback_count}/{level_config["max_fallbacks"]})',
            'hierarchy_info': {'level': level_name, 'fallback_count': fallback_count}
        }

# ================= FONCTIONS DE COMPATIBILITÉ =================

def compute_saint_graal_indicators(df):
    """Compatibilité avec code existant"""
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
        high=df['high'], low=df['low'], close=df['close'], window=5, smooth_window=3
    )
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    return df

def get_signal_with_metadata(df, signal_count=0, total_signals=8):
    """Fonction principale pour le bot"""
    try:
        if df is None or len(df) < 30:
            level_name, _ = get_hierarchy_level(signal_count)
            return create_minimal_fallback(signal_count, total_signals, level_name)
        
        df_with_indicators = compute_saint_graal_indicators(df)
        
        if df_with_indicators.empty:
            level_name, _ = get_hierarchy_level(signal_count)
            return create_minimal_fallback(signal_count, total_signals, level_name)
        
        result = rule_signal_saint_graal_hierarchical(df_with_indicators, signal_count, total_signals)
        
        if result and result['signal'] != 'NO_SIGNAL':
            quality_display = {
                'EXCELLENT': '⭐⭐⭐⭐⭐',
                'HIGH': '⭐⭐⭐⭐',
                'ACCEPTABLE': '⭐⭐⭐',
                'MINIMUM': '⭐⭐',
                'CRITICAL': '⭐',
                'WAIT': '⏳'
            }.get(result['quality'], '⭐')
            
            reason = f"{quality_display} {result['signal']} | Score: {result['score']:.0f} | {result['mode']}"
            
            return {
                'direction': result['signal'],
                'mode': result['mode'],
                'quality': result['quality'],
                'score': float(result['score']),
                'reason': reason,
                'hierarchy_info': result.get('hierarchy_info', {}),
                'session_info': {
                    'current_signal': signal_count + 1,
                    'total_signals': total_signals,
                    'level': result.get('hierarchy_info', {}).get('level', 'UNKNOWN'),
                    'quality_level': result['quality']
                }
            }
        
        elif result and result['signal'] == 'NO_SIGNAL':
            return {
                'direction': 'WAIT',
                'mode': 'WAIT',
                'quality': 'WAIT',
                'score': 0.0,
                'reason': 'Aucun signal valide - Attente',
                'session_info': {
                    'current_signal': signal_count + 1,
                    'total_signals': total_signals,
                    'level': 'WAIT',
                    'quality_level': 'WAIT'
                }
            }
        
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
    
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
            'level': 'ERROR',
            'quality_level': 'CRITICAL'
        }
    }

def compute_indicators(df, ema_fast=8, ema_slow=21, rsi_len=14, bb_len=20):
    """Wrapper pour compatibilité"""
    return compute_saint_graal_indicators(df)

def rule_signal(df):
    """Version par défaut (compatibilité)"""
    result = rule_signal_saint_graal_hierarchical(df, signal_count=0, total_signals_needed=8)
    return result['signal'] if result and result['signal'] != 'NO_SIGNAL' else 'CALL'

def format_signal_reason(direction, confidence, indicators):
    """Compatibilité"""
    return f"{direction} | Confiance: {confidence}%"

# ================= INITIALISATION =================

if __name__ == "__main__":
    print("🎯 STRATÉGIE FOREX M1 - APPROCHE HIÉRARCHIQUE")
    print("📊 Version: 4.0 - Architecture à 4 niveaux")
    print("\n📋 HIÉRARCHIE DES SIGNAUX:")
    print("1. TOP (1-4): 85+ score, timing strict, pas de contre-tendance")
    print("2. HIGH (5-6): 75+ score, timing modéré, contre-tendance pénalisé")
    print("3. STANDARD (7): 65+ score, timing lenient, contre-tendance accepté")
    print("4. FALLBACK (8): 55+ score, règles minimales")
    print("\n🎯 OBJECTIF: 8 signaux avec qualité contrôlée")
    print("✅ Prêt pour le trading")
