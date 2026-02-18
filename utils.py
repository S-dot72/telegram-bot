"""
🚀 STRATÉGIE BINAIRE M1 PRO - VERSION 10.1 MULTI-TIMEFRAME
🔥 APPROCHE "THE SNIPER" : M15 (direction) + M5 (divergence/patterns) + M1 (entrée)
"""

import copy
import pandas as pd
import numpy as np
from datetime import datetime
from contextlib import contextmanager
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIGURATION AVANCÉE =================

SAINT_GRAAL_CONFIG = {
    'expiration_minutes': 5,

    # Mode Sniper (multi-timeframe strict)
    'sniper_mode': True,  # active la confluence M15 + M5 + M1

    # Paramètres pour les niveaux S/R
    'support_resistance': {
        'lookback_bars': 50,      # nombre de bougies M15 à analyser
        'tolerance_pips': 5,      # tolérance en pips pour considérer qu'on est sur un niveau
    },

    'forbidden_zones': {
        'no_buy_zone': {
            'enabled': True,
            'stoch_fast_max': 80,
            'rsi_max': 70,
            'bb_position_max': 75,
            'strict_mode': False,
            'penalty': 10,
        },
        'no_sell_zone': {
            'enabled': True,
            'stoch_fast_min': 20,
            'rsi_min': 30,
            'bb_position_min': 25,
            'strict_mode': False,
            'penalty': 10,
        },
        'swing_filter': {
            'enabled': True,
            'lookback_bars': 5,
            'no_buy_at_swing_high': True,
            'no_sell_at_swing_low': True,
            'strict_mode': False,
            'swing_penalty': 10,
            'swing_momentum_threshold': 999,
        }
    },

    'momentum_rules': {
        'buy_conditions': {
            'rsi_max': 62,
            'rsi_oversold': 35,
            'stoch_max': 55,
            'stoch_oversold': 25,
            'require_stoch_rising': True,
        },
        'sell_conditions': {
            'rsi_min': 45,
            'rsi_overbought': 62,
            'stoch_min': 45,
            'stoch_overbought': 70,
            'require_stoch_falling': True,
        },
        'momentum_gate_diff': 5,
        'smart_gate': True,
    },

    'micro_momentum': {
        'enabled': True,
        'lookback_bars': 3,
        'min_bullish_bars': 2,
        'min_bearish_bars': 2,
        'require_trend_alignment': True,
        'weight': 8,
    },

    'bollinger_config': {
        'window': 20,
        'window_dev': 2,
        'oversold_zone': 25,
        'overbought_zone': 75,
        'buy_zone_max': 50,
        'sell_zone_min': 50,
        'middle_band_weight': 10,
        'strict_mode': False,
        'penalty': 6,
    },

    'atr_filter': {
        'enabled': True,
        'window': 14,
        'min_atr_pips': 1,
        'max_atr_pips': 50,
        'optimal_range': [3, 25],
    },

    'm5_filter': {
        'enabled': True,
        'ema_fast': 21,
        'ema_slow': 50,
        'min_bars_required': 55,
        'weight': 8,
        'soft_veto': True,
        'max_score_against_trend': 999,
    },

    'market_state': {
        'enabled': True,
        'adx_threshold': 20,
        'rsi_range_threshold': 40,
        'prioritize_bb_in_range': True,
        'prioritize_momentum_in_trend': True,
    },

    'signal_validation': {
        'min_score': 55,
        'max_score_realistic': 120,
        'confidence_zones': {
            55: 62,
            65: 68,
            75: 74,
            85: 80,
            95: 86,
            105: 91,
        },
        'cooldown_bars': 2,
    },
}

# ================= MODE DÉGRADÉ =================

@contextmanager
def degraded_mode_config():
    """Applique temporairement une config allégée pour les données limitées."""
    original = copy.deepcopy(SAINT_GRAAL_CONFIG)
    try:
        SAINT_GRAAL_CONFIG['m5_filter']['enabled'] = False
        SAINT_GRAAL_CONFIG['market_state']['enabled'] = False
        SAINT_GRAAL_CONFIG['signal_validation']['min_score'] = 45
        yield
    finally:
        SAINT_GRAAL_CONFIG.update(original)

# ================= UTILITAIRES D'AGRÉGATION =================

def aggregate_to_timeframe(df_m1, timeframe_minutes):
    """
    Agrège des données M1 en M5 ou M15.
    - df_m1 : DataFrame avec colonnes ['open','high','low','close'] et index datetime
    - timeframe_minutes : 5 ou 15
    Retourne un DataFrame OHLC au timeframe demandé.
    """
    rule = f'{timeframe_minutes}T'
    ohlc = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }
    df_resampled = df_m1.resample(rule).agg(ohlc).dropna()
    return df_resampled

# ================= DÉTECTION DES NIVEAUX S/R SUR M15 =================

def detect_support_resistance(df_m15, lookback=None, tolerance_pips=None):
    """
    Détecte les niveaux de support et résistance à partir des swing highs/lows sur M15.
    Retourne deux listes : supports, resistances (prix).
    """
    if lookback is None:
        lookback = SAINT_GRAAL_CONFIG['support_resistance']['lookback_bars']
    if tolerance_pips is None:
        tolerance_pips = SAINT_GRAAL_CONFIG['support_resistance']['tolerance_pips']

    if len(df_m15) < lookback:
        return [], []

    highs = df_m15['high'].values[-lookback:]
    lows = df_m15['low'].values[-lookback:]

    # Détection des pivots (simplifiée : un point est un pivot s'il est le plus haut/bas sur 2 bars de chaque côté)
    swing_highs = []
    swing_lows = []
    for i in range(2, len(highs)-2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append(lows[i])

    # Conversion de la tolérance en valeur absolue (en fonction du prix moyen)
    avg_price = (df_m15['close'].mean())
    tolerance_abs = tolerance_pips * 0.0001 if avg_price < 50 else tolerance_pips * 0.01  # approximation

    def cluster(values, tol):
        if not values:
            return []
        values.sort()
        clusters = [[values[0]]]
        for v in values[1:]:
            if abs(v - clusters[-1][-1]) <= tol:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        return [np.mean(c) for c in clusters]

    resistances = cluster(swing_highs, tolerance_abs)
    supports = cluster(swing_lows, tolerance_abs)
    return supports, resistances

def is_near_level(price, levels, tolerance_pips, avg_price):
    """Vérifie si price est proche d'un niveau (tolérance en pips convertie)."""
    if not levels:
        return False
    tolerance_abs = tolerance_pips * 0.0001 if avg_price < 50 else tolerance_pips * 0.01
    for level in levels:
        if abs(price - level) <= tolerance_abs:
            return True
    return False

# ================= DÉTECTION DES DIVERGENCES RSI SUR M5 =================

def detect_rsi_divergence(df_m5, period=30):
    """
    Détecte une divergence haussière ou baissière sur les 30 dernières bougies M5.
    Retourne 'bullish', 'bearish' ou None.
    """
    if len(df_m5) < period:
        return None
    closes = df_m5['close'].values[-period:]
    rsi = RSIIndicator(close=df_m5['close'], window=14).rsi().values[-period:]

    # Trouver les minima et maxima locaux (simplifié : on utilise une fenêtre de 3)
    def find_pivots(series, order=3):
        highs = []
        lows = []
        for i in range(order, len(series)-order):
            if series[i] == max(series[i-order:i+order+1]):
                highs.append((i, series[i]))
            if series[i] == min(series[i-order:i+order+1]):
                lows.append((i, series[i]))
        return highs, lows

    price_highs, price_lows = find_pivots(closes)
    rsi_highs, rsi_lows = find_pivots(rsi)

    if len(price_lows) >= 2 and len(rsi_lows) >= 2:
        # Deux derniers plus bas
        pl1, pl2 = price_lows[-2], price_lows[-1]
        rl1, rl2 = rsi_lows[-2], rsi_lows[-1]
        if pl2[1] < pl1[1] and rl2[1] > rl1[1]:
            return 'bullish'

    if len(price_highs) >= 2 and len(rsi_highs) >= 2:
        ph1, ph2 = price_highs[-2], price_highs[-1]
        rh1, rh2 = rsi_highs[-2], rsi_highs[-1]
        if ph2[1] > ph1[1] and rh2[1] < rh1[1]:
            return 'bearish'

    return None

# ================= DÉTECTION DES PATTERNS DE BOUGIES =================

def is_pin_bar(candle, direction=None):
    """
    candle : dict avec 'open','high','low','close'
    direction : 'buy' pour pin bar bas (longue mèche basse), 'sell' pour haute.
    Retourne True si pattern détecté.
    """
    body = abs(candle['close'] - candle['open'])
    high_low = candle['high'] - candle['low']
    if high_low == 0:
        return False
    body_ratio = body / high_low
    if body_ratio > 0.3:  # corps trop grand
        return False
    lower_wick = min(candle['open'], candle['close']) - candle['low']
    upper_wick = candle['high'] - max(candle['open'], candle['close'])
    if direction == 'buy' and lower_wick / high_low > 0.6:
        return True
    if direction == 'sell' and upper_wick / high_low > 0.6:
        return True
    if direction is None:
        return (lower_wick / high_low > 0.6) or (upper_wick / high_low > 0.6)
    return False

def is_engulfing(prev, curr):
    """
    Vérifie si la bougie curr englobe la précédente.
    Retourne 'bullish' ou 'bearish' ou None.
    """
    prev_body = abs(prev['close'] - prev['open'])
    curr_body = abs(curr['close'] - curr['open'])
    prev_dir = 1 if prev['close'] > prev['open'] else -1
    curr_dir = 1 if curr['close'] > curr['open'] else -1
    if curr_dir != prev_dir and curr_body > prev_body:
        if curr_dir == 1 and curr['close'] > prev['high']:
            return 'bullish'
        if curr_dir == -1 and curr['close'] < prev['low']:
            return 'bearish'
    return None

# ================= FONCTIONS D'ANALYSE EXISTANTES (légèrement adaptées) =================

def detect_market_state(df):
    """Détecte si le marché est en TREND ou RANGE (inchangé)"""
    if len(df) < 25:
        return {'state': 'NEUTRAL', 'adx': 0, 'reason': 'Données insuffisantes'}
    try:
        adx_indicator = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
        adx = float(adx_indicator.adx().iloc[-1])
        if np.isnan(adx):
            adx = 0
    except Exception:
        adx = 0
    rsi = RSIIndicator(close=df['close'], window=14).rsi()
    current_rsi = float(rsi.iloc[-1])
    if np.isnan(current_rsi):
        current_rsi = 50
    rsi_range_threshold = SAINT_GRAAL_CONFIG['market_state']['rsi_range_threshold']
    in_rsi_range = abs(current_rsi - 50) < (50 - rsi_range_threshold)
    if adx >= SAINT_GRAAL_CONFIG['market_state']['adx_threshold']:
        state = "TREND"
        reason = f"ADX fort: {adx:.1f}"
    elif in_rsi_range:
        state = "RANGE"
        reason = f"Range RSI: {current_rsi:.1f}"
    else:
        state = "NEUTRAL"
        reason = f"ADX: {adx:.1f}, RSI: {current_rsi:.1f}"
    return {'state': state, 'adx': adx, 'rsi': current_rsi, 'reason': reason}

def calculate_momentum_gate(df, direction, momentum_data):
    """Gate momentum réduit à 1/3 conditions (inchangé)"""
    if not SAINT_GRAAL_CONFIG['momentum_rules']['smart_gate']:
        stoch_diff = abs(momentum_data['stoch_k'] - momentum_data['stoch_d'])
        return stoch_diff >= SAINT_GRAAL_CONFIG['momentum_rules']['momentum_gate_diff'], {}
    gate_score = 0
    stoch_diff = abs(momentum_data['stoch_k'] - momentum_data['stoch_d'])
    if stoch_diff >= SAINT_GRAAL_CONFIG['momentum_rules']['momentum_gate_diff']:
        gate_score += 1
    if direction == "BUY":
        if momentum_data['rsi'] >= momentum_data.get('prev_rsi', momentum_data['rsi']):
            gate_score += 1
    else:
        if momentum_data['rsi'] <= momentum_data.get('prev_rsi', momentum_data['rsi']):
            gate_score += 1
    if len(df) >= 3:
        last_closes = df['close'].values[-3:]
        if direction == "BUY":
            if last_closes[-1] > last_closes[-2] or last_closes[-2] > last_closes[-3]:
                gate_score += 1
        else:
            if last_closes[-1] < last_closes[-2] or last_closes[-2] < last_closes[-3]:
                gate_score += 1
    debug_info = {'direction': direction, 'gate_score': gate_score, 'stoch_diff': stoch_diff}
    return gate_score >= 1, debug_info

def analyze_momentum_with_filters(df):
    """Analyse momentum (inchangé)"""
    if len(df) < 20:
        return {
            'rsi': 50, 'stoch_k': 50, 'stoch_d': 50, 'prev_rsi': 50,
            'buy': {'allowed': True, 'veto': False, 'score': 0, 'penalty': 0, 'reason': 'Données insuffisantes'},
            'sell': {'allowed': True, 'veto': False, 'score': 0, 'penalty': 0, 'reason': 'Données insuffisantes'},
            'gate_buy': False, 'gate_sell': False, 'violations': []
        }
    rsi_series = RSIIndicator(close=df['close'], window=14).rsi()
    current_rsi = float(rsi_series.iloc[-1])
    prev_rsi = float(rsi_series.iloc[-2]) if len(rsi_series) > 1 else current_rsi
    if np.isnan(current_rsi): current_rsi = 50
    if np.isnan(prev_rsi): prev_rsi = current_rsi
    stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    stoch_k_series = stoch.stoch()
    stoch_d_series = stoch.stoch_signal()
    current_stoch_k = float(stoch_k_series.iloc[-1])
    current_stoch_d = float(stoch_d_series.iloc[-1])
    prev_stoch_k = float(stoch_k_series.iloc[-2]) if len(stoch_k_series) > 1 else current_stoch_k
    if np.isnan(current_stoch_k): current_stoch_k = 50
    if np.isnan(current_stoch_d): current_stoch_d = 50
    if np.isnan(prev_stoch_k): prev_stoch_k = current_stoch_k

    buy_result = {'allowed': True, 'veto': False, 'score': 0, 'penalty': 0, 'reason': []}
    sell_result = {'allowed': True, 'veto': False, 'score': 0, 'penalty': 0, 'reason': []}
    violations = []

    no_buy_zone = SAINT_GRAAL_CONFIG['forbidden_zones']['no_buy_zone']
    no_sell_zone = SAINT_GRAAL_CONFIG['forbidden_zones']['no_sell_zone']

    if no_buy_zone['enabled']:
        if current_stoch_k > no_buy_zone['stoch_fast_max']:
            buy_result['penalty'] += no_buy_zone['penalty']
            buy_result['reason'].append(f"Stoch haut pénalité: -{no_buy_zone['penalty']}")
            violations.append(f"⚠️ BUY pénalité: Stoch {current_stoch_k:.1f} > {no_buy_zone['stoch_fast_max']}")
        if current_rsi > no_buy_zone['rsi_max']:
            buy_result['penalty'] += no_buy_zone['penalty']
            buy_result['reason'].append(f"RSI haut pénalité: -{no_buy_zone['penalty']}")
            violations.append(f"⚠️ BUY pénalité: RSI {current_rsi:.1f} > {no_buy_zone['rsi_max']}")
    if no_sell_zone['enabled']:
        if current_stoch_k < no_sell_zone['stoch_fast_min']:
            sell_result['penalty'] += no_sell_zone['penalty']
            sell_result['reason'].append(f"Stoch bas pénalité: -{no_sell_zone['penalty']}")
            violations.append(f"⚠️ SELL pénalité: Stoch {current_stoch_k:.1f} < {no_sell_zone['stoch_fast_min']}")
        if current_rsi < no_sell_zone['rsi_min']:
            sell_result['penalty'] += no_sell_zone['penalty']
            sell_result['reason'].append(f"RSI bas pénalité: -{no_sell_zone['penalty']}")
            violations.append(f"⚠️ SELL pénalité: RSI {current_rsi:.1f} < {no_sell_zone['rsi_min']}")

    momentum_data = {
        'rsi': current_rsi, 'stoch_k': current_stoch_k,
        'stoch_d': current_stoch_d, 'prev_rsi': prev_rsi
    }

    # Score BUY
    buy_score = 0
    buy_conds = SAINT_GRAAL_CONFIG['momentum_rules']['buy_conditions']
    if current_rsi < buy_conds['rsi_max']:
        buy_score += 15
        buy_result['reason'].append(f"RSI OK: {current_rsi:.1f}")
        if current_rsi < buy_conds['rsi_oversold']:
            buy_score += 10
            buy_result['reason'].append("RSI OVERSOLD +10")
    if current_stoch_k < buy_conds['stoch_max']:
        buy_score += 12
        buy_result['reason'].append(f"Stoch OK: {current_stoch_k:.1f}")
        if current_stoch_k < buy_conds['stoch_oversold']:
            buy_score += 8
            buy_result['reason'].append("Stoch OVERSOLD +8")
    if buy_conds['require_stoch_rising']:
        if current_stoch_k > prev_stoch_k:
            buy_score += 8
            buy_result['reason'].append("Stoch rising +8")
    if current_stoch_k > current_stoch_d and prev_stoch_k <= current_stoch_d:
        buy_score += 6
        buy_result['reason'].append("Stoch K/D cross bullish +6")
    buy_score = max(0, buy_score - buy_result['penalty'])
    buy_result['score'] = buy_score

    # Score SELL
    sell_score = 0
    sell_conds = SAINT_GRAAL_CONFIG['momentum_rules']['sell_conditions']
    if current_rsi > sell_conds['rsi_min']:
        sell_score += 15
        sell_result['reason'].append(f"RSI haut: {current_rsi:.1f}")
        if current_rsi > sell_conds['rsi_overbought']:
            sell_score += 10
            sell_result['reason'].append("RSI OVERBOUGHT +10")
    if current_stoch_k > sell_conds['stoch_min']:
        sell_score += 12
        sell_result['reason'].append(f"Stoch haut: {current_stoch_k:.1f}")
        if current_stoch_k > sell_conds['stoch_overbought']:
            sell_score += 8
            sell_result['reason'].append("Stoch OVERBOUGHT +8")
    if sell_conds['require_stoch_falling']:
        if current_stoch_k < prev_stoch_k:
            sell_score += 8
            sell_result['reason'].append("Stoch falling +8")
    if current_stoch_k < current_stoch_d and prev_stoch_k >= current_stoch_d:
        sell_score += 6
        sell_result['reason'].append("Stoch K/D cross bearish +6")
    sell_score = max(0, sell_score - sell_result['penalty'])
    sell_result['score'] = sell_score

    gate_buy, debug_buy = calculate_momentum_gate(df, "BUY", momentum_data)
    gate_sell, debug_sell = calculate_momentum_gate(df, "SELL", momentum_data)

    buy_result['reason'] = " | ".join(buy_result['reason']) if buy_result['reason'] else "Neutre"
    sell_result['reason'] = " | ".join(sell_result['reason']) if sell_result['reason'] else "Neutre"

    return {
        'rsi': current_rsi, 'stoch_k': current_stoch_k, 'stoch_d': current_stoch_d, 'prev_rsi': prev_rsi,
        'buy': buy_result, 'sell': sell_result,
        'gate_buy': gate_buy, 'gate_sell': gate_sell,
        'gate_debug': {'buy': debug_buy, 'sell': debug_sell},
        'violations': violations
    }

def analyze_bollinger_bands(df):
    """Analyse BB (inchangé)"""
    if len(df) < SAINT_GRAAL_CONFIG['bollinger_config']['window']:
        return {
            'bb_position': 50,
            'buy': {'allowed': True, 'veto': False, 'score': 0, 'penalty': 0, 'reason': 'Données insuffisantes'},
            'sell': {'allowed': True, 'veto': False, 'score': 0, 'penalty': 0, 'reason': 'Données insuffisantes'},
            'price_above_middle': False
        }
    bb = BollingerBands(close=df['close'], window=SAINT_GRAAL_CONFIG['bollinger_config']['window'], window_dev=SAINT_GRAAL_CONFIG['bollinger_config']['window_dev'])
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_middle = bb.bollinger_mavg()
    current_price = float(df.iloc[-1]['close'])
    current_upper = float(bb_upper.iloc[-1])
    current_lower = float(bb_lower.iloc[-1])
    current_middle = float(bb_middle.iloc[-1])
    if current_upper != current_lower:
        bb_position = ((current_price - current_lower) / (current_upper - current_lower)) * 100
    else:
        bb_position = 50

    bb_config = SAINT_GRAAL_CONFIG['bollinger_config']
    no_buy_zone = SAINT_GRAAL_CONFIG['forbidden_zones']['no_buy_zone']
    no_sell_zone = SAINT_GRAAL_CONFIG['forbidden_zones']['no_sell_zone']

    buy_result = {'allowed': True, 'veto': False, 'score': 0, 'penalty': 0, 'reason': []}
    sell_result = {'allowed': True, 'veto': False, 'score': 0, 'penalty': 0, 'reason': []}

    if no_buy_zone['enabled'] and bb_position > no_buy_zone['bb_position_max']:
        buy_result['penalty'] += bb_config['penalty']
        buy_result['reason'].append(f"BB zone haute: -{bb_config['penalty']}")
    if no_sell_zone['enabled'] and bb_position < no_sell_zone['bb_position_min']:
        sell_result['penalty'] += bb_config['penalty']
        sell_result['reason'].append(f"BB zone basse: -{bb_config['penalty']}")

    if bb_position < bb_config['buy_zone_max']:
        buy_result['score'] = 20
        if bb_position < bb_config['oversold_zone']:
            buy_result['score'] += 12
            buy_result['reason'].append("BB OVERSOLD +12")
        else:
            buy_result['reason'].append(f"BB zone BUY ({bb_position:.0f}%)")
    buy_result['score'] = max(0, buy_result['score'] - buy_result['penalty'])

    if bb_position > bb_config['sell_zone_min']:
        sell_result['score'] = 20
        if bb_position > bb_config['overbought_zone']:
            sell_result['score'] += 12
            sell_result['reason'].append("BB OVERBOUGHT +12")
        else:
            sell_result['reason'].append(f"BB zone SELL ({bb_position:.0f}%)")
    sell_result['score'] = max(0, sell_result['score'] - sell_result['penalty'])

    if len(df) >= 2:
        prev_price = float(df.iloc[-2]['close'])
        if prev_price <= current_middle < current_price:
            buy_result['score'] += bb_config['middle_band_weight']
            buy_result['reason'].append("Bullish cross médiane +10")
        elif prev_price >= current_middle > current_price:
            sell_result['score'] += bb_config['middle_band_weight']
            sell_result['reason'].append("Bearish cross médiane +10")

    buy_result['reason'] = " | ".join(buy_result['reason']) if buy_result['reason'] else f"BB: {bb_position:.1f}%"
    sell_result['reason'] = " | ".join(sell_result['reason']) if sell_result['reason'] else f"BB: {bb_position:.1f}%"

    return {
        'bb_position': bb_position,
        'buy': buy_result, 'sell': sell_result,
        'price_above_middle': current_price > current_middle
    }

def analyze_atr_volatility(df):
    """Analyse ATR (inchangé)"""
    if len(df) < 15:
        return {'valid': True, 'reason': 'ATR ignoré (peu de données)', 'score': 5, 'atr_pips': 0}
    try:
        atr_indicator = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=SAINT_GRAAL_CONFIG['atr_filter']['window'])
        atr = float(atr_indicator.average_true_range().iloc[-1])
        if np.isnan(atr) or atr == 0:
            return {'valid': True, 'reason': 'ATR non calculable', 'score': 5, 'atr_pips': 0}
    except Exception:
        return {'valid': True, 'reason': 'ATR erreur calcul', 'score': 5, 'atr_pips': 0}
    last_price = float(df.iloc[-1]['close'])
    if last_price > 50:
        atr_pips = atr * 100
    else:
        atr_pips = atr * 10000
    config = SAINT_GRAAL_CONFIG['atr_filter']
    if not config['enabled']:
        return {'valid': True, 'reason': 'ATR désactivé', 'score': 5, 'atr_pips': atr_pips}
    if atr_pips < config['min_atr_pips']:
        return {'valid': True, 'reason': f'ATR faible: {atr_pips:.1f} pips', 'score': 2, 'atr_pips': atr_pips}
    if atr_pips > config['max_atr_pips']:
        return {'valid': True, 'reason': f'ATR élevé: {atr_pips:.1f} pips', 'score': 2, 'atr_pips': atr_pips}
    if config['optimal_range'][0] <= atr_pips <= config['optimal_range'][1]:
        return {'valid': True, 'reason': f'ATR optimal: {atr_pips:.1f} pips', 'score': 10, 'atr_pips': atr_pips}
    return {'valid': True, 'reason': f'ATR acceptable: {atr_pips:.1f} pips', 'score': 5, 'atr_pips': atr_pips}

def analyze_m5_trend(df):
    """Analyse tendance M5 avec EMA21/EMA50 (inchangé)"""
    min_bars = SAINT_GRAAL_CONFIG['m5_filter']['min_bars_required']
    if len(df) < min_bars:
        return {'trend': 'NEUTRAL', 'reason': f'Données insuffisantes ({len(df)}/{min_bars})', 'score': 5}
    if not SAINT_GRAAL_CONFIG['m5_filter']['enabled']:
        return {'trend': 'NEUTRAL', 'reason': 'Filtre M5 désactivé', 'score': 5}
    try:
        ema_fast = EMAIndicator(close=df['close'], window=SAINT_GRAAL_CONFIG['m5_filter']['ema_fast']).ema_indicator()
        ema_slow = EMAIndicator(close=df['close'], window=SAINT_GRAAL_CONFIG['m5_filter']['ema_slow']).ema_indicator()
        current_fast = float(ema_fast.iloc[-1])
        current_slow = float(ema_slow.iloc[-1])
        if np.isnan(current_fast) or np.isnan(current_slow):
            return {'trend': 'NEUTRAL', 'reason': 'EMA NaN', 'score': 5}
    except Exception:
        return {'trend': 'NEUTRAL', 'reason': 'Erreur calcul EMA', 'score': 5}
    ema_fast_n = SAINT_GRAAL_CONFIG['m5_filter']['ema_fast']
    ema_slow_n = SAINT_GRAAL_CONFIG['m5_filter']['ema_slow']
    if current_fast > current_slow * 1.0005:
        return {'trend': 'BULLISH', 'reason': f"Tendance haussière: EMA{ema_fast_n} > EMA{ema_slow_n}", 'score': SAINT_GRAAL_CONFIG['m5_filter']['weight']}
    elif current_fast < current_slow * 0.9995:
        return {'trend': 'BEARISH', 'reason': f"Tendance baissière: EMA{ema_fast_n} < EMA{ema_slow_n}", 'score': SAINT_GRAAL_CONFIG['m5_filter']['weight']}
    else:
        return {'trend': 'NEUTRAL', 'reason': f"Tendance neutre: EMA{ema_fast_n} ≈ EMA{ema_slow_n}", 'score': 5}

def detect_swing_extremes(df):
    """Détecte les swing highs et lows (inchangé)"""
    lookback = SAINT_GRAAL_CONFIG['forbidden_zones']['swing_filter']['lookback_bars']
    if len(df) < lookback + 1:
        return {'is_swing_high': False, 'is_swing_low': False}
    highs = df['high'].values[-lookback:]
    lows = df['low'].values[-lookback:]
    is_swing_high = float(highs[-1]) == float(max(highs))
    is_swing_low = float(lows[-1]) == float(min(lows))
    return {'is_swing_high': is_swing_high, 'is_swing_low': is_swing_low}

def analyze_micro_momentum(df, direction):
    """Analyse micro momentum (inchangé)"""
    if not SAINT_GRAAL_CONFIG['micro_momentum']['enabled']:
        return {'valid': True, 'score': 0, 'reason': 'Micro momentum désactivé'}
    lookback = SAINT_GRAAL_CONFIG['micro_momentum']['lookback_bars']
    if len(df) < lookback + 1:
        return {'valid': True, 'score': 0, 'reason': 'Données insuffisantes — ignoré'}
    closes = df['close'].values[-(lookback + 1):]
    if direction == "BUY":
        bullish = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i - 1])
        if bullish >= SAINT_GRAAL_CONFIG['micro_momentum']['min_bullish_bars']:
            return {'valid': True, 'score': SAINT_GRAAL_CONFIG['micro_momentum']['weight'], 'reason': f'Micro momentum haussier: {bullish}/{lookback} bougies vertes'}
    else:
        bearish = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i - 1])
        if bearish >= SAINT_GRAAL_CONFIG['micro_momentum']['min_bearish_bars']:
            return {'valid': True, 'score': SAINT_GRAAL_CONFIG['micro_momentum']['weight'], 'reason': f'Micro momentum baissier: {bearish}/{lookback} bougies rouges'}
    return {'valid': False, 'score': 0, 'reason': 'Micro momentum insuffisant'}

def check_confidence_killers(df, direction, momentum_data):
    """Vérifie les facteurs réduisant la confiance (inchangé)"""
    confidence_reduction = 0
    killers = []
    if len(df) >= 15:
        try:
            closes = df['close'].values[-15:]
            rsis = RSIIndicator(close=pd.Series(closes), window=14).rsi().values
            valid = rsis[~np.isnan(rsis)]
            if len(valid) >= 5:
                rsi_trend = np.polyfit(range(5), valid[-5:], 1)[0]
                price_trend = np.polyfit(range(5), closes[-5:], 1)[0]
                if direction == "BUY" and price_trend > 0 and rsi_trend < 0:
                    confidence_reduction += 5
                    killers.append("Divergence RSI baissière")
                elif direction == "SELL" and price_trend < 0 and rsi_trend > 0:
                    confidence_reduction += 5
                    killers.append("Divergence RSI haussière")
        except Exception:
            pass
    try:
        candle = df.iloc[-1]
        body_size = abs(float(candle['close']) - float(candle['open']))
        total_range = float(candle['high']) - float(candle['low'])
        if total_range > 0 and body_size > 0:
            if direction == "BUY":
                upper_wick = float(candle['high']) - max(float(candle['open']), float(candle['close']))
                if upper_wick > body_size * 2.5:
                    confidence_reduction += 3
                    killers.append("Grande mèche haute")
            else:
                lower_wick = min(float(candle['open']), float(candle['close'])) - float(candle['low'])
                if lower_wick > body_size * 2.5:
                    confidence_reduction += 3
                    killers.append("Grande mèche basse")
    except Exception:
        pass
    return confidence_reduction, killers

def calculate_confidence(score):
    """Calcule la confiance à partir du score (inchangé)"""
    zones = sorted(SAINT_GRAAL_CONFIG['signal_validation']['confidence_zones'].items())
    max_realistic = SAINT_GRAAL_CONFIG['signal_validation']['max_score_realistic']
    normalized = min(score, max_realistic)
    base_confidence = 60
    for threshold, conf in zones:
        if normalized >= threshold:
            base_confidence = conf
    for i in range(len(zones) - 1):
        ct, cc = zones[i]
        nt, nc = zones[i + 1]
        if ct <= normalized < nt:
            progress = (normalized - ct) / (nt - ct)
            base_confidence = cc + (nc - cc) * progress
            break
    return min(95, max(60, int(base_confidence)))

# ================= FONCTION PRINCIPALE MULTI-TIMEFRAME =================

def analyze_pair_for_signals(df_m1, df_m5=None, df_m15=None):
    """
    Analyse complète avec multi-timeframe.
    - df_m1 : DataFrame M1 obligatoire
    - df_m5 : DataFrame M5 (optionnel, sera agrégé si non fourni)
    - df_m15 : DataFrame M15 (optionnel, sera agrégé si non fourni)
    """
    # Agrégation si nécessaire
    if df_m5 is None:
        df_m5 = aggregate_to_timeframe(df_m1, 5)
    if df_m15 is None:
        df_m15 = aggregate_to_timeframe(df_m1, 15)

    # Vérification de la longueur minimale
    if len(df_m1) < 60:
        print(f"⚠️  Données M1 limitées ({len(df_m1)} bougies) — mode dégradé activé")
        ctx = degraded_mode_config()
        ctx.__enter__()
    else:
        ctx = None

    try:
        return _run_analysis_multi(df_m1, df_m5, df_m15)
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)

def _run_analysis_multi(df_m1, df_m5, df_m15):
    current_price = float(df_m1.iloc[-1]['close'])
    print(f"\n{'='*60}")
    print(f"🔍 ANALYSE MULTI-TIMEFRAME — Prix M1: {current_price:.5f}")
    print(f"{'='*60}")

    # ========== ANALYSE M15 ==========
    market = detect_market_state(df_m15)
    print(f"📊 M15: {market['state']} — {market['reason']}")

    # Détection des niveaux S/R sur M15
    supports, resistances = detect_support_resistance(df_m15)
    avg_price = df_m15['close'].mean()
    tol_pips = SAINT_GRAAL_CONFIG['support_resistance']['tolerance_pips']
    near_support = is_near_level(current_price, supports, tol_pips, avg_price)
    near_resistance = is_near_level(current_price, resistances, tol_pips, avg_price)
    if near_support:
        print(f"   ✅ Prix proche d'un support M15")
    if near_resistance:
        print(f"   ✅ Prix proche d'une résistance M15")

    # ========== ANALYSE M5 ==========
    # Divergence RSI
    divergence = detect_rsi_divergence(df_m5)
    if divergence:
        print(f"📈 M5: Divergence {divergence} détectée")

    # Patterns sur la dernière bougie M5
    last_m5 = df_m5.iloc[-1].to_dict()
    prev_m5 = df_m5.iloc[-2].to_dict() if len(df_m5) > 1 else None
    pin_m5_buy = is_pin_bar(last_m5, 'buy')
    pin_m5_sell = is_pin_bar(last_m5, 'sell')
    engulf_m5 = is_engulfing(prev_m5, last_m5) if prev_m5 is not None else None
    if pin_m5_buy or pin_m5_sell or engulf_m5:
        print(f"📊 M5: Pattern détecté (Pin/Engulfing)")

    # ========== ANALYSE M1 ==========
    # Indicateurs classiques (momentum, BB, etc.) sur M1
    momentum = analyze_momentum_with_filters(df_m1)
    print(f"📈 MOMENTUM M1: RSI {momentum['rsi']:.1f} | Stoch {momentum['stoch_k']:.1f}/{momentum['stoch_d']:.1f}")
    print(f"   BUY  score: {momentum['buy']['score']}  | Gate: {'✅' if momentum['gate_buy'] else '❌'}")
    print(f"   SELL score: {momentum['sell']['score']} | Gate: {'✅' if momentum['gate_sell'] else '❌'}")
    for v in momentum['violations']:
        print(f"   {v}")

    bb = analyze_bollinger_bands(df_m1)
    print(f"📊 BB M1: position {bb['bb_position']:.1f}% | BUY={bb['buy']['score']} SELL={bb['sell']['score']}")

    swings = detect_swing_extremes(df_m1)
    swing_filter = SAINT_GRAAL_CONFIG['forbidden_zones']['swing_filter']
    swing_adj = {'buy': 0, 'sell': 0}
    swing_killers = {'buy': [], 'sell': []}
    if swing_filter['enabled']:
        if swing_filter['no_buy_at_swing_high'] and swings['is_swing_high']:
            swing_adj['buy'] = -swing_filter['swing_penalty']
            swing_killers['buy'].append(f"Swing High: -{swing_filter['swing_penalty']}")
        if swing_filter['no_sell_at_swing_low'] and swings['is_swing_low']:
            swing_adj['sell'] = -swing_filter['swing_penalty']
            swing_killers['sell'].append(f"Swing Low: -{swing_filter['swing_penalty']}")

    atr = analyze_atr_volatility(df_m1)
    print(f"📏 ATR M1: {atr['reason']}")

    m5_trend = analyze_m5_trend(df_m5)  # tendance M5 via EMA
    print(f"⏰ M5 trend: {m5_trend['reason']}")

    # Patterns sur la dernière bougie M1 (pour confirmation d'entrée)
    last_m1 = df_m1.iloc[-1].to_dict()
    prev_m1 = df_m1.iloc[-2].to_dict() if len(df_m1) > 1 else None
    pin_m1_buy = is_pin_bar(last_m1, 'buy')
    pin_m1_sell = is_pin_bar(last_m1, 'sell')
    engulf_m1 = is_engulfing(prev_m1, last_m1) if prev_m1 is not None else None

    # ========== CALCUL DES SCORES (compatible avec ancien mode) ==========
    buy_score = momentum['buy']['score'] + bb['buy']['score'] + swing_adj['buy'] + atr['score']
    sell_score = momentum['sell']['score'] + bb['sell']['score'] + swing_adj['sell'] + atr['score']

    if m5_trend['trend'] == "BULLISH":
        buy_score += m5_trend['score']
    elif m5_trend['trend'] == "BEARISH":
        sell_score += m5_trend['score']
    else:
        buy_score += m5_trend['score'] // 2
        sell_score += m5_trend['score'] // 2

    if SAINT_GRAAL_CONFIG['market_state']['enabled']:
        if market['state'] == "RANGE" and SAINT_GRAAL_CONFIG['market_state']['prioritize_bb_in_range']:
            if buy_score > 0:
                buy_score = buy_score * 0.75 + bb['buy']['score'] * 0.25
            if sell_score > 0:
                sell_score = sell_score * 0.75 + bb['sell']['score'] * 0.25
        elif market['state'] == "TREND" and SAINT_GRAAL_CONFIG['market_state']['prioritize_momentum_in_trend']:
            if buy_score > 0:
                buy_score = buy_score * 0.8 + momentum['buy']['score'] * 0.2
            if sell_score > 0:
                sell_score = sell_score * 0.8 + momentum['sell']['score'] * 0.2

    print(f"\n🎯 SCORES: BUY {buy_score:.1f} | SELL {sell_score:.1f} | MIN requis: {SAINT_GRAAL_CONFIG['signal_validation']['min_score']}")

    min_score = SAINT_GRAAL_CONFIG['signal_validation']['min_score']
    sniper_mode = SAINT_GRAAL_CONFIG.get('sniper_mode', False)

    # ========== CONDITIONS DE SIGNAL ==========
    # Conditions de base (toujours requises)
    buy_conditions_base = (
        momentum['buy']['allowed'] and
        bb['buy']['allowed'] and
        momentum['gate_buy'] and
        buy_score >= min_score
    )
    sell_conditions_base = (
        momentum['sell']['allowed'] and
        bb['sell']['allowed'] and
        momentum['gate_sell'] and
        sell_score >= min_score
    )

    if sniper_mode:
        # Conditions supplémentaires pour le mode Sniper
        buy_sniper = (
            (near_support or market['state'] == 'RANGE') and  # soit sur support, soit en range
            (divergence == 'bullish' or pin_m5_buy or engulf_m5 == 'bullish') and
            (pin_m1_buy or engulf_m1 == 'bullish')  # confirmation sur M1
        )
        sell_sniper = (
            (near_resistance or market['state'] == 'RANGE') and
            (divergence == 'bearish' or pin_m5_sell or engulf_m5 == 'bearish') and
            (pin_m1_sell or engulf_m1 == 'bearish')
        )
        buy_conditions = buy_conditions_base and buy_sniper
        sell_conditions = sell_conditions_base and sell_sniper
    else:
        buy_conditions = buy_conditions_base
        sell_conditions = sell_conditions_base

    signal = None
    final_score = 0
    quality = "MINIMUM"
    final_confidence = 60
    reason_text = ""
    micro = {'valid': False, 'score': 0, 'reason': ''}
    confidence_killers = []

    if buy_conditions:
        micro = analyze_micro_momentum(df_m1, "BUY")
        final_score = buy_score + (micro['score'] if micro['valid'] else 0)
        if final_score >= min_score:
            confidence_reduction, killers = check_confidence_killers(df_m1, "BUY", momentum)
            confidence_killers.extend(killers)
            signal = "CALL"
            reason_text = (f"BUY Score: {final_score:.1f} | RSI: {momentum['rsi']:.1f} "
                           f"| Stoch: {momentum['stoch_k']:.1f} | BB: {bb['bb_position']:.1f}%")
    elif sell_conditions:
        micro = analyze_micro_momentum(df_m1, "SELL")
        final_score = sell_score + (micro['score'] if micro['valid'] else 0)
        if final_score >= min_score:
            confidence_reduction, killers = check_confidence_killers(df_m1, "SELL", momentum)
            confidence_killers.extend(killers)
            signal = "PUT"
            reason_text = (f"SELL Score: {final_score:.1f} | RSI: {momentum['rsi']:.1f} "
                           f"| Stoch: {momentum['stoch_k']:.1f} | BB: {bb['bb_position']:.1f}%")

    if signal:
        if final_score >= 105:
            quality = "PREMIUM"
        elif final_score >= 95:
            quality = "EXCELLENT"
        elif final_score >= 85:
            quality = "HIGH"
        elif final_score >= 75:
            quality = "GOOD"
        elif final_score >= 65:
            quality = "SOLID"
        else:
            quality = "MINIMUM"
        base_confidence = calculate_confidence(final_score)
        final_confidence = max(60, base_confidence - confidence_reduction)
        print(f"\n✅ SIGNAL {signal} DÉTECTÉ!")
        print(f"   Score: {final_score:.1f} | Qualité: {quality} | Confiance: {final_confidence}%")
        if confidence_killers:
            print(f"   Killers: {', '.join(confidence_killers)}")
        if micro['valid']:
            print(f"   Micro: {micro['reason']}")
        return {
            'direction': signal,
            'quality': quality,
            'score': round(final_score, 1),
            'confidence': final_confidence,
            'expiration_minutes': 5,
            'reason': reason_text,
            'details': {
                'market_state': market['state'],
                'momentum_score': max(momentum['buy']['score'], momentum['sell']['score']),
                'bb_score': max(bb['buy']['score'], bb['sell']['score']),
                'micro_score': micro['score'] if micro['valid'] else 0,
                'atr_score': atr['score'],
                'm5_trend': m5_trend['trend'],
                'rsi': momentum['rsi'],
                'stoch': momentum['stoch_k'],
                'bb_position': bb['bb_position'],
                'atr_pips': atr['atr_pips'],
                'gate_buy': momentum['gate_buy'],
                'gate_sell': momentum['gate_sell'],
                'confidence_killers': confidence_killers,
                'swing_adjustment': swing_adj,
                'near_support': near_support,
                'near_resistance': near_resistance,
                'divergence': divergence,
                'pin_m5': {'buy': pin_m5_buy, 'sell': pin_m5_sell},
                'engulf_m5': engulf_m5,
                'pin_m1': {'buy': pin_m1_buy, 'sell': pin_m1_sell},
                'engulf_m1': engulf_m1,
            }
        }
    else:
        print(f"\n❌ AUCUN SIGNAL VALIDE")
        print(f"   Score BUY:  {buy_score:.1f}  | Gate BUY:  {'✅' if momentum['gate_buy'] else '❌'} | Conditions: {'✅' if buy_conditions else '❌'}")
        print(f"   Score SELL: {sell_score:.1f} | Gate SELL: {'✅' if momentum['gate_sell'] else '❌'} | Conditions: {'✅' if sell_conditions else '❌'}")
        if swing_killers['buy']:
            print(f"   Swing BUY:  {swing_killers['buy']}")
        if swing_killers['sell']:
            print(f"   Swing SELL: {swing_killers['sell']}")
        if momentum.get('gate_debug'):
            print(f"   Gate debug BUY:  {momentum['gate_debug']['buy']}")
            print(f"   Gate debug SELL: {momentum['gate_debug']['sell']}")
        if sniper_mode:
            print(f"   Sniper conditions BUY:  near_support={near_support}, divergence/pattern M5={divergence or pin_m5_buy or engulf_m5}, pattern M1={pin_m1_buy or engulf_m1}")
            print(f"   Sniper conditions SELL: near_resistance={near_resistance}, divergence/pattern M5={divergence or pin_m5_sell or engulf_m5}, pattern M1={pin_m1_sell or engulf_m1}")
        return None

# ================= INTERFACE POUR LE BOT =================

def get_signal_saint_graal(df, signal_count=0, total_signals=8, return_dict=False):
    """
    Interface de compatibilité pour signal_bot.py
    Ici, on suppose que df est en M1. On agrège pour obtenir M5 et M15.
    """
    print(f"\n🎯 ANALYSE MULTI-TIMEFRAME — Signal #{signal_count} | {len(df)} bougies M1")
    signal = analyze_pair_for_signals(df)  # df_m5 et df_m15 seront agrégés automatiquement
    if signal:
        signal['signal_count'] = signal_count
        signal['total_signals'] = total_signals
        if 'mode' not in signal:
            signal['mode'] = "V10.1 MTF"
        print(f"✅ Signal: {signal['direction']} — Score: {signal['score']} — Qualité: {signal['quality']}")
        return signal
    print(f"❌ Pas de signal (score minimum: {SAINT_GRAAL_CONFIG['signal_validation']['min_score']})")
    return None

get_binary_signal = get_signal_saint_graal

if __name__ == "__main__":
    print("🚀 STRATÉGIE BINAIRE M1 PRO — VERSION 10.1 MULTI-TIMEFRAME")
    print("=" * 60)
    print("APPROCHE 'THE SNIPER' : M15 (direction/niveaux) + M5 (divergence/patterns) + M1 (entrée)")
    print("=" * 60)
