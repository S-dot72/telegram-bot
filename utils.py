"""
utils.py - STRATÉGIE BINAIRE M1 PRO - VERSION 5.0 ULTIMATE PLUS COMPLÈTE (CORRIGÉE)
VERSION FINALE - Une seule fonction qui s'adapte au besoin
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIGURATION =================

SAINT_GRAAL_CONFIG = {
    'expiration_minutes': 5,
    
    'micro_momentum_filter': {
        'enabled': True,
        'lookback_bars': 3,
        'min_bullish_bars': 2,
        'min_bearish_bars': 2,
        'require_price_alignment': True,
        'require_volume_confirmation': False,
        'weight': 15,
    },
    
    'atr_filter': {
        'enabled': True,
        'window': 14,
        'min_atr_pips': 2,
        'max_atr_pips': 25,
        'optimal_atr_pips': [5, 15],
        'atr_trend_weight': 10,
        'squeeze_detection': True,
    },
    
    'bb_crossover': {
        'enabled': True,
        'lookback_bars': 2,
        'require_confirmation': True,
        'weight': 12,
        'min_candle_size_pips': 3,
        'strict_mode': True,
    },
    
    'buy_rules': {
        'stoch_period': 7,
        'stoch_smooth': 3,
        'rsi_max_for_buy': 45,
        'rsi_oversold': 32,
        'require_swing_confirmation': True,
        'min_signal_duration_bars': 2,
        'bb_confirmation': True,
        'score_threshold': 70,
    },
    
    'sell_rules': {
        'stoch_period': 9,
        'stoch_smooth': 3,
        'rsi_min_for_sell': 60,
        'stoch_min_overbought': 68,
        'require_swing_break': True,
        'max_swing_distance_pips': 6,
        'momentum_gate_diff': 12,
        'min_signal_duration_bars': 3,
        'bb_confirmation': True,
        'score_threshold': 75,
    },
    
    'm5_filter': {
        'enabled': True,
        'ema_fast': 50,
        'ema_slow': 200,
        'min_required_m5_bars': 50,
        'weight': 20,
        'strict_mode': True,
    },
    
    'bollinger_config': {
        'window': 20,
        'window_dev': 2,
        'oversold_zone': 30,
        'overbought_zone': 70,
        'middle_band_weight': 25,
        'crossover_weight': 15,
    },
    
    'signal_config': {
        'require_m5_alignment': True,
        'min_quality_score': 90,
        'max_signals_per_session': 6,
        'cooldown_bars_after_signal': 3
    }
}

# ================= FONCTION POUR CALCULER LE POURCENTAGE DE CONFIANCE =================

def calculate_confidence_percentage(final_score):
    """Calcule le pourcentage de confiance basé sur le score final"""
    min_score = 90
    max_score = 200
    
    if final_score <= min_score:
        return 50
    elif final_score >= max_score:
        return 100
    else:
        percentage = 50 + ((final_score - min_score) / (max_score - min_score)) * 50
        return min(100, max(50, round(percentage)))

# ================= FONCTION POUR GÉNÉRER LE MESSAGE DE SIGNAL =================

def generate_signal_message(signal_data, pairs_analyzed=5, batches=1):
    """Génère un message formaté pour le signal"""
    if not signal_data or not isinstance(signal_data, dict):
        return None
    
    required_keys = ['direction', 'quality', 'score']
    for key in required_keys:
        if key not in signal_data:
            return None
    
    direction = signal_data['direction']
    quality = signal_data['quality']
    score = signal_data['score']
    confidence = signal_data.get('confidence', calculate_confidence_percentage(score))
    
    if direction == "CALL":
        direction_emoji = "↗️"
        direction_text = "CALL ↗️"
    else:
        direction_emoji = "↘️"
        direction_text = "PUT ↘️"
    
    quality_stars = {
        'EXCELLENT': '⭐⭐⭐⭐⭐',
        'HIGH': '⭐⭐⭐⭐',
        'SOLID': '⭐⭐⭐',
        'MINIMUM': '⭐⭐',
        'CRITICAL': '⭐'
    }.get(quality, '⭐')
    
    current_time = datetime.now().strftime("%H:%M")
    
    message = f"""
🎯 **SIGNAL #1 - ROTATION ITÉRATIVE**
━━━━━━━━━━━━━━━━━━━━
💱 ETH/USD
📈 Direction: **{direction_text}**
⏰ Heure entrée: **{current_time}**
💪 Confiance: **{confidence}%**
{quality_stars} Qualité: **{quality}**

🔄 {pairs_analyzed} paires analysées ({batches} batches)
⏱️ Timeframe: 1 minute
🎯 Expiration: 5 minutes
📊 Score système: **{score:.0f}/200**
━━━━━━━━━━━━━━━━━━━━
📋 Détails techniques:
• Momentum aligné: ✓
• Bollinger Bands: ✓
• Filtre ATR: ✓  
• Micro momentum: ✓
• Croisement BB médiane: ✓
━━━━━━━━━━━━━━━━━━━━
⚠️ **RAPPEL RISQUES**
• Maximum 3-5% du capital par trade
• Stop loss mental obligatoire
• Pas de revenge trading
━━━━━━━━━━━━━━━━━━━━
🚀 **ACTION IMMÉDIATE**
1. Vérifier confluence sur M5
2. Entrée au prix marché
3. Expiration: 5 minutes
4. TP: 75-85% | SL: 0%
━━━━━━━━━━━━━━━━━━━━
"""
    return message

# ================= MICRO GARDE-FOU MOMENTUM =================

def check_micro_momentum(df, direction, lookback=3):
    """Vérifie la cohérence des dernières bougies M1 avec la direction"""
    if len(df) < lookback + 2:
        return False, 0, "Données insuffisantes pour micro momentum"
    
    recent = df.tail(lookback).copy()
    closes = recent['close'].values
    opens = recent['open'].values
    highs = recent['high'].values
    lows = recent['low'].values
    
    bullish_count = 0
    bearish_count = 0
    price_alignment = 0
    
    for i in range(len(recent)):
        if closes[i] > opens[i]:
            bullish_count += 1
        elif closes[i] < opens[i]:
            bearish_count += 1
        
        if i > 0:
            if closes[i] > closes[i-1]:
                price_alignment += 1
            elif closes[i] < closes[i-1]:
                price_alignment -= 1
    
    if direction == "BUY":
        min_bullish = SAINT_GRAAL_CONFIG['micro_momentum_filter']['min_bullish_bars']
        
        if bullish_count < min_bullish:
            return False, -20, f"Seulement {bullish_count}/{lookback} bougies haussières"
        
        if (SAINT_GRAAL_CONFIG['micro_momentum_filter']['require_price_alignment'] and 
            price_alignment < 1):
            return False, -15, f"Alignement prix faible: {price_alignment}"
        
        highs_increasing = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        if highs_increasing >= 2:
            micro_score = SAINT_GRAAL_CONFIG['micro_momentum_filter']['weight']
            return True, micro_score, f"Micro momentum BUY: {bullish_count}/{lookback} haussier, prix alignés"
        else:
            return True, 8, f"Micro momentum BUY faible: {bullish_count}/{lookback} haussier"
    
    elif direction == "SELL":
        min_bearish = SAINT_GRAAL_CONFIG['micro_momentum_filter']['min_bearish_bars']
        
        if bearish_count < min_bearish:
            return False, -20, f"Seulement {bearish_count}/{lookback} bougies baissières"
        
        if (SAINT_GRAAL_CONFIG['micro_momentum_filter']['require_price_alignment'] and 
            price_alignment > -1):
            return False, -15, f"Alignement prix faible: {price_alignment}"
        
        lows_decreasing = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
        if lows_decreasing >= 2:
            micro_score = SAINT_GRAAL_CONFIG['micro_momentum_filter']['weight']
            return True, micro_score, f"Micro momentum SELL: {bearish_count}/{lookback} baissier, prix alignés"
        else:
            return True, 8, f"Micro momentum SELL faible: {bearish_count}/{lookback} baissier"
    
    return False, 0, "Direction non reconnue"

# ================= FILTRE ATR =================

def calculate_atr_filter(df):
    """🔥 FILTRE ATR - Analyse de la volatilité"""
    if len(df) < 20:
        return {
            'enabled': False,
            'atr_value': 0,
            'atr_pips': 0,
            'signal': 'NO_DATA',
            'score': 0,
            'reason': 'Données insuffisantes pour ATR',
            'is_squeeze': False,
            'atr_trend': 'NEUTRAL',
        }
    
    atr_indicator = AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=SAINT_GRAAL_CONFIG['atr_filter']['window']
    )
    
    atr_values = atr_indicator.average_true_range()
    current_atr = float(atr_values.iloc[-1])
    atr_pips = current_atr / 0.0001
    
    if len(atr_values) > 1:
        prev_atr = float(atr_values.iloc[-2])
        atr_trend = "RISING" if current_atr > prev_atr else "FALLING"
    else:
        atr_trend = "NEUTRAL"
    
    avg_atr = atr_values.tail(50).mean() if len(atr_values) >= 50 else current_atr
    is_squeeze = current_atr < avg_atr * 0.6
    
    min_atr = SAINT_GRAAL_CONFIG['atr_filter']['min_atr_pips']
    max_atr = SAINT_GRAAL_CONFIG['atr_filter']['max_atr_pips']
    optimal_range = SAINT_GRAAL_CONFIG['atr_filter']['optimal_atr_pips']
    
    score = 0
    signal = "NEUTRAL"
    reason = ""
    
    if atr_pips < min_atr:
        signal = "AVOID_LOW_VOL"
        score = -25
        reason = f"ATR trop bas: {atr_pips:.1f} pips < {min_atr} pips"
    
    elif atr_pips > max_atr:
        signal = "AVOID_HIGH_VOL"
        score = -20
        reason = f"ATR trop haut: {atr_pips:.1f} pips > {max_atr} pips"
    
    elif optimal_range[0] <= atr_pips <= optimal_range[1]:
        signal = "OPTIMAL_VOL"
        score = 15
        reason = f"ATR optimal: {atr_pips:.1f} pips"
        
        if atr_trend == "RISING":
            score += SAINT_GRAAL_CONFIG['atr_filter']['atr_trend_weight']
            reason += f" (hausse, momentum favorable)"
    
    else:
        signal = "ACCEPTABLE_VOL"
        score = 5
        reason = f"ATR acceptable: {atr_pips:.1f} pips"
    
    if (is_squeeze and SAINT_GRAAL_CONFIG['atr_filter']['squeeze_detection'] and
        signal not in ["AVOID_LOW_VOL", "AVOID_HIGH_VOL"]):
        score += 5
        reason += " [SQUEEZE détecté]"
    
    return {
        'enabled': True,
        'atr_value': current_atr,
        'atr_pips': atr_pips,
        'signal': signal,
        'score': score,
        'reason': reason,
        'is_squeeze': is_squeeze,
        'atr_trend': atr_trend,
    }

# ================= LOGIQUE CROISEMENT BANDE MÉDIANE BB =================

def check_bb_middle_crossover(df, direction):
    """🔥 Vérifie le croisement de la bande médiane des Bollinger Bands"""
    if not SAINT_GRAAL_CONFIG['bb_crossover']['enabled']:
        return True, 0, "Croisement BB désactivé"
    
    if len(df) < SAINT_GRAAL_CONFIG['bollinger_config']['window'] + 5:
        return False, 0, "Données insuffisantes pour BB crossover"
    
    bb = BollingerBands(
        close=df['close'],
        window=SAINT_GRAAL_CONFIG['bollinger_config']['window'],
        window_dev=SAINT_GRAAL_CONFIG['bollinger_config']['window_dev']
    )
    
    bb_middle = bb.bollinger_mavg()
    
    lookback = SAINT_GRAAL_CONFIG['bb_crossover']['lookback_bars']
    recent_data = df.tail(lookback + 1).copy()
    
    if len(recent_data) < lookback + 1:
        return False, 0, f"Pas assez de données: {len(recent_data)} < {lookback + 1}"
    
    recent_middle = bb_middle.iloc[-(lookback + 1):].values
    recent_closes = recent_data['close'].values
    recent_opens = recent_data['open'].values
    
    crossover_detected = False
    crossover_strength = 0
    reason = ""
    
    if direction == "BUY":
        current_close = recent_closes[-1]
        current_middle = recent_middle[-1]
        
        prev_close = recent_closes[-2] if lookback >= 1 else None
        prev_middle = recent_middle[-2] if lookback >= 1 else None
        
        if current_close > current_middle:
            crossover_detected = True
            
            distance_pips = (current_close - current_middle) / 0.0001
            crossover_strength = min(SAINT_GRAAL_CONFIG['bb_crossover']['weight'], 
                                    distance_pips * 2)
            
            if SAINT_GRAAL_CONFIG['bb_crossover']['require_confirmation']:
                if prev_close is not None and prev_middle is not None:
                    if prev_close < prev_middle:
                        crossover_strength += 5
                        reason = f"Vrai croisement BUY: {distance_pips:.1f} pips au-dessus"
                    else:
                        reason = f"BUY: Déjà au-dessus, {distance_pips:.1f} pips"
                else:
                    reason = f"BUY: {distance_pips:.1f} pips au-dessus de la médiane"
            else:
                reason = f"BUY: {distance_pips:.1f} pips au-dessus de la médiane"
            
            current_open = recent_opens[-1]
            candle_size = abs(current_close - current_open) / 0.0001
            if candle_size >= SAINT_GRAAL_CONFIG['bb_crossover']['min_candle_size_pips']:
                crossover_strength += 3
                reason += f" (bougie forte: {candle_size:.1f} pips)"
            else:
                reason += f" (bougie faible: {candle_size:.1f} pips)"
        else:
            reason = f"BUY rejeté: Fermeture {current_close} < Bande médiane {current_middle:.5f}"
    
    elif direction == "SELL":
        current_close = recent_closes[-1]
        current_middle = recent_middle[-1]
        
        prev_close = recent_closes[-2] if lookback >= 1 else None
        prev_middle = recent_middle[-2] if lookback >= 1 else None
        
        if current_close < current_middle:
            crossover_detected = True
            
            distance_pips = (current_middle - current_close) / 0.0001
            crossover_strength = min(SAINT_GRAAL_CONFIG['bb_crossover']['weight'], 
                                    distance_pips * 2)
            
            if SAINT_GRAAL_CONFIG['bb_crossover']['require_confirmation']:
                if prev_close is not None and prev_middle is not None:
                    if prev_close > prev_middle:
                        crossover_strength += 5
                        reason = f"Vrai croisement SELL: {distance_pips:.1f} pips en-dessous"
                    else:
                        reason = f"SELL: Déjà en-dessous, {distance_pips:.1f} pips"
                else:
                    reason = f"SELL: {distance_pips:.1f} pips en-dessous de la médiane"
            else:
                reason = f"SELL: {distance_pips:.1f} pips en-dessous de la médiane"
            
            current_open = recent_opens[-1]
            candle_size = abs(current_close - current_open) / 0.0001
            if candle_size >= SAINT_GRAAL_CONFIG['bb_crossover']['min_candle_size_pips']:
                crossover_strength += 3
                reason += f" (bougie forte: {candle_size:.1f} pips)"
            else:
                reason += f" (bougie faible: {candle_size:.1f} pips)"
        else:
            reason = f"SELL rejeté: Fermeture {current_close} > Bande médiane {current_middle:.5f}"
    
    if (SAINT_GRAAL_CONFIG['bb_crossover']['strict_mode'] and 
        crossover_detected and 
        crossover_strength < 5):
        return False, 0, f"Croisement trop faible: {crossover_strength:.1f}"
    
    return crossover_detected, crossover_strength, reason

# ================= FONCTIONS DE BASE =================

def calculate_m5_filter(df_m1):
    """Filtre M5 pour analyse de tendance"""
    if len(df_m1) < 300:
        return {
            'trend': 'NEUTRAL',
            'score': 0,
            'reason': 'Données M5 insuffisantes',
            'ema_fast': None,
            'ema_slow': None
        }
    
    df_m5 = df_m1.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    if len(df_m5) < SAINT_GRAAL_CONFIG['m5_filter']['min_required_m5_bars']:
        return {
            'trend': 'NEUTRAL',
            'score': 0,
            'reason': 'Bougies M5 insuffisantes après resample',
            'ema_fast': None,
            'ema_slow': None
        }
    
    ema_fast = EMAIndicator(
        close=df_m5['close'],
        window=SAINT_GRAAL_CONFIG['m5_filter']['ema_fast']
    ).ema_indicator()
    
    ema_slow = EMAIndicator(
        close=df_m5['close'],
        window=SAINT_GRAAL_CONFIG['m5_filter']['ema_slow']
    ).ema_indicator()
    
    current_ema_fast = float(ema_fast.iloc[-1])
    current_ema_slow = float(ema_slow.iloc[-1])
    
    price = float(df_m5.iloc[-1]['close'])
    
    if current_ema_fast > current_ema_slow * 1.002:
        trend = "BULLISH"
        score = SAINT_GRAAL_CONFIG['m5_filter']['weight']
        reason = f"M5 BULLISH (EMA{SAINT_GRAAL_CONFIG['m5_filter']['ema_fast']}>{SAINT_GRAAL_CONFIG['m5_filter']['ema_slow']})"
        
        if price > current_ema_fast:
            score += 5
            reason += " - Prix > EMA rapide"
    
    elif current_ema_slow > current_ema_fast * 1.002:
        trend = "BEARISH"
        score = SAINT_GRAAL_CONFIG['m5_filter']['weight']
        reason = f"M5 BEARISH (EMA{SAINT_GRAAL_CONFIG['m5_filter']['ema_slow']}>{SAINT_GRAAL_CONFIG['m5_filter']['ema_fast']})"
        
        if price < current_ema_fast:
            score += 5
            reason += " - Prix < EMA rapide"
    
    else:
        trend = "NEUTRAL"
        score = 0
        reason = "M5 NEUTRAL (EMAs alignées)"
    
    return {
        'trend': trend,
        'score': score,
        'reason': reason,
        'ema_fast': current_ema_fast,
        'ema_slow': current_ema_slow
    }

def analyze_market_structure(df, lookback=20):
    """Analyse la structure du marché"""
    if len(df) < lookback:
        return "NEUTRAL", 0
    
    recent_data = df.tail(lookback).copy()
    
    highs = recent_data['high'].values
    lows = recent_data['low'].values
    
    swing_highs = []
    swing_lows = []
    
    for i in range(2, len(highs)-2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append(lows[i])
    
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        last_2_highs = sorted(swing_highs)[-2:]
        last_2_lows = sorted(swing_lows)[-2:]
        
        if last_2_highs[-1] > last_2_highs[-2] and last_2_lows[-1] > last_2_lows[-2]:
            trend_strength = ((last_2_highs[-1] - last_2_highs[-2]) / last_2_highs[-2] * 100 + 
                            (last_2_lows[-1] - last_2_lows[-2]) / last_2_lows[-2] * 100) / 2
            return "UPTREND", trend_strength
        
        elif last_2_highs[-1] < last_2_highs[-2] and last_2_lows[-1] < last_2_lows[-2]:
            trend_strength = ((last_2_highs[-2] - last_2_highs[-1]) / last_2_highs[-2] * 100 + 
                            (last_2_lows[-2] - last_2_lows[-1]) / last_2_lows[-2] * 100) / 2
            return "DOWNTREND", trend_strength
    
    avg_range = (recent_data['high'].max() - recent_data['low'].min()) / recent_data['close'].mean() * 100
    
    if avg_range < 0.1:
        return "CONSOLIDATION", avg_range
    else:
        return "NEUTRAL", avg_range

def analyze_momentum_asymmetric_optimized(df):
    """Analyse de momentum avec paramètres asymétriques"""
    if len(df) < 50:
        return {
            'rsi': 50,
            'stoch_k_fast': 50,
            'stoch_d_fast': 50,
            'stoch_k_slow': 50,
            'stoch_d_slow': 50,
            'buy_score': 0,
            'sell_score': 0,
            'dominant': 'NEUTRAL',
            'momentum_gate_passed': False
        }
    
    rsi = RSIIndicator(close=df['close'], window=14).rsi()
    current_rsi = float(rsi.iloc[-1])
    
    stoch_fast = StochasticOscillator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=SAINT_GRAAL_CONFIG['buy_rules']['stoch_period'],
        smooth_window=SAINT_GRAAL_CONFIG['buy_rules']['stoch_smooth']
    )
    stoch_k_fast = stoch_fast.stoch()
    stoch_d_fast = stoch_fast.stoch_signal()
    
    current_stoch_k_fast = float(stoch_k_fast.iloc[-1])
    current_stoch_d_fast = float(stoch_d_fast.iloc[-1])
    
    stoch_slow = StochasticOscillator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=SAINT_GRAAL_CONFIG['sell_rules']['stoch_period'],
        smooth_window=SAINT_GRAAL_CONFIG['sell_rules']['stoch_smooth']
    )
    stoch_k_slow = stoch_slow.stoch()
    stoch_d_slow = stoch_slow.stoch_signal()
    
    current_stoch_k_slow = float(stoch_k_slow.iloc[-1])
    current_stoch_d_slow = float(stoch_d_slow.iloc[-1])
    
    buy_score = 0
    sell_score = 0
    
    if current_rsi < SAINT_GRAAL_CONFIG['buy_rules']['rsi_max_for_buy']:
        buy_score += 25
        
        if current_rsi < SAINT_GRAAL_CONFIG['buy_rules']['rsi_oversold']:
            buy_score += 15
            buy_score += (SAINT_GRAAL_CONFIG['buy_rules']['rsi_oversold'] - current_rsi) * 2
    
    if current_stoch_k_fast < 20 and current_stoch_d_fast < 20:
        buy_score += 20
    elif current_stoch_k_fast < 30 and current_stoch_d_fast < 30:
        buy_score += 15
    
    if current_rsi > SAINT_GRAAL_CONFIG['sell_rules']['rsi_min_for_sell']:
        sell_score += 25
        
        if current_rsi > 70:
            sell_score += 15
            sell_score += (current_rsi - 70) * 2
    
    if current_stoch_k_slow > SAINT_GRAAL_CONFIG['sell_rules']['stoch_min_overbought']:
        sell_score += 20
    elif current_stoch_k_slow > 75:
        sell_score += 25
    
    momentum_gate_diff_buy = abs(current_stoch_k_fast - current_stoch_d_fast)
    momentum_gate_diff_sell = abs(current_stoch_k_slow - current_stoch_d_slow)
    
    momentum_gate_passed = (
        (buy_score > 0 and momentum_gate_diff_buy >= SAINT_GRAAL_CONFIG['sell_rules']['momentum_gate_diff']) or
        (sell_score > 0 and momentum_gate_diff_sell >= SAINT_GRAAL_CONFIG['sell_rules']['momentum_gate_diff'])
    )
    
    dominant = "NEUTRAL"
    if buy_score > sell_score + 10:
        dominant = "BUY"
    elif sell_score > buy_score + 15:
        dominant = "SELL"
    
    return {
        'rsi': current_rsi,
        'stoch_k_fast': current_stoch_k_fast,
        'stoch_d_fast': current_stoch_d_fast,
        'stoch_k_slow': current_stoch_k_slow,
        'stoch_d_slow': current_stoch_d_slow,
        'buy_score': buy_score,
        'sell_score': sell_score,
        'dominant': dominant,
        'momentum_gate_passed': momentum_gate_passed
    }

def calculate_bollinger_signals(df):
    """Calcule les signaux des Bandes de Bollinger avec croisement médiane"""
    if len(df) < SAINT_GRAAL_CONFIG['bollinger_config']['window'] + 10:
        return {
            'bb_position': 50,
            'bb_signal': 'NO_DATA',
            'bb_width': 0,
            'bb_squeeze': False,
            'bb_upper': 0,
            'bb_lower': 0,
            'bb_middle': 0,
            'price_above_middle': False,
            'price_below_middle': False,
            'middle_crossover': 'NEUTRAL'
        }
    
    bb = BollingerBands(
        close=df['close'],
        window=SAINT_GRAAL_CONFIG['bollinger_config']['window'],
        window_dev=SAINT_GRAAL_CONFIG['bollinger_config']['window_dev']
    )
    
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_middle = bb.bollinger_mavg()
    bb_width = bb.bollinger_wband()
    
    current_price = float(df.iloc[-1]['close'])
    current_upper = float(bb_upper.iloc[-1])
    current_lower = float(bb_lower.iloc[-1])
    current_middle = float(bb_middle.iloc[-1])
    
    if current_upper != current_lower:
        bb_position = ((current_price - current_lower) / (current_upper - current_lower)) * 100
    else:
        bb_position = 50
    
    avg_width = bb_width.tail(20).mean()
    current_width = float(bb_width.iloc[-1])
    bb_squeeze = current_width < avg_width * 0.7
    
    price_above_middle = current_price > current_middle
    price_below_middle = current_price < current_middle
    
    middle_crossover = "NEUTRAL"
    if len(df) >= 2:
        prev_price = float(df.iloc[-2]['close'])
        prev_middle = float(bb_middle.iloc[-2]) if len(bb_middle) >= 2 else current_middle
        
        if prev_price <= prev_middle and current_price > current_middle:
            middle_crossover = "BULLISH_CROSS"
        elif prev_price >= prev_middle and current_price < current_middle:
            middle_crossover = "BEARISH_CROSS"
        elif price_above_middle:
            middle_crossover = "ABOVE_MIDDLE"
        elif price_below_middle:
            middle_crossover = "BELOW_MIDDLE"
    
    bb_signal = "NEUTRAL"
    
    if bb_position < SAINT_GRAAL_CONFIG['bollinger_config']['oversold_zone']:
        bb_signal = "OVERSOLD"
    elif bb_position > SAINT_GRAAL_CONFIG['bollinger_config']['overbought_zone']:
        bb_signal = "OVERBOUGHT"
    elif abs(current_price - current_middle) / current_middle * 100 < 0.1:
        bb_signal = "MIDDLE_BAND"
    
    return {
        'bb_position': bb_position,
        'bb_signal': bb_signal,
        'bb_width': current_width,
        'bb_squeeze': bb_squeeze,
        'bb_upper': current_upper,
        'bb_lower': current_lower,
        'bb_middle': current_middle,
        'price_above_middle': price_above_middle,
        'price_below_middle': price_below_middle,
        'middle_crossover': middle_crossover
    }

def get_bb_confirmation_score(bb_signal, direction, stochastic_value):
    """Calcule le score de confirmation Bollinger Bands avec croisement médiane"""
    score = 0
    reason = ""
    
    if direction == "BUY":
        if bb_signal['bb_position'] < 30:
            score += 35
            reason += "BB OVERSOLD"
        elif bb_signal['bb_position'] < 40:
            score += 25
            reason += "BB Près du bas"
        elif bb_signal['bb_position'] < 50:
            score += 15
            reason += "BB Zone neutre basse"
        
        if bb_signal['bb_squeeze']:
            score += 10
            reason += " + SQUEEZE"
        
        if stochastic_value < 30:
            score += 20
            reason += " + Stoch OVERSOLD"
        elif stochastic_value < 40:
            score += 10
            reason += " + Stoch Bas"
        
        if bb_signal['price_above_middle']:
            score += SAINT_GRAAL_CONFIG['bollinger_config']['crossover_weight']
            reason += " + Au-dessus médiane"
            
            if bb_signal['middle_crossover'] == "BULLISH_CROSS":
                score += 8
                reason += " (Croisement haussier)"
        
        if (SAINT_GRAAL_CONFIG['bb_crossover']['strict_mode'] and 
            bb_signal['price_below_middle'] and 
            bb_signal['middle_crossover'] != "BULLISH_CROSS"):
            score -= 15
            reason += " - En-dessous médiane (veto)"
    
    elif direction == "SELL":
        if bb_signal['bb_position'] > 70:
            score += 35
            reason += "BB OVERBOUGHT"
        elif bb_signal['bb_position'] > 60:
            score += 25
            reason += "BB Près du haut"
        elif bb_signal['bb_position'] > 50:
            score += 15
            reason += "BB Zone neutre haute"
        
        if bb_signal['bb_squeeze']:
            score += 10
            reason += " + SQUEEZE"
        
        if stochastic_value > 70:
            score += 20
            reason += " + Stoch OVERBOUGHT"
        elif stochastic_value > 60:
            score += 10
            reason += " + Stoch Haut"
        
        if bb_signal['price_below_middle']:
            score += SAINT_GRAAL_CONFIG['bollinger_config']['crossover_weight']
            reason += " + En-dessous médiane"
            
            if bb_signal['middle_crossover'] == "BEARISH_CROSS":
                score += 8
                reason += " (Croisement baissier)"
        
        if (SAINT_GRAAL_CONFIG['bb_crossover']['strict_mode'] and 
            bb_signal['price_above_middle'] and 
            bb_signal['middle_crossover'] != "BEARISH_CROSS"):
            score -= 15
            reason += " - Au-dessus médiane (veto)"
    
    current_diff_to_band = 0
    if direction == "BUY":
        current_diff_to_band = abs(bb_signal['bb_lower'] - bb_signal['bb_middle'])
    else:
        current_diff_to_band = abs(bb_signal['bb_upper'] - bb_signal['bb_middle'])
    
    if current_diff_to_band > 0:
        band_proximity = min(100, (current_diff_to_band / bb_signal['bb_middle'] * 10000))
        if band_proximity < 15:
            score += 15
            reason += " + Proche bande"
    
    return min(score, 70), reason

def check_m5_alignment(m5_filter, direction):
    """Vérifie l'alignement avec la tendance M5"""
    if m5_filter['trend'] == 'NEUTRAL':
        return True, "M5 Neutre (pas de conflit)", 5
    
    if direction == "BUY":
        if m5_filter['trend'] == "BULLISH":
            return True, "M5 aligné BULLISH", 15
        elif m5_filter['trend'] == "BEARISH":
            return False, "M5 en conflit BEARISH", -20
        else:
            return True, "M5 Neutre", 5
    
    elif direction == "SELL":
        if m5_filter['trend'] == "BEARISH":
            return True, "M5 aligné BEARISH", 15
        elif m5_filter['trend'] == "BULLISH":
            return False, "M5 en conflit BULLISH", -20
        else:
            return True, "M5 Neutre", 5
    
    return False, "Direction inconnue", 0

def validate_candle_for_5min_buy(df):
    """Valide la configuration de bougie pour un signal BUY"""
    if len(df) < 3:
        return False, "NO_DATA", 0, "Données insuffisantes"
    
    current = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    
    current_close = float(current['close'])
    current_open = float(current['open'])
    prev_close = float(prev['close'])
    prev_open = float(prev['open'])
    
    is_bullish = current_close > current_open
    
    candle_size = abs(current_close - current_open) / 0.0001
    
    volume_ok = True
    
    pattern = "NORMAL"
    confidence = 40
    
    if is_bullish and candle_size > 5:
        confidence += 20
        pattern = "BULLISH_STRONG"
    
    lower_shadow = min(current_open, current_close) - float(current['low'])
    upper_shadow = float(current['high']) - max(current_open, current_close)
    
    if lower_shadow > candle_size * 2 and candle_size < lower_shadow * 0.3:
        confidence += 25
        pattern = "HAMMER"
    
    if (is_bullish and not (prev_close > prev_open) and 
        current_close > prev_open and current_open < prev_close):
        confidence += 30
        pattern = "BULLISH_ENGULFING"
    
    if (prev2['close'] < prev2['open'] and
        abs(prev_close - prev_open) < candle_size * 0.3 and
        is_bullish and current_close > prev2['close']):
        confidence += 35
        pattern = "MORNING_STAR"
    
    if not volume_ok:
        confidence -= 10
    
    if current_close > df['close'].tail(20).max():
        confidence -= 15
    
    valid = confidence >= 50
    
    reason = f"{pattern} (Conf: {confidence}%)" if valid else f"Configuration faible: {pattern}"
    
    return valid, pattern, confidence, reason

def validate_candle_for_5min_sell(df):
    """Valide la configuration de bougie pour un signal SELL"""
    if len(df) < 3:
        return False, "NO_DATA", 0, "Données insuffisantes"
    
    current = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    
    current_close = float(current['close'])
    current_open = float(current['open'])
    prev_close = float(prev['close'])
    prev_open = float(prev['open'])
    
    is_bearish = current_close < current_open
    
    candle_size = abs(current_close - current_open) / 0.0001
    
    volume_ok = True
    
    pattern = "NORMAL"
    confidence = 40
    
    if is_bearish and candle_size > 5:
        confidence += 20
        pattern = "BEARISH_STRONG"
    
    upper_shadow = float(current['high']) - max(current_open, current_close)
    lower_shadow = min(current_open, current_close) - float(current['low'])
    
    if upper_shadow > candle_size * 2 and candle_size < upper_shadow * 0.3:
        confidence += 25
        pattern = "SHOOTING_STAR"
    
    if (is_bearish and not (prev_close < prev_open) and 
        current_close < prev_open and current_open > prev_close):
        confidence += 30
        pattern = "BEARISH_ENGULFING"
    
    if (prev2['close'] > prev2['open'] and
        abs(prev_close - prev_open) < candle_size * 0.3 and
        is_bearish and current_close < prev2['close']):
        confidence += 35
        pattern = "EVENING_STAR"
    
    if not volume_ok:
        confidence -= 10
    
    if current_close < df['close'].tail(20).min():
        confidence -= 15
    
    valid = confidence >= 50
    
    reason = f"{pattern} (Conf: {confidence}%)" if valid else f"Configuration faible: {pattern}"
    
    return valid, pattern, confidence, reason

# ================= FONCTION PRINCIPALE V4 =================

def rule_signal_saint_graal_5min_pro_v4(df, signal_count=0, total_signals_needed=6):
    """🔥 VERSION 4.9 : AVEC CROISEMENT BANDE MÉDIANE BB + MICRO MOMENTUM + FILTRE ATR"""
    print(f"\n{'='*70}")
    print(f"🎯 BINAIRE 5 MIN V4.9 - Signal #{signal_count+1}/{total_signals_needed}")
    print(f"{'='*70}")
    
    if len(df) < 100:
        print(f"❌ Données insuffisantes: {len(df)} < 100")
        return None
    
    current_price = float(df.iloc[-1]['close'])
    print(f"💰 Prix actuel: {current_price:.5f}")
    
    m5_filter = calculate_m5_filter(df)
    print(f"📈 Filtre M5: {m5_filter['reason']}")
    
    structure, trend_strength = analyze_market_structure(df)
    print(f"🏗️  Structure: {structure} | Force: {trend_strength:.1f}%")
    
    momentum = analyze_momentum_asymmetric_optimized(df)
    print(f"⚡ Momentum: RSI {momentum['rsi']:.1f} | StochF {momentum['stoch_k_fast']:.1f} | StochS {momentum['stoch_k_slow']:.1f}")
    
    bb_signal = calculate_bollinger_signals(df)
    print(f"📊 BB: Position {bb_signal['bb_position']:.1f}% | Signal: {bb_signal['bb_signal']} | Croisement: {bb_signal['middle_crossover']}")
    
    atr_filter = calculate_atr_filter(df)
    print(f"📏 ATR: {atr_filter['reason']}")
    
    bb_buy_score, bb_buy_reason = get_bb_confirmation_score(
        bb_signal, "BUY", momentum['stoch_k_fast']
    )
    
    bb_sell_score, bb_sell_reason = get_bb_confirmation_score(
        bb_signal, "SELL", momentum['stoch_k_slow']
    )
    
    print(f"✅ BB Confirmation: BUY {bb_buy_score}/70 | SELL {bb_sell_score}/70")
    
    sell_score_total = 0
    buy_score_total = 0
    
    sell_score_total += momentum['sell_score']
    buy_score_total += momentum['buy_score']
    
    sell_score_total += bb_sell_score
    buy_score_total += bb_buy_score
    
    if atr_filter['enabled']:
        buy_score_total += atr_filter['score']
        sell_score_total += atr_filter['score']
        print(f"📏 Score ATR ajouté: {atr_filter['score']} points")
    
    if SAINT_GRAAL_CONFIG['m5_filter']['strict_mode']:
        if m5_filter['trend'] == "BULLISH":
            buy_score_total += m5_filter['score']
        elif m5_filter['trend'] == "BEARISH":
            buy_score_total -= 10
        
        if m5_filter['trend'] == "BEARISH":
            sell_score_total += m5_filter['score']
        elif m5_filter['trend'] == "BULLISH":
            sell_score_total -= 10
    
    if structure == "DOWNTREND" and momentum['dominant'] == "SELL":
        sell_score_total += 15
    
    if structure == "UPTREND" and momentum['dominant'] == "BUY":
        buy_score_total += 15
    
    print(f"🎯 Scores avant micro: SELL {sell_score_total:.0f}/200 - BUY {buy_score_total:.0f}/200")
    
    direction = None
    final_score = 0
    decision_details = []
    micro_valid = False
    micro_score = 0
    micro_reason = ""
    bb_crossover_valid = False
    bb_crossover_score = 0
    bb_crossover_reason = ""
    
    if (buy_score_total >= 70 and momentum['momentum_gate_passed']):
        
        micro_valid, micro_score, micro_reason = check_micro_momentum(df, "BUY")
        bb_crossover_valid, bb_crossover_score, bb_crossover_reason = check_bb_middle_crossover(df, "BUY")
        
        if not micro_valid:
            print(f"❌ Micro momentum BUY échoué: {micro_reason}")
        elif not bb_crossover_valid:
            print(f"❌ Croisement BB BUY échoué: {bb_crossover_reason}")
        else:
            m5_aligned, m5_reason, m5_bonus = check_m5_alignment(m5_filter, "BUY")
            
            if m5_aligned or not SAINT_GRAAL_CONFIG['signal_config']['require_m5_alignment']:
                candle_valid, pattern, pattern_conf, candle_reason = validate_candle_for_5min_buy(df)
                
                if candle_valid:
                    direction = "BUY"
                    final_score = (buy_score_total + pattern_conf + m5_bonus + 
                                 micro_score + bb_crossover_score)
                    decision_details.append(f"BUY validé: {pattern} ({pattern_conf}%)")
                    decision_details.append(f"Micro: {micro_reason}")
                    decision_details.append(f"BB: {bb_crossover_reason}")
                    decision_details.append(m5_reason)
                else:
                    print(f"❌ BUY rejeté: {candle_reason}")
            else:
                print(f"❌ BUY rejeté: {m5_reason}")
    
    elif (sell_score_total >= 75 and momentum['momentum_gate_passed']):
        
        micro_valid, micro_score, micro_reason = check_micro_momentum(df, "SELL")
        bb_crossover_valid, bb_crossover_score, bb_crossover_reason = check_bb_middle_crossover(df, "SELL")
        
        if not micro_valid:
            print(f"❌ Micro momentum SELL échoué: {micro_reason}")
        elif not bb_crossover_valid:
            print(f"❌ Croisement BB SELL échoué: {bb_crossover_reason}")
        else:
            m5_aligned, m5_reason, m5_bonus = check_m5_alignment(m5_filter, "SELL")
            
            if m5_aligned or not SAINT_GRAAL_CONFIG['signal_config']['require_m5_alignment']:
                candle_valid, pattern, pattern_conf, candle_reason = validate_candle_for_5min_sell(df)
                
                if candle_valid:
                    direction = "SELL"
                    final_score = (sell_score_total + pattern_conf + m5_bonus + 
                                 micro_score + bb_crossover_score)
                    decision_details.append(f"SELL validé: {pattern} ({pattern_conf}%)")
                    decision_details.append(f"Micro: {micro_reason}")
                    decision_details.append(f"BB: {bb_crossover_reason}")
                    decision_details.append(m5_reason)
                else:
                    print(f"❌ SELL rejeté: {candle_reason}")
            else:
                print(f"❌ SELL rejeté: {m5_reason}")
    
    if direction:
        if atr_filter['enabled'] and atr_filter['signal'] in ["AVOID_LOW_VOL", "AVOID_HIGH_VOL"]:
            print(f"❌ Signal rejeté par ATR: {atr_filter['reason']}")
            return None
        
        if final_score >= 90:
            if final_score >= 140:
                quality = "EXCELLENT"
                mode = "5MIN_MAX"
            elif final_score >= 120:
                quality = "HIGH"
                mode = "5MIN_PRO"
            elif final_score >= 100:
                quality = "SOLID"
                mode = "5MIN_STANDARD"
            else:
                quality = "MINIMUM"
                mode = "5MIN_MIN"
            
            direction_display = "CALL" if direction == "BUY" else "PUT"
            
            print(f"✅ SIGNAL {direction_display} {quality}")
            print(f"   Score total: {final_score:.1f}")
            print(f"   Détails: {' | '.join(decision_details[:3])}")
            
            return {
                'signal': direction_display,
                'mode': mode,
                'quality': quality,
                'score': float(final_score),
                'reason': f"{direction_display} | Score {final_score:.1f} | {structure} | ATR:{atr_filter['atr_pips']:.1f}pips | BB:{bb_signal['middle_crossover']}",
                'expiration_minutes': 5,
                'details': {
                    'momentum_score': momentum['buy_score'] if direction == "BUY" else momentum['sell_score'],
                    'bb_score': bb_buy_score if direction == "BUY" else bb_sell_score,
                    'micro_momentum_score': micro_score,
                    'bb_crossover_score': bb_crossover_score,
                    'atr_score': atr_filter['score'],
                    'm5_alignment': m5_filter['trend'],
                    'structure': structure,
                    'bb_crossover': bb_signal['middle_crossover'],
                }
            }
    
    print(f"❌ Aucun signal valide - Score insuffisant ou filtres échoués")
    return None

# ================= FONCTION UNIQUE ET UNIVERSALE =================

def get_signal_saint_graal(df, signal_count=0, total_signals=6, return_dict=True, pairs_analyzed=5, batches=1):
    """
    🔥 FONCTION UNIVERSELLE - VERSION 5.0
    Retourne soit un dict (pour analyse multi-marchés) soit un tuple (pour envoi de signal)
    
    Paramètres:
    - df: DataFrame OHLC
    - signal_count: numéro du signal dans la session
    - total_signals: nombre total de signaux dans la session
    - return_dict: True pour retourner un dict, False pour retourner un tuple (dict, message)
    - pairs_analyzed: nombre de paires analysées (pour le message)
    - batches: nombre de batches analysés (pour le message)
    
    Retour:
    - Si return_dict=True: dict ou None
    - Si return_dict=False: tuple (dict, message) ou (None, None)
    """
    try:
        if df is None or len(df) < 100:
            print("❌ Données insuffisantes pour analyse")
            return None if return_dict else (None, None)
        
        # Obtenir le résultat de base
        result = rule_signal_saint_graal_5min_pro_v4(df, signal_count, total_signals)
        
        if result is not None:
            direction_display = result['signal']
            final_score = result['score']
            
            if final_score >= 140:
                quality = "EXCELLENT"
                mode = "5MIN_MAX"
            elif final_score >= 120:
                quality = "HIGH"
                mode = "5MIN_PRO"
            elif final_score >= 100:
                quality = "SOLID"
                mode = "5MIN_STANDARD"
            else:
                quality = "MINIMUM"
                mode = "5MIN_MIN"
            
            # Créer le dictionnaire de signal
            signal_dict = {
                'direction': direction_display,
                'mode': mode,
                'quality': quality,
                'score': float(final_score),
                'confidence': calculate_confidence_percentage(final_score),
                'expiration_minutes': 5,
                'reason': result.get('reason', 'N/A'),
                'details': result.get('details', {}),
                'raw_result': result
            }
            
            if return_dict:
                return signal_dict
            else:
                # Générer le message
                message = generate_signal_message(signal_dict, pairs_analyzed, batches)
                return signal_dict, message
        
        print(f"🎯 Aucun signal valide - Session {signal_count+1}/{total_signals}")
        return None if return_dict else (None, None)
        
    except Exception as e:
        print(f"❌ Erreur dans get_signal_saint_graal: {str(e)}")
        import traceback
        traceback.print_exc()
        return None if return_dict else (None, None)

# ================= FONCTIONS DE COMPATIBILITÉ =================

def get_signal_dict_only(df, signal_count=0, total_signals=6):
    """Alias pour get_signal_saint_graal avec return_dict=True"""
    return get_signal_saint_graal(df, signal_count, total_signals, return_dict=True)

def get_signal_with_metadata(df, signal_count=0, total_signals=6):
    """Alias pour get_signal_saint_graal avec return_dict=False"""
    return get_signal_saint_graal(df, signal_count, total_signals, return_dict=False)

def get_signal_with_metadata_v2(df, signal_count=0, total_signals=6, pairs_analyzed=5, batches=1):
    """Alias pour get_signal_saint_graal avec return_dict=False et paramètres personnalisés"""
    return get_signal_saint_graal(df, signal_count, total_signals, return_dict=False, 
                                  pairs_analyzed=pairs_analyzed, batches=batches)

# ================= POINT D'ENTRÉE PRINCIPAL =================

if __name__ == "__main__":
    print("🎯 DESK PRO BINAIRE - VERSION 5.0 ULTIMATE PLUS COMPLÈTE")
    print("🔥 FONCTION UNIVERSELLE: get_signal_saint_graal()")
    print("\n📋 MODES D'UTILISATION:")
    print("""
    # 1. Pour analyse multi-marchés (retourne un dict)
    signal_dict = get_signal_saint_graal(df, return_dict=True)
    # ou
    signal_dict = get_signal_dict_only(df)
    
    # 2. Pour envoi de signal (retourne un tuple)
    signal_dict, message = get_signal_saint_graal(df, return_dict=False)
    # ou
    signal_dict, message = get_signal_with_metadata(df)
    
    # 3. Avec paramètres personnalisés
    signal_dict, message = get_signal_with_metadata_v2(df, pairs_analyzed=5, batches=2)
    """)
    
    # Test avec des données simulées
    print("\n🧪 TEST SIMULATION:")
    try:
        import pandas as pd
        import numpy as np
        
        # Créer des données de test
        dates = pd.date_range(start='2024-01-01', periods=250, freq='1min')
        data = {
            'open': np.random.normal(1.1000, 0.0005, 250),
            'high': np.random.normal(1.1010, 0.0005, 250),
            'low': np.random.normal(1.0990, 0.0005, 250),
            'close': np.random.normal(1.1005, 0.0005, 250)
        }
        
        df_test = pd.DataFrame(data, index=dates)
        
        # Test 1: Mode dict seulement
        print("\n🔍 Test 1: Mode dict (pour analyse multi-marchés)")
        signal_dict = get_signal_dict_only(df_test, signal_count=0, total_signals=6)
        print(f"Résultat: {type(signal_dict)}")
        if signal_dict:
            print(f"Signal: {signal_dict['direction']} | Score: {signal_dict['score']}")
        
        # Test 2: Mode tuple avec message
        print("\n🔍 Test 2: Mode tuple avec message")
        signal_tuple = get_signal_with_metadata(df_test, signal_count=0, total_signals=6)
        print(f"Résultat: {type(signal_tuple)}")
        if signal_tuple and signal_tuple[0]:
            print(f"Signal détecté: {signal_tuple[0]['direction']}")
        
    except Exception as e:
        print(f"❌ Erreur pendant le test: {str(e)}")
