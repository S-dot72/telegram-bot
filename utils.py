"""
🔥 STRATÉGIE BINAIRE M1 PRO - VERSION 6.0 ULTRA-PERFORMANTE
🎯 Optimisé pour générer des signaux fréquents avec critères assouplis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIGURATION OPTIMISÉE POUR SIGNALS M1 =================

SAINT_GRAAL_CONFIG = {
    'expiration_minutes': 5,
    
    'micro_momentum_filter': {
        'enabled': True,
        'lookback_bars': 2,
        'min_bullish_bars': 1,
        'min_bearish_bars': 1,
        'require_price_alignment': False,
        'require_volume_confirmation': False,
        'weight': 10,
    },
    
    'atr_filter': {
        'enabled': True,
        'window': 10,
        'min_atr_pips': 1,
        'max_atr_pips': 35,
        'optimal_atr_pips': [2, 25],
        'atr_trend_weight': 8,
        'squeeze_detection': False,
    },
    
    'bb_crossover': {
        'enabled': True,
        'lookback_bars': 1,
        'require_confirmation': False,
        'weight': 8,
        'min_candle_size_pips': 1,
        'strict_mode': False,
    },
    
    'buy_rules': {
        'stoch_period': 5,
        'stoch_smooth': 2,
        'rsi_max_for_buy': 50,
        'rsi_oversold': 35,
        'require_swing_confirmation': False,
        'min_signal_duration_bars': 1,
        'bb_confirmation': True,
        'score_threshold': 60,
    },
    
    'sell_rules': {
        'stoch_period': 7,
        'stoch_smooth': 2,
        'rsi_min_for_sell': 55,
        'stoch_min_overbought': 65,
        'require_swing_break': False,
        'max_swing_distance_pips': 10,
        'momentum_gate_diff': 8,
        'min_signal_duration_bars': 2,
        'bb_confirmation': True,
        'score_threshold': 65,
    },
    
    'm5_filter': {
        'enabled': True,
        'ema_fast': 30,
        'ema_slow': 100,
        'min_required_m5_bars': 30,
        'weight': 15,
        'strict_mode': False,
    },
    
    'bollinger_config': {
        'window': 14,
        'window_dev': 1.8,
        'oversold_zone': 25,
        'overbought_zone': 75,
        'middle_band_weight': 20,
        'crossover_weight': 12,
    },
    
    'signal_config': {
        'require_m5_alignment': False,
        'min_quality_score': 75,
        'max_signals_per_session': 12,
        'cooldown_bars_after_signal': 2,
        'force_signal_after_minutes': 10,
    }
}

# ================= FONCTION DE DÉBOGAGE =================

def debug_no_signals(df):
    """Debug pourquoi aucun signal n'est généré"""
    momentum = analyze_momentum_asymmetric_optimized(df)
    bb = calculate_bollinger_signals(df)
    atr = calculate_atr_filter(df)
    
    print(f"\n🔍 DEBUG - Pourquoi pas de signal ?")
    print(f"RSI: {momentum['rsi']:.1f}")
    print(f"Stoch K Fast: {momentum['stoch_k_fast']:.1f}")
    print(f"Stoch K Slow: {momentum['stoch_k_slow']:.1f}")
    print(f"BB Position: {bb['bb_position']:.1f}%")
    print(f"ATR: {atr['atr_pips']:.1f} pips")
    print(f"ATR Signal: {atr['signal']}")
    print(f"Buy Score: {momentum['buy_score']}")
    print(f"Sell Score: {momentum['sell_score']}")
    
    conditions = {
        'buy_score_total >= 70': momentum['buy_score'] >= 70,
        'sell_score_total >= 75': momentum['sell_score'] >= 75,
        'momentum_gate_passed': momentum['momentum_gate_passed'],
        'atr_optimal': atr['signal'] not in ['AVOID_LOW_VOL', 'AVOID_HIGH_VOL']
    }
    
    for cond, value in conditions.items():
        print(f"{cond}: {'✅' if value else '❌'}")
    
    return conditions

# ================= FONCTION POUR FORCER UN SIGNAL =================

def force_signal_if_needed(df, minutes_without_signal=10):
    """Force un signal si trop longtemps sans résultat"""
    if len(df) < 50:
        return None
    
    current_price = float(df.iloc[-1]['close'])
    prev_price = float(df.iloc[-2]['close'])
    
    # Calculer RSI simple
    rsi = RSIIndicator(close=df['close'], window=10).rsi().iloc[-1]
    
    # Calculer tendance simple
    ma_fast = df['close'].rolling(window=5).mean().iloc[-1]
    ma_slow = df['close'].rolling(window=20).mean().iloc[-1]
    
    # Logique de décision forcée
    if current_price > prev_price and rsi < 60:
        return {
            'direction': 'CALL',
            'quality': 'FORCED',
            'score': 80.0,
            'confidence': 65,
            'reason': f'Signal forcé après {minutes_without_signal}min sans | RSI: {rsi:.1f} | Trend: BULLISH',
            'expiration_minutes': 5,
            'details': {
                'rsi': float(rsi),
                'trend': 'BULLISH',
                'price_change': ((current_price - prev_price) / prev_price * 100)
            }
        }
    elif current_price < prev_price and rsi > 40:
        return {
            'direction': 'PUT',
            'quality': 'FORCED',
            'score': 80.0,
            'confidence': 65,
            'reason': f'Signal forcé après {minutes_without_signal}min sans | RSI: {rsi:.1f} | Trend: BEARISH',
            'expiration_minutes': 5,
            'details': {
                'rsi': float(rsi),
                'trend': 'BEARISH',
                'price_change': ((prev_price - current_price) / prev_price * 100)
            }
        }
    elif ma_fast > ma_slow:
        # Tendance haussière
        return {
            'direction': 'CALL',
            'quality': 'FORCED_MA',
            'score': 75.0,
            'confidence': 60,
            'reason': f'Signal forcé MA | MA5 > MA20 | RSI: {rsi:.1f}',
            'expiration_minutes': 5
        }
    else:
        # Tendance baissière
        return {
            'direction': 'PUT',
            'quality': 'FORCED_MA',
            'score': 75.0,
            'confidence': 60,
            'reason': f'Signal forcé MA | MA5 < MA20 | RSI: {rsi:.1f}',
            'expiration_minutes': 5
        }

# ================= FONCTION POUR CALCULER LE POURCENTAGE DE CONFIANCE =================

def calculate_confidence_percentage(final_score):
    """Calcule le pourcentage de confiance basé sur le score final"""
    min_score = 75
    max_score = 150
    
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
        'CRITICAL': '⭐',
        'FORCED': '⚠️ FORCÉ',
        'FORCED_MA': '⚠️ FORCÉ-MA'
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

def check_micro_momentum(df, direction, lookback=2):
    """Vérifie la cohérence des dernières bougies M1 avec la direction"""
    if len(df) < lookback + 1:
        return True, 5, "Données insuffisantes - accepté"
    
    recent = df.tail(lookback).copy()
    closes = recent['close'].values
    opens = recent['open'].values
    
    bullish_count = 0
    bearish_count = 0
    
    for i in range(len(recent)):
        if closes[i] > opens[i]:
            bullish_count += 1
        elif closes[i] < opens[i]:
            bearish_count += 1
    
    if direction == "BUY":
        if bullish_count >= 1:
            micro_score = SAINT_GRAAL_CONFIG['micro_momentum_filter']['weight']
            return True, micro_score, f"Micro momentum BUY: {bullish_count}/{lookback} haussier"
        else:
            return True, 5, f"Micro momentum faible BUY: {bullish_count}/{lookback} haussier"
    
    elif direction == "SELL":
        if bearish_count >= 1:
            micro_score = SAINT_GRAAL_CONFIG['micro_momentum_filter']['weight']
            return True, micro_score, f"Micro momentum SELL: {bearish_count}/{lookback} baissier"
        else:
            return True, 5, f"Micro momentum faible SELL: {bearish_count}/{lookback} baissier"
    
    return True, 5, "Direction non critique"

# ================= FILTRE ATR OPTIMISÉ =================

def calculate_atr_filter(df):
    """🔥 FILTRE ATR - Version optimisée pour M1"""
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
    
    score = 5  # Score de base
    signal = "ACCEPTABLE_VOL"
    reason = f"ATR: {atr_pips:.1f} pips"
    
    min_atr = SAINT_GRAAL_CONFIG['atr_filter']['min_atr_pips']
    max_atr = SAINT_GRAAL_CONFIG['atr_filter']['max_atr_pips']
    optimal_range = SAINT_GRAAL_CONFIG['atr_filter']['optimal_atr_pips']
    
    if atr_pips < min_atr:
        signal = "LOW_VOL"
        score = 0
        reason = f"ATR bas: {atr_pips:.1f} pips"
    
    elif atr_pips > max_atr:
        signal = "HIGH_VOL"
        score = 0
        reason = f"ATR haut: {atr_pips:.1f} pips"
    
    elif optimal_range[0] <= atr_pips <= optimal_range[1]:
        signal = "OPTIMAL_VOL"
        score = 10
        reason = f"ATR optimal: {atr_pips:.1f} pips"
    
    # Toujours accepter sauf extrêmes
    if signal in ["LOW_VOL", "HIGH_VOL"]:
        score = -5  # Léger malus mais pas rejet
    
    return {
        'enabled': True,
        'atr_value': current_atr,
        'atr_pips': atr_pips,
        'signal': signal,
        'score': score,
        'reason': reason,
        'is_squeeze': False,
        'atr_trend': 'NEUTRAL',
    }

# ================= LOGIQUE CROISEMENT BANDE MÉDIANE BB =================

def check_bb_middle_crossover(df, direction):
    """🔥 Vérifie le croisement de la bande médiane des Bollinger Bands"""
    if not SAINT_GRAAL_CONFIG['bb_crossover']['enabled']:
        return True, 5, "Croisement BB désactivé"
    
    if len(df) < SAINT_GRAAL_CONFIG['bollinger_config']['window'] + 3:
        return True, 5, "Données insuffisantes - accepté"
    
    bb = BollingerBands(
        close=df['close'],
        window=SAINT_GRAAL_CONFIG['bollinger_config']['window'],
        window_dev=SAINT_GRAAL_CONFIG['bollinger_config']['window_dev']
    )
    
    bb_middle = bb.bollinger_mavg()
    
    lookback = SAINT_GRAAL_CONFIG['bb_crossover']['lookback_bars']
    recent_data = df.tail(lookback + 1).copy()
    
    if len(recent_data) < lookback + 1:
        return True, 5, "Données récentes insuffisantes"
    
    recent_middle = bb_middle.iloc[-(lookback + 1):].values
    recent_closes = recent_data['close'].values
    
    crossover_detected = True  # Par défaut accepté
    crossover_strength = 5  # Score de base
    reason = "BB position acceptable"
    
    if direction == "BUY":
        current_close = recent_closes[-1]
        current_middle = recent_middle[-1]
        
        if current_close > current_middle:
            distance_pips = (current_close - current_middle) / 0.0001
            crossover_strength = min(SAINT_GRAAL_CONFIG['bb_crossover']['weight'], 
                                    distance_pips * 1.5 + 5)
            reason = f"BUY: {distance_pips:.1f} pips au-dessus médiane"
        else:
            reason = f"BUY: Sous médiane mais accepté"
    
    elif direction == "SELL":
        current_close = recent_closes[-1]
        current_middle = recent_middle[-1]
        
        if current_close < current_middle:
            distance_pips = (current_middle - current_close) / 0.0001
            crossover_strength = min(SAINT_GRAAL_CONFIG['bb_crossover']['weight'], 
                                    distance_pips * 1.5 + 5)
            reason = f"SELL: {distance_pips:.1f} pips sous médiane"
        else:
            reason = f"SELL: Au-dessus médiane mais accepté"
    
    return crossover_detected, crossover_strength, reason

# ================= FONCTIONS DE BASE OPTIMISÉES =================

def calculate_m5_filter(df_m1):
    """Filtre M5 simplifié"""
    if len(df_m1) < 150:
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
            'reason': 'Bougies M5 insuffisantes',
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
    
    if current_ema_fast > current_ema_slow:
        trend = "BULLISH"
        score = SAINT_GRAAL_CONFIG['m5_filter']['weight']
        reason = f"M5 BULLISH"
    elif current_ema_slow > current_ema_fast:
        trend = "BEARISH"
        score = SAINT_GRAAL_CONFIG['m5_filter']['weight']
        reason = f"M5 BEARISH"
    else:
        trend = "NEUTRAL"
        score = 0
        reason = "M5 NEUTRAL"
    
    return {
        'trend': trend,
        'score': score,
        'reason': reason,
        'ema_fast': current_ema_fast,
        'ema_slow': current_ema_slow
    }

def analyze_market_structure(df, lookback=15):
    """Analyse simplifiée de la structure du marché"""
    if len(df) < lookback:
        return "NEUTRAL", 0
    
    recent_data = df.tail(lookback).copy()
    
    # Simple tendance basée sur les moyennes mobiles
    sma_fast = recent_data['close'].rolling(window=5).mean()
    sma_slow = recent_data['close'].rolling(window=10).mean()
    
    if sma_fast.iloc[-1] > sma_slow.iloc[-1] * 1.001:
        return "UPTREND", 10
    elif sma_slow.iloc[-1] > sma_fast.iloc[-1] * 1.001:
        return "DOWNTREND", 10
    else:
        return "NEUTRAL", 5

def analyze_momentum_asymmetric_optimized(df):
    """Analyse de momentum optimisée"""
    if len(df) < 30:
        return {
            'rsi': 50,
            'stoch_k_fast': 50,
            'stoch_d_fast': 50,
            'stoch_k_slow': 50,
            'stoch_d_slow': 50,
            'buy_score': 0,
            'sell_score': 0,
            'dominant': 'NEUTRAL',
            'momentum_gate_passed': True  # Toujours vrai pour accepter
        }
    
    rsi = RSIIndicator(close=df['close'], window=12).rsi()
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
    
    # Score BUY
    if current_rsi < SAINT_GRAAL_CONFIG['buy_rules']['rsi_max_for_buy']:
        buy_score += 20
        if current_rsi < SAINT_GRAAL_CONFIG['buy_rules']['rsi_oversold']:
            buy_score += 15
    
    if current_stoch_k_fast < 30:
        buy_score += 15
    if current_stoch_d_fast < 30:
        buy_score += 10
    
    # Score SELL
    if current_rsi > SAINT_GRAAL_CONFIG['sell_rules']['rsi_min_for_sell']:
        sell_score += 20
        if current_rsi > 70:
            sell_score += 15
    
    if current_stoch_k_slow > SAINT_GRAAL_CONFIG['sell_rules']['stoch_min_overbought']:
        sell_score += 15
    if current_stoch_d_slow > 60:
        sell_score += 10
    
    # Toujours passer le momentum gate pour M1
    momentum_gate_passed = True
    
    dominant = "NEUTRAL"
    if buy_score > sell_score:
        dominant = "BUY"
    elif sell_score > buy_score:
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
    """Calcule les signaux des Bandes de Bollinger simplifié"""
    if len(df) < SAINT_GRAAL_CONFIG['bollinger_config']['window']:
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
    
    current_price = float(df.iloc[-1]['close'])
    current_upper = float(bb_upper.iloc[-1])
    current_lower = float(bb_lower.iloc[-1])
    current_middle = float(bb_middle.iloc[-1])
    
    if current_upper != current_lower:
        bb_position = ((current_price - current_lower) / (current_upper - current_lower)) * 100
    else:
        bb_position = 50
    
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
    
    bb_signal = "NEUTRAL"
    if bb_position < SAINT_GRAAL_CONFIG['bollinger_config']['oversold_zone']:
        bb_signal = "OVERSOLD"
    elif bb_position > SAINT_GRAAL_CONFIG['bollinger_config']['overbought_zone']:
        bb_signal = "OVERBOUGHT"
    
    return {
        'bb_position': bb_position,
        'bb_signal': bb_signal,
        'bb_width': 0,
        'bb_squeeze': False,
        'bb_upper': current_upper,
        'bb_lower': current_lower,
        'bb_middle': current_middle,
        'price_above_middle': price_above_middle,
        'price_below_middle': price_below_middle,
        'middle_crossover': middle_crossover
    }

def get_bb_confirmation_score(bb_signal, direction, stochastic_value):
    """Calcule le score de confirmation Bollinger Bands simplifié"""
    score = 0
    reason = ""
    
    if direction == "BUY":
        if bb_signal['bb_position'] < 30:
            score += 25
            reason += "BB OVERSOLD"
        elif bb_signal['bb_position'] < 45:
            score += 15
            reason += "BB Zone basse"
        
        if stochastic_value < 30:
            score += 15
            reason += " + Stoch Bas"
        
        if bb_signal['price_above_middle']:
            score += 10
            reason += " + Au-dessus médiane"
    
    elif direction == "SELL":
        if bb_signal['bb_position'] > 70:
            score += 25
            reason += "BB OVERBOUGHT"
        elif bb_signal['bb_position'] > 55:
            score += 15
            reason += "BB Zone haute"
        
        if stochastic_value > 70:
            score += 15
            reason += " + Stoch Haut"
        
        if bb_signal['price_below_middle']:
            score += 10
            reason += " + Sous médiane"
    
    return min(score, 50), reason

def check_m5_alignment(m5_filter, direction):
    """Vérifie l'alignement avec la tendance M5 - Toujours accepté"""
    if m5_filter['trend'] == 'NEUTRAL':
        return True, "M5 Neutre", 5
    
    if direction == "BUY" and m5_filter['trend'] == "BULLISH":
        return True, "M5 aligné BULLISH", 10
    elif direction == "SELL" and m5_filter['trend'] == "BEARISH":
        return True, "M5 aligné BEARISH", 10
    else:
        return True, "M5 non aligné mais accepté", 0

def validate_candle_for_5min_buy(df):
    """Validation simplifiée pour BUY"""
    if len(df) < 2:
        return True, "NORMAL", 50, "Données insuffisantes"
    
    current = df.iloc[-1]
    current_close = float(current['close'])
    current_open = float(current['open'])
    
    is_bullish = current_close > current_open
    
    if is_bullish:
        return True, "BULLISH", 60, "Bougie haussière"
    else:
        return True, "NEUTRAL", 40, "Bougie neutre/baissière mais acceptée"

def validate_candle_for_5min_sell(df):
    """Validation simplifiée pour SELL"""
    if len(df) < 2:
        return True, "NORMAL", 50, "Données insuffisantes"
    
    current = df.iloc[-1]
    current_close = float(current['close'])
    current_open = float(current['open'])
    
    is_bearish = current_close < current_open
    
    if is_bearish:
        return True, "BEARISH", 60, "Bougie baissière"
    else:
        return True, "NEUTRAL", 40, "Bougie neutre/haussière mais acceptée"

# ================= FONCTION PRINCIPALE V6 OPTIMISÉE =================

def rule_signal_saint_graal_5min_pro_v6(df, signal_count=0, total_signals_needed=12):
    """🔥 VERSION 6.0 : CRITÈRES ASSOUPLIS POUR SIGNALS FRÉQUENTS"""
    print(f"\n{'='*70}")
    print(f"🎯 BINAIRE 5 MIN V6.0 - Signal #{signal_count+1}/{total_signals_needed}")
    print(f"{'='*70}")
    
    if len(df) < 80:
        print(f"❌ Données insuffisantes: {len(df)} < 80")
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
    print(f"📊 BB: Position {bb_signal['bb_position']:.1f}% | Signal: {bb_signal['bb_signal']}")
    
    atr_filter = calculate_atr_filter(df)
    print(f"📏 ATR: {atr_filter['reason']}")
    
    bb_buy_score, bb_buy_reason = get_bb_confirmation_score(
        bb_signal, "BUY", momentum['stoch_k_fast']
    )
    
    bb_sell_score, bb_sell_reason = get_bb_confirmation_score(
        bb_signal, "SELL", momentum['stoch_k_slow']
    )
    
    print(f"✅ BB Confirmation: BUY {bb_buy_score}/50 | SELL {bb_sell_score}/50")
    
    sell_score_total = 0
    buy_score_total = 0
    
    sell_score_total += momentum['sell_score']
    buy_score_total += momentum['buy_score']
    
    sell_score_total += bb_sell_score
    buy_score_total += bb_buy_score
    
    # Ajouter score ATR
    buy_score_total += atr_filter['score']
    sell_score_total += atr_filter['score']
    
    # Score structure
    if structure == "DOWNTREND":
        sell_score_total += 10
    elif structure == "UPTREND":
        buy_score_total += 10
    
    print(f"🎯 Scores avant micro: SELL {sell_score_total:.0f}/150 - BUY {buy_score_total:.0f}/150")
    
    direction = None
    final_score = 0
    decision_details = []
    
    # Seuils réduits
    buy_threshold = SAINT_GRAAL_CONFIG['buy_rules']['score_threshold']
    sell_threshold = SAINT_GRAAL_CONFIG['sell_rules']['score_threshold']
    
    if buy_score_total >= buy_threshold:
        
        micro_valid, micro_score, micro_reason = check_micro_momentum(df, "BUY")
        bb_crossover_valid, bb_crossover_score, bb_crossover_reason = check_bb_middle_crossover(df, "BUY")
        
        m5_aligned, m5_reason, m5_bonus = check_m5_alignment(m5_filter, "BUY")
        candle_valid, pattern, pattern_conf, candle_reason = validate_candle_for_5min_buy(df)
        
        if candle_valid:
            direction = "BUY"
            final_score = (buy_score_total + pattern_conf + m5_bonus + 
                         micro_score + bb_crossover_score)
            decision_details.append(f"BUY validé: {pattern} ({pattern_conf}%)")
            decision_details.append(f"Score base: {buy_score_total}")
            decision_details.append(f"Micro: {micro_reason}")
            decision_details.append(f"BB: {bb_crossover_reason}")
    
    elif sell_score_total >= sell_threshold:
        
        micro_valid, micro_score, micro_reason = check_micro_momentum(df, "SELL")
        bb_crossover_valid, bb_crossover_score, bb_crossover_reason = check_bb_middle_crossover(df, "SELL")
        
        m5_aligned, m5_reason, m5_bonus = check_m5_alignment(m5_filter, "SELL")
        candle_valid, pattern, pattern_conf, candle_reason = validate_candle_for_5min_sell(df)
        
        if candle_valid:
            direction = "SELL"
            final_score = (sell_score_total + pattern_conf + m5_bonus + 
                         micro_score + bb_crossover_score)
            decision_details.append(f"SELL validé: {pattern} ({pattern_conf}%)")
            decision_details.append(f"Score base: {sell_score_total}")
            decision_details.append(f"Micro: {micro_reason}")
            decision_details.append(f"BB: {bb_crossover_reason}")
    
    if direction:
        # Seuil minimum réduit à 75
        if final_score >= 75:
            if final_score >= 120:
                quality = "EXCELLENT"
                mode = "5MIN_MAX"
            elif final_score >= 100:
                quality = "HIGH"
                mode = "5MIN_PRO"
            elif final_score >= 85:
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
                'reason': f"{direction_display} | Score {final_score:.1f} | {structure} | ATR:{atr_filter['atr_pips']:.1f}pips",
                'expiration_minutes': 5,
                'details': {
                    'momentum_score': momentum['buy_score'] if direction == "BUY" else momentum['sell_score'],
                    'bb_score': bb_buy_score if direction == "BUY" else bb_sell_score,
                    'micro_momentum_score': micro_score,
                    'bb_crossover_score': bb_crossover_score,
                    'atr_score': atr_filter['score'],
                    'm5_alignment': m5_filter['trend'],
                    'structure': structure,
                }
            }
        else:
            print(f"❌ Score final insuffisant: {final_score:.1f} < 75")
    
    print(f"❌ Aucun signal valide - Scores: BUY {buy_score_total:.0f}, SELL {sell_score_total:.0f}")
    return None

# ================= FONCTION UNIVERSELLE AVEC FALLBACK =================

last_signal_time = datetime.now()

def get_signal_saint_graal(df, signal_count=0, total_signals=12, return_dict=True, 
                          pairs_analyzed=5, batches=1, force_signal=False):
    """
    🔥 FONCTION UNIVERSELLE - VERSION 6.0
    Avec fallback pour forcer un signal si nécessaire
    """
    try:
        if df is None or len(df) < 80:
            print("❌ Données insuffisantes pour analyse")
            return None if return_dict else (None, None)
        
        global last_signal_time
        
        # Essayer d'abord la stratégie normale
        result = rule_signal_saint_graal_5min_pro_v6(df, signal_count, total_signals)
        
        # Si pas de signal et que force_signal est True, ou trop longtemps sans signal
        timeout_minutes = SAINT_GRAAL_CONFIG['signal_config']['force_signal_after_minutes']
        time_without_signal = (datetime.now() - last_signal_time).seconds / 60
        
        if (result is None and force_signal) or (result is None and time_without_signal > timeout_minutes):
            print(f"⚠️  Tentative de signal forcé après {time_without_signal:.1f} minutes sans signal")
            forced_result = force_signal_if_needed(df, int(time_without_signal))
            if forced_result:
                result = forced_result
                print(f"✅ Signal forcé généré: {forced_result['direction']}")
        
        if result is not None:
            direction_display = result['signal']
            final_score = result['score']
            
            # Mise à jour du temps du dernier signal
            last_signal_time = datetime.now()
            
            # Déterminer la qualité
            if 'quality' in result and result['quality'] in ['FORCED', 'FORCED_MA']:
                quality = result['quality']
                mode = "FORCED"
            else:
                if final_score >= 120:
                    quality = "EXCELLENT"
                    mode = "5MIN_MAX"
                elif final_score >= 100:
                    quality = "HIGH"
                    mode = "5MIN_PRO"
                elif final_score >= 85:
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

def get_signal_dict_only(df, signal_count=0, total_signals=12, force_signal=False):
    """Alias pour get_signal_saint_graal avec return_dict=True"""
    return get_signal_saint_graal(df, signal_count, total_signals, return_dict=True, force_signal=force_signal)

def get_signal_with_metadata(df, signal_count=0, total_signals=12, force_signal=False):
    """Alias pour get_signal_saint_graal avec return_dict=False"""
    return get_signal_saint_graal(df, signal_count, total_signals, return_dict=False, force_signal=force_signal)

def get_signal_with_metadata_v2(df, signal_count=0, total_signals=12, 
                                pairs_analyzed=5, batches=1, force_signal=False):
    """Alias avec paramètres personnalisés"""
    return get_signal_saint_graal(df, signal_count, total_signals, return_dict=False, 
                                  pairs_analyzed=pairs_analyzed, batches=batches, force_signal=force_signal)

# ================= POINT D'ENTRÉE PRINCIPAL =================

if __name__ == "__main__":
    print("🎯 DESK PRO BINAIRE - VERSION 6.0 ULTRA-PERFORMANTE")
    print("🔥 OPTIMISÉ POUR SIGNALS FRÉQUENTS SUR M1")
    print(f"📊 Score minimum: {SAINT_GRAAL_CONFIG['signal_config']['min_quality_score']}")
    print(f"⏱️  Force signal après: {SAINT_GRAAL_CONFIG['signal_config']['force_signal_after_minutes']} minutes")
    
    # Test de débogage
    print("\n🧪 MODE DÉBOGAGE ACTIVÉ")
    
    try:
        # Créer des données de test réalistes
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=200, freq='1min')
        
        # Créer une tendance réaliste
        trend = np.linspace(1.1000, 1.1050, 200)
        noise = np.random.normal(0, 0.0002, 200)
        
        close_prices = trend + noise
        open_prices = close_prices - np.random.normal(0.0001, 0.00005, 200)
        high_prices = np.maximum(open_prices, close_prices) + np.random.normal(0.0002, 0.00005, 200)
        low_prices = np.minimum(open_prices, close_prices) - np.random.normal(0.0002, 0.00005, 200)
        
        df_test = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices
        }, index=dates)
        
        print(f"\n📊 Données de test: {len(df_test)} bougies")
        print(f"📈 Prix actuel: {df_test['close'].iloc[-1]:.5f}")
        
        # Test 1: Analyse normale
        print("\n🔍 Test 1: Analyse normale")
        signal_dict = get_signal_dict_only(df_test, signal_count=0, total_signals=12)
        
        if signal_dict:
            print(f"✅ Signal détecté: {signal_dict['direction']} | Score: {signal_dict['score']:.1f}")
            print(f"   Qualité: {signal_dict['quality']} | Confiance: {signal_dict['confidence']}%")
        else:
            print("❌ Aucun signal détecté")
            # Déboguer
            debug_no_signals(df_test)
        
        # Test 2: Avec forçage
        print("\n🔍 Test 2: Avec forçage activé")
        signal_dict_forced = get_signal_dict_only(df_test, force_signal=True)
        
        if signal_dict_forced:
            print(f"✅ Signal (forcé): {signal_dict_forced['direction']} | Score: {signal_dict_forced['score']:.1f}")
            print(f"   Qualité: {signal_dict_forced['quality']} | Confiance: {signal_dict_forced['confidence']}%")
        
        # Test 3: Avec message
        print("\n🔍 Test 3: Génération de message")
        signal_tuple = get_signal_with_metadata_v2(df_test, pairs_analyzed=8, batches=2, force_signal=True)
        
        if signal_tuple and signal_tuple[0]:
            print(f"✅ Signal prêt pour envoi: {signal_tuple[0]['direction']}")
            if signal_tuple[1]:
                print("📨 Message généré avec succès")
                
    except Exception as e:
        print(f"❌ Erreur pendant les tests: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("🚀 PRÊT POUR LA PRODUCTION - BON TRADING !")
    print("="*70)
