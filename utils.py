"""
utils.py - STRATÉGIE BINAIRE M1 PRO - VERSION 4.6 ULTIMATE PLUS COMPLÈTE
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
    """
    Calcule le pourcentage de confiance basé sur le score final
    Score min: 90 = 50% de confiance
    Score max: 200 = 100% de confiance
    """
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
    """
    Génère un message formaté pour le signal
    """
    if not signal_data:
        return None
    
    direction = signal_data['direction']
    quality = signal_data['quality']
    score = signal_data['score']
    confidence = calculate_confidence_percentage(score)
    
    # Déterminer l'emoji et le texte de direction
    if direction == "CALL":
        direction_emoji = "↗️"
        direction_text = "CALL ↗️"
    else:
        direction_emoji = "↘️"
        direction_text = "PUT ↘️"
    
    # Déterminer les étoiles de qualité
    quality_stars = {
        'EXCELLENT': '⭐⭐⭐⭐⭐',
        'HIGH': '⭐⭐⭐⭐',
        'SOLID': '⭐⭐⭐',
        'MINIMUM': '⭐⭐',
        'CRITICAL': '⭐'
    }.get(quality, '⭐')
    
    # Heure actuelle
    current_time = datetime.now().strftime("%H:%M")
    
    # Créer le message formaté
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
    """
    🔥 MICRO GARDE-FOU MOMENTUM
    Vérifie la cohérence des dernières bougies M1 avec la direction
    """
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
    
    # Analyse des dernières bougies
    for i in range(len(recent)):
        # Bougie haussière
        if closes[i] > opens[i]:
            bullish_count += 1
        # Bougie baissière
        elif closes[i] < opens[i]:
            bearish_count += 1
        
        # Alignement des prix (tendance micro)
        if i > 0:
            if closes[i] > closes[i-1]:
                price_alignment += 1
            elif closes[i] < closes[i-1]:
                price_alignment -= 1
    
    # 🔥 LOGIQUE DE CONFIRMATION MICRO
    if direction == "BUY":
        min_bullish = SAINT_GRAAL_CONFIG['micro_momentum_filter']['min_bullish_bars']
        
        # Vérification 1: Nombre de bougies haussières
        if bullish_count < min_bullish:
            return False, -20, f"Seulement {bullish_count}/{lookback} bougies haussières"
        
        # Vérification 2: Alignement des prix
        if (SAINT_GRAAL_CONFIG['micro_momentum_filter']['require_price_alignment'] and 
            price_alignment < 1):
            return False, -15, f"Alignement prix faible: {price_alignment}"
        
        # Vérification 3: Momentum des hauts
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
        
        # Momentum des bas
        lows_decreasing = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
        if lows_decreasing >= 2:
            micro_score = SAINT_GRAAL_CONFIG['micro_momentum_filter']['weight']
            return True, micro_score, f"Micro momentum SELL: {bearish_count}/{lookback} baissier, prix alignés"
        else:
            return True, 8, f"Micro momentum SELL faible: {bearish_count}/{lookback} baissier"
    
    return False, 0, "Direction non reconnue"

# ================= FILTRE ATR =================

def calculate_atr_filter(df):
    """
    🔥 FILTRE ATR - Analyse de la volatilité
    """
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
    
    # Calcul ATR
    atr_indicator = AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=SAINT_GRAAL_CONFIG['atr_filter']['window']
    )
    
    atr_values = atr_indicator.average_true_range()
    current_atr = float(atr_values.iloc[-1])
    atr_pips = current_atr / 0.0001  # Conversion en pips
    
    # ATR précédent pour tendance
    if len(atr_values) > 1:
        prev_atr = float(atr_values.iloc[-2])
        atr_trend = "RISING" if current_atr > prev_atr else "FALLING"
    else:
        atr_trend = "NEUTRAL"
    
    # Détection squeeze (volatilité très basse)
    avg_atr = atr_values.tail(50).mean() if len(atr_values) >= 50 else current_atr
    is_squeeze = current_atr < avg_atr * 0.6
    
    # Évaluation ATR
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
        
        # Bonus si ATR en hausse (momentum)
        if atr_trend == "RISING":
            score += SAINT_GRAAL_CONFIG['atr_filter']['atr_trend_weight']
            reason += f" (hausse, momentum favorable)"
    
    else:
        signal = "ACCEPTABLE_VOL"
        score = 5
        reason = f"ATR acceptable: {atr_pips:.1f} pips"
    
    # Bonus squeeze pour breakout potentiel
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
    """
    🔥 NOUVEAU: Vérifie le croisement de la bande médiane des Bollinger Bands
    Retourne True si la bougie actuelle ferme au-dessus (BUY) ou en-dessous (SELL) de la bande médiane
    """
    if not SAINT_GRAAL_CONFIG['bb_crossover']['enabled']:
        return True, 0, "Croisement BB désactivé"
    
    if len(df) < SAINT_GRAAL_CONFIG['bollinger_config']['window'] + 5:
        return False, 0, "Données insuffisantes pour BB crossover"
    
    # Calcul des Bandes de Bollinger
    bb = BollingerBands(
        close=df['close'],
        window=SAINT_GRAAL_CONFIG['bollinger_config']['window'],
        window_dev=SAINT_GRAAL_CONFIG['bollinger_config']['window_dev']
    )
    
    bb_middle = bb.bollinger_mavg()
    
    # Vérifier les dernières bougies
    lookback = SAINT_GRAAL_CONFIG['bb_crossover']['lookback_bars']
    recent_data = df.tail(lookback + 1).copy()  # +1 pour comparer avec la précédente
    
    if len(recent_data) < lookback + 1:
        return False, 0, f"Pas assez de données: {len(recent_data)} < {lookback + 1}"
    
    # Obtenir les valeurs de bande médiane correspondantes
    recent_middle = bb_middle.iloc[-(lookback + 1):].values
    recent_closes = recent_data['close'].values
    recent_opens = recent_data['open'].values
    
    crossover_detected = False
    crossover_strength = 0
    reason = ""
    
    if direction == "BUY":
        # Vérifier si la bougie actuelle ferme AU-DESSUS de la bande médiane
        current_close = recent_closes[-1]
        current_middle = recent_middle[-1]
        
        # Vérifier la bougie précédente (pour confirmation)
        prev_close = recent_closes[-2] if lookback >= 1 else None
        prev_middle = recent_middle[-2] if lookback >= 1 else None
        
        # Logique BUY: Fermeture au-dessus de la bande médiane
        if current_close > current_middle:
            crossover_detected = True
            
            # Calculer la force du croisement (distance en pips)
            distance_pips = (current_close - current_middle) / 0.0001
            crossover_strength = min(SAINT_GRAAL_CONFIG['bb_crossover']['weight'], 
                                    distance_pips * 2)  # Bonus proportionnel
            
            # Vérification de confirmation
            if SAINT_GRAAL_CONFIG['bb_crossover']['require_confirmation']:
                # Si la bougie précédente était en-dessous, c'est un vrai croisement
                if prev_close is not None and prev_middle is not None:
                    if prev_close < prev_middle:
                        crossover_strength += 5  # Bonus pour vrai croisement
                        reason = f"Vrai croisement BUY: {distance_pips:.1f} pips au-dessus"
                    else:
                        reason = f"BUY: Déjà au-dessus, {distance_pips:.1f} pips"
                else:
                    reason = f"BUY: {distance_pips:.1f} pips au-dessus de la médiane"
            else:
                reason = f"BUY: {distance_pips:.1f} pips au-dessus de la médiane"
            
            # Vérifier taille de bougie
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
        # Vérifier si la bougie actuelle ferme EN-DESSOUS de la bande médiane
        current_close = recent_closes[-1]
        current_middle = recent_middle[-1]
        
        # Vérifier la bougie précédente (pour confirmation)
        prev_close = recent_closes[-2] if lookback >= 1 else None
        prev_middle = recent_middle[-2] if lookback >= 1 else None
        
        # Logique SELL: Fermeture en-dessous de la bande médiane
        if current_close < current_middle:
            crossover_detected = True
            
            # Calculer la force du croisement (distance en pips)
            distance_pips = (current_middle - current_close) / 0.0001
            crossover_strength = min(SAINT_GRAAL_CONFIG['bb_crossover']['weight'], 
                                    distance_pips * 2)  # Bonus proportionnel
            
            # Vérification de confirmation
            if SAINT_GRAAL_CONFIG['bb_crossover']['require_confirmation']:
                # Si la bougie précédente était au-dessus, c'est un vrai croisement
                if prev_close is not None and prev_middle is not None:
                    if prev_close > prev_middle:
                        crossover_strength += 5  # Bonus pour vrai croisement
                        reason = f"Vrai croisement SELL: {distance_pips:.1f} pips en-dessous"
                    else:
                        reason = f"SELL: Déjà en-dessous, {distance_pips:.1f} pips"
                else:
                    reason = f"SELL: {distance_pips:.1f} pips en-dessous de la médiane"
            else:
                reason = f"SELL: {distance_pips:.1f} pips en-dessous de la médiane"
            
            # Vérifier taille de bougie
            current_open = recent_opens[-1]
            candle_size = abs(current_close - current_open) / 0.0001
            if candle_size >= SAINT_GRAAL_CONFIG['bb_crossover']['min_candle_size_pips']:
                crossover_strength += 3
                reason += f" (bougie forte: {candle_size:.1f} pips)"
            else:
                reason += f" (bougie faible: {candle_size:.1f} pips)"
        else:
            reason = f"SELL rejeté: Fermeture {current_close} > Bande médiane {current_middle:.5f}"
    
    # Mode strict: Rejeter si le croisement est trop faible
    if (SAINT_GRAAL_CONFIG['bb_crossover']['strict_mode'] and 
        crossover_detected and 
        crossover_strength < 5):
        return False, 0, f"Croisement trop faible: {crossover_strength:.1f}"
    
    return crossover_detected, crossover_strength, reason

# ================= FONCTIONS DE BASE =================

def calculate_m5_filter(df_m1):
    """Filtre M5 pour analyse de tendance"""
    if len(df_m1) < 300:  # Au moins 300 bougies M1 pour avoir des données M5 fiables
        return {
            'trend': 'NEUTRAL',
            'score': 0,
            'reason': 'Données M5 insuffisantes',
            'ema_fast': None,
            'ema_slow': None
        }
    
    # Resample en M5
    df_m5 = df_m1.resample('5T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum' if 'volume' in df_m1.columns else 'sum'
    }).dropna()
    
    if len(df_m5) < SAINT_GRAAL_CONFIG['m5_filter']['min_required_m5_bars']:
        return {
            'trend': 'NEUTRAL',
            'score': 0,
            'reason': 'Bougies M5 insuffisantes après resample',
            'ema_fast': None,
            'ema_slow': None
        }
    
    # Calcul des EMAs M5
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
    
    # Détermination de la tendance
    price = float(df_m5.iloc[-1]['close'])
    
    # Logique de tendance M5
    if current_ema_fast > current_ema_slow * 1.002:  # 0.2% de marge
        trend = "BULLISH"
        score = SAINT_GRAAL_CONFIG['m5_filter']['weight']
        reason = f"M5 BULLISH (EMA{SAINT_GRAAL_CONFIG['m5_filter']['ema_fast']}>{SAINT_GRAAL_CONFIG['m5_filter']['ema_slow']})"
        
        # Bonus si prix au-dessus des deux EMAs
        if price > current_ema_fast:
            score += 5
            reason += " - Prix > EMA rapide"
    
    elif current_ema_slow > current_ema_fast * 1.002:
        trend = "BEARISH"
        score = SAINT_GRAAL_CONFIG['m5_filter']['weight']
        reason = f"M5 BEARISH (EMA{SAINT_GRAAL_CONFIG['m5_filter']['ema_slow']}>{SAINT_GRAAL_CONFIG['m5_filter']['ema_fast']})"
        
        # Bonus si prix en-dessous des deux EMAs
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
    """Analyse la structure du marché (tendances, supports, résistances)"""
    if len(df) < lookback:
        return "NEUTRAL", 0
    
    recent_data = df.tail(lookback).copy()
    
    # Calcul des pivots (highs et lows)
    highs = recent_data['high'].values
    lows = recent_data['low'].values
    
    # Identification des swings
    swing_highs = []
    swing_lows = []
    
    for i in range(2, len(highs)-2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append(lows[i])
    
    # Analyse de tendance
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        last_2_highs = sorted(swing_highs)[-2:]
        last_2_lows = sorted(swing_lows)[-2:]
        
        # Tendance haussière: highs et lows croissants
        if last_2_highs[-1] > last_2_highs[-2] and last_2_lows[-1] > last_2_lows[-2]:
            # Calcul de la force (pourcentage de hausse)
            trend_strength = ((last_2_highs[-1] - last_2_highs[-2]) / last_2_highs[-2] * 100 + 
                            (last_2_lows[-1] - last_2_lows[-2]) / last_2_lows[-2] * 100) / 2
            return "UPTREND", trend_strength
        
        # Tendance baissière: highs et lows décroissants
        elif last_2_highs[-1] < last_2_highs[-2] and last_2_lows[-1] < last_2_lows[-2]:
            trend_strength = ((last_2_highs[-2] - last_2_highs[-1]) / last_2_highs[-2] * 100 + 
                            (last_2_lows[-2] - last_2_lows[-1]) / last_2_lows[-2] * 100) / 2
            return "DOWNTREND", trend_strength
    
    # Range ou neutre
    avg_range = (recent_data['high'].max() - recent_data['low'].min()) / recent_data['close'].mean() * 100
    
    if avg_range < 0.1:  # 0.1% de range
        return "CONSOLIDATION", avg_range
    else:
        return "NEUTRAL", avg_range

def analyze_momentum_asymmetric_optimized(df):
    """Analyse de momentum avec paramètres asymétriques pour BUY/SELL"""
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
    
    # RSI standard
    rsi = RSIIndicator(close=df['close'], window=14).rsi()
    current_rsi = float(rsi.iloc[-1])
    
    # Stochastique rapide (pour BUY)
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
    
    # Stochastique lent (pour SELL)
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
    
    # Calcul des scores
    buy_score = 0
    sell_score = 0
    
    # Logique BUY
    if current_rsi < SAINT_GRAAL_CONFIG['buy_rules']['rsi_max_for_buy']:
        buy_score += 25
        
        if current_rsi < SAINT_GRAAL_CONFIG['buy_rules']['rsi_oversold']:
            buy_score += 15
            buy_score += (SAINT_GRAAL_CONFIG['buy_rules']['rsi_oversold'] - current_rsi) * 2
    
    if current_stoch_k_fast < 20 and current_stoch_d_fast < 20:
        buy_score += 20
    elif current_stoch_k_fast < 30 and current_stoch_d_fast < 30:
        buy_score += 15
    
    # Logique SELL
    if current_rsi > SAINT_GRAAL_CONFIG['sell_rules']['rsi_min_for_sell']:
        sell_score += 25
        
        if current_rsi > 70:
            sell_score += 15
            sell_score += (current_rsi - 70) * 2
    
    if current_stoch_k_slow > SAINT_GRAAL_CONFIG['sell_rules']['stoch_min_overbought']:
        sell_score += 20
    elif current_stoch_k_slow > 75:
        sell_score += 25
    
    # Momentum gate (différence entre K et D)
    momentum_gate_diff_buy = abs(current_stoch_k_fast - current_stoch_d_fast)
    momentum_gate_diff_sell = abs(current_stoch_k_slow - current_stoch_d_slow)
    
    momentum_gate_passed = (
        (buy_score > 0 and momentum_gate_diff_buy >= SAINT_GRAAL_CONFIG['sell_rules']['momentum_gate_diff']) or
        (sell_score > 0 and momentum_gate_diff_sell >= SAINT_GRAAL_CONFIG['sell_rules']['momentum_gate_diff'])
    )
    
    # Détermination du momentum dominant
    dominant = "NEUTRAL"
    if buy_score > sell_score + 10:
        dominant = "BUY"
    elif sell_score > buy_score + 15:  # Plus strict pour SELL
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
    
    # Position en pourcentage (0 = bas de bande, 100 = haut de bande)
    if current_upper != current_lower:
        bb_position = ((current_price - current_lower) / (current_upper - current_lower)) * 100
    else:
        bb_position = 50
    
    # Détection squeeze (volatilité faible)
    avg_width = bb_width.tail(20).mean()
    current_width = float(bb_width.iloc[-1])
    bb_squeeze = current_width < avg_width * 0.7
    
    # 🔥 AJOUT: Position par rapport à la bande médiane
    price_above_middle = current_price > current_middle
    price_below_middle = current_price < current_middle
    
    # 🔥 AJOUT: Détection de croisement
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
    
    # Détermination du signal traditionnel
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
        # Score de position
        if bb_signal['bb_position'] < 30:
            score += 35
            reason += "BB OVERSOLD"
        elif bb_signal['bb_position'] < 40:
            score += 25
            reason += "BB Près du bas"
        elif bb_signal['bb_position'] < 50:
            score += 15
            reason += "BB Zone neutre basse"
        
        # Bonus squeeze pour rebond potentiel
        if bb_signal['bb_squeeze']:
            score += 10
            reason += " + SQUEEZE"
        
        # Alignement avec stochastique
        if stochastic_value < 30:
            score += 20
            reason += " + Stoch OVERSOLD"
        elif stochastic_value < 40:
            score += 10
            reason += " + Stoch Bas"
        
        # 🔥 NOUVEAU: Bonus pour position au-dessus de la bande médiane
        if bb_signal['price_above_middle']:
            score += SAINT_GRAAL_CONFIG['bollinger_config']['crossover_weight']
            reason += " + Au-dessus médiane"
            
            # Bonus supplémentaire pour vrai croisement
            if bb_signal['middle_crossover'] == "BULLISH_CROSS":
                score += 8
                reason += " (Croisement haussier)"
        
        # 🔥 VÉTO: Si prix en-dessous de la bande médiane en mode strict
        if (SAINT_GRAAL_CONFIG['bb_crossover']['strict_mode'] and 
            bb_signal['price_below_middle'] and 
            bb_signal['middle_crossover'] != "BULLISH_CROSS"):
            score -= 15
            reason += " - En-dessous médiane (vété)"
    
    elif direction == "SELL":
        # Score de position
        if bb_signal['bb_position'] > 70:
            score += 35
            reason += "BB OVERBOUGHT"
        elif bb_signal['bb_position'] > 60:
            score += 25
            reason += "BB Près du haut"
        elif bb_signal['bb_position'] > 50:
            score += 15
            reason += "BB Zone neutre haute"
        
        # Bonus squeeze
        if bb_signal['bb_squeeze']:
            score += 10
            reason += " + SQUEEZE"
        
        # Alignement avec stochastique
        if stochastic_value > 70:
            score += 20
            reason += " + Stoch OVERBOUGHT"
        elif stochastic_value > 60:
            score += 10
            reason += " + Stoch Haut"
        
        # 🔥 NOUVEAU: Bonus pour position en-dessous de la bande médiane
        if bb_signal['price_below_middle']:
            score += SAINT_GRAAL_CONFIG['bollinger_config']['crossover_weight']
            reason += " + En-dessous médiane"
            
            # Bonus supplémentaire pour vrai croisement
            if bb_signal['middle_crossover'] == "BEARISH_CROSS":
                score += 8
                reason += " (Croisement baissier)"
        
        # 🔥 VÉTO: Si prix au-dessus de la bande médiane en mode strict
        if (SAINT_GRAAL_CONFIG['bb_crossover']['strict_mode'] and 
            bb_signal['price_above_middle'] and 
            bb_signal['middle_crossover'] != "BEARISH_CROSS"):
            score -= 15
            reason += " - Au-dessus médiane (vété)"
    
    # Bonus si prix proche de la bande
    current_diff_to_band = 0
    if direction == "BUY":
        current_diff_to_band = abs(bb_signal['bb_lower'] - bb_signal['bb_middle'])
    else:
        current_diff_to_band = abs(bb_signal['bb_upper'] - bb_signal['bb_middle'])
    
    if current_diff_to_band > 0:
        band_proximity = min(100, (current_diff_to_band / bb_signal['bb_middle'] * 10000))
        if band_proximity < 15:  # Moins de 0.15% de la bande
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
    
    # Bougie haussière
    is_bullish = current_close > current_open
    
    # Taille de la bougie (en pips)
    candle_size = abs(current_close - current_open) / 0.0001
    
    # Volume (si disponible)
    volume_ok = True
    if 'volume' in df.columns:
        current_volume = float(current['volume']) if pd.notnull(current['volume']) else 0
        avg_volume = df['volume'].tail(20).mean()
        volume_ok = current_volume > avg_volume * 0.7
    
    # Configuration de bougie optimale
    pattern = "NORMAL"
    confidence = 40  # Base
    
    # Bougie haussière forte
    if is_bullish and candle_size > 5:  # Plus de 5 pips
        confidence += 20
        pattern = "BULLISH_STRONG"
    
    # Hammer ou inverted hammer
    lower_shadow = min(current_open, current_close) - float(current['low'])
    upper_shadow = float(current['high']) - max(current_open, current_close)
    
    if lower_shadow > candle_size * 2 and candle_size < lower_shadow * 0.3:
        confidence += 25
        pattern = "HAMMER"
    
    # Engulfing haussier
    if (is_bullish and not (prev_close > prev_open) and 
        current_close > prev_open and current_open < prev_close):
        confidence += 30
        pattern = "BULLISH_ENGULFING"
    
    # Morning star pattern (simplifié)
    if (prev2['close'] < prev2['open'] and  # Bougie baissière
        abs(prev_close - prev_open) < candle_size * 0.3 and  # Doji ou petite bougie
        is_bullish and current_close > prev2['close']):
        confidence += 35
        pattern = "MORNING_STAR"
    
    # Vérifications finales
    if not volume_ok:
        confidence -= 10
    
    # Vérifier qu'on n'achète pas au sommet
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
    
    # Bougie baissière
    is_bearish = current_close < current_open
    
    # Taille de la bougie
    candle_size = abs(current_close - current_open) / 0.0001
    
    # Volume
    volume_ok = True
    if 'volume' in df.columns:
        current_volume = float(current['volume']) if pd.notnull(current['volume']) else 0
        avg_volume = df['volume'].tail(20).mean()
        volume_ok = current_volume > avg_volume * 0.7
    
    # Configuration de bougie
    pattern = "NORMAL"
    confidence = 40
    
    # Bougie baissière forte
    if is_bearish and candle_size > 5:
        confidence += 20
        pattern = "BEARISH_STRONG"
    
    # Shooting star ou hanging man
    upper_shadow = float(current['high']) - max(current_open, current_close)
    lower_shadow = min(current_open, current_close) - float(current['low'])
    
    if upper_shadow > candle_size * 2 and candle_size < upper_shadow * 0.3:
        confidence += 25
        pattern = "SHOOTING_STAR"
    
    # Engulfing baissier
    if (is_bearish and not (prev_close < prev_open) and 
        current_close < prev_open and current_open > prev_close):
        confidence += 30
        pattern = "BEARISH_ENGULFING"
    
    # Evening star pattern (simplifié)
    if (prev2['close'] > prev2['open'] and  # Bougie haussière
        abs(prev_close - prev_open) < candle_size * 0.3 and  # Doji ou petite bougie
        is_bearish and current_close < prev2['close']):
        confidence += 35
        pattern = "EVENING_STAR"
    
    # Vérifications finales
    if not volume_ok:
        confidence -= 10
    
    # Vérifier qu'on ne vend pas au plus bas
    if current_close < df['close'].tail(20).min():
        confidence -= 15
    
    valid = confidence >= 50
    
    reason = f"{pattern} (Conf: {confidence}%)" if valid else f"Configuration faible: {pattern}"
    
    return valid, pattern, confidence, reason

# ================= FONCTION PRINCIPALE V4 =================

def rule_signal_saint_graal_5min_pro_v4(df, signal_count=0, total_signals_needed=6):
    """
    🔥 VERSION 4.6 : AVEC CROISEMENT BANDE MÉDIANE BB + MICRO MOMENTUM + FILTRE ATR
    """
    print(f"\n{'='*70}")
    print(f"🎯 BINAIRE 5 MIN V4.6 - Signal #{signal_count+1}/{total_signals_needed}")
    print(f"{'='*70}")
    
    if len(df) < 100:
        print(f"❌ Données insuffisantes: {len(df)} < 100")
        return None
    
    current_price = float(df.iloc[-1]['close'])
    
    # ===== 1. FILTRE M5 =====
    m5_filter = calculate_m5_filter(df)
    print(f"📈 Filtre M5: {m5_filter['reason']}")
    
    # ===== 2. ANALYSE STRUCTURE =====
    structure, trend_strength = analyze_market_structure(df)
    print(f"🏗️  Structure: {structure} | Force: {trend_strength:.1f}%")
    
    # ===== 3. MOMENTUM =====
    momentum = analyze_momentum_asymmetric_optimized(df)
    print(f"⚡ Momentum: RSI {momentum['rsi']:.1f} | StochF {momentum['stoch_k_fast']:.1f} | StochS {momentum['stoch_k_slow']:.1f}")
    
    # ===== 4. BOLLINGER BANDS AVEC CROISEMENT =====
    bb_signal = calculate_bollinger_signals(df)
    print(f"📊 BB: Position {bb_signal['bb_position']:.1f}% | Signal: {bb_signal['bb_signal']} | Croisement: {bb_signal['middle_crossover']}")
    
    # ===== 5. 🔥 FILTRE ATR =====
    atr_filter = calculate_atr_filter(df)
    print(f"📏 ATR: {atr_filter['reason']}")
    
    # ===== 6. LOGIQUE ZIGZAG-BB-STOCHASTIC =====
    bb_buy_score, bb_buy_reason = get_bb_confirmation_score(
        bb_signal, "BUY", momentum['stoch_k_fast']
    )
    
    bb_sell_score, bb_sell_reason = get_bb_confirmation_score(
        bb_signal, "SELL", momentum['stoch_k_slow']
    )
    
    print(f"✅ BB Confirmation: BUY {bb_buy_score}/70 | SELL {bb_sell_score}/70")
    
    # ===== 7. CALCUL SCORES COMPLETS =====
    sell_score_total = 0
    buy_score_total = 0
    sell_details = []
    buy_details = []
    
    # Score momentum
    sell_score_total += momentum['sell_score']
    buy_score_total += momentum['buy_score']
    
    # Score Bollinger
    sell_score_total += bb_sell_score
    buy_score_total += bb_buy_score
    
    # Score ATR (ajouté)
    if atr_filter['enabled']:
        # ATR affecte les deux côtés équitablement
        buy_score_total += atr_filter['score']
        sell_score_total += atr_filter['score']
        print(f"📏 Score ATR ajouté: {atr_filter['score']} points")
    
    # Filtre M5
    if SAINT_GRAAL_CONFIG['m5_filter']['strict_mode']:
        if m5_filter['trend'] == "BULLISH":
            buy_score_total += m5_filter['score']
            buy_details.append(f"Tendance M5: {m5_filter['trend']}")
        elif m5_filter['trend'] == "BEARISH":
            buy_score_total -= 10
        
        if m5_filter['trend'] == "BEARISH":
            sell_score_total += m5_filter['score']
            sell_details.append(f"Tendance M5: {m5_filter['trend']}")
        elif m5_filter['trend'] == "BULLISH":
            sell_score_total -= 10
    
    # Bonus structure
    if structure == "DOWNTREND" and momentum['dominant'] == "SELL":
        sell_score_total += 15
        sell_details.append("Tendance alignée")
    
    if structure == "UPTREND" and momentum['dominant'] == "BUY":
        buy_score_total += 15
        buy_details.append("Tendance alignée")
    
    print(f"🎯 Scores avant micro: SELL {sell_score_total}/200 - BUY {buy_score_total}/200")
    
    # ===== 8. DÉCISION AVEC NOUVEAUX FILTRES =====
    direction = None
    final_score = 0
    decision_details = []
    
    # 🔥 DÉCISION BUY
    if (buy_score_total >= 70 and momentum['momentum_gate_passed']):
        
        # 🔥 Vérification micro momentum
        micro_valid, micro_score, micro_reason = check_micro_momentum(df, "BUY")
        
        # 🔥 NOUVEAU: Vérification croisement bande médiane BB
        bb_crossover_valid, bb_crossover_score, bb_crossover_reason = check_bb_middle_crossover(df, "BUY")
        
        if not micro_valid:
            print(f"❌ Micro momentum BUY échoué: {micro_reason}")
        elif not bb_crossover_valid:
            print(f"❌ Croisement BB BUY échoué: {bb_crossover_reason}")
        else:
            # Vérification alignement M5
            m5_aligned, m5_reason, m5_bonus = check_m5_alignment(m5_filter, "BUY")
            
            if m5_aligned or not SAINT_GRAAL_CONFIG['signal_config']['require_m5_alignment']:
                # Validation bougie
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
                print(f"❌ BUY rejeté: {m5_reason}")
    
    # 🔥 DÉCISION SELL
    elif (sell_score_total >= 75 and momentum['momentum_gate_passed']):
        
        micro_valid, micro_score, micro_reason = check_micro_momentum(df, "SELL")
        
        # 🔥 NOUVEAU: Vérification croisement bande médiane BB
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
                print(f"❌ SELL rejeté: {m5_reason}")
    
    # ===== 9. VÉRIFICATION FINALE AVEC ATR =====
    if direction:
        # 🔥 VÉTO ATR: Rejeter si volatilité inappropriée
        if atr_filter['enabled'] and atr_filter['signal'] in ["AVOID_LOW_VOL", "AVOID_HIGH_VOL"]:
            print(f"❌ Signal rejeté par ATR: {atr_filter['reason']}")
            return None
        
        # Vérification score final
        if final_score >= 90:
            # Déterminer la qualité
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
    
    # ===== 10. PAS DE SIGNAL =====
    print(f"❌ Aucun signal valide - Score insuffisant ou filtres échoués")
    return None

# ================= FONCTION PRINCIPALE CORRIGÉE =================

def get_signal_with_metadata_v2(df, signal_count=0, total_signals=6, pairs_analyzed=5, batches=1):
    """
    🔥 VERSION CORRIGÉE avec calcul de confiance et message formaté
    """
    try:
        if df is None or len(df) < 100:
            print("❌ Données insuffisantes pour analyse")
            return None, None
        
        # Utiliser la version avec croisement BB, micro momentum et ATR
        result = rule_signal_saint_graal_5min_pro_v4(df, signal_count, total_signals)
        
        if result is not None:
            direction_display = result['signal']
            
            # Qualité basée sur le score
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
            
            # Créer l'objet signal
            signal_obj = {
                'direction': direction_display,
                'mode': mode,
                'quality': quality,
                'score': float(final_score),
                'confidence': calculate_confidence_percentage(final_score),
                'expiration_minutes': 5,
                'details': result.get('details', {}),
                'raw_result': result
            }
            
            # Générer le message formaté
            message = generate_signal_message(signal_obj, pairs_analyzed, batches)
            
            return signal_obj, message
        
        print(f"🎯 Aucun signal valide - Session {signal_count+1}/{total_signals}")
        return None, None
        
    except Exception as e:
        print(f"❌ Erreur critique: {str(e)}")
        return None, None

# ================= FONCTION DE COMPATIBILITÉ (pour code existant) =================

def get_signal_with_metadata(df, signal_count=0, total_signals=6):
    """
    Fonction de compatibilité pour l'ancien code
    Utilise get_signal_with_metadata_v2 avec valeurs par défaut
    """
    return get_signal_with_metadata_v2(df, signal_count, total_signals, 5, 1)

# ================= POINT D'ENTRÉE PRINCIPAL =================

if __name__ == "__main__":
    print("🎯 DESK PRO BINAIRE - VERSION 4.6 ULTIMATE PLUS COMPLÈTE")
    print("🔥 FONCTIONS DISPONIBLES:")
    print("   1. get_signal_with_metadata_v2() - Nouvelle version avec message formaté")
    print("   2. get_signal_with_metadata()    - Compatibilité ancien code")
    print("   3. rule_signal_saint_graal_5min_pro_v4() - Logique complète V4.6")
    print("\n✅ Toutes les fonctions sont intégrées et opérationnelles!")
    
    # Exemple d'utilisation
    print("\n📋 EXEMPLE D'UTILISATION:")
    print("""
    import pandas as pd
    
    # Simuler des données
    data = {
        'open': [1.1000, 1.1010, 1.1020, 1.1015, 1.1005] * 50,
        'high': [1.1010, 1.1025, 1.1030, 1.1020, 1.1015] * 50,
        'low': [1.0995, 1.1005, 1.1015, 1.1000, 1.0995] * 50,
        'close': [1.1005, 1.1020, 1.1025, 1.1010, 1.1000] * 50
    }
    
    df = pd.DataFrame(data)
    df.index = pd.date_range(start='2024-01-01', periods=250, freq='1min')
    
    # Obtenir le signal avec message formaté
    signal, message = get_signal_with_metadata_v2(
        df, 
        signal_count=0,
        total_signals=6,
        pairs_analyzed=5,
        batches=1
    )
    
    if signal:
        print(f"✅ Signal détecté!")
        print(f"   Direction: {signal['direction']}")
        print(f"   Score: {signal['score']:.0f}/200")
        print(f"   Confiance: {signal['confidence']}%")
        print(f"   Qualité: {signal['quality']}")
        print("\n📨 Message formaté à envoyer:")
        print(message)
    else:
        print("❌ Aucun signal détecté")
    """)
