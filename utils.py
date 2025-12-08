import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange


def round_to_m5_candle(dt):
    """
    Arrondit un datetime à la bougie M5 la plus proche
    Exemple: 14:23:47 -> 14:20:00
            14:27:12 -> 14:25:00
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    # Arrondir les minutes à un multiple de 5
    minute = (dt.minute // 5) * 5
    
    # Remettre secondes et microsecondes à 0
    return dt.replace(minute=minute, second=0, microsecond=0)


def get_next_m5_candle(dt):
    """
    Retourne le début de la PROCHAINE bougie M5
    Exemple: 14:23:47 -> 14:25:00
            14:20:00 -> 14:25:00
    """
    current_candle = round_to_m5_candle(dt)
    return current_candle + timedelta(minutes=5)


def get_m5_candle_range(dt):
    """
    Retourne le début et la fin de la bougie M5 contenant ce datetime
    Exemple: 14:23:47 -> (14:20:00, 14:25:00)
    """
    start = round_to_m5_candle(dt)
    end = start + timedelta(minutes=5)
    return start, end


def compute_indicators(df, ema_fast=8, ema_slow=21, rsi_len=14, bb_len=20):
    """
    Calcule des indicateurs techniques avancés pour M5
    Optimisé pour timeframe 5 minutes
    """
    df = df.copy()
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    
    # EMA (Exponential Moving Average)
    df['ema_fast'] = EMAIndicator(close=df['close'], window=ema_fast).ema_indicator()
    df['ema_slow'] = EMAIndicator(close=df['close'], window=ema_slow).ema_indicator()
    df['ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
    df['ema_200'] = EMAIndicator(close=df['close'], window=200).ema_indicator()
    
    # MACD (Moving Average Convergence Divergence)
    macd = MACD(close=df['close'])
    df['MACD_12_26_9'] = macd.macd()
    df['MACDs_12_26_9'] = macd.macd_signal()
    df['MACDh_12_26_9'] = macd.macd_diff()
    
    # RSI (Relative Strength Index)
    df['rsi'] = RSIIndicator(close=df['close'], window=rsi_len).rsi()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['close'], window=bb_len, window_dev=2)
    df['BBL_20_2.0'] = bb.bollinger_lband()
    df['BBM_20_2.0'] = bb.bollinger_mavg()
    df['BBU_20_2.0'] = bb.bollinger_hband()
    df['BB_width'] = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0']
    
    # ATR (Average True Range) - Adapté pour M5
    df['atr'] = AverageTrueRange(
        high=df['high'], 
        low=df['low'], 
        close=df['close'], 
        window=14
    ).average_true_range()
    
    # ADX (Average Directional Index) - Force de la tendance
    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()
    
    # Stochastic Oscillator
    stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Volume profile (si disponible)
    if 'volume' in df.columns:
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
    
    # Momentum sur 3 bougies M5 (15 minutes)
    df['momentum_3'] = df['close'].pct_change(periods=3) * 100
    
    # Momentum sur 5 bougies (25 minutes) - Plus stable
    df['momentum_5'] = df['close'].pct_change(periods=5) * 100
    
    # Support/Resistance sur 20 bougies M5 (100 minutes)
    df['resistance'] = df['high'].rolling(window=20).max()
    df['support'] = df['low'].rolling(window=20).min()
    
    # Distance par rapport aux bandes de Bollinger
    df['bb_position'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
    
    return df


def rule_signal_ultra_strict(df):
    """
    Stratégie ULTRA-CONSERVATRICE pour M5 - Win rate cible 70-80%
    
    VERSION 3.0 - CORRECTIONS APRÈS 0% WIN RATE:
    
    1. ADX minimum augmenté: 12 → 15 (tendance plus forte)
    2. Seuil de décision augmenté: 2/5 → 3/5 critères (60%)
    3. RSI resserré: 15-85 → 25-75 (zone plus sûre)
    4. Momentum validé sur 2 périodes (3 et 5 bougies)
    5. MACD histogram doit être significatif (>0.0001)
    6. Vérification position Bollinger Bands
    7. Confirmation directionnelle EMA 50/200
    8. Rejet si prix dans zone neutre BB (40-60%)
    
    Cette version vise 3-5 signaux/jour avec 70-80% win rate
    """
    
    if len(df) < 50:
        return None
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Vérifications de base
    rsi = last.get('rsi')
    adx = last.get('adx')
    stoch_k = last.get('stoch_k')
    stoch_d = last.get('stoch_d')
    macd = last.get('MACD_12_26_9')
    macd_signal = last.get('MACDs_12_26_9')
    macd_hist = last.get('MACDh_12_26_9')
    bb_position = last.get('bb_position')
    
    if None in [rsi, adx, stoch_k, stoch_d, macd, macd_signal, macd_hist]:
        return None
    
    # ============================================
    # FILTRES DE BASE (MUST-HAVE) - RENFORCÉS
    # ============================================
    
    # CRITERE 1: TENDANCE FORTE REQUISE
    # ADX > 15 (au lieu de 12) = tendance claire nécessaire
    if adx < 15:
        return None
    
    # CRITERE 2: VOLATILITÉ MODÉRÉE
    atr = last.get('atr', 0)
    atr_sma = df['atr'].rolling(20).mean().iloc[-1]
    # Volatilité entre 0.5x et 2.5x (ni trop faible, ni trop forte)
    if atr < atr_sma * 0.5 or atr > atr_sma * 2.5:
        return None
    
    # CRITERE 3: RSI ZONE RESSERRÉE (évite extrêmes ET zone neutre)
    # Zone: 25-75 (plus conservateur que 15-85)
    if rsi < 25 or rsi > 75:
        return None
    
    # CRITERE 4: MOMENTUM DOUBLE VALIDATION
    momentum_3 = last.get('momentum_3', 0)
    momentum_5 = last.get('momentum_5', 0)
    
    # Momentum sur 3 bougies: 0.02-2.0%
    if abs(momentum_3) < 0.02 or abs(momentum_3) > 2.0:
        return None
    
    # Momentum sur 5 bougies: 0.05-3.0%
    if abs(momentum_5) < 0.05 or abs(momentum_5) > 3.0:
        return None
    
    # Les deux momentum doivent être dans la même direction
    if (momentum_3 > 0 and momentum_5 < 0) or (momentum_3 < 0 and momentum_5 > 0):
        return None
    
    # CRITERE 5: MACD HISTOGRAM SIGNIFICATIF
    # Rejeter si signal trop faible
    if abs(macd_hist) < 0.0001:
        return None
    
    # CRITERE 6: POSITION BOLLINGER BANDS
    # Rejeter si dans zone neutre (40-60%)
    if bb_position and 0.4 < bb_position < 0.6:
        return None
    
    # ============================================
    # ANALYSE CALL (BUY) - 3/5 CRITERES (60%)
    # ============================================
    
    call_signals = []
    
    # 1. Direction EMA principale + confirmation EMA 50
    ema_bullish_main = last['ema_fast'] > last['ema_slow']
    ema_50_confirm = last['ema_slow'] > last['ema_50'] if 'ema_50' in last.index else True
    call_signals.append(ema_bullish_main and ema_50_confirm)
    
    # 2. MACD haussier + histogram en croissance
    macd_bullish = macd > macd_signal and macd_hist > 0
    macd_growing = macd_hist > prev.get('MACDh_12_26_9', 0) if 'MACDh_12_26_9' in prev.index else True
    call_signals.append(macd_bullish and macd_growing)
    
    # 3. RSI dans zone haussière optimale (45-70)
    rsi_bullish = 45 < rsi < 70
    call_signals.append(rsi_bullish)
    
    # 4. Stochastic confirme + zone valide
    stoch_bullish = stoch_k > stoch_d and 20 < stoch_k < 85
    stoch_momentum = stoch_k > prev.get('stoch_k', 0) if 'stoch_k' in prev.index else True
    call_signals.append(stoch_bullish and stoch_momentum)
    
    # 5. ADX tendance haussière + momentum positifs alignés
    adx_bullish = (last['adx_pos'] > last['adx_neg']) and momentum_3 > 0 and momentum_5 > 0
    call_signals.append(adx_bullish)
    
    # DECISION CALL: 3/5 critères (60%) REQUIS
    call_score = sum(call_signals)
    if call_score >= 3:
        # Vérification finale: pas trop proche de la résistance
        resistance = last.get('resistance')
        if resistance and last['close'] > resistance * 0.995:
            # Trop proche de la résistance (< 0.5%), rejeter
            return None
        
        # Vérification BB: doit être en zone haussière (> 60%)
        if bb_position and bb_position < 0.55:
            return None
        
        return 'CALL'
    
    # ============================================
    # ANALYSE PUT (SELL) - 3/5 CRITERES (60%)
    # ============================================
    
    put_signals = []
    
    # 1. Direction EMA principale + confirmation EMA 50
    ema_bearish_main = last['ema_fast'] < last['ema_slow']
    ema_50_confirm = last['ema_slow'] < last['ema_50'] if 'ema_50' in last.index else True
    put_signals.append(ema_bearish_main and ema_50_confirm)
    
    # 2. MACD baissier + histogram en décroissance
    macd_bearish = macd < macd_signal and macd_hist < 0
    macd_falling = macd_hist < prev.get('MACDh_12_26_9', 0) if 'MACDh_12_26_9' in prev.index else True
    put_signals.append(macd_bearish and macd_falling)
    
    # 3. RSI dans zone baissière optimale (30-55)
    rsi_bearish = 30 < rsi < 55
    put_signals.append(rsi_bearish)
    
    # 4. Stochastic confirme + zone valide
    stoch_bearish = stoch_k < stoch_d and 15 < stoch_k < 80
    stoch_momentum = stoch_k < prev.get('stoch_k', 0) if 'stoch_k' in prev.index else True
    put_signals.append(stoch_bearish and stoch_momentum)
    
    # 5. ADX tendance baissière + momentum négatifs alignés
    adx_bearish = (last['adx_neg'] > last['adx_pos']) and momentum_3 < 0 and momentum_5 < 0
    put_signals.append(adx_bearish)
    
    # DECISION PUT: 3/5 critères (60%) REQUIS
    put_score = sum(put_signals)
    if put_score >= 3:
        # Vérification finale: pas trop proche du support
        support = last.get('support')
        if support and last['close'] < support * 1.005:
            # Trop proche du support (< 0.5%), rejeter
            return None
        
        # Vérification BB: doit être en zone baissière (< 40%)
        if bb_position and bb_position > 0.45:
            return None
        
        return 'PUT'
    
    # Si moins de 3/5 critères, NE PAS TRADER
    return None


def rule_signal(df):
    """
    Stratégie standard (fallback)
    Utilise les mêmes critères que rule_signal_ultra_strict
    """
    return rule_signal_ultra_strict(df)


def get_signal_quality_score(df):
    """
    Calcule un score de qualité du signal (0-100)
    Version améliorée avec plus de critères
    """
    if len(df) < 10:
        return 0
    
    last = df.iloc[-1]
    score = 0
    
    # ADX score (max 20 points)
    adx = last.get('adx', 0)
    if adx > 30:
        score += 20
    elif adx > 25:
        score += 15
    elif adx > 20:
        score += 10
    elif adx > 15:
        score += 5
    
    # RSI position (max 20 points)
    rsi = last.get('rsi', 50)
    if 48 < rsi < 52:
        score += 20  # Zone neutre parfaite
    elif 45 < rsi < 55:
        score += 15
    elif 40 < rsi < 60:
        score += 10
    elif 35 < rsi < 65:
        score += 5
    
    # MACD alignement + force (max 25 points)
    macd = last.get('MACD_12_26_9', 0)
    macd_signal = last.get('MACDs_12_26_9', 0)
    macd_hist = last.get('MACDh_12_26_9', 0)
    
    if (macd > macd_signal and macd_hist > 0) or (macd < macd_signal and macd_hist < 0):
        score += 15
        # Bonus si histogram fort
        if abs(macd_hist) > 0.0005:
            score += 10
    
    # Volatilité optimale (max 20 points)
    atr = last.get('atr', 0)
    atr_sma = df['atr'].rolling(20).mean().iloc[-1] if len(df) >= 20 else atr
    if atr_sma > 0:
        volatility_ratio = atr / atr_sma
        if 0.9 < volatility_ratio < 1.3:
            score += 20  # Volatilité idéale
        elif 0.7 < volatility_ratio < 1.7:
            score += 10
    
    # EMA alignment (max 15 points)
    ema_fast = last.get('ema_fast', 0)
    ema_slow = last.get('ema_slow', 0)
    ema_50 = last.get('ema_50', 0)
    
    # Tendance claire avec EMA 50 confirmée
    if ema_fast > ema_slow > ema_50:
        score += 15  # Tendance haussière parfaite
    elif ema_fast < ema_slow < ema_50:
        score += 15  # Tendance baissière parfaite
    elif ema_fast > ema_slow or ema_fast < ema_slow:
        score += 8  # Tendance partielle
    
    return min(score, 100)


def is_kill_zone_optimal(hour_utc):
    """
    Détermine si l'heure UTC est dans une zone optimale
    Retourne (is_optimal, zone_name, priority)
    """
    # London/NY Overlap (12h-14h UTC) - Meilleure zone
    if 12 <= hour_utc < 14:
        return True, "London/NY Overlap", 5
    
    # London Open (07h-10h UTC)
    if 7 <= hour_utc < 10:
        return True, "London Open", 3
    
    # NY Open (13h-16h UTC)
    if 13 <= hour_utc < 16:
        return True, "NY Open", 3
    
    # Asian Session (00h-03h UTC)
    if 0 <= hour_utc < 3:
        return True, "Asian Session", 1
    
    return False, None, 0


def format_signal_reason(direction, confidence, indicators):
    """
    Formate une raison lisible pour le signal
    """
    last = indicators.iloc[-1]
    
    reason_parts = []
    
    # Direction
    direction_text = "Haussier" if direction == "CALL" else "Baissier"
    reason_parts.append(f"{direction_text}")
    
    # Confiance ML
    reason_parts.append(f"ML {int(confidence*100)}%")
    
    # ADX
    adx = last.get('adx', 0)
    if adx > 25:
        reason_parts.append(f"ADX fort ({adx:.0f})")
    elif adx > 20:
        reason_parts.append(f"ADX moyen ({adx:.0f})")
    elif adx > 15:
        reason_parts.append(f"ADX léger ({adx:.0f})")
    
    # RSI
    rsi = last.get('rsi', 50)
    if direction == "CALL" and 45 < rsi < 65:
        reason_parts.append(f"RSI optimal ({rsi:.0f})")
    elif direction == "PUT" and 35 < rsi < 55:
        reason_parts.append(f"RSI optimal ({rsi:.0f})")
    
    return " | ".join(reason_parts)


def validate_m5_timing(entry_time):
    """
    Valide que l'heure d'entrée est bien alignée sur une bougie M5
    Retourne (is_valid, corrected_time)
    """
    if isinstance(entry_time, str):
        entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
    
    # Arrondir à la bougie M5
    corrected = round_to_m5_candle(entry_time)
    
    # Vérifier si on a dû corriger
    is_valid = (entry_time == corrected)
    
    return is_valid, corrected


def get_m5_entry_exit_times(signal_time):
    """
    À partir d'un signal_time, calcule les temps d'entrée et sortie M5 arrondis
    
    Returns:
        entry_time: Début de la bougie M5 d'entrée
        exit_time: Fin de la bougie M5 d'entrée (= début bougie suivante)
    """
    if isinstance(signal_time, str):
        signal_time = datetime.fromisoformat(signal_time.replace('Z', '+00:00'))
    
    # Entry = prochaine bougie M5 après le signal
    entry_time = get_next_m5_candle(signal_time)
    
    # Exit = fin de la bougie d'entrée
    exit_time = entry_time + timedelta(minutes=5)
    
    return entry_time, exit_time
