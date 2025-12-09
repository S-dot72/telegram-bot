import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange


def round_to_m5_candle(dt):
    """Arrondit un datetime à la bougie M5 la plus proche"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    minute = (dt.minute // 5) * 5
    return dt.replace(minute=minute, second=0, microsecond=0)


def get_next_m5_candle(dt):
    """Retourne le début de la PROCHAINE bougie M5"""
    current_candle = round_to_m5_candle(dt)
    return current_candle + timedelta(minutes=5)


def get_m5_candle_range(dt):
    """Retourne le début et la fin de la bougie M5"""
    start = round_to_m5_candle(dt)
    end = start + timedelta(minutes=5)
    return start, end


def compute_indicators(df, ema_fast=8, ema_slow=21, rsi_len=14, bb_len=20):
    """Calcule des indicateurs techniques pour M5"""
    df = df.copy()
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    
    # EMA
    df['ema_fast'] = EMAIndicator(close=df['close'], window=ema_fast).ema_indicator()
    df['ema_slow'] = EMAIndicator(close=df['close'], window=ema_slow).ema_indicator()
    df['ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
    df['ema_200'] = EMAIndicator(close=df['close'], window=200).ema_indicator()
    
    # MACD
    macd = MACD(close=df['close'])
    df['MACD_12_26_9'] = macd.macd()
    df['MACDs_12_26_9'] = macd.macd_signal()
    df['MACDh_12_26_9'] = macd.macd_diff()
    
    # RSI
    df['rsi'] = RSIIndicator(close=df['close'], window=rsi_len).rsi()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['close'], window=bb_len, window_dev=2)
    df['BBL_20_2.0'] = bb.bollinger_lband()
    df['BBM_20_2.0'] = bb.bollinger_mavg()
    df['BBU_20_2.0'] = bb.bollinger_hband()
    df['BB_width'] = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0']
    
    # ATR
    df['atr'] = AverageTrueRange(
        high=df['high'], low=df['low'], close=df['close'], window=14
    ).average_true_range()
    
    # ADX
    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()
    
    # Stochastic
    stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Volume
    if 'volume' in df.columns:
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
    
    # Momentum
    df['momentum_3'] = df['close'].pct_change(periods=3) * 100
    df['momentum_5'] = df['close'].pct_change(periods=5) * 100
    
    # Support/Resistance
    df['resistance'] = df['high'].rolling(window=20).max()
    df['support'] = df['low'].rolling(window=20).min()
    
    # Position BB
    df['bb_position'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
    
    return df


def rule_signal_ultra_strict(df, session_priority=3):
    """
    Stratégie HYBRIDE V2.5 - Adaptatif selon priorité session
    
    VERSION 2.5 - MEILLEUR COMPROMIS QUANTITÉ/QUALITÉ:
    
    ADAPTABILITÉ PAR SESSION:
    - Session priorité 5 (London/NY Overlap) : Mode STRICT (3/5 critères)
    - Session priorité 3-4 (London, NY) : Mode MODÉRÉ (2.5/5 critères)
    - Session priorité 1-2 (Asian, Evening) : Mode SOUPLE (2/5 critères)
    
    RÉSULTATS ATTENDUS:
    - Signaux/jour: 8-12
    - Win rate: 65-75%
    - Wins réels: 5-9/jour
    
    MEILLEUR COMPROMIS entre V2 (0% WR) et V3 (3-5 signaux/jour)
    """
    
    if len(df) < 50:
        return None
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Déterminer mode selon priorité session
    if session_priority >= 5:
        mode = "STRICT"
        required_score = 3  # 3/5 critères
        adx_min = 15
        rsi_range = (30, 70)
        momentum_min = 0.02
    elif session_priority >= 3:
        mode = "MODÉRÉ"
        required_score = 2.5  # 2.5/5 critères (arrondi dynamique)
        adx_min = 13
        rsi_range = (25, 75)
        momentum_min = 0.015
    else:
        mode = "SOUPLE"
        required_score = 2  # 2/5 critères
        adx_min = 12
        rsi_range = (20, 80)
        momentum_min = 0.01
    
    # Vérifications de base
    rsi = last.get('rsi')
    adx = last.get('adx')
    stoch_k = last.get('stoch_k')
    stoch_d = last.get('stoch_d')
    macd = last.get('MACD_12_26_9')
    macd_signal = last.get('MACDs_12_26_9')
    macd_hist = last.get('MACDh_12_26_9')
    
    if None in [rsi, adx, stoch_k, stoch_d, macd, macd_signal, macd_hist]:
        return None
    
    # ============================================
    # FILTRES DE BASE (ADAPTATIFS)
    # ============================================
    
    # CRITERE 1: TENDANCE (adaptatif)
    if adx < adx_min:
        return None
    
    # CRITERE 2: VOLATILITÉ
    atr = last.get('atr', 0)
    atr_sma = df['atr'].rolling(20).mean().iloc[-1]
    if atr < atr_sma * 0.5 or atr > atr_sma * 2.8:
        return None
    
    # CRITERE 3: RSI (adaptatif)
    if rsi < rsi_range[0] or rsi > rsi_range[1]:
        return None
    
    # CRITERE 4: MOMENTUM (adaptatif)
    momentum_3 = last.get('momentum_3', 0)
    if abs(momentum_3) < momentum_min or abs(momentum_3) > 2.5:
        return None
    
    # CRITERE 5: MACD HISTOGRAM (strict en mode STRICT uniquement)
    if mode == "STRICT" and abs(macd_hist) < 0.0001:
        return None
    
    # ============================================
    # ANALYSE CALL (BUY)
    # ============================================
    
    call_signals = []
    call_weights = []  # Poids pour mode MODÉRÉ
    
    # 1. Direction EMA (poids 1.0)
    ema_bullish = last['ema_fast'] > last['ema_slow']
    if mode == "STRICT":
        ema_50_confirm = last['ema_slow'] > last.get('ema_50', 0)
        call_signals.append(ema_bullish and ema_50_confirm)
    else:
        call_signals.append(ema_bullish)
    call_weights.append(1.0)
    
    # 2. MACD haussier (poids 1.0)
    macd_bullish = macd > macd_signal and macd_hist > 0
    if mode == "STRICT":
        macd_growing = macd_hist > prev.get('MACDh_12_26_9', 0)
        call_signals.append(macd_bullish and macd_growing)
    else:
        call_signals.append(macd_bullish)
    call_weights.append(1.0)
    
    # 3. RSI zone haussière (poids 0.8)
    if mode == "STRICT":
        rsi_bullish = 45 < rsi < 70
    elif mode == "MODÉRÉ":
        rsi_bullish = 40 < rsi < 75
    else:
        rsi_bullish = 35 < rsi < 80
    call_signals.append(rsi_bullish)
    call_weights.append(0.8)
    
    # 4. Stochastic (poids 0.7)
    stoch_bullish = stoch_k > stoch_d and 15 < stoch_k < 90
    call_signals.append(stoch_bullish)
    call_weights.append(0.7)
    
    # 5. ADX + momentum (poids 1.0)
    momentum_5 = last.get('momentum_5', momentum_3)
    if mode == "STRICT":
        adx_bullish = (last['adx_pos'] > last['adx_neg']) and momentum_3 > 0 and momentum_5 > 0
    else:
        adx_bullish = (last['adx_pos'] > last['adx_neg']) or momentum_3 > 0
    call_signals.append(adx_bullish)
    call_weights.append(1.0)
    
    # DECISION CALL
    if mode == "MODÉRÉ":
        # Score pondéré
        call_score = sum(s * w for s, w in zip(call_signals, call_weights))
        threshold = required_score
    else:
        # Score simple
        call_score = sum(call_signals)
        threshold = int(required_score)
    
    if call_score >= threshold:
        # Vérification finale (strict en mode STRICT)
        if mode == "STRICT":
            resistance = last.get('resistance')
            if resistance and last['close'] > resistance * 0.995:
                return None
            bb_position = last.get('bb_position')
            if bb_position and bb_position < 0.55:
                return None
        
        return 'CALL'
    
    # ============================================
    # ANALYSE PUT (SELL)
    # ============================================
    
    put_signals = []
    put_weights = []
    
    # 1. Direction EMA (poids 1.0)
    ema_bearish = last['ema_fast'] < last['ema_slow']
    if mode == "STRICT":
        ema_50_confirm = last['ema_slow'] < last.get('ema_50', 0)
        put_signals.append(ema_bearish and ema_50_confirm)
    else:
        put_signals.append(ema_bearish)
    put_weights.append(1.0)
    
    # 2. MACD baissier (poids 1.0)
    macd_bearish = macd < macd_signal and macd_hist < 0
    if mode == "STRICT":
        macd_falling = macd_hist < prev.get('MACDh_12_26_9', 0)
        put_signals.append(macd_bearish and macd_falling)
    else:
        put_signals.append(macd_bearish)
    put_weights.append(1.0)
    
    # 3. RSI zone baissière (poids 0.8)
    if mode == "STRICT":
        rsi_bearish = 30 < rsi < 55
    elif mode == "MODÉRÉ":
        rsi_bearish = 25 < rsi < 60
    else:
        rsi_bearish = 20 < rsi < 65
    put_signals.append(rsi_bearish)
    put_weights.append(0.8)
    
    # 4. Stochastic (poids 0.7)
    stoch_bearish = stoch_k < stoch_d and 10 < stoch_k < 85
    put_signals.append(stoch_bearish)
    put_weights.append(0.7)
    
    # 5. ADX + momentum (poids 1.0)
    if mode == "STRICT":
        adx_bearish = (last['adx_neg'] > last['adx_pos']) and momentum_3 < 0 and momentum_5 < 0
    else:
        adx_bearish = (last['adx_neg'] > last['adx_pos']) or momentum_3 < 0
    put_signals.append(adx_bearish)
    put_weights.append(1.0)
    
    # DECISION PUT
    if mode == "MODÉRÉ":
        put_score = sum(s * w for s, w in zip(put_signals, put_weights))
        threshold = required_score
    else:
        put_score = sum(put_signals)
        threshold = int(required_score)
    
    if put_score >= threshold:
        # Vérification finale (strict en mode STRICT)
        if mode == "STRICT":
            support = last.get('support')
            if support and last['close'] < support * 1.005:
                return None
            bb_position = last.get('bb_position')
            if bb_position and bb_position > 0.45:
                return None
        
        return 'PUT'
    
    return None


def rule_signal(df):
    """Stratégie standard - Utilise mode MODÉRÉ par défaut"""
    return rule_signal_ultra_strict(df, session_priority=3)


def get_signal_quality_score(df):
    """Calcule un score de qualité du signal (0-100)"""
    if len(df) < 10:
        return 0
    
    last = df.iloc[-1]
    score = 0
    
    # ADX
    adx = last.get('adx', 0)
    if adx > 30:
        score += 20
    elif adx > 25:
        score += 15
    elif adx > 20:
        score += 10
    elif adx > 15:
        score += 5
    
    # RSI
    rsi = last.get('rsi', 50)
    if 48 < rsi < 52:
        score += 20
    elif 45 < rsi < 55:
        score += 15
    elif 40 < rsi < 60:
        score += 10
    
    # MACD
    macd = last.get('MACD_12_26_9', 0)
    macd_signal = last.get('MACDs_12_26_9', 0)
    macd_hist = last.get('MACDh_12_26_9', 0)
    if (macd > macd_signal and macd_hist > 0) or (macd < macd_signal and macd_hist < 0):
        score += 20
    
    # Volatilité
    atr = last.get('atr', 0)
    atr_sma = df['atr'].rolling(20).mean().iloc[-1] if len(df) >= 20 else atr
    if atr_sma > 0:
        volatility_ratio = atr / atr_sma
        if 0.9 < volatility_ratio < 1.3:
            score += 20
        elif 0.7 < volatility_ratio < 1.7:
            score += 10
    
    # EMA
    ema_fast = last.get('ema_fast', 0)
    ema_slow = last.get('ema_slow', 0)
    ema_50 = last.get('ema_50', 0)
    if ema_fast > ema_slow > ema_50 or ema_fast < ema_slow < ema_50:
        score += 20
    elif ema_fast > ema_slow or ema_fast < ema_slow:
        score += 10
    
    return min(score, 100)


def is_kill_zone_optimal(hour_utc):
    """Détermine si l'heure UTC est dans une zone optimale"""
    if 12 <= hour_utc < 14:
        return True, "London/NY Overlap", 5
    if 7 <= hour_utc < 10:
        return True, "London Open", 3
    if 13 <= hour_utc < 16:
        return True, "NY Open", 3
    if 0 <= hour_utc < 3:
        return True, "Asian Session", 1
    return False, None, 0


def format_signal_reason(direction, confidence, indicators):
    """Formate une raison lisible pour le signal"""
    last = indicators.iloc[-1]
    reason_parts = []
    
    direction_text = "Haussier" if direction == "CALL" else "Baissier"
    reason_parts.append(f"{direction_text}")
    reason_parts.append(f"ML {int(confidence*100)}%")
    
    adx = last.get('adx', 0)
    if adx > 25:
        reason_parts.append(f"ADX fort ({adx:.0f})")
    elif adx > 20:
        reason_parts.append(f"ADX moyen ({adx:.0f})")
    
    rsi = last.get('rsi', 50)
    if direction == "CALL" and 45 < rsi < 65:
        reason_parts.append(f"RSI optimal ({rsi:.0f})")
    elif direction == "PUT" and 35 < rsi < 55:
        reason_parts.append(f"RSI optimal ({rsi:.0f})")
    
    return " | ".join(reason_parts)


def validate_m5_timing(entry_time):
    """Valide que l'heure d'entrée est bien alignée sur une bougie M5"""
    if isinstance(entry_time, str):
        entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
    corrected = round_to_m5_candle(entry_time)
    is_valid = (entry_time == corrected)
    return is_valid, corrected


def get_m5_entry_exit_times(signal_time):
    """Calcule les temps d'entrée et sortie M5 arrondis"""
    if isinstance(signal_time, str):
        signal_time = datetime.fromisoformat(signal_time.replace('Z', '+00:00'))
    entry_time = get_next_m5_candle(signal_time)
    exit_time = entry_time + timedelta(minutes=5)
    return entry_time, exit_time
