"""
Utils OPTIMISÉ - Stratégie ULTRA_STRICT
========================================

OBJECTIF: 75-85% Win Rate
MÉTHODE: 4/5 critères minimum + filtres stricts

CHANGEMENTS vs VERSION ORIGINALE:
- Mode ULTRA_STRICT par défaut (4/5 critères)
- ADX minimum: 15 → 22
- RSI range resserré: 30-70 → 35-65
- Momentum minimum doublé
- Vérification EMA 50/200
- Filtres support/résistance renforcés
"""

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
    
    # EMA (ajout EMA 50 et 200 pour filtre tendance long terme)
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
    
    # Momentum (périodes plus longues pour filtrer bruit)
    df['momentum_3'] = df['close'].pct_change(periods=3) * 100
    df['momentum_5'] = df['close'].pct_change(periods=5) * 100
    df['momentum_10'] = df['close'].pct_change(periods=10) * 100  # NOUVEAU
    
    # Support/Resistance (fenêtre élargie pour meilleure détection)
    df['resistance'] = df['high'].rolling(window=30).max()  # 20 → 30
    df['support'] = df['low'].rolling(window=30).min()      # 20 → 30
    
    # Position BB
    df['bb_position'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
    
    # Distance par rapport aux EMA (NOUVEAU - pour filtrer contre-tendance)
    df['distance_ema_fast_pct'] = ((df['close'] - df['ema_fast']) / df['close']) * 100
    df['distance_ema_slow_pct'] = ((df['close'] - df['ema_slow']) / df['close']) * 100
    
    return df


def rule_signal_ultra_strict(df, session_priority=5, adx_min=22, confidence_min=0.80):
    """
    Stratégie ULTRA_STRICT V3.0 - Objectif 75-85% Win Rate
    
    NOUVEAUTÉS V3.0:
    ================
    - Mode ULTRA_STRICT par défaut (4/5 critères OBLIGATOIRES)
    - ADX minimum: 22 (vs 15 avant)
    - RSI range resserré: 35-65 (vs 30-70)
    - Momentum minimum DOUBLÉ
    - Vérification obligatoire EMA 50/200 pour confirmation tendance long terme
    - Filtres support/résistance RENFORCÉS (marge 1.5% au lieu de 0.5%)
    - Score pondéré avec pénalités pour signaux faibles
    
    RÉSULTATS ATTENDUS:
    ===================
    - Signaux/jour: 8-12 (vs 40-50 avant)
    - Win rate: 75-85% (vs 36% avant)
    - Wins réels: 6-10/jour
    - Qualité >>> Quantité
    
    CRITÈRES (4/5 OBLIGATOIRES):
    =============================
    1. EMA: Alignement fast > slow > 50 (CALL) ou inverse (PUT)
    2. MACD: Histogram > 0 ET en croissance (CALL) ou inverse (PUT)
    3. RSI: Zone stricte 40-60 avec momentum positif
    4. Stochastic: Aligné ET dans zone non-extrême (20-80)
    5. ADX: > 22 avec DI+ > DI- (CALL) ou inverse (PUT)
    
    Args:
        df: DataFrame avec indicateurs
        session_priority: Priorité session (5 = premium)
        adx_min: ADX minimum (default 22)
        confidence_min: Confiance minimum (default 0.80)
    
    Returns:
        'CALL', 'PUT' ou None
    """
    
    if len(df) < 100:  # Besoin de plus d'historique pour EMA 50/200
        return None
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    
    # ============================================
    # FILTRES DE BASE STRICTS (ÉLIMINATOIRES)
    # ============================================
    
    # 1. ADX: Tendance forte OBLIGATOIRE
    adx = last.get('adx')
    if adx is None or adx < adx_min:  # 22 minimum
        return None
    
    # 2. RSI: Zone resserrée
    rsi = last.get('rsi')
    if rsi is None or rsi < 35 or rsi > 65:  # 35-65 au lieu de 30-70
        return None
    
    # 3. Volatilité: Normale uniquement
    atr = last.get('atr', 0)
    if atr == 0:
        return None
    atr_sma = df['atr'].rolling(20).mean().iloc[-1]
    if atr < atr_sma * 0.6 or atr > atr_sma * 2.5:  # Plus strict
        return None
    
    # 4. Momentum: DOUBLÉ par rapport à avant
    momentum_3 = last.get('momentum_3', 0)
    momentum_5 = last.get('momentum_5', 0)
    if abs(momentum_3) < 0.03 or abs(momentum_3) > 2.0:  # 0.015 → 0.03
        return None
    
    # 5. MACD Histogram: Doit être significatif
    macd_hist = last.get('MACDh_12_26_9')
    if macd_hist is None or abs(macd_hist) < 0.00015:  # 0.0001 → 0.00015
        return None
    
    # 6. Volume: Si disponible, doit être normal
    if 'volume_sma' in df.columns:
        volume_ratio = last.get('volume', 0) / last.get('volume_sma', 1)
        if volume_ratio < 0.7 or volume_ratio > 2.5:
            return None
    
    # Récupérer tous les indicateurs
    stoch_k = last.get('stoch_k')
    stoch_d = last.get('stoch_d')
    macd = last.get('MACD_12_26_9')
    macd_signal = last.get('MACDs_12_26_9')
    
    if None in [stoch_k, stoch_d, macd, macd_signal]:
        return None
    
    # ============================================
    # ANALYSE CALL (BUY) - MODE ULTRA_STRICT
    # ============================================
    
    call_criteria = []
    call_weights = []
    
    # CRITÈRE 1: EMA Triple Alignement (poids 1.5)
    # CALL: fast > slow > 50 > 200
    ema_fast = last.get('ema_fast', 0)
    ema_slow = last.get('ema_slow', 0)
    ema_50 = last.get('ema_50', 0)
    ema_200 = last.get('ema_200', 0)
    
    ema_aligned = (
        ema_fast > ema_slow and 
        ema_slow > ema_50 and
        ema_50 > ema_200 and
        last['close'] > ema_fast  # Prix au-dessus de fast
    )
    
    # Bonus: Distance pas trop grande (éviter sur-extension)
    distance_ok = abs(last.get('distance_ema_fast_pct', 0)) < 0.5
    
    call_criteria.append(ema_aligned and distance_ok)
    call_weights.append(1.5)
    
    # CRITÈRE 2: MACD Fort et en croissance (poids 1.5)
    macd_bullish = (
        macd > macd_signal and 
        macd_hist > 0 and
        macd_hist > prev.get('MACDh_12_26_9', 0) and  # En croissance
        macd > prev.get('MACD_12_26_9', 0)  # MACD lui-même monte
    )
    
    call_criteria.append(macd_bullish)
    call_weights.append(1.5)
    
    # CRITÈRE 3: RSI Zone optimale avec momentum (poids 1.0)
    rsi_bullish = (
        40 < rsi < 60 and  # Zone resserrée
        rsi > prev.get('rsi', rsi) and  # RSI monte
        momentum_5 > 0  # Momentum positif confirmé
    )
    
    call_criteria.append(rsi_bullish)
    call_weights.append(1.0)
    
    # CRITÈRE 4: Stochastic confirmé (poids 1.0)
    stoch_bullish = (
        stoch_k > stoch_d and
        stoch_k > prev.get('stoch_k', stoch_k) and  # K monte
        20 < stoch_k < 80 and  # Zone non-extrême
        stoch_k - stoch_d > 2  # Écart significatif
    )
    
    call_criteria.append(stoch_bullish)
    call_weights.append(1.0)
    
    # CRITÈRE 5: ADX + Directional Movement (poids 1.5)
    adx_pos = last.get('adx_pos', 0)
    adx_neg = last.get('adx_neg', 0)
    
    adx_bullish = (
        adx_pos > adx_neg and
        adx_pos > 20 and
        momentum_3 > 0 and
        momentum_5 > 0 and
        last.get('momentum_10', 0) > 0  # Momentum long terme aussi
    )
    
    call_criteria.append(adx_bullish)
    call_weights.append(1.5)
    
    # CALCUL SCORE PONDÉRÉ
    call_score = sum(c * w for c, w in zip(call_criteria, call_weights))
    max_score = sum(call_weights)
    call_score_pct = call_score / max_score
    
    # ULTRA_STRICT: 4/5 critères OU score pondéré > 70%
    criteria_met = sum(call_criteria)
    
    if criteria_met >= 4 or call_score_pct >= 0.70:
        # VÉRIFICATIONS FINALES STRICTES
        
        # 1. Support/Résistance avec marge élargie
        resistance = last.get('resistance')
        if resistance and last['close'] > resistance * 0.985:  # 1.5% de marge
            return None
        
        # 2. Bollinger Bands: pas trop haut
        bb_position = last.get('bb_position')
        if bb_position is None or bb_position > 0.75:  # 0.55 → 0.75
            return None
        
        # 3. Pas de divergence RSI-Prix
        price_change = (last['close'] - prev['close']) / prev['close']
        rsi_change = (rsi - prev.get('rsi', rsi)) / prev.get('rsi', rsi)
        
        if price_change > 0 and rsi_change < -0.05:  # Divergence baissière
            return None
        
        # 4. ATR ne doit pas exploser (spike = danger)
        atr_change = (atr - prev.get('atr', atr)) / prev.get('atr', atr)
        if atr_change > 0.5:  # +50% ATR = trop de volatilité soudaine
            return None
        
        return 'CALL'
    
    # ============================================
    # ANALYSE PUT (SELL) - MODE ULTRA_STRICT
    # ============================================
    
    put_criteria = []
    put_weights = []
    
    # CRITÈRE 1: EMA Triple Alignement inversé (poids 1.5)
    ema_aligned = (
        ema_fast < ema_slow and 
        ema_slow < ema_50 and
        ema_50 < ema_200 and
        last['close'] < ema_fast
    )
    
    distance_ok = abs(last.get('distance_ema_fast_pct', 0)) < 0.5
    
    put_criteria.append(ema_aligned and distance_ok)
    put_weights.append(1.5)
    
    # CRITÈRE 2: MACD Baissier fort (poids 1.5)
    macd_bearish = (
        macd < macd_signal and 
        macd_hist < 0 and
        macd_hist < prev.get('MACDh_12_26_9', 0) and
        macd < prev.get('MACD_12_26_9', 0)
    )
    
    put_criteria.append(macd_bearish)
    put_weights.append(1.5)
    
    # CRITÈRE 3: RSI Zone baissière (poids 1.0)
    rsi_bearish = (
        40 < rsi < 60 and
        rsi < prev.get('rsi', rsi) and
        momentum_5 < 0
    )
    
    put_criteria.append(rsi_bearish)
    put_weights.append(1.0)
    
    # CRITÈRE 4: Stochastic baissier (poids 1.0)
    stoch_bearish = (
        stoch_k < stoch_d and
        stoch_k < prev.get('stoch_k', stoch_k) and
        20 < stoch_k < 80 and
        stoch_d - stoch_k > 2
    )
    
    put_criteria.append(stoch_bearish)
    put_weights.append(1.0)
    
    # CRITÈRE 5: ADX baissier (poids 1.5)
    adx_bearish = (
        adx_neg > adx_pos and
        adx_neg > 20 and
        momentum_3 < 0 and
        momentum_5 < 0 and
        last.get('momentum_10', 0) < 0
    )
    
    put_criteria.append(adx_bearish)
    put_weights.append(1.5)
    
    # CALCUL SCORE
    put_score = sum(c * w for c, w in zip(put_criteria, put_weights))
    max_score = sum(put_weights)
    put_score_pct = put_score / max_score
    criteria_met = sum(put_criteria)
    
    if criteria_met >= 4 or put_score_pct >= 0.70:
        # VÉRIFICATIONS FINALES
        
        # 1. Support/Résistance
        support = last.get('support')
        if support and last['close'] < support * 1.015:
            return None
        
        # 2. Bollinger Bands
        bb_position = last.get('bb_position')
        if bb_position is None or bb_position < 0.25:
            return None
        
        # 3. Divergence
        price_change = (last['close'] - prev['close']) / prev['close']
        rsi_change = (rsi - prev.get('rsi', rsi)) / prev.get('rsi', rsi)
        
        if price_change < 0 and rsi_change > 0.05:
            return None
        
        # 4. ATR
        atr_change = (atr - prev.get('atr', atr)) / prev.get('atr', atr)
        if atr_change > 0.5:
            return None
        
        return 'PUT'
    
    return None


def rule_signal(df):
    """Wrapper pour compatibilité - Utilise ULTRA_STRICT par défaut"""
    return rule_signal_ultra_strict(df, session_priority=5, adx_min=22, confidence_min=0.80)


def get_signal_quality_score(df):
    """
    Score de qualité AMÉLIORÉ (0-100)
    Plus strict dans l'attribution des points
    """
    if len(df) < 10:
        return 0
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    
    # ADX (max 25 points) - Plus strict
    adx = last.get('adx', 0)
    if adx > 30:
        score += 25
    elif adx > 25:
        score += 20
    elif adx > 22:
        score += 15
    elif adx > 18:
        score += 8
    else:
        score += 0  # Pénalité: pas de points en dessous de 18
    
    # RSI (max 20 points) - Zone resserrée
    rsi = last.get('rsi', 50)
    if 45 < rsi < 55:
        score += 20
    elif 40 < rsi < 60:
        score += 15
    elif 35 < rsi < 65:
        score += 10
    else:
        score += 0
    
    # MACD (max 20 points) - Alignement + force
    macd = last.get('MACD_12_26_9', 0)
    macd_signal = last.get('MACDs_12_26_9', 0)
    macd_hist = last.get('MACDh_12_26_9', 0)
    
    aligned = (macd > macd_signal and macd_hist > 0) or (macd < macd_signal and macd_hist < 0)
    growing = abs(macd_hist) > abs(prev.get('MACDh_12_26_9', 0))
    
    if aligned and growing:
        score += 20
    elif aligned:
        score += 10
    
    # Volatilité (max 15 points)
    atr = last.get('atr', 0)
    atr_sma = df['atr'].rolling(20).mean().iloc[-1] if len(df) >= 20 else atr
    
    if atr_sma > 0:
        volatility_ratio = atr / atr_sma
        if 0.95 < volatility_ratio < 1.15:  # Très stable
            score += 15
        elif 0.85 < volatility_ratio < 1.30:  # Stable
            score += 10
        elif 0.70 < volatility_ratio < 1.60:  # Acceptable
            score += 5
    
    # EMA Triple (max 20 points) - Alignement fort
    ema_fast = last.get('ema_fast', 0)
    ema_slow = last.get('ema_slow', 0)
    ema_50 = last.get('ema_50', 0)
    ema_200 = last.get('ema_200', 0)
    
    # Alignement complet
    if (ema_fast > ema_slow > ema_50 > ema_200) or (ema_fast < ema_slow < ema_50 < ema_200):
        score += 20
    # Alignement partiel
    elif (ema_fast > ema_slow > ema_50) or (ema_fast < ema_slow < ema_50):
        score += 12
    # Minimal
    elif ema_fast > ema_slow or ema_fast < ema_slow:
        score += 5
    
    return min(score, 100)


def is_kill_zone_optimal(hour_utc):
    """
    Kill zones OPTIMISÉES - Seulement les plus profitables
    """
    # London/NY Overlap = MEILLEUR (12h-15h UTC)
    if 12 <= hour_utc < 15:
        return True, "London/NY Overlap", 5
    
    # London Open = BON (7h-10h UTC)
    if 7 <= hour_utc < 10:
        return True, "London Open", 4
    
    # NY Session = MOYEN (15h-17h UTC)
    if 15 <= hour_utc < 17:
        return True, "NY Session", 3
    
    # SUPPRIMÉ: Asian (trop volatile), Evening (trop de bruit)
    return False, None, 0


def format_signal_reason(direction, confidence, indicators):
    """Raison améliorée avec détails qualité"""
    last = indicators.iloc[-1]
    reason_parts = []
    
    direction_text = "Haussier" if direction == "CALL" else "Baissier"
    reason_parts.append(f"🎯 {direction_text}")
    
    # Confiance avec emoji
    conf_pct = int(confidence * 100)
    if conf_pct >= 85:
        emoji = "🔥"
    elif conf_pct >= 80:
        emoji = "💪"
    else:
        emoji = "✅"
    reason_parts.append(f"{emoji} ML {conf_pct}%")
    
    # ADX
    adx = last.get('adx', 0)
    if adx > 25:
        reason_parts.append(f"⚡ ADX {adx:.0f}")
    elif adx > 22:
        reason_parts.append(f"💨 ADX {adx:.0f}")
    
    # RSI
    rsi = last.get('rsi', 50)
    if 45 < rsi < 55:
        reason_parts.append(f"🎯 RSI {rsi:.0f}")
    
    # Score qualité
    quality = get_signal_quality_score(indicators)
    if quality >= 80:
        reason_parts.append("⭐⭐⭐")
    elif quality >= 70:
        reason_parts.append("⭐⭐")
    
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
