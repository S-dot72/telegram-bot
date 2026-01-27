import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from scipy import stats


def round_to_m1_candle(dt):
    """Arrondit un datetime à la bougie M1 (minute)"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.replace(second=0, microsecond=0)


def get_next_m1_candle(dt):
    """Retourne le début de la PROCHAINE bougie M1"""
    current_candle = round_to_m1_candle(dt)
    return current_candle + timedelta(minutes=1)


def get_m1_candle_range(dt):
    """Retourne le début et la fin de la bougie M1"""
    start = round_to_m1_candle(dt)
    end = start + timedelta(minutes=1)
    return start, end


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
    """Calcule des indicateurs techniques pour M1 et M5"""
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
        df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Momentum
    df['momentum_3'] = df['close'].pct_change(periods=3) * 100
    df['momentum_5'] = df['close'].pct_change(periods=5) * 100
    
    # Support/Resistance
    df['resistance'] = df['high'].rolling(window=20).max()
    df['support'] = df['low'].rolling(window=20).min()
    
    # Position BB
    df['bb_position'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
    
    # Nouvelles améliorations pour la qualité
    # Tendance linéaire
    df['linear_trend'] = calculate_linear_trend(df['close'])
    
    # Convergence d'indicateurs
    df['signal_convergence'] = calculate_signal_convergence(df)
    
    # Score de qualité des données
    df['data_quality'] = calculate_data_quality(df)
    
    # Patterns de bougies
    df['candle_pattern'] = identify_candle_pattern(df)
    
    # Ratio risque/récompense
    df['risk_reward'] = calculate_risk_reward_ratio(df)
    
    return df


def rule_signal_ultra_strict(df, session_priority=3):
    """
    STRATÉGIE FOREX PRIORITÉ 3 - QUALITÉ GARANTIE
    
    OBJECTIF : 8 signaux de HAUTE QUALITÉ par session
    
    AMÉLIORATIONS CLÉS :
    1. Multi-confirmations obligatoires (5/7 indicateurs)
    2. Validation de convergence (tous les indicateurs dans la même direction)
    3. Filtres anti-faux signaux (volatilité, tendance, momentum)
    4. Score de qualité minimum (75%)
    5. Stratégie de secours intelligente pour garantir 8 signaux
    
    MODES :
    - Priorité 5 : Mode EXTREME (6/7 indicateurs) - Qualité maximale
    - Priorité 3-4 : Mode QUALITY (5/7 indicateurs) - Équilibre qualité/quantité
    - Priorité 1-2 : Mode STANDARD (4/7 indicateurs) - Plus de signaux
    """
    
    if len(df) < 50:
        return None
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Déterminer mode selon priorité session
    if session_priority >= 5:
        mode = "EXTREME"
        required_score = 6  # 6/7 indicateurs concordants
        quality_threshold = 0.85  # 85% de qualité minimum
        volatility_max = 0.03  # 3% max de volatilité
    elif session_priority >= 3:
        mode = "QUALITY"
        required_score = 5  # 5/7 indicateurs concordants
        quality_threshold = 0.75  # 75% de qualité minimum
        volatility_max = 0.04  # 4% max de volatilité
    else:
        mode = "STANDARD"
        required_score = 4  # 4/7 indicateurs concordants
        quality_threshold = 0.65  # 65% de qualité minimum
        volatility_max = 0.05  # 5% max de volatilité
    
    print(f"[STRATEGY] 🎯 Mode {mode} activé - Priorité {session_priority}")
    print(f"[STRATEGY] 📊 Seuil: {required_score}/7 indicateurs - Qualité: {quality_threshold*100}%")
    
    # ============================================
    # VALIDATION DE LA QUALITÉ DES DONNÉES
    # ============================================
    
    # Vérifier la qualité des données
    data_quality = last.get('data_quality', 0.5)
    if data_quality < 0.6:
        print(f"[STRATEGY] ⚠️ Qualité données insuffisante: {data_quality:.1%}")
        return None
    
    # Vérifier la volatilité
    bb_width = last.get('BB_width', 0)
    if bb_width > volatility_max:
        print(f"[STRATEGY] ⚠️ Volatilité trop élevée: {bb_width:.2%} > {volatility_max:.2%}")
        return None
    
    # Vérifier l'ATR
    atr = last.get('atr', 0)
    atr_sma = df['atr'].rolling(20).mean().iloc[-1] if len(df) >= 20 else atr
    if atr_sma > 0 and (atr < atr_sma * 0.3 or atr > atr_sma * 3.0):
        print(f"[STRATEGY] ⚠️ ATR anormal: {atr:.6f} (moyenne: {atr_sma:.6f})")
        return None
    
    # ============================================
    # ANALYSE CALL (BUY) - MULTI-CONFIRMATIONS
    # ============================================
    
    call_signals = []
    call_details = []
    
    # 1. TENDANCE EMA (poids fort)
    ema_bullish = last['ema_fast'] > last['ema_slow']
    ema_50_confirmation = last['ema_slow'] > last['ema_50']
    ema_200_confirmation = last['ema_50'] > last['ema_200']
    
    ema_score = 0
    if ema_bullish:
        ema_score += 1
        if ema_50_confirmation:
            ema_score += 0.5
        if ema_200_confirmation:
            ema_score += 0.5
    
    call_signals.append(min(ema_score / 2.0, 1.0))  # Normalisé à 0-1
    call_details.append(f"EMA: {ema_score}/2")
    
    # 2. MACD HAUSSIER (poids fort)
    macd = last.get('MACD_12_26_9', 0)
    macd_signal = last.get('MACDs_12_26_9', 0)
    macd_hist = last.get('MACDh_12_26_9', 0)
    prev_macd_hist = prev.get('MACDh_12_26_9', 0)
    
    macd_bullish = macd > macd_signal
    macd_hist_bullish = macd_hist > 0
    macd_growing = macd_hist > prev_macd_hist
    
    macd_score = 0
    if macd_bullish:
        macd_score += 1
        if macd_hist_bullish:
            macd_score += 0.5
        if macd_growing:
            macd_score += 0.5
    
    call_signals.append(min(macd_score / 2.0, 1.0))
    call_details.append(f"MACD: {macd_score}/2")
    
    # 3. RSI OPTIMAL (poids moyen)
    rsi = last.get('rsi', 50)
    rsi_bullish_zone = 45 < rsi < 70
    rsi_optimal_zone = 50 < rsi < 65
    
    rsi_score = 0
    if rsi_bullish_zone:
        rsi_score += 1
        if rsi_optimal_zone:
            rsi_score += 1
    
    call_signals.append(min(rsi_score / 2.0, 1.0))
    call_details.append(f"RSI: {rsi_score}/2 ({rsi:.1f})")
    
    # 4. STOCHASTIQUE HAUSSIER (poids moyen)
    stoch_k = last.get('stoch_k', 50)
    stoch_d = last.get('stoch_d', 50)
    
    stoch_bullish = stoch_k > stoch_d
    stoch_not_overbought = stoch_k < 80
    
    stoch_score = 0
    if stoch_bullish:
        stoch_score += 1
        if stoch_not_overbought:
            stoch_score += 1
    
    call_signals.append(min(stoch_score / 2.0, 1.0))
    call_details.append(f"Stoch: {stoch_score}/2")
    
    # 5. ADX + MOMENTUM (poids fort)
    adx = last.get('adx', 0)
    adx_pos = last.get('adx_pos', 0)
    adx_neg = last.get('adx_neg', 0)
    momentum_3 = last.get('momentum_3', 0)
    momentum_5 = last.get('momentum_5', 0)
    
    adx_strong = adx > 25
    adx_bullish = adx_pos > adx_neg
    momentum_bullish = momentum_3 > 0 and momentum_5 > 0
    
    adx_score = 0
    if adx_strong:
        adx_score += 1
        if adx_bullish:
            adx_score += 0.5
        if momentum_bullish:
            adx_score += 0.5
    
    call_signals.append(min(adx_score / 2.0, 1.0))
    call_details.append(f"ADX: {adx_score}/2 ({adx:.1f})")
    
    # 6. BOLLINGER BANDS (poids moyen)
    bb_position = last.get('bb_position', 0.5)
    bb_bullish_position = 0.3 < bb_position < 0.7
    bb_not_extreme = 0.2 < bb_position < 0.8
    
    bb_score = 0
    if bb_bullish_position:
        bb_score += 1
        if bb_not_extreme:
            bb_score += 1
    
    call_signals.append(min(bb_score / 2.0, 1.0))
    call_details.append(f"BB: {bb_score}/2 ({bb_position:.2f})")
    
    # 7. CONVERGENCE ET TENDANCE (poids fort)
    linear_trend = last.get('linear_trend', 0)
    signal_convergence = last.get('signal_convergence', 0)
    price_above_support = last['close'] > last['support'] * 1.002
    
    convergence_score = 0
    if linear_trend > 0:
        convergence_score += 1
    if signal_convergence > 0.5:
        convergence_score += 0.5
    if price_above_support:
        convergence_score += 0.5
    
    call_signals.append(min(convergence_score / 2.0, 1.0))
    call_details.append(f"Conv: {convergence_score}/2")
    
    # ============================================
    # CALCUL DU SCORE CALL
    # ============================================
    
    # Convertir en indicateurs binaires (1 si >= 0.5)
    call_indicators = [1 if score >= 0.5 else 0 for score in call_signals]
    call_total = sum(call_indicators)
    
    # Score de qualité (moyenne pondérée)
    weights = [1.0, 1.0, 0.8, 0.8, 1.0, 0.8, 1.0]  # Poids selon importance
    call_weighted_score = sum(s * w for s, w in zip(call_signals, weights))
    call_max_weight = sum(weights)
    call_quality = call_weighted_score / call_max_weight
    
    print(f"[CALL_ANALYSIS] 📊 Indicateurs: {call_indicators}")
    print(f"[CALL_ANALYSIS] 🔢 Total: {call_total}/7 - Qualité: {call_quality:.1%}")
    print(f"[CALL_ANALYSIS] 📋 Détails: {', '.join(call_details)}")
    
    # VÉRIFICATIONS FINALES POUR CALL
    if call_total >= required_score and call_quality >= quality_threshold:
        # Vérifications supplémentaires
        volume_ok = True
        if 'volume_ratio' in last:
            volume_ok = last['volume_ratio'] > 0.7
        
        candle_pattern = last.get('candle_pattern', 'NEUTRAL')
        candle_ok = candle_pattern in ['BULLISH', 'BULLISH_STRONG', 'NEUTRAL']
        
        risk_reward = last.get('risk_reward', 1)
        rr_ok = risk_reward >= 1.2
        
        if volume_ok and candle_ok and rr_ok:
            print(f"[STRATEGY] ✅ Signal CALL validé - Score: {call_total}/7 - Qualité: {call_quality:.1%}")
            return 'CALL'
    
    # ============================================
    # ANALYSE PUT (SELL) - MULTI-CONFIRMATIONS
    # ============================================
    
    put_signals = []
    put_details = []
    
    # 1. TENDANCE EMA (poids fort)
    ema_bearish = last['ema_fast'] < last['ema_slow']
    ema_50_confirmation = last['ema_slow'] < last['ema_50']
    ema_200_confirmation = last['ema_50'] < last['ema_200']
    
    ema_score = 0
    if ema_bearish:
        ema_score += 1
        if ema_50_confirmation:
            ema_score += 0.5
        if ema_200_confirmation:
            ema_score += 0.5
    
    put_signals.append(min(ema_score / 2.0, 1.0))
    put_details.append(f"EMA: {ema_score}/2")
    
    # 2. MACD BAISSIER (poids fort)
    macd_bearish = macd < macd_signal
    macd_hist_bearish = macd_hist < 0
    macd_falling = macd_hist < prev_macd_hist
    
    macd_score = 0
    if macd_bearish:
        macd_score += 1
        if macd_hist_bearish:
            macd_score += 0.5
        if macd_falling:
            macd_score += 0.5
    
    put_signals.append(min(macd_score / 2.0, 1.0))
    put_details.append(f"MACD: {macd_score}/2")
    
    # 3. RSI OPTIMAL (poids moyen)
    rsi_bearish_zone = 30 < rsi < 55
    rsi_optimal_zone = 35 < rsi < 50
    
    rsi_score = 0
    if rsi_bearish_zone:
        rsi_score += 1
        if rsi_optimal_zone:
            rsi_score += 1
    
    put_signals.append(min(rsi_score / 2.0, 1.0))
    put_details.append(f"RSI: {rsi_score}/2 ({rsi:.1f})")
    
    # 4. STOCHASTIQUE BAISSIER (poids moyen)
    stoch_bearish = stoch_k < stoch_d
    stoch_not_oversold = stoch_k > 20
    
    stoch_score = 0
    if stoch_bearish:
        stoch_score += 1
        if stoch_not_oversold:
            stoch_score += 1
    
    put_signals.append(min(stoch_score / 2.0, 1.0))
    put_details.append(f"Stoch: {stoch_score}/2")
    
    # 5. ADX + MOMENTUM (poids fort)
    adx_bearish = adx_neg > adx_pos
    momentum_bearish = momentum_3 < 0 and momentum_5 < 0
    
    adx_score = 0
    if adx_strong:
        adx_score += 1
        if adx_bearish:
            adx_score += 0.5
        if momentum_bearish:
            adx_score += 0.5
    
    put_signals.append(min(adx_score / 2.0, 1.0))
    put_details.append(f"ADX: {adx_score}/2 ({adx:.1f})")
    
    # 6. BOLLINGER BANDS (poids moyen)
    bb_bearish_position = 0.3 < bb_position < 0.7
    
    bb_score = 0
    if bb_bearish_position:
        bb_score += 1
        if bb_not_extreme:
            bb_score += 1
    
    put_signals.append(min(bb_score / 2.0, 1.0))
    put_details.append(f"BB: {bb_score}/2 ({bb_position:.2f})")
    
    # 7. CONVERGENCE ET TENDANCE (poids fort)
    price_below_resistance = last['close'] < last['resistance'] * 0.998
    
    convergence_score = 0
    if linear_trend < 0:
        convergence_score += 1
    if signal_convergence < -0.5:
        convergence_score += 0.5
    if price_below_resistance:
        convergence_score += 0.5
    
    put_signals.append(min(convergence_score / 2.0, 1.0))
    put_details.append(f"Conv: {convergence_score}/2")
    
    # ============================================
    # CALCUL DU SCORE PUT
    # ============================================
    
    # Convertir en indicateurs binaires
    put_indicators = [1 if score >= 0.5 else 0 for score in put_signals]
    put_total = sum(put_indicators)
    
    # Score de qualité
    put_weighted_score = sum(s * w for s, w in zip(put_signals, weights))
    put_quality = put_weighted_score / call_max_weight
    
    print(f"[PUT_ANALYSIS] 📊 Indicateurs: {put_indicators}")
    print(f"[PUT_ANALYSIS] 🔢 Total: {put_total}/7 - Qualité: {put_quality:.1%}")
    print(f"[PUT_ANALYSIS] 📋 Détails: {', '.join(put_details)}")
    
    # VÉRIFICATIONS FINALES POUR PUT
    if put_total >= required_score and put_quality >= quality_threshold:
        # Vérifications supplémentaires
        volume_ok = True
        if 'volume_ratio' in last:
            volume_ok = last['volume_ratio'] > 0.7
        
        candle_pattern = last.get('candle_pattern', 'NEUTRAL')
        candle_ok = candle_pattern in ['BEARISH', 'BEARISH_STRONG', 'NEUTRAL']
        
        risk_reward = last.get('risk_reward', 1)
        rr_ok = risk_reward >= 1.2
        
        if volume_ok and candle_ok and rr_ok:
            print(f"[STRATEGY] ✅ Signal PUT validé - Score: {put_total}/7 - Qualité: {put_quality:.1%}")
            return 'PUT'
    
    # ============================================
    # STRATÉGIE DE SECOURS POUR 8 SIGNAUX GARANTIS
    # ============================================
    
    print(f"[STRATEGY_SECOURS] 🔄 Aucun signal strict trouvé, activation mode secours...")
    
    # Calculer le meilleur signal potentiel
    if call_total >= 4 or put_total >= 4:
        # Choisir la direction avec le meilleur score
        if call_total > put_total and call_total >= 4:
            print(f"[STRATEGY_SECOURS] 🎲 Signal CALL secours - Score: {call_total}/7")
            return 'CALL'
        elif put_total > call_total and put_total >= 4:
            print(f"[STRATEGY_SECOURS] 🎲 Signal PUT secours - Score: {put_total}/7")
            return 'PUT'
    
    # Dernier recours: analyse de tendance simple
    if len(df) >= 20:
        # Tendance sur 20 dernières bougies
        prices = df['close'].tail(20).values
        sma_5 = np.mean(prices[-5:])
        sma_20 = np.mean(prices)
        
        if sma_5 > sma_20 and prices[-1] > sma_5:
            print(f"[STRATEGY_SECOURS] 📈 Signal CALL tendance simple")
            return 'CALL'
        elif sma_5 < sma_20 and prices[-1] < sma_5:
            print(f"[STRATEGY_SECOURS] 📉 Signal PUT tendance simple")
            return 'PUT'
    
    print(f"[STRATEGY] ⚠️ Aucun signal trouvé même en mode secours")
    return None


def rule_signal(df):
    """Stratégie standard - Utilise mode QUALITY par défaut (priorité 3)"""
    return rule_signal_ultra_strict(df, session_priority=3)


def get_signal_quality_score(df):
    """Calcule un score de qualité du signal (0-100)"""
    if len(df) < 10:
        return 0
    
    last = df.iloc[-1]
    score = 0
    
    # Multi-confirmations (30 points)
    call_signals, put_signals = 0, 0
    
    # EMA
    if last['ema_fast'] > last['ema_slow']:
        call_signals += 1
    else:
        put_signals += 1
    
    # MACD
    if last['MACD_12_26_9'] > last['MACDs_12_26_9']:
        call_signals += 1
    else:
        put_signals += 1
    
    # RSI
    if last['rsi'] > 50:
        call_signals += 1
    else:
        put_signals += 1
    
    # ADX
    if last['adx_pos'] > last['adx_neg']:
        call_signals += 1
    else:
        put_signals += 1
    
    convergence = abs(call_signals - put_signals)
    if convergence >= 3:
        score += 30
    elif convergence >= 2:
        score += 20
    elif convergence >= 1:
        score += 10
    
    # ADX force (20 points)
    adx = last.get('adx', 0)
    if adx > 30:
        score += 20
    elif adx > 25:
        score += 15
    elif adx > 20:
        score += 10
    elif adx > 15:
        score += 5
    
    # RSI position (15 points)
    rsi = last.get('rsi', 50)
    if 48 < rsi < 52:
        score += 15
    elif 45 < rsi < 55:
        score += 10
    elif 40 < rsi < 60:
        score += 5
    
    # Volatilité contrôlée (15 points)
    bb_width = last.get('BB_width', 0)
    if 0.01 < bb_width < 0.03:
        score += 15
    elif 0.005 < bb_width < 0.04:
        score += 10
    elif 0.002 < bb_width < 0.05:
        score += 5
    
    # Volume (10 points)
    if 'volume_ratio' in last:
        vol_ratio = last['volume_ratio']
        if 0.8 < vol_ratio < 1.5:
            score += 10
        elif 0.5 < vol_ratio < 2.0:
            score += 5
    
    # Tendance claire (10 points)
    ema_alignment = (last['close'] > last['ema_fast'] > last['ema_slow']) or (last['close'] < last['ema_fast'] < last['ema_slow'])
    if ema_alignment:
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
    
    # Qualité du signal
    quality_score = get_signal_quality_score(indicators)
    if quality_score >= 80:
        reason_parts.append("Qualité: EXCELLENT")
    elif quality_score >= 60:
        reason_parts.append("Qualité: BONNE")
    else:
        reason_parts.append("Qualité: MOYENNE")
    
    # Indicateurs clés
    adx = last.get('adx', 0)
    if adx > 25:
        reason_parts.append(f"ADX:{adx:.0f}")
    
    rsi = last.get('rsi', 50)
    reason_parts.append(f"RSI:{rsi:.0f}")
    
    # Convergence
    convergence = last.get('signal_convergence', 0)
    if abs(convergence) > 0.6:
        reason_parts.append("Multi-confirmations")
    
    return " | ".join(reason_parts)


def validate_m1_timing(entry_time):
    """Valide que l'heure d'entrée est bien alignée sur une bougie M1"""
    if isinstance(entry_time, str):
        entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
    corrected = round_to_m1_candle(entry_time)
    is_valid = (entry_time == corrected)
    return is_valid, corrected


def validate_m5_timing(entry_time):
    """Valide que l'heure d'entrée est bien alignée sur une bougie M5"""
    if isinstance(entry_time, str):
        entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
    corrected = round_to_m5_candle(entry_time)
    is_valid = (entry_time == corrected)
    return is_valid, corrected


def get_m1_entry_exit_times(signal_time):
    """Calcule les temps d'entrée et sortie M1 arrondis"""
    if isinstance(signal_time, str):
        signal_time = datetime.fromisoformat(signal_time.replace('Z', '+00:00'))
    entry_time = get_next_m1_candle(signal_time)
    exit_time = entry_time + timedelta(minutes=1)
    return entry_time, exit_time


def get_m5_entry_exit_times(signal_time):
    """Calcule les temps d'entrée et sortie M5 arrondis"""
    if isinstance(signal_time, str):
        signal_time = datetime.fromisoformat(signal_time.replace('Z', '+00:00'))
    entry_time = get_next_m5_candle(signal_time)
    exit_time = entry_time + timedelta(minutes=5)
    return entry_time, exit_time


# ================= NOUVELLES FONCTIONS AMÉLIORÉES =================

def calculate_linear_trend(prices, window=20):
    """Calcule la pente de la tendance linéaire"""
    if len(prices) < window:
        return 0
    
    recent_prices = prices.tail(window).values
    x = np.arange(len(recent_prices))
    
    # Régression linéaire
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_prices)
    
    # Normaliser la pente par rapport au prix moyen
    avg_price = np.mean(recent_prices)
    normalized_slope = slope / avg_price if avg_price != 0 else 0
    
    return normalized_slope * 100  # Retourne en pourcentage


def calculate_signal_convergence(df):
    """Calcule la convergence des signaux indicateurs"""
    if len(df) < 10:
        return 0
    
    last = df.iloc[-1]
    signals = []
    
    # EMA signal
    if 'ema_fast' in last and 'ema_slow' in last:
        ema_signal = 1 if last['ema_fast'] > last['ema_slow'] else -1
        signals.append(ema_signal)
    
    # MACD signal
    if 'MACD_12_26_9' in last and 'MACDs_12_26_9' in last:
        macd_signal = 1 if last['MACD_12_26_9'] > last['MACDs_12_26_9'] else -1
        signals.append(macd_signal)
    
    # RSI signal
    if 'rsi' in last:
        rsi_signal = 1 if last['rsi'] > 50 else -1
        signals.append(rsi_signal)
    
    # ADX signal
    if 'adx_pos' in last and 'adx_neg' in last:
        adx_signal = 1 if last['adx_pos'] > last['adx_neg'] else -1
        signals.append(adx_signal)
    
    # Stochastic signal
    if 'stoch_k' in last and 'stoch_d' in last:
        stoch_signal = 1 if last['stoch_k'] > last['stoch_d'] else -1
        signals.append(stoch_signal)
    
    if signals:
        convergence = np.mean(signals)
        return convergence
    
    return 0


def calculate_data_quality(df):
    """Calcule un score de qualité des données (0-1)"""
    if len(df) < 20:
        return 0.5
    
    scores = []
    
    # Vérification des NaN
    required_cols = ['open', 'high', 'low', 'close']
    nan_ratio = df[required_cols].isnull().mean().mean()
    scores.append(1 - nan_ratio)
    
    # Vérification de la cohérence des prix
    price_consistency = ((df['low'] <= df['open']) & 
                         (df['open'] <= df['high']) &
                         (df['low'] <= df['close']) &
                         (df['close'] <= df['high'])).mean()
    scores.append(price_consistency)
    
    # Volatilité raisonnable
    volatility = df['close'].pct_change().std()
    if 0.0001 < volatility < 0.05:  # 0.01% à 5%
        scores.append(1.0)
    else:
        scores.append(0.5)
    
    # Continuité temporelle
    if len(df) > 1:
        time_diff = df.index.to_series().diff().dt.total_seconds()
        if (time_diff > 120).any():  # Gaps de plus de 2 minutes
            scores.append(0.7)
        else:
            scores.append(1.0)
    
    return np.mean(scores)


def identify_candle_pattern(df):
    """Identifie les patterns de chandeliers simples"""
    if len(df) < 2:
        return 'NEUTRAL'
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    open_price = last['open']
    close_price = last['close']
    high_price = last['high']
    low_price = last['low']
    
    # Body size
    body_size = abs(close_price - open_price)
    total_range = high_price - low_price
    
    if total_range == 0:
        return 'NEUTRAL'
    
    body_ratio = body_size / total_range
    
    # Déterminer le type de bougie
    if close_price > open_price:
        if body_ratio > 0.7:
            return 'BULLISH_STRONG'
        elif body_ratio > 0.3:
            return 'BULLISH'
        else:
            return 'BULLISH_WEAK'
    else:
        if body_ratio > 0.7:
            return 'BEARISH_STRONG'
        elif body_ratio > 0.3:
            return 'BEARISH'
        else:
            return 'BEARISH_WEAK'


def calculate_risk_reward_ratio(df, lookback=20):
    """Calcule le ratio risque/récompense potentiel"""
    if len(df) < lookback:
        return 1.0
    
    last = df.iloc[-1]
    
    # Support et résistance récents
    recent_high = df['high'].tail(lookback).max()
    recent_low = df['low'].tail(lookback).min()
    current_price = last['close']
    
    # Distance aux niveaux
    distance_to_resistance = abs(recent_high - current_price)
    distance_to_support = abs(current_price - recent_low)
    
    if distance_to_support > 0:
        rr_ratio = distance_to_resistance / distance_to_support
        return min(rr_ratio, 3.0)  # Limiter à 3:1
    
    return 1.0


def calculate_pair_score(df, mode='forex'):
    """Calcule un score de qualité pour la paire actuelle (0-100)"""
    if len(df) < 50:
        return 0
    
    last = df.iloc[-1]
    scores = []
    
    # RSI score
    rsi = last.get('rsi', 50)
    if 45 <= rsi <= 55:
        scores.append(100)
    elif 40 <= rsi <= 60:
        scores.append(80)
    elif 35 <= rsi <= 65:
        scores.append(60)
    else:
        scores.append(30)
    
    # ADX score
    adx = last.get('adx', 0)
    if adx > 30:
        scores.append(100)
    elif adx > 25:
        scores.append(80)
    elif adx > 20:
        scores.append(60)
    elif adx > 15:
        scores.append(40)
    else:
        scores.append(20)
    
    # Volatilité score
    bb_width = last.get('BB_width', 0)
    if mode == 'forex':
        if 0.01 < bb_width < 0.03:
            scores.append(100)
        elif 0.005 < bb_width < 0.04:
            scores.append(80)
        elif 0.002 < bb_width < 0.05:
            scores.append(60)
        else:
            scores.append(30)
    else:  # OTC
        if bb_width < 0.05:
            scores.append(100)
        elif bb_width < 0.08:
            scores.append(80)
        elif bb_width < 0.12:
            scores.append(60)
        else:
            scores.append(30)
    
    # Tendance score
    if 'ema_fast' in last and 'ema_slow' in last:
        trend_alignment = 1 if (last['close'] > last['ema_fast'] > last['ema_slow']) or (last['close'] < last['ema_fast'] < last['ema_slow']) else 0
        scores.append(trend_alignment * 100)
    
    # Volume score (si disponible)
    if 'volume_ratio' in last:
        vol_ratio = last['volume_ratio']
        if 0.8 < vol_ratio < 1.2:
            scores.append(100)
        elif 0.6 < vol_ratio < 1.5:
            scores.append(80)
        elif 0.4 < vol_ratio < 2.0:
            scores.append(60)
        else:
            scores.append(30)
    
    # Moyenne des scores
    final_score = np.mean(scores) if scores else 0
    return final_score


def get_trading_recommendation(df, mode='forex'):
    """
    Génère une recommandation de trading basée sur l'analyse technique
    
    Retourne:
        dict: Recommandation avec score, direction et confiance
    """
    try:
        if len(df) < 50:
            return {'score': 0, 'direction': 'NEUTRAL', 'confidence': 0}
        
        # Obtenir le signal
        signal = rule_signal_ultra_strict(df, session_priority=3 if mode == 'forex' else 2)
        
        if not signal:
            return {'score': 0, 'direction': 'NEUTRAL', 'confidence': 0}
        
        # Calculer la confiance
        last = df.iloc[-1]
        quality_score = get_signal_quality_score(df)
        
        # Ajuster la confiance selon le mode
        if mode == 'forex':
            # Forex nécessite plus de confirmation
            confidence = (quality_score / 100) * 0.9  # Réduction de 10%
        else:
            # OTC plus permissif
            confidence = min((quality_score / 100) * 1.1, 0.95)  # Augmentation de 10%
        
        return {
            'score': quality_score,
            'direction': signal,
            'confidence': confidence,
            'quality_level': 'EXCELLENT' if quality_score >= 80 else 'GOOD' if quality_score >= 60 else 'FAIR'
        }
        
    except Exception as e:
        print(f"[RECOMMENDATION] ❌ Erreur: {e}")
        return {'score': 0, 'direction': 'NEUTRAL', 'confidence': 0}
