
"""
utils.py - STRATÉGIE SAINT GRAAL FOREX M1 AVEC GARANTIE
Version ultra-optimisée pour le trading binaire M1 avec expiration 1 minute
Garantie de 8 signaux par session grâce au mode secours intelligent
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from ta.trend import EMAIndicator, MACD, ADXIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange, DonchianChannel
from ta.volume import VolumeWeightedAveragePrice, AccDistIndexIndicator
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIGURATION SAINT GRAAL AVEC GARANTIE =================

SAINT_GRAAL_CONFIG = {
    # Timeframes optimisés pour M1
    'rsi_period': 7,
    'ema_fast': 5,
    'ema_slow': 13,
    'stoch_period': 5,
    'macd_fast': 6,
    'macd_slow': 13,
    'macd_signal': 5,
    'bb_period': 20,
    'adx_period': 10,
    
    # Seuils pour mode STRICT
    'strict': {
        'rsi_overbought': 68,
        'rsi_oversold': 32,
        'adx_min': 22,
        'min_score_required': 7.0,
        'min_percentage_required': 65.0
    },
    
    # Seuils pour mode GARANTIE (plus souples)
    'guarantee': {
        'rsi_overbought': 72,
        'rsi_oversold': 28,
        'adx_min': 18,
        'min_score_required': 5.0,
        'min_percentage_required': 55.0
    },
    
    # Paramètres généraux
    'min_volume_ratio': 0.7,
    'max_volatility': 0.04,
    'min_price_change': 0.0002,
}

# ================= FONCTIONS DE BASE =================

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

# ================= INDICATEURS SAINT GRAAL =================

def compute_saint_graal_indicators(df):
    """
    Calcule TOUS les indicateurs pour la stratégie Saint Graal Forex M1
    """
    df = df.copy()
    
    # Assurer les types numériques
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remplir les NaN
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    config = SAINT_GRAAL_CONFIG
    
    # ===== 1. INDICATEURS DE TENDANCE OPTIMISÉS =====
    
    df['ema_3'] = EMAIndicator(close=df['close'], window=3).ema_indicator()
    df['ema_5'] = EMAIndicator(close=df['close'], window=config['ema_fast']).ema_indicator()
    df['ema_13'] = EMAIndicator(close=df['close'], window=config['ema_slow']).ema_indicator()
    df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
    
    # MACD rapide (6,13,5)
    macd = MACD(
        close=df['close'],
        window_fast=config['macd_fast'],
        window_slow=config['macd_slow'],
        window_sign=config['macd_signal']
    )
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # MACD ultra-rapide (3,7,2)
    macd_ultra = MACD(close=df['close'], window_fast=3, window_slow=7, window_sign=2)
    df['macd_ultra'] = macd_ultra.macd()
    df['macd_signal_ultra'] = macd_ultra.macd_signal()
    
    # ADX rapide
    adx = ADXIndicator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=config['adx_period']
    )
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()
    
    # ===== 2. INDICATEURS DE MOMENTUM OPTIMISÉS =====
    
    df['rsi_7'] = RSIIndicator(close=df['close'], window=config['rsi_period']).rsi()
    df['rsi_3'] = RSIIndicator(close=df['close'], window=3).rsi()
    
    # Stochastique rapide
    stoch = StochasticOscillator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=config['stoch_period'],
        smooth_window=3
    )
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Rate of Change
    df['roc_5'] = ROCIndicator(close=df['close'], window=5).roc()
    df['roc_10'] = ROCIndicator(close=df['close'], window=10).roc()
    
    # ===== 3. VOLATILITÉ ET VOLUME =====
    
    # Bollinger Bands
    bb = BollingerBands(
        close=df['close'],
        window=config['bb_period'],
        window_dev=2
    )
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ATR pour le risque
    df['atr_10'] = AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=10
    ).average_true_range()
    
    # Donchian Channel
    donchian = DonchianChannel(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=10
    )
    df['donchian_high'] = donchian.donchian_channel_hband()
    df['donchian_low'] = donchian.donchian_channel_lband()
    
    # ===== 4. INDICATEURS AVANCÉS =====
    
    # Ichimoku Cloud (version light)
    ichimoku = IchimokuIndicator(
        high=df['high'],
        low=df['low'],
        window1=9,
        window2=26,
        window3=52
    )
    df['ichimoku_a'] = ichimoku.ichimoku_a()
    df['ichimoku_b'] = ichimoku.ichimoku_b()
    
    # Volume Weighted Average Price
    if 'volume' in df.columns:
        vwap = VolumeWeightedAveragePrice(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume'],
            window=20
        )
        df['vwap'] = vwap.volume_weighted_average_price()
        df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    
    # ===== 5. MOMENTUM ET PRICE ACTION =====
    
    df['momentum_1'] = df['close'].pct_change(1) * 100
    df['momentum_3'] = df['close'].pct_change(3) * 100
    df['momentum_5'] = df['close'].pct_change(5) * 100
    
    df['acceleration'] = df['momentum_1'].diff()
    df['volatility_10'] = df['close'].rolling(10).std() / df['close'].rolling(10).mean()
    df['range_ratio'] = (df['high'] - df['low']) / df['close']
    
    # ===== 6. PATTERNS DE BOUGIES =====
    
    df['candle_body'] = df['close'] - df['open']
    df['candle_size'] = df['high'] - df['low']
    df['body_ratio'] = abs(df['candle_body']) / df['candle_size'].replace(0, 0.00001)
    
    df['bullish_candle'] = (df['close'] > df['open']).astype(int)
    df['bearish_candle'] = (df['close'] < df['open']).astype(int)
    
    # ===== 7. CONVERGENCE =====
    
    df['trend_score'] = (
        (df['ema_5'] > df['ema_13']).astype(int) +
        (df['macd'] > df['macd_signal']).astype(int) +
        (df['adx_pos'] > df['adx_neg']).astype(int) +
        (df['stoch_k'] > df['stoch_d']).astype(int) +
        (df['rsi_7'] > 50).astype(int)
    ) / 5.0
    
    # ===== 8. QUALITÉ DES DONNÉES =====
    
    df['data_quality'] = (
        (df['close'].notna()).astype(int) +
        (df['volume'].notna() if 'volume' in df.columns else 1) +
        (df['volatility_10'] < 0.1).astype(int) +
        (df['range_ratio'] > 0).astype(int)
    ) / 4.0
    
    # Remplir les derniers NaN
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    return df

# ================= STRATÉGIE SAINT GRAAL AVEC GARANTIE =================

def rule_signal_saint_graal_with_guarantee(df, session_priority=3, signal_count=0, total_signals_needed=8):
    """
    STRATÉGIE SAINT GRAAL FOREX M1 AVEC GARANTIE DE 8 SIGNAUX
    
    Logique :
    1. D'abord essayer mode STRICT (haute qualité)
    2. Si pas de signal strict ET besoin de garantie → mode GARANTIE
    3. Mode GARANTIE : conditions plus souples mais filtrées
    4. Dernier recours : signal minimal pour compléter la session
    """
    
    if len(df) < 30:
        print("[SAINT-GRAAL] ⚠️ Données insuffisantes")
        return None
    
    # Calculer le besoin de garantie
    signals_still_needed = total_signals_needed - signal_count
    signals_remaining = total_signals_needed - signal_count if signal_count < total_signals_needed else 0
    
    # Déterminer la sévérité basée sur la progression de la session
    if signal_count == 0:
        # Premier signal : mode strict
        mode = "STRICT"
        guarantee_needed = False
    elif signals_still_needed > 3:
        # Encore beaucoup de signaux nécessaires : mode normal
        mode = "STRICT"
        guarantee_needed = False
    elif signals_still_needed > 0:
        # Peu de signaux restants : activer garantie si nécessaire
        mode = "STRICT"
        guarantee_needed = True
    else:
        # Session complète
        return None
    
    print(f"\n[SAINT-GRAAL] 🔍 Mode: {mode} | Signal: {signal_count+1}/8")
    print(f"[SAINT-GRAAL] 📊 Signaux restants: {signals_still_needed}")
    
    # ===== 1. ESSAYER MODE STRICT D'ABORD =====
    
    strict_signal = rule_signal_saint_graal_strict(df)
    
    if strict_signal:
        print(f"[SAINT-GRAAL] ✅ Signal STRICT trouvé: {strict_signal}")
        return {
            'signal': strict_signal,
            'mode': 'STRICT',
            'quality': 'HIGH',
            'score': calculate_signal_quality_score(df)
        }
    
    print(f"[SAINT-GRAAL] ⚠️ Pas de signal strict")
    
    # ===== 2. SI GARANTIE NÉCESSAIRE, ESSAYER MODE GARANTIE =====
    
    if guarantee_needed and signals_still_needed > 0:
        print(f"[SAINT-GRAAL] 🔄 Activation mode GARANTIE (signaux restants: {signals_still_needed})")
        
        guarantee_signal = rule_signal_saint_graal_guarantee(df)
        
        if guarantee_signal:
            print(f"[SAINT-GRAAL] ✅ Signal GARANTIE trouvé: {guarantee_signal}")
            return {
                'signal': guarantee_signal,
                'mode': 'GUARANTEE',
                'quality': 'MEDIUM',
                'score': calculate_signal_quality_score(df) * 0.8  # Réduction de 20% pour garantie
            }
        
        print(f"[SAINT-GRAAL] ⚠️ Pas de signal garantie")
    
    # ===== 3. DERNIER RECOURS POUR GARANTIR LA SESSION =====
    
    if signals_still_needed > 0 and signal_count < total_signals_needed:
        # Calculer l'urgence (plus on approche de la fin, plus on est urgent)
        urgency = (total_signals_needed - signal_count) / total_signals_needed
        
        if urgency > 0.5:  # Plus de la moitié des signaux manquants
            print(f"[SAINT-GRAAL] 🚨 DERNIER RECOURS (urgence: {urgency:.0%})")
            
            last_resort_signal = rule_signal_last_resort(df)
            
            if last_resort_signal:
                print(f"[SAINT-GRAAL] ✅ Signal DERNIER RECOURS: {last_resort_signal}")
                return {
                    'signal': last_resort_signal,
                    'mode': 'LAST_RESORT',
                    'quality': 'LOW',
                    'score': calculate_signal_quality_score(df) * 0.6  # Réduction de 40%
                }
    
    print(f"[SAINT-GRAAL] ❌ Aucun signal possible")
    return None

def rule_signal_saint_graal_strict(df):
    """Mode STRICT - Haute qualité, seuils élevés"""
    if len(df) < 30:
        return None
    
    config = SAINT_GRAAL_CONFIG['strict']
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # ===== FILTRES DE SÉCURITÉ STRICTS =====
    
    if last['data_quality'] < 0.8:
        return None
    
    if last['bb_width'] > 0.035:
        return None
    
    if last['adx'] < config['adx_min']:
        return None
    
    if last['rsi_7'] > config['rsi_overbought'] or last['rsi_7'] < config['rsi_oversold']:
        return None
    
    # ===== ANALYSE CALL STRICTE =====
    
    call_score = 0
    max_call_score = 12
    
    # 1. Tendance (3 points)
    if last['ema_5'] > last['ema_13'] > last['ema_20']:
        call_score += 3
    
    # 2. MACD (2 points)
    if last['macd'] > last['macd_signal'] and last['macd_diff'] > 0:
        call_score += 2
    
    # 3. RSI optimal (2 points)
    if 55 < last['rsi_7'] < 65:
        call_score += 2
    
    # 4. ADX +DI (2 points)
    if last['adx_pos'] > last['adx_neg'] and last['adx_pos'] > 25:
        call_score += 2
    
    # 5. Stochastic (1 point)
    if last['stoch_k'] > last['stoch_d'] and 50 < last['stoch_k'] < 75:
        call_score += 1
    
    # 6. Price action (2 points)
    if last['bullish_candle'] == 1 and last['body_ratio'] > 0.4:
        call_score += 2
    
    call_percentage = (call_score / max_call_score) * 100
    
    # ===== ANALYSE PUT STRICTE =====
    
    put_score = 0
    max_put_score = 12
    
    # 1. Tendance (3 points)
    if last['ema_5'] < last['ema_13'] < last['ema_20']:
        put_score += 3
    
    # 2. MACD (2 points)
    if last['macd'] < last['macd_signal'] and last['macd_diff'] < 0:
        put_score += 2
    
    # 3. RSI optimal (2 points)
    if 35 < last['rsi_7'] < 45:
        put_score += 2
    
    # 4. ADX -DI (2 points)
    if last['adx_neg'] > last['adx_pos'] and last['adx_neg'] > 25:
        put_score += 2
    
    # 5. Stochastic (1 point)
    if last['stoch_k'] < last['stoch_d'] and 25 < last['stoch_k'] < 50:
        put_score += 1
    
    # 6. Price action (2 points)
    if last['bearish_candle'] == 1 and last['body_ratio'] > 0.4:
        put_score += 2
    
    put_percentage = (put_score / max_put_score) * 100
    
    # ===== DÉCISION =====
    
    min_score = config['min_score_required']
    min_percentage = config['min_percentage_required']
    
    if call_score >= min_score and call_percentage >= min_percentage and call_score > put_score:
        return 'CALL'
    
    if put_score >= min_score and put_percentage >= min_percentage and put_score > call_score:
        return 'PUT'
    
    return None

def rule_signal_saint_graal_guarantee(df):
    """Mode GARANTIE - Conditions plus souples mais filtrées"""
    if len(df) < 20:
        return None
    
    config = SAINT_GRAAL_CONFIG['guarantee']
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # ===== FILTRES DE SÉCURITÉ MODÉRÉS =====
    
    if last['data_quality'] < 0.6:
        return None
    
    if last['bb_width'] > 0.045:
        return None
    
    if last['adx'] < config['adx_min']:
        return None
    
    # ===== ANALYSE CALL GARANTIE =====
    
    call_conditions = []
    
    # Conditions plus souples
    call_conditions.append(last['ema_5'] > last['ema_13'])
    call_conditions.append(last['macd'] > last['macd_signal'])
    call_conditions.append(50 < last['rsi_7'] < 70)
    call_conditions.append(last['adx_pos'] > last['adx_neg'])
    call_conditions.append(last['stoch_k'] > last['stoch_d'])
    call_conditions.append(last['close'] > prev['close'])
    
    call_score = sum(call_conditions)
    
    # ===== ANALYSE PUT GARANTIE =====
    
    put_conditions = []
    
    put_conditions.append(last['ema_5'] < last['ema_13'])
    put_conditions.append(last['macd'] < last['macd_signal'])
    put_conditions.append(30 < last['rsi_7'] < 50)
    put_conditions.append(last['adx_neg'] > last['adx_pos'])
    put_conditions.append(last['stoch_k'] < last['stoch_d'])
    put_conditions.append(last['close'] < prev['close'])
    
    put_score = sum(put_conditions)
    
    # ===== DÉCISION =====
    
    min_conditions = 4  # 4/6 conditions minimum
    
    if call_score >= min_conditions and call_score > put_score:
        return 'CALL'
    
    if put_score >= min_conditions and put_score > call_score:
        return 'PUT'
    
    return None

def rule_signal_last_resort(df):
    """Dernier recours - Conditions minimales pour compléter la session"""
    if len(df) < 10:
        return None
    
    last = df.iloc[-1]
    
    # Analyse très simple basée sur RSI et prix
    if last['rsi_7'] > 50 and last['close'] > last['ema_5']:
        return 'CALL'
    elif last['rsi_7'] < 50 and last['close'] < last['ema_5']:
        return 'PUT'
    
    # Dernier recours absolu : direction du dernier mouvement
    prices = df['close'].tail(5).values
    if prices[-1] > prices[-2]:
        return 'CALL'
    else:
        return 'PUT'

# ================= FONCTIONS DE COMPATIBILITÉ =================

def compute_indicators(df, ema_fast=8, ema_slow=21, rsi_len=14, bb_len=20):
    """Wrapper pour compatibilité"""
    return compute_saint_graal_indicators(df)

def rule_signal_ultra_strict(df, session_priority=3):
    """Wrapper pour compatibilité avec le bot existant"""
    # Pour utiliser avec le bot, nous avons besoin de signal_count et total_signals
    # Par défaut, nous utilisons une version simple
    result = rule_signal_saint_graal_with_guarantee(
        df, 
        session_priority, 
        signal_count=0,  # À remplacer par le vrai count dans le bot
        total_signals_needed=8
    )
    
    if result:
        return result['signal']
    return None

def rule_signal(df):
    """Par défaut, utilise Saint Graal avec garantie"""
    result = rule_signal_saint_graal_with_guarantee(df, signal_count=0, total_signals_needed=8)
    if result:
        return result['signal']
    return None

def calculate_signal_quality_score(df):
    """Calcule un score de qualité global du signal (0-100)"""
    if len(df) < 20:
        return 0
    
    last = df.iloc[-1]
    score = 0
    
    # Convergence (30 points max)
    convergence = last.get('trend_score', 0.5)
    score += convergence * 30
    
    # Force de la tendance (25 points)
    adx = last.get('adx', 0)
    if adx > 30:
        score += 25
    elif adx > 25:
        score += 20
    elif adx > 20:
        score += 15
    elif adx > 15:
        score += 10
    
    # Alignement des indicateurs (20 points)
    aligned_indicators = 0
    if last.get('ema_5', 0) > last.get('ema_13', 0):
        aligned_indicators += 1
    if last.get('macd', 0) > last.get('macd_signal', 0):
        aligned_indicators += 1
    if last.get('rsi_7', 50) > 50:
        aligned_indicators += 1
    if last.get('stoch_k', 50) > last.get('stoch_d', 50):
        aligned_indicators += 1
    
    score += (aligned_indicators / 4) * 20
    
    # Volatilité contrôlée (15 points)
    bb_width = last.get('bb_width', 0)
    if 0.01 < bb_width < 0.03:
        score += 15
    elif 0.005 < bb_width < 0.04:
        score += 10
    elif 0.002 < bb_width < 0.05:
        score += 5
    
    # Volume (10 points) - si disponible
    if 'price_vs_vwap' in last:
        if abs(last['price_vs_vwap']) < 0.002:
            score += 10
        elif abs(last['price_vs_vwap']) < 0.005:
            score += 5
    
    return min(score, 100)

def format_signal_reason(direction, confidence, indicators):
    """Formate une raison lisible pour le signal"""
    last = indicators.iloc[-1]
    
    reason_parts = [f"SAINT-GRAAL {direction}"]
    
    # Qualité du signal
    quality_score = calculate_signal_quality_score(indicators)
    if quality_score >= 80:
        reason_parts.append("QUALITÉ: EXCELLENTE")
    elif quality_score >= 70:
        reason_parts.append("QUALITÉ: BONNE")
    elif quality_score >= 60:
        reason_parts.append("QUALITÉ: MOYENNE")
    else:
        reason_parts.append("QUALITÉ: FAIBLE")
    
    # Indicateurs clés
    reason_parts.append(f"RSI7: {last.get('rsi_7', 0):.1f}")
    reason_parts.append(f"ADX: {last.get('adx', 0):.1f}")
    
    # Convergence
    convergence = last.get('trend_score', 0)
    if convergence > 0.7:
        reason_parts.append("CONVERGENCE: FORTE")
    
    return " | ".join(reason_parts)

def is_kill_zone_optimal(hour_utc):
    """Zones temporelles optimales"""
    if 7 <= hour_utc < 10:
        return True, "London Open", 5
    if 13 <= hour_utc < 16:
        return True, "NY Open", 5
    if 10 <= hour_utc < 12:
        return True, "London/NY Overlap", 5
    if 1 <= hour_utc < 4:
        return True, "Asia Close", 3
    return False, None, 0

# ================= FONCTIONS DE GESTION =================

def get_signal_with_metadata(df, signal_count=0, total_signals=8):
    """
    Fonction principale pour obtenir un signal avec métadonnées
    À utiliser dans le bot pour suivre la progression
    """
    result = rule_signal_saint_graal_with_guarantee(
        df, 
        signal_count=signal_count,
        total_signals_needed=total_signals
    )
    
    if result:
        return {
            'direction': result['signal'],
            'mode': result['mode'],
            'quality': result['quality'],
            'score': result['score'],
            'reason': format_signal_reason(
                result['signal'], 
                result['score']/100, 
                df
            )
        }
    
    return None
