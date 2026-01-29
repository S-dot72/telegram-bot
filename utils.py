"""
utils.py - STRATÉGIE FOREX M1 - 6 SIGNAUX QUALITÉ MAXIMALE
Version optimisée pour 6 signaux haute qualité, extensible à 8 si conditions optimales
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIGURATION QUALITÉ MAXIMALE =================

SAINT_GRAAL_CONFIG = {
    # Indicateurs M1 optimisés
    'rsi_period': 7,
    'ema_fast': 5,
    'ema_slow': 13,
    'stoch_period': 5,
    
    # Seuils QUALITÉ MAXIMALE (6 signaux)
    'max_quality': {
        'rsi_overbought': 67,
        'rsi_oversold': 33,
        'adx_min': 24,
        'stoch_overbought': 73,
        'stoch_oversold': 27,
        'min_confluence_points': 8,  # 8/10 points minimum
        'min_body_ratio': 0.45,
        'max_wick_ratio': 0.4,
    },
    
    # Seuils QUALITÉ ÉLEVÉE (extensible à 8 signaux)
    'high_quality': {
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'adx_min': 20,
        'stoch_overbought': 75,
        'stoch_oversold': 25,
        'min_confluence_points': 6,  # 6/10 points minimum
        'min_body_ratio': 0.35,
        'max_wick_ratio': 0.5,
    },
    
    # Filtres anti-manipulation renforcés
    'anti_manip': {
        'max_wick_ratio': 0.55,
        'max_candle_size_ratio': 2.2,
        'min_ema_spread': 0.0008,
        'max_volatility': 0.035,
        'min_data_quality': 0.85,
    },
    
    # Paramètres généraux
    'target_signals': 6,  # Objectif principal
    'max_signals': 8,     # Maximum possible
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

# ================= INDICATEURS QUALITÉ MAX =================

def compute_saint_graal_indicators(df):
    """
    Calcule les indicateurs pour qualité maximale
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
    
    # ===== 1. INDICATEURS PRINCIPAUX =====
    
    # EMA 5 & 13 (tendance)
    df['ema_5'] = EMAIndicator(close=df['close'], window=config['ema_fast']).ema_indicator()
    df['ema_13'] = EMAIndicator(close=df['close'], window=config['ema_slow']).ema_indicator()
    df['ema_spread'] = abs(df['ema_5'] - df['ema_13']) / df['close']
    df['ema_trend'] = (df['ema_5'] > df['ema_13']).astype(int)
    
    # RSI 7 (momentum)
    df['rsi_7'] = RSIIndicator(close=df['close'], window=config['rsi_period']).rsi()
    df['rsi_trend'] = (df['rsi_7'] > 50).astype(int)
    
    # Stochastique 5,3,3 (sur-achat/vente)
    stoch = StochasticOscillator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=config['stoch_period'],
        smooth_window=3
    )
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df['stoch_trend'] = (df['stoch_k'] > df['stoch_d']).astype(int)
    
    # ADX 10 (force de tendance)
    adx = ADXIndicator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=10
    )
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()
    df['adx_trend'] = (df['adx_pos'] > df['adx_neg']).astype(int)
    
    # ===== 2. VOLATILITÉ ET RISQUE =====
    
    # ATR 10
    df['atr_10'] = AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=10
    ).average_true_range()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ===== 3. PRICE ACTION DÉTAILLÉE =====
    
    # Bougie actuelle
    df['candle_body'] = df['close'] - df['open']
    df['candle_size'] = df['high'] - df['low']
    df['body_ratio'] = abs(df['candle_body']) / df['candle_size'].replace(0, 0.00001)
    
    # Mèches
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['total_wick'] = df['upper_wick'] + df['lower_wick']
    df['wick_ratio'] = df['total_wick'] / abs(df['candle_body']).replace(0, 0.00001)
    
    # Tendance prix
    df['price_trend'] = (df['close'] > df['open']).astype(int)
    df['momentum_1'] = df['close'].pct_change(1) * 100
    
    # ===== 4. QUALITÉ ET CONVERGENCE =====
    
    # Score de convergence (5 indicateurs alignés)
    df['convergence_raw'] = (
        df['ema_trend'] + 
        df['rsi_trend'] + 
        df['stoch_trend'] + 
        df['adx_trend'] + 
        df['price_trend']
    )
    df['convergence_score'] = df['convergence_raw'] / 5.0
    
    # Qualité globale
    df['data_quality'] = (
        (df['close'].notna()).astype(int) +
        (df['bb_width'] < 0.04).astype(int) +
        (df['wick_ratio'] < 0.6).astype(int) +
        (df['body_ratio'] > 0.2).astype(int)
    ) / 4.0
    
    # Remplir les derniers NaN
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    return df

# ================= FILTRES ANTI-MANIPULATION RENFORCÉS =================

def check_anti_manipulation(df, strict_mode=True):
    """Vérifie les conditions anti-manipulation renforcées"""
    if len(df) < 15:
        return False, "Données insuffisantes"
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    anti = SAINT_GRAAL_CONFIG['anti_manip']
    
    # 1. Qualité des données minimale
    if last['data_quality'] < anti['min_data_quality']:
        return False, f"Qualité données faible: {last['data_quality']:.2f}"
    
    # 2. Mèches suspectes
    max_wick = anti['max_wick_ratio'] * (0.9 if strict_mode else 1.0)
    if last['wick_ratio'] > max_wick:
        return False, f"Mèche suspecte: {last['wick_ratio']:.1%}"
    
    # 3. Bougie anormale
    if last['candle_size'] > last['atr_10'] * 3:
        return False, f"Bougie trop grande: {last['candle_size']:.5f} > ATRx3"
    
    # 4. Volatilité excessive
    if last['bb_width'] > anti['max_volatility']:
        return False, f"Volatilité excessive: {last['bb_width']:.3%}"
    
    # 5. EMA trop plates
    if last['ema_spread'] < anti['min_ema_spread']:
        return False, f"EMA plates: écart {last['ema_spread']:.5%}"
    
    # 6. ADX trop faible (pas de tendance)
    if last['adx'] < 15:
        return False, f"ADX faible: {last['adx']:.1f}"
    
    return True, "OK"

# ================= STRATÉGIE 6 SIGNAUX QUALITÉ MAX =================

def rule_signal_saint_graal_with_guarantee(df, signal_count=0, total_signals_needed=6):
    """
    STRATÉGIE - 6 signaux qualité maximale
    
    Priorités absolues:
    1. QUALITÉ MAXIMALE pour 6 signaux
    2. Pas de compromis sur les filtres
    3. Si conditions optimales, extension à 8 signaux
    """
    
    if len(df) < 30:
        print("[STRATEGIE] ⚠️ Données insuffisantes")
        return None
    
    # Objectif principal: 6 signaux qualité max
    target_signals = SAINT_GRAAL_CONFIG['target_signals']
    max_signals = SAINT_GRAAL_CONFIG['max_signals']
    
    # Calculer l'urgence réelle
    signals_remaining = target_signals - signal_count
    urgency = signals_remaining / target_signals if signals_remaining > 0 else 0
    
    print(f"\n[STRATEGIE] 🎯 Signal #{signal_count+1}/{target_signals} | Urgence: {urgency:.0%}")
    
    # Vérifier anti-manipulation STRICT (toujours)
    manip_ok, manip_reason = check_anti_manipulation(df, strict_mode=True)
    if not manip_ok:
        print(f"[STRATEGIE] ⚠️ Anti-manip: {manip_reason}")
        return None
    
    # ===== 1. QUALITÉ MAXIMALE (signaux 1-6) =====
    if signal_count < target_signals:
        max_quality_signal = rule_signal_max_quality(df)
        if max_quality_signal:
            quality_score = calculate_signal_quality(df, max_quality_signal)
            if quality_score >= 85:  # Seuil très élevé
                print(f"[STRATEGIE] ✅ Signal QUALITÉ MAX #{signal_count+1} | Score: {quality_score}/100")
                return {
                    'signal': max_quality_signal,
                    'mode': 'MAX_QUALITY',
                    'quality': 'EXCELLENT',
                    'score': quality_score
                }
            else:
                print(f"[STRATEGIE] ⚠️ Qualité insuffisante: {quality_score}/100")
    
    # ===== 2. EXTENSION OPTIONNELLE (signaux 7-8) =====
    elif signal_count < max_signals:
        print(f"[STRATEGIE] 🔄 Mode extension (signal {signal_count+1})")
        extended_signal = rule_signal_high_quality(df)
        if extended_signal:
            quality_score = calculate_signal_quality(df, extended_signal)
            if quality_score >= 70:  # Seuil élevé pour extension
                print(f"[STRATEGIE] ✅ Signal EXTENSION #{signal_count+1} | Score: {quality_score}/100")
                return {
                    'signal': extended_signal,
                    'mode': 'EXTENSION',
                    'quality': 'HIGH',
                    'score': quality_score
                }
    
    print(f"[STRATEGIE] ❌ Pas de signal qualité suffisante")
    return None

def rule_signal_max_quality(df):
    """QUALITÉ MAXIMALE - Filtres très stricts"""
    if len(df) < 25:
        return None
    
    config = SAINT_GRAAL_CONFIG['max_quality']
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # ===== FILTRES ABSOLUS =====
    
    # 1. Qualité données exceptionnelle
    if last['data_quality'] < 0.9:
        return None
    
    # 2. ADX fort (tendance claire)
    if last['adx'] < config['adx_min']:
        return None
    
    # 3. Convergence très forte (4+/5 indicateurs alignés)
    if last['convergence_raw'] < 4:
        return None
    
    # 4. Bougie de qualité (corps significatif)
    if last['body_ratio'] < config['min_body_ratio']:
        return None
    
    # 5. Pas de mèches suspectes
    if last['wick_ratio'] > config['max_wick_ratio']:
        return None
    
    # ===== ANALYSE CALL MAX QUALITY =====
    
    call_points = 0
    max_points = 10
    
    # 1. EMA alignées et écarté (3 points)
    if last['ema_5'] > last['ema_13'] and last['ema_spread'] > 0.001:
        call_points += 3
    
    # 2. RSI optimal (58-67) (2 points)
    if 58 < last['rsi_7'] < config['rsi_overbought']:
        call_points += 2
    
    # 3. Stochastique directionnel mais pas overbought (2 points)
    if last['stoch_k'] > last['stoch_d'] and last['stoch_k'] < config['stoch_overbought']:
        call_points += 2
    
    # 4. ADX +DI fort (2 points)
    if last['adx_pos'] > last['adx_neg'] and last['adx_pos'] > 28:
        call_points += 2
    
    # 5. Bougie haussière forte (1 point)
    if last['price_trend'] == 1 and last['candle_body'] > 0:
        call_points += 1
    
    # ===== ANALYSE PUT MAX QUALITY =====
    
    put_points = 0
    
    # 1. EMA alignées et écarté
    if last['ema_5'] < last['ema_13'] and last['ema_spread'] > 0.001:
        put_points += 3
    
    # 2. RSI optimal (33-42)
    if config['rsi_oversold'] < last['rsi_7'] < 42:
        put_points += 2
    
    # 3. Stochastique directionnel mais pas oversold
    if last['stoch_k'] < last['stoch_d'] and last['stoch_k'] > config['stoch_oversold']:
        put_points += 2
    
    # 4. ADX -DI fort
    if last['adx_neg'] > last['adx_pos'] and last['adx_neg'] > 28:
        put_points += 2
    
    # 5. Bougie baissière forte
    if last['price_trend'] == 0 and last['candle_body'] < 0:
        put_points += 1
    
    # ===== DÉCISION STRICTE =====
    
    min_points = config['min_confluence_points']
    
    if call_points >= min_points and call_points > put_points:
        return 'CALL'
    if put_points >= min_points and put_points > call_points:
        return 'PUT'
    
    # Si égalité mais points élevés, choisir basé sur RSI
    if call_points == put_points and call_points >= min_points:
        if last['rsi_7'] > 50:
            return 'CALL'
        else:
            return 'PUT'
    
    return None

def rule_signal_high_quality(df):
    """QUALITÉ ÉLEVÉE - Pour extension à 8 signaux"""
    if len(df) < 20:
        return None
    
    config = SAINT_GRAAL_CONFIG['high_quality']
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Filtres toujours stricts
    if last['data_quality'] < 0.8:
        return None
    if last['adx'] < config['adx_min']:
        return None
    
    # Conditions simplifiées mais robustes
    call_conditions = [
        last['ema_5'] > last['ema_13'],
        last['rsi_7'] > 52,
        last['stoch_k'] > last['stoch_d'],
        last['adx_pos'] > last['adx_neg'],
        last['close'] > prev['close'],
        last['body_ratio'] > config['min_body_ratio'],
        last['wick_ratio'] < config['max_wick_ratio'],
    ]
    
    put_conditions = [
        last['ema_5'] < last['ema_13'],
        last['rsi_7'] < 48,
        last['stoch_k'] < last['stoch_d'],
        last['adx_neg'] > last['adx_pos'],
        last['close'] < prev['close'],
        last['body_ratio'] > config['min_body_ratio'],
        last['wick_ratio'] < config['max_wick_ratio'],
    ]
    
    call_score = sum(call_conditions)
    put_score = sum(put_conditions)
    
    min_conditions = config['min_confluence_points']
    
    if call_score >= min_conditions and call_score > put_score:
        return 'CALL'
    if put_score >= min_conditions and put_score > call_score:
        return 'PUT'
    
    return None

# ================= FONCTIONS DE QUALITÉ =================

def calculate_signal_quality(df, direction):
    """Calcule un score de qualité (0-100) très strict"""
    if len(df) < 20:
        return 0
    
    last = df.iloc[-1]
    score = 0
    
    # 1. CONVERGENCE (40 points max)
    convergence = last['convergence_raw']
    if convergence == 5:
        score += 40
    elif convergence == 4:
        score += 30
    elif convergence == 3:
        score += 20
    
    # 2. FORCE DE TENDANCE (30 points)
    if last['adx'] > 30:
        score += 30
    elif last['adx'] > 25:
        score += 25
    elif last['adx'] > 20:
        score += 20
    
    # 3. QUALITÉ BOUGIE (20 points)
    if last['body_ratio'] > 0.5:
        score += 15
    elif last['body_ratio'] > 0.4:
        score += 10
    elif last['body_ratio'] > 0.3:
        score += 5
    
    if last['wick_ratio'] < 0.3:
        score += 5
    
    # 4. MOMENTUM (10 points)
    if direction == 'CALL':
        if 60 < last['rsi_7'] < 65:
            score += 10
        elif 55 < last['rsi_7'] < 70:
            score += 7
        elif last['rsi_7'] > 50:
            score += 4
    else:  # PUT
        if 35 < last['rsi_7'] < 40:
            score += 10
        elif 30 < last['rsi_7'] < 45:
            score += 7
        elif last['rsi_7'] < 50:
            score += 4
    
    return min(score, 100)

def format_signal_reason(direction, quality_score, indicators):
    """Formate la raison du signal"""
    last = indicators.iloc[-1]
    
    reason_parts = [f"🎯 {direction} - QUALITÉ MAX"]
    
    # Évaluation qualité
    if quality_score >= 90:
        reason_parts.append("⭐⭐⭐⭐⭐")
    elif quality_score >= 85:
        reason_parts.append("⭐⭐⭐⭐")
    elif quality_score >= 80:
        reason_parts.append("⭐⭐⭐")
    elif quality_score >= 75:
        reason_parts.append("⭐⭐")
    else:
        reason_parts.append("⭐")
    
    # Indicateurs clés
    reason_parts.append(f"RSI7: {last['rsi_7']:.1f}")
    reason_parts.append(f"ADX: {last['adx']:.1f}")
    
    # Convergence
    if last['convergence_raw'] >= 4:
        reason_parts.append("CONVERGENCE: EXCELLENTE")
    elif last['convergence_raw'] >= 3:
        reason_parts.append("CONVERGENCE: BONNE")
    
    return " | ".join(reason_parts)

# ================= FONCTIONS DE COMPATIBILITÉ =================

def compute_indicators(df, ema_fast=8, ema_slow=21, rsi_len=14, bb_len=20):
    """Wrapper pour compatibilité"""
    return compute_saint_graal_indicators(df)

def rule_signal(df):
    """Version par défaut (6 signaux qualité max)"""
    result = rule_signal_saint_graal_with_guarantee(df, signal_count=0, total_signals_needed=6)
    return result['signal'] if result else None

def get_signal_with_metadata(df, signal_count=0, total_signals=6):
    """Fonction principale pour le bot"""
    result = rule_signal_saint_graal_with_guarantee(df, signal_count, total_signals)
    
    if result:
        # Adapter total_signals si extension
        if signal_count >= 6 and result['mode'] == 'EXTENSION':
            total_signals = 8
        
        quality_text = {
            'EXCELLENT': 'EXCELLENTE',
            'HIGH': 'ÉLEVÉE',
            'EXTENSION': 'EXTENSION'
        }.get(result['quality'], 'STANDARD')
        
        return {
            'direction': result['signal'],
            'mode': result['mode'],
            'quality': result['quality'],
            'score': result['score'],
            'reason': format_signal_reason(
                result['signal'], 
                result['score'], 
                df
            ),
            'session_info': {
                'current_signal': signal_count + 1,
                'total_signals': total_signals,
                'target_quality': 'MAXIMALE'
            }
        }
    
    return None

# ================= FONCTIONS DE TIMING OPTIMAL =================

def is_optimal_trading_hour(hour_utc):
    """Heures de trading optimales (London/NY)"""
    # London Open (7-10 UTC) - Meilleure liquidité
    if 7 <= hour_utc < 10:
        return True, "London Open", 10
    
    # NY Open (13-16 UTC) - Volatilité élevée
    if 13 <= hour_utc < 16:
        return True, "NY Open", 9
    
    # Overlap London/NY (10-12 UTC)
    if 10 <= hour_utc < 12:
        return True, "London/NY Overlap", 8
    
    # Asia Close (1-4 UTC) - Mouvements techniques
    if 1 <= hour_utc < 4:
        return True, "Asia Close", 6
    
    return False, "Heure non optimale", 3

def check_timing_quality(current_minute, current_second):
    """Vérifie la qualité du timing"""
    # Éviter début/fin de bougie
    if current_second < 10:
        return False, f"Début bougie ({current_second}s)"
    
    if current_second > 50:
        return False, f"Fin bougie ({current_second}s)"
    
    # Préférer milieu de bougie
    if 20 <= current_second <= 40:
        return True, "Timing optimal"
    
    return True, "Timing acceptable"
