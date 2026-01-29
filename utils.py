"""
utils.py - STRATÉGIE FOREX M1 - 8 SIGNAUX QUALITÉ MAXIMALE AVEC GARANTIE
Version optimisée pour 8 signaux haute qualité avec garantie
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIGURATION 8 SIGNAUX QUALITÉ MAX =================

SAINT_GRAAL_CONFIG = {
    # Indicateurs M1 optimisés
    'rsi_period': 7,
    'ema_fast': 5,
    'ema_slow': 13,
    'stoch_period': 5,
    
    # Seuils QUALITÉ MAXIMALE POUR 8 SIGNAUX
    'max_quality': {
        'rsi_overbought': 68,
        'rsi_oversold': 32,
        'adx_min': 25,
        'stoch_overbought': 74,
        'stoch_oversold': 26,
        'min_confluence_points': 7,  # 7/10 points minimum pour garantir 8 signaux
        'min_body_ratio': 0.4,
        'max_wick_ratio': 0.4,
        'min_quality_score': 80,  # Score minimum pour qualité maximale
    },
    
    # Seuils QUALITÉ ÉLEVÉE (pour compléter les 8 signaux si nécessaire)
    'high_quality': {
        'rsi_overbought': 72,
        'rsi_oversold': 28,
        'adx_min': 22,
        'stoch_overbought': 78,
        'stoch_oversold': 22,
        'min_confluence_points': 6,  # 6/10 points pour qualité élevée
        'min_body_ratio': 0.35,
        'max_wick_ratio': 0.5,
        'min_quality_score': 70,
    },
    
    # Mode GARANTIE (dernier recours pour les 8 signaux)
    'guarantee_mode': {
        'rsi_overbought': 75,
        'rsi_oversold': 25,
        'adx_min': 20,
        'stoch_overbought': 80,
        'stoch_oversold': 20,
        'min_confluence_points': 5,  # 5/10 points minimum en garantie
        'min_body_ratio': 0.3,
        'max_wick_ratio': 0.6,
        'min_quality_score': 60,
    },
    
    # Filtres anti-manipulation renforcés
    'anti_manip': {
        'max_wick_ratio': 0.65,
        'max_candle_size_ratio': 2.5,
        'min_ema_spread': 0.0006,
        'max_volatility': 0.04,
        'min_data_quality': 0.8,
    },
    
    # Paramètres généraux
    'target_signals': 8,  # Objectif GARANTI de 8 signaux
    'max_signals': 8,     # Maximum = objectif
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
    """Retourne le range de la bougie M1 actuelle"""
    current_candle = round_to_m1_candle(dt)
    start_time = current_candle
    end_time = current_candle + timedelta(minutes=1)
    return start_time, end_time

# ================= INDICATEURS QUALITÉ MAX =================

def compute_saint_graal_indicators(df):
    """
    Calcule les indicateurs pour qualité maximale - Version 8 signaux
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

# ================= FONCTION MANQUANTE calculate_signal_quality_score =================

def calculate_signal_quality_score(df):
    """
    Calcule un score de qualité global du signal (0-100)
    Compatibilité avec l'ancien code
    """
    if len(df) < 20:
        return 0
    
    last = df.iloc[-1]
    score = 0
    
    # Convergence (30 points max)
    convergence = last.get('convergence_score', 0.5)
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
    if last.get('ema_trend', 0) == 1:
        aligned_indicators += 1
    if last.get('rsi_trend', 0) == 1:
        aligned_indicators += 1
    if last.get('stoch_trend', 0) == 1:
        aligned_indicators += 1
    if last.get('adx_trend', 0) == 1:
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
    
    # Qualité de la bougie (10 points)
    body_ratio = last.get('body_ratio', 0)
    wick_ratio = last.get('wick_ratio', 0)
    
    if body_ratio > 0.4 and wick_ratio < 0.3:
        score += 10
    elif body_ratio > 0.3 and wick_ratio < 0.4:
        score += 5
    
    return min(score, 100)

# ================= FILTRES ANTI-MANIPULATION RENFORCÉS =================

def check_anti_manipulation(df, strict_mode=True):
    """Vérifie les conditions anti-manipulation renforcées"""
    if len(df) < 15:
        return False, "Données insuffisantes"
    
    last = df.iloc[-1]
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

# ================= STRATÉGIE 8 SIGNAUX QUALITÉ MAX =================

def rule_signal_saint_graal_with_guarantee(df, signal_count=0, total_signals_needed=8):
    """
    STRATÉGIE - 8 signaux qualité maximale avec GARANTIE
    
    Nouvelle logique pour 8 signaux garantis :
    1. Essayer QUALITÉ MAXIMALE (signaux 1-6)
    2. Si échec après 3 tentatives, passer à QUALITÉ ÉLEVÉE
    3. En dernier recours, utiliser MODE GARANTIE
    4. Résultat : TOUJOURS un signal pour compléter les 8
    """
    
    if len(df) < 30:
        print("[STRATEGIE] ⚠️ Données insuffisantes")
        return None
    
    target_signals = SAINT_GRAAL_CONFIG['target_signals']
    max_signals = SAINT_GRAAL_CONFIG['max_signals']
    
    print(f"\n[STRATEGIE] 🎯 Signal #{signal_count+1}/{target_signals}")
    
    # ===== 1. QUALITÉ MAXIMALE (signaux 1-6 préférés) =====
    if signal_count < 6:
        print(f"[STRATEGIE] 🔵 Mode QUALITÉ MAXIMALE")
        
        # Vérifier anti-manipulation strict
        manip_ok, manip_reason = check_anti_manipulation(df, strict_mode=True)
        if not manip_ok:
            print(f"[STRATEGIE] ⚠️ Anti-manip: {manip_reason}")
            # Pas d'échec immédiat, on essaie quand même l'analyse
            
        max_quality_signal = rule_signal_max_quality(df)
        if max_quality_signal:
            quality_score = calculate_signal_quality(df, max_quality_signal)
            min_score = SAINT_GRAAL_CONFIG['max_quality']['min_quality_score']
            
            if quality_score >= min_score:
                print(f"[STRATEGIE] ✅ Signal QUALITÉ MAX #{signal_count+1} | Score: {quality_score}/100")
                return {
                    'signal': max_quality_signal,
                    'mode': 'MAX_QUALITY',
                    'quality': 'EXCELLENT',
                    'score': quality_score
                }
            else:
                print(f"[STRATEGIE] ⚠️ Qualité insuffisante: {quality_score}/{min_score}")
                # Continuer vers QUALITÉ ÉLEVÉE
    
    # ===== 2. QUALITÉ ÉLEVÉE (signaux 7-8 ou backup) =====
    print(f"[STRATEGIE] 🟡 Mode QUALITÉ ÉLEVÉE")
    
    # Anti-manipulation un peu moins strict
    manip_ok, manip_reason = check_anti_manipulation(df, strict_mode=False)
    if not manip_ok:
        print(f"[STRATEGIE] ⚠️ Anti-manip (toléré): {manip_reason}")
        # En mode qualité élevée, on tolère plus
    
    high_quality_signal = rule_signal_high_quality(df)
    if high_quality_signal:
        quality_score = calculate_signal_quality_for_mode(df, high_quality_signal, 'HIGH_QUALITY')
        min_score = SAINT_GRAAL_CONFIG['high_quality']['min_quality_score']
        
        if quality_score >= min_score:
            print(f"[STRATEGIE] ✅ Signal QUALITÉ ÉLEVÉE #{signal_count+1} | Score: {quality_score}/100")
            return {
                'signal': high_quality_signal,
                'mode': 'HIGH_QUALITY',
                'quality': 'HIGH',
                'score': quality_score
            }
        else:
            print(f"[STRATEGIE] ⚠️ Qualité élevée insuffisante: {quality_score}/{min_score}")
            # Continuer vers MODE GARANTIE
    
    # ===== 3. MODE GARANTIE (dernier recours pour les 8 signaux) =====
    print(f"[STRATEGIE] 🟠 Mode GARANTIE (dernier recours)")
    
    # Filtres minimaux seulement
    guarantee_signal = rule_signal_guarantee_mode(df)
    if guarantee_signal:
        quality_score = calculate_signal_quality_for_mode(df, guarantee_signal, 'GUARANTEE')
        min_score = SAINT_GRAAL_CONFIG['guarantee_mode']['min_quality_score']
        
        if quality_score >= min_score:
            print(f"[STRATEGIE] ✅ Signal GARANTIE #{signal_count+1} | Score: {quality_score}/100")
            return {
                'signal': guarantee_signal,
                'mode': 'GUARANTEE',
                'quality': 'ACCEPTABLE',
                'score': quality_score
            }
    
    # ===== 4. FORCÉ (absolument aucun signal trouvé) =====
    print(f"[STRATEGIE] ⚡ Mode FORCÉ pour garantir le signal #{signal_count+1}")
    
    # Forcer un signal basé sur les indicateurs simples
    forced_signal = force_signal_based_on_indicators(df)
    if forced_signal:
        print(f"[STRATEGIE] ✅ Signal FORCÉ #{signal_count+1} (garantie des 8 signaux)")
        return {
            'signal': forced_signal,
            'mode': 'FORCED',
            'quality': 'MINIMUM',
            'score': 50  # Score minimum
        }
    
    # Si même le forçage échoue (très improbable)
    print(f"[STRATEGIE] ❌ ERREUR CRITIQUE: Impossible de générer un signal même en mode forcé")
    return None

def rule_signal_max_quality(df):
    """QUALITÉ MAXIMALE - Filtres très stricts"""
    if len(df) < 25:
        return None
    
    config = SAINT_GRAAL_CONFIG['max_quality']
    last = df.iloc[-1]
    
    # ===== FILTRES ABSOLUS =====
    
    # 1. Qualité données exceptionnelle
    if last['data_quality'] < 0.85:
        return None
    
    # 2. ADX fort (tendance claire)
    if last['adx'] < config['adx_min']:
        return None
    
    # 3. Convergence forte (3+/5 indicateurs alignés)
    if last['convergence_raw'] < 3:
        return None
    
    # ===== ANALYSE CALL MAX QUALITY =====
    
    call_points = 0
    max_points = 10
    
    # 1. EMA alignées et écarté (3 points)
    if last['ema_5'] > last['ema_13'] and last['ema_spread'] > 0.001:
        call_points += 3
    elif last['ema_5'] > last['ema_13']:
        call_points += 2
    
    # 2. RSI optimal (58-68) (2 points)
    if 58 < last['rsi_7'] < config['rsi_overbought']:
        call_points += 2
    elif 55 < last['rsi_7'] < 70:
        call_points += 1
    
    # 3. Stochastique directionnel (2 points)
    if last['stoch_k'] > last['stoch_d'] and last['stoch_k'] < config['stoch_overbought']:
        call_points += 2
    elif last['stoch_k'] > last['stoch_d']:
        call_points += 1
    
    # 4. ADX +DI fort (2 points)
    if last['adx_pos'] > last['adx_neg'] and last['adx_pos'] > 25:
        call_points += 2
    elif last['adx_pos'] > last['adx_neg']:
        call_points += 1
    
    # 5. Bougie haussière (1 point)
    if last['price_trend'] == 1:
        call_points += 1
    
    # 6. Bonne formation bougie (points bonus)
    if last['body_ratio'] > config['min_body_ratio']:
        call_points += 0.5
    if last['wick_ratio'] < config['max_wick_ratio']:
        call_points += 0.5
    
    # ===== ANALYSE PUT MAX QUALITY =====
    
    put_points = 0
    
    # 1. EMA alignées et écarté
    if last['ema_5'] < last['ema_13'] and last['ema_spread'] > 0.001:
        put_points += 3
    elif last['ema_5'] < last['ema_13']:
        put_points += 2
    
    # 2. RSI optimal (32-42)
    if config['rsi_oversold'] < last['rsi_7'] < 42:
        put_points += 2
    elif 30 < last['rsi_7'] < 45:
        put_points += 1
    
    # 3. Stochastique directionnel
    if last['stoch_k'] < last['stoch_d'] and last['stoch_k'] > config['stoch_oversold']:
        put_points += 2
    elif last['stoch_k'] < last['stoch_d']:
        put_points += 1
    
    # 4. ADX -DI fort
    if last['adx_neg'] > last['adx_pos'] and last['adx_neg'] > 25:
        put_points += 2
    elif last['adx_neg'] > last['adx_pos']:
        put_points += 1
    
    # 5. Bougie baissière
    if last['price_trend'] == 0:
        put_points += 1
    
    # 6. Bonne formation bougie
    if last['body_ratio'] > config['min_body_ratio']:
        put_points += 0.5
    if last['wick_ratio'] < config['max_wick_ratio']:
        put_points += 0.5
    
    # ===== DÉCISION STRICTE =====
    
    min_points = config['min_confluence_points']
    
    if call_points >= min_points and call_points > put_points:
        return 'CALL'
    if put_points >= min_points and put_points > call_points:
        return 'PUT'
    
    # Si égalité mais points élevés, choisir basé sur RSI
    if call_points == put_points and call_points >= min_points - 1:
        if last['rsi_7'] > 50:
            return 'CALL'
        else:
            return 'PUT'
    
    return None

def rule_signal_high_quality(df):
    """QUALITÉ ÉLEVÉE - Pour compléter les 8 signaux"""
    if len(df) < 20:
        return None
    
    config = SAINT_GRAAL_CONFIG['high_quality']
    last = df.iloc[-1]
    
    # Filtres moins stricts
    if last['data_quality'] < 0.75:
        return None
    if last['adx'] < config['adx_min']:
        return None
    
    # Conditions simplifiées mais robustes
    call_conditions = [
        last['ema_5'] > last['ema_13'],
        last['rsi_7'] > 52,
        last['stoch_k'] > last['stoch_d'],
        last['adx_pos'] > last['adx_neg'],
        last['price_trend'] == 1,
        last['body_ratio'] > config['min_body_ratio'],
        last['wick_ratio'] < config['max_wick_ratio'],
    ]
    
    put_conditions = [
        last['ema_5'] < last['ema_13'],
        last['rsi_7'] < 48,
        last['stoch_k'] < last['stoch_d'],
        last['adx_neg'] > last['adx_pos'],
        last['price_trend'] == 0,
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

def rule_signal_guarantee_mode(df):
    """MODE GARANTIE - Dernier recours pour les 8 signaux"""
    if len(df) < 15:
        return None
    
    config = SAINT_GRAAL_CONFIG['guarantee_mode']
    last = df.iloc[-1]
    
    # Filtres minimaux
    if last['data_quality'] < 0.65:
        return None
    
    # Conditions très simples
    call_conditions = [
        last['ema_5'] > last['ema_13'],
        last['rsi_7'] > 50,
        last['stoch_k'] > last['stoch_d'],
        last['price_trend'] == 1,
    ]
    
    put_conditions = [
        last['ema_5'] < last['ema_13'],
        last['rsi_7'] < 50,
        last['stoch_k'] < last['stoch_d'],
        last['price_trend'] == 0,
    ]
    
    call_score = sum(call_conditions)
    put_score = sum(put_conditions)
    
    min_conditions = config['min_confluence_points']
    
    if call_score >= min_conditions:
        return 'CALL'
    if put_score >= min_conditions:
        return 'PUT'
    
    # Si aucune condition n'est remplie, choisir basé sur RSI
    if last['rsi_7'] > 50:
        return 'CALL'
    else:
        return 'PUT'

def force_signal_based_on_indicators(df):
    """Force un signal basé sur les indicateurs les plus simples"""
    if len(df) < 10:
        return None
    
    last = df.iloc[-1]
    
    # Règle simple : RSI + EMA
    if last['rsi_7'] > 50 and last['ema_5'] > last['ema_13']:
        return 'CALL'
    elif last['rsi_7'] < 50 and last['ema_5'] < last['ema_13']:
        return 'PUT'
    
    # Fallback : RSI seul
    if last['rsi_7'] > 50:
        return 'CALL'
    else:
        return 'PUT'

# ================= FONCTIONS DE QUALITÉ =================

def calculate_signal_quality(df, direction):
    """Calcule un score de qualité (0-100) très strict"""
    if len(df) < 20:
        return 0
    
    last = df.iloc[-1]
    score = 0
    
    # 1. CONVERGENCE (40 points max)
    convergence = last['convergence_raw']
    if convergence >= 4:
        score += 40
    elif convergence >= 3:
        score += 30
    elif convergence >= 2:
        score += 20
    else:
        score += 10
    
    # 2. FORCE DE TENDANCE (30 points)
    if last['adx'] > 30:
        score += 30
    elif last['adx'] > 25:
        score += 25
    elif last['adx'] > 20:
        score += 20
    elif last['adx'] > 15:
        score += 15
    else:
        score += 5
    
    # 3. QUALITÉ BOUGIE (20 points)
    if last['body_ratio'] > 0.5:
        score += 15
    elif last['body_ratio'] > 0.4:
        score += 10
    elif last['body_ratio'] > 0.3:
        score += 5
    
    if last['wick_ratio'] < 0.3:
        score += 5
    elif last['wick_ratio'] < 0.4:
        score += 3
    elif last['wick_ratio'] < 0.5:
        score += 1
    
    # 4. ALIGNEMENT INDICATEURS (10 points)
    aligned = 0
    if direction == 'CALL':
        if last['ema_5'] > last['ema_13']:
            aligned += 1
        if last['rsi_trend'] == 1:
            aligned += 1
        if last['stoch_trend'] == 1:
            aligned += 1
        if last['adx_trend'] == 1:
            aligned += 1
    else:  # PUT
        if last['ema_5'] < last['ema_13']:
            aligned += 1
        if last['rsi_trend'] == 0:
            aligned += 1
        if last['stoch_trend'] == 0:
            aligned += 1
        if last['adx_trend'] == 0:
            aligned += 1
    
    score += (aligned / 4) * 10
    
    return min(score, 100)

def calculate_signal_quality_for_mode(df, direction, mode):
    """Calcule le score de qualité adapté au mode"""
    last = df.iloc[-1]
    score = 0
    
    # Base score par mode
    base_scores = {
        'MAX_QUALITY': 80,
        'HIGH_QUALITY': 70,
        'GUARANTEE': 60
    }
    
    score = base_scores.get(mode, 60)
    
    # Bonus selon les indicateurs
    if direction == 'CALL':
        if last['ema_5'] > last['ema_13']:
            score += 5
        if last['rsi_7'] > 55:
            score += 5
        if last['stoch_k'] > last['stoch_d']:
            score += 5
        if last['adx_pos'] > last['adx_neg']:
            score += 5
    else:
        if last['ema_5'] < last['ema_13']:
            score += 5
        if last['rsi_7'] < 45:
            score += 5
        if last['stoch_k'] < last['stoch_d']:
            score += 5
        if last['adx_neg'] > last['adx_pos']:
            score += 5
    
    # Bonus convergence
    if last['convergence_raw'] >= 3:
        score += 5
    
    # Bonus qualité bougie
    if last['body_ratio'] > 0.3:
        score += 5
    
    return min(score, 100)

def format_signal_reason(direction, confidence, indicators):
    """Formate la raison du signal - Compatibilité avec ancien code"""
    last = indicators.iloc[-1]
    
    # Utiliser le score de qualité global
    quality_score = calculate_signal_quality_score(indicators)
    
    reason_parts = [f"🎯 {direction}"]
    
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
        reason_parts.append("CONV: EXCELLENTE")
    elif last['convergence_raw'] >= 3:
        reason_parts.append("CONV: BONNE")
    
    # Confiance
    reason_parts.append(f"CONF: {int(confidence)}%")
    
    return " | ".join(reason_parts)

# ================= FONCTIONS DE COMPATIBILITÉ =================

def compute_indicators(df, ema_fast=8, ema_slow=21, rsi_len=14, bb_len=20):
    """Wrapper pour compatibilité"""
    return compute_saint_graal_indicators(df)

def rule_signal(df):
    """Version par défaut (8 signaux qualité max)"""
    result = rule_signal_saint_graal_with_guarantee(df, signal_count=0, total_signals_needed=8)
    return result['signal'] if result else None

def get_signal_with_metadata(df, signal_count=0, total_signals=8):
    """Fonction principale pour le bot - RETOURNE TOUJOURS UN SIGNAL"""
    result = rule_signal_saint_graal_with_guarantee(df, signal_count, total_signals)
    
    if result:
        mode_display = {
            'MAX_QUALITY': '🔵 STRICT',
            'HIGH_QUALITY': '🟡 HIGH',
            'GUARANTEE': '🟠 GARANTIE',
            'FORCED': '⚡ FORCED'
        }.get(result['mode'], '⚪ STANDARD')
        
        quality_text = {
            'EXCELLENT': 'EXCELLENTE',
            'HIGH': 'ÉLEVÉE',
            'ACCEPTABLE': 'ACCEPTABLE',
            'MINIMUM': 'MINIMUM'
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
                'mode_used': result['mode'],
                'quality_level': result['quality']
            }
        }
    
    # Cette ligne ne devrait JAMAIS être atteinte grâce au mode FORCED
    print(f"[STRATEGIE] ⚠️ ERREUR: Aucun signal retourné même après tous les modes")
    
    # Fallback absolu
    last = df.iloc[-1] if len(df) > 0 else None
    if last is not None:
        forced_direction = 'CALL' if last.get('rsi_7', 50) > 50 else 'PUT'
    else:
        forced_direction = 'CALL'  # Default
    
    return {
        'direction': forced_direction,
        'mode': 'FALLBACK',
        'quality': 'MINIMUM',
        'score': 50,
        'reason': f"FALLBACK {forced_direction} - Aucun signal trouvé",
        'session_info': {
            'current_signal': signal_count + 1,
            'total_signals': total_signals,
            'mode_used': 'FALLBACK',
            'quality_level': 'MINIMUM'
        }
    }

# ================= FONCTIONS DE TIMING OPTIMAL =================

def is_kill_zone_optimal(hour_utc):
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
