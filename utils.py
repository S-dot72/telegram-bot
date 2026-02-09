"""
🚀 STRATÉGIE BINAIRE M1 PRO - VERSION 9.4 PRODUCTION
🔥 OPTIMISÉ POUR GÉNÉRATION DE SIGNAUX
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIGURATION OPTIMISÉE =================

SAINT_GRAAL_CONFIG = {
    'expiration_minutes': 5,
    
    # 🔥 ZONES D'INTERDICTION ÉQUILIBRÉES
    'forbidden_zones': {
        'no_buy_zone': {
            'enabled': True,
            'stoch_fast_max': 80,
            'rsi_max': 65,
            'bb_position_max': 75,
            'strict_mode': False,  # SOFT VETO
            'penalty': 10,
        },
        'no_sell_zone': {
            'enabled': True,
            'stoch_fast_min': 20,
            'rsi_min': 35,
            'bb_position_min': 25,
            'strict_mode': False,  # SOFT VETO
            'penalty': 10,
        },
        'swing_filter': {
            'enabled': False,  # DÉSACTIVÉ
            'lookback_bars': 8,
            'no_buy_at_swing_high': False,
            'no_sell_at_swing_low': False,
            'strict_mode': False,
            'swing_penalty': 0,
            'swing_momentum_threshold': 0,
        }
    },
    
    # 🔥 MOMENTUM GATE OPTIMISÉ
    'momentum_rules': {
        'buy_conditions': {
            'rsi_max': 58,
            'rsi_oversold': 28,
            'stoch_max': 40,
            'stoch_oversold': 18,
            'require_stoch_rising': True,
        },
        'sell_conditions': {
            'rsi_min': 48,
            'rsi_overbought': 72,
            'stoch_min': 60,
            'stoch_overbought': 80,
            'require_stoch_falling': True,
        },
        'momentum_gate_diff': 5,
        'smart_gate': True,
    },
    
    'micro_momentum': {
        'enabled': True,
        'lookback_bars': 3,
        'min_bullish_bars': 1,  # Réduit
        'min_bearish_bars': 1,  # Réduit
        'require_trend_alignment': False,
        'weight': 8,
    },
    
    'bollinger_config': {
        'window': 20,
        'window_dev': 2,
        'oversold_zone': 20,
        'overbought_zone': 80,
        'buy_zone_max': 50,
        'sell_zone_min': 50,
        'middle_band_weight': 10,
        'strict_mode': False,
        'penalty': 5,
    },
    
    'atr_filter': {
        'enabled': False,  # DÉSACTIVÉ
        'window': 14,
        'min_atr_pips': 1,
        'max_atr_pips': 100,
        'optimal_range': [1, 100],
    },
    
    # 🔥 M5 SIMPLIFIÉ
    'm5_filter': {
        'enabled': False,  # DÉSACTIVÉ
        'ema_fast': 50,
        'ema_slow': 200,
        'weight': 0,
        'soft_veto': False,
        'max_score_against_trend': 100,
    },
    
    # 🔥 ÉTAT DE MARCHÉ SIMPLIFIÉ
    'market_state': {
        'enabled': False,  # DÉSACTIVÉ
        'adx_threshold': 25,
        'rsi_range_threshold': 45,
        'prioritize_bb_in_range': True,
        'prioritize_momentum_in_trend': True,
    },
    
    'signal_validation': {
        'min_score': 70,           # ← CRITIQUE : 70 au lieu de 85
        'max_score_realistic': 120,
        'confidence_zones': {
            70: 65,    # MINIMUM
            80: 72,    # SOLID
            90: 78,    # GOOD
            100: 85,   # HIGH
            110: 90,   # EXCELLENT
            120: 92,   # PREMIUM
        },
        'cooldown_bars': 2,
    },
    
    # 🔥 COOLDOWN SIMPLIFIÉ
    'risk_management': {
        'dynamic_cooldown': False,
        'normal_cooldown': 2,
        'cooldown_by_quality': {
            'EXCELLENT': 1,
            'HIGH': 2,
            'SOLID': 3,
            'MINIMUM': 4,
        },
        'max_daily_trades': 30,
        'max_consecutive_losses': 5,
    }
}

# ================= ÉTAT DU TRADING =================

class TradingState:
    """Gestion simplifiée de l'état"""
    def __init__(self):
        self.last_trade_time = None
        self.consecutive_losses = 0
        self.daily_trades = 0
        
    def can_trade(self, current_time):
        """Toujours autorisé en mode simplifié"""
        return True, "OK"

trading_state = TradingState()

# ================= FONCTIONS OPTIMISÉES =================

def analyze_momentum_with_filters(df):
    """Analyse momentum optimisée"""
    if len(df) < 20:
        return {
            'rsi': 50,
            'stoch_k': 50,
            'stoch_d': 50,
            'prev_rsi': 50,
            'buy': {'allowed': True, 'veto': False, 'score': 25, 'penalty': 0, 'reason': 'Données limitées'},
            'sell': {'allowed': True, 'veto': False, 'score': 25, 'penalty': 0, 'reason': 'Données limitées'},
            'gate_buy': True,
            'gate_sell': True,
            'violations': []
        }
    
    try:
        # RSI
        rsi = RSIIndicator(close=df['close'], window=14).rsi()
        current_rsi = float(rsi.iloc[-1])
        
        # Stochastique
        stoch = StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14,
            smooth_window=3
        )
        stoch_k = float(stoch.stoch().iloc[-1])
        stoch_d = float(stoch.stoch_signal().iloc[-1])
        
    except:
        current_rsi = 50
        stoch_k = 50
        stoch_d = 50
    
    # Scores de base
    buy_score = 25
    sell_score = 25
    
    # Score RSI
    if current_rsi < SAINT_GRAAL_CONFIG['momentum_rules']['buy_conditions']['rsi_max']:
        buy_score += 15
        if current_rsi < SAINT_GRAAL_CONFIG['momentum_rules']['buy_conditions']['rsi_oversold']:
            buy_score += 10
    
    if current_rsi > SAINT_GRAAL_CONFIG['momentum_rules']['sell_conditions']['rsi_min']:
        sell_score += 15
        if current_rsi > SAINT_GRAAL_CONFIG['momentum_rules']['sell_conditions']['rsi_overbought']:
            sell_score += 10
    
    # Score Stochastique
    if stoch_k < SAINT_GRAAL_CONFIG['momentum_rules']['buy_conditions']['stoch_max']:
        buy_score += 12
        if stoch_k < SAINT_GRAAL_CONFIG['momentum_rules']['buy_conditions']['stoch_oversold']:
            buy_score += 8
    
    if stoch_k > SAINT_GRAAL_CONFIG['momentum_rules']['sell_conditions']['stoch_min']:
        sell_score += 12
        if stoch_k > SAINT_GRAAL_CONFIG['momentum_rules']['sell_conditions']['stoch_overbought']:
            sell_score += 8
    
    # Gate simplifié
    gate_buy = stoch_k < 50 and current_rsi < 55
    gate_sell = stoch_k > 50 and current_rsi > 45
    
    return {
        'rsi': current_rsi,
        'stoch_k': stoch_k,
        'stoch_d': stoch_d,
        'prev_rsi': current_rsi,
        'buy': {'allowed': True, 'veto': False, 'score': buy_score, 'penalty': 0, 'reason': f'RSI:{current_rsi:.1f}'},
        'sell': {'allowed': True, 'veto': False, 'score': sell_score, 'penalty': 0, 'reason': f'RSI:{current_rsi:.1f}'},
        'gate_buy': gate_buy,
        'gate_sell': gate_sell,
        'violations': []
    }

def analyze_bollinger_bands(df):
    """Analyse BB optimisée"""
    if len(df) < 20:
        return {
            'bb_position': 50,
            'buy': {'allowed': True, 'veto': False, 'score': 20, 'penalty': 0, 'reason': 'Données limitées'},
            'sell': {'allowed': True, 'veto': False, 'score': 20, 'penalty': 0, 'reason': 'Données limitées'}
        }
    
    try:
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        current_price = float(df.iloc[-1]['close'])
        current_upper = float(bb.bollinger_hband().iloc[-1])
        current_lower = float(bb.bollinger_lband().iloc[-1])
        
        if current_upper != current_lower:
            bb_position = ((current_price - current_lower) / (current_upper - current_lower)) * 100
        else:
            bb_position = 50
    except:
        bb_position = 50
    
    # Scores
    buy_score = 20
    sell_score = 20
    
    if bb_position < SAINT_GRAAL_CONFIG['bollinger_config']['buy_zone_max']:
        buy_score += 15
        if bb_position < SAINT_GRAAL_CONFIG['bollinger_config']['oversold_zone']:
            buy_score += 10
    
    if bb_position > SAINT_GRAAL_CONFIG['bollinger_config']['sell_zone_min']:
        sell_score += 15
        if bb_position > SAINT_GRAAL_CONFIG['bollinger_config']['overbought_zone']:
            sell_score += 10
    
    return {
        'bb_position': bb_position,
        'buy': {'allowed': True, 'veto': False, 'score': buy_score, 'penalty': 0, 'reason': f'BB:{bb_position:.1f}%'},
        'sell': {'allowed': True, 'veto': False, 'score': sell_score, 'penalty': 0, 'reason': f'BB:{bb_position:.1f}%'}
    }

def analyze_micro_momentum(df, direction):
    """Micro momentum simplifié"""
    if len(df) < 4:
        return {'valid': True, 'score': 5, 'reason': 'Données limitées'}
    
    try:
        closes = df['close'].values[-3:]
        if direction == "BUY":
            if closes[-1] > closes[-2]:
                return {'valid': True, 'score': SAINT_GRAAL_CONFIG['micro_momentum']['weight'], 'reason': 'Dernière bougie verte'}
        else:
            if closes[-1] < closes[-2]:
                return {'valid': True, 'score': SAINT_GRAAL_CONFIG['micro_momentum']['weight'], 'reason': 'Dernière bougie rouge'}
    except:
        pass
    
    return {'valid': True, 'score': 5, 'reason': 'Neutre'}

def calculate_confidence(score):
    """Confiance simple"""
    if score >= 110:
        return 90
    elif score >= 100:
        return 85
    elif score >= 90:
        return 80
    elif score >= 80:
        return 75
    elif score >= 70:
        return 70
    else:
        return 65

# ================= FONCTION PRINCIPALE V9.4 =================

def analyze_pair_for_signals(df):
    """
    🔥 Analyse optimisée - VERSION 9.4
    """
    if df is None or len(df) < 30:
        return None
    
    current_price = float(df.iloc[-1]['close'])
    
    # 1. Momentum
    momentum = analyze_momentum_with_filters(df)
    # 2. Bollinger Bands
    bb = analyze_bollinger_bands(df)
    # 3. Micro momentum
    micro_buy = analyze_micro_momentum(df, "BUY")
    micro_sell = analyze_micro_momentum(df, "SELL")
    
    # Scores totaux
    buy_score_total = momentum['buy']['score'] + bb['buy']['score'] + micro_buy['score']
    sell_score_total = momentum['sell']['score'] + bb['sell']['score'] + micro_sell['score']
    
    min_score = SAINT_GRAAL_CONFIG['signal_validation']['min_score']
    
    # Décision
    if (not momentum['buy']['veto'] and not bb['buy']['veto'] and 
        momentum['gate_buy'] and buy_score_total >= min_score and 
        buy_score_total >= sell_score_total):
        
        final_score = buy_score_total
        direction = "CALL"
        reason = f"BUY Score: {final_score:.1f} | RSI: {momentum['rsi']:.1f} | BB: {bb['bb_position']:.1f}%"
        confidence = calculate_confidence(final_score)
        
        return {
            'direction': direction,
            'quality': "SOLID" if final_score >= 80 else "MINIMUM",
            'score': round(final_score, 1),
            'confidence': confidence,
            'expiration_minutes': 5,
            'reason': reason,
            'details': {
                'market_state': 'NEUTRAL',
                'momentum_score': momentum['buy']['score'],
                'bb_score': bb['buy']['score'],
                'micro_score': micro_buy['score'],
                'rsi': momentum['rsi'],
                'stoch': momentum['stoch_k'],
                'bb_position': bb['bb_position'],
            }
        }
    
    elif (not momentum['sell']['veto'] and not bb['sell']['veto'] and 
          momentum['gate_sell'] and sell_score_total >= min_score):
        
        final_score = sell_score_total
        direction = "PUT"
        reason = f"SELL Score: {final_score:.1f} | RSI: {momentum['rsi']:.1f} | BB: {bb['bb_position']:.1f}%"
        confidence = calculate_confidence(final_score)
        
        return {
            'direction': direction,
            'quality': "SOLID" if final_score >= 80 else "MINIMUM",
            'score': round(final_score, 1),
            'confidence': confidence,
            'expiration_minutes': 5,
            'reason': reason,
            'details': {
                'market_state': 'NEUTRAL',
                'momentum_score': momentum['sell']['score'],
                'bb_score': bb['sell']['score'],
                'micro_score': micro_sell['score'],
                'rsi': momentum['rsi'],
                'stoch': momentum['stoch_k'],
                'bb_position': bb['bb_position'],
            }
        }
    
    return None

def get_signal_saint_graal(df, signal_count=0, total_signals=8, return_dict=False):
    """
    🔥 Fonction principale pour le bot
    """
    signal = analyze_pair_for_signals(df)
    
    if signal:
        signal['signal_count'] = signal_count
        signal['total_signals'] = total_signals
        signal['mode'] = "V9.4"
        return signal
    
    return None

# Alias pour compatibilité
get_binary_signal = get_signal_saint_graal

# ================= INITIALISATION =================

if __name__ == "__main__":
    print("🚀 STRATÉGIE BINAIRE M1 PRO - VERSION 9.4")
    print("🔥 OPTIMISÉE POUR GÉNÉRATION DE SIGNAUX")
    print("\n" + "="*60)
    print("CONFIGURATION:")
    print("✅ Score minimum: 70")
    print("✅ Momentum + Bollinger + Micro")
    print("✅ Filtres simplifiés mais efficaces")
    print("✅ Compatible signal_bot.py")
    print("="*60)
