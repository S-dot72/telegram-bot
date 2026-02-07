"""
utils.py - STRATÉGIE BINAIRE M1 PRO - VERSION 4.5 ULTIMATE PLUS
Ajout: Micro garde-fou momentum + Filtre ATR
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIGURATION AVEC FILTRES AJOUTÉS =================

SAINT_GRAAL_CONFIG = {
    'expiration_minutes': 5,
    
    # 🔥 AJOUT: MICRO GARDE-FOU MOMENTUM
    'micro_momentum_filter': {
        'enabled': True,
        'lookback_bars': 3,           # Dernières 3 bougies M1
        'min_bullish_bars': 2,        # Minimum 2/3 bougies haussières pour BUY
        'min_bearish_bars': 2,        # Minimum 2/3 bougies baissières pour SELL
        'require_price_alignment': True,  # Prix doit suivre la direction
        'require_volume_confirmation': False,  # Optionnel selon les données
        'weight': 15,                 # Poids dans le score total
    },
    
    # 🔥 AJOUT: FILTRE ATR
    'atr_filter': {
        'enabled': True,
        'window': 14,                  # Période ATR standard
        'min_atr_pips': 2,            # Volatilité minimale requise (2 pips)
        'max_atr_pips': 25,           # Volatilité maximale autorisée (25 pips)
        'optimal_atr_pips': [5, 15],  # Zone optimale 5-15 pips
        'atr_trend_weight': 10,       # Bonus si ATR en hausse (momentum)
        'squeeze_detection': True,    # Détection de squeeze ATR
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
    
    'momentum_context': {
        'trend_overbought': 65,
        'trend_oversold': 35,
        'range_overbought': 72,
        'range_oversold': 28,
        'strong_trend_threshold': 1.2,
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
    },
}

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

# ================= FONCTION PRINCIPALE MISE À JOUR =================

def rule_signal_saint_graal_5min_pro_v3(df, signal_count=0, total_signals_needed=6):
    """
    🔥 VERSION 4.5 : AVEC MICRO MOMENTUM + FILTRE ATR
    """
    print(f"\n{'='*70}")
    print(f"🎯 BINAIRE 5 MIN V4.5 - Signal #{signal_count+1}/{total_signals_needed}")
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
    
    # ===== 4. BOLLINGER BANDS =====
    bb_signal = calculate_bollinger_signals(df)
    print(f"📊 BB: Position {bb_signal['bb_position']:.1f}% | Signal: {bb_signal['bb_signal']}")
    
    # ===== 5. 🔥 NOUVEAU: FILTRE ATR =====
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
        
        # 🔥 NOUVEAU: Vérification micro momentum
        micro_valid, micro_score, micro_reason = check_micro_momentum(df, "BUY")
        
        if not micro_valid:
            print(f"❌ Micro momentum BUY échoué: {micro_reason}")
        else:
            # Vérification alignement M5
            m5_aligned, m5_reason, m5_bonus = check_m5_alignment(m5_filter, "BUY")
            
            if m5_aligned or not SAINT_GRAAL_CONFIG['signal_config']['require_m5_alignment']:
                # Validation bougie
                candle_valid, pattern, pattern_conf, candle_reason = validate_candle_for_5min_buy(df)
                
                if candle_valid:
                    direction = "BUY"
                    final_score = buy_score_total + pattern_conf + m5_bonus + micro_score
                    decision_details.append(f"BUY validé: {pattern} ({pattern_conf}%)")
                    decision_details.append(f"Micro: {micro_reason}")
                    decision_details.append(m5_reason)
            else:
                print(f"❌ BUY rejeté: {m5_reason}")
    
    # 🔥 DÉCISION SELL
    elif (sell_score_total >= 75 and momentum['momentum_gate_passed']):
        
        micro_valid, micro_score, micro_reason = check_micro_momentum(df, "SELL")
        
        if not micro_valid:
            print(f"❌ Micro momentum SELL échoué: {micro_reason}")
        else:
            m5_aligned, m5_reason, m5_bonus = check_m5_alignment(m5_filter, "SELL")
            
            if m5_aligned or not SAINT_GRAAL_CONFIG['signal_config']['require_m5_alignment']:
                candle_valid, pattern, pattern_conf, candle_reason = validate_candle_for_5min_sell(df)
                
                if candle_valid:
                    direction = "SELL"
                    final_score = sell_score_total + pattern_conf + m5_bonus + micro_score
                    decision_details.append(f"SELL validé: {pattern} ({pattern_conf}%)")
                    decision_details.append(f"Micro: {micro_reason}")
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
            print(f"   Détails: {' | '.join(decision_details[:2])}")
            
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
                    'atr_score': atr_filter['score'],
                    'm5_alignment': m5_filter['trend'],
                    'structure': structure,
                }
            }
    
    # ===== 10. PAS DE SIGNAL =====
    print(f"❌ Aucun signal valide - Score insuffisant ou filtres échoués")
    return None

# ================= FONCTIONS DE COMPATIBILITÉ MISES À JOUR =================

def get_signal_with_metadata(df, signal_count=0, total_signals=6):
    """
    🔥 FONCTION PRINCIPALE AVEC NOUVEAUX FILTRES
    """
    try:
        if df is None or len(df) < 100:
            print("❌ Données insuffisantes pour analyse")
            return None
        
        # Utiliser la version avec micro momentum et ATR
        result = rule_signal_saint_graal_5min_pro_v3(df, signal_count, total_signals)
        
        if result is not None:
            direction_display = result['signal']
            quality_display = {
                'EXCELLENT': '⭐⭐⭐⭐⭐',
                'HIGH': '⭐⭐⭐⭐',
                'SOLID': '⭐⭐⭐',
                'MINIMUM': '⭐⭐',
                'CRITICAL': '⭐'
            }.get(result['quality'], '⭐')
            
            reason = f"{quality_display} {direction_display} (5min) | Score: {result['score']:.0f}"
            
            return {
                'direction': direction_display,
                'mode': result['mode'],
                'quality': result['quality'],
                'score': float(result['score']),
                'reason': reason,
                'expiration_minutes': 5,
                'session_info': {
                    'current_signal': signal_count + 1,
                    'total_signals': total_signals,
                    'timeframe': 'M1',
                    'expiration': '5MIN',
                    'filters': 'MICRO_MOMENTUM+ATR+M5',
                }
            }
        
        print(f"🎯 Aucun signal valide - Session {signal_count+1}/{total_signals}")
        return None
        
    except Exception as e:
        print(f"❌ Erreur critique: {str(e)}")
        return None

# ================= FONCTIONS EXISTANTES (À CONSERVER) =================

def calculate_m5_filter(df_m1):
    """Filtre M5 (inchangé)"""
    # ... (code existant)
    pass

def analyze_market_structure(df, lookback=20):
    """Analyse structure (inchangé)"""
    # ... (code existant)
    pass

def analyze_momentum_asymmetric_optimized(df):
    """Momentum optimisé (inchangé)"""
    # ... (code existant)
    pass

def calculate_bollinger_signals(df):
    """Bollinger Bands (inchangé)"""
    # ... (code existant)
    pass

def get_bb_confirmation_score(bb_signal, direction, stochastic_value):
    """Score BB (inchangé)"""
    # ... (code existant)
    pass

def check_m5_alignment(m5_filter, direction):
    """Alignement M5 (inchangé)"""
    # ... (code existant)
    pass

def validate_candle_for_5min_buy(df):
    """Validation bougie BUY (inchangé)"""
    # ... (code existant)
    pass

def validate_candle_for_5min_sell(df):
    """Validation bougie SELL (inchangé)"""
    # ... (code existant)
    pass

if __name__ == "__main__":
    print("🎯 DESK PRO BINAIRE - VERSION 4.5 ULTIMATE PLUS")
    print("🔥 NOUVEAUX FILTRES AJOUTÉS:")
    print("   1. Micro garde-fou momentum (cohérence dernières bougies M1)")
    print("   2. Filtre ATR (volatilité optimale 5-15 pips)")
    print("   3. Vétos ATR pour basse/haute volatilité")
    print("   4. Bonus squeeze ATR pour breakouts potentiels")
    print("\n✅ Système de filtrage multicouche optimal pour Pocket Option 5min!")
