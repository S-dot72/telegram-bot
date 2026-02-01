"""
utils.py - STRATÉGIE BINAIRE M1 PRO - VERSION 4.2
Niveau desk pro avec asymétrie BUY/SELL
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIGURATION DESK PRO =================

SAINT_GRAAL_CONFIG = {
    # 🔥 ASYMÉTRIE BUY/SELL
    'buy_rules': {
        'stoch_period': 5,      # Rapide pour les BUY
        'stoch_smooth': 3,
        'rsi_max_for_buy': 45,  # RSI maximum pour BUY
        'rsi_oversold': 35,     # Seuil oversold standard
        'require_swing_confirmation': False,  # BUY plus réactif
    },
    
    'sell_rules': {
        'stoch_period': 9,      # Lent pour les SELL
        'stoch_smooth': 3,
        'rsi_min_for_sell': 58,  # 🔥 Augmenté à 58
        'stoch_min_overbought': 65,
        'require_swing_break': True,  # 🔥 Confirmation swing obligatoire
        'max_swing_distance_pips': 5,  # Distance max au swing
        'momentum_gate_diff': 12,      # 🔥 Augmenté à 12
    },
    
    # Seuils contextuels optimisés
    'momentum_context': {
        'trend_overbought': 62,  # 🔥 Baissé à 62
        'trend_oversold': 38,    # 🔥 Hausse à 38
        'range_overbought': 72,
        'range_oversold': 28,
        'strong_trend_threshold': 1.0,  # 🔥 Hausse à 1.0%
    },
    
    # Structure rules PRO
    'structure_rules': {
        'sell_in_uptrend': {
            'require_swing_break': True,
            'min_stoch_overbought': 68,
            'allow_only_at_resistance': True,
            'max_rsi': 62,
        },
        'buy_in_downtrend': {
            'require_swing_break': True,
            'max_stoch_oversold': 32,
            'allow_only_at_support': True,
            'min_rsi': 38,
        },
        'range_trading': {
            'min_zone_strength': 25,
            'require_bounce_confirmation': True,
            'max_entry_from_zone_pips': 3,
        }
    },
}

# ================= DÉTECTION SWING INTERNE (NOUVEAU) =================

def detect_internal_swings(df, lookback=10):
    """
    🔥 NOUVEAU : Détection swings internes pour confirmation structure
    """
    if len(df) < lookback:
        return None, None
    
    recent = df.tail(lookback).copy()
    
    # Dernier swing high interne
    internal_high_idx = recent['high'].idxmax()
    internal_high = {
        'price': float(recent.loc[internal_high_idx, 'high']),
        'index': internal_high_idx,
        'bars_ago': len(recent) - 1 - recent.index.get_loc(internal_high_idx)
    }
    
    # Dernier swing low interne
    internal_low_idx = recent['low'].idxmin()
    internal_low = {
        'price': float(recent.loc[internal_low_idx, 'low']),
        'index': internal_low_idx,
        'bars_ago': len(recent) - 1 - recent.index.get_loc(internal_low_idx)
    }
    
    return internal_high, internal_low

def check_swing_break(current_price, internal_swing, direction, max_distance_pips=5):
    """
    Vérifie si le prix a cassé un swing interne
    """
    if internal_swing is None:
        return False, 0
    
    distance_pips = abs(current_price - internal_swing['price']) / 0.0001
    
    if direction == "SELL":
        # Pour SELL: besoin de casser le swing low interne
        is_broken = current_price < internal_swing['price']
    else:  # BUY
        # Pour BUY: besoin de casser le swing high interne
        is_broken = current_price > internal_swing['price']
    
    is_near = distance_pips <= max_distance_pips
    
    return is_broken and is_near, distance_pips

# ================= MOMENTUM ASYMÉTRIQUE (NOUVEAU) =================

def analyze_momentum_asymmetric(df):
    """
    🔥 NOUVEAU : Momentum asymétrique BUY/SELL
    Stochastic rapide pour BUY, lent pour SELL
    """
    if len(df) < 30:
        return {"status": "INSUFFICIENT_DATA"}
    
    # RSI commun
    rsi = RSIIndicator(close=df['close'], window=7).rsi()
    current_rsi = rsi.iloc[-1]
    
    # 🔥 STOCHASTIC RAPIDE POUR BUY (5,3,3)
    stoch_fast = StochasticOscillator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=SAINT_GRAAL_CONFIG['buy_rules']['stoch_period'],
        smooth_window=SAINT_GRAAL_CONFIG['buy_rules']['stoch_smooth']
    )
    stoch_k_fast = stoch_fast.stoch()
    stoch_d_fast = stoch_fast.stoch_signal()
    
    # 🔥 STOCHASTIC LENT POUR SELL (9,3,3)
    stoch_slow = StochasticOscillator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=SAINT_GRAAL_CONFIG['sell_rules']['stoch_period'],
        smooth_window=SAINT_GRAAL_CONFIG['sell_rules']['stoch_smooth']
    )
    stoch_k_slow = stoch_slow.stoch()
    stoch_d_slow = stoch_slow.stoch_signal()
    
    # Détection structure pour seuils contextuels
    structure, trend_strength = analyze_market_structure(df.tail(15))
    is_strong_trend = abs(trend_strength) > SAINT_GRAAL_CONFIG['momentum_context']['strong_trend_threshold']
    
    if is_strong_trend:
        overbought_threshold = SAINT_GRAAL_CONFIG['momentum_context']['trend_overbought']
        oversold_threshold = SAINT_GRAAL_CONFIG['momentum_context']['trend_oversold']
    else:
        overbought_threshold = SAINT_GRAAL_CONFIG['momentum_context']['range_overbought']
        oversold_threshold = SAINT_GRAAL_CONFIG['momentum_context']['range_oversold']
    
    # 🔥 DÉTECTION PIC STOCHASTIC LENT (pour SELL)
    stoch_slow_values = stoch_k_slow.tail(10).values
    stoch_peak_detected = False
    stoch_peak_value = 0
    
    if len(stoch_slow_values) >= 8:
        for i in range(3, len(stoch_slow_values)-3):
            if (stoch_slow_values[i] > stoch_slow_values[i-1] and 
                stoch_slow_values[i] > stoch_slow_values[i-2] and
                stoch_slow_values[i] > stoch_slow_values[i-3] and
                stoch_slow_values[i] > stoch_slow_values[i+1] and
                stoch_slow_values[i] > stoch_slow_values[i+2] and
                stoch_slow_values[i] > stoch_slow_values[i+3]):
                
                if stoch_slow_values[i] >= overbought_threshold:
                    stoch_peak_detected = True
                    stoch_peak_value = stoch_slow_values[i]
                    break
    
    # Turning bearish sur slow
    stoch_turning_bearish = False
    if len(stoch_slow_values) >= 4:
        recent_slow = stoch_slow_values[-4:]
        if (recent_slow[0] > recent_slow[1] > recent_slow[2] > recent_slow[3] and
            recent_slow[0] - recent_slow[3] > 8.0):  # 🔥 8 points minimum
            stoch_turning_bearish = True
    
    # Turning bullish sur fast
    stoch_fast_values = stoch_k_fast.tail(6).values
    stoch_turning_bullish = False
    if len(stoch_fast_values) >= 4:
        recent_fast = stoch_fast_values[-4:]
        if (recent_fast[0] < recent_fast[1] < recent_fast[2] < recent_fast[3] and
            recent_fast[3] - recent_fast[0] > 6.0):
            stoch_turning_bullish = True
    
    # MACD
    macd_ind = MACD(
        close=df['close'],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )
    macd_line = macd_ind.macd().iloc[-1]
    macd_signal = macd_ind.macd_signal().iloc[-1]
    macd_histogram = macd_line - macd_signal
    
    # 🔥 CALCUL SCORES ASYMÉTRIQUES
    sell_score = 0
    buy_score = 0
    sell_reasons = []
    buy_reasons = []
    
    # ========== CONDITIONS SELL (STRICTISSIMES) ==========
    current_stoch_slow = stoch_k_slow.iloc[-1]
    
    # RÈGLE 1: RSI minimum 58
    if current_rsi >= SAINT_GRAAL_CONFIG['sell_rules']['rsi_min_for_sell']:
        sell_score += 15
        sell_reasons.append(f"RSI:{current_rsi:.1f}")
    
    # RÈGLE 2: Stochastic lent overbought + turning
    if (current_stoch_slow >= overbought_threshold and 
        stoch_turning_bearish and
        current_stoch_slow <= 82):  # 🔥 Éviter extrêmes tardifs
        
        sell_score += 25
        sell_reasons.append(f"StochS:{current_stoch_slow:.1f}↓")
    
    # RÈGLE 3: Pic détecté + baisse confirmée
    elif stoch_peak_detected and stoch_turning_bearish:
        sell_score += 30  # 🔥 Bonus pic
        sell_reasons.append(f"Pic:{stoch_peak_value:.1f}↓")
    
    # RÈGLE 4: MACD baissier fort
    if macd_histogram < -0.00015:  # 🔥 Seuil plus strict
        sell_score += 12
        sell_reasons.append("MACD↓")
    
    # ========== CONDITIONS BUY (RÉACTIVES) ==========
    current_stoch_fast = stoch_k_fast.iloc[-1]
    
    # RSI oversold ou bas
    if current_rsi <= SAINT_GRAAL_CONFIG['buy_rules']['rsi_max_for_buy']:
        buy_score += 18  # 🔥 Plus généreux
        buy_reasons.append(f"RSI:{current_rsi:.1f}")
    
    # Stochastic fast oversold ou turning
    if (current_stoch_fast <= oversold_threshold or 
        stoch_turning_bullish):
        
        buy_score += 28
        buy_reasons.append(f"StochF:{current_stoch_fast:.1f}↑")
    
    # MACD haussier
    if macd_histogram > 0.0001:
        buy_score += 10
        buy_reasons.append("MACD↑")
    
    # 🔥 VÉTO MOMENTUM STRICT
    momentum_gate_passed = True
    gate_reason = ""
    
    if sell_score > buy_score:
        diff = sell_score - buy_score
        if diff < SAINT_GRAAL_CONFIG['sell_rules']['momentum_gate_diff']:
            momentum_gate_passed = False
            gate_reason = f"Momentum SELL insuffisant ({diff} < {SAINT_GRAAL_CONFIG['sell_rules']['momentum_gate_diff']})"
    
    elif buy_score > sell_score:
        diff = buy_score - sell_score
        if diff < 8:  # 🔥 Gate plus léger pour BUY
            momentum_gate_passed = False
            gate_reason = f"Momentum BUY insuffisant ({diff} < 8)"
    
    return {
        'rsi': float(current_rsi),
        'stoch_k_fast': float(stoch_k_fast.iloc[-1]),
        'stoch_d_fast': float(stoch_d_fast.iloc[-1]),
        'stoch_k_slow': float(stoch_k_slow.iloc[-1]),
        'stoch_d_slow': float(stoch_d_slow.iloc[-1]),
        'stoch_turning_bearish': stoch_turning_bearish,
        'stoch_turning_bullish': stoch_turning_bullish,
        'stoch_peak_detected': stoch_peak_detected,
        'stoch_peak_value': float(stoch_peak_value),
        'macd_histogram': float(macd_histogram),
        'sell_score': sell_score,
        'buy_score': buy_score,
        'sell_reasons': sell_reasons,
        'buy_reasons': buy_reasons,
        'overbought_threshold': overbought_threshold,
        'oversold_threshold': oversold_threshold,
        'dominant': "SELL" if sell_score > buy_score else "BUY" if buy_score > sell_score else "NEUTRAL",
        'strength': abs(sell_score - buy_score),
        'momentum_gate_passed': momentum_gate_passed,
        'gate_reason': gate_reason,
        'structure': structure,
        'trend_strength': trend_strength,
    }

# ================= STRUCTURE SCORE PRO (NOUVEAU) =================

def calculate_structure_score_pro_m1(structure, direction, momentum_info, internal_swing_break):
    """
    🔥 NOUVEAU : Structure score avec règles desk pro
    """
    score = 0
    reasons = []
    
    if direction == "SELL":
        # 🔥 RÈGLE DESK: SELL en uptrend = conditions ultra strictes
        if "UPTREND" in structure:
            rules = SAINT_GRAAL_CONFIG['structure_rules']['sell_in_uptrend']
            
            if momentum_info['rsi'] > rules['max_rsi']:
                score = -20
                reasons.append(f"RSI {momentum_info['rsi']:.1f} > {rules['max_rsi']}")
            
            elif not internal_swing_break and rules['require_swing_break']:
                score = -25
                reasons.append("Pas de break swing interne")
            
            elif momentum_info['stoch_k_slow'] < rules['min_stoch_overbought']:
                score = -15
                reasons.append(f"Stoch {momentum_info['stoch_k_slow']:.1f} < {rules['min_stoch_overbought']}")
            
            else:
                score = 15  # 🔥 Score réduit pour SELL contre tendance
                reasons.append("SELL uptrend strict")
        
        # SELL en downtrend = favorable
        elif "DOWNTREND" in structure:
            score = 25
            reasons.append("DOWNTREND fort")
        
        # SELL en range = conditions moyennes
        elif "RANGE" in structure:
            score = 18
            reasons.append("Range + zone")
        
        # SELL en wedge/chao = éviter
        elif "WEDGE" in structure or "CHAOTIC" in structure:
            score = -10
            reasons.append("Structure défavorable")
    
    else:  # BUY
        # 🔥 RÈGLE DESK: BUY en downtrend = conditions strictes
        if "DOWNTREND" in structure:
            rules = SAINT_GRAAL_CONFIG['structure_rules']['buy_in_downtrend']
            
            if momentum_info['rsi'] < rules['min_rsi']:
                score = -20
                reasons.append(f"RSI {momentum_info['rsi']:.1f} < {rules['min_rsi']}")
            
            elif not internal_swing_break and rules['require_swing_break']:
                score = -25
                reasons.append("Pas de break swing interne")
            
            elif momentum_info['stoch_k_fast'] > rules['max_stoch_oversold']:
                score = -15
                reasons.append(f"Stoch {momentum_info['stoch_k_fast']:.1f} > {rules['max_stoch_oversold']}")
            
            else:
                score = 15
                reasons.append("BUY downtrend strict")
        
        # BUY en uptrend = favorable
        elif "UPTREND" in structure:
            score = 25
            reasons.append("UPTREND fort")
        
        # BUY en range
        elif "RANGE" in structure:
            score = 18
            reasons.append("Range + zone")
        
        # BUY en wedge/chao
        elif "WEDGE" in structure or "CHAOTIC" in structure:
            score = -10
            reasons.append("Structure défavorable")
    
    return score, " | ".join(reasons)

# ================= FONCTION PRINCIPALE V4.2 =================

def rule_signal_saint_graal_m1_pro_v2(df, signal_count=0, total_signals_needed=8):
    """
    🔥 VERSION 4.2 : LOGIQUE DESK PRO
    Asymétrie BUY/SELL + règles structure strictes
    """
    print(f"\n{'='*70}")
    print(f"🚀 DESK PRO M1 - Signal #{signal_count+1}")
    print(f"{'='*70}")
    
    if len(df) < 50:
        return pro_fallback_intelligent(df, signal_count, total_signals_needed)
    
    current_price = float(df.iloc[-1]['close'])
    
    # ===== 1. ANALYSE COMPLÈTE =====
    structure, trend_strength = analyze_market_structure(df)
    internal_high, internal_low = detect_internal_swings(df.tail(10))
    
    print(f"🏗️  Structure: {structure} | Force: {trend_strength:.1f}%")
    if internal_high:
        print(f"📊 Swings internes: H={internal_high['price']:.5f}({internal_high['bars_ago']}b) | L={internal_low['price']:.5f}({internal_low['bars_ago']}b)")
    
    # ===== 2. MOMENTUM ASYMÉTRIQUE =====
    momentum = analyze_momentum_asymmetric(df)
    print(f"⚡ Momentum: RSI {momentum['rsi']:.1f} | StochF {momentum['stoch_k_fast']:.1f} | StochS {momentum['stoch_k_slow']:.1f}")
    print(f"   Pic:{momentum['stoch_peak_detected']}({momentum['stoch_peak_value']:.1f}) | Gate:{'PASS' if momentum['momentum_gate_passed'] else 'BLOCK'} {momentum.get('gate_reason', '')}")
    
    # ===== 3. ZONES S/R =====
    supports, resistances = detect_key_zones(df)
    near_support, nearest_support, dist_support = is_price_near_zone_pro(current_price, supports, 8)  # 🔥 Distance réduite
    near_resistance, nearest_resistance, dist_resistance = is_price_near_zone_pro(current_price, resistances, 8)
    
    print(f"📍 Zones: S {near_support}({dist_support:.1f}p) | R {near_resistance}({dist_resistance:.1f}p)")
    
    # ===== 4. VÉRIFICATION SWING BREAK =====
    swing_break_sell = False
    swing_break_buy = False
    
    if internal_low:
        swing_break_sell, dist_break_sell = check_swing_break(
            current_price, internal_low, "SELL", 
            SAINT_GRAAL_CONFIG['sell_rules']['max_swing_distance_pips']
        )
    
    if internal_high:
        swing_break_buy, dist_break_buy = check_swing_break(
            current_price, internal_high, "BUY", 5
        )
    
    print(f"🔨 Swing break: SELL {swing_break_sell}({dist_break_sell if 'dist_break_sell' in locals() else 0:.1f}p) | BUY {swing_break_buy}({dist_break_buy if 'dist_break_buy' in locals() else 0:.1f}p)")
    
    # ===== 5. SCORING DESK PRO =====
    sell_score = 0
    buy_score = 0
    sell_details = []
    buy_details = []
    
    # 🔥 Structure score avec règles pro
    structure_score_sell, structure_reason_sell = calculate_structure_score_pro_m1(
        structure, "SELL", momentum, swing_break_sell
    )
    sell_score += structure_score_sell
    if structure_score_sell != 0:
        sell_details.append(structure_reason_sell)
    
    structure_score_buy, structure_reason_buy = calculate_structure_score_pro_m1(
        structure, "BUY", momentum, swing_break_buy
    )
    buy_score += structure_score_buy
    if structure_score_buy != 0:
        buy_details.append(structure_reason_buy)
    
    # Momentum scores
    sell_score += momentum['sell_score']
    buy_score += momentum['buy_score']
    
    if momentum['sell_score'] > 0:
        sell_details.extend(momentum['sell_reasons'])
    if momentum['buy_score'] > 0:
        buy_details.extend(momentum['buy_reasons'])
    
    # Zones bonus (avec vérification distance)
    if near_resistance and dist_resistance <= 5:  # 🔥 Zone proche seulement
        zone_bonus, zone_reason = calculate_zone_strength(nearest_resistance)
        sell_score += int(zone_bonus * 0.8)  # 🔥 Bonus réduit
        sell_details.append(f"Zone:{zone_reason}")
    
    if near_support and dist_support <= 5:
        zone_bonus, zone_reason = calculate_zone_strength(nearest_support)
        buy_score += int(zone_bonus * 0.8)
        buy_details.append(f"Zone:{zone_reason}")
    
    print(f"🎯 Scores: SELL {sell_score}/100 - BUY {buy_score}/100")
    
    # ===== 6. DÉCISION AVEC FILTRES DESK =====
    direction = None
    final_score = 0
    validation_issues = []
    
    # 🔥 VÉTO ABSOLU POUR SELL
    if sell_score > buy_score:
        # VÉTO 1: Momentum gate
        if not momentum['momentum_gate_passed']:
            validation_issues.append(f"Momentum: {momentum['gate_reason']}")
        
        # VÉTO 2: Structure contre tendance sans break
        elif "UPTREND" in structure and not swing_break_sell:
            validation_issues.append("Uptrend sans break swing")
        
        # VÉTO 3: RSI trop bas
        elif momentum['rsi'] < 52:  # 🔥 Seuil relevé
            validation_issues.append(f"RSI {momentum['rsi']:.1f} trop bas pour SELL")
        
        # VÉTO 4: Stochastic lent insuffisant
        elif momentum['stoch_k_slow'] < 58 and not momentum['stoch_peak_detected']:
            validation_issues.append(f"Stoch lent {momentum['stoch_k_slow']:.1f} < 58 sans pic")
        
        else:
            # Validation bougie
            candle_valid, pattern, pattern_conf, candle_reason = validate_candle_for_binary_m1(
                df, "SELL", require_rejection=True
            )
            
            if candle_valid:
                direction = "SELL"
                final_score = sell_score + (pattern_conf / 10)
                print(f"✅ SELL confirmé: {pattern} ({pattern_conf}%) | RSI:{momentum['rsi']:.1f} StochS:{momentum['stoch_k_slow']:.1f}")
            else:
                validation_issues.append(f"Bougie: {candle_reason}")
    
    # 🔥 BUY (moins strict)
    elif buy_score > sell_score:
        if not momentum['momentum_gate_passed']:
            validation_issues.append(f"Momentum: {momentum['gate_reason']}")
        
        # VÉTO BUY en downtrend sans break
        elif "DOWNTREND" in structure and not swing_break_buy:
            validation_issues.append("Downtrend sans break swing")
        
        else:
            candle_valid, pattern, pattern_conf, candle_reason = validate_candle_for_binary_m1(
                df, "BUY", require_rejection=False
            )
            
            if candle_valid:
                direction = "BUY"
                final_score = buy_score + (pattern_conf / 10)
                print(f"✅ BUY confirmé: {pattern} ({pattern_conf}%)")
            else:
                validation_issues.append(f"Bougie: {candle_reason}")
    
    # ===== 7. DÉCISION FINALE =====
    if direction and final_score >= 65:  # 🔥 Seuil relevé à 65
        # Qualité basée sur score et structure
        if final_score >= 88 and ("TREND" in structure or swing_break_sell or swing_break_buy):
            quality = "EXCELLENT"
            mode = "DESK_MAX"
        elif final_score >= 78:
            quality = "HIGH"
            mode = "DESK_PRO"
        elif final_score >= 68:
            quality = "SOLID"
            mode = "DESK_STANDARD"
        else:
            quality = "MINIMUM"
            mode = "DESK_MIN"
        
        print(f"\n🎉 DÉCISION: {direction} | Score: {final_score:.1f}/100 | Qualité: {quality}")
        print(f"   Structure: {structure} | Swing break: {'SELL' if swing_break_sell else 'BUY' if swing_break_buy else 'Non'}")
        
        direction_display = "CALL" if direction == "BUY" else "PUT"
        
        return {
            'signal': direction_display,
            'mode': mode,
            'quality': quality,
            'score': float(final_score),
            'reason': f"{direction} | {quality} | Score {final_score:.1f} | {structure}",
            'structure_info': {
                'market_structure': structure,
                'trend_strength': float(trend_strength),
                'swing_break': swing_break_sell if direction == "SELL" else swing_break_buy,
                'momentum_info': {
                    'rsi': momentum['rsi'],
                    'stoch_fast': momentum['stoch_k_fast'],
                    'stoch_slow': momentum['stoch_k_slow'],
                    'peak_detected': momentum['stoch_peak_detected'],
                }
            }
        }
    
    # ===== 8. FALLBACK SÉCURISÉ =====
    if validation_issues:
        print(f"⚡ Fallback: {validation_issues[0]}")
    else:
        print("⚡ Fallback: Aucune direction valide")
    
    return pro_fallback_intelligent(df, signal_count, total_signals_needed)

# ================= FONCTION MAINTAINED POUR COMPATIBILITÉ =================

def get_signal_with_metadata(df, signal_count=0, total_signals=8):
    """Wrapper pour compatibilité avec l'architecture existante"""
    try:
        if df is None or len(df) < 30:
            return create_minimal_fallback("Données insuffisantes", signal_count, total_signals)
        
        # Utiliser la version PRO
        result = rule_signal_saint_graal_m1_pro_v2(df, signal_count, total_signals)
        
        if result:
            direction_display = result['signal']
            quality_display = {
                'EXCELLENT': '⭐⭐⭐⭐⭐',
                'HIGH': '⭐⭐⭐⭐',
                'SOLID': '⭐⭐⭐',
                'MINIMUM': '⭐⭐',
                'CRITICAL': '⭐'
            }.get(result['quality'], '⭐')
            
            reason = f"{quality_display} {direction_display} | Score: {result['score']:.0f}/100"
            
            return {
                'direction': direction_display,
                'mode': result['mode'],
                'quality': result['quality'],
                'score': float(result['score']),
                'reason': reason,
                'structure_info': result.get('structure_info', {}),
                'session_info': {
                    'current_signal': signal_count + 1,
                    'total_signals': total_signals,
                    'mode_used': result['mode'],
                }
            }
        
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
    
    # Fallback absolu
    return {
        'direction': 'CALL',
        'mode': 'ERROR',
        'quality': 'CRITICAL',
        'score': 45.0,
        'reason': 'Erreur système',
        'session_info': {
            'current_signal': signal_count + 1,
            'total_signals': total_signals,
            'mode_used': 'ERROR',
        }
    }

if __name__ == "__main__":
    print("🚀 DESK PRO BINAIRE M1 - VERSION 4.2")
    print("📊 Niveau: Trading desk professionnel")
    print("\n🔥 AMÉLIORATIONS CLÉS:")
    print("✅ 1. ASYMÉTRIE BUY/SELL: Stoch 5 pour BUY, Stoch 9 pour SELL")
    print("✅ 2. RSI minimum SELL: 58 (au lieu de 55)")
    print("✅ 3. SWING BREAK obligatoire: Pas de SELL en uptrend sans break")
    print("✅ 4. MOMENTUM GATE renforcé: Différence > 12 points")
    print("✅ 5. STRUCTURE RULES desk: Conditions strictes par type de structure")
    print("\n🎯 Objectif réaliste: 62-68% winrate (hors news)")
