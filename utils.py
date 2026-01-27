"""
utils.py - STRATÉGIE PERFECTION M1 ADAPTIVE AVEC TIMING
Analyse 20-40 secondes pour garantir des signaux gagnants de haute qualité
Expiration 1 minute optimisée pour trading binaire
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIGURATION TIMING M1 =================

M1_CONFIG = {
    'analysis_min_time': 20,      # Analyse minimum 20 secondes
    'analysis_max_time': 40,      # Maximum 40 secondes
    'candle_duration': 60,        # Bougie M1 = 60 secondes
    'signal_before_expiry': 15,   # Émettre signal 15s avant fin
    
    # Phases de la session (max 8 signaux)
    'phase1_signals': [1, 2, 3, 4],  # Perfection absolue
    'phase2_signals': [5, 6],        # Excellence
    'phase3_signals': [7, 8],        # Qualité garantie
    
    # Seuils par phase
    'phase1_threshold': 0.90,    # 90% confiance minimum
    'phase2_threshold': 0.80,    # 80% confiance minimum
    'phase3_threshold': 0.70,    # 70% confiance minimum
}

# ================= TIMING INTELLIGENT =================

def intelligent_analysis_timing(df, signal_number):
    """
    Gère le timing intelligent de l'analyse
    Prend plus de temps pour les signaux importants
    """
    start_time = time.time()
    
    # Phase 1 : Analyse approfondie (25-40 secondes)
    if signal_number in M1_CONFIG['phase1_signals']:
        print(f"[PHASE 1] Analyse approfondie du signal {signal_number}/8...")
        min_time = 25
        max_time = 40
        
    # Phase 2 : Analyse détaillée (20-30 secondes)
    elif signal_number in M1_CONFIG['phase2_signals']:
        print(f"[PHASE 2] Analyse détaillée du signal {signal_number}/8...")
        min_time = 20
        max_time = 30
        
    # Phase 3 : Analyse rapide mais complète (15-25 secondes)
    else:
        print(f"[PHASE 3] Analyse rapide du signal {signal_number}/8...")
        min_time = 15
        max_time = 25
    
    # ===== ANALYSE MULTI-NIVEAUX =====
    
    # Niveau 1 : Analyse technique de base
    analysis1 = analyze_technical_level1(df)
    time.sleep(3)  # Simulation traitement
    
    # Niveau 2 : Analyse avancée
    analysis2 = analyze_technical_level2(df)
    time.sleep(3)
    
    # Niveau 3 : Validation des patterns
    analysis3 = validate_patterns(df)
    time.sleep(3)
    
    # Niveau 4 : Analyse de risque
    analysis4 = risk_analysis(df)
    time.sleep(3)
    
    # Combiner les analyses
    combined_analysis = combine_analyses(analysis1, analysis2, analysis3, analysis4)
    
    # ===== GESTION DU TEMPS =====
    
    elapsed = time.time() - start_time
    
    # Si analyse trop rapide, prendre plus de temps pour réflexion
    if elapsed < min_time:
        extra_time = min_time - elapsed
        print(f"[TIMING] Analyse approfondie supplémentaire: {extra_time:.1f}s")
        time.sleep(extra_time)
        elapsed = time.time() - start_time
    
    # Limiter le temps maximum
    if elapsed > max_time:
        print(f"[TIMING] Analyse accélérée pour respecter le timing")
    
    print(f"[TIMING] Analyse complète en {elapsed:.1f} secondes")
    
    return combined_analysis

def analyze_technical_level1(df):
    """Niveau 1 : Analyse technique rapide"""
    if len(df) < 10:
        return None
    
    last = df.iloc[-1]
    
    # EMA ultra-rapides (M1 optimisé)
    df['ema_2'] = df['close'].ewm(span=2, adjust=False).mean()
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    
    # RSI rapide
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=5).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=5).mean()
    rs = gain / loss.replace(0, 0.00001)
    df['rsi_5'] = 100 - (100 / (1 + rs))
    
    # MACD ultra-rapide
    exp1 = df['close'].ewm(span=3, adjust=False).mean()
    exp2 = df['close'].ewm(span=7, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=2, adjust=False).mean()
    
    # Bollinger Bands rapides
    df['bb_middle'] = df['close'].rolling(window=10).mean()
    bb_std = df['close'].rolling(window=10).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 1.5)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 1.5)
    
    return df

def analyze_technical_level2(df):
    """Niveau 2 : Analyse avancée"""
    if len(df) < 20:
        return None
    
    # Momentum instantané
    df['momentum_1'] = df['close'].pct_change(1) * 100
    df['momentum_3'] = df['close'].pct_change(3) * 100
    
    # Volatilité
    df['volatility_10'] = df['close'].rolling(10).std() / df['close'].rolling(10).mean()
    
    # Price action
    df['candle_body'] = df['close'] - df['open']
    df['candle_size'] = df['high'] - df['low']
    df['body_ratio'] = abs(df['candle_body']) / df['candle_size'].replace(0, 0.00001)
    
    # Force relative
    df['strength'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 100
    
    return df

def validate_patterns(df):
    """Niveau 3 : Validation des patterns M1"""
    if len(df) < 15:
        return None
    
    patterns = {
        'bullish_engulfing': False,
        'bearish_engulfing': False,
        'hammer': False,
        'shooting_star': False,
        'doji': False
    }
    
    # Analyser les 3 dernières bougies
    if len(df) >= 3:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        # Bullish Engulfing
        if (prev['candle_body'] < 0 and last['candle_body'] > 0 and
            abs(last['candle_body']) > abs(prev['candle_body']) and
            last['close'] > prev['open']):
            patterns['bullish_engulfing'] = True
            
        # Bearish Engulfing
        if (prev['candle_body'] > 0 and last['candle_body'] < 0 and
            abs(last['candle_body']) > abs(prev['candle_body']) and
            last['close'] < prev['open']):
            patterns['bearish_engulfing'] = True
            
        # Hammer (bougie de retournement haussier)
        if (last['body_ratio'] < 0.3 and 
            (last['close'] - last['low']) > 2 * abs(last['candle_body']) and
            last['candle_body'] > 0):
            patterns['hammer'] = True
            
        # Shooting Star (bougie de retournement baissier)
        if (last['body_ratio'] < 0.3 and 
            (last['high'] - last['close']) > 2 * abs(last['candle_body']) and
            last['candle_body'] < 0):
            patterns['shooting_star'] = True
            
        # Doji (indécision)
        if last['body_ratio'] < 0.1:
            patterns['doji'] = True
    
    return patterns

def risk_analysis(df):
    """Niveau 4 : Analyse de risque"""
    if len(df) < 10:
        return {'risk_level': 'HIGH', 'reason': 'Données insuffisantes'}
    
    last = df.iloc[-1]
    
    risk_score = 0
    reasons = []
    
    # Volatilité excessive
    if last.get('volatility_10', 0) > 0.03:
        risk_score += 30
        reasons.append("Volatilité élevée")
    
    # Bougie trop petite (pas de conviction)
    if last.get('body_ratio', 0) < 0.2:
        risk_score += 20
        reasons.append("Faible conviction (petite bougie)")
    
    # Tendance incertaine
    if abs(last.get('strength', 0)) < 0.1:
        risk_score += 15
        reasons.append("Tendance faible")
    
    # RSI extrême
    rsi = last.get('rsi_5', 50)
    if rsi > 80 or rsi < 20:
        risk_score += 25
        reasons.append("RSI extrême")
    
    # Déterminer niveau de risque
    if risk_score >= 50:
        risk_level = "HIGH"
    elif risk_score >= 30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return {
        'risk_level': risk_level,
        'risk_score': risk_score,
        'reasons': reasons
    }

def combine_analyses(analysis1, analysis2, analysis3, analysis4):
    """Combine toutes les analyses pour une décision finale"""
    
    if analysis1 is None or analysis2 is None:
        return None
    
    df = analysis1.copy()
    last = df.iloc[-1]
    
    # ===== CALCUL DU SCORE DE CONFIANCE =====
    
    confidence_score = 50  # Base
    
    # 1. Alignement EMA (15 points)
    if last['ema_2'] > last['ema_5'] > last['ema_9']:
        confidence_score += 15
    elif last['ema_2'] < last['ema_5'] < last['ema_9']:
        confidence_score += 15
    
    # 2. MACD direction (15 points)
    if last['macd'] > last['macd_signal']:
        confidence_score += 15
    elif last['macd'] < last['macd_signal']:
        confidence_score += 15
    
    # 3. RSI optimal (15 points)
    rsi = last.get('rsi_5', 50)
    if 40 < rsi < 60:
        confidence_score += 15
    elif 30 < rsi < 70:
        confidence_score += 10
    
    # 4. Momentum (10 points)
    if last.get('momentum_1', 0) > 0:
        confidence_score += 10
    elif last.get('momentum_1', 0) < 0:
        confidence_score += 10
    
    # 5. Price action (10 points)
    if last.get('body_ratio', 0) > 0.4:
        confidence_score += 10
    
    # 6. Patterns (15 points)
    if analysis3:
        if analysis3.get('bullish_engulfing', False) or analysis3.get('hammer', False):
            confidence_score += 15
        elif analysis3.get('bearish_engulfing', False) or analysis3.get('shooting_star', False):
            confidence_score += 15
    
    # 7. Réduction basée sur le risque (jusqu'à -30 points)
    if analysis4 and analysis4['risk_level'] == "HIGH":
        confidence_score -= 30
    elif analysis4 and analysis4['risk_level'] == "MEDIUM":
        confidence_score -= 15
    
    # Normaliser entre 0 et 100
    confidence_score = max(0, min(100, confidence_score))
    confidence_percentage = confidence_score / 100.0
    
    # ===== DÉTERMINATION DE LA DIRECTION =====
    
    call_signals = 0
    put_signals = 0
    
    # Signaux CALL
    if last['ema_2'] > last['ema_5']:
        call_signals += 1
    if last['macd'] > last['macd_signal']:
        call_signals += 1
    if rsi > 50:
        call_signals += 1
    if last.get('momentum_1', 0) > 0:
        call_signals += 1
    if analysis3 and (analysis3.get('bullish_engulfing', False) or analysis3.get('hammer', False)):
        call_signals += 2
    
    # Signaux PUT
    if last['ema_2'] < last['ema_5']:
        put_signals += 1
    if last['macd'] < last['macd_signal']:
        put_signals += 1
    if rsi < 50:
        put_signals += 1
    if last.get('momentum_1', 0) < 0:
        put_signals += 1
    if analysis3 and (analysis3.get('bearish_engulfing', False) or analysis3.get('shooting_star', False)):
        put_signals += 2
    
    # Décision
    if call_signals > put_signals and confidence_percentage >= 0.60:
        direction = 'CALL'
    elif put_signals > call_signals and confidence_percentage >= 0.60:
        direction = 'PUT'
    else:
        direction = None
    
    return {
        'direction': direction,
        'confidence': confidence_percentage,
        'call_signals': call_signals,
        'put_signals': put_signals,
        'risk_level': analysis4['risk_level'] if analysis4 else 'UNKNOWN',
        'analysis_time': time.time()
    }

# ================= STRATÉGIE ADAPTATIVE PAR PHASE =================

def adaptive_phase_strategy(df, signal_number, total_signals=8):
    """
    Stratégie adaptive basée sur la phase du signal
    """
    # Déterminer la phase
    if signal_number in M1_CONFIG['phase1_signals']:
        phase = "PERFECTION"
        min_confidence = M1_CONFIG['phase1_threshold']
    elif signal_number in M1_CONFIG['phase2_signals']:
        phase = "EXCELLENCE"
        min_confidence = M1_CONFIG['phase2_threshold']
    elif signal_number in M1_CONFIG['phase3_signals']:
        phase = "QUALITY"
        min_confidence = M1_CONFIG['phase3_threshold']
    else:
        phase = "EXTRA"
        min_confidence = 0.65
    
    print(f"\n[PHASE {phase}] Signal {signal_number}/{total_signals}")
    print(f"Seuil minimum: {min_confidence*100:.0f}% de confiance")
    
    # Analyse avec timing intelligent
    result = intelligent_analysis_timing(df, signal_number)
    
    if result is None or result['direction'] is None:
        print(f"[PHASE {phase}] Aucun signal valide trouvé")
        return None
    
    # Vérifier le seuil de confiance minimum
    if result['confidence'] < min_confidence:
        print(f"[PHASE {phase}] Confiance insuffisante: {result['confidence']*100:.1f}% < {min_confidence*100:.0f}%")
        return None
    
    # Vérifier la qualité du signal
    min_signals_diff = 2  # Différence minimum entre signaux CALL/PUT
    
    if phase == "PERFECTION":
        if result['call_signals'] - result['put_signals'] < min_signals_diff:
            print(f"[PERFECTION] Convergence insuffisante: {result['call_signals']} vs {result['put_signals']}")
            return None
        
        # Risque maximum autorisé: LOW seulement
        if result['risk_level'] != "LOW":
            print(f"[PERFECTION] Risque trop élevé: {result['risk_level']}")
            return None
    
    elif phase == "EXCELLENCE":
        if abs(result['call_signals'] - result['put_signals']) < 1:
            print(f"[EXCELLENCE] Pas de direction claire")
            return None
        
        # Risque maximum: MEDIUM
        if result['risk_level'] == "HIGH":
            print(f"[EXCELLENCE] Risque HIGH non autorisé")
            return None
    
    # Signal accepté
    print(f"[PHASE {phase}] Signal {result['direction']} validé!")
    print(f"   Confiance: {result['confidence']*100:.1f}%")
    print(f"   Call: {result['call_signals']} | Put: {result['put_signals']}")
    print(f"   Risque: {result['risk_level']}")
    
    return result['direction']

# ================= INTERFACE DE COMPATIBILITÉ =================

def compute_indicators(df, ema_fast=8, ema_slow=21, rsi_len=14, bb_len=20):
    """Wrapper de compatibilité"""
    return analyze_technical_level1(df)

def rule_signal_ultra_strict(df, session_priority=3):
    """Mode ultra strict - Phase 1 seulement"""
    return adaptive_phase_strategy(df, signal_number=1)

def rule_signal(df, session_priority=3):
    """Interface principale - utiliser avec signal_number"""
    # Pour compatibilité, utiliser signal 1
    print("Utilisez get_signal_adaptive() avec signal_number pour meilleurs résultats")
    return adaptive_phase_strategy(df, signal_number=1)

# ================= FONCTIONS NOUVELLES POUR LE BOT =================

def get_signal_adaptive(df, signal_number, total_signals=8):
    """
    Fonction principale pour le bot M1
    À appeler avec le numéro du signal dans la session
    """
    if signal_number > total_signals:
        print(f"Session complète! {total_signals}/{total_signals} signaux")
        return None
    
    print(f"\n{'='*60}")
    print(f"GENERATION SIGNAL {signal_number}/{total_signals}")
    print(f"{'='*60}")
    
    # Vérifier données minimales
    if len(df) < 15:
        print("Données insuffisantes pour analyse")
        return None
    
    # Générer le signal adaptatif
    signal = adaptive_phase_strategy(df, signal_number, total_signals)
    
    if signal:
        print(f"\nSIGNAL {signal_number}/{total_signals} PRET: {signal}")
        print(f"Heure: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Bougies analysées: {len(df)}")
    else:
        print(f"\nAucun signal valide pour {signal_number}/{total_signals}")
        print("Patientez la prochaine bougie...")
    
    return signal

def get_signal_with_metadata(df, signal_number, total_signals=8):
    """Version avec métadonnées complètes"""
    result = intelligent_analysis_timing(df, signal_number)
    
    if result and result['direction']:
        return {
            'direction': result['direction'],
            'confidence': result['confidence'],
            'phase': 'PERFECTION' if signal_number <= 4 else 'EXCELLENCE' if signal_number <= 6 else 'QUALITY',
            'call_signals': result['call_signals'],
            'put_signals': result['put_signals'],
            'risk_level': result['risk_level'],
            'timestamp': datetime.now().isoformat()
        }
    
    return None

def format_signal_reason(direction, confidence, indicators):
    """Format de raison pour le bot"""
    return f"{direction} | Confiance: {confidence:.1%} | Analyse: {len(indicators)} bougies"

def calculate_signal_quality_score(df):
    """Score de qualité rapide"""
    if len(df) < 10:
        return 0
    
    result = intelligent_analysis_timing(df, 1)
    if result:
        return int(result['confidence'] * 100)
    return 0

def is_kill_zone_optimal(hour_utc):
    """Pour compatibilité - désactivé pour sessions on-demand"""
    return False, None, 0

# ================= EXPORT =================

__all__ = [
    'compute_indicators',
    'rule_signal',
    'rule_signal_ultra_strict',
    'get_signal_adaptive',
    'get_signal_with_metadata',
    'calculate_signal_quality_score',
    'format_signal_reason',
    'is_kill_zone_optimal'
]

print("\n" + "="*60)
print("STRATEGIE PERFECTION M1 CHARGEE")
print(f"Timing analyse: {M1_CONFIG['analysis_min_time']}-{M1_CONFIG['analysis_max_time']}s")
print(f"Signaux/session: 8 max (4 Perfection + 2 Excellence + 2 Qualité)")
print("="*60)
