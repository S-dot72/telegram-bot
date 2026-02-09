"""
🚀 STRATÉGIE BINAIRE M1 PRO - VERSION 9.3 DEBUG
🔥 MODE DÉBOGAGE COMPLET - ANALYSE CHAQUE FILTRE
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIGURATION ULTRA PERMISSIVE =================

SAINT_GRAAL_CONFIG = {
    'expiration_minutes': 5,
    
    # 🔥 ZONES D'INTERDICTION DÉSACTIVÉES TEMPORAIREMENT
    'forbidden_zones': {
        'no_buy_zone': {
            'enabled': False,  # DÉSACTIVÉ
            'stoch_fast_max': 80,
            'rsi_max': 70,
            'bb_position_max': 80,
            'strict_mode': False,
            'penalty': 0,
        },
        'no_sell_zone': {
            'enabled': False,  # DÉSACTIVÉ
            'stoch_fast_min': 20,
            'rsi_min': 30,
            'bb_position_min': 20,
            'strict_mode': False,
            'penalty': 0,
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
    
    # 🔥 MOMENTUM GATE TRÈS PERMISSIF
    'momentum_rules': {
        'buy_conditions': {
            'rsi_max': 70,        # Très haut
            'rsi_oversold': 20,   # Très bas
            'stoch_max': 50,      # Très haut
            'stoch_oversold': 10, # Très bas
            'require_stoch_rising': False,  # DÉSACTIVÉ
        },
        'sell_conditions': {
            'rsi_min': 30,        # Très bas
            'rsi_overbought': 80, # Très haut
            'stoch_min': 50,      # Très bas
            'stoch_overbought': 90, # Très haut
            'require_stoch_falling': False, # DÉSACTIVÉ
        },
        'momentum_gate_diff': 2,  # Très bas
        'smart_gate': False,      # DÉSACTIVÉ
    },
    
    'micro_momentum': {
        'enabled': False,  # DÉSACTIVÉ
        'lookback_bars': 3,
        'min_bullish_bars': 1,
        'min_bearish_bars': 1,
        'require_trend_alignment': False,
        'weight': 0,
    },
    
    'bollinger_config': {
        'window': 20,
        'window_dev': 2,
        'oversold_zone': 15,
        'overbought_zone': 85,
        'buy_zone_max': 60,     # Très haut
        'sell_zone_min': 40,    # Très bas
        'middle_band_weight': 10,
        'strict_mode': False,   # DÉSACTIVÉ
        'penalty': 0,
    },
    
    'atr_filter': {
        'enabled': False,  # DÉSACTIVÉ
        'window': 14,
        'min_atr_pips': 1,
        'max_atr_pips': 50,
        'optimal_range': [1, 50],
    },
    
    # 🔥 M5 DÉSACTIVÉ
    'm5_filter': {
        'enabled': False,  # DÉSACTIVÉ
        'ema_fast': 50,
        'ema_slow': 200,
        'weight': 0,
        'soft_veto': False,
        'max_score_against_trend': 100,
    },
    
    # 🔥 ÉTAT DE MARCHÉ DÉSACTIVÉ
    'market_state': {
        'enabled': False,  # DÉSACTIVÉ
        'adx_threshold': 20,
        'rsi_range_threshold': 40,
        'prioritize_bb_in_range': False,
        'prioritize_momentum_in_trend': False,
    },
    
    'signal_validation': {
        'min_score': 50,           # TRÈS BAS !!!
        'max_score_realistic': 200,
        'confidence_zones': {
            50: 60,    # Très bas
            60: 65,
            70: 70,
            80: 75,
            90: 80,
            100: 85,
        },
        'cooldown_bars': 0,
    },
    
    # 🔥 COOLDOWN DÉSACTIVÉ
    'risk_management': {
        'dynamic_cooldown': False,
        'normal_cooldown': 0,
        'cooldown_by_quality': {
            'EXCELLENT': 0,
            'HIGH': 0,
            'SOLID': 0,
            'MINIMUM': 0,
        },
        'max_daily_trades': 100,
        'max_consecutive_losses': 10,
    }
}

# ================= ÉTAT DU TRADING SIMPLIFIÉ =================

class TradingState:
    """Gestion simplifiée pour le debug"""
    def __init__(self):
        self.last_trade_time = None
        
    def can_trade(self, current_time):
        return True, "OK"

trading_state = TradingState()

# ================= DÉTECTION ÉTAT DE MARCHÉ =================

def detect_market_state(df):
    """Détection simplifiée"""
    return {'state': 'DEBUG', 'adx': 0, 'rsi': 50, 'reason': 'Mode debug'}

# ================= MOMENTUM GATE =================

def calculate_momentum_gate(df, direction, momentum_data):
    """Gate toujours vrai en mode debug"""
    return True, {'direction': direction, 'gate_score': 3, 'stoch_diff': 10, 'rsi_slope_ok': True, 'price_momentum_ok': True}

# ================= ANALYSE MOMENTUM =================

def analyze_momentum_with_filters(df):
    """Analyse momentum avec debug détaillé"""
    if len(df) < 10:
        print("⚠️ MOMENTUM: Moins de 10 bougies")
        return {
            'rsi': 50,
            'stoch_k': 50,
            'stoch_d': 50,
            'prev_rsi': 50,
            'buy': {'allowed': True, 'veto': False, 'score': 0, 'penalty': 0, 'reason': 'Données insuffisantes'},
            'sell': {'allowed': True, 'veto': False, 'score': 0, 'penalty': 0, 'reason': 'Données insuffisantes'},
            'gate_buy': True,
            'gate_sell': True,
            'violations': []
        }
    
    try:
        # Calcul RSI
        rsi = RSIIndicator(close=df['close'], window=14).rsi()
        current_rsi = float(rsi.iloc[-1]) if len(rsi) > 0 else 50
        
        # Calcul Stochastique
        stoch = StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14,
            smooth_window=3
        )
        stoch_k = stoch.stoch()
        stoch_d = stoch.stoch_signal()
        
        current_stoch_k = float(stoch_k.iloc[-1]) if len(stoch_k) > 0 else 50
        current_stoch_d = float(stoch_d.iloc[-1]) if len(stoch_d) > 0 else 50
        
        print(f"📈 MOMENTUM CALCULÉ: RSI={current_rsi:.1f}, Stoch K={current_stoch_k:.1f}, Stoch D={current_stoch_d:.1f}")
        
    except Exception as e:
        print(f"❌ ERREUR MOMENTUM: {e}")
        current_rsi = 50
        current_stoch_k = 50
        current_stoch_d = 50
    
    # Scores basiques
    buy_score = 25  # Score de base pour BUY
    sell_score = 25  # Score de base pour SELL
    
    # Ajustement basé sur RSI
    if current_rsi < 50:
        buy_score += 10
        sell_score -= 5
    else:
        sell_score += 10
        buy_score -= 5
    
    # Ajustement basé sur Stoch
    if current_stoch_k < 50:
        buy_score += 10
        sell_score -= 5
    else:
        sell_score += 10
        buy_score -= 5
    
    return {
        'rsi': current_rsi,
        'stoch_k': current_stoch_k,
        'stoch_d': current_stoch_d,
        'prev_rsi': current_rsi,
        'buy': {'allowed': True, 'veto': False, 'score': buy_score, 'penalty': 0, 'reason': f'RSI:{current_rsi:.1f}, Stoch:{current_stoch_k:.1f}'},
        'sell': {'allowed': True, 'veto': False, 'score': sell_score, 'penalty': 0, 'reason': f'RSI:{current_rsi:.1f}, Stoch:{current_stoch_k:.1f}'},
        'gate_buy': True,
        'gate_sell': True,
        'gate_debug': {'buy': {}, 'sell': {}},
        'violations': []
    }

# ================= BOLLINGER BANDS =================

def analyze_bollinger_bands(df):
    """Analyse BB avec debug détaillé"""
    if len(df) < 20:
        print("⚠️ BB: Moins de 20 bougies")
        return {
            'bb_position': 50,
            'buy': {'allowed': True, 'veto': False, 'score': 20, 'penalty': 0, 'reason': 'Données insuffisantes'},
            'sell': {'allowed': True, 'veto': False, 'score': 20, 'penalty': 0, 'reason': 'Données insuffisantes'},
            'price_above_middle': True
        }
    
    try:
        bb = BollingerBands(
            close=df['close'],
            window=SAINT_GRAAL_CONFIG['bollinger_config']['window'],
            window_dev=SAINT_GRAAL_CONFIG['bollinger_config']['window_dev']
        )
        
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        bb_middle = bb.bollinger_mavg()
        
        current_price = float(df.iloc[-1]['close'])
        current_upper = float(bb_upper.iloc[-1])
        current_lower = float(bb_lower.iloc[-1])
        
        # Position BB
        if current_upper != current_lower:
            bb_position = ((current_price - current_lower) / (current_upper - current_lower)) * 100
        else:
            bb_position = 50
        
        print(f"📊 BB CALCULÉ: Position={bb_position:.1f}%, Price={current_price:.5f}, Upper={current_upper:.5f}, Lower={current_lower:.5f}")
        
        # Scores basiques
        buy_score = 20
        sell_score = 20
        
        # Ajustement position
        if bb_position < 50:
            buy_score += 15
            sell_score -= 5
        else:
            sell_score += 15
            buy_score -= 5
        
        return {
            'bb_position': bb_position,
            'buy': {'allowed': True, 'veto': False, 'score': buy_score, 'penalty': 0, 'reason': f'BB Pos:{bb_position:.1f}%'},
            'sell': {'allowed': True, 'veto': False, 'score': sell_score, 'penalty': 0, 'reason': f'BB Pos:{bb_position:.1f}%'},
            'price_above_middle': current_price > float(bb_middle.iloc[-1]) if len(bb_middle) > 0 else True
        }
        
    except Exception as e:
        print(f"❌ ERREUR BB: {e}")
        return {
            'bb_position': 50,
            'buy': {'allowed': True, 'veto': False, 'score': 20, 'penalty': 0, 'reason': f'Erreur:{str(e)[:50]}'},
            'sell': {'allowed': True, 'veto': False, 'score': 20, 'penalty': 0, 'reason': f'Erreur:{str(e)[:50]}'},
            'price_above_middle': True
        }

# ================= FONCTIONS DE FILTRAGE SIMPLIFIÉES =================

def analyze_atr_volatility(df):
    """ATR toujours valide en debug"""
    return {'valid': True, 'reason': 'ATR désactivé', 'score': 10, 'atr_pips': 10}

def analyze_m5_trend(df):
    """M5 toujours neutre en debug"""
    return {'trend': 'NEUTRAL', 'reason': 'M5 désactivé', 'score': 10}

def detect_swing_extremes(df):
    """Swing désactivé"""
    return {'is_swing_high': False, 'is_swing_low': False}

def analyze_micro_momentum(df, direction):
    """Micro momentum toujours valide"""
    return {'valid': True, 'score': 10, 'reason': 'Micro momentum activé'}

def check_confidence_killers(df, direction, momentum_data):
    """Pas de killers en debug"""
    return 0, []

def calculate_confidence(score):
    """Confiance basique"""
    return max(60, min(95, int(score / 2 + 50)))

# ================= FONCTION PRINCIPALE DEBUG =================

def analyze_pair_for_signals(df):
    """
    🔥 Analyse en mode DEBUG - Version ultra permissive
    """
    print(f"\n{'='*80}")
    print(f"🔍 DEBUG COMPLET - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*80}")
    
    if df is None or len(df) == 0:
        print("❌ DONNÉES: DataFrame vide ou None")
        return None
    
    print(f"📊 DONNÉES: {len(df)} bougies, Colonnes: {list(df.columns)}")
    
    # Vérifier les colonnes nécessaires
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ COLONNES MANQUANTES: {missing_cols}")
        print(f"   Colonnes disponibles: {list(df.columns)}")
        return None
    
    # Afficher les dernières bougies
    print(f"📈 DERNIÈRES BOUGIES:")
    for i in range(min(3, len(df))):
        idx = -1 - i
        row = df.iloc[idx]
        print(f"   [{i+1}] O:{row['open']:.5f} H:{row['high']:.5f} L:{row['low']:.5f} C:{row['close']:.5f}")
    
    current_price = float(df.iloc[-1]['close'])
    print(f"\n💰 PRIX ACTUEL: {current_price:.5f}")
    
    # 1. Momentum
    print(f"\n{'='*40}")
    print(f"1️⃣  ANALYSE MOMENTUM")
    print(f"{'='*40}")
    momentum = analyze_momentum_with_filters(df)
    print(f"   RSI: {momentum['rsi']:.1f}")
    print(f"   Stoch K/D: {momentum['stoch_k']:.1f}/{momentum['stoch_d']:.1f}")
    print(f"   Score BUY: {momentum['buy']['score']}")
    print(f"   Score SELL: {momentum['sell']['score']}")
    
    # 2. Bollinger Bands
    print(f"\n{'='*40}")
    print(f"2️⃣  ANALYSE BOLLINGER")
    print(f"{'='*40}")
    bb = analyze_bollinger_bands(df)
    print(f"   Position BB: {bb['bb_position']:.1f}%")
    print(f"   Score BUY: {bb['buy']['score']}")
    print(f"   Score SELL: {bb['sell']['score']}")
    
    # Calcul des scores totaux
    buy_score_total = momentum['buy']['score'] + bb['buy']['score']
    sell_score_total = momentum['sell']['score'] + bb['sell']['score']
    
    print(f"\n{'='*40}")
    print(f"🎯 SCORES TOTAUX")
    print(f"{'='*40}")
    print(f"   BUY: {momentum['buy']['score']} (M) + {bb['buy']['score']} (BB) = {buy_score_total:.1f}")
    print(f"   SELL: {momentum['sell']['score']} (M) + {bb['sell']['score']} (BB) = {sell_score_total:.1f}")
    
    min_score = SAINT_GRAAL_CONFIG['signal_validation']['min_score']
    print(f"\n📏 SEUIL MINIMUM: {min_score}")
    
    # Vérification des conditions
    conditions_buy = (
        not momentum['buy']['veto'] and 
        not bb['buy']['veto'] and 
        momentum['buy']['allowed'] and 
        bb['buy']['allowed'] and
        momentum['gate_buy'] and
        buy_score_total >= min_score
    )
    
    conditions_sell = (
        not momentum['sell']['veto'] and 
        not bb['sell']['veto'] and 
        momentum['sell']['allowed'] and 
        bb['sell']['allowed'] and
        momentum['gate_sell'] and
        sell_score_total >= min_score
    )
    
    print(f"\n✅ CONDITIONS BUY: {'VRAI' if conditions_buy else 'FAUX'}")
    print(f"   - Pas de veto MOMENTUM: {'OUI' if not momentum['buy']['veto'] else 'NON'}")
    print(f"   - Pas de veto BB: {'OUI' if not bb['buy']['veto'] else 'NON'}")
    print(f"   - Momentum autorisé: {'OUI' if momentum['buy']['allowed'] else 'NON'}")
    print(f"   - BB autorisé: {'OUI' if bb['buy']['allowed'] else 'NON'}")
    print(f"   - Gate BUY: {'OUI' if momentum['gate_buy'] else 'NON'}")
    print(f"   - Score >= {min_score}: {'OUI' if buy_score_total >= min_score else f'NON ({buy_score_total:.1f})'}")
    
    print(f"\n✅ CONDITIONS SELL: {'VRAI' if conditions_sell else 'FAUX'}")
    print(f"   - Pas de veto MOMENTUM: {'OUI' if not momentum['sell']['veto'] else 'NON'}")
    print(f"   - Pas de veto BB: {'OUI' if not bb['sell']['veto'] else 'NON'}")
    print(f"   - Momentum autorisé: {'OUI' if momentum['sell']['allowed'] else 'NON'}")
    print(f"   - BB autorisé: {'OUI' if bb['sell']['allowed'] else 'NON'}")
    print(f"   - Gate SELL: {'OUI' if momentum['gate_sell'] else 'NON'}")
    print(f"   - Score >= {min_score}: {'OUI' if sell_score_total >= min_score else f'NON ({sell_score_total:.1f})'}")
    
    # Décision finale
    if conditions_buy and buy_score_total >= sell_score_total:
        final_score = buy_score_total
        direction = "CALL"
        reason = f"BUY Score: {final_score:.1f} | RSI: {momentum['rsi']:.1f} | Stoch: {momentum['stoch_k']:.1f} | BB: {bb['bb_position']:.1f}%"
        confidence = calculate_confidence(final_score)
        
        print(f"\n🎯 SIGNAL BUY DÉTECTÉ!")
        print(f"   Direction: {direction}")
        print(f"   Score: {final_score:.1f}")
        print(f"   Confiance: {confidence}%")
        print(f"   Raison: {reason}")
        
        return {
            'direction': direction,
            'quality': "DEBUG",
            'score': round(final_score, 1),
            'confidence': confidence,
            'expiration_minutes': 5,
            'reason': reason,
            'details': {
                'market_state': 'DEBUG',
                'momentum_score': momentum['buy']['score'],
                'bb_score': bb['buy']['score'],
                'micro_score': 0,
                'atr_score': 0,
                'm5_trend': 'NEUTRAL',
                'rsi': momentum['rsi'],
                'stoch': momentum['stoch_k'],
                'bb_position': bb['bb_position'],
                'atr_pips': 0,
                'gate_buy': momentum['gate_buy'],
                'gate_sell': momentum['gate_sell'],
                'confidence_killers': [],
                'swing_adjustment': {'buy': 0, 'sell': 0}
            }
        }
    
    elif conditions_sell:
        final_score = sell_score_total
        direction = "PUT"
        reason = f"SELL Score: {final_score:.1f} | RSI: {momentum['rsi']:.1f} | Stoch: {momentum['stoch_k']:.1f} | BB: {bb['bb_position']:.1f}%"
        confidence = calculate_confidence(final_score)
        
        print(f"\n🎯 SIGNAL SELL DÉTECTÉ!")
        print(f"   Direction: {direction}")
        print(f"   Score: {final_score:.1f}")
        print(f"   Confiance: {confidence}%")
        print(f"   Raison: {reason}")
        
        return {
            'direction': direction,
            'quality': "DEBUG",
            'score': round(final_score, 1),
            'confidence': confidence,
            'expiration_minutes': 5,
            'reason': reason,
            'details': {
                'market_state': 'DEBUG',
                'momentum_score': momentum['sell']['score'],
                'bb_score': bb['sell']['score'],
                'micro_score': 0,
                'atr_score': 0,
                'm5_trend': 'NEUTRAL',
                'rsi': momentum['rsi'],
                'stoch': momentum['stoch_k'],
                'bb_position': bb['bb_position'],
                'atr_pips': 0,
                'gate_buy': momentum['gate_buy'],
                'gate_sell': momentum['gate_sell'],
                'confidence_killers': [],
                'swing_adjustment': {'buy': 0, 'sell': 0}
            }
        }
    
    else:
        print(f"\n❌ AUCUN SIGNAL VALIDE")
        
        # Vérifier quelle condition a échoué
        if buy_score_total < min_score and sell_score_total < min_score:
            print(f"   ❌ Score insuffisant: BUY={buy_score_total:.1f}, SELL={sell_score_total:.1f} (minimum={min_score})")
        elif momentum['buy']['veto'] or momentum['sell']['veto']:
            print(f"   ❌ Veto momentum actif")
        elif bb['buy']['veto'] or bb['sell']['veto']:
            print(f"   ❌ Veto BB actif")
        elif not momentum['gate_buy'] or not momentum['gate_sell']:
            print(f"   ❌ Gate momentum bloqué")
        
        return None

# ================= FONCTIONS DE COMPATIBILITÉ =================

def get_signal_saint_graal(df, signal_count=0, total_signals=8, return_dict=False):
    """
    🔥 Fonction principale pour le bot - VERSION DEBUG
    """
    print(f"\n{'='*80}")
    print(f"🎯 APPEL get_signal_saint_graal - Signal #{signal_count}")
    print(f"{'='*80}")
    
    if df is None:
        print("❌ DataFrame est None")
        return None
    
    if len(df) < 5:
        print(f"⚠️  Données insuffisantes: {len(df)} bougies")
        # Créer des données de test si besoin
        try:
            dates = pd.date_range(end=datetime.now(), periods=20, freq='1min')
            test_data = {
                'open': np.random.randn(20).cumsum() + 100,
                'high': np.random.randn(20).cumsum() + 101,
                'low': np.random.randn(20).cumsum() + 99,
                'close': np.random.randn(20).cumsum() + 100,
            }
            df = pd.DataFrame(test_data, index=dates)
            print(f"✅ Données de test générées: {len(df)} bougies")
        except:
            return None
    
    signal = analyze_pair_for_signals(df)
    
    if signal:
        signal['signal_count'] = signal_count
        signal['total_signals'] = total_signals
        signal['mode'] = "DEBUG V9.3"
        
        print(f"\n✅ SIGNAL RETOURNÉ AU BOT:")
        print(f"   Direction: {signal['direction']}")
        print(f"   Score: {signal['score']}")
        print(f"   Confiance: {signal['confidence']}%")
        
        return signal
    else:
        print(f"\n❌ AUCUN SIGNAL RETOURNÉ")
        return None

# Alias pour compatibilité
get_binary_signal = get_signal_saint_graal

# ================= FONCTION DE TEST =================

def test_with_sample_data():
    """Test avec des données d'exemple"""
    print("\n🧪 TEST AVEC DONNÉES D'EXEMPLE")
    print("="*60)
    
    # Créer des données de test
    dates = pd.date_range(end=datetime.now(), periods=100, freq='1min')
    np.random.seed(42)
    
    # Prix qui tendent à la hausse pour BUY
    base_price = 100
    trend = np.linspace(0, 10, 100)
    noise = np.random.randn(100) * 2
    closes = base_price + trend + noise
    
    # Calculer OHLC
    opens = closes - np.random.rand(100) * 0.5
    highs = closes + np.random.rand(100) * 0.8
    lows = closes - np.random.rand(100) * 0.8
    
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes
    }, index=dates)
    
    print(f"📊 Données de test créées: {len(df)} bougies")
    print(f"   Prix: {df['close'].iloc[-1]:.5f}")
    print(f"   Plage: {df['close'].min():.5f} - {df['close'].max():.5f}")
    
    # Tester la fonction
    signal = get_signal_saint_graal(df, 1, 8, True)
    
    if signal:
        print(f"\n✅ TEST RÉUSSI: Signal {signal['direction']} détecté")
        return True
    else:
        print(f"\n❌ TEST ÉCHOUÉ: Aucun signal détecté")
        return False

# ================= INITIALISATION =================

if __name__ == "__main__":
    print("🚀 STRATÉGIE BINAIRE M1 PRO - VERSION 9.3 DEBUG")
    print("🔥 MODE DÉBOGAGE COMPLET - TRÈS PERMISSIF")
    print("\n" + "="*80)
    print("CONFIGURATION ULTRA PERMISSIVE:")
    print("✅ Tous les veto désactivés")
    print("✅ Score minimum: 50 seulement")
    print("✅ Gates momentum désactivés")
    print("✅ Micro momentum désactivé")
    print("✅ ATR désactivé")
    print("✅ M5 désactivé")
    print("✅ Swing filter désactivé")
    print("✅ Mode marché désactivé")
    print("="*80)
    
    print("\n🎯 TEST AUTOMATIQUE:")
    if test_with_sample_data():
        print("✅ La logique fonctionne avec des données de test")
    else:
        print("❌ Problème détecté même avec données de test")
    
    print("\n" + "="*80)
    print("INSTRUCTIONS:")
    print("1. Utilisez ce fichier comme utils.py")
    print("2. Lancez votre bot signal_bot.py")
    print("3. Regardez les logs pour comprendre le problème")
    print("4. Le système devrait générer des signaux maintenant")
    print("="*80)
