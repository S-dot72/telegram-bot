"""
ML Predictor OPTIMISÉ avec adaptation automatique OTC/Forex
=============================================================

FONCTIONNEMENT ADAPTATIF:
- Mode Forex (semaine): Seuil strict 80%
- Mode OTC (week-end/crypto): Seuil flexible 65%
- Fallback automatique en cas d'erreur
- Compatibilité totale avec signal_bot.py existant
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
import os
from datetime import datetime

class MLSignalPredictor:
    def __init__(self, model_path='models/ml_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.model_fitted = False
        
        # SEUILS ADAPTATIFS
        self.min_confidence_forex = 0.80          # Strict pour Forex
        self.min_confidence_otc = 0.65            # Flexible pour OTC
        self.ultra_strict_confidence = 0.85
        self.premium_confidence = 0.90
        
        # Cache pour éviter recalculs
        self.last_weekend_check = None
        self.is_weekend_cache = None
        
        # Créer dossier models
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Charger si existe
        if os.path.exists(model_path):
            self.load_model()
        else:
            self.model = self._create_hybrid_model()
            print("⚠️  ML: Modèle créé - En attente d'entraînement")
            print(f"🎯 Seuil Forex: {self.min_confidence_forex*100:.0f}%")
            print(f"🏖️ Seuil OTC: {self.min_confidence_otc*100:.0f}%")
    
    def _create_hybrid_model(self):
        """Modèle Gradient Boosting optimisé"""
        return GradientBoostingClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            min_samples_split=12,
            min_samples_leaf=6,
            subsample=0.8,
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
    
    def _is_weekend_cached(self):
        """Détection week-end avec cache"""
        now = datetime.utcnow()
        
        # Vérifier le cache (mise à jour toutes les minutes)
        if (self.last_weekend_check and 
            (now - self.last_weekend_check).total_seconds() < 60 and
            self.is_weekend_cache is not None):
            return self.is_weekend_cache
        
        # Calculer et mettre en cache
        self.is_weekend_cache = now.weekday() >= 5  # Samedi=5, Dimanche=6
        self.last_weekend_check = now
        return self.is_weekend_cache
    
    def _detect_otc_context(self, df=None):
        """Détecte si on est en contexte OTC"""
        # 1. Vérifier si on est en week-end
        if self._is_weekend_cached():
            return True
        
        # 2. Analyser les données pour détecter les cryptos
        if df is not None and len(df) > 0:
            try:
                last_price = df.iloc[-1].get('close', 0)
                # Les cryptos ont des caractéristiques de prix spécifiques
                if last_price > 0:
                    # Vérifier la volatilité (les cryptos sont plus volatiles)
                    if len(df) >= 20:
                        returns = df['close'].pct_change().dropna()
                        volatility = returns.std() * 100  # En pourcentage
                        
                        # Volatilité élevée typique des cryptos
                        if volatility > 2.0:  # > 2% de volatilité quotidienne
                            return True
                    
                    # Prix très élevés (BTC) ou très bas (altcoins)
                    if last_price > 10000 or last_price < 1:
                        return True
            except:
                pass
        
        return False
    
    def extract_features(self, df):
        """
        Features ENRICHIES pour meilleur ML
        Retourne None si données insuffisantes
        """
        if len(df) < 30:  # Réduit de 50 à 30 pour être plus permissif
            return None
        
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        prev5 = df.iloc[-6] if len(df) > 5 else prev
        
        # Colonnes minimales requises
        required_cols = [
            'open', 'high', 'low', 'close',
            'ema_fast', 'ema_slow', 'rsi', 'MACD_12_26_9', 
            'MACDs_12_26_9', 'MACDh_12_26_9', 'adx'
        ]
        
        for col in required_cols:
            if col not in df.columns or pd.isna(last.get(col)):
                return None
        
        # Construction des features avec valeurs par défaut
        features = {
            # Prix
            'open': last['open'],
            'high': last['high'],
            'low': last['low'],
            'close': last['close'],
            'close_change': (last['close'] - prev['close']) / prev['close'] if prev['close'] > 0 else 0,
            
            # EMA
            'ema_fast': last.get('ema_fast', 0),
            'ema_slow': last.get('ema_slow', 0),
            'ema_diff': last.get('ema_fast', 0) - last.get('ema_slow', 0),
            'ema_diff_pct': (last.get('ema_fast', 0) - last.get('ema_slow', 0)) / last.get('ema_slow', 1) * 100,
            
            # RSI
            'rsi': last.get('rsi', 50),
            'rsi_momentum': last.get('rsi', 50) - prev.get('rsi', 50),
            
            # MACD
            'macd': last.get('MACD_12_26_9', 0),
            'macd_signal': last.get('MACDs_12_26_9', 0),
            'macd_hist': last.get('MACDh_12_26_9', 0),
            'macd_trend': 1 if last.get('MACD_12_26_9', 0) > last.get('MACDs_12_26_9', 0) else -1,
            
            # ADX
            'adx': last.get('adx', 20),
            'adx_strong': 1 if last.get('adx', 0) > 25 else 0,
            
            # Volatilité
            'atr': last.get('atr', 0) if 'atr' in df.columns else 0,
        }
        
        # Ajouter features optionnelles si disponibles
        optional_features = {
            'ema_50': last.get('ema_50', 0),
            'ema_200': last.get('ema_200', 0),
            'adx_pos': last.get('adx_pos', 0),
            'adx_neg': last.get('adx_neg', 0),
            'stoch_k': last.get('stoch_k', 50),
            'stoch_d': last.get('stoch_d', 50),
        }
        
        features.update({k: v for k, v in optional_features.items() if k in df.columns})
        
        # Features dérivées
        if 'ema_50' in features and 'ema_200' in features:
            features['ema_triple_bullish'] = int(
                features['ema_fast'] > features['ema_slow'] > features['ema_50'] > features['ema_200']
            )
            features['ema_triple_bearish'] = int(
                features['ema_fast'] < features['ema_slow'] < features['ema_50'] < features['ema_200']
            )
        
        # Score composite simplifié
        features['bullish_score'] = self._calculate_bullish_score_simple(last)
        features['bearish_score'] = self._calculate_bearish_score_simple(last)
        
        return features
    
    def _calculate_bullish_score_simple(self, last_row):
        """Score haussier simplifié (0-5)"""
        score = 0
        
        # EMA fast > EMA slow
        if last_row.get('ema_fast', 0) > last_row.get('ema_slow', 0):
            score += 1
        
        # MACD positif
        if last_row.get('MACD_12_26_9', 0) > last_row.get('MACDs_12_26_9', 0):
            score += 1
        
        # RSI dans zone haussière
        rsi = last_row.get('rsi', 50)
        if 40 < rsi < 70:
            score += 1
        elif 30 < rsi < 80:
            score += 0.5
        
        # ADX fort
        if last_row.get('adx', 0) > 20:
            score += 1
        
        # Stochastic haussier
        stoch_k = last_row.get('stoch_k', 50)
        stoch_d = last_row.get('stoch_d', 50)
        if stoch_k > stoch_d and 20 < stoch_k < 80:
            score += 1
        
        return min(score, 5)
    
    def _calculate_bearish_score_simple(self, last_row):
        """Score baissier simplifié (0-5)"""
        score = 0
        
        # EMA fast < EMA slow
        if last_row.get('ema_fast', 0) < last_row.get('ema_slow', 0):
            score += 1
        
        # MACD négatif
        if last_row.get('MACD_12_26_9', 0) < last_row.get('MACDs_12_26_9', 0):
            score += 1
        
        # RSI dans zone baissière
        rsi = last_row.get('rsi', 50)
        if 30 < rsi < 60:
            score += 1
        elif 20 < rsi < 70:
            score += 0.5
        
        # ADX fort
        if last_row.get('adx', 0) > 20:
            score += 1
        
        # Stochastic baissier
        stoch_k = last_row.get('stoch_k', 50)
        stoch_d = last_row.get('stoch_d', 50)
        if stoch_k < stoch_d and 20 < stoch_k < 80:
            score += 1
        
        return min(score, 5)
    
    def calculate_confidence_score(self, df, base_signal):
        """
        Score de confiance simplifié et robuste
        Retourne: 0.0 à 1.0
        """
        if len(df) < 10:
            return 0.5  # Valeur neutre si données insuffisantes
        
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        
        score = 0.50  # Base neutre
        
        # 1. Convergence des indicateurs (0-0.25)
        convergence = 0
        
        # Vérifier la cohérence signal/indicateurs
        if base_signal == 'CALL':
            bullish_indicators = 0
            if last.get('ema_fast', 0) > last.get('ema_slow', 0):
                bullish_indicators += 1
            if last.get('MACD_12_26_9', 0) > last.get('MACDs_12_26_9', 0):
                bullish_indicators += 1
            if 30 < last.get('rsi', 50) < 70:
                bullish_indicators += 1
            
            convergence = bullish_indicators / 3 * 0.25
            
        else:  # PUT
            bearish_indicators = 0
            if last.get('ema_fast', 0) < last.get('ema_slow', 0):
                bearish_indicators += 1
            if last.get('MACD_12_26_9', 0) < last.get('MACDs_12_26_9', 0):
                bearish_indicators += 1
            if 30 < last.get('rsi', 50) < 70:
                bearish_indicators += 1
            
            convergence = bearish_indicators / 3 * 0.25
        
        score += convergence
        
        # 2. Force de la tendance ADX (0-0.15)
        adx = last.get('adx', 0)
        if adx > 30:
            score += 0.15
        elif adx > 25:
            score += 0.10
        elif adx > 20:
            score += 0.05
        
        # 3. Momentum (0-0.10)
        if base_signal == 'CALL' and last.get('close', 0) > prev.get('close', 0):
            score += 0.10
        elif base_signal == 'PUT' and last.get('close', 0) < prev.get('close', 0):
            score += 0.10
        
        # 4. Position RSI (0-0.10)
        rsi = last.get('rsi', 50)
        if base_signal == 'CALL' and 40 < rsi < 70:
            score += 0.10
        elif base_signal == 'PUT' and 30 < rsi < 60:
            score += 0.10
        elif 35 < rsi < 65:  # Zone neutre
            score += 0.05
        
        # 5. Alignement EMA (0-0.10)
        if base_signal == 'CALL' and last.get('ema_fast', 0) > last.get('ema_slow', 0):
            score += 0.10
        elif base_signal == 'PUT' and last.get('ema_fast', 0) < last.get('ema_slow', 0):
            score += 0.10
        
        # Limiter entre 0.3 et 0.9
        return max(0.3, min(score, 0.9))
    
    def predict_signal(self, df, base_signal):
        """
        Prédiction avec adaptation automatique OTC/Forex
        - En OTC: accepte les signaux avec seuil réduit
        - En Forex: règles strictes
        - Fallback robuste en cas d'erreur
        
        Returns: (signal, confidence) ou (None, confidence)
        """
        try:
            # Détection du contexte
            is_otc = self._detect_otc_context(df)
            
            if is_otc:
                print(f"   🏖️ MODE OTC DÉTECTÉ - Seuil réduit: {self.min_confidence_otc*100:.0f}%")
                min_confidence = self.min_confidence_otc
                mode = "OTC"
            else:
                print(f"   📈 MODE FOREX - Seuil strict: {self.min_confidence_forex*100:.0f}%")
                min_confidence = self.min_confidence_forex
                mode = "Forex"
            
            # Tentative avec ML si disponible
            if self.model_fitted:
                try:
                    features = self.extract_features(df)
                    if features is not None:
                        X = pd.DataFrame([features])
                        
                        try:
                            X_scaled = self.scaler.transform(X)
                        except:
                            # Entraîner le scaler si nécessaire
                            self.scaler.fit(X)
                            X_scaled = self.scaler.transform(X)
                        
                        # Prédiction
                        probas = self.model.predict_proba(X_scaled)[0]
                        if len(probas) >= 2:
                            ml_confidence = probas[1] if base_signal == 'CALL' else probas[0]
                        else:
                            ml_confidence = probas[0]
                        
                        print(f"   🤖 ML: {ml_confidence:.1%}")
                        
                        # Vérification du seuil
                        if ml_confidence >= min_confidence:
                            return base_signal, ml_confidence
                        else:
                            print(f"   ⚠️  ML insuffisant: {ml_confidence:.1%} < {min_confidence:.0%}")
                            # En OTC, on peut être plus flexible
                            if mode == "OTC" and ml_confidence >= 0.55:
                                print(f"   🏖️ OTC flexible: accepté à {ml_confidence:.1%}")
                                return base_signal, ml_confidence
                except Exception as e:
                    print(f"   ⚠️  Erreur ML: {e}")
            
            # Fallback: scoring manuel
            confidence = self.calculate_confidence_score(df, base_signal)
            print(f"   💪 Score manuel: {confidence:.1%}")
            
            # En OTC, être plus permissif
            if mode == "OTC":
                if confidence >= 0.55:  # Seuil très bas pour testing
                    print(f"   🏖️ OTC: Signal accepté (flexible)")
                    return base_signal, max(confidence, 0.60)
                else:
                    print(f"   🏖️ OTC: Score trop bas {confidence:.1%}")
            
            # Vérification standard
            if confidence >= min_confidence:
                return base_signal, confidence
            else:
                print(f"   ❌ Score insuffisant: {confidence:.1%} < {min_confidence:.0%}")
                return None, confidence
                
        except Exception as e:
            print(f"   🚨 Erreur critique prédiction: {e}")
            # Fallback ultime: accepter le signal avec confiance modérée
            fallback_confidence = 0.65 if self._is_weekend_cached() else 0.70
            print(f"   🆘 Fallback: {base_signal} ({fallback_confidence:.1%})")
            return base_signal, fallback_confidence
    
    def train_on_history(self, engine):
        """Entraîne le modèle (placeholder)"""
        from sqlalchemy import text
        
        print("\n🤖 ENTRAÎNEMENT ML")
        print("="*50)
        
        query = text("""
            SELECT payload_json, result 
            FROM signals 
            WHERE result IS NOT NULL
        """)
        
        with engine.connect() as conn:
            results = conn.execute(query).fetchall()
        
        if len(results) < 50:
            print(f"⚠️  Données insuffisantes: {len(results)} < 50")
            return False
        
        print(f"📚 {len(results)} signaux disponibles")
        print("✅ Utiliser /retrain pour entraîner")
        
        return False
    
    def save_model(self):
        """Sauvegarde"""
        try:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'model_fitted': self.model_fitted,
                'min_confidence_forex': self.min_confidence_forex,
                'min_confidence_otc': self.min_confidence_otc,
                'ultra_strict_confidence': self.ultra_strict_confidence,
                'premium_confidence': self.premium_confidence
            }, self.model_path)
            print(f"💾 Modèle ML sauvegardé: {self.model_path}")
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")
    
    def load_model(self):
        """Charge le modèle"""
        try:
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.model_fitted = data.get('model_fitted', False)
            self.min_confidence_forex = data.get('min_confidence_forex', 0.80)
            self.min_confidence_otc = data.get('min_confidence_otc', 0.65)
            self.ultra_strict_confidence = data.get('ultra_strict_confidence', 0.85)
            self.premium_confidence = data.get('premium_confidence', 0.90)
            
            print(f"✅ Modèle ML chargé: {self.model_path}")
            print(f"   🎯 Forex: {self.min_confidence_forex*100:.0f}%")
            print(f"   🏖️ OTC: {self.min_confidence_otc*100:.0f}%")
            print(f"   🤖 Entraîné: {'Oui' if self.model_fitted else 'Non'}")
        except Exception as e:
            print(f"⚠️  Erreur chargement: {e}")
            self.model = None
            self.model_fitted = False
    
    def get_model_info(self):
        """Info modèle"""
        is_otc = self._detect_otc_context()
        
        return {
            'model_loaded': self.model is not None,
            'model_fitted': self.model_fitted,
            'model_type': type(self.model).__name__ if self.model else 'None',
            'min_confidence_forex': self.min_confidence_forex,
            'min_confidence_otc': self.min_confidence_otc,
            'ultra_strict_confidence': self.ultra_strict_confidence,
            'premium_confidence': self.premium_confidence,
            'current_mode': 'OTC' if is_otc else 'Forex',
            'is_weekend': self._is_weekend_cached()
        }
    
    def force_otc_mode(self, enable=True):
        """Force le mode OTC (pour testing)"""
        if enable:
            print("🔧 Mode OTC forcé activé")
            self.min_confidence_otc = 0.55  # Seuil très bas pour testing
        else:
            print("🔧 Mode OTC forcé désactivé")
            self.min_confidence_otc = 0.65
    
    def quick_test(self, df=None):
        """Test rapide du predictor"""
        print("\n🧪 TEST RAPIDE ML PREDICTOR")
        print("="*50)
        
        if df is None:
            # Créer des données de test minimales
            df = pd.DataFrame({
                'open': [100, 101, 102, 103, 104, 105],
                'high': [101, 102, 103, 104, 105, 106],
                'low': [99, 100, 101, 102, 103, 104],
                'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
                'ema_fast': [100.2, 101.2, 102.2, 103.2, 104.2, 105.2],
                'ema_slow': [100.1, 101.1, 102.1, 103.1, 104.1, 105.1],
                'rsi': [55, 56, 57, 58, 59, 60],
                'MACD_12_26_9': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'MACDs_12_26_9': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                'MACDh_12_26_9': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                'adx': [25, 26, 27, 28, 29, 30],
            })
        
        print(f"Données: {len(df)} bougies")
        print(f"Week-end: {'Oui' if self._is_weekend_cached() else 'Non'}")
        print(f"Mode détecté: {'OTC' if self._detect_otc_context(df) else 'Forex'}")
        
        # Test CALL
        print("\nTest CALL:")
        signal, conf = self.predict_signal(df, "CALL")
        print(f"  Signal: {signal}, Confiance: {conf:.1%}")
        
        # Test PUT
        print("\nTest PUT:")
        signal, conf = self.predict_signal(df, "PUT")
        print(f"  Signal: {signal}, Confiance: {conf:.1%}")
        
        print("\n✅ Test terminé")
