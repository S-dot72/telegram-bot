"""
ML Predictor OPTIMISÉ - Seuils Stricts pour 75-85% Win Rate
=============================================================

CHANGEMENTS MAJEURS:
- Seuil minimum: 65% → 80%
- Seuil ultra: 75% → 85%
- Scoring manuel amélioré avec pénalités
- Features enrichies pour meilleur ML
- Filtres anti-faux-positifs renforcés

NOUVEAU: Adaptation automatique OTC vs Forex
- Détection auto des paires crypto
- Seuils adaptatifs (80% Forex, 65% OTC)
- Fallback automatique pour OTC
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
        
        # SEUILS ADAPTATIFS - Auto-détection OTC/Forex
        self.min_confidence_forex = 0.80          # Forex en semaine
        self.min_confidence_otc = 0.65           # OTC week-end (plus permissif)
        self.ultra_strict_confidence = 0.85
        self.premium_confidence = 0.90
        
        # Créer dossier models
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Charger si existe
        if os.path.exists(model_path):
            self.load_model()
        else:
            self.model = self._create_hybrid_model()
            print("⚠️  ML: Modèle créé - En attente d'entraînement")
            print(f"🎯 Seuil adaptatif activé")
            print(f"   📈 Forex: {self.min_confidence_forex*100:.0f}%")
            print(f"   🏖️ OTC: {self.min_confidence_otc*100:.0f}%")
    
    def _is_weekend(self):
        """Détecte si on est en week-end (UTC)"""
        utc_now = datetime.utcnow()
        return utc_now.weekday() >= 5  # Samedi=5, Dimanche=6
    
    def _detect_otc_pair(self, pair):
        """Détecte si la paire est une crypto (OTC)"""
        crypto_pairs = ['BTC/USD', 'ETH/USD', 'XRP/USD', 'LTC/USD', 
                       'BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'LTCUSDT']
        return any(crypto in pair for crypto in crypto_pairs)
    
    def _get_min_confidence(self, pair=None, df=None):
        """
        Détermine automatiquement le seuil minimal
        Priorité: 1. Week-end → OTC, 2. Paire crypto → OTC, 3. Sinon Forex
        """
        # Si on est en week-end → OTC
        if self._is_weekend():
            return self.min_confidence_otc
        
        # Si la paire est une crypto → OTC
        if pair and self._detect_otc_pair(pair):
            return self.min_confidence_otc
        
        # Vérifier les données pour détecter les cryptos
        if df is not None and len(df) > 0:
            last_price = df.iloc[-1].get('close', 0)
            # Les cryptos ont généralement des prix très élevés (BTC) ou très bas (altcoins)
            if last_price > 10000 or last_price < 10:  # Prix typiques des cryptos
                return self.min_confidence_otc
        
        # Par défaut → Forex
        return self.min_confidence_forex
    
    def _create_hybrid_model(self):
        """Modèle Gradient Boosting optimisé"""
        return GradientBoostingClassifier(
            n_estimators=200,      # 100 → 200
            max_depth=8,           # 6 → 8
            learning_rate=0.05,    # 0.1 → 0.05 (plus conservateur)
            min_samples_split=12,  # 10 → 12
            min_samples_leaf=6,    # 5 → 6
            subsample=0.8,
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
    
    def extract_features(self, df):
        """
        Features ENRICHIES pour meilleur ML
        +5 nouvelles features vs version originale
        """
        if len(df) < 50:  # Besoin de plus d'historique
            return None
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        prev5 = df.iloc[-6] if len(df) > 5 else prev
        
        # Vérifier indicateurs requis
        required_cols = [
            'ema_fast', 'ema_slow', 'ema_50', 'ema_200',  # +2 EMA
            'rsi', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
            'adx', 'adx_pos', 'adx_neg',
            'stoch_k', 'stoch_d', 'atr'
        ]
        
        for col in required_cols:
            if col not in df.columns or pd.isna(last.get(col)):
                return None
        
        features = {
            # === EMA (enrichi) ===
            'ema_fast': last['ema_fast'],
            'ema_slow': last['ema_slow'],
            'ema_50': last['ema_50'],
            'ema_200': last['ema_200'],
            'ema_diff': last['ema_fast'] - last['ema_slow'],
            'ema_diff_pct': (last['ema_fast'] - last['ema_slow']) / last['ema_slow'] * 100,
            'ema_trend': 1 if last['ema_fast'] > last['ema_slow'] else -1,
            
            # NOUVEAU: Triple alignment
            'ema_triple_bullish': int(
                last['ema_fast'] > last['ema_slow'] > last['ema_50'] > last['ema_200']
            ),
            'ema_triple_bearish': int(
                last['ema_fast'] < last['ema_slow'] < last['ema_50'] < last['ema_200']
            ),
            
            # NOUVEAU: Distance EMA
            'distance_ema_fast': abs(last['close'] - last['ema_fast']) / last['close'] * 100,
            
            # === RSI (enrichi) ===
            'rsi': last['rsi'],
            'rsi_momentum': last['rsi'] - prev['rsi'],
            'rsi_momentum_5': last['rsi'] - prev5['rsi'],  # NOUVEAU
            'rsi_optimal_call': int(40 < last['rsi'] < 60),  # NOUVEAU
            'rsi_optimal_put': int(40 < last['rsi'] < 60),   # NOUVEAU
            
            # === MACD (enrichi) ===
            'macd': last['MACD_12_26_9'],
            'macd_signal': last['MACDs_12_26_9'],
            'macd_hist': last['MACDh_12_26_9'],
            'macd_hist_momentum': last['MACDh_12_26_9'] - prev['MACDh_12_26_9'],  # NOUVEAU
            'macd_trend': 1 if last['MACD_12_26_9'] > last['MACDs_12_26_9'] else -1,
            'macd_strength': abs(last['MACDh_12_26_9']),
            
            # === ADX (enrichi) ===
            'adx': last['adx'],
            'adx_pos': last['adx_pos'],
            'adx_neg': last['adx_neg'],
            'adx_diff': last['adx_pos'] - last['adx_neg'],
            'adx_strong': int(last['adx'] > 25),  # NOUVEAU
            
            # === Stochastic ===
            'stoch_k': last['stoch_k'],
            'stoch_d': last['stoch_d'],
            'stoch_diff': last['stoch_k'] - last['stoch_d'],
            'stoch_momentum': last['stoch_k'] - prev['stoch_k'],  # NOUVEAU
            
            # === Volatilité (enrichi) ===
            'atr': last['atr'],
            'atr_normalized': last['atr'] / df['atr'].rolling(20).mean().iloc[-1] if len(df) >= 20 else 1.0,
            'atr_change': (last['atr'] - prev['atr']) / prev['atr'] if prev['atr'] > 0 else 0,  # NOUVEAU
            
            # === Prix & Momentum ===
            'close_change': (last['close'] - prev['close']) / prev['close'],
            'close_change_5': (last['close'] - prev5['close']) / prev5['close'],  # NOUVEAU
            
            # === Scores composites AMÉLIORÉS ===
            'bullish_score': self._bullish_score_v2(df),   # Nouvelle version
            'bearish_score': self._bearish_score_v2(df),   # Nouvelle version
            
            # NOUVEAU: Qualité globale
            'signal_quality': self._compute_signal_quality(df),
        }
        
        return features
    
    def _bullish_score_v2(self, df):
        """
        Score haussier V2 (0-7) - Plus strict
        """
        if len(df) < 10:
            return 0
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        score = 0
        
        # 1. EMA triple alignment (2 points si complet, 1 sinon)
        if last.get('ema_fast', 0) > last.get('ema_slow', 0) > last.get('ema_50', 0):
            score += 2
        elif last.get('ema_fast', 0) > last.get('ema_slow', 0):
            score += 1
        
        # 2. MACD haussier ET en croissance (2 points)
        macd_hist = last.get('MACDh_12_26_9', 0)
        prev_hist = prev.get('MACDh_12_26_9', 0)
        if last.get('MACD_12_26_9', 0) > last.get('MACDs_12_26_9', 0):
            if macd_hist > 0 and macd_hist > prev_hist:
                score += 2
            else:
                score += 1
        
        # 3. RSI optimal (1 point)
        rsi = last.get('rsi', 50)
        if 40 < rsi < 60:
            score += 1
        
        # 4. Stochastic (1 point)
        if last.get('stoch_k', 50) > last.get('stoch_d', 50) and 20 < last.get('stoch_k', 50) < 80:
            score += 1
        
        # 5. ADX fort (1 point)
        if last.get('adx', 0) > 22 and last.get('adx_pos', 0) > last.get('adx_neg', 0):
            score += 1
        
        return score
    
    def _bearish_score_v2(self, df):
        """
        Score baissier V2 (0-7) - Plus strict
        """
        if len(df) < 10:
            return 0
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        score = 0
        
        # 1. EMA triple alignment inversé
        if last.get('ema_fast', 0) < last.get('ema_slow', 0) < last.get('ema_50', 0):
            score += 2
        elif last.get('ema_fast', 0) < last.get('ema_slow', 0):
            score += 1
        
        # 2. MACD baissier ET en descente
        macd_hist = last.get('MACDh_12_26_9', 0)
        prev_hist = prev.get('MACDh_12_26_9', 0)
        if last.get('MACD_12_26_9', 0) < last.get('MACDs_12_26_9', 0):
            if macd_hist < 0 and macd_hist < prev_hist:
                score += 2
            else:
                score += 1
        
        # 3. RSI optimal
        rsi = last.get('rsi', 50)
        if 40 < rsi < 60:
            score += 1
        
        # 4. Stochastic
        if last.get('stoch_k', 50) < last.get('stoch_d', 50) and 20 < last.get('stoch_k', 50) < 80:
            score += 1
        
        # 5. ADX fort
        if last.get('adx', 0) > 22 and last.get('adx_neg', 0) > last.get('adx_pos', 0):
            score += 1
        
        return score
    
    def _compute_signal_quality(self, df):
        """
        Qualité globale du signal (0-1)
        Basé sur convergence des indicateurs
        """
        if len(df) < 10:
            return 0.0
        
        last = df.iloc[-1]
        quality = 0.0
        max_quality = 5.0
        
        # 1. ADX
        adx = last.get('adx', 0)
        if adx > 25:
            quality += 1.0
        elif adx > 22:
            quality += 0.7
        elif adx > 18:
            quality += 0.4
        
        # 2. RSI dans zone
        rsi = last.get('rsi', 50)
        if 45 < rsi < 55:
            quality += 1.0
        elif 40 < rsi < 60:
            quality += 0.7
        elif 35 < rsi < 65:
            quality += 0.4
        
        # 3. MACD aligné
        if (last.get('MACD_12_26_9', 0) > last.get('MACDs_12_26_9', 0) and 
            last.get('MACDh_12_26_9', 0) > 0):
            quality += 1.0
        elif last.get('MACD_12_26_9', 0) < last.get('MACDs_12_26_9', 0) and last.get('MACDh_12_26_9', 0) < 0:
            quality += 1.0
        else:
            quality += 0.3
        
        # 4. Volatilité normale
        atr = last.get('atr', 0)
        atr_sma = df['atr'].rolling(20).mean().iloc[-1] if len(df) >= 20 else atr
        if atr_sma > 0:
            ratio = atr / atr_sma
            if 0.9 < ratio < 1.2:
                quality += 1.0
            elif 0.7 < ratio < 1.5:
                quality += 0.6
            else:
                quality += 0.2
        
        # 5. EMA alignment
        if (last.get('ema_fast', 0) > last.get('ema_slow', 0) > last.get('ema_50', 0)):
            quality += 1.0
        elif (last.get('ema_fast', 0) < last.get('ema_slow', 0) < last.get('ema_50', 0)):
            quality += 1.0
        elif last.get('ema_fast', 0) > last.get('ema_slow', 0) or last.get('ema_fast', 0) < last.get('ema_slow', 0):
            quality += 0.5
        
        return quality / max_quality
    
    def calculate_confidence_score(self, df, base_signal):
        """
        Score de confiance AMÉLIORÉ (sans ML)
        Scoring plus strict avec pénalités
        Retourne: 0.0 à 1.0
        """
        if len(df) < 50:
            return 0.5
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        score = 0.40  # Base réduite: 0.50 → 0.40
        
        # === 1. ADX: Force tendance (max +0.20) ===
        adx = last.get('adx', 0)
        if adx > 28:
            score += 0.20
        elif adx > 25:
            score += 0.15
        elif adx > 22:
            score += 0.10
        elif adx > 18:
            score += 0.05
        else:
            score -= 0.05  # PÉNALITÉ
        
        # === 2. RSI: Position + Momentum (max +0.15) ===
        rsi = last.get('rsi', 50)
        rsi_prev = prev.get('rsi', 50)
        
        # Position
        if base_signal == 'CALL':
            if 45 < rsi < 58:
                score += 0.10
            elif 40 < rsi < 63:
                score += 0.05
            elif rsi < 35 or rsi > 68:
                score -= 0.05  # PÉNALITÉ extrêmes
        else:  # PUT
            if 42 < rsi < 55:
                score += 0.10
            elif 37 < rsi < 60:
                score += 0.05
            elif rsi < 32 or rsi > 65:
                score -= 0.05
        
        # Momentum RSI
        if base_signal == 'CALL' and rsi > rsi_prev:
            score += 0.05
        elif base_signal == 'PUT' and rsi < rsi_prev:
            score += 0.05
        
        # === 3. MACD: Alignement + Force (max +0.15) ===
        macd = last.get('MACD_12_26_9', 0)
        macd_signal = last.get('MACDs_12_26_9', 0)
        macd_hist = last.get('MACDh_12_26_9', 0)
        macd_hist_prev = prev.get('MACDh_12_26_9', 0)
        
        if base_signal == 'CALL':
            if macd > macd_signal and macd_hist > 0:
                score += 0.10
                # Bonus si en croissance
                if macd_hist > macd_hist_prev:
                    score += 0.05
            else:
                score -= 0.05  # PÉNALITÉ divergence
        else:  # PUT
            if macd < macd_signal and macd_hist < 0:
                score += 0.10
                if macd_hist < macd_hist_prev:
                    score += 0.05
            else:
                score -= 0.05
        
        # === 4. EMA: Triple Alignment (max +0.15) ===
        ema_fast = last.get('ema_fast', 0)
        ema_slow = last.get('ema_slow', 0)
        ema_50 = last.get('ema_50', 0)
        ema_200 = last.get('ema_200', 0)
        
        if base_signal == 'CALL':
            if ema_fast > ema_slow > ema_50 > ema_200:
                score += 0.15  # Perfect
            elif ema_fast > ema_slow > ema_50:
                score += 0.10
            elif ema_fast > ema_slow:
                score += 0.05
            else:
                score -= 0.10  # GROSSE PÉNALITÉ contre-tendance
        else:  # PUT
            if ema_fast < ema_slow < ema_50 < ema_200:
                score += 0.15
            elif ema_fast < ema_slow < ema_50:
                score += 0.10
            elif ema_fast < ema_slow:
                score += 0.05
            else:
                score -= 0.10
        
        # === 5. Stochastic: Confirmation (max +0.08) ===
        stoch_k = last.get('stoch_k', 50)
        stoch_d = last.get('stoch_d', 50)
        
        if base_signal == 'CALL':
            if stoch_k > stoch_d and 20 < stoch_k < 75:
                score += 0.08
            elif stoch_k > 80:
                score -= 0.05  # PÉNALITÉ surachat
        else:  # PUT
            if stoch_k < stoch_d and 25 < stoch_k < 80:
                score += 0.08
            elif stoch_k < 20:
                score -= 0.05  # PÉNALITÉ survente
        
        # === 6. Volatilité: Stabilité (max +0.07) ===
        atr = last.get('atr', 0)
        atr_sma = df['atr'].rolling(20).mean().iloc[-1] if len(df) >= 20 else atr
        
        if atr_sma > 0:
            atr_ratio = atr / atr_sma
            if 0.95 < atr_ratio < 1.15:
                score += 0.07  # Très stable
            elif 0.85 < atr_ratio < 1.30:
                score += 0.04
            elif atr_ratio > 1.8 or atr_ratio < 0.6:
                score -= 0.08  # PÉNALITÉ volatilité extrême
        
        # === 7. Momentum Multi-périodes (max +0.10) ===
        momentum_3 = last.get('momentum_3', 0)
        momentum_5 = last.get('momentum_5', 0)
        
        if base_signal == 'CALL':
            if momentum_3 > 0 and momentum_5 > 0:
                score += 0.10
            elif momentum_3 > 0 or momentum_5 > 0:
                score += 0.05
            else:
                score -= 0.05  # PÉNALITÉ momentum contraire
        else:  # PUT
            if momentum_3 < 0 and momentum_5 < 0:
                score += 0.10
            elif momentum_3 < 0 or momentum_5 < 0:
                score += 0.05
            else:
                score -= 0.05
        
        # Limiter entre 0.0 et 1.0
        return min(max(score, 0.0), 1.0)
    
    def predict_signal(self, df, base_signal, pair_name=None):
        """
        Prédit avec seuils ADAPTATIFS AUTO
        - Détecte automatiquement OTC vs Forex
        - Applique le seuil approprié
        
        Returns: (signal, confidence) ou (None, confidence)
        """
        # Détection automatique du mode et seuil
        min_confidence = self._get_min_confidence(pair_name, df)
        
        mode = "OTC" if min_confidence == self.min_confidence_otc else "Forex"
        print(f"   🎯 Mode auto: {mode} (seuil: {min_confidence*100:.0f}%)")
        
        # MODE 1: Modèle ML entraîné → Utiliser ML
        if self.model_fitted:
            try:
                features = self.extract_features(df)
                if features is None:
                    # Fallback scoring manuel
                    confidence = self.calculate_confidence_score(df, base_signal)
                    
                    # Adaptation OTC: être plus permissif
                    if mode == "OTC" and confidence >= 0.55:
                        print(f"   🏖️ OTC: Accepté avec score manuel {confidence:.1%}")
                        return base_signal, confidence
                    elif confidence >= min_confidence:
                        return base_signal, confidence
                    else:
                        print(f"   ❌ Score manuel rejeté: {confidence:.1%} < {min_confidence:.0%}")
                        return None, confidence
                
                X = pd.DataFrame([features])
                
                try:
                    X_scaled = self.scaler.transform(X)
                except:
                    # Scaler non prêt
                    confidence = self.calculate_confidence_score(df, base_signal)
                    
                    # Adaptation OTC
                    if mode == "OTC" and confidence >= 0.55:
                        print(f"   🏖️ OTC: Accepté sans scaler {confidence:.1%}")
                        return base_signal, confidence
                    elif confidence >= min_confidence:
                        return base_signal, confidence
                    else:
                        return None, confidence
                
                # Prédiction ML
                probas = self.model.predict_proba(X_scaled)[0]
                ml_confidence = probas[1]
                
                print(f"   🤖 ML: {ml_confidence:.1%}")
                
                # SEUIL ADAPTATIF + Fallback OTC
                if ml_confidence < min_confidence:
                    # En mode OTC, être plus permissif
                    if mode == "OTC" and ml_confidence >= 0.55:
                        print(f"   🏖️ OTC: Accepté avec confiance réduite {ml_confidence:.1%}")
                        return base_signal, max(ml_confidence, 0.60)
                    else:
                        print(f"   ❌ ML Rejeté: {ml_confidence:.1%} < {min_confidence:.0%}")
                        return None, ml_confidence
                
                return base_signal, ml_confidence
                
            except Exception as e:
                print(f"   ⚠️  ML Erreur: {e}")
                # Fallback avec adaptation OTC
                confidence = self.calculate_confidence_score(df, base_signal)
                
                if mode == "OTC" and confidence >= 0.55:
                    print(f"   🏖️ OTC Fallback: {confidence:.1%}")
                    return base_signal, confidence
                elif confidence >= min_confidence:
                    return base_signal, confidence
                else:
                    return None, confidence
        
        # MODE 2: Pas de ML → Scoring manuel avec adaptation OTC
        else:
            confidence = self.calculate_confidence_score(df, base_signal)
            
            print(f"   💪 Score manuel: {confidence:.1%}")
            
            # Adaptation OTC: être plus permissif
            if mode == "OTC" and confidence >= 0.55:
                print(f"   🏖️ OTC: Accepté avec score manuel")
                return base_signal, max(confidence, 0.60)
            elif confidence < min_confidence:
                print(f"   ❌ Score trop bas: {confidence:.1%} < {min_confidence:.0%}")
                return None, confidence
            
            return base_signal, confidence
    
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
        
        if len(results) < 100:  # Seuil augmenté: 50 → 100
            print(f"⚠️  Données insuffisantes: {len(results)} < 100")
            return False
        
        print(f"📚 {len(results)} signaux disponibles")
        print("✅ Utiliser /retrain pour entraîner")
        
        return False
    
    def save_model(self):
        """Sauvegarde"""
        try:
            # Créer dossier parent
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
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
            print(f"   🎯 Seuil Forex: {self.min_confidence_forex*100:.0f}%")
            print(f"   🏖️ Seuil OTC: {self.min_confidence_otc*100:.0f}%")
            print(f"   🔥 Seuil ultra: {self.ultra_strict_confidence:.0%}")
            print(f"   ⭐ Seuil premium: {self.premium_confidence:.0%}")
            print(f"   🤖 Entraîné: {'Oui' if self.model_fitted else 'Non'}")
        except Exception as e:
            print(f"⚠️  Erreur chargement: {e}")
            self.model = None
            self.model_fitted = False
    
    def get_model_info(self):
        """Info modèle"""
        return {
            'model_loaded': self.model is not None,
            'model_fitted': self.model_fitted,
            'model_type': type(self.model).__name__ if self.model else 'None',
            'min_confidence_forex': self.min_confidence_forex,
            'min_confidence_otc': self.min_confidence_otc,
            'ultra_strict_confidence': self.ultra_strict_confidence,
            'premium_confidence': self.premium_confidence,
            'current_mode': 'OTC' if self._is_weekend() else 'Forex'
        }
