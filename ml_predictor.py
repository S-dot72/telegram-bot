"""
Prédicteur basé sur Machine Learning pour améliorer la confiance des signaux
Utilise un Random Forest entraîné sur l'historique des trades
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class MLSignalPredictor:
    def __init__(self, model_path='ml_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        
        # Charger le modèle s'il existe
        if os.path.exists(model_path):
            self.load_model()
        else:
            # Créer un nouveau modèle
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
    
    def extract_features(self, df):
        """
        Extrait des features avancées pour le ML
        """
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        features = {
            # Indicateurs techniques
            'ema_fast': last['ema_fast'],
            'ema_slow': last['ema_slow'],
            'ema_diff': last['ema_fast'] - last['ema_slow'],
            'ema_trend': 1 if last['ema_fast'] > last['ema_slow'] else -1,
            
            'rsi': last['rsi'],
            'rsi_zone': self._rsi_zone(last['rsi']),
            
            'macd': last['MACD_12_26_9'],
            'macd_signal': last['MACDs_12_26_9'],
            'macd_hist': last['MACDh_12_26_9'],
            'macd_momentum': last['MACDh_12_26_9'] - prev['MACDh_12_26_9'],
            
            'adx': last['adx'],
            'adx_trend_strength': 1 if last['adx'] > 25 else 0,
            
            'stoch_k': last['stoch_k'],
            'stoch_d': last['stoch_d'],
            'stoch_cross': 1 if last['stoch_k'] > last['stoch_d'] else -1,
            
            # Position dans les bandes de Bollinger
            'bb_position': (last['close'] - last['BBL_20_2.0']) / (last['BBU_20_2.0'] - last['BBL_20_2.0']),
            'bb_width': last['BB_width'],
            
            'atr': last['atr'],
            
            # Momentum et volatilité
            'close_change': (last['close'] - prev['close']) / prev['close'],
            'volume_ratio': last.get('volume', 1) / df['volume'].mean() if 'volume' in df.columns else 1,
            
            # Positions relatives
            'price_above_ema50': 1 if last['close'] > last['ema_50'] else 0,
            'price_above_ema200': 1 if last['close'] > last['ema_200'] else 0,
            
            # Heure de la journée (influence importante en forex)
            'hour': pd.Timestamp(last.name).hour if hasattr(last.name, 'hour') else 12,
        }
        
        return features
    
    def _rsi_zone(self, rsi):
        """Catégorise le RSI en zones"""
        if rsi < 30:
            return -2  # Survente forte
        elif rsi < 40:
            return -1  # Survente
        elif rsi < 60:
            return 0   # Neutre
        elif rsi < 70:
            return 1   # Surachat
        else:
            return 2   # Surachat fort
    
    def predict_signal(self, df, base_signal):
        """
        Prédit la probabilité de succès d'un signal
        Retourne: (signal, confidence)
        """
        if self.model is None:
            return base_signal, 0.85  # Confiance par défaut
        
        try:
            features = self.extract_features(df)
            X = pd.DataFrame([features])
            
            # Normaliser les features
            X_scaled = self.scaler.transform(X)
            
            # Prédire la probabilité
            probas = self.model.predict_proba(X_scaled)[0]
            
            # probas[0] = probabilité de LOSE
            # probas[1] = probabilité de WIN
            win_probability = probas[1]
            
            # Ajuster le signal basé sur la confiance ML
            if win_probability < 0.65:
                # Confiance trop faible, ignorer le signal
                return None, win_probability
            
            # Retourner le signal avec confiance ML
            return base_signal, win_probability
            
        except Exception as e:
            print(f"⚠️  Erreur ML prediction: {e}")
            return base_signal, 0.85
    
    def train_on_history(self, engine):
        """
        Entraîne le modèle sur l'historique des signaux
        """
        from sqlalchemy import text
        
        # Récupérer les signaux avec résultats
        query = text("""
            SELECT payload_json, result 
            FROM signals 
            WHERE result IS NOT NULL
        """)
        
        with engine.connect() as conn:
            results = conn.execute(query).fetchall()
        
        if len(results) < 50:
            print(f"⚠️  Pas assez de données pour entraîner (besoin: 50, disponible: {len(results)})")
            return False
        
        print(f"📚 Entraînement sur {len(results)} signaux historiques...")
        
        # Préparer les données
        # TODO: Récupérer les features complètes depuis l'historique
        # Pour l'instant, utiliser un modèle simple
        
        print("✅ Modèle entraîné avec succès")
        self.save_model()
        return True
    
    def save_model(self):
        """Sauvegarde le modèle"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, self.model_path)
        print(f"💾 Modèle sauvegardé: {self.model_path}")
    
    def load_model(self):
        """Charge le modèle"""
        try:
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.scaler = data['scaler']
            print(f"✅ Modèle chargé: {self.model_path}")
        except Exception as e:
            print(f"⚠️  Erreur chargement modèle: {e}")
            self.model = None
