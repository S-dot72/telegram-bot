"""
Système d'apprentissage continu ultra-optimisé pour 90% Win Rate
Réentraîne automatiquement le modèle avec les nouveaux résultats
Supporte Gradient Boosting et analyse avancée des performances
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from sqlalchemy import text
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

class ContinuousLearning:
    def __init__(self, engine, model_dir='models'):
        self.engine = engine
        self.model_dir = model_dir
        self.HAITI_TZ = ZoneInfo("America/Port-au-Prince")
        
        # Créer le dossier models s'il n'existe pas
        os.makedirs(model_dir, exist_ok=True)
        
        # Chemins des fichiers
        self.model_path = os.path.join(model_dir, 'ml_model.pkl')
        self.scaler_path = os.path.join(model_dir, 'scaler.pkl')
        self.history_path = os.path.join(model_dir, 'training_history.json')
        self.backup_dir = os.path.join(model_dir, 'backups')
        
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Historique d'entraînement
        self.training_history = self.load_training_history()
        
        # Seuils pour accepter un nouveau modèle
        self.min_accuracy_target = 0.85  # Minimum 85% pour être accepté
        self.optimal_accuracy_target = 0.90  # Objectif 90%
    
    def load_training_history(self):
        """Charge l'historique des entraînements"""
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'trainings': [],
            'best_accuracy': 0.0,
            'best_precision_win': 0.0,
            'total_signals_trained': 0,
            'model_version': 1
        }
    
    def save_training_history(self):
        """Sauvegarde l'historique"""
        with open(self.history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def get_verified_signals(self, min_signals=50, days_back=None):
        """
        Récupère tous les signaux vérifiés (WIN/LOSE) de la base
        
        Args:
            min_signals: Nombre minimum de signaux requis
            days_back: Limiter aux N derniers jours (None = tous)
        
        Returns:
            DataFrame avec les features et résultats, ou None si insuffisant
        """
        try:
            # Construire la requête
            where_clause = "WHERE result IN ('WIN', 'LOSE')"
            
            if days_back:
                cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
                where_clause += f" AND ts_enter >= '{cutoff_date}'"
            
            query = text(f"""
                SELECT 
                    pair,
                    direction,
                    confidence,
                    result,
                    gale_level,
                    ts_enter,
                    ts_send,
                    payload_json,
                    reason
                FROM signals
                {where_clause}
                ORDER BY ts_enter ASC
            """)
            
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)
            
            print(f"📊 Signaux vérifiés trouvés: {len(df)}")
            
            if len(df) < min_signals:
                print(f"⚠️ Pas assez de signaux ({len(df)} < {min_signals})")
                return None
            
            # Statistiques rapides
            win_rate = (df['result'] == 'WIN').mean()
            print(f"📈 Win rate actuel: {win_rate*100:.1f}%")
            print(f"✅ Wins: {(df['result'] == 'WIN').sum()}")
            print(f"❌ Loses: {(df['result'] == 'LOSE').sum()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Erreur get_verified_signals: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def extract_advanced_features(self, df):
        """
        Extrait des features avancées depuis les signaux
        Compatible avec le nouveau ml_predictor.py
        
        Returns:
            DataFrame avec features enrichies
        """
        try:
            print("\n🔍 Extraction des features avancées...")
            
            # Features de base
            df['direction_encoded'] = df['direction'].map({'CALL': 1, 'PUT': 0})
            df['pair_encoded'] = df['pair'].astype('category').cat.codes
            df['result_binary'] = df['result'].map({'WIN': 1, 'LOSE': 0})
            
            # Features temporelles
            df['ts_enter_dt'] = pd.to_datetime(df['ts_enter'])
            df['hour'] = df['ts_enter_dt'].dt.hour
            df['day_of_week'] = df['ts_enter_dt'].dt.dayofweek
            df['is_london_session'] = ((df['hour'] >= 7) & (df['hour'] <= 16)).astype(int)
            df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] <= 21)).astype(int)
            
            # Feature de confiance
            df['confidence_squared'] = df['confidence'] ** 2
            df['confidence_high'] = (df['confidence'] >= 0.85).astype(int)
            df['confidence_ultra'] = (df['confidence'] >= 0.90).astype(int)
            
            # Statistiques par paire
            pair_stats = df.groupby('pair')['result_binary'].agg(['mean', 'count']).reset_index()
            pair_stats.columns = ['pair', 'pair_win_rate', 'pair_count']
            df = df.merge(pair_stats, on='pair', how='left')
            
            # Statistiques par direction
            direction_stats = df.groupby('direction')['result_binary'].agg(['mean', 'count']).reset_index()
            direction_stats.columns = ['direction', 'direction_win_rate', 'direction_count']
            df = df.merge(direction_stats, on='direction', how='left')
            
            # Statistiques par heure
            hour_stats = df.groupby('hour')['result_binary'].agg(['mean', 'count']).reset_index()
            hour_stats.columns = ['hour', 'hour_win_rate', 'hour_count']
            df = df.merge(hour_stats, on='hour', how='left')
            
            # Features d'interaction
            df['confidence_x_pair_wr'] = df['confidence'] * df['pair_win_rate']
            df['confidence_x_hour_wr'] = df['confidence'] * df['hour_win_rate']
            
            # Features sans gale (pour nouveau système)
            df['no_gale'] = (df['gale_level'] == 0).astype(int)
            
            print(f"✅ Features extraites: {len(df.columns)} colonnes")
            
            return df
            
        except Exception as e:
            print(f"❌ Erreur extract_advanced_features: {e}")
            import traceback
            traceback.print_exc()
            return df
    
    def prepare_training_data(self, df):
        """
        Prépare les données pour l'entraînement avec features avancées
        
        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        try:
            # Enrichir avec features avancées
            df = self.extract_advanced_features(df)
            
            # Sélectionner les features pour l'entraînement
            feature_cols = [
                'direction_encoded',
                'pair_encoded',
                'confidence',
                'confidence_squared',
                'confidence_high',
                'confidence_ultra',
                'hour',
                'day_of_week',
                'is_london_session',
                'is_ny_session',
                'pair_win_rate',
                'direction_win_rate',
                'hour_win_rate',
                'confidence_x_pair_wr',
                'confidence_x_hour_wr',
                'no_gale'
            ]
            
            # Vérifier que toutes les colonnes existent
            available_features = [col for col in feature_cols if col in df.columns]
            missing_features = [col for col in feature_cols if col not in df.columns]
            
            if missing_features:
                print(f"⚠️ Features manquantes (ignorées): {missing_features}")
            
            print(f"📊 Features utilisées: {len(available_features)}")
            
            X = df[available_features].values
            y = df['result_binary'].values
            
            # Split stratifié 80/20
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"\n📊 Dataset:")
            print(f"   Train: {len(X_train)} signaux ({y_train.mean()*100:.1f}% WIN)")
            print(f"   Test: {len(X_test)} signaux ({y_test.mean()*100:.1f}% WIN)")
            
            return X_train, X_test, y_train, y_test, available_features
            
        except Exception as e:
            print(f"❌ Erreur prepare_training_data: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None, []
    
    def create_model(self, model_type='gradient_boosting'):
        """
        Crée un modèle optimisé pour 90% Win Rate
        
        Args:
            model_type: 'gradient_boosting' ou 'random_forest'
        """
        if model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42
            )
        else:
            return RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=8,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1
            )
    
    def train_new_model(self, X_train, y_train, X_test, y_test, model_type='gradient_boosting'):
        """
        Entraîne un nouveau modèle avec validation croisée
        
        Returns:
            model, scaler, accuracy, detailed_metrics
        """
        try:
            print(f"\n🤖 Entraînement du modèle ({model_type})...")
            
            # Normaliser les features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Créer le modèle
            model = self.create_model(model_type)
            
            # Validation croisée (5-fold)
            print("📊 Validation croisée...")
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            print(f"   CV Accuracy: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")
            
            # Entraîner sur tout le train set
            model.fit(X_train_scaled, y_train)
            
            # Prédictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Probabilités
            y_proba_test = model.predict_proba(X_test_scaled)[:, 1]
            
            # Métriques
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            
            # Classification report
            report = classification_report(y_test, y_pred_test, output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred_test)
            
            # Métriques détaillées
            detailed_metrics = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'precision_win': report['1']['precision'],
                'recall_win': report['1']['recall'],
                'f1_win': report['1']['f1-score'],
                'precision_lose': report['0']['precision'],
                'recall_lose': report['0']['recall'],
                'true_positives': int(cm[1][1]),
                'false_positives': int(cm[0][1]),
                'true_negatives': int(cm[0][0]),
                'false_negatives': int(cm[1][0]),
                'overfitting_gap': train_accuracy - test_accuracy
            }
            
            # Affichage
            print(f"\n📊 RÉSULTATS:")
            print(f"   {'='*50}")
            print(f"   Train Accuracy: {train_accuracy*100:.2f}%")
            print(f"   Test Accuracy:  {test_accuracy*100:.2f}%")
            print(f"   Overfitting:    {detailed_metrics['overfitting_gap']*100:+.2f}%")
            print(f"   {'='*50}")
            print(f"   Precision WIN:  {detailed_metrics['precision_win']*100:.2f}%")
            print(f"   Recall WIN:     {detailed_metrics['recall_win']*100:.2f}%")
            print(f"   F1-Score WIN:   {detailed_metrics['f1_win']*100:.2f}%")
            print(f"   {'='*50}")
            print(f"   Matrice de confusion:")
            print(f"     TN={cm[0][0]}  FP={cm[0][1]}")
            print(f"     FN={cm[1][0]}  TP={cm[1][1]}")
            print(f"   {'='*50}")
            
            return model, scaler, test_accuracy, detailed_metrics
            
        except Exception as e:
            print(f"❌ Erreur train_new_model: {e}")
            import traceback
            traceback.print_exc()
            return None, None, 0.0, {}
    
    def save_model(self, model, scaler, accuracy, metrics, backup=True):
        """
        Sauvegarde le modèle et le scaler
        """
        try:
            # Sauvegarder le modèle
            with open(self.model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Sauvegarder le scaler
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            print(f"💾 Modèle sauvegardé: {self.model_path}")
            print(f"💾 Scaler sauvegardé: {self.scaler_path}")
            
            # Créer un backup avec timestamp
            if backup:
                now = datetime.now(self.HAITI_TZ)
                backup_name = f"model_v{self.training_history['model_version']}_{now.strftime('%Y%m%d_%H%M%S')}_acc{accuracy:.3f}.pkl"
                backup_path = os.path.join(self.backup_dir, backup_name)
                
                # Sauvegarder modèle + scaler + metrics
                backup_data = {
                    'model': model,
                    'scaler': scaler,
                    'accuracy': accuracy,
                    'metrics': metrics,
                    'timestamp': now.isoformat()
                }
                
                with open(backup_path, 'wb') as f:
                    pickle.dump(backup_data, f)
                
                print(f"💾 Backup créé: {backup_name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur save_model: {e}")
            return False
    
    def retrain_model(self, min_signals=50, min_accuracy_improvement=0.01, days_back=None):
        """
        Réentraîne le modèle avec les nouveaux signaux
        
        Args:
            min_signals: Nombre minimum de signaux
            min_accuracy_improvement: Amélioration minimale requise
            days_back: Limiter aux N derniers jours
        
        Returns:
            dict avec les résultats
        """
        try:
            print("\n" + "="*70)
            print("🔄 RÉENTRAÎNEMENT ML - MODE ULTRA STRICT (90% WR)")
            print("="*70)
            
            now_haiti = datetime.now(self.HAITI_TZ)
            
            # 1. Récupérer les signaux
            df = self.get_verified_signals(min_signals, days_back)
            
            if df is None:
                return {
                    'success': False,
                    'reason': f'Pas assez de signaux (min: {min_signals})',
                    'signals_count': 0
                }
            
            # 2. Préparer les données
            X_train, X_test, y_train, y_test, features = self.prepare_training_data(df)
            
            if X_train is None:
                return {
                    'success': False,
                    'reason': 'Erreur préparation données',
                    'signals_count': len(df)
                }
            
            # 3. Entraîner le nouveau modèle
            new_model, scaler, new_accuracy, metrics = self.train_new_model(
                X_train, y_train, X_test, y_test, model_type='gradient_boosting'
            )
            
            if new_model is None:
                return {
                    'success': False,
                    'reason': 'Erreur entraînement modèle',
                    'signals_count': len(df)
                }
            
            # 4. Évaluation stricte
            best_accuracy = self.training_history.get('best_accuracy', 0.0)
            improvement = new_accuracy - best_accuracy
            
            print(f"\n{'='*70}")
            print(f"📊 ÉVALUATION:")
            print(f"{'='*70}")
            print(f"   Meilleur précédent:  {best_accuracy*100:.2f}%")
            print(f"   Nouveau modèle:      {new_accuracy*100:.2f}%")
            print(f"   Amélioration:        {improvement*100:+.2f}%")
            print(f"   Objectif minimum:    {self.min_accuracy_target*100:.0f}%")
            print(f"   Objectif optimal:    {self.optimal_accuracy_target*100:.0f}%")
            print(f"{'='*70}")
            
            # 5. Décision d'acceptation
            accept_model = False
            accept_reason = ""
            
            if new_accuracy >= self.optimal_accuracy_target:
                accept_model = True
                accept_reason = f"✨ Objectif optimal atteint ({new_accuracy*100:.1f}% ≥ 90%)"
            elif new_accuracy >= self.min_accuracy_target and improvement >= min_accuracy_improvement:
                accept_model = True
                accept_reason = f"✅ Amélioration significative ({improvement*100:+.1f}%)"
            elif best_accuracy == 0.0 and new_accuracy >= self.min_accuracy_target:
                accept_model = True
                accept_reason = "🎯 Premier modèle valide"
            else:
                accept_reason = f"⚠️ Critères non atteints (accuracy: {new_accuracy*100:.1f}%, amélioration: {improvement*100:+.1f}%)"
            
            print(f"\n{'='*70}")
            print(f"{'✅ MODÈLE ACCEPTÉ' if accept_model else '❌ MODÈLE REJETÉ'}")
            print(f"{accept_reason}")
            print(f"{'='*70}\n")
            
            # 6. Sauvegarder si accepté
            if accept_model:
                self.save_model(new_model, scaler, new_accuracy, metrics, backup=True)
                
                # Incrémenter version
                self.training_history['model_version'] += 1
                self.training_history['best_accuracy'] = new_accuracy
                self.training_history['best_precision_win'] = metrics['precision_win']
                self.training_history['total_signals_trained'] = len(df)
                self.training_history['last_training'] = now_haiti.isoformat()
            
            # 7. Enregistrer dans l'historique
            training_entry = {
                'timestamp': now_haiti.isoformat(),
                'model_version': self.training_history['model_version'],
                'signals_count': len(df),
                'accuracy': new_accuracy,
                'improvement': improvement,
                'metrics': metrics,
                'accepted': accept_model,
                'reason': accept_reason
            }
            
            self.training_history['trainings'].append(training_entry)
            self.save_training_history()
            
            return {
                'success': True,
                'accepted': accept_model,
                'signals_count': len(df),
                'accuracy': new_accuracy,
                'improvement': improvement,
                'metrics': metrics,
                'reason': accept_reason
            }
        
        except Exception as e:
            print(f"❌ Erreur retrain_model: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'reason': f'Erreur: {str(e)}',
                'signals_count': 0
            }
    
    def get_training_stats(self):
        """Retourne les statistiques d'entraînement"""
        return {
            'total_trainings': len(self.training_history.get('trainings', [])),
            'best_accuracy': self.training_history.get('best_accuracy', 0.0),
            'best_precision_win': self.training_history.get('best_precision_win', 0.0),
            'total_signals': self.training_history.get('total_signals_trained', 0),
            'last_training': self.training_history.get('last_training', 'Jamais'),
            'model_version': self.training_history.get('model_version', 0),
            'recent_trainings': self.training_history.get('trainings', [])[-5:]
        }


# === Fonction pour intégration dans le bot ===

async def scheduled_retraining(engine, telegram_app=None, admin_chat_ids=None):
    """
    Réentraînement nocturne automatique avec notification
    
    Args:
        engine: SQLAlchemy engine
        telegram_app: Application Telegram (optionnel)
        admin_chat_ids: Liste des IDs admin à notifier
    """
    try:
        print("\n🌙 RÉENTRAÎNEMENT NOCTURNE PROGRAMMÉ")
        print("="*70)
        
        learner = ContinuousLearning(engine)
        
        # Réentraîner avec minimum 50 signaux, amélioration min 1%
        result = learner.retrain_model(
            min_signals=50,
            min_accuracy_improvement=0.01,
            days_back=30  # Limiter aux 30 derniers jours
        )
        
        # Créer le message de notification
        if result['success']:
            if result['accepted']:
                emoji = "✅"
                status = "ACCEPTÉ"
                metrics = result.get('metrics', {})
                msg = (
                    f"{emoji} **Réentraînement ML {status}**\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"📊 Signaux: {result['signals_count']}\n"
                    f"🎯 Accuracy: {result['accuracy']*100:.2f}%\n"
                    f"📈 Amélioration: {result['improvement']*100:+.2f}%\n"
                    f"✨ Precision WIN: {metrics.get('precision_win', 0)*100:.1f}%\n"
                    f"🔄 Recall WIN: {metrics.get('recall_win', 0)*100:.1f}%\n\n"
                    f"✨ {result['reason']}"
                )
            else:
                emoji = "⚠️"
                status = "REJETÉ"
                msg = (
                    f"{emoji} **Réentraînement ML {status}**\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"📊 Signaux: {result['signals_count']}\n"
                    f"🎯 Accuracy: {result['accuracy']*100:.2f}%\n"
                    f"📉 Amélioration: {result['improvement']*100:+.2f}%\n\n"
                    f"ℹ️ {result['reason']}"
                )
        else:
            emoji = "❌"
            msg = (
                f"{emoji} **Réentraînement ML ÉCHOUÉ**\n"
                f"━━━━━━━━━━━━━━━━━━━━\n\n"
                f"❌ {result['reason']}\n"
                f"📊 Signaux disponibles: {result['signals_count']}"
            )
        
        print(f"\n{msg}\n")
        
        # Envoyer notification aux admins
        if telegram_app and admin_chat_ids:
            for admin_id in admin_chat_ids:
                try:
                    await telegram_app.bot.send_message(chat_id=admin_id, text=msg)
                    print(f"📤 Notification envoyée à admin {admin_id}")
                except Exception as e:
                    print(f"❌ Erreur envoi notification à {admin_id}: {e}")
        
        return result
        
    except Exception as e:
        print(f"❌ Erreur scheduled_retraining: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'reason': str(e)}
