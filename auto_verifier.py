"""
Système de vérification automatique des résultats
Vérifie si les signaux ont gagné ou perdu en analysant les prix après l'entrée
"""

import asyncio
from datetime import datetime, timedelta, timezone
from sqlalchemy import text
import requests

class AutoResultVerifier:
    def __init__(self, engine, twelvedata_api_key):
        self.engine = engine
        self.api_key = twelvedata_api_key
        self.base_url = 'https://api.twelvedata.com/time_series'
    
    async def verify_pending_signals(self):
        """
        Vérifie tous les signaux qui n'ont pas encore de résultat
        et dont l'heure d'entrée + 10 minutes est passée
        """
        print("🔍 Vérification automatique des résultats...")
        
        # Récupérer les signaux sans résultat et passés
        query = text("""
            SELECT id, pair, direction, ts_enter, confidence
            FROM signals 
            WHERE result IS NULL 
            AND datetime(ts_enter) < datetime('now', '-10 minutes')
            ORDER BY ts_enter DESC
            LIMIT 20
        """)
        
        with self.engine.connect() as conn:
            pending = conn.execute(query).fetchall()
        
        if not pending:
            print("✅ Aucun signal en attente de vérification")
            return
        
        print(f"📊 {len(pending)} signaux à vérifier")
        
        verified_count = 0
        for signal in pending:
            try:
                result = await self._verify_single_signal(signal)
                if result:
                    self._update_signal_result(signal.id, result)
                    verified_count += 1
                    await asyncio.sleep(2)  # Respecter limite API
            except Exception as e:
                print(f"❌ Erreur vérification signal {signal.id}: {e}")
        
        print(f"✅ {verified_count}/{len(pending)} signaux vérifiés")
        
        # Entraîner le modèle ML si assez de données
        self._trigger_ml_retraining()
    
    async def _verify_single_signal(self, signal):
        """
        Vérifie un signal individuel
        Logique:
        - BUY (CALL): WIN si prix > prix_entrée après 5 min
        - SELL (PUT): WIN si prix < prix_entrée après 5 min
        """
        entry_time = datetime.fromisoformat(signal.ts_enter.replace('Z', '+00:00'))
        check_time = entry_time + timedelta(minutes=5)  # Vérifier 5 min après
        
        print(f"🔎 Vérification {signal.pair} {signal.direction} entrée à {entry_time}")
        
        # Récupérer les prix au moment de l'entrée et 5 min après
        entry_price = self._get_price_at_time(signal.pair, entry_time)
        exit_price = self._get_price_at_time(signal.pair, check_time)
        
        if entry_price is None or exit_price is None:
            print(f"⚠️  Prix non disponibles pour {signal.pair}")
            return None
        
        # Déterminer WIN ou LOSE
        if signal.direction == 'CALL':
            # BUY: on gagne si le prix monte
            result = 'WIN' if exit_price > entry_price else 'LOSE'
        else:  # PUT
            # SELL: on gagne si le prix descend
            result = 'WIN' if exit_price < entry_price else 'LOSE'
        
        pips_diff = abs(exit_price - entry_price) * 10000  # Différence en pips
        
        print(f"{'✅' if result == 'WIN' else '❌'} {signal.pair}: {result} "
              f"(entrée: {entry_price:.5f}, sortie: {exit_price:.5f}, {pips_diff:.1f} pips)")
        
        return result
    
    def _get_price_at_time(self, pair, timestamp):
        """
        Récupère le prix d'une paire à un moment donné
        """
        try:
            # Formater la date pour TwelveData
            date_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            
            params = {
                'symbol': pair,
                'interval': '1min',
                'outputsize': 10,
                'apikey': self.api_key,
                'format': 'JSON',
                'start_date': date_str
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'values' in data and len(data['values']) > 0:
                # Prendre le prix de clôture de la première bougie
                return float(data['values'][0]['close'])
            
            return None
            
        except Exception as e:
            print(f"⚠️  Erreur récupération prix: {e}")
            return None
    
    def _update_signal_result(self, signal_id, result):
        """Met à jour le résultat d'un signal dans la DB"""
        query = text("""
            UPDATE signals 
            SET result = :result, ts_result = :ts_result 
            WHERE id = :id
        """)
        
        with self.engine.begin() as conn:
            conn.execute(query, {
                'result': result,
                'ts_result': datetime.utcnow().isoformat(),
                'id': signal_id
            })
    
    def _trigger_ml_retraining(self):
        """Déclenche le réentraînement du modèle ML si nécessaire"""
        query = text("""
            SELECT COUNT(*) as count 
            FROM signals 
            WHERE result IS NOT NULL
        """)
        
        with self.engine.connect() as conn:
            count = conn.execute(query).scalar()
        
        if count >= 100 and count % 50 == 0:
            print(f"🎓 {count} résultats disponibles, réentraînement du modèle ML recommandé")
            # TODO: Appeler ml_predictor.train_on_history()
    
    def get_performance_stats(self):
        """Calcule les statistiques de performance"""
        query = text("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                AVG(confidence) as avg_confidence
            FROM signals 
            WHERE result IS NOT NULL
        """)
        
        with self.engine.connect() as conn:
            stats = conn.execute(query).fetchone()
        
        if stats.total > 0:
            winrate = (stats.wins / stats.total) * 100
            return {
                'total': stats.total,
                'wins': stats.wins,
                'losses': stats.total - stats.wins,
                'winrate': winrate,
                'avg_confidence': stats.avg_confidence
            }
        
        return None
