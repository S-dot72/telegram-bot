"""
Système de vérification automatique des résultats
Vérifie si les signaux ont gagné ou perdu en analysant les prix après l'entrée
"""

import asyncio
from datetime import datetime, timedelta, timezone
from sqlalchemy import text
import requests

class AutoResultVerifier:
    def __init__(self, engine, twelvedata_api_key, bot=None):
        self.engine = engine
        self.api_key = twelvedata_api_key
        self.base_url = 'https://api.twelvedata.com/time_series'
        self.bot = bot  # Pour envoyer des notifications
        self.admin_chat_ids = []  # Liste des admins à notifier
    
    def set_bot(self, bot):
        """Configure le bot pour les notifications"""
        self.bot = bot
    
    def add_admin(self, chat_id):
        """Ajoute un admin pour recevoir les rapports"""
        if chat_id not in self.admin_chat_ids:
            self.admin_chat_ids.append(chat_id)
    
    async def verify_pending_signals(self):
        """
        Vérifie tous les signaux qui n'ont pas encore de résultat
        et dont l'heure d'entrée + 10 minutes est passée
        """
        print("\n" + "="*60)
        print(f"🔍 VÉRIFICATION AUTOMATIQUE - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print("="*60)
        
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
            print("="*60 + "\n")
            return
        
        print(f"📊 {len(pending)} signaux à vérifier")
        print("-"*60)
        
        results = []
        verified_count = 0
        
        for signal in pending:
            try:
                print(f"\n🔎 Signal #{signal.id} - {signal.pair} {signal.direction}")
                result, details = await self._verify_single_signal(signal)
                
                if result:
                    self._update_signal_result(signal.id, result)
                    verified_count += 1
                    results.append({
                        'signal': signal,
                        'result': result,
                        'details': details
                    })
                    
                    # Log détaillé
                    emoji = "✅" if result == 'WIN' else "❌"
                    print(f"{emoji} Résultat: {result}")
                    print(f"   Entrée: {details['entry_price']:.5f}")
                    print(f"   Sortie: {details['exit_price']:.5f}")
                    print(f"   Diff: {details['pips']:.1f} pips")
                
                await asyncio.sleep(2)  # Respecter limite API
                
            except Exception as e:
                print(f"❌ Erreur vérification signal {signal.id}: {e}")
        
        print("\n" + "-"*60)
        print(f"📈 RÉSUMÉ: {verified_count}/{len(pending)} signaux vérifiés")
        print("="*60 + "\n")
        
        # Envoyer rapport aux admins
        if verified_count > 0 and self.bot and self.admin_chat_ids:
            await self._send_verification_report(results)
        
        # Vérifier si réentraînement nécessaire
        self._check_ml_retraining()
    
    async def _verify_single_signal(self, signal):
        """
        Vérifie un signal individuel
        Retourne: (result, details)
        """
        entry_time = datetime.fromisoformat(signal.ts_enter.replace('Z', '+00:00'))
        check_time = entry_time + timedelta(minutes=5)
        
        # Récupérer les prix
        entry_price = await self._get_price_at_time(signal.pair, entry_time)
        await asyncio.sleep(1)  # Petite pause entre les appels API
        exit_price = await self._get_price_at_time(signal.pair, check_time)
        
        if entry_price is None or exit_price is None:
            print(f"⚠️  Prix non disponibles")
            return None, None
        
        # Déterminer WIN ou LOSE
        if signal.direction == 'CALL':
            result = 'WIN' if exit_price > entry_price else 'LOSE'
        else:  # PUT
            result = 'WIN' if exit_price < entry_price else 'LOSE'
        
        pips_diff = abs(exit_price - entry_price) * 10000
        
        details = {
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pips': pips_diff,
            'entry_time': entry_time,
            'check_time': check_time
        }
        
        return result, details
    
    async def _get_price_at_time(self, pair, timestamp):
        """Récupère le prix d'une paire à un moment donné"""
        try:
            # Utiliser end_date pour obtenir les données jusqu'à ce moment
            end_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            start_str = (timestamp - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')
            
            params = {
                'symbol': pair,
                'interval': '1min',
                'outputsize': 5,
                'apikey': self.api_key,
                'format': 'JSON',
                'start_date': start_str,
                'end_date': end_str
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'values' in data and len(data['values']) > 0:
                # Trouver la bougie la plus proche du timestamp
                for candle in data['values']:
                    return float(candle['close'])
            
            return None
            
        except Exception as e:
            print(f"⚠️  Erreur API: {e}")
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
        
        print(f"💾 Résultat sauvegardé: Signal #{signal_id} = {result}")
    
    async def _send_verification_report(self, results):
        """Envoie un rapport de vérification aux admins"""
        wins = sum(1 for r in results if r['result'] == 'WIN')
        losses = len(results) - wins
        winrate = (wins / len(results)) * 100
        
        report = f"📊 **Rapport de Vérification**\n\n"
        report += f"Signaux vérifiés: {len(results)}\n"
        report += f"✅ Gains: {wins}\n"
        report += f"❌ Pertes: {losses}\n"
        report += f"📈 Win rate: {winrate:.1f}%\n\n"
        report += f"Détails:\n"
        
        for r in results[:5]:  # Montrer max 5 derniers
            emoji = "✅" if r['result'] == 'WIN' else "❌"
            sig = r['signal']
            det = r['details']
            report += f"{emoji} {sig.pair} {sig.direction}: {det['pips']:.1f} pips\n"
        
        # Envoyer à tous les admins
        for chat_id in self.admin_chat_ids:
            try:
                await self.bot.send_message(chat_id=chat_id, text=report)
            except Exception as e:
                print(f"⚠️  Erreur envoi rapport à {chat_id}: {e}")
    
    def _check_ml_retraining(self):
        """Vérifie si réentraînement ML nécessaire"""
        query = text("""
            SELECT COUNT(*) as count 
            FROM signals 
            WHERE result IS NOT NULL
        """)
        
        with self.engine.connect() as conn:
            count = conn.execute(query).scalar()
        
        if count >= 100 and count % 50 == 0:
            print(f"\n🎓 {count} résultats disponibles")
            print(f"💡 Réentraînement du modèle ML recommandé")
            print(f"   Utilisez /train pour améliorer la précision\n")
    
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
    
    def get_recent_results(self, limit=10):
        """Récupère les derniers résultats vérifiés"""
        query = text("""
            SELECT pair, direction, result, confidence, ts_enter, ts_result
            FROM signals 
            WHERE result IS NOT NULL
            ORDER BY ts_result DESC
            LIMIT :limit
        """)
        
        with self.engine.connect() as conn:
            results = conn.execute(query, {'limit': limit}).fetchall()
        
        return results
