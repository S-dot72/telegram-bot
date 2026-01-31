"""
AUTO VERIFIER M1 - VERSION CORRIGÉE
Correction pour trading M1 (1 minute) au lieu de M5
"""

import asyncio
from datetime import datetime, timedelta, timezone
from sqlalchemy import text
import requests

class AutoResultVerifierM1:
    def __init__(self, engine, twelvedata_api_key, bot=None):
        self.engine = engine
        self.api_key = twelvedata_api_key
        self.base_url = 'https://api.twelvedata.com/time_series'
        self.bot = bot
        self.admin_chat_ids = []
        
        self._session = requests.Session()
        
        # Paramètres pour M1
        self.default_timeframe = 1  # ✅ 1 minute
        self.default_max_gales = 0
        
        # RATE LIMITING
        self.api_calls_count = 0
        self.api_calls_reset_time = datetime.now()
        self.MAX_API_CALLS_PER_MINUTE = 6
        
        print("[VERIF-M1] ✅ AutoResultVerifier M1 initialisé")
        print("[VERIF-M1] 🎯 Mode: Trading M1 (1 minute)")

    def set_bot(self, bot):
        """Configure le bot pour les notifications"""
        self.bot = bot
        print("[VERIF-M1] ✅ Bot configuré")

    def add_admin(self, chat_id):
        """Ajoute un admin pour recevoir les rapports"""
        if chat_id not in self.admin_chat_ids:
            self.admin_chat_ids.append(chat_id)
            print(f"[VERIF-M1] ✅ Admin {chat_id} ajouté")

    async def _wait_if_rate_limited(self):
        """Attend si limite API atteinte"""
        now = datetime.now()
        
        if (now - self.api_calls_reset_time).total_seconds() >= 60:
            self.api_calls_count = 0
            self.api_calls_reset_time = now
        
        if self.api_calls_count >= self.MAX_API_CALLS_PER_MINUTE:
            wait_time = 60 - (now - self.api_calls_reset_time).total_seconds()
            if wait_time > 0:
                print(f"[VERIF-M1] ⏳ Limite API - attente {wait_time:.0f}s")
                await asyncio.sleep(wait_time + 1)
                self.api_calls_count = 0
                self.api_calls_reset_time = datetime.now()

    def _increment_api_call(self):
        """Incrémente compteur appels API"""
        self.api_calls_count += 1

    def _is_weekend(self, timestamp):
        """Vérifie si le timestamp tombe le week-end"""
        if isinstance(timestamp, str):
            ts_clean = timestamp.replace('Z', '').replace('+00:00', '').split('.')[0]
            try:
                dt = datetime.fromisoformat(ts_clean)
            except:
                try:
                    dt = datetime.strptime(ts_clean, '%Y-%m-%d %H:%M:%S')
                except:
                    return True
        else:
            dt = timestamp
        
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        weekday = dt.weekday()
        hour = dt.hour
        
        # Samedi : fermé
        if weekday == 5:
            return True
        
        # Dimanche : fermé avant 22h UTC
        if weekday == 6 and hour < 22:
            return True
        
        # Vendredi : fermé après 22h UTC
        if weekday == 4 and hour >= 22:
            return True
        
        return False

    def _round_to_m1(self, dt):
        """Arrondit à la bougie M1 (minute pleine)"""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.replace(second=0, microsecond=0)

    def _calculate_next_m1_candle(self, signal_time):
        """
        CORRECTION M1 : Calcule la PROCHAINE bougie M1
        
        Logique M1 :
        - Signal à 14:23:47 → Prochaine bougie = 14:24:00
        - Signal à 14:23:01 → Prochaine bougie = 14:24:00
        - Signal à 14:23:00 → Prochaine bougie = 14:24:00
        
        Returns:
            (candle_start, candle_end)
        """
        # S'assurer que c'est en UTC
        if signal_time.tzinfo is None:
            signal_time = signal_time.replace(tzinfo=timezone.utc)
        
        current_second = signal_time.second
        
        # Pour M1, la prochaine bougie est toujours la minute suivante
        if current_second == 0:
            # Signal pile au début d'une minute
            # On prend la minute suivante pour être sûr que le trade a eu lieu
            candle_start = signal_time + timedelta(minutes=1)
        else:
            # Signal pendant une minute → prendre la minute suivante
            candle_start = signal_time.replace(second=0, microsecond=0) + timedelta(minutes=1)
        
        candle_end = candle_start + timedelta(minutes=1)
        
        return candle_start, candle_end

    async def verify_single_signal(self, signal_id):
        """
        Vérifie UN signal M1 - VERSION CORRIGÉE
        """
        try:
            print(f"\n{'='*70}")
            print(f"[VERIF-M1] 🔍 Vérification signal #{signal_id}")
            print(f"{'='*70}")
            
            # Récupérer le signal
            with self.engine.connect() as conn:
                signal = conn.execute(
                    text("""
                        SELECT id, pair, direction, ts_enter, confidence, kill_zone
                        FROM signals
                        WHERE id = :sid AND result IS NULL
                    """),
                    {"sid": signal_id}
                ).fetchone()
            
            if not signal:
                print(f"[VERIF-M1] ⚠️ Signal #{signal_id} déjà vérifié ou inexistant")
                return None
            
            signal_id, pair, direction, ts_enter, confidence, kill_zone = signal
            
            kz_text = f" [{kill_zone}]" if kill_zone else ""
            print(f"[VERIF-M1] 📊 {pair} {direction}{kz_text}")
            print(f"[VERIF-M1] 💪 Confiance: {confidence:.1%}")
            
            # Parser timestamp
            if isinstance(ts_enter, str):
                ts_clean = ts_enter.replace('Z', '').replace('+00:00', '').split('.')[0]
                try:
                    signal_time = datetime.fromisoformat(ts_clean)
                except:
                    signal_time = datetime.strptime(ts_clean, '%Y-%m-%d %H:%M:%S')
            else:
                signal_time = ts_enter
            
            if signal_time.tzinfo is None:
                signal_time = signal_time.replace(tzinfo=timezone.utc)
            
            # Vérifier week-end
            if self._is_weekend(signal_time):
                print(f"[VERIF-M1] 🏖️ Week-end - Marqué LOSE")
                self._update_signal_result(signal_id, 'LOSE', {
                    'entry_price': 0,
                    'exit_price': 0,
                    'pips': 0,
                    'gale_level': 0,
                    'reason': 'Marché fermé (week-end)'
                })
                return 'LOSE'
            
            # CORRECTION M1 : Calculer la PROCHAINE bougie M1
            trade_start, trade_end = self._calculate_next_m1_candle(signal_time)
            
            print(f"[VERIF-M1] 🕐 Signal envoyé  : {signal_time.strftime('%H:%M:%S')} UTC")
            print(f"[VERIF-M1] 📊 Bougie M1 tradée : {trade_start.strftime('%H:%M')} - {trade_end.strftime('%H:%M')} UTC")
            
            # Vérifier que la bougie M1 est complète
            now_utc = datetime.now(timezone.utc)
            if now_utc < trade_end:
                remaining = (trade_end - now_utc).total_seconds()
                print(f"[VERIF-M1] ⏳ Bougie M1 pas complète - reste {remaining:.0f}s")
                return None
            
            # Récupérer les prix de CETTE bougie M1 spécifique
            result, details = await self._verify_m1_candle(signal_id, pair, direction, trade_start)
            
            if result:
                self._update_signal_result(signal_id, result, details)
                emoji = "✅" if result == 'WIN' else "❌"
                print(f"[VERIF-M1] {emoji} Résultat M1: {result}")
                
                if details and details.get('pips'):
                    print(f"[VERIF-M1] 📊 {details['pips']:.1f} pips")
                
                return result
            else:
                print(f"[VERIF-M1] ⚠️ Impossible de vérifier")
                return None
                
        except Exception as e:
            print(f"[VERIF-M1] ❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _verify_m1_candle(self, signal_id, pair, direction, candle_start):
        """
        Vérifie une bougie M1 spécifique
        
        Args:
            signal_id: ID du signal
            pair: Paire tradée
            direction: CALL ou PUT
            candle_start: Début de la bougie M1 à vérifier
        
        Returns:
            (result, details) ou (None, None)
        """
        try:
            print(f"[VERIF-M1] 🔍 Récupération bougie M1 {candle_start.strftime('%H:%M')}...")
            
            # Récupérer le prix d'ouverture M1
            entry_price = await self._get_exact_m1_candle_price(pair, candle_start, 'open')
            if entry_price is None:
                print(f"[VERIF-M1] ❌ Prix d'ouverture M1 non disponible")
                return None, None
            
            await asyncio.sleep(2)  # Délai entre appels API
            
            # Récupérer le prix de fermeture M1
            exit_price = await self._get_exact_m1_candle_price(pair, candle_start, 'close')
            if exit_price is None:
                print(f"[VERIF-M1] ❌ Prix de fermeture M1 non disponible")
                return None, None
            
            # Calculer le résultat
            price_diff = exit_price - entry_price
            pips_diff = abs(price_diff) * 10000
            
            print(f"[VERIF-M1] 💰 Prix M1 entrée : {entry_price:.5f}")
            print(f"[VERIF-M1] 💰 Prix M1 sortie : {exit_price:.5f}")
            print(f"[VERIF-M1] 📊 Différence M1   : {price_diff:+.5f} ({pips_diff:.1f} pips)")
            
            # Vérification par direction
            if direction == 'CALL':
                is_winning = exit_price > entry_price
                print(f"[VERIF-M1] 🎯 CALL: {exit_price:.5f} > {entry_price:.5f} ? {is_winning}")
            else:  # PUT
                is_winning = exit_price < entry_price
                print(f"[VERIF-M1] 🎯 PUT: {exit_price:.5f} < {entry_price:.5f} ? {is_winning}")
            
            result = 'WIN' if is_winning else 'LOSE'
            
            details = {
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pips': pips_diff,
                'gale_level': 0
            }
            
            return result, details
            
        except Exception as e:
            print(f"[VERIF-M1] ❌ Erreur _verify_m1_candle: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    async def _get_exact_m1_candle_price(self, pair, candle_start, price_type='close'):
        """
        Récupère le prix d'UNE bougie M1 SPÉCIFIQUE
        
        CORRECTION M1 : Plage ÉTROITE et intervalle 1min
        
        Args:
            pair: Paire à récupérer
            candle_start: Début EXACT de la bougie M1
            price_type: 'open' ou 'close'
        
        Returns:
            float prix ou None
        """
        try:
            await self._wait_if_rate_limited()
            
            # Plage ÉTROITE pour M1 : ±2 minutes
            start_dt = candle_start - timedelta(minutes=2)
            end_dt = candle_start + timedelta(minutes=3)
            
            params = {
                'symbol': pair,
                'interval': '1min',  # ✅ M1 !
                'outputsize': 5,
                'apikey': self.api_key,
                'format': 'JSON',
                'start_date': start_dt.strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': end_dt.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            print(f"[VERIF-M1] 🔍 API M1: {pair} {price_type} à {candle_start.strftime('%H:%M')} UTC")
            
            resp = self._session.get(self.base_url, params=params, timeout=12)
            self._increment_api_call()
            
            resp.raise_for_status()
            data = resp.json()
            
            # Vérifier limite API
            if 'code' in data and data['code'] == 429:
                print(f"[VERIF-M1] ⚠️ Limite API atteinte")
                await asyncio.sleep(60)
                self.api_calls_count = 0
                self.api_calls_reset_time = datetime.now()
                return None
            
            if 'values' not in data or not data['values']:
                print(f"[VERIF-M1] ❌ Aucune donnée M1 retournée")
                return None
            
            # Chercher LA bougie M1 exacte
            for candle in data['values']:
                try:
                    candle_time = datetime.fromisoformat(candle['datetime'].replace('Z', '+00:00'))
                except:
                    try:
                        candle_time = datetime.strptime(candle['datetime'], '%Y-%m-%d %H:%M:%S')
                        if candle_time.tzinfo is None:
                            candle_time = candle_time.replace(tzinfo=timezone.utc)
                    except:
                        continue
                
                # Arrondir à M1 (minute pleine)
                candle_time_m1 = self._round_to_m1(candle_time)
                
                # Comparaison EXACTE pour M1 (tolérance 30 secondes)
                time_diff = abs((candle_time_m1 - candle_start).total_seconds())
                
                if time_diff < 30:  # ✅ Tolérance 30s pour M1
                    try:
                        price = float(candle[price_type])
                        print(f"[VERIF-M1] ✅ Bougie M1 trouvée - {price_type}: {price:.5f}")
                        return price
                    except:
                        # Fallback sur close
                        try:
                            price = float(candle['close'])
                            print(f"[VERIF-M1] ⚠️ Fallback M1 close: {price:.5f}")
                            return price
                        except:
                            continue
            
            print(f"[VERIF-M1] ❌ Bougie M1 {candle_start.strftime('%H:%M')} NON trouvée")
            return None
            
        except Exception as e:
            print(f"[VERIF-M1] ❌ Erreur API M1: {e}")
            return None

    def _update_signal_result(self, signal_id, result, details):
        """Met à jour le résultat dans la DB"""
        try:
            gale_level = 0
            reason = details.get('reason', '') if details else ''
            
            query = text("""
                UPDATE signals
                SET result = :result, gale_level = :gale_level, reason = :reason
                WHERE id = :id
            """)
            
            with self.engine.begin() as conn:
                conn.execute(query, {
                    'result': result,
                    'gale_level': gale_level,
                    'reason': reason,
                    'id': signal_id
                })
            
            print(f"[VERIF-M1] 💾 Résultat M1 sauvegardé: {result}")
            
        except Exception as e:
            print(f"[VERIF-M1] ❌ Erreur sauvegarde: {e}")
            try:
                query = text("UPDATE signals SET result = :result WHERE id = :id")
                with self.engine.begin() as conn:
                    conn.execute(query, {'result': result, 'id': signal_id})
                print(f"[VERIF-M1] 💾 Sauvegardé (version simple)")
            except Exception as e2:
                print(f"[VERIF-M1] ❌ Échec total: {e2}")

    async def verify_pending_signals(self):
        """Vérifie tous les signaux M1 en attente"""
        try:
            now_utc = datetime.now(timezone.utc)
            print(f"\n{'='*70}")
            print(f"[VERIF-M1] 🔍 VÉRIFICATION AUTO M1 - {now_utc.strftime('%H:%M:%S')} UTC")
            print(f"{'='*70}")

            query = text("""
                SELECT id, pair, direction, ts_enter, confidence, kill_zone
                FROM signals
                WHERE result IS NULL
                ORDER BY ts_enter ASC
                LIMIT 50
            """)
            
            with self.engine.connect() as conn:
                pending = conn.execute(query).fetchall()
            
            print(f"[VERIF-M1] 📊 Signaux M1 en attente: {len(pending)}")
            
            if not pending:
                print(f"[VERIF-M1] ✅ Aucun signal M1 à vérifier")
                return
            
            # Limite : 3 signaux max par cycle (API rate limiting)
            MAX_PER_CYCLE = 3
            
            if len(pending) > MAX_PER_CYCLE:
                print(f"[VERIF-M1] ⚠️ {len(pending)} signaux - vérification de {MAX_PER_CYCLE}")
                to_verify = pending[:MAX_PER_CYCLE]
            else:
                to_verify = pending
            
            verified_count = 0
            skipped_count = 0
            error_count = 0
            
            for signal_row in to_verify:
                try:
                    signal_id = signal_row[0]
                    
                    result = await self.verify_single_signal(signal_id)
                    
                    if result:
                        verified_count += 1
                    elif result is None:
                        skipped_count += 1
                    else:
                        error_count += 1
                    
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    error_count += 1
                    print(f"[VERIF-M1] ❌ Erreur signal: {e}")
            
            print(f"\n{'='*70}")
            print(f"[VERIF-M1] 📈 RÉSUMÉ M1: {verified_count} vérifiés, {skipped_count} en attente, {error_count} erreurs")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"[VERIF-M1] ❌ Erreur globale: {e}")
            import traceback
            traceback.print_exc()


# Fonctions utilitaires M1 à ajouter dans utils.py
def round_to_m1_candle(dt):
    """Arrondit un datetime à la bougie M1 la plus proche"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.replace(second=0, microsecond=0)


def get_next_m1_candle(dt):
    """Retourne le début de la PROCHAINE bougie M1"""
    current_candle = round_to_m1_candle(dt)
    return current_candle + timedelta(minutes=1)


def get_m1_candle_range(dt):
    """Retourne le début et la fin de la bougie M1"""
    start = round_to_m1_candle(dt)
    end = start + timedelta(minutes=1)
    return start, end
