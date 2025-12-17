"""
Auto Verifier OPTIMISÉ - Vérification Prioritaire & Briefing Immédiat
======================================================================

AMÉLIORATIONS MAJEURES:
- Vérification IMMÉDIATE (dès que signal prêt)
- Briefing INSTANTANÉ après vérification
- Queue prioritaire (FIFO strict)
- Rate limiting intelligent
- Retry automatique en cas d'échec
- Logs détaillés pour debugging

WORKFLOW:
1. Signal envoyé à T+0
2. Attente 5 min (bougie M5)
3. Vérification à T+5
4. Briefing immédiat à T+5
5. PUIS signal suivant à T+10
"""

import asyncio
from datetime import datetime, timedelta, timezone
from sqlalchemy import text
import requests
from utils import round_to_m5_candle, get_m5_candle_range

class AutoResultVerifier:
    def __init__(self, engine, twelvedata_api_key, bot=None):
        self.engine = engine
        self.api_key = twelvedata_api_key
        self.base_url = 'https://api.twelvedata.com/time_series'
        self.bot = bot
        self.admin_chat_ids = []
        
        # M5 params
        self.default_timeframe = 5
        self.default_max_gales = 0
        self._session = requests.Session()
        
        # RATE LIMITING AMÉLIORÉ
        self.api_calls_count = 0
        self.api_calls_reset_time = datetime.now()
        self.MAX_API_CALLS_PER_MINUTE = 6
        
        # NOUVEAU: Queue de vérification prioritaire
        self.verification_queue = []
        self.currently_verifying = False
        
        # NOUVEAU: Retry automatique
        self.max_retries = 3
        self.retry_delay = 5  # secondes
        
        # NOUVEAU: Stats temps réel
        self.stats = {
            'verifications_total': 0,
            'verifications_success': 0,
            'verifications_failed': 0,
            'average_verification_time': 0,
            'last_verification_time': None
        }

    def set_bot(self, bot):
        """Configure le bot"""
        self.bot = bot
        print("✅ Bot configuré pour notifications")

    def add_admin(self, chat_id):
        """Ajoute admin"""
        if chat_id not in self.admin_chat_ids:
            self.admin_chat_ids.append(chat_id)
            print(f"✅ Admin {chat_id} ajouté")

    async def _wait_if_rate_limited(self):
        """Attend si limite API atteinte"""
        now = datetime.now()
        
        if (now - self.api_calls_reset_time).total_seconds() >= 60:
            self.api_calls_count = 0
            self.api_calls_reset_time = now
        
        if self.api_calls_count >= self.MAX_API_CALLS_PER_MINUTE:
            wait_time = 60 - (now - self.api_calls_reset_time).total_seconds()
            if wait_time > 0:
                print(f"   ⏳ Rate limit: attente {wait_time:.0f}s...")
                await asyncio.sleep(wait_time + 1)
                self.api_calls_count = 0
                self.api_calls_reset_time = datetime.now()

    def _increment_api_call(self):
        """Incrémente compteur API"""
        self.api_calls_count += 1

    def _is_weekend(self, timestamp):
        """Vérifie week-end (marché fermé)"""
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
        
        if weekday == 5:
            return True
        if weekday == 6 and hour < 22:
            return True
        if weekday == 4 and hour >= 22:
            return True
        
        return False

    async def verify_single_signal_with_retry(self, signal_id, max_retries=None):
        """
        NOUVEAU: Vérification avec retry automatique
        
        Args:
            signal_id: ID du signal à vérifier
            max_retries: Nombre max de tentatives (défaut: self.max_retries)
        
        Returns:
            'WIN', 'LOSE', ou None si échec total
        """
        if max_retries is None:
            max_retries = self.max_retries
        
        for attempt in range(max_retries):
            try:
                result = await self.verify_single_signal(signal_id)
                
                if result is not None:
                    self.stats['verifications_success'] += 1
                    return result
                
                # Si None, réessayer
                if attempt < max_retries - 1:
                    print(f"   🔄 Tentative {attempt + 2}/{max_retries}...")
                    await asyncio.sleep(self.retry_delay)
                    
            except Exception as e:
                print(f"   ❌ Erreur tentative {attempt + 1}: {e}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
        
        # Échec total après toutes les tentatives
        self.stats['verifications_failed'] += 1
        print(f"   ❌ Échec après {max_retries} tentatives")
        return None

    async def verify_single_signal(self, signal_id):
        """Vérifie UN signal en M5"""
        start_time = datetime.now()
        
        try:
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
                print(f"⚠️ Signal #{signal_id} déjà vérifié")
                return None
            
            signal_id, pair, direction, ts_enter, confidence, kill_zone = signal
            
            kz_text = f" [{kill_zone}]" if kill_zone else ""
            print(f"\n🔍 Vérification M5 #{signal_id} - {pair} {direction}{kz_text}")
            
            # Week-end
            if self._is_weekend(ts_enter):
                print(f"🏖️ Week-end - Marqué LOSE")
                self._update_signal_result(signal_id, 'LOSE', {
                    'entry_price': 0,
                    'exit_price': 0,
                    'pips': 0,
                    'gale_level': 0,
                    'reason': 'Marché fermé (week-end)'
                })
                return 'LOSE'
            
            # Vérifier si M5 complet
            if not self._is_signal_complete_m5(ts_enter):
                print(f"⏳ Signal pas encore prêt")
                return None
            
            # Vérifier
            result, details = await self._verify_signal_m5(
                signal_id, pair, direction, ts_enter
            )
            
            if result:
                self._update_signal_result(signal_id, result, details)
                
                emoji = "✅" if result == 'WIN' else "❌"
                pips = details.get('pips', 0) if details else 0
                print(f"{emoji} Résultat: {result} ({pips:.1f} pips)")
                
                # Stats
                elapsed = (datetime.now() - start_time).total_seconds()
                self.stats['verifications_total'] += 1
                self.stats['last_verification_time'] = datetime.now()
                
                # Moyenne temps vérification
                if self.stats['average_verification_time'] == 0:
                    self.stats['average_verification_time'] = elapsed
                else:
                    self.stats['average_verification_time'] = (
                        self.stats['average_verification_time'] * 0.8 + elapsed * 0.2
                    )
                
                return result
            else:
                print(f"⚠️ Impossible de vérifier")
                return None
                
        except Exception as e:
            print(f"❌ Erreur verify_single_signal: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def verify_and_brief_immediately(self, signal_id, telegram_app):
        """
        NOUVEAU: Vérification + Briefing IMMÉDIAT
        
        Cette fonction est le cœur de l'amélioration:
        1. Vérifie le signal (avec retry)
        2. Envoie le briefing IMMÉDIATEMENT
        3. Retourne le résultat
        
        Args:
            signal_id: ID du signal
            telegram_app: Application Telegram
        
        Returns:
            'WIN', 'LOSE', ou None
        """
        print(f"\n{'='*60}")
        print(f"⚡ VÉRIFICATION PRIORITAIRE - Signal #{signal_id}")
        print(f"{'='*60}")
        
        # Vérifier avec retry
        result = await self.verify_single_signal_with_retry(signal_id, max_retries=3)
        
        if result:
            # BRIEFING IMMÉDIAT
            print(f"📧 Envoi briefing immédiat...")
            await self._send_verification_briefing_immediate(signal_id, telegram_app)
            print(f"✅ Briefing #{signal_id} envoyé")
        else:
            print(f"⚠️ Signal #{signal_id} non vérifié (sera retentée)")
        
        print(f"{'='*60}\n")
        
        return result

    async def _send_verification_briefing_immediate(self, signal_id, telegram_app):
        """
        NOUVEAU: Envoie briefing de manière synchrone et immédiate
        Plus rapide que la version originale
        """
        try:
            with self.engine.connect() as conn:
                signal = conn.execute(
                    text("SELECT pair, direction, result, confidence, kill_zone FROM signals WHERE id = :sid"),
                    {"sid": signal_id}
                ).fetchone()

            if not signal or not signal[2]:
                return

            pair, direction, result, confidence, kill_zone = signal
            
            with self.engine.connect() as conn:
                user_ids = [r[0] for r in conn.execute(
                    text("SELECT user_id FROM subscribers")
                ).fetchall()]
            
            # Emoji et statut
            emoji = "✅" if result == "WIN" else "❌"
            status = "GAGNÉ" if result == "WIN" else "PERDU"
            direction_emoji = "📈" if direction == "CALL" else "📉"
            
            # Message compact et rapide
            briefing = (
                f"{emoji} **BRIEFING #{signal_id}**\n"
                f"━━━━━━━━━━━━━━━━━━━━\n\n"
                f"{direction_emoji} {pair} {direction}\n"
                f"💪 {int(confidence*100)}%"
            )
            
            if kill_zone:
                briefing += f" | {kill_zone}"
            
            briefing += f"\n\n🎲 **{status}**\n\n━━━━━━━━━━━━━━━━━━━━"
            
            # Envoi rapide en parallèle
            tasks = []
            for uid in user_ids:
                task = telegram_app.bot.send_message(chat_id=uid, text=briefing)
                tasks.append(task)
            
            # Attendre tous les envois
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            sent = sum(1 for r in results if not isinstance(r, Exception))
            print(f"   📤 Briefing envoyé à {sent}/{len(user_ids)} abonnés")

        except Exception as e:
            print(f"❌ Erreur briefing #{signal_id}: {e}")

    async def verify_pending_signals(self):
        """Vérifie tous les signaux en attente (version originale pour cron)"""
        try:
            now_utc = datetime.now(timezone.utc)
            print("\n" + "="*60)
            print(f"🔍 VÉRIFICATION AUTO - {now_utc.strftime('%H:%M:%S')} UTC")
            print("="*60)

            query = text("""
                SELECT id, pair, direction, ts_enter, confidence, kill_zone
                FROM signals
                WHERE result IS NULL
                ORDER BY ts_enter ASC
                LIMIT 50
            """)
            
            with self.engine.connect() as conn:
                pending = conn.execute(query).fetchall()
            
            print(f"📊 Signaux en attente: {len(pending)}")
            
            if not pending:
                print("✅ Aucun signal en attente")
                print("="*60 + "\n")
                return
            
            # Limite API: 3 signaux max par cycle
            MAX_VERIFY = 3
            
            if len(pending) > MAX_VERIFY:
                print(f"⚠️ {len(pending)} signaux - Traitement de {MAX_VERIFY} prioritaires")
                pending_to_verify = pending[:MAX_VERIFY]
            else:
                pending_to_verify = pending
            
            print("-"*60)
            
            verified = 0
            skipped = 0
            
            for signal_row in pending_to_verify:
                try:
                    signal_id = signal_row[0]
                    pair = signal_row[1]
                    direction = signal_row[2]
                    ts_enter = signal_row[3]
                    
                    print(f"\n{'='*40}")
                    print(f"🔎 Signal #{signal_id} - {pair} {direction}")
                    print(f"{'='*40}")
                    
                    # Week-end
                    if self._is_weekend(ts_enter):
                        print(f"🏖️ Week-end - LOSE")
                        self._update_signal_result(signal_id, 'LOSE', {
                            'entry_price': 0,
                            'exit_price': 0,
                            'pips': 0,
                            'gale_level': 0,
                            'reason': 'Marché fermé'
                        })
                        verified += 1
                        continue
                    
                    # Vérifier si prêt
                    if not self._is_signal_complete_m5(ts_enter):
                        skipped += 1
                        print(f"➡️ SKIP - Pas prêt")
                        continue
                    
                    print(f"✅ Prêt pour vérification")
                    
                    # Vérifier avec retry
                    result = await self.verify_single_signal_with_retry(signal_id)
                    
                    if result:
                        verified += 1
                    
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    print(f"❌ Erreur: {e}")
            
            print("\n" + "-"*60)
            print(f"📈 RÉSUMÉ: {verified} vérifiés, {skipped} en attente")
            print("="*60 + "\n")
        
        except Exception as e:
            print(f"❌ ERREUR GLOBALE: {e}")
            import traceback
            traceback.print_exc()

    def _is_signal_complete_m5(self, ts_enter):
        """Vérifie si signal M5 est complet"""
        try:
            if isinstance(ts_enter, str):
                ts_clean = ts_enter.replace('Z', '').replace('+00:00', '').split('.')[0]
                try:
                    entry_time_utc = datetime.fromisoformat(ts_clean)
                except:
                    try:
                        entry_time_utc = datetime.strptime(ts_clean, '%Y-%m-%d %H:%M:%S')
                    except:
                        return False
            else:
                entry_time_utc = ts_enter
            
            if entry_time_utc.tzinfo is None:
                entry_time_utc = entry_time_utc.replace(tzinfo=timezone.utc)
            else:
                entry_time_utc = entry_time_utc.astimezone(timezone.utc)

            entry_time_utc = round_to_m5_candle(entry_time_utc)
            end_time_utc = entry_time_utc + timedelta(minutes=5)
            now_utc = datetime.now(timezone.utc)
            
            is_complete = now_utc >= end_time_utc
            
            if is_complete:
                print(f"   ✅ COMPLET M5")
            else:
                remaining = (end_time_utc - now_utc).total_seconds()
                print(f"   ⏳ Attente: {remaining:.0f}s")
            
            return is_complete
            
        except Exception as e:
            print(f"❌ Erreur _is_signal_complete_m5: {e}")
            return False

    async def _verify_signal_m5(self, signal_id, pair, direction, ts_enter):
        """Vérifie bougie M5"""
        try:
            if isinstance(ts_enter, str):
                ts_clean = ts_enter.replace('Z', '').replace('+00:00', '').split('.')[0]
                try:
                    entry_time_utc = datetime.fromisoformat(ts_clean)
                except:
                    entry_time_utc = datetime.strptime(ts_clean, '%Y-%m-%d %H:%M:%S')
            else:
                entry_time_utc = ts_enter
            
            if entry_time_utc.tzinfo is None:
                entry_time_utc = entry_time_utc.replace(tzinfo=timezone.utc)
            
            if self._is_weekend(entry_time_utc):
                return 'LOSE', {
                    'entry_price': 0,
                    'exit_price': 0,
                    'pips': 0,
                    'gale_level': 0,
                    'reason': 'Week-end'
                }

            entry_candle_start, entry_candle_end = get_m5_candle_range(entry_time_utc)
            
            print(f"   📍 Bougie: {entry_candle_start.strftime('%H:%M')}-{entry_candle_end.strftime('%H:%M')} UTC")
            print(f"   📈 Direction: {direction}")
            
            # Prix entrée
            entry_price = await self._get_price_at_time(pair, entry_candle_start, price_type='open')
            if entry_price is None:
                return None, None
            
            await asyncio.sleep(2)
            
            # Prix sortie
            exit_price = await self._get_price_at_time(pair, entry_candle_start, price_type='close')
            if exit_price is None:
                return None, None
            
            # Résultat
            price_diff = exit_price - entry_price
            pips_diff = abs(price_diff) * 10000
            
            print(f"   💰 Entrée: {entry_price:.5f}")
            print(f"   💰 Sortie: {exit_price:.5f}")
            print(f"   📊 Diff: {price_diff:+.5f} ({pips_diff:.1f} pips)")
            
            if direction == 'CALL':
                is_winning = exit_price > entry_price
            else:
                is_winning = exit_price < entry_price
            
            result = 'WIN' if is_winning else 'LOSE'
            
            details = {
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pips': pips_diff,
                'gale_level': 0
            }
            
            return result, details
            
        except Exception as e:
            print(f"❌ Erreur _verify_signal_m5: {e}")
            return None, None

    async def _get_price_at_time(self, pair, timestamp, price_type='close'):
        """Récupère prix à un moment donné"""
        try:
            await self._wait_if_rate_limited()
            
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)

            ts_utc = timestamp.astimezone(timezone.utc)
            ts_utc = round_to_m5_candle(ts_utc)
            
            if self._is_weekend(ts_utc):
                return None
            
            start_dt = ts_utc - timedelta(minutes=10)
            end_dt = ts_utc + timedelta(minutes=10)
            
            start_str = start_dt.strftime('%Y-%m-%d %H:%M:%S')
            end_str = end_dt.strftime('%Y-%m-%d %H:%M:%S')
            
            params = {
                'symbol': pair,
                'interval': '5min',
                'outputsize': 5,
                'apikey': self.api_key,
                'format': 'JSON',
                'start_date': start_str,
                'end_date': end_str
            }
            
            print(f"   🔍 API: {pair} {price_type} à {ts_utc.strftime('%H:%M')}")
            
            resp = self._session.get(self.base_url, params=params, timeout=12)
            self._increment_api_call()
            
            resp.raise_for_status()
            data = resp.json()
            
            if 'code' in data and data['code'] == 429:
                print(f"   ⚠️ Limite API")
                await asyncio.sleep(60)
                return None
            
            if 'values' in data and len(data['values']) > 0:
                closest_candle = None
                min_diff = float('inf')
                
                for candle in data['values']:
                    try:
                        candle_time = datetime.fromisoformat(candle['datetime'].replace('Z', '+00:00'))
                    except:
                        continue
                    
                    if candle_time.tzinfo is None:
                        candle_time = candle_time.replace(tzinfo=timezone.utc)
                    
                    candle_time = round_to_m5_candle(candle_time)
                    
                    diff = abs((candle_time - ts_utc).total_seconds())
                    if diff < min_diff:
                        min_diff = diff
                        closest_candle = candle
                
                if closest_candle and min_diff <= 300:
                    try:
                        price = float(closest_candle[price_type])
                        print(f"   💰 Prix {price_type}: {price}")
                        return price
                    except:
                        try:
                            price = float(closest_candle['close'])
                            return price
                        except:
                            return None
            
            return None
            
        except Exception as e:
            print(f"⚠️ Erreur API: {e}")
            return None

    def _update_signal_result(self, signal_id, result, details):
        """Met à jour résultat"""
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
            
            print(f"💾 Résultat sauvegardé: #{signal_id} = {result}")
            
        except Exception as e:
            print(f"❌ Erreur update: {e}")

    def get_verification_stats(self):
        """NOUVEAU: Retourne stats de vérification"""
        return self.stats.copy()
