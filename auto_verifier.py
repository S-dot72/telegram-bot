import asyncio
from datetime import datetime, timedelta, timezone
from sqlalchemy import text
import requests

class AutoResultVerifier:
    def __init__(self, engine, twelvedata_api_key):
        self.engine = engine
        self.api_key = twelvedata_api_key
        self.base_url = 'https://api.twelvedata.com/time_series'
        self._session = requests.Session()
        
        # Rate limiting
        self.api_calls_count = 0
        self.api_calls_reset_time = datetime.now()
        self.MAX_API_CALLS_PER_MINUTE = 6

    async def _wait_if_rate_limited(self):
        """Attend si limite API atteinte"""
        now = datetime.now()
        
        if (now - self.api_calls_reset_time).total_seconds() >= 60:
            self.api_calls_count = 0
            self.api_calls_reset_time = now
        
        if self.api_calls_count >= self.MAX_API_CALLS_PER_MINUTE:
            wait_time = 60 - (now - self.api_calls_reset_time).total_seconds()
            if wait_time > 0:
                print(f"   ⏳ Limite API, attente {wait_time:.0f}s...")
                await asyncio.sleep(wait_time + 1)
                self.api_calls_count = 0
                self.api_calls_reset_time = datetime.now()

    def _increment_api_call(self):
        self.api_calls_count += 1

    def _is_weekend(self, timestamp):
        """Vérifie si le timestamp tombe le week-end"""
        if isinstance(timestamp, str):
            ts_clean = timestamp.replace('Z', '').replace('+00:00', '').split('.')[0]
            try:
                dt = datetime.fromisoformat(ts_clean)
            except:
                dt = datetime.strptime(ts_clean, '%Y-%m-%d %H:%M:%S')
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

    def _round_to_m1_candle(self, dt):
        """Arrondit à la minute (bougie M1)"""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.replace(second=0, microsecond=0)

    def _get_m1_candle_range(self, dt):
        """Retourne début et fin bougie M1"""
        start = self._round_to_m1_candle(dt)
        end = start + timedelta(minutes=1)
        return start, end

    async def verify_single_signal(self, signal_id):
        """Vérifie un signal M1"""
        try:
            with self.engine.connect() as conn:
                signal = conn.execute(
                    text("""
                        SELECT id, pair, direction, ts_enter, confidence, timeframe
                        FROM signals
                        WHERE id = :sid AND result IS NULL
                    """),
                    {"sid": signal_id}
                ).fetchone()
            
            if not signal:
                print(f"⚠️ Signal #{signal_id} déjà vérifié")
                return None
            
            signal_id, pair, direction, ts_enter, confidence, timeframe = signal
            
            print(f"\n🔍 Vérification M1 signal #{signal_id} - {pair} {direction}")
            
            # Vérifier week-end
            if self._is_weekend(ts_enter):
                print(f"🏖️ Week-end - Marqué LOSE")
                self._update_signal_result(signal_id, 'LOSE', {
                    'reason': 'Marché fermé (week-end)'
                })
                return 'LOSE'
            
            # Vérifier si signal M1 complet
            if not self._is_signal_complete_m1(ts_enter):
                print(f"⏳ Signal M1 pas encore prêt")
                return None
            
            # Vérifier signal M1
            result, details = await self._verify_signal_m1(
                signal_id, pair, direction, ts_enter
            )
            
            if result:
                self._update_signal_result(signal_id, result, details)
                emoji = "✅" if result == 'WIN' else "❌"
                print(f"{emoji} Résultat M1: {result}")
                return result
            else:
                print(f"⚠️ Impossible de vérifier")
                return None
                
        except Exception as e:
            print(f"❌ Erreur verify_single_signal: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _is_signal_complete_m1(self, ts_enter):
        """Vérifie si signal M1 est complet (1 minute écoulée)"""
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
            
            # Arrondir à la minute
            entry_time_utc = self._round_to_m1_candle(entry_time_utc)
            
            # M1: vérifier 1 minute après l'entrée
            end_time_utc = entry_time_utc + timedelta(minutes=1)
            
            now_utc = datetime.now(timezone.utc)
            is_complete = now_utc >= end_time_utc
            
            print(f"   📅 Entrée M1: {entry_time_utc.strftime('%H:%M:%S')}")
            print(f"   📅 Fin M1: {end_time_utc.strftime('%H:%M:%S')}")
            print(f"   📅 Maintenant: {now_utc.strftime('%H:%M:%S')}")
            
            if is_complete:
                print(f"   ✅ COMPLET M1")
            else:
                remaining = (end_time_utc - now_utc).total_seconds()
                print(f"   ⏳ PAS COMPLET - {remaining:.0f}s")
            
            return is_complete
            
        except Exception as e:
            print(f"❌ Erreur _is_signal_complete_m1: {e}")
            return False

    async def _verify_signal_m1(self, signal_id, pair, direction, ts_enter):
        """Vérifie bougie M1"""
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
                return 'LOSE', {'reason': 'Week-end'}
            
            # Arrondir à la bougie M1
            entry_candle_start, entry_candle_end = self._get_m1_candle_range(entry_time_utc)
            
            print(f"   📍 M1: {entry_candle_start.strftime('%H:%M')}-{entry_candle_end.strftime('%H:%M')}")
            print(f"   📈 Direction: {direction}")
            
            # Prix d'entrée (open de la bougie M1)
            entry_price = await self._get_price_at_time(pair, entry_candle_start, price_type='open')
            if entry_price is None:
                print(f"   ⚠️ Prix d'entrée M1 indisponible")
                return None, None
            
            await asyncio.sleep(2)
            
            # Prix de sortie (close de la bougie M1)
            exit_price = await self._get_price_at_time(pair, entry_candle_start, price_type='close')
            if exit_price is None:
                print(f"   ⚠️ Prix de sortie M1 indisponible")
                return None, None
            
            # Calculer résultat
            price_diff = exit_price - entry_price
            pips_diff = abs(price_diff) * 10000
            
            print(f"   💰 Entrée: {entry_price:.5f}")
            print(f"   💰 Sortie: {exit_price:.5f}")
            print(f"   📊 Diff: {price_diff:+.5f} ({pips_diff:.1f} pips)")
            
            if direction == 'CALL':
                is_winning = exit_price > entry_price
                print(f"   🎯 CALL: {exit_price:.5f} > {entry_price:.5f} ? {is_winning}")
            else:
                is_winning = exit_price < entry_price
                print(f"   🎯 PUT: {exit_price:.5f} < {entry_price:.5f} ? {is_winning}")
            
            result = 'WIN' if is_winning else 'LOSE'
            details = {
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pips': pips_diff,
                'gale_level': 0
            }
            
            emoji = "✅" if is_winning else "❌"
            print(f"   {emoji} {result} M1 ({pips_diff:+.1f} pips)")
            
            return result, details
            
        except Exception as e:
            print(f"❌ Erreur _verify_signal_m1: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    async def _get_price_at_time(self, pair, timestamp, price_type='close'):
        """Récupère prix à un moment donné (M1)"""
        try:
            await self._wait_if_rate_limited()
            
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            
            ts_utc = timestamp.astimezone(timezone.utc)
            ts_utc = self._round_to_m1_candle(ts_utc)
            
            if self._is_weekend(ts_utc):
                print(f"   🏖️ Week-end - Pas d'appel API")
                return None
            
            # Plage M1: ±5 minutes
            start_dt = ts_utc - timedelta(minutes=5)
            end_dt = ts_utc + timedelta(minutes=5)
            
            params = {
                'symbol': pair,
                'interval': '1min',
                'outputsize': 10,
                'apikey': self.api_key,
                'format': 'JSON',
                'start_date': start_dt.strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': end_dt.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            print(f"   🔍 API M1: {pair} {price_type} à {ts_utc.strftime('%H:%M')}")
            
            resp = self._session.get(self.base_url, params=params, timeout=12)
            self._increment_api_call()
            
            resp.raise_for_status()
            data = resp.json()
            
            if 'code' in data and data['code'] == 429:
                print(f"   ⚠️ LIMITE API")
                await asyncio.sleep(60)
                self.api_calls_count = 0
                return None
            
            if 'values' in data and len(data['values']) > 0:
                closest_candle = None
                min_diff = float('inf')
                
                for candle in data['values']:
                    try:
                        candle_time = datetime.fromisoformat(candle['datetime'].replace('Z', '+00:00'))
                    except:
                        candle_time = datetime.strptime(candle['datetime'], '%Y-%m-%d %H:%M:%S')
                    
                    if candle_time.tzinfo is None:
                        candle_time = candle_time.replace(tzinfo=timezone.utc)
                    
                    candle_time = self._round_to_m1_candle(candle_time)
                    
                    diff = abs((candle_time - ts_utc).total_seconds())
                    if diff < min_diff:
                        min_diff = diff
                        closest_candle = candle
                
                # M1: tolérance 1 minute
                if closest_candle and min_diff <= 60:
                    try:
                        price = float(closest_candle[price_type])
                        print(f"   💰 Prix {price_type}: {price} (diff: {min_diff:.0f}s)")
                        return price
                    except:
                        try:
                            price = float(closest_candle['close'])
                            print(f"   💰 Prix close (fallback): {price}")
                            return price
                        except:
                            return None
            
            print(f"   ⚠️ Aucune bougie M1 trouvée")
            return None
            
        except Exception as e:
            print(f"⚠️ Erreur API M1: {e}")
            return None

    def _update_signal_result(self, signal_id, result, details):
        """Met à jour résultat dans DB"""
        try:
            reason = details.get('reason', '') if details else ''
            
            query = text("""
                UPDATE signals
                SET result = :result, gale_level = 0, reason = :reason
                WHERE id = :id
            """)
            
            with self.engine.begin() as conn:
                conn.execute(query, {
                    'result': result,
                    'reason': reason,
                    'id': signal_id
                })
            
            print(f"💾 Résultat M1 sauvegardé: #{signal_id} = {result}")
            
        except Exception as e:
            print(f"❌ Erreur _update_signal_result: {e}")
