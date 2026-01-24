import asyncio
from datetime import datetime, timedelta, timezone
from sqlalchemy import text
import requests
import pandas as pd
import json

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
        
        # Samedi ou dimanche = week-end
        return weekday in [5, 6]

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
            print(f"\n🔍 Vérification signal #{signal_id}")
            
            with self.engine.connect() as conn:
                signal = conn.execute(
                    text("""
                        SELECT id, pair, direction, ts_enter, confidence, timeframe, payload_json
                        FROM signals
                        WHERE id = :sid
                    """),
                    {"sid": signal_id}
                ).fetchone()
            
            if not signal:
                print(f"⚠️ Signal #{signal_id} non trouvé")
                return None
            
            signal_id, pair, direction, ts_enter, confidence, timeframe, payload_json = signal
            
            # Vérifier si déjà vérifié
            with self.engine.connect() as conn:
                already_verified = conn.execute(
                    text("SELECT result FROM signals WHERE id = :sid AND result IS NOT NULL"),
                    {"sid": signal_id}
                ).fetchone()
            
            if already_verified:
                result = already_verified[0]
                print(f"✅ Signal #{signal_id} déjà vérifié: {result}")
                return result
            
            print(f"📊 Vérification M1 signal #{signal_id} - {pair} {direction}")
            
            # Analyser le payload pour voir si c'était en mode OTC
            is_otc = False
            original_pair = None
            if payload_json:
                try:
                    payload = json.loads(payload_json)
                    mode = payload.get('mode', 'Forex')
                    original_pair = payload.get('original_pair', pair)
                    
                    if mode == 'OTC':
                        is_otc = True
                        print(f"🏖️ Mode OTC détecté: {original_pair} → {pair}")
                except:
                    pass
            
            # Vérifier si signal M1 complet
            if not self._is_signal_complete_m1(ts_enter):
                print(f"⏳ Signal M1 pas encore prêt")
                return None
            
            # Vérifier signal M1
            if is_otc:
                print(f"⚠️ Mode OTC - Vérification limitée (pas d'API crypto)")
                # En mode OTC, on ne peut pas vérifier via TwelveData
                # On pourrait implémenter une vérification manuelle ou via autre API
                result = None
                details = {'reason': 'Vérification OTC non disponible'}
            else:
                # Mode Forex - vérifier via TwelveData
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
            
            if is_complete:
                print(f"   ✅ COMPLET M1 (attendu: {end_time_utc.strftime('%H:%M:%S')})")
            else:
                remaining = (end_time_utc - now_utc).total_seconds()
                print(f"   ⏳ PAS COMPLET - {remaining:.0f}s restants")
            
            return is_complete
            
        except Exception as e:
            print(f"❌ Erreur _is_signal_complete_m1: {e}")
            return False

    async def _verify_signal_m1(self, signal_id, pair, direction, ts_enter):
        """Vérifie bougie M1 pour Forex"""
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
            
            # Vérifier week-end - pour Forex seulement
            if self._is_weekend(entry_time_utc):
                print(f"   🏖️ Week-end Forex - Vérification impossible")
                return None, {'reason': 'Forex fermé le week-end'}
            
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
            exit_price = await self._get_price_at_time(pair, entry_candle_end, price_type='close')
            if exit_price is None:
                print(f"   ⚠️ Prix de sortie M1 indisponible")
                # Essayer avec le début de la bougie suivante
                next_candle_start = entry_candle_end
                exit_price = await self._get_price_at_time(pair, next_candle_start, price_type='open')
                if exit_price is None:
                    return None, None
            
            # Calculer résultat
            price_diff = exit_price - entry_price
            pips_diff = abs(price_diff) * 10000
            
            print(f"   💰 Entrée (open): {entry_price:.5f}")
            print(f"   💰 Sortie (close): {exit_price:.5f}")
            print(f"   📊 Diff: {price_diff:+.5f} ({pips_diff:.1f} pips)")
            
            if direction == 'CALL':
                is_winning = exit_price > entry_price
                print(f"   🎯 CALL: {exit_price:.5f} > {entry_price:.5f} ? {is_winning}")
            else:
                is_winning = exit_price < entry_price
                print(f"   🎯 PUT: {exit_price:.5f} < {entry_price:.5f} ? {is_winning}")
            
            result = 'WIN' if is_winning else 'LOSE'
            details = {
                'entry_price': float(entry_price),
                'exit_price': float(exit_price),
                'pips': float(pips_diff),
                'gale_level': 0,
                'reason': f'M1 vérifié - Diff: {price_diff:+.5f}'
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
            
            # Vérifier week-end
            if self._is_weekend(ts_utc):
                print(f"   🏖️ Week-end - Pas d'appel API pour Forex")
                return None
            
            # Plage M1: ±5 minutes pour être sûr
            start_dt = ts_utc - timedelta(minutes=5)
            end_dt = ts_utc + timedelta(minutes=5)
            
            params = {
                'symbol': pair,
                'interval': '1min',
                'outputsize': 10,
                'apikey': self.api_key,
                'format': 'JSON'
            }
            
            print(f"   🔍 API M1: {pair} {price_type} à {ts_utc.strftime('%H:%M')}")
            
            try:
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
                    # L'API renvoie les bougies les plus récentes en premier
                    for candle in data['values']:
                        try:
                            candle_time = datetime.fromisoformat(candle['datetime'].replace('Z', '+00:00'))
                        except:
                            candle_time = datetime.strptime(candle['datetime'], '%Y-%m-%d %H:%M:%S')
                        
                        if candle_time.tzinfo is None:
                            candle_time = candle_time.replace(tzinfo=timezone.utc)
                        
                        candle_time = self._round_to_m1_candle(candle_time)
                        
                        # Vérifier si c'est la bougie qu'on cherche
                        if candle_time == ts_utc:
                            try:
                                price = float(candle[price_type])
                                print(f"   💰 Prix {price_type}: {price}")
                                return price
                            except:
                                # Fallback au prix close
                                try:
                                    price = float(candle['close'])
                                    print(f"   💰 Prix close (fallback): {price}")
                                    return price
                                except:
                                    return None
                
                print(f"   ⚠️ Bougie M1 à {ts_utc.strftime('%H:%M')} non trouvée")
                return None
                
            except requests.exceptions.RequestException as e:
                print(f"   ⚠️ Erreur réseau API: {e}")
                return None
            
        except Exception as e:
            print(f"⚠️ Erreur _get_price_at_time: {e}")
            return None

    def _update_signal_result(self, signal_id, result, details):
        """Met à jour résultat dans DB"""
        try:
            reason = details.get('reason', '') if details else ''
            entry_price = details.get('entry_price')
            exit_price = details.get('exit_price')
            pips = details.get('pips')
            
            # D'abord, vérifier que le signal existe et n'a pas déjà un résultat
            with self.engine.connect() as conn:
                existing = conn.execute(
                    text("SELECT result FROM signals WHERE id = :id"),
                    {"id": signal_id}
                ).fetchone()
                
                if existing and existing[0] is not None:
                    print(f"⚠️ Signal #{signal_id} a déjà un résultat: {existing[0]}")
                    return
            
            # Mettre à jour avec toutes les informations
            query = text("""
                UPDATE signals
                SET result = :result, 
                    gale_level = 0, 
                    reason = :reason,
                    entry_price = :entry_price,
                    exit_price = :exit_price,
                    pips = :pips,
                    ts_exit = :ts_exit
                WHERE id = :id
            """)
            
            with self.engine.begin() as conn:
                conn.execute(query, {
                    'result': result,
                    'reason': reason,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pips': pips,
                    'ts_exit': datetime.now(timezone.utc).isoformat(),
                    'id': signal_id
                })
            
            print(f"💾 Résultat M1 sauvegardé: #{signal_id} = {result}")
            print(f"   📊 Entry: {entry_price}, Exit: {exit_price}, Pips: {pips}")
            
        except Exception as e:
            print(f"❌ Erreur _update_signal_result: {e}")
            import traceback
            traceback.print_exc()
    
    async def manual_verify_signal(self, signal_id, result, entry_price=None, exit_price=None):
        """Vérification manuelle d'un signal"""
        try:
            print(f"🔄 Vérification manuelle signal #{signal_id}: {result}")
            
            # Récupérer les infos du signal
            with self.engine.connect() as conn:
                signal = conn.execute(
                    text("SELECT pair, direction, ts_enter FROM signals WHERE id = :sid"),
                    {"sid": signal_id}
                ).fetchone()
            
            if not signal:
                print(f"❌ Signal #{signal_id} non trouvé")
                return False
            
            pair, direction, ts_enter = signal
            
            details = {
                'reason': f'Vérification manuelle - {result}',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pips': abs(exit_price - entry_price) * 10000 if entry_price and exit_price else 0,
                'gale_level': 0
            }
            
            self._update_signal_result(signal_id, result, details)
            print(f"✅ Signal #{signal_id} mis à jour manuellement: {result}")
            return True
            
        except Exception as e:
            print(f"❌ Erreur manual_verify_signal: {e}")
            return False
    
    def get_signal_status(self, signal_id):
        """Récupère le statut d'un signal"""
        try:
            with self.engine.connect() as conn:
                signal = conn.execute(
                    text("""
                        SELECT id, pair, direction, result, ts_enter, ts_exit, 
                               entry_price, exit_price, pips, reason
                        FROM signals
                        WHERE id = :sid
                    """),
                    {"sid": signal_id}
                ).fetchone()
            
            if not signal:
                return None
            
            return {
                'id': signal[0],
                'pair': signal[1],
                'direction': signal[2],
                'result': signal[3],
                'ts_enter': signal[4],
                'ts_exit': signal[5],
                'entry_price': signal[6],
                'exit_price': signal[7],
                'pips': signal[8],
                'reason': signal[9]
            }
            
        except Exception as e:
            print(f"❌ Erreur get_signal_status: {e}")
            return None
