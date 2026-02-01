"""
AUTO VERIFIER M1 - VERSION POCKET OPTION RÉELLE SANS DONNÉES FICTIVES
"""

import asyncio
from datetime import datetime, timedelta, timezone
from sqlalchemy import text
import requests
import json
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoResultVerifier:
    def __init__(self, engine, twelvedata_api_key, bot=None):
        self.engine = engine
        self.api_key = twelvedata_api_key
        self.base_url = 'https://api.twelvedata.com/time_series'
        self.bot = bot
        self.admin_chat_ids = []
        
        self._session = requests.Session()
        
        # Paramètres pour M1
        self.default_timeframe = 1
        self.default_max_gales = 0
        
        # Endpoints pour OTC (crypto)
        self.crypto_endpoints = {
            'binance': 'https://api.binance.com/api/v3/klines',
            'bybit': 'https://api.bybit.com/v5/market/kline',
            'kucoin': 'https://api.kucoin.com/api/v1/market/candles',
        }
        
        # Mapping des paires OTC
        self.otc_symbol_mapping = {
            'binance': {
                'BTC/USD': 'BTCUSDT',
                'ETH/USD': 'ETHUSDT',
                'TRX/USD': 'TRXUSDT',
                'LTC/USD': 'LTCUSDT',
                'EUR/USD': 'EURUSDT',
                'GBP/USD': 'GBPUSDT',
                'USD/JPY': 'JPYUSDT',
                'AUD/USD': 'AUDUSDT',
                'AUD/CAD': 'AUDCAD',
                'EUR/GBP': 'EURGBP',
                'XAU/USD': 'XAUUSDT',
                'XAG/USD': 'XAGUSDT',
            },
            'bybit': {
                'BTC/USD': 'BTCUSDT',
                'ETH/USD': 'ETHUSDT',
                'TRX/USD': 'TRXUSDT',
                'LTC/USD': 'LTCUSDT',
                'AUD/CAD': 'AUDCAD',
                'EUR/GBP': 'EURGBP',
                'XAU/USD': 'XAUUSDT',
                'XAG/USD': 'XAGUSDT',
            },
            'kucoin': {
                'BTC/USD': 'BTC-USDT',
                'ETH/USD': 'ETH-USDT',
                'TRX/USD': 'TRX-USDT',
                'LTC/USD': 'LTC-USDT',
                'AUD/CAD': 'AUD-CAD',
                'EUR/GBP': 'EUR-GBP',
                'XAU/USD': 'XAU-USDT',
                'XAG/USD': 'XAGUSDT',
            }
        }
        
        # RATE LIMITING
        self.api_calls_count = 0
        self.api_calls_reset_time = datetime.now()
        self.MAX_API_CALLS_PER_MINUTE = 6
        
        print("[VERIF-M1] ✅ AutoResultVerifier M1 initialisé")
        print("[VERIF-M1] 🎯 Mode: Trading M1 (1 minute)")
        print("[VERIF-M1] 🔥 Support OTC/CRYPTO activé")
        print("[VERIF-M1] ⚠️ DONNÉES FICTIVES INTERDITES - Seules les données réelles sont acceptées")

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

    def _map_pair_to_symbol(self, pair: str, exchange: str = 'bybit') -> str:
        """Convertit une paire format TradingView en symbole d'API OTC"""
        return self.otc_symbol_mapping.get(exchange, {}).get(pair, pair.replace('/', ''))

    def _calculate_correct_m1_candle(self, signal_time):
        """
        Calcule la BONNE bougie M1 pour Pocket Option
        """
        if signal_time.tzinfo is None:
            signal_time = signal_time.replace(tzinfo=timezone.utc)
        
        # Sur Pocket Option : tu trade la bougie qui DÉBUTE à l'heure arrondie à la minute
        candle_start = self._round_to_m1(signal_time)
        candle_end = candle_start + timedelta(minutes=1)
        
        return candle_start, candle_end

    async def verify_single_signal(self, signal_id):
        """
        Vérifie UN signal M1 - SANS DONNÉES FICTIVES
        """
        try:
            print(f"\n{'='*70}")
            print(f"[VERIF-M1] 🔍 Vérification signal #{signal_id}")
            print(f"{'='*70}")
            
            # Récupérer le signal
            with self.engine.connect() as conn:
                signal = conn.execute(
                    text("""
                        SELECT id, pair, direction, ts_enter, confidence, kill_zone, payload_json
                        FROM signals
                        WHERE id = :sid AND result IS NULL
                    """),
                    {"sid": signal_id}
                ).fetchone()
            
            if not signal:
                print(f"[VERIF-M1] ⚠️ Signal #{signal_id} déjà vérifié ou inexistant")
                return None
            
            signal_id, pair, direction, ts_enter, confidence, kill_zone, payload_json = signal
            
            # Détecter le mode OTC/CRYPTO
            is_otc = False
            exchange = 'bybit'
            mode = 'Forex'
            
            if payload_json:
                try:
                    payload = json.loads(payload_json)
                    mode = payload.get('mode', 'Forex')
                    is_otc = (mode == 'OTC' or mode == 'CRYPTO' or mode == 'CRYPTO_OTC')
                    exchange = payload.get('exchange', 'bybit')
                except Exception as e:
                    print(f"[VERIF-M1] ⚠️ Erreur lecture payload: {e}")
            
            kz_text = f" [{kill_zone}]" if kill_zone else ""
            print(f"[VERIF-M1] 📊 {pair} {direction}{kz_text}")
            print(f"[VERIF-M1] 🎮 Mode: {mode} ({'OTC' if is_otc else 'Forex'})")
            
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
            
            print(f"[VERIF-M1] 🕐 Signal envoyé à: {signal_time.strftime('%H:%M:%S')} UTC")
            
            # Vérifier week-end
            if not is_otc and self._is_weekend(signal_time):
                print(f"[VERIF-M1] ⏳ Week-end - Laisser en attente (pas de données)")
                return None  # Pas de vérification pendant le week-end
            
            # Calculer la bougie
            trade_start, trade_end = self._calculate_correct_m1_candle(signal_time)
            
            print(f"[VERIF-M1] 📊 Bougie tradée: {trade_start.strftime('%H:%M')} → {trade_end.strftime('%H:%M')} UTC")
            
            # Vérifier si la bougie est terminée
            now_utc = datetime.now(timezone.utc)
            
            if now_utc < trade_start:
                time_until_start = (trade_start - now_utc).total_seconds()
                print(f"[VERIF-M1] ⏳ Bougie M1 pas encore commencée - commence dans {time_until_start:.0f}s")
                return None
            
            if now_utc < trade_end:
                remaining = (trade_end - now_utc).total_seconds()
                print(f"[VERIF-M1] ⏳ Bougie M1 en cours - fin dans {remaining:.0f}s")
                return None
            
            print(f"[VERIF-M1] ✅ Bougie M1 terminée - vérification en cours...")
            
            # Récupérer les prix RÉELS
            result, details = await self._verify_m1_candle_real_prices(
                signal_id, pair, direction, trade_start, is_otc, exchange
            )
            
            if result and details:
                details['verification_method'] = 'AUTO_M1_POCKET_' + ('OTC' if is_otc else 'FOREX')
                self._update_signal_result_real(signal_id, result, details)
                emoji = "✅" if result == 'WIN' else "❌"
                print(f"[VERIF-M1] {emoji} Résultat M1: {result}")
                
                if details.get('pips'):
                    print(f"[VERIF-M1] 📊 {details['pips']:.1f} pips")
                
                return result
            else:
                print(f"[VERIF-M1] ⚠️ Impossible de vérifier - données réelles manquantes")
                return None
                
        except Exception as e:
            print(f"[VERIF-M1] ❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _verify_m1_candle_real_prices(self, signal_id, pair, direction, candle_start, is_otc=False, exchange='bybit'):
        """
        Vérifie une bougie M1 avec prix RÉELS uniquement
        Retourne None si les prix ne sont pas disponibles
        """
        try:
            print(f"[VERIF-M1] 🔍 Récupération PRIX RÉELS bougie M1 {candle_start.strftime('%H:%M')}...")
            
            # Récupérer le prix d'ouverture M1
            entry_price = await self._get_exact_m1_candle_price_real(
                pair, candle_start, 'open', is_otc, exchange
            )
            
            await asyncio.sleep(1)
            
            # Récupérer le prix de fermeture M1
            exit_price = await self._get_exact_m1_candle_price_real(
                pair, candle_start, 'close', is_otc, exchange
            )
            
            # VÉRIFICATION CRITIQUE : Si un des prix est None, on ARRÊTE
            if entry_price is None:
                print(f"[VERIF-M1] ❌ Prix d'ouverture RÉEL non disponible - ABANDON")
                return None, None
            
            if exit_price is None:
                print(f"[VERIF-M1] ❌ Prix de fermeture RÉEL non disponible - ABANDON")
                return None, None
            
            # Calculer le résultat avec les prix RÉELS
            price_diff = exit_price - entry_price
            pips_diff = abs(price_diff) * 10000
            
            print(f"[VERIF-M1] 💰 Prix M1 entrée RÉEL : {entry_price:.5f}")
            print(f"[VERIF-M1] 💰 Prix M1 sortie RÉEL : {exit_price:.5f}")
            print(f"[VERIF-M1] 📊 Différence M1 RÉELLE : {price_diff:+.5f} ({pips_diff:.1f} pips)")
            
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
                'gale_level': 0,
                'mode': 'OTC' if is_otc else 'Forex',
                'exchange': exchange if is_otc else 'twelvedata',
                'reason': f"Bougie M1 RÉELLE {candle_start.strftime('%H:%M')}"
            }
            
            return result, details
            
        except Exception as e:
            print(f"[VERIF-M1] ❌ Erreur _verify_m1_candle_real_prices: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    async def _get_exact_m1_candle_price_real(self, pair, candle_start, price_type='close', is_otc=False, exchange='bybit'):
        """
        Récupère le prix d'UNE bougie M1 SPÉCIFIQUE
        Retourne None si le prix n'est pas disponible
        """
        try:
            if is_otc:
                return await self._get_otc_candle_price_real(pair, candle_start, price_type, exchange)
            else:
                return await self._get_forex_candle_price_real(pair, candle_start, price_type)
                
        except Exception as e:
            print(f"[VERIF-M1] ❌ Erreur _get_exact_m1_candle_price_real: {e}")
            return None

    async def _get_forex_candle_price_real(self, pair, candle_start, price_type='close'):
        """
        Récupère le prix RÉEL depuis TwelveData (Forex)
        Retourne None si non disponible
        """
        try:
            await self._wait_if_rate_limited()
            
            # Plage plus large pour être sûr de trouver la bougie
            start_dt = candle_start - timedelta(minutes=3)
            end_dt = candle_start + timedelta(minutes=3)
            
            params = {
                'symbol': pair,
                'interval': '1min',
                'outputsize': 10,
                'apikey': self.api_key,
                'format': 'JSON',
                'start_date': start_dt.strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': end_dt.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            print(f"[VERIF-M1] 🔍 Forex API RÉELLE: {pair} {price_type} à {candle_start.strftime('%H:%M')} UTC")
            
            resp = self._session.get(self.base_url, params=params, timeout=15)
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
                print(f"[VERIF-M1] ❌ Aucune donnée Forex RÉELLE retournée")
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
                
                # Arrondir à M1
                candle_time_m1 = self._round_to_m1(candle_time)
                
                # Comparaison EXACTE pour M1
                time_diff = abs((candle_time_m1 - candle_start).total_seconds())
                
                if time_diff < 10:  # Tolérance réduite à 10s
                    try:
                        price = float(candle[price_type])
                        print(f"[VERIF-M1] ✅ Prix Forex RÉEL trouvé - {price_type}: {price:.5f}")
                        return price
                    except:
                        # Fallback sur close
                        try:
                            price = float(candle['close'])
                            print(f"[VERIF-M1] ⚠️ Fallback Forex close: {price:.5f}")
                            return price
                        except:
                            continue
            
            print(f"[VERIF-M1] ❌ Prix Forex RÉEL {candle_start.strftime('%H:%M')} NON trouvé")
            return None
            
        except Exception as e:
            print(f"[VERIF-M1] ❌ Erreur API Forex RÉELLE: {e}")
            return None

    async def _get_otc_candle_price_real(self, pair, candle_start, price_type='close', exchange='bybit'):
        """
        Récupère le prix RÉEL depuis un exchange OTC
        Retourne None si non disponible
        """
        try:
            await self._wait_if_rate_limited()
            
            # Convertir le symbole
            symbol = self._map_pair_to_symbol(pair, exchange)
            print(f"[VERIF-M1] 🔄 OTC RÉEL: {pair} -> {symbol} sur {exchange}")
            
            # Timestamp en millisecondes
            start_ms = int(candle_start.timestamp() * 1000)
            
            if exchange == 'binance':
                url = self.crypto_endpoints['binance']
                params = {
                    'symbol': symbol,
                    'interval': '1m',
                    'startTime': start_ms,
                    'limit': 3
                }
                
                resp = self._session.get(url, params=params, timeout=10)
                self._increment_api_call()
                
                if resp.status_code != 200:
                    print(f"[VERIF-M1] ❌ Binance API error: {resp.status_code}")
                    return None
                
                data = resp.json()
                
                if isinstance(data, list) and len(data) > 0:
                    for candle in data:
                        candle_time = datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc)
                        time_diff = abs((candle_time - candle_start).total_seconds())
                        
                        if time_diff < 10:
                            if price_type == 'open':
                                price = float(candle[1])
                            else:
                                price = float(candle[4])
                            
                            print(f"[VERIF-M1] ✅ Prix Binance RÉEL: {price_type}={price:.6f}")
                            return price
            
            elif exchange == 'bybit':
                url = self.crypto_endpoints['bybit']
                params = {
                    'category': 'spot',
                    'symbol': symbol,
                    'interval': '1',
                    'start': start_ms,
                    'limit': 3
                }
                
                resp = self._session.get(url, params=params, timeout=10)
                self._increment_api_call()
                
                if resp.status_code != 200:
                    print(f"[VERIF-M1] ❌ Bybit API error: {resp.status_code}")
                    return None
                
                data = resp.json()
                
                if data.get('retCode') == 0 and data.get('result', {}).get('list'):
                    candles = data['result']['list']
                    for candle in candles:
                        candle_time = datetime.fromtimestamp(int(candle[0]) / 1000, tz=timezone.utc)
                        time_diff = abs((candle_time - candle_start).total_seconds())
                        
                        if time_diff < 10:
                            if price_type == 'open':
                                price = float(candle[1])
                            else:
                                price = float(candle[4])
                            
                            print(f"[VERIF-M1] ✅ Prix Bybit RÉEL: {price_type}={price:.6f}")
                            return price
            
            print(f"[VERIF-M1] ❌ Prix OTC RÉEL {candle_start.strftime('%H:%M')} NON trouvé sur {exchange}")
            return None
            
        except Exception as e:
            print(f"[VERIF-M1] ❌ Erreur API OTC RÉELLE ({exchange}): {e}")
            return None

    def _update_signal_result_real(self, signal_id, result, details):
        """
        Met à jour le résultat avec les prix RÉELS uniquement
        """
        try:
            gale_level = details.get('gale_level', 0)
            reason = details.get('reason', '')
            verification_method = details.get('verification_method', 'AUTO_M1_REAL')
            entry_price = details.get('entry_price', 0)
            exit_price = details.get('exit_price', 0)
            pips = details.get('pips', 0)
            
            print(f"[VERIF-M1] 💾 Sauvegarde avec prix RÉELS:")
            print(f"[VERIF-M1]   • entry_price: {entry_price:.5f}")
            print(f"[VERIF-M1]   • exit_price: {exit_price:.5f}")
            print(f"[VERIF-M1]   • pips: {pips:.1f}")
            
            # VÉRIFICATION CRITIQUE : Ne pas sauvegarder si les prix sont 0
            if entry_price == 0 or exit_price == 0:
                print(f"[VERIF-M1] ⚠️ PRIX À 0 DÉTECTÉS - ABANDON DE LA SAUVEGARDE")
                return
            
            with self.engine.begin() as conn:
                # Mise à jour avec TOUS les champs nécessaires
                conn.execute(
                    text("""
                        UPDATE signals 
                        SET result = :result,
                            ts_exit = :ts_exit,
                            entry_price = :entry_price,
                            exit_price = :exit_price,
                            pips = :pips,
                            gale_level = :gale_level,
                            reason = :reason,
                            verification_method = :verification_method
                        WHERE id = :id
                    """),
                    {
                        'result': result,
                        'ts_exit': datetime.now(timezone.utc).isoformat(),
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pips': pips,
                        'gale_level': gale_level,
                        'reason': reason,
                        'verification_method': verification_method,
                        'id': signal_id
                    }
                )
            
            print(f"[VERIF-M1] 💾 Résultat RÉEL sauvegardé: {result}")
            
        except Exception as e:
            print(f"[VERIF-M1] ❌ Erreur sauvegarde RÉELLE: {e}")
            import traceback
            traceback.print_exc()

    async def verify_pending_signals_real_only(self):
        """
        Vérifie tous les signaux M1 en attente - DONNÉES RÉELLES UNIQUEMENT
        """
        try:
            now_utc = datetime.now(timezone.utc)
            print(f"\n{'='*70}")
            print(f"[VERIF-M1] 🔍 VÉRIFICATION AUTO M1 - DONNÉES RÉELLES UNIQUEMENT")
            print(f"[VERIF-M1] 🕐 {now_utc.strftime('%H:%M:%S')} UTC")
            print(f"{'='*70}")

            # Récupérer les signaux vérifiables (au moins 2 minutes après l'entrée)
            with self.engine.connect() as conn:
                pending = conn.execute(
                    text("""
                        SELECT id, pair, direction, ts_enter, confidence, kill_zone, payload_json
                        FROM signals
                        WHERE result IS NULL
                          AND datetime(ts_enter, '+2 minutes') <= datetime('now')
                        ORDER BY ts_enter ASC
                        LIMIT 5
                    """)
                ).fetchall()
            
            print(f"[VERIF-M1] 📊 Signaux M1 vérifiables: {len(pending)}")
            
            if not pending:
                print(f"[VERIF-M1] ✅ Aucun signal M1 à vérifier")
                return
            
            verified = 0
            waiting = 0
            no_data = 0
            
            for signal_row in pending:
                try:
                    signal_id = signal_row[0]
                    
                    result = await self.verify_single_signal(signal_id)
                    
                    if result == 'WIN' or result == 'LOSE':
                        verified += 1
                    elif result is None:
                        waiting += 1
                        no_data += 1
                    
                    await asyncio.sleep(2)  # Délai pour éviter rate limiting
                    
                except Exception as e:
                    print(f"[VERIF-M1] ❌ Erreur signal #{signal_row[0]}: {e}")
            
            print(f"\n{'='*70}")
            print(f"[VERIF-M1] 📈 RÉSUMÉ M1 RÉEL:")
            print(f"[VERIF-M1]   • Vérifiés: {verified}")
            print(f"[VERIF-M1]   • En attente (données manquantes): {waiting}")
            print(f"[VERIF-M1]   • Sans données: {no_data}")
            print(f"[VERIF-M1]   • Total traités: {len(pending)}")
            print(f"[VERIF-M1] ⚠️ Les signaux sans données RÉELLES restent en attente")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"[VERIF-M1] ❌ Erreur globale vérification réelle: {e}")
            import traceback
            traceback.print_exc()

    async def repair_real_missing_prices(self, limit: int = 20):
        """
        Répare UNIQUEMENT les signaux avec prix RÉELS manquants
        """
        try:
            print(f"\n{'='*70}")
            print(f"[REPAIR REAL] 🔧 Réparation prix RÉELS manquants")
            print(f"{'='*70}")
            
            with self.engine.connect() as conn:
                # Trouver les signaux vérifiés mais avec prix à 0 ou NULL
                signals_to_repair = conn.execute(
                    text("""
                        SELECT id, pair, direction, ts_enter, result, payload_json
                        FROM signals
                        WHERE result IN ('WIN', 'LOSE')
                          AND (entry_price IS NULL OR entry_price = 0 
                               OR exit_price IS NULL OR exit_price = 0)
                        ORDER BY id DESC
                        LIMIT :limit
                    """),
                    {"limit": limit}
                ).fetchall()
            
            print(f"[REPAIR REAL] 📊 Signaux à réparer: {len(signals_to_repair)}")
            
            if not signals_to_repair:
                print(f"[REPAIR REAL] ✅ Tous les signaux ont déjà des prix RÉELS")
                return
            
            repaired = 0
            failed = 0
            skipped = 0
            
            for signal in signals_to_repair:
                signal_id, pair, direction, ts_enter, result, payload_json = signal
                
                print(f"\n[REPAIR REAL] 🔧 Signal #{signal_id} ({pair}) - {result}...")
                
                # Vérifier si le signal a plus de 24h (trop vieux)
                if isinstance(ts_enter, str):
                    ts_clean = ts_enter.replace('Z', '').replace('+00:00', '').split('.')[0]
                    try:
                        signal_time = datetime.fromisoformat(ts_clean)
                    except:
                        signal_time = datetime.strptime(ts_clean, '%Y-%m-%d %H:%M:%S')
                else:
                    signal_time = ts_enter
                
                age_hours = (datetime.now(timezone.utc) - signal_time).total_seconds() / 3600
                
                if age_hours > 24:
                    print(f"[REPAIR REAL] ⚠️ Signal trop vieux ({age_hours:.1f}h) - SKIP")
                    skipped += 1
                    continue
                
                try:
                    # Déterminer le mode
                    is_otc = False
                    exchange = 'bybit'
                    
                    if payload_json:
                        try:
                            payload = json.loads(payload_json)
                            mode = payload.get('mode', 'Forex')
                            is_otc = (mode == 'OTC' or mode == 'CRYPTO' or mode == 'CRYPTO_OTC')
                            exchange = payload.get('exchange', 'bybit')
                        except:
                            pass
                    
                    if signal_time.tzinfo is None:
                        signal_time = signal_time.replace(tzinfo=timezone.utc)
                    
                    # Calculer la bougie M1
                    candle_start, _ = self._calculate_correct_m1_candle(signal_time)
                    
                    # Récupérer les prix RÉELS
                    entry_price = await self._get_exact_m1_candle_price_real(
                        pair, candle_start, 'open', is_otc, exchange
                    )
                    
                    await asyncio.sleep(1)
                    
                    exit_price = await self._get_exact_m1_candle_price_real(
                        pair, candle_start, 'close', is_otc, exchange
                    )
                    
                    # VÉRIFICATION : Si un prix est manquant, on ABANDONNE
                    if entry_price is None:
                        print(f"[REPAIR REAL] ❌ Prix entrée RÉEL non disponible pour #{signal_id}")
                        failed += 1
                        continue
                    
                    if exit_price is None:
                        print(f"[REPAIR REAL] ❌ Prix sortie RÉEL non disponible pour #{signal_id}")
                        failed += 1
                        continue
                    
                    # VÉRIFICATION : Ne pas accepter des prix à 0
                    if entry_price == 0 or exit_price == 0:
                        print(f"[REPAIR REAL] ⚠️ Prix à 0 détectés - ABANDON")
                        failed += 1
                        continue
                    
                    # Calculer les pips
                    price_diff = exit_price - entry_price
                    pips_diff = abs(price_diff) * 10000
                    
                    # Mettre à jour avec les prix RÉELS
                    with self.engine.begin() as conn:
                        conn.execute(
                            text("""
                                UPDATE signals
                                SET entry_price = :entry_price,
                                    exit_price = :exit_price,
                                    pips = :pips
                                WHERE id = :signal_id
                            """),
                            {
                                "entry_price": entry_price,
                                "exit_price": exit_price,
                                "pips": pips_diff,
                                "signal_id": signal_id
                            }
                        )
                    
                    print(f"[REPAIR REAL] ✅ #{signal_id} réparé avec prix RÉELS:")
                    print(f"[REPAIR REAL]   • Entrée: {entry_price:.5f}")
                    print(f"[REPAIR REAL]   • Sortie: {exit_price:.5f}")
                    print(f"[REPAIR REAL]   • Pips: {pips_diff:.1f}")
                    repaired += 1
                    
                    await asyncio.sleep(3)  # Délai important pour éviter rate limiting
                    
                except Exception as e:
                    print(f"[REPAIR REAL] ❌ Erreur réparation #{signal_id}: {e}")
                    failed += 1
            
            print(f"\n{'='*70}")
            print(f"[REPAIR REAL] 📈 RÉPARATION TERMINÉE:")
            print(f"[REPAIR REAL]   • Réparés avec données RÉELLES: {repaired}")
            print(f"[REPAIR REAL]   • Échecs (données manquantes): {failed}")
            print(f"[REPAIR REAL]   • Skippés (trop vieux): {skipped}")
            print(f"[REPAIR REAL]   • Total: {len(signals_to_repair)}")
            print(f"{'='*70}")
            
        except Exception as e:
            print(f"[REPAIR REAL] ❌ Erreur globale réparation: {e}")
            import traceback
            traceback.print_exc()
