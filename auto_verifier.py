"""
AUTO VERIFIER M1 - VERSION POCKET OPTION RÉELLE
Correction pour timing réel Pocket Option (trade la bougie en cours)
"""

import asyncio
from datetime import datetime, timedelta, timezone
from sqlalchemy import text
import requests
import json

class AutoResultVerifier:
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
                'XAG/USD': 'XAG-USDT',
            }
        }
        
        # RATE LIMITING
        self.api_calls_count = 0
        self.api_calls_reset_time = datetime.now()
        self.MAX_API_CALLS_PER_MINUTE = 6
        
        print("[VERIF-M1] ✅ AutoResultVerifier M1 initialisé")
        print("[VERIF-M1] 🎯 Mode: Trading M1 (1 minute)")
        print("[VERIF-M1] 🔥 Support OTC/CRYPTO activé")
        print("[VERIF-M1] ⏰ TIMING POCKET OPTION: Trade la bougie EN COURS")

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
        CORRECTION CRITIQUE : Calcule la BONNE bougie M1 pour Pocket Option
        
        LOGIQUE POCKET OPTION RÉELLE :
        - Signal à 19:49:00 → Trade la bougie 19:49:00-19:50:00
        - Signal à 19:49:30 → Trade la bougie 19:49:00-19:50:00 (bougie en cours)
        
        Sur Pocket Option, quand tu cliques "CALL" ou "PUT", tu trade la bougie qui est EN COURS,
        pas la suivante !
        
        Returns:
            (candle_start, candle_end) - LA BOUGIE EN COURS au moment du signal
        """
        # S'assurer que c'est en UTC
        if signal_time.tzinfo is None:
            signal_time = signal_time.replace(tzinfo=timezone.utc)
        
        # Sur Pocket Option : tu trade la bougie qui DÉBUTE à l'heure arrondie à la minute
        # Ex: 19:49:00 → bougie 19:49:00-19:50:00
        # Ex: 19:49:30 → bougie 19:49:00-19:50:00 (même bougie en cours)
        
        candle_start = self._round_to_m1(signal_time)
        candle_end = candle_start + timedelta(minutes=1)
        
        return candle_start, candle_end

    async def verify_single_signal(self, signal_id):
        """
        Vérifie UN signal M1 - TIMING POCKET OPTION CORRIGÉ
        """
        try:
            print(f"\n{'='*70}")
            print(f"[VERIF-M1] 🔍 Vérification signal #{signal_id}")
            print(f"{'='*70}")
            
            # Récupérer le signal avec payload_json
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
            
            # Détecter le mode OTC/CRYPTO depuis payload_json
            is_otc = False
            exchange = 'bybit'  # exchange par défaut pour OTC
            mode = 'Forex'  # mode par défaut
            
            if payload_json:
                try:
                    payload = json.loads(payload_json)
                    mode = payload.get('mode', 'Forex')
                    is_otc = (mode == 'OTC' or mode == 'CRYPTO' or mode == 'CRYPTO_OTC')
                    exchange = payload.get('exchange', 'bybit')
                    
                    print(f"[VERIF-M1] 🔥 Mode détecté: {mode}")
                    print(f"[VERIF-M1] 💱 Exchange: {exchange}")
                except Exception as e:
                    print(f"[VERIF-M1] ⚠️ Erreur lecture payload: {e}")
            
            kz_text = f" [{kill_zone}]" if kill_zone else ""
            print(f"[VERIF-M1] 📊 {pair} {direction}{kz_text}")
            print(f"[VERIF-M1] 💪 Confiance: {confidence:.1%}")
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
            
            # Vérifier week-end (sauf pour OTC)
            if not is_otc and self._is_weekend(signal_time):
                print(f"[VERIF-M1] 🏖️ Week-end - Marqué LOSE")
                self._update_signal_result(signal_id, 'LOSE', {
                    'entry_price': 0,
                    'exit_price': 0,
                    'pips': 0,
                    'gale_level': 0,
                    'reason': 'Marché fermé (week-end)',
                    'verification_method': 'AUTO_WEEKEND'
                })
                return 'LOSE'
            
            # CORRECTION CRITIQUE : Calculer la BOUGIE EN COURS POCKET OPTION
            trade_start, trade_end = self._calculate_correct_m1_candle(signal_time)
            
            print(f"[VERIF-M1] ⚡ LOGIQUE POCKET OPTION:")
            print(f"[VERIF-M1] 📊 Tu as tradé la bougie: {trade_start.strftime('%H:%M')} → {trade_end.strftime('%H:%M')} UTC")
            print(f"[VERIF-M1] 💡 Sur Pocket Option, tu trade la bougie EN COURS quand tu cliques!")
            
            # Calculer combien de temps il reste avant la fin de la bougie
            now_utc = datetime.now(timezone.utc)
            
            # Si la bougie n'est PAS ENCORE COMMENCÉE (futur)
            if now_utc < trade_start:
                time_until_start = (trade_start - now_utc).total_seconds()
                print(f"[VERIF-M1] ⏳ Bougie M1 pas encore commencée - commence dans {time_until_start:.0f}s")
                return None
            
            # Si la bougie est EN COURS
            if now_utc < trade_end:
                remaining = (trade_end - now_utc).total_seconds()
                print(f"[VERIF-M1] ⏳ Bougie M1 en cours - fin dans {remaining:.0f}s")
                print(f"[VERIF-M1] ⚠️ Attendre la fin de la bougie pour vérifier...")
                return None
            
            # Si la bougie est TERMINÉE
            print(f"[VERIF-M1] ✅ Bougie M1 terminée - vérification en cours...")
            
            # Récupérer les prix de CETTE bougie M1 spécifique
            result, details = await self._verify_m1_candle(
                signal_id, pair, direction, trade_start, is_otc, exchange
            )
            
            if result:
                details['verification_method'] = 'AUTO_M1_POCKET_' + ('OTC' if is_otc else 'FOREX')
                self._update_signal_result(signal_id, result, details)
                emoji = "✅" if result == 'WIN' else "❌"
                print(f"[VERIF-M1] {emoji} Résultat M1: {result}")
                
                if details and details.get('pips'):
                    print(f"[VERIF-M1] 📊 {details['pips']:.1f} pips")
                
                return result
            else:
                print(f"[VERIF-M1] ⚠️ Impossible de vérifier - réessai plus tard")
                return None
                
        except Exception as e:
            print(f"[VERIF-M1] ❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _verify_m1_candle(self, signal_id, pair, direction, candle_start, is_otc=False, exchange='bybit'):
        """
        Vérifie une bougie M1 spécifique avec support OTC
        """
        try:
            print(f"[VERIF-M1] 🔍 Récupération bougie M1 {candle_start.strftime('%H:%M')}...")
            
            # Récupérer le prix d'ouverture M1
            entry_price = await self._get_exact_m1_candle_price(
                pair, candle_start, 'open', is_otc, exchange
            )
            if entry_price is None:
                print(f"[VERIF-M1] ❌ Prix d'ouverture M1 non disponible")
                return None, None
            
            await asyncio.sleep(1)  # Délai court entre appels API
            
            # Récupérer le prix de fermeture M1
            exit_price = await self._get_exact_m1_candle_price(
                pair, candle_start, 'close', is_otc, exchange
            )
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
                'gale_level': 0,
                'mode': 'OTC' if is_otc else 'Forex',
                'exchange': exchange if is_otc else 'twelvedata',
                'reason': f"Bougie M1 {candle_start.strftime('%H:%M')}-{(candle_start + timedelta(minutes=1)).strftime('%H:%M')}"
            }
            
            return result, details
            
        except Exception as e:
            print(f"[VERIF-M1] ❌ Erreur _verify_m1_candle: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    async def _get_exact_m1_candle_price(self, pair, candle_start, price_type='close', is_otc=False, exchange='bybit'):
        """
        Récupère le prix d'UNE bougie M1 SPÉCIFIQUE avec support OTC
        """
        try:
            if is_otc:
                return await self._get_otc_candle_price(pair, candle_start, price_type, exchange)
            else:
                return await self._get_forex_candle_price(pair, candle_start, price_type)
                
        except Exception as e:
            print(f"[VERIF-M1] ❌ Erreur _get_exact_m1_candle_price: {e}")
            return None

    async def _get_forex_candle_price(self, pair, candle_start, price_type='close'):
        """
        Récupère le prix depuis TwelveData (Forex)
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
            
            print(f"[VERIF-M1] 🔍 Forex API: {pair} {price_type} à {candle_start.strftime('%H:%M')} UTC")
            
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
                print(f"[VERIF-M1] ❌ Aucune donnée Forex retournée")
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
                
                # Comparaison EXACTE pour M1
                time_diff = abs((candle_time_m1 - candle_start).total_seconds())
                
                if time_diff < 10:  # ✅ Tolérance réduite à 10s pour précision
                    try:
                        price = float(candle[price_type])
                        print(f"[VERIF-M1] ✅ Bougie Forex trouvée - {price_type}: {price:.5f}")
                        return price
                    except:
                        # Fallback sur close
                        try:
                            price = float(candle['close'])
                            print(f"[VERIF-M1] ⚠️ Fallback Forex close: {price:.5f}")
                            return price
                        except:
                            continue
            
            print(f"[VERIF-M1] ❌ Bougie Forex {candle_start.strftime('%H:%M')} NON trouvée")
            return None
            
        except Exception as e:
            print(f"[VERIF-M1] ❌ Erreur API Forex: {e}")
            return None

    async def _get_otc_candle_price(self, pair, candle_start, price_type='close', exchange='bybit'):
        """
        Récupère le prix depuis un exchange OTC (crypto)
        """
        try:
            await self._wait_if_rate_limited()
            
            # Convertir le symbole
            symbol = self._map_pair_to_symbol(pair, exchange)
            print(f"[VERIF-M1] 🔄 OTC: {pair} -> {symbol} sur {exchange}")
            
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
                    # Binance retourne: [timestamp, open, high, low, close, ...]
                    for candle in data:
                        candle_time = datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc)
                        time_diff = abs((candle_time - candle_start).total_seconds())
                        
                        if time_diff < 10:  # Tolérance 10 secondes
                            if price_type == 'open':
                                price = float(candle[1])
                            else:
                                price = float(candle[4])
                            
                            print(f"[VERIF-M1] ✅ Binance: {price_type}={price:.6f}")
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
                        # Bybit v5 retourne: [timestamp, open, high, low, close, volume, turnover]
                        candle_time = datetime.fromtimestamp(int(candle[0]) / 1000, tz=timezone.utc)
                        time_diff = abs((candle_time - candle_start).total_seconds())
                        
                        if time_diff < 10:
                            if price_type == 'open':
                                price = float(candle[1])
                            else:
                                price = float(candle[4])
                            
                            print(f"[VERIF-M1] ✅ Bybit: {price_type}={price:.6f}")
                            return price
            
            print(f"[VERIF-M1] ❌ Bougie OTC {candle_start.strftime('%H:%M')} NON trouvée sur {exchange}")
            return None
            
        except Exception as e:
            print(f"[VERIF-M1] ❌ Erreur API OTC ({exchange}): {e}")
            return None

    def _update_signal_result(self, signal_id, result, details):
        """Met à jour le résultat dans la DB"""
        try:
            gale_level = details.get('gale_level', 0) if details else 0
            reason = details.get('reason', '') if details else ''
            verification_method = details.get('verification_method', 'AUTO_M1') if details else 'AUTO_M1'
            entry_price = details.get('entry_price', 0) if details else 0
            exit_price = details.get('exit_price', 0) if details else 0
            pips = details.get('pips', 0) if details else 0
            
            # Vérifier d'abord si la colonne verification_method existe
            with self.engine.connect() as conn:
                # Vérifier la structure de la table
                table_info = conn.execute(
                    text("PRAGMA table_info(signals)")
                ).fetchall()
                
                columns = [row[1] for row in table_info]
                
                # Construire la requête dynamiquement
                set_clauses = []
                values = {'id': signal_id}
                
                # Colonnes de base
                set_clauses.append("result = :result")
                values['result'] = result
                
                set_clauses.append("ts_exit = :ts_exit")
                values['ts_exit'] = datetime.now(timezone.utc).isoformat()
                
                # Ajouter les colonnes optionnelles si elles existent
                if 'gale_level' in columns:
                    set_clauses.append("gale_level = :gale_level")
                    values['gale_level'] = gale_level
                
                if 'reason' in columns:
                    set_clauses.append("reason = :reason")
                    values['reason'] = reason
                
                if 'verification_method' in columns:
                    set_clauses.append("verification_method = :verification_method")
                    values['verification_method'] = verification_method
                
                if 'entry_price' in columns:
                    set_clauses.append("entry_price = :entry_price")
                    values['entry_price'] = entry_price
                
                if 'exit_price' in columns:
                    set_clauses.append("exit_price = :exit_price")
                    values['exit_price'] = exit_price
                
                if 'pips' in columns:
                    set_clauses.append("pips = :pips")
                    values['pips'] = pips
                
                # Exécuter la mise à jour
                query = text(f"""
                    UPDATE signals
                    SET {', '.join(set_clauses)}
                    WHERE id = :id
                """)
                
                conn.execute(query, values)
                conn.commit()
            
            print(f"[VERIF-M1] 💾 Résultat M1 sauvegardé: {result}")
            
        except Exception as e:
            print(f"[VERIF-M1] ❌ Erreur sauvegarde: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback simple sans les colonnes optionnelles
            try:
                query = text("UPDATE signals SET result = :result WHERE id = :id")
                with self.engine.begin() as conn:
                    conn.execute(query, {'result': result, 'id': signal_id})
                print(f"[VERIF-M1] 💾 Sauvegardé (version simple)")
            except Exception as e2:
                print(f"[VERIF-M1] ❌ Échec total: {e2}")

    async def verify_pending_signals(self):
        """Vérifie tous les signaux M1 en attente avec support OTC"""
        try:
            now_utc = datetime.now(timezone.utc)
            print(f"\n{'='*70}")
            print(f"[VERIF-M1] 🔍 VÉRIFICATION AUTO M1 - {now_utc.strftime('%H:%M:%S')} UTC")
            print(f"{'='*70}")

            # Ne vérifier que les signaux dont la bougie est terminée (au moins 1 minute après le début)
            with self.engine.connect() as conn:
                # Requête optimisée pour ne prendre que les signaux vérifiables
                pending = conn.execute(
                    text("""
                        SELECT id, pair, direction, ts_enter, confidence, kill_zone, payload_json
                        FROM signals
                        WHERE result IS NULL
                          AND datetime(ts_enter, '+2 minutes') <= datetime('now')
                        ORDER BY ts_enter ASC
                        LIMIT 10
                    """)
                ).fetchall()
            
            print(f"[VERIF-M1] 📊 Signaux M1 vérifiables: {len(pending)}")
            
            if not pending:
                print(f"[VERIF-M1] ✅ Aucun signal M1 à vérifier pour le moment")
                return
            
            # Limite : 2 signaux max par cycle (éviter rate limiting)
            MAX_PER_CYCLE = 2
            
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
                    
                    await asyncio.sleep(1)
                    
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
