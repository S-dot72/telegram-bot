"""
AUTO VERIFIER M1 - VERSION POCKET OPTION RÉELLE SANS DONNÉES FICTIVES
Utilise OTCDataProvider pour la cohérence des données
"""

import asyncio
from datetime import datetime, timedelta, timezone
from sqlalchemy import text
import requests
import json
import logging
import pandas as pd

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoResultVerifier:
    def __init__(self, engine, twelvedata_api_key, otc_provider=None, bot=None):
        """
        Initialise le vérificateur avec otc_provider pour la cohérence des données OTC
        
        Args:
            engine: Connexion à la base de données
            twelvedata_api_key: Clé API pour Forex
            otc_provider: Instance de OTCDataProvider pour les données Crypto OTC
            bot: Instance du bot Telegram (optionnel)
        """
        self.engine = engine
        self.api_key = twelvedata_api_key
        self.otc_provider = otc_provider  # <-- NOUVEAU: utiliser le même provider que le bot
        self.base_url = 'https://api.twelvedata.com/time_series'
        self.bot = bot
        self.admin_chat_ids = []
        
        self._session = requests.Session()
        
        # Paramètres pour M1
        self.default_timeframe = 1
        self.default_max_gales = 0
        
        # RATE LIMITING
        self.api_calls_count = 0
        self.api_calls_reset_time = datetime.now()
        self.MAX_API_CALLS_PER_MINUTE = 8
        
        print("[VERIF-M1] ✅ AutoResultVerifier M1 initialisé")
        print("[VERIF-M1] 🎯 Mode: Trading M1 (1 minute)")
        print("[VERIF-M1] 🔥 Support OTC/CRYPTO activé")
        print(f"[VERIF-M1] 🔧 OTC Provider: {'✅ Disponible' if otc_provider else '❌ Non disponible'}")
        print("[VERIF-M1] ⚠️ DONNÉES FICTIVES INTERDITES - Seules les données réelles sont acceptées")
        print("[VERIF-M1] 📊 Version: 3.0 - Utilise OTCDataProvider pour cohérence")

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
        
        # Utiliser otc_provider pour déterminer si c'est le week-end
        if self.otc_provider:
            return self.otc_provider.is_weekend()
        
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

    def _parse_datetime_robust(self, dt_str):
        """Parse robuste des dates de différentes APIs"""
        if not dt_str:
            return None
            
        dt_str = str(dt_str).replace('Z', '+00:00')
        
        formats = [
            '%Y-%m-%d %H:%M:%S%z',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%dT%H:%M:%S',
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(dt_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except:
                continue
        
        print(f"[VERIF-M1] ⚠️ Format de date non reconnu: {dt_str}")
        return None

    async def verify_single_signal(self, signal_id):
        """
        Vérifie UN signal M1 - SANS DONNÉES FICTIVES - VERSION CORRIGÉE
        Utilise otc_provider pour les données OTC
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
            mode = 'Forex'
            
            if payload_json:
                try:
                    payload = json.loads(payload_json)
                    mode = payload.get('mode', 'Forex')
                    is_otc = (mode == 'OTC' or mode == 'CRYPTO' or mode == 'CRYPTO_OTC')
                except Exception as e:
                    print(f"[VERIF-M1] ⚠️ Erreur lecture payload: {e}")
            
            kz_text = f" [{kill_zone}]" if kill_zone else ""
            print(f"[VERIF-M1] 📊 {pair} {direction}{kz_text}")
            print(f"[VERIF-M1] 🎮 Mode: {mode} ({'OTC' if is_otc else 'Forex'})")
            
            # Parser timestamp
            signal_time = self._parse_datetime_robust(ts_enter)
            if signal_time is None:
                print(f"[VERIF-M1] ❌ Impossible de parser ts_enter: {ts_enter}")
                return None
            
            if signal_time.tzinfo is None:
                signal_time = signal_time.replace(tzinfo=timezone.utc)
            
            print(f"[VERIF-M1] 🕐 Signal envoyé à: {signal_time.strftime('%H:%M:%S')} UTC")
            
            # Calculer la bougie
            trade_start, trade_end = self._calculate_correct_m1_candle(signal_time)
            
            print(f"[VERIF-M1] 📊 Bougie tradée: {trade_start.strftime('%H:%M')} → {trade_end.strftime('%H:%M')} UTC")
            
            # Vérifier si la bougie est terminée ET si les données sont disponibles
            now_utc = datetime.now(timezone.utc)
            
            # Les données ne sont disponibles que 2-3 minutes APRÈS la fin de la bougie
            data_available_time = trade_end + timedelta(minutes=3)
            
            if now_utc < data_available_time:
                wait_seconds = (data_available_time - now_utc).total_seconds()
                if wait_seconds > 0:
                    print(f"[VERIF-M1] ⏳ Données historiques disponibles dans {wait_seconds:.0f}s")
                    
                    # Si l'attente est courte (< 5 min), on attend
                    if wait_seconds < 300:
                        print(f"[VERIF-M1] ⏳ Attente de {wait_seconds:.0f}s pour données...")
                        await asyncio.sleep(wait_seconds)
                    else:
                        print(f"[VERIF-M1] ⏳ Trop long à attendre, on retente plus tard")
                        return None
            
            print(f"[VERIF-M1] ✅ Bougie M1 terminée - vérification en cours...")
            
            # Récupérer les prix RÉELS - UTILISE otc_provider pour OTC
            result, details = await self._verify_m1_candle_real_prices(
                signal_id, pair, direction, trade_start, is_otc
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

    async def verify_single_signal_with_retry(self, signal_id, max_retries=3):
        """Version avec retry - garde la compatibilité"""
        for attempt in range(max_retries):
            try:
                print(f"[VERIF-M1] 🔄 Tentative {attempt+1}/{max_retries} pour signal #{signal_id}")
                
                result = await self.verify_single_signal(signal_id)
                if result is not None:
                    return result
                
                # Si None, attendre avant de réessayer
                wait_time = (attempt + 1) * 60
                print(f"[VERIF-M1] 🔄 Attente {wait_time}s avant nouvelle tentative...")
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                print(f"[VERIF-M1] ❌ Erreur tentative {attempt+1}: {e}")
        
        print(f"[VERIF-M1] ❌ Échec après {max_retries} tentatives")
        return None

    async def _verify_m1_candle_real_prices(self, signal_id, pair, direction, candle_start, is_otc=False):
        """
        Vérifie une bougie M1 avec prix RÉELS uniquement - VERSION AMÉLIORÉE
        Utilise otc_provider pour les données OTC
        """
        try:
            print(f"[VERIF-M1] 🔍 Récupération PRIX RÉELS bougie M1 {candle_start.strftime('%H:%M')}...")
            
            # Récupérer le prix d'ouverture M1
            entry_price = await self._get_exact_m1_candle_price_real(
                pair, candle_start, 'open', is_otc
            )
            
            await asyncio.sleep(1)  # Petit délai entre les requêtes
            
            # Récupérer le prix de fermeture M1
            exit_price = await self._get_exact_m1_candle_price_real(
                pair, candle_start, 'close', is_otc
            )
            
            # Si un des prix est None, essayer avec fallback étendu
            if entry_price is None:
                print(f"[VERIF-M1] ⚠️ Prix d'ouverture RÉEL non disponible - tentative fallback étendu")
                entry_price = await self._get_price_with_extended_fallback(pair, candle_start, 'open', is_otc)
            
            if exit_price is None:
                print(f"[VERIF-M1] ⚠️ Prix de fermeture RÉEL non disponible - tentative fallback étendu")
                exit_price = await self._get_price_with_extended_fallback(pair, candle_start, 'close', is_otc)
            
            # Si toujours None, on abandonne
            if entry_price is None:
                print(f"[VERIF-M1] ❌ Prix d'ouverture RÉEL non disponible malgré fallback")
                return None, None
            
            if exit_price is None:
                print(f"[VERIF-M1] ❌ Prix de fermeture RÉEL non disponible malgré fallback")
                return None, None
            
            # Vérifier que les prix ne sont pas à 0 (erreur API)
            if entry_price == 0 or exit_price == 0:
                print(f"[VERIF-M1] ❌ Prix à 0 détectés - probable erreur API")
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
                'reason': f"Bougie M1 RÉELLE {candle_start.strftime('%H:%M')}"
            }
            
            return result, details
            
        except Exception as e:
            print(f"[VERIF-M1] ❌ Erreur _verify_m1_candle_real_prices: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    async def _get_price_with_extended_fallback(self, pair, candle_start, price_type='close', is_otc=False):
        """
        Fallback étendu qui essaie toutes les méthodes disponibles
        """
        print(f"[VERIF-M1] 🔄 Fallback étendu pour {pair} {candle_start.strftime('%H:%M')}")
        
        # 1. Si OTC et otc_provider disponible, utiliser le provider
        if is_otc and self.otc_provider:
            print(f"[VERIF-M1]  1. Essai via otc_provider...")
            price = await self._get_otc_price_via_provider(pair, candle_start, price_type)
            if price is not None and price != 0:
                print(f"[VERIF-M1]   ✅ otc_provider réussi: {price:.5f}")
                return price
        
        # 2. Pour OTC, essayer les APIs directes comme fallback
        if is_otc:
            print(f"[VERIF-M1]  2. Essai APIs directes OTC...")
            price = await self._get_otc_price_direct_api(pair, candle_start, price_type)
            if price is not None and price != 0:
                print(f"[VERIF-M1]   ✅ API directe OTC réussi: {price:.5f}")
                return price
        
        # 3. Si Forex, utiliser TwelveData
        if not is_otc:
            print(f"[VERIF-M1]  3. Essai Forex via TwelveData...")
            price = await self._get_forex_candle_price_real(pair, candle_start, price_type)
            if price is not None and price != 0:
                print(f"[VERIF-M1]   ✅ Forex TwelveData réussi: {price:.5f}")
                return price
        
        print(f"[VERIF-M1]  ❌ Toutes les méthodes ont échoué")
        return None

    async def _get_exact_m1_candle_price_real(self, pair, candle_start, price_type='close', is_otc=False):
        """
        Récupère le prix d'UNE bougie M1 SPÉCIFIQUE
        Utilise otc_provider pour les données OTC
        """
        try:
            if is_otc:
                # PRIORITÉ : Utiliser otc_provider si disponible
                if self.otc_provider:
                    price = await self._get_otc_price_via_provider(pair, candle_start, price_type)
                    if price is not None:
                        return price
                
                # Fallback : APIs directes
                return await self._get_otc_price_direct_api(pair, candle_start, price_type)
            else:
                return await self._get_forex_candle_price_real(pair, candle_start, price_type)
                
        except Exception as e:
            print(f"[VERIF-M1] ❌ Erreur _get_exact_m1_candle_price_real: {e}")
            return None

    async def _get_otc_price_via_provider(self, pair, candle_start, price_type='close'):
        """
        Récupère le prix OTC via otc_provider (même source que le générateur de signaux)
        """
        if not self.otc_provider:
            print(f"[VERIF-M1] ⚠️ otc_provider non disponible pour {pair}")
            return None
        
        try:
            print(f"[VERIF-M1] 🔄 Utilisation otc_provider pour {pair}...")
            
            # Récupérer les données via otc_provider (méthode synchrone)
            # On l'exécute dans un thread séparé pour ne pas bloquer
            import asyncio
            import functools
            
            df = await asyncio.get_event_loop().run_in_executor(
                None,
                functools.partial(
                    self.otc_provider.get_otc_data,
                    pair=pair,
                    interval='1min',
                    limit=10
                )
            )
            
            if df is None or df.empty:
                print(f"[VERIF-M1] ❌ otc_provider retourné DataFrame vide pour {pair}")
                return None
            
            print(f"[VERIF-M1] 📊 otc_provider retourné {len(df)} bougies pour {pair}")
            
            # Chercher la bougie la plus proche de candle_start
            # Convertir les index en datetime timezone-aware
            df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            
            # Trouver la bougie la plus proche
            time_diffs = abs(df.index - candle_start)
            min_diff_idx = time_diffs.idxmin()
            min_diff = time_diffs.min()
            
            # Vérifier que la différence est acceptable (< 2 minutes)
            if min_diff > timedelta(minutes=2):
                print(f"[VERIF-M1] ⚠️ Aucune bougie OTC proche ({min_diff.total_seconds():.0f}s de différence)")
                return None
            
            # Récupérer le prix demandé
            if price_type == 'open':
                price = float(df.loc[min_diff_idx, 'open'])
            else:
                price = float(df.loc[min_diff_idx, 'close'])
            
            print(f"[VERIF-M1] ✅ Prix OTC via provider: {price:.5f} (diff: {min_diff.total_seconds():.0f}s)")
            return price
            
        except Exception as e:
            print(f"[VERIF-M1] ❌ Erreur otc_provider pour {pair}: {e}")
            return None

    async def _get_otc_price_direct_api(self, pair, candle_start, price_type='close'):
        """
        Fallback: Récupère le prix OTC via API directe (conservé pour compatibilité)
        """
        print(f"[VERIF-M1] 🔄 Fallback API directe pour {pair}")
        
        # Liste des exchanges à essayer
        exchanges = ['bybit', 'binance', 'kucoin']
        
        for exchange in exchanges:
            try:
                print(f"[VERIF-M1]  Essai {exchange}...")
                price = await self._get_otc_price_direct_exchange(pair, candle_start, price_type, exchange)
                if price is not None and price != 0:
                    print(f"[VERIF-M1]  ✅ {exchange} réussi: {price:.5f}")
                    return price
            except Exception as e:
                print(f"[VERIF-M1]  ⚠️ {exchange} échoué: {e}")
                continue
        
        print(f"[VERIF-M1] ❌ Tous les exchanges OTC ont échoué")
        return None

    async def _get_otc_price_direct_exchange(self, pair, candle_start, price_type='close', exchange='bybit'):
        """
        Récupère le prix d'un exchange OTC spécifique (méthode directe)
        """
        try:
            await self._wait_if_rate_limited()
            
            # Mapping des paires
            symbol_mapping = {
                'bybit': {
                    'BTC/USD': 'BTCUSDT',
                    'ETH/USD': 'ETHUSDT',
                    'TRX/USD': 'TRXUSDT',
                    'LTC/USD': 'LTCUSDT',
                },
                'binance': {
                    'BTC/USD': 'BTCUSDT',
                    'ETH/USD': 'ETHUSDT',
                    'TRX/USD': 'TRXUSDT',
                    'LTC/USD': 'LTCUSDT',
                },
                'kucoin': {
                    'BTC/USD': 'BTC-USDT',
                    'ETH/USD': 'ETH-USDT',
                    'TRX/USD': 'TRX-USDT',
                    'LTC/USD': 'LTC-USDT',
                }
            }
            
            symbol = symbol_mapping.get(exchange, {}).get(pair, pair.replace('/', ''))
            
            # Timestamp en millisecondes
            start_ms = int(candle_start.timestamp() * 1000)
            
            if exchange == 'binance':
                url = 'https://api.binance.com/api/v3/klines'
                params = {
                    'symbol': symbol,
                    'interval': '1m',
                    'startTime': start_ms - 120000,
                    'limit': 5
                }
                
                resp = self._session.get(url, params=params, timeout=10)
                self._increment_api_call()
                
                if resp.status_code != 200:
                    return None
                
                data = resp.json()
                
                if isinstance(data, list) and len(data) > 0:
                    for candle in data:
                        candle_time = datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc)
                        time_diff = abs((candle_time - candle_start).total_seconds())
                        
                        if time_diff < 60:
                            if price_type == 'open':
                                return float(candle[1])
                            else:
                                return float(candle[4])
            
            elif exchange == 'bybit':
                url = 'https://api.bybit.com/v5/market/kline'
                params = {
                    'category': 'spot',
                    'symbol': symbol,
                    'interval': '1',
                    'start': start_ms - 120000,
                    'limit': 5
                }
                
                resp = self._session.get(url, params=params, timeout=10)
                self._increment_api_call()
                
                if resp.status_code != 200:
                    return None
                
                data = resp.json()
                
                if data.get('retCode') == 0 and data.get('result', {}).get('list'):
                    candles = data['result']['list']
                    for candle in candles:
                        candle_time = datetime.fromtimestamp(int(candle[0]) / 1000, tz=timezone.utc)
                        time_diff = abs((candle_time - candle_start).total_seconds())
                        
                        if time_diff < 60:
                            if price_type == 'open':
                                return float(candle[1])
                            else:
                                return float(candle[4])
            
            return None
            
        except Exception as e:
            print(f"[VERIF-M1] ❌ Erreur API directe {exchange}: {e}")
            return None

    async def _get_forex_candle_price_real(self, pair, candle_start, price_type='close'):
        """
        Récupère le prix RÉEL depuis TwelveData (Forex) - VERSION AMÉLIORÉE
        Retourne None si non disponible
        """
        try:
            await self._wait_if_rate_limited()
            
            params = {
                'symbol': pair,
                'interval': '1min',
                'outputsize': 30,
                'apikey': self.api_key,
                'format': 'JSON'
            }
            
            print(f"[VERIF-M1] 🔍 Forex API RÉELLE: {pair} {price_type} autour de {candle_start.strftime('%H:%M')} UTC")
            
            resp = self._session.get(self.base_url, params=params, timeout=20)
            self._increment_api_call()
            
            resp.raise_for_status()
            data = resp.json()
            
            if 'code' in data and data['code'] == 429:
                print(f"[VERIF-M1] ⚠️ Limite API atteinte - attente 60s")
                await asyncio.sleep(60)
                self.api_calls_count = 0
                self.api_calls_reset_time = datetime.now()
                return None
            
            if 'values' not in data or not data['values']:
                print(f"[VERIF-M1] ❌ Aucune donnée Forex RÉELLE retournée")
                return None
            
            # Chercher LA bougie M1 la plus proche
            best_candle = None
            best_diff = float('inf')
            
            for candle in data['values']:
                try:
                    candle_time = self._parse_datetime_robust(candle['datetime'])
                    if candle_time is None:
                        continue
                    
                    candle_time_m1 = self._round_to_m1(candle_time)
                    time_diff = abs((candle_time_m1 - candle_start).total_seconds())
                    
                    if time_diff < best_diff:
                        best_diff = time_diff
                        best_candle = candle
                        
                except Exception as e:
                    continue
            
            if best_candle and best_diff < 120:
                try:
                    price = float(best_candle[price_type])
                    print(f"[VERIF-M1] ✅ Prix Forex RÉEL trouvé (diff: {best_diff:.0f}s) - {price_type}: {price:.5f}")
                    return price
                except KeyError:
                    try:
                        price = float(best_candle['close'])
                        print(f"[VERIF-M1] ⚠️ Fallback Forex close: {price:.5f}")
                        return price
                    except:
                        pass
            
            print(f"[VERIF-M1] ❌ Prix Forex RÉEL {candle_start.strftime('%H:%M')} NON trouvé (meilleure diff: {best_diff:.0f}s)")
            return None
            
        except Exception as e:
            print(f"[VERIF-M1] ❌ Erreur API Forex RÉELLE: {e}")
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
            
            if entry_price == 0 or exit_price == 0:
                print(f"[VERIF-M1] ⚠️ PRIX À 0 DÉTECTÉS - Marquage comme en attente")
                verification_method = 'PENDING_REAL_DATA'
            
            with self.engine.begin() as conn:
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

            with self.engine.connect() as conn:
                pending = conn.execute(
                    text("""
                        SELECT id, pair, direction, ts_enter, confidence, kill_zone, payload_json
                        FROM signals
                        WHERE result IS NULL
                          AND datetime(ts_enter, '+5 minutes') <= datetime('now')
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
                    
                    result = await self.verify_single_signal_with_retry(signal_id, max_retries=2)
                    
                    if result == 'WIN' or result == 'LOSE':
                        verified += 1
                    elif result is None:
                        waiting += 1
                        no_data += 1
                    
                    await asyncio.sleep(3)
                    
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
        Utilise otc_provider pour les données OTC
        """
        try:
            print(f"\n{'='*70}")
            print(f"[REPAIR REAL] 🔧 Réparation prix RÉELS manquants avec otc_provider")
            print(f"{'='*70}")
            
            with self.engine.connect() as conn:
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
                
                signal_time = self._parse_datetime_robust(ts_enter)
                if signal_time is None:
                    print(f"[REPAIR REAL] ❌ Impossible de parser ts_enter - SKIP")
                    skipped += 1
                    continue
                
                age_hours = (datetime.now(timezone.utc) - signal_time).total_seconds() / 3600
                
                if age_hours > 48:
                    print(f"[REPAIR REAL] ⚠️ Signal trop vieux ({age_hours:.1f}h) - SKIP")
                    skipped += 1
                    continue
                
                try:
                    # Déterminer le mode
                    is_otc = False
                    
                    if payload_json:
                        try:
                            payload = json.loads(payload_json)
                            mode = payload.get('mode', 'Forex')
                            is_otc = (mode == 'OTC' or mode == 'CRYPTO' or mode == 'CRYPTO_OTC')
                        except:
                            pass
                    
                    # Calculer la bougie M1
                    candle_start, _ = self._calculate_correct_m1_candle(signal_time)
                    
                    # Récupérer les prix RÉELS avec priorité à otc_provider pour OTC
                    if is_otc and self.otc_provider:
                        print(f"[REPAIR REAL] 🔄 Utilisation otc_provider pour OTC...")
                        entry_price = await self._get_otc_price_via_provider(pair, candle_start, 'open')
                        await asyncio.sleep(1)
                        exit_price = await self._get_otc_price_via_provider(pair, candle_start, 'close')
                    else:
                        entry_price = await self._get_exact_m1_candle_price_real(pair, candle_start, 'open', is_otc)
                        await asyncio.sleep(1)
                        exit_price = await self._get_exact_m1_candle_price_real(pair, candle_start, 'close', is_otc)
                    
                    # Si échec, essayer fallback étendu
                    if entry_price is None:
                        entry_price = await self._get_price_with_extended_fallback(pair, candle_start, 'open', is_otc)
                    
                    if exit_price is None:
                        exit_price = await self._get_price_with_extended_fallback(pair, candle_start, 'close', is_otc)
                    
                    if entry_price is None:
                        print(f"[REPAIR REAL] ❌ Prix entrée RÉEL non disponible pour #{signal_id}")
                        failed += 1
                        continue
                    
                    if exit_price is None:
                        print(f"[REPAIR REAL] ❌ Prix sortie RÉEL non disponible pour #{signal_id}")
                        failed += 1
                        continue
                    
                    if entry_price == 0 or exit_price == 0:
                        print(f"[REPAIR REAL] ⚠️ Prix à 0 détectés - ABANDON")
                        failed += 1
                        continue
                    
                    price_diff = exit_price - entry_price
                    pips_diff = abs(price_diff) * 10000
                    
                    with self.engine.begin() as conn:
                        conn.execute(
                            text("""
                                UPDATE signals
                                SET entry_price = :entry_price,
                                    exit_price = :exit_price,
                                    pips = :pips,
                                    verification_method = 'REPAIRED_HISTORICAL'
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
                    
                    await asyncio.sleep(3)
                    
                except Exception as e:
                    print(f"[REPAIR REAL] ❌ Erreur réparation #{signal_id}: {e}")
                    failed += 1
            
            print(f"\n{'='*70}")
            print(f"[REPAIR REAL] 📈 RÉPARATION TERMINÉE:")
            print(f"[REPAIR REAL]   • Réparés avec données RÉELLES: {repaired}")
            print(f"[REPAIR REAL]   • Échecs (données manquantes): {failed}")
            print(f"[REPAIR REAL]   • Skippés (trop vieux): {skipped}")
            print(f"[REPAIR REAL]   • Total: {len(signals_to_repair)}")
            print(f"[REPAIR REAL] 🔧 otc_provider utilisé: {'✅ Oui' if self.otc_provider else '❌ Non'}")
            print(f"{'='*70}")
            
        except Exception as e:
            print(f"[REPAIR REAL] ❌ Erreur globale réparation: {e}")
            import traceback
            traceback.print_exc()
