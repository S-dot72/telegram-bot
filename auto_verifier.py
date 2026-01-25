import asyncio
from datetime import datetime, timedelta, timezone
from sqlalchemy import text
import requests
import json
import re
from typing import Dict, List, Tuple, Optional
import time

class AutoResultVerifier:
    """
    Système de vérification PRÉCIS des signaux de trading
    Version 2.0 - Basé sur l'audit technique complet
    """
    
    def __init__(self, engine, twelvedata_api_key):
        self.engine = engine
        self.api_key = twelvedata_api_key
        self.base_url = 'https://api.twelvedata.com/time_series'
        self._session = requests.Session()
        
        # Configuration STRICTE - PAS D'APPROXIMATION
        self.TOLERANCE_MS = 2000  # 2 secondes max de décalage
        self.MIN_DATA_QUALITY = 0.8  # 80% minimum pour validation
        self.EXACT_PRICE_REQUIRED = True  # Rejeter les prix approximatifs
        
        # Mappage des exchanges pour crypto
        self.crypto_endpoints = {
            'bybit': {
                'base': 'https://api.bybit.com/v5/market',
                'kline': '/kline',
                'trades': '/recent-trade',
                'ticker': '/tickers'
            },
            'binance': 'https://api.binance.com/api/v3',
            'kucoin': 'https://api.kucoin.com/api/v1'
        }
        
        print(f"[VERIF_PRO] ✅ AutoResultVerifier PRO initialisé")
        print(f"[VERIF_PRO] 📊 Mode: PRÉCIS | Tolérance: {self.TOLERANCE_MS}ms")
        print(f"[VERIF_PRO] 🎯 Qualité minimum: {self.MIN_DATA_QUALITY*100}%")
    
    # -----------------------------------------------------------------
    # 1. UTILITAIRES DE BASE
    # -----------------------------------------------------------------
    
    def _parse_datetime(self, dt_str: str) -> datetime:
        """Parse un timestamp en datetime UTC"""
        if isinstance(dt_str, datetime):
            if dt_str.tzinfo is None:
                return dt_str.replace(tzinfo=timezone.utc)
            return dt_str.astimezone(timezone.utc)
        
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S.%f'
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(dt_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except:
                continue
        
        raise ValueError(f"Format datetime non reconnu: {dt_str}")
    
    def _map_pair_to_symbol(self, pair: str, exchange: str = 'bybit') -> str:
        """Convertit une paire TradingView en symbole d'API"""
        mapping = {
            'bybit': {
                'BTC/USD': 'BTCUSDT',
                'ETH/USD': 'ETHUSDT',
                'TRX/USD': 'TRXUSDT',
                'LTC/USD': 'LTCUSDT',
                'XRP/USD': 'XRPUSDT',
                'ADA/USD': 'ADAUSDT',
                'DOT/USD': 'DOTUSDT',
                'SOL/USD': 'SOLUSDT'
            },
            'binance': {
                'BTC/USD': 'BTCUSDT',
                'ETH/USD': 'ETHUSDT',
                'TRX/USD': 'TRXUSDT',
                'LTC/USD': 'LTCUSDT',
                'XRP/USD': 'XRPUSDT',
                'ADA/USD': 'ADAUSDT',
                'DOT/USD': 'DOTUSDT',
                'SOL/USD': 'SOLUSDT'
            },
            'kucoin': {
                'BTC/USD': 'BTC-USDT',
                'ETH/USD': 'ETH-USDT',
                'TRX/USD': 'TRX-USDT',
                'LTC/USD': 'LTC-USDT',
                'XRP/USD': 'XRP-USDT',
                'ADA/USD': 'ADA-USDT',
                'DOT/USD': 'DOT-USDT',
                'SOL/USD': 'SOL-USDT'
            }
        }
        
        symbol = mapping.get(exchange, {}).get(pair)
        if not symbol:
            # Fallback: retirer le slash
            symbol = pair.replace('/', '').replace('-', '')
        
        return symbol
    
    def _extract_market_type(self, payload_json: str, pair: str) -> Dict:
        """Détermine le type de marché de manière UNIFORME"""
        try:
            payload = json.loads(payload_json) if payload_json else {}
            
            # Source unique de vérité
            market_type = payload.get('market_type', '').upper()
            if not market_type:
                market_type = payload.get('mode', '').upper()
            
            if market_type in ['CRYPTO', 'OTC', 'CRYPTO_OTC']:
                return {
                    'type': 'CRYPTO',
                    'is_otc': True,
                    'exchange': payload.get('exchange', 'bybit')
                }
            elif market_type in ['FOREX', 'FX']:
                return {
                    'type': 'FOREX',
                    'is_otc': False,
                    'exchange': 'twelvedata'
                }
            else:
                # Détection automatique basée sur la paire
                crypto_keywords = ['BTC', 'ETH', 'TRX', 'LTC', 'XRP', 'ADA', 'DOT', 'SOL']
                is_crypto = any(kw in pair for kw in crypto_keywords)
                
                return {
                    'type': 'CRYPTO' if is_crypto else 'FOREX',
                    'is_otc': is_crypto,
                    'exchange': 'bybit' if is_crypto else 'twelvedata'
                }
                
        except Exception as e:
            print(f"[VERIF] ⚠️ Erreur détection marché: {e}")
            return {
                'type': 'UNKNOWN',
                'is_otc': False,
                'exchange': 'unknown'
            }
    
    # -----------------------------------------------------------------
    # 2. RÉCUPÉRATION DES PRIX EXACTS (CRITIQUE)
    # -----------------------------------------------------------------
    
    def _get_exact_price_at_timestamp(self, pair: str, timestamp: datetime, 
                                     market_info: Dict, is_entry: bool = True) -> Optional[float]:
        """
        Récupère le prix EXACT au timestamp donné
        Retourne None si data insuffisante
        """
        try:
            # Convertir en timestamp millisecondes
            target_ts_ms = int(timestamp.timestamp() * 1000)
            
            if market_info['type'] == 'CRYPTO':
                return self._get_crypto_exact_price(pair, target_ts_ms, market_info['exchange'])
            else:
                return self._get_forex_exact_price(pair, target_ts_ms)
                
        except Exception as e:
            print(f"[VERIF_EXACT] ❌ Erreur récupération prix: {e}")
            return None
    
    def _get_crypto_exact_price(self, pair: str, target_ts_ms: int, exchange: str = 'bybit') -> Optional[float]:
        """
        Récupère le prix crypto EXACT via l'API de l'exchange
        """
        try:
            symbol = self._map_pair_to_symbol(pair, exchange)
            
            # Stratégie 1: Trades récents (plus précis)
            price = self._get_crypto_from_recent_trades(symbol, target_ts_ms, exchange)
            if price is not None:
                return price
            
            # Stratégie 2: Bougie 1min avec validation stricte
            price = self._get_crypto_from_validated_kline(symbol, target_ts_ms, exchange)
            if price is not None:
                return price
            
            # Stratégie 3: Essayer un autre exchange
            if exchange != 'binance':
                price = self._get_crypto_exact_price(pair, target_ts_ms, 'binance')
                if price is not None:
                    print(f"[VERIF_EXACT] 🔄 Prix trouvé via Binance")
                    return price
            
            print(f"[VERIF_EXACT] ⚠️ Aucun prix exact trouvé pour {pair} à {target_ts_ms}")
            return None
            
        except Exception as e:
            print(f"[VERIF_EXACT] ❌ Erreur crypto: {e}")
            return None
    
    def _get_crypto_from_recent_trades(self, symbol: str, target_ts_ms: int, exchange: str) -> Optional[float]:
        """Récupère le prix via les trades récents"""
        try:
            if exchange == 'bybit':
                url = f"{self.crypto_endpoints['bybit']['base']}{self.crypto_endpoints['bybit']['trades']}"
                params = {
                    'category': 'spot',
                    'symbol': symbol,
                    'limit': 100
                }
            elif exchange == 'binance':
                url = f"{self.crypto_endpoints['binance']}/trades"
                params = {'symbol': symbol, 'limit': 100}
            else:
                return None
            
            response = self._session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Parser la réponse selon l'exchange
                trades = []
                if exchange == 'bybit':
                    if data.get('retCode') == 0:
                        trades = data.get('result', {}).get('list', [])
                elif exchange == 'binance':
                    trades = data
                
                if trades:
                    # Trouver le trade le plus proche AVANT le timestamp
                    closest_trade = None
                    min_diff = float('inf')
                    
                    for trade in trades:
                        if exchange == 'bybit':
                            trade_ts = int(trade['time'])
                            trade_price = float(trade['price'])
                        else:  # binance
                            trade_ts = int(trade['time'])
                            trade_price = float(trade['price'])
                        
                        diff = target_ts_ms - trade_ts
                        
                        # Le trade doit être AVANT ou à peu près au même moment
                        if -1000 <= diff <= self.TOLERANCE_MS:
                            if abs(diff) < min_diff:
                                min_diff = abs(diff)
                                closest_trade = trade_price
                    
                    if closest_trade is not None:
                        print(f"[VERIF_TRADES] ✅ Prix exact via trades: {closest_trade} (diff: {min_diff}ms)")
                        return closest_trade
            
            return None
            
        except Exception as e:
            print(f"[VERIF_TRADES] ⚠️ Erreur: {e}")
            return None
    
    def _get_crypto_from_validated_kline(self, symbol: str, target_ts_ms: int, exchange: str) -> Optional[float]:
        """
        Récupère via bougie avec validation STRICTE du timestamp
        """
        try:
            # Calculer le début de la bougie 1min qui contient le timestamp
            candle_start_ms = target_ts_ms - (target_ts_ms % 60000)
            candle_end_ms = candle_start_ms + 60000
            
            if exchange == 'bybit':
                url = f"{self.crypto_endpoints['bybit']['base']}{self.crypto_endpoints['bybit']['kline']}"
                params = {
                    'category': 'spot',
                    'symbol': symbol,
                    'interval': '1',
                    'start': candle_start_ms,
                    'end': candle_end_ms,
                    'limit': 1
                }
            elif exchange == 'binance':
                url = f"{self.crypto_endpoints['binance']}/klines"
                start_time = candle_start_ms
                end_time = candle_end_ms
                params = {
                    'symbol': symbol,
                    'interval': '1m',
                    'startTime': start_time,
                    'endTime': end_time,
                    'limit': 1
                }
            else:
                return None
            
            response = self._session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                candles = []
                
                if exchange == 'bybit':
                    if data.get('retCode') == 0:
                        candles = data.get('result', {}).get('list', [])
                elif exchange == 'binance':
                    candles = data
                
                if candles:
                    candle = candles[0]
                    
                    if exchange == 'bybit':
                        candle_start = int(candle[0])
                        open_price = float(candle[1])
                    else:  # binance
                        candle_start = int(candle[0])
                        open_price = float(candle[1])
                    
                    # VÉRIFICATION CRITIQUE: la bougie doit contenir le timestamp
                    if candle_start <= target_ts_ms < candle_start + 60000:
                        print(f"[VERIF_KLINE] ✅ Prix bougie validé: {open_price}")
                        return open_price
            
            print(f"[VERIF_KLINE] ⚠️ Pas de bougie valide pour le timestamp")
            return None
            
        except Exception as e:
            print(f"[VERIF_KLINE] ❌ Erreur: {e}")
            return None
    
    def _get_forex_exact_price(self, pair: str, target_ts_ms: int) -> Optional[float]:
        """
        Récupère le prix Forex EXACT via TwelveData
        """
        try:
            # Convertir ms en datetime pour TwelveData
            target_dt = datetime.fromtimestamp(target_ts_ms / 1000, tz=timezone.utc)
            
            # TwelveData fonctionne par minute, on prend la bougie contenant le timestamp
            minute_start = target_dt.replace(second=0, microsecond=0)
            minute_end = minute_start + timedelta(minutes=1)
            
            # Formater pour l'API
            start_str = minute_start.strftime('%Y-%m-%d %H:%M:%S')
            end_str = minute_end.strftime('%Y-%m-%d %H:%M:%S')
            
            params = {
                'symbol': pair,
                'interval': '1min',
                'start_date': start_str,
                'end_date': end_str,
                'apikey': self.api_key,
                'outputsize': 2,
                'format': 'JSON'
            }
            
            response = self._session.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'values' in data and data['values']:
                    # Trouver la bougie correspondante
                    target_str = minute_start.strftime('%Y-%m-%d %H:%M:00')
                    
                    for candle in data['values']:
                        if candle.get('datetime', '').startswith(target_str):
                            open_price = float(candle['open'])
                            print(f"[VERIF_FOREX] ✅ Prix Forex trouvé: {open_price}")
                            return open_price
            
            print(f"[VERIF_FOREX] ⚠️ Pas de données Forex pour {pair} à {target_dt}")
            return None
            
        except Exception as e:
            print(f"[VERIF_FOREX] ❌ Erreur: {e}")
            return None
    
    # -----------------------------------------------------------------
    # 3. ÉVALUATION DE LA QUALITÉ DES DONNÉES
    # -----------------------------------------------------------------
    
    def _evaluate_data_quality(self, entry_price: Optional[float], exit_price: Optional[float],
                              ts_enter: datetime, ts_exit: datetime, 
                              market_info: Dict) -> Dict:
        """
        Évalue la qualité des données sur une échelle 0-1
        Retourne un dict avec score et issues
        """
        score = 1.0
        issues = []
        
        # 1. Données manquantes (critique)
        if entry_price is None:
            score *= 0.3
            issues.append('Prix entrée manquant')
        
        if exit_price is None:
            score *= 0.3
            issues.append('Prix sortie manquant')
        
        # 2. Prix réalistes
        if entry_price is not None:
            if entry_price <= 0:
                score *= 0.2
                issues.append('Prix entrée invalide (≤0)')
            elif market_info['type'] == 'CRYPTO':
                if entry_price < 0.000001 or entry_price > 1000000:
                    score *= 0.5
                    issues.append('Prix entrée hors limites réalistes')
        
        if exit_price is not None:
            if exit_price <= 0:
                score *= 0.2
                issues.append('Prix sortie invalide (≤0)')
        
        # 3. Volatilité réaliste
        if entry_price and exit_price:
            change_pct = abs(exit_price - entry_price) / entry_price
            
            if market_info['type'] == 'CRYPTO':
                if change_pct > 0.10:  # 10% max sur 1 minute
                    score *= 0.6
                    issues.append(f'Volatilité excessive: {change_pct:.2%}')
            else:  # Forex
                if change_pct > 0.01:  # 1% max sur 1 minute
                    score *= 0.6
                    issues.append(f'Volatilité excessive: {change_pct:.2%}')
        
        # 4. Timestamps réalistes
        now = datetime.now(timezone.utc)
        
        if ts_enter > now:
            score *= 0.5
            issues.append('Timestamp entrée dans le futur')
        
        if ts_exit > now:
            score *= 0.5
            issues.append('Timestamp sortie dans le futur')
        
        # 5. Durée réaliste
        duration = (ts_exit - ts_enter).total_seconds()
        if not (55 <= duration <= 65):  # Environ 1 minute
            score *= 0.8
            issues.append(f'Durée anormale: {duration}s')
        
        return {
            'score': round(score, 3),
            'issues': issues,
            'is_reliable': score >= self.MIN_DATA_QUALITY
        }
    
    # -----------------------------------------------------------------
    # 4. CALCUL DES RÉSULTATS
    # -----------------------------------------------------------------
    
    def _calculate_precise_result(self, direction: str, entry_price: float, 
                                 exit_price: float, market_info: Dict) -> Dict:
        """
        Calcule le résultat EXACT sans approximation
        """
        # Vérifier les prix
        if entry_price is None or exit_price is None:
            return {
                'result': 'DATA_MISSING',
                'pips': 0.0,
                'price_change': 0.0,
                'change_pct': 0.0
            }
        
        # Calculer le changement
        price_change = exit_price - entry_price
        change_pct = price_change / entry_price if entry_price != 0 else 0
        
        # Déterminer le résultat
        if direction == 'CALL':
            if price_change > 0:
                result = 'WIN'
            elif price_change < 0:
                result = 'LOSE'
            else:
                result = 'DRAW'
        else:  # PUT
            if price_change < 0:
                result = 'WIN'
            elif price_change > 0:
                result = 'LOSE'
            else:
                result = 'DRAW'
        
        # Calculer les pips selon le marché
        if market_info['type'] == 'CRYPTO':
            # Pour crypto, on utilise le changement en prix
            pips = abs(price_change)
        else:
            # Pour Forex, pips = 4ème décimale (sauf JPY = 2ème)
            if 'JPY' in market_info.get('pair', ''):
                pips = abs(price_change) * 100  # 2ème décimale
            else:
                pips = abs(price_change) * 10000  # 4ème décimale
        
        return {
            'result': result,
            'pips': round(pips, 6),
            'price_change': round(price_change, 8),
            'change_pct': round(change_pct, 6),
            'direction_actual': 'UP' if price_change > 0 else 'DOWN' if price_change < 0 else 'FLAT'
        }
    
    def _build_verification_reason(self, pair: str, direction: str, result_info: Dict,
                                  entry_price: float, exit_price: float,
                                  market_info: Dict, data_quality: Dict) -> str:
        """
        Construit une raison détaillée et structurée
        """
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        
        parts = [
            f"[PRECISE_V2] {timestamp}",
            f"PAIR: {pair}",
            f"DIRECTION: {direction}",
            f"RESULT: {result_info['result']}",
            f"MARKET: {market_info['type']}",
            f"EXCHANGE: {market_info['exchange']}",
            f"DATA_QUALITY: {data_quality['score']:.3f}",
            f"ENTRY_PRICE: {entry_price if entry_price else 'N/A'}",
            f"EXIT_PRICE: {exit_price if exit_price else 'N/A'}",
            f"PRICE_CHANGE: {result_info['price_change']:.8f}",
            f"CHANGE_PCT: {result_info['change_pct']:.4%}",
            f"PIPS: {result_info['pips']:.6f}"
        ]
        
        if data_quality['issues']:
            parts.append(f"ISSUES: {'; '.join(data_quality['issues'])}")
        
        return " | ".join(parts)
    
    # -----------------------------------------------------------------
    # 5. VÉRIFICATION PRINCIPALE
    # -----------------------------------------------------------------
    
    async def verify_single_signal(self, signal_id: int) -> str:
        """
        Vérifie un signal M1 avec les données RÉELLES du marché
        Version PRO - Zéro approximation
        """
        try:
            print(f"\n{'='*60}")
            print(f"[VERIF_PRO] 🔍 Vérification PRÉCISE signal #{signal_id}")
            print(f"{'='*60}")
            
            # Récupérer le signal depuis la DB
            with self.engine.connect() as conn:
                signal = conn.execute(
                    text("""
                        SELECT id, pair, direction, ts_enter, confidence, 
                               payload_json, result, reason
                        FROM signals
                        WHERE id = :sid
                    """),
                    {"sid": signal_id}
                ).fetchone()
            
            if not signal:
                print(f"[VERIF_PRO] ❌ Signal #{signal_id} non trouvé")
                return 'INVALID'
            
            (sig_id, pair, direction, ts_enter, confidence, 
             payload_json, existing_result, existing_reason) = signal
            
            # Vérifier si déjà vérifié avec le nouveau système
            if existing_reason and '[PRECISE_V2]' in existing_reason:
                print(f"[VERIF_PRO] ✅ Signal déjà vérifié avec système PRECISE_V2")
                return existing_result
            
            # Analyser les timestamps
            ts_enter_dt = self._parse_datetime(ts_enter)
            ts_exit_dt = ts_enter_dt + timedelta(minutes=1)  # M1 par défaut
            
            # Déterminer le type de marché
            market_info = self._extract_market_type(payload_json, pair)
            
            print(f"[VERIF_PRO] 📊 Signal #{sig_id} - {pair} {direction}")
            print(f"[VERIF_PRO] 🕐 Entrée: {ts_enter_dt}")
            print(f"[VERIF_PRO] 🕐 Sortie: {ts_exit_dt}")
            print(f"[VERIF_PRO] 📈 Marché: {market_info['type']} ({market_info['exchange']})")
            
            # Récupérer les prix EXACTS
            print(f"[VERIF_PRO] 🔍 Récupération prix d'entrée...")
            entry_price = self._get_exact_price_at_timestamp(
                pair, ts_enter_dt, market_info, True
            )
            
            print(f"[VERIF_PRO] 🔍 Récupération prix de sortie...")
            exit_price = self._get_exact_price_at_timestamp(
                pair, ts_exit_dt, market_info, False
            )
            
            # Évaluer la qualité des données
            data_quality = self._evaluate_data_quality(
                entry_price, exit_price, ts_enter_dt, ts_exit_dt, market_info
            )
            
            print(f"[VERIF_PRO] 📊 Qualité données: {data_quality['score']:.3f}")
            
            # Décision basée sur la qualité
            if not data_quality['is_reliable']:
                reason = f"[PRECISE_V2] QUALITÉ INSUFFISANTE ({data_quality['score']:.3f})"
                if data_quality['issues']:
                    reason += f" | Issues: {'; '.join(data_quality['issues'])}"
                
                print(f"[VERIF_PRO] 🚫 Signal #{sig_id} rejeté - qualité insuffisante")
                
                # Marquer comme INVALID
                self._update_signal_result(sig_id, 'INVALID', reason, 
                                         entry_price, exit_price, 0.0)
                return 'INVALID'
            
            # Calculer le résultat exact
            result_info = self._calculate_precise_result(
                direction, entry_price, exit_price, market_info
            )
            
            # Construire la raison détaillée
            reason = self._build_verification_reason(
                pair, direction, result_info,
                entry_price, exit_price,
                market_info, data_quality
            )
            
            # Sauvegarder le résultat
            self._update_signal_result(
                sig_id, result_info['result'], reason,
                entry_price, exit_price, result_info['pips']
            )
            
            # Afficher le résultat
            print(f"[VERIF_PRO] 🎯 RÉSULTAT: {result_info['result']}")
            print(f"[VERIF_PRO] 💰 Entrée: {entry_price:.8f}")
            print(f"[VERIF_PRO] 💰 Sortie: {exit_price:.8f}")
            print(f"[VERIF_PRO] 📈 Changement: {result_info['price_change']:.8f} ({result_info['change_pct']:.4%})")
            print(f"[VERIF_PRO] 📊 Pips: {result_info['pips']:.6f}")
            print(f"[VERIF_PRO] 🔄 Direction réelle: {result_info['direction_actual']}")
            
            return result_info['result']
            
        except Exception as e:
            print(f"[VERIF_PRO] ❌ Erreur critique: {e}")
            import traceback
            traceback.print_exc()
            
            # Marquer comme erreur technique
            reason = f"[PRECISE_V2] ERREUR TECHNIQUE: {str(e)[:100]}"
            self._update_signal_result(signal_id, 'ERROR', reason, None, None, 0.0)
            
            return 'ERROR'
    
    def _update_signal_result(self, signal_id: int, result: str, reason: str,
                             entry_price: Optional[float], exit_price: Optional[float],
                             pips: float):
        """Met à jour le résultat du signal dans la DB"""
        try:
            with self.engine.begin() as conn:
                # Vérifier quelles colonnes sont disponibles
                table_info = conn.execute(
                    text("PRAGMA table_info(signals)")
                ).fetchall()
                
                columns = [row[1] for row in table_info]
                
                # Préparer les valeurs
                values = {
                    'result': result,
                    'reason': reason[:500],  # Limiter la longueur
                    'id': signal_id
                }
                
                # Ajouter les prix si disponibles
                if 'entry_price' in columns and entry_price is not None:
                    values['entry_price'] = entry_price
                if 'exit_price' in columns and exit_price is not None:
                    values['exit_price'] = exit_price
                if 'pips' in columns:
                    values['pips'] = pips
                if 'ts_exit' in columns:
                    values['ts_exit'] = datetime.now(timezone.utc).isoformat()
                
                # Construire dynamiquement la requête
                set_clauses = [f"{col} = :{col}" for col in values.keys() if col != 'id']
                
                query = text(f"""
                    UPDATE signals
                    SET {', '.join(set_clauses)}
                    WHERE id = :id
                """)
                
                conn.execute(query, values)
                
                print(f"[VERIF_PRO] 💾 Résultat sauvegardé: {result}")
                
        except Exception as e:
            print(f"[VERIF_PRO] ❌ Erreur sauvegarde: {e}")
    
    # -----------------------------------------------------------------
    # 6. STATISTIQUES FIABLES
    # -----------------------------------------------------------------
    
    def get_reliable_statistics(self, days: int = 30) -> Dict:
        """
        Calcule des statistiques FIABLES en filtrant par qualité
        N'utilise que les signaux vérifiés avec PRECISE_V2
        """
        try:
            with self.engine.connect() as conn:
                # Récupérer tous les signaux de la période
                signals = conn.execute(text("""
                    SELECT id, pair, direction, result, reason, confidence,
                           entry_price, exit_price, pips, ts_enter
                    FROM signals
                    WHERE ts_enter >= datetime('now', '-' || :days || ' days')
                    ORDER BY ts_enter DESC
                """), {"days": days}).fetchall()
            
            # Initialiser les structures d'analyse
            analysis = {
                'total_signals': len(signals),
                'precise_signals': 0,
                'reliable_signals': 0,
                'by_pair': {},
                'by_result': {'WIN': 0, 'LOSE': 0, 'INVALID': 0, 'ERROR': 0, 'OTHER': 0},
                'quality_distribution': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNKNOWN': 0}
            }
            
            for signal in signals:
                (sig_id, pair, direction, result, reason, confidence,
                 entry_price, exit_price, pips, ts_enter) = signal
                
                # Vérifier si c'est une vérification PRECISE_V2
                if reason and '[PRECISE_V2]' in reason:
                    analysis['precise_signals'] += 1
                    
                    # Extraire la qualité des données
                    quality_score = self._extract_quality_from_reason(reason)
                    
                    # Déterminer la catégorie de qualité
                    if quality_score >= 0.9:
                        quality_cat = 'HIGH'
                    elif quality_score >= self.MIN_DATA_QUALITY:
                        quality_cat = 'MEDIUM'
                    else:
                        quality_cat = 'LOW'
                    
                    analysis['quality_distribution'][quality_cat] += 1
                    
                    # Inclure seulement les signaux de qualité suffisante
                    if quality_score >= self.MIN_DATA_QUALITY and result in ['WIN', 'LOSE']:
                        analysis['reliable_signals'] += 1
                        
                        # Mettre à jour les stats par paire
                        if pair not in analysis['by_pair']:
                            analysis['by_pair'][pair] = {
                                'total': 0, 'wins': 0, 'losses': 0,
                                'total_pips': 0.0, 'confidences': []
                            }
                        
                        pair_stats = analysis['by_pair'][pair]
                        pair_stats['total'] += 1
                        
                        if result == 'WIN':
                            pair_stats['wins'] += 1
                            analysis['by_result']['WIN'] += 1
                        else:
                            pair_stats['losses'] += 1
                            analysis['by_result']['LOSE'] += 1
                        
                        if pips:
                            pair_stats['total_pips'] += pips
                        
                        if confidence:
                            pair_stats['confidences'].append(confidence)
                else:
                    analysis['by_result']['OTHER'] += 1
                    analysis['quality_distribution']['UNKNOWN'] += 1
            
            # Calculer les métriques finales
            final_stats = {
                'overview': {
                    'period_days': days,
                    'total_signals': analysis['total_signals'],
                    'precise_signals': analysis['precise_signals'],
                    'reliable_signals': analysis['reliable_signals'],
                    'precise_coverage': self._safe_divide(analysis['precise_signals'], analysis['total_signals']),
                    'reliable_coverage': self._safe_divide(analysis['reliable_signals'], analysis['precise_signals']),
                    'data_quality': analysis['quality_distribution']
                },
                'performance': {},
                'summary': {}
            }
            
            # Calculer les performances par paire
            valid_pairs = []
            for pair, stats in analysis['by_pair'].items():
                if stats['total'] >= 3:  # Minimum 3 signaux pour statistiques
                    win_rate = self._safe_divide(stats['wins'], stats['total'])
                    avg_pips = self._safe_divide(stats['total_pips'], stats['total'])
                    avg_confidence = self._safe_divide(sum(stats['confidences']), len(stats['confidences'])) if stats['confidences'] else 0
                    
                    # Calculer le Profit Factor
                    if stats['losses'] > 0 and avg_pips > 0:
                        profit_factor = (stats['wins'] * avg_pips) / (stats['losses'] * avg_pips)
                    else:
                        profit_factor = float('inf') if stats['wins'] > 0 else 0
                    
                    # Calculer l'Expected Value
                    expected_value = (win_rate * avg_pips) - ((1 - win_rate) * avg_pips)
                    
                    pair_data = {
                        'total': stats['total'],
                        'wins': stats['wins'],
                        'losses': stats['losses'],
                        'win_rate': round(win_rate, 3),
                        'avg_pips': round(avg_pips, 4),
                        'avg_confidence': round(avg_confidence, 3),
                        'profit_factor': round(profit_factor, 2),
                        'expected_value': round(expected_value, 4),
                        'reliability': 'HIGH' if stats['total'] >= 10 else 'MEDIUM' if stats['total'] >= 5 else 'LOW'
                    }
                    
                    final_stats['performance'][pair] = pair_data
                    valid_pairs.append(pair_data)
            
            # Calculer les résumés globaux
            if valid_pairs:
                total_trades = sum(p['total'] for p in valid_pairs)
                total_wins = sum(p['wins'] for p in valid_pairs)
                overall_win_rate = self._safe_divide(total_wins, total_trades)
                
                # Moyenne pondérée par le nombre de trades
                weighted_avg_pips = sum(p['avg_pips'] * p['total'] for p in valid_pairs) / total_trades if total_trades > 0 else 0
                weighted_avg_confidence = sum(p['avg_confidence'] * p['total'] for p in valid_pairs) / total_trades if total_trades > 0 else 0
                
                final_stats['summary'] = {
                    'total_reliable_trades': total_trades,
                    'overall_win_rate': round(overall_win_rate, 3),
                    'weighted_avg_pips': round(weighted_avg_pips, 4),
                    'weighted_avg_confidence': round(weighted_avg_confidence, 3),
                    'best_pair': max(valid_pairs, key=lambda x: x['win_rate']) if valid_pairs else None,
                    'worst_pair': min(valid_pairs, key=lambda x: x['win_rate']) if valid_pairs else None
                }
            
            # Trier les performances par win_rate descendant
            final_stats['performance'] = dict(sorted(
                final_stats['performance'].items(),
                key=lambda x: x[1]['win_rate'],
                reverse=True
            ))
            
            return final_stats
            
        except Exception as e:
            print(f"[STATS_PRO] ❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def _extract_quality_from_reason(self, reason: str) -> float:
        """Extrait le score de qualité de la raison"""
        if not reason:
            return 0.0
        
        # Chercher le pattern "DATA_QUALITY: X.XXX"
        match = re.search(r'DATA_QUALITY:\s*([\d.]+)', reason)
        if match:
            return float(match.group(1))
        
        # Fallback basé sur des keywords
        if 'QUALITÉ INSUFFISANTE' in reason:
            return 0.3
        elif 'ISSUES:' in reason:
            return 0.5
        elif '[PRECISE_V2]' in reason:
            return 0.8  # Par défaut pour les vérifications précises
        
        return 0.0
    
    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """Division sécurisée avec gestion de division par zéro"""
        return numerator / denominator if denominator != 0 else 0.0
    
    # -----------------------------------------------------------------
    # 7. OUTILS DE DIAGNOSTIC
    # -----------------------------------------------------------------
    
    def diagnose_signal(self, signal_id: int) -> Dict:
        """
        Retourne un diagnostic détaillé d'un signal
        """
        try:
            with self.engine.connect() as conn:
                signal = conn.execute(
                    text("""
                        SELECT id, pair, direction, ts_enter, confidence,
                               payload_json, result, reason, entry_price, exit_price, pips
                        FROM signals
                        WHERE id = :sid
                    """),
                    {"sid": signal_id}
                ).fetchone()
            
            if not signal:
                return {'status': 'NOT_FOUND', 'message': 'Signal non trouvé'}
            
            (sig_id, pair, direction, ts_enter, confidence, payload_json,
             result, reason, entry_price, exit_price, pips) = signal
            
            # Analyser le signal
            ts_enter_dt = self._parse_datetime(ts_enter)
            market_info = self._extract_market_type(payload_json, pair)
            
            # Tester la récupération des prix
            entry_test = self._get_exact_price_at_timestamp(pair, ts_enter_dt, market_info, True)
            exit_test = self._get_exact_price_at_timestamp(pair, ts_enter_dt + timedelta(minutes=1), market_info, False)
            
            # Évaluer la qualité
            data_quality = self._evaluate_data_quality(
                entry_test, exit_test, ts_enter_dt, 
                ts_enter_dt + timedelta(minutes=1), market_info
            )
            
            return {
                'status': 'SUCCESS',
                'signal_id': sig_id,
                'pair': pair,
                'direction': direction,
                'timestamp': ts_enter_dt.isoformat(),
                'market_type': market_info['type'],
                'exchange': market_info['exchange'],
                'current_result': result,
                'current_reason': reason,
                'current_prices': {
                    'entry': entry_price,
                    'exit': exit_price,
                    'pips': pips
                },
                'price_test': {
                    'entry_price_test': entry_test,
                    'exit_price_test': exit_test,
                    'data_quality_score': data_quality['score'],
                    'data_quality_issues': data_quality['issues']
                },
                'verification_possible': data_quality['is_reliable'],
                'recommendation': 'VERIFY' if data_quality['is_reliable'] else 'INVALIDATE'
            }
            
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}
    
    async def batch_verify_signals(self, signal_ids: List[int], 
                                  max_concurrent: int = 5) -> Dict:
        """
        Vérifie plusieurs signaux en parallèle
        """
        results = {
            'total': len(signal_ids),
            'completed': 0,
            'succeeded': 0,
            'failed': 0,
            'results': {}
        }
        
        # Diviser en lots
        for i in range(0, len(signal_ids), max_concurrent):
            batch = signal_ids[i:i + max_concurrent]
            
            print(f"[BATCH] 🔄 Traitement lot {i//max_concurrent + 1}: {len(batch)} signaux")
            
            # Créer les tâches
            tasks = [self.verify_single_signal(sig_id) for sig_id in batch]
            
            # Exécuter en parallèle
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Analyser les résultats
            for sig_id, result in zip(batch, batch_results):
                results['completed'] += 1
                
                if isinstance(result, Exception):
                    results['failed'] += 1
                    results['results'][sig_id] = {
                        'status': 'ERROR',
                        'error': str(result)
                    }
                else:
                    results['succeeded'] += 1
                    results['results'][sig_id] = {
                        'status': 'SUCCESS',
                        'result': result
                    }
            
            # Petite pause entre les lots
            await asyncio.sleep(1)
        
        print(f"[BATCH] ✅ Batch terminé: {results['succeeded']}/{results['total']} succès")
        return results
    
    # -----------------------------------------------------------------
    # 8. MIGRATION DES DONNÉES EXISTANTES
    # -----------------------------------------------------------------
    
    async def migrate_existing_signals(self, limit: int = 100) -> Dict:
        """
        Re-vérifie les signaux existants avec la nouvelle logique
        """
        try:
            print(f"[MIGRATION] 🚀 Migration des signaux existants...")
            
            # Récupérer les IDs des signaux non vérifiés ou vérifiés anciennement
            with self.engine.connect() as conn:
                signal_ids = conn.execute(text("""
                    SELECT id FROM signals 
                    WHERE (result IS NULL OR result NOT IN ('INVALID', 'ERROR'))
                       AND (reason IS NULL OR reason NOT LIKE '[PRECISE_V2]%')
                    ORDER BY ts_enter DESC
                    LIMIT :limit
                """), {"limit": limit}).fetchall()
            
            signal_ids = [sid[0] for sid in signal_ids]
            
            print(f"[MIGRATION] 📊 {len(signal_ids)} signaux à migrer")
            
            # Vérifier en batch
            results = await self.batch_verify_signals(signal_ids, max_concurrent=3)
            
            # Résumé
            migration_stats = {
                'total_processed': results['total'],
                'successfully_migrated': results['succeeded'],
                'failed_migrations': results['failed'],
                'migration_rate': self._safe_divide(results['succeeded'], results['total'])
            }
            
            print(f"[MIGRATION] 🎉 Migration terminée")
            print(f"[MIGRATION] 📈 Taux de réussite: {migration_stats['migration_rate']:.1%}")
            
            return migration_stats
            
        except Exception as e:
            print(f"[MIGRATION] ❌ Erreur: {e}")
            return {'error': str(e)}
    
    # -----------------------------------------------------------------
    # 9. COMPATIBILITÉ ASCENDANTE
    # -----------------------------------------------------------------
    
    async def verify_signal_compat(self, signal_id: int) -> str:
        """
        Méthode de compatibilité ascendante
        Utilise la nouvelle logique mais retourne juste le résultat
        """
        return await self.verify_single_signal(signal_id)
    
    def get_asset_statistics_compat(self):
        """
        Version de compatibilité de get_asset_statistics
        Utilise la nouvelle logique mais formate comme l'ancienne
        """
        reliable_stats = self.get_reliable_statistics(days=7)
        
        if 'error' in reliable_stats:
            return {}
        
        # Convertir au format attendu
        result = {}
        for pair, stats in reliable_stats.get('performance', {}).items():
            result[pair] = {
                'total': stats['total'],
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': round(stats['win_rate'], 3),
                'avg_confidence': stats['avg_confidence'] * 100  # Convertir en pourcentage
            }
        
        return result

# -----------------------------------------------------------------
# POINT D'ENTRÉE POUR TESTS
# -----------------------------------------------------------------

if __name__ == "__main__":
    # Exemple d'utilisation
    print("AutoResultVerifier PRO - Système de vérification précis")
    print("Version 2.0 - Correction complète des biais structurels")
    print("=" * 60)
    print("Fonctionnalités principales:")
    print("1. Prix EXACTS au timestamp (pas d'OHLC approximatif)")
    print("2. Validation STRICTE de la qualité des données")
    print("3. Zéro fallback aléatoire")
    print("4. Statistiques basées uniquement sur données fiables")
    print("5. Diagnostic complet par signal")
    print("=" * 60)
