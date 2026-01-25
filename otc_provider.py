import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from typing import Optional, Dict, List
import time

class OTCDataProvider:
    """Fournisseur de données OTC avec multiple sources crypto"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.cache = {}
        self.cache_expiry = {}
        
    def is_weekend(self) -> bool:
        """Vérifie si c'est le week-end (UTC)"""
        utc_now = datetime.utcnow()
        return utc_now.weekday() >= 5  # Samedi=5, Dimanche=6
    
    def get_available_pairs(self) -> List[str]:
        """Retourne la liste des paires OTC disponibles"""
        return ['BTC/USD', 'ETH/USD', 'TRX/USD', 'LTC/USD']
    
    def _map_pair_to_symbol(self, pair: str, exchange: str = 'bybit') -> str:
        """Convertit une paire format TradingView en symbole d'API"""
        mapping = {
            'bybit': {
                'BTC/USD': 'BTCUSDT',
                'ETH/USD': 'ETHUSDT',
                'TRX/USD': 'TRXUSDT',
                'LTC/USD': 'LTCUSDT'
            },
            'binance': {
                'BTC/USD': 'BTCUSDT',
                'ETH/USD': 'ETHUSDT',
                'TRX/USD': 'TRXUSDT',
                'LTC/USD': 'LTCUSDT'
            },
            'kucoin': {
                'BTC/USD': 'BTC-USDT',
                'ETH/USD': 'ETH-USDT',
                'TRX/USD': 'TRX-USDT',
                'LTC/USD': 'LTC-USDT'
            }
        }
        return mapping.get(exchange, {}).get(pair, pair.replace('/', '').replace('USD', 'USDT'))
    
    def _interval_to_api_format(self, interval: str, exchange: str = 'bybit') -> str:
        """Convertit l'intervalle au format de l'API"""
        interval_map = {
            'bybit': {
                '1min': '1',
                '5min': '5',
                '15min': '15',
                '30min': '30',
                '1h': '60',
                '4h': '240',
                '1d': 'D'
            },
            'binance': {
                '1min': '1m',
                '5min': '5m',
                '15min': '15m',
                '30min': '30m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d'
            },
            'kucoin': {
                '1min': '1min',
                '5min': '5min',
                '15min': '15min',
                '30min': '30min',
                '1h': '1hour',
                '4h': '4hour',
                '1d': '1day'
            }
        }
        return interval_map.get(exchange, {}).get(interval, '1')
    
    def get_bybit_data(self, pair: str, interval: str = '1min', limit: int = 300) -> Optional[pd.DataFrame]:
        """Récupère les données depuis Bybit (Spot API) - Très fiable"""
        try:
            symbol = self._map_pair_to_symbol(pair, 'bybit')
            api_interval = self._interval_to_api_format(interval, 'bybit')
            
            # Essayer plusieurs endpoints de Bybit
            endpoints = [
                'https://api.bybit.com',
                'https://api.bytick.com',
                'https://api.bybit.org'
            ]
            
            for endpoint in endpoints:
                try:
                    url = f"{endpoint}/v5/market/kline"
                    params = {
                        'category': 'spot',
                        'symbol': symbol,
                        'interval': api_interval,
                        'limit': min(limit, 1000)
                    }
                    
                    response = requests.get(url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data.get('retCode') == 0 and data.get('result'):
                            klines = data['result']['list']
                            
                            if not klines:
                                continue
                            
                            # Convertir en DataFrame
                            df = pd.DataFrame(klines, columns=[
                                'timestamp', 'open', 'high', 'low', 'close', 
                                'volume', 'turnover'
                            ])
                            
                            # Convertir timestamp
                            df['timestamp'] = pd.to_numeric(df['timestamp'])
                            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                            df.set_index('datetime', inplace=True)
                            
                            # Convertir les prix en float
                            for col in ['open', 'high', 'low', 'close', 'volume']:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                            print(f"✅ Bybit ({endpoint}) réussi: {len(df)} bougies")
                            return df[['open', 'high', 'low', 'close', 'volume']]
                            
                except Exception as e:
                    print(f"⚠️ Bybit endpoint {endpoint} échoué: {str(e)[:50]}")
                    continue
            
            print("❌ Tous les endpoints Bybit ont échoué")
            return None
            
        except Exception as e:
            print(f"❌ Erreur Bybit: {e}")
            return None
    
    def get_binance_data(self, pair: str, interval: str = '1min', limit: int = 300) -> Optional[pd.DataFrame]:
        """Récupère les données depuis Binance (Spot API)"""
        try:
            symbol = self._map_pair_to_symbol(pair, 'binance')
            api_interval = self._interval_to_api_format(interval, 'binance')
            
            # Essayer plusieurs endpoints de Binance
            endpoints = [
                'https://api.binance.com',
                'https://api1.binance.com',
                'https://api2.binance.com',
                'https://api3.binance.com'
            ]
            
            for endpoint in endpoints:
                try:
                    url = f"{endpoint}/api/v3/klines"
                    params = {
                        'symbol': symbol,
                        'interval': api_interval,
                        'limit': min(limit, 1000)
                    }
                    
                    response = requests.get(url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data and len(data) > 0:
                            # Convertir en DataFrame
                            df = pd.DataFrame(data, columns=[
                                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                'taker_buy_quote', 'ignore'
                            ])
                            
                            # Convertir timestamp
                            df['timestamp'] = pd.to_numeric(df['timestamp'])
                            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                            df.set_index('datetime', inplace=True)
                            
                            # Convertir les prix en float
                            for col in ['open', 'high', 'low', 'close', 'volume']:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                            print(f"✅ Binance ({endpoint}) réussi: {len(df)} bougies")
                            return df[['open', 'high', 'low', 'close', 'volume']]
                            
                except Exception as e:
                    print(f"⚠️ Binance endpoint {endpoint} échoué: {str(e)[:50]}")
                    continue
            
            print("❌ Tous les endpoints Binance ont échoué")
            return None
            
        except Exception as e:
            print(f"❌ Erreur Binance: {e}")
            return None
    
    def get_kucoin_data(self, pair: str, interval: str = '1min', limit: int = 300) -> Optional[pd.DataFrame]:
        """Récupère les données depuis KuCoin (Spot API)"""
        try:
            symbol = self._map_pair_to_symbol(pair, 'kucoin')
            api_interval = self._interval_to_api_format(interval, 'kucoin')
            
            url = "https://api.kucoin.com/api/v1/market/candles"
            
            # Calculer les timestamps
            end_time = int(datetime.utcnow().timestamp())
            start_time = end_time - (limit * 60)  # Approximatif
            
            params = {
                'type': api_interval,
                'symbol': symbol,
                'startAt': start_time,
                'endAt': end_time
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('code') == '200000' and data.get('data'):
                    candles = data['data']
                    
                    if not candles:
                        return None
                    
                    # KuCoin retourne les données dans l'ordre inverse
                    df = pd.DataFrame(candles, columns=[
                        'timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'
                    ])
                    
                    # Inverser l'ordre
                    df = df.iloc[::-1].reset_index(drop=True)
                    
                    # Convertir timestamp
                    df['timestamp'] = pd.to_numeric(df['timestamp'])
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                    df.set_index('datetime', inplace=True)
                    
                    # Convertir les prix en float
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    print(f"✅ KuCoin réussi: {len(df)} bougies")
                    return df[['open', 'high', 'low', 'close', 'volume']]
            
            return None
            
        except Exception as e:
            print(f"❌ Erreur KuCoin: {e}")
            return None
    
    def get_coingecko_data(self, pair: str, interval: str = '1min', limit: int = 300) -> Optional[pd.DataFrame]:
        """Récupère les données depuis CoinGecko"""
        try:
            # Mapping des paires aux IDs CoinGecko
            cg_ids = {
                'BTC/USD': 'bitcoin',
                'ETH/USD': 'ethereum',
                'TRX/USD': 'tron',
                'LTC/USD': 'litecoin'
            }
            
            coin_id = cg_ids.get(pair)
            if not coin_id:
                return None
            
            # Pour CoinGecko, nous devons utiliser l'API OHLC
            days = 1  # 1 jour pour les données M1
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
            
            params = {
                'vs_currency': 'usd',
                'days': days
            }
            
            # Ajouter un header User-Agent
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data and len(data) > 0:
                    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                    
                    # Convertir timestamp
                    df['timestamp'] = pd.to_numeric(df['timestamp'])
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('datetime', inplace=True)
                    
                    # Convertir les prix en float
                    for col in ['open', 'high', 'low', 'close']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Ajouter un volume synthétique
                    df['volume'] = np.random.uniform(100, 1000, size=len(df))
                    
                    print(f"✅ CoinGecko réussi: {len(df)} bougies")
                    return df[['open', 'high', 'low', 'close', 'volume']]
            
            return None
            
        except Exception as e:
            print(f"❌ Erreur CoinGecko: {e}")
            return None
    
    def get_coincap_data(self, pair: str, interval: str = '1min', limit: int = 300) -> Optional[pd.DataFrame]:
        """Récupère les données depuis CoinCap (bon pour données historiques)"""
        try:
            # CoinCap utilise des IDs différents
            coincap_ids = {
                'BTC/USD': 'bitcoin',
                'ETH/USD': 'ethereum',
                'TRX/USD': 'tron',
                'LTC/USD': 'litecoin'
            }
            
            coin_id = coincap_ids.get(pair)
            if not coin_id:
                return None
            
            # Récupérer les données historiques (gratuit jusqu'à 2000 requêtes/jour)
            url = f"https://api.coincap.io/v2/assets/{coin_id}/history"
            
            end_time = int(datetime.utcnow().timestamp() * 1000)
            start_time = end_time - (limit * 60 * 1000)  # En millisecondes
            
            params = {
                'interval': 'm1',  # 1 minute
                'start': start_time,
                'end': end_time
            }
            
            headers = {
                'Authorization': 'Bearer e6de2e7d-5b25-47ba-bf7c-7d1c93c4b6d1'  # API key publique (limites réduites)
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('data'):
                    history = data['data']
                    
                    if not history:
                        return None
                    
                    df = pd.DataFrame(history)
                    
                    # Renommer les colonnes
                    df = df.rename(columns={
                        'time': 'datetime',
                        'priceUsd': 'close'
                    })
                    
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                    
                    # Convertir en float
                    df['close'] = pd.to_numeric(df['close'], errors='coerce')
                    
                    # Pour CoinCap, nous n'avons qu'un prix, donc générer OHLC
                    df['open'] = df['close'].shift(1).fillna(df['close'])
                    df['high'] = df[['open', 'close']].max(axis=1)
                    df['low'] = df[['open', 'close']].min(axis=1)
                    df['volume'] = np.random.uniform(100, 1000, size=len(df))
                    
                    print(f"✅ CoinCap réussi: {len(df)} bougies")
                    return df[['open', 'high', 'low', 'close', 'volume']]
            
            return None
            
        except Exception as e:
            print(f"❌ Erreur CoinCap: {e}")
            return None
    
    def generate_synthetic_data(self, pair: str, interval: str = '1min', limit: int = 300) -> pd.DataFrame:
        """Génère des données synthétiques réalistes basées sur les prix actuels"""
        print(f"🔧 Génération données synthétiques pour {pair}")
        
        try:
            # Essayer d'obtenir le prix réel d'abord
            real_price = self.get_current_price(pair)
            if real_price:
                base_price = real_price
                print(f"💰 Prix réel obtenu: ${base_price:.4f}")
            else:
                # Prix par défaut si échec
                base_prices = {
                    'BTC/USD': 45000.0,
                    'ETH/USD': 2500.0,
                    'TRX/USD': 0.10,  # Prix TRX autour de 0.10 USD
                    'LTC/USD': 70.0
                }
                base_price = base_prices.get(pair, 100.0)
                print(f"⚠️ Utilisation prix par défaut: ${base_price:.4f}")
            
            # Générer des données réalistes
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=limit)
            timestamps = pd.date_range(start=start_time, end=end_time, periods=limit)
            
            # Volatilité réaliste - TRX a généralement une volatilité plus élevée
            if 'BTC' in pair or 'ETH' in pair:
                volatility = 0.005
            elif 'TRX' in pair:
                volatility = 0.015  # TRX est plus volatile
            else:
                volatility = 0.008
            
            prices = []
            current_price = base_price
            
            for i in range(limit):
                # Mouvement aléatoire réaliste
                change = np.random.normal(0, volatility)
                current_price = current_price * (1 + change)
                
                # Générer OHLC réalistes
                open_price = current_price
                high_price = open_price * (1 + abs(np.random.normal(0, volatility/2)))
                low_price = open_price * (1 - abs(np.random.normal(0, volatility/2)))
                close_price = np.random.uniform(low_price, high_price)
                volume = np.random.uniform(100, 1000)
                
                prices.append({
                    'datetime': timestamps[i],
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
                
                current_price = close_price
            
            df = pd.DataFrame(prices)
            df.set_index('datetime', inplace=True)
            
            print(f"✅ Données synthétiques générées: {len(df)} bougies, dernier prix: ${df.iloc[-1]['close']:.4f}")
            return df
            
        except Exception as e:
            print(f"❌ Erreur génération synthétique: {e}")
            return None
    
    def get_current_price(self, pair: str) -> Optional[float]:
        """Récupère le prix actuel depuis une API simple"""
        try:
            # Utiliser CoinGecko pour le prix actuel
            cg_ids = {
                'BTC/USD': 'bitcoin',
                'ETH/USD': 'ethereum',
                'TRX/USD': 'tron',
                'LTC/USD': 'litecoin'
            }
            
            coin_id = cg_ids.get(pair)
            if not coin_id:
                return None
            
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if coin_id in data:
                    return data[coin_id].get('usd')
            
            return None
            
        except Exception as e:
            print(f"⚠️ Erreur prix actuel: {e}")
            return None
    
    def get_otc_data(self, pair: str, interval: str = '1min', limit: int = 300) -> Optional[pd.DataFrame]:
        """
        Récupère les données OTC depuis plusieurs sources en cascade.
        Priorité: Bybit > Binance > KuCoin > CoinGecko > CoinCap > Synthétique
        """
        print(f"🔍 Recherche données OTC pour {pair}")
        
        # Vérifier le cache
        cache_key = f"{pair}_{interval}_{limit}"
        current_time = datetime.utcnow()
        
        if cache_key in self.cache:
            cached_data, expiry_time = self.cache[cache_key], self.cache_expiry.get(cache_key)
            if expiry_time and current_time < expiry_time:
                print(f"📦 Utilisation données en cache pour {pair}")
                return cached_data
        
        # Liste des sources à essayer dans l'ordre
        sources = [
            ('Bybit', lambda: self.get_bybit_data(pair, interval, limit)),
            ('Binance', lambda: self.get_binance_data(pair, interval, limit)),
            ('KuCoin', lambda: self.get_kucoin_data(pair, interval, limit)),
            ('CoinGecko', lambda: self.get_coingecko_data(pair, interval, limit)),
            ('CoinCap', lambda: self.get_coincap_data(pair, interval, limit))
        ]
        
        for source_name, source_func in sources:
            print(f"  {len(sources) - sources.index((source_name, source_func))}. Essai {source_name}...")
            df = source_func()
            
            if df is not None and len(df) > 0:
                print(f"    ✅ {source_name} réussi: {len(df)} bougies")
                
                # Mettre en cache pendant 30 secondes
                self.cache[cache_key] = df
                self.cache_expiry[cache_key] = current_time + timedelta(seconds=30)
                
                return df
        
        # Fallback sur données synthétiques
        print(f"  ⚠️ Toutes les sources API ont échoué, basculement sur synthétique")
        df = self.generate_synthetic_data(pair, interval, limit)
        
        if df is not None:
            # Mettre en cache pendant 5 secondes seulement pour synthétique
            self.cache[cache_key] = df
            self.cache_expiry[cache_key] = current_time + timedelta(seconds=5)
        
        return df
