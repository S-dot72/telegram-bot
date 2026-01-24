"""
Module OTC pour trading week-end
Fournit des données synthétiques basées sur les crypto-monnaies
qui tradent 24/7 y compris le week-end
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone, timedelta

class OTCDataProvider:
    """
    Fournisseur de données OTC pour le week-end
    
    Sources disponibles:
    1. Crypto (BTC, ETH) - Tradent 24/7
    2. Données synthétiques basées sur historique
    3. API alternatives (Binance, CoinGecko)
    """
    
    def __init__(self, twelvedata_api_key=None):
        self.twelvedata_key = twelvedata_api_key
        self.binance_base = 'https://api.binance.com/api/v3'
        self.coingecko_base = 'https://api.coingecko.com/api/v3'
        
        # Paires OTC disponibles le week-end
        self.otc_pairs = {
            'BTC/USD': 'BTC/USDT',
            'ETH/USD': 'ETH/USDT',
            'TRX/USD': 'TRX/USDT',
            'LTC/USD': 'LTC/USDT',
            'BCH/USD': 'BCHUSDT',
            'ADA/USD': 'ADAUSDT',
            'DOT/USD': 'DOTUSDT',
            'LINK/USD': 'LINKUSDT'
        }
    
    def is_weekend(self):
        """Vérifie si c'est le week-end (marché Forex fermé)"""
        now_utc = datetime.now(timezone.utc)
        weekday = now_utc.weekday()
        hour = now_utc.hour
        
        # Samedi
        if weekday == 5:
            return True
        # Dimanche avant 22h UTC
        if weekday == 6 and hour < 22:
            return True
        # Vendredi après 22h UTC
        if weekday == 4 and hour >= 22:
            return True
        
        return False
    
    def get_otc_data_binance(self, pair, interval='1m', limit=500):
        """
        Récupère données crypto depuis Binance (gratuit, illimité)
        
        Args:
            pair: Paire OTC (ex: BTC/USD)
            interval: 1m, 5m, 15m, 1h, etc.
            limit: Nombre de bougies (max 1000)
        
        Returns:
            DataFrame avec OHLCV
        """
        try:
            # Convertir paire OTC en symbole Binance
            if pair not in self.otc_pairs:
                raise ValueError(f"Paire {pair} non disponible en OTC")
            
            symbol = self.otc_pairs[pair]
            
            # Mapping interval
            interval_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d'
            }
            
            binance_interval = interval_map.get(interval, '1m')
            
            # Requête Binance
            url = f"{self.binance_base}/klines"
            params = {
                'symbol': symbol,
                'interval': binance_interval,
                'limit': limit
            }
            
            print(f"   📡 Binance: {symbol} {binance_interval}")
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Convertir en DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Nettoyer et formater
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Sélectionner colonnes nécessaires
            df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            df.index = df['datetime']
            
            print(f"   ✅ {len(df)} bougies Binance récupérées")
            
            return df
            
        except Exception as e:
            print(f"   ❌ Erreur Binance: {e}")
            return None
    
    def get_otc_data_coingecko(self, pair, days=1):
        """
        Récupère données crypto depuis CoinGecko (gratuit, limité)
        Alternative à Binance
        
        Args:
            pair: Paire OTC (ex: BTC/USD)
            days: Nombre de jours d'historique
        
        Returns:
            DataFrame avec OHLC
        """
        try:
            # Mapping paires vers IDs CoinGecko
            coin_map = {
                'BTC/USD': 'bitcoin',
                'ETH/USD': 'ethereum',
                'XRP/USD': 'ripple',
                'LTC/USD': 'litecoin',
                'BCH/USD': 'bitcoin-cash',
                'ADA/USD': 'cardano',
                'DOT/USD': 'polkadot',
                'LINK/USD': 'chainlink'
            }
            
            if pair not in coin_map:
                raise ValueError(f"Paire {pair} non disponible")
            
            coin_id = coin_map[pair]
            
            # Requête CoinGecko
            url = f"{self.coingecko_base}/coins/{coin_id}/ohlc"
            params = {
                'vs_currency': 'usd',
                'days': days
            }
            
            print(f"   📡 CoinGecko: {coin_id}")
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Convertir en DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['volume'] = 0  # CoinGecko OHLC ne fournit pas le volume
            
            df.index = df['datetime']
            
            print(f"   ✅ {len(df)} bougies CoinGecko récupérées")
            
            return df
            
        except Exception as e:
            print(f"   ❌ Erreur CoinGecko: {e}")
            return None
    
    def get_synthetic_data(self, pair, interval='1m', num_candles=500):
        """
        Génère données synthétiques réalistes basées sur patterns crypto
        Utilisé en dernier recours si APIs indisponibles
        
        Args:
            pair: Paire (pour le prix de base)
            interval: Timeframe
            num_candles: Nombre de bougies
        
        Returns:
            DataFrame avec OHLCV synthétiques
        """
        print(f"   🎭 Génération données synthétiques pour {pair}")
        
        # Prix de base selon la paire
        base_prices = {
            'BTC/USD': 42000.0,
            'ETH/USD': 2200.0,
            'XRP/USD': 0.60,
            'LTC/USD': 70.0,
            'BCH/USD': 200.0,
            'ADA/USD': 0.50,
            'DOT/USD': 7.0,
            'LINK/USD': 15.0
        }
        
        base_price = base_prices.get(pair, 100.0)
        
        # Générer timestamp
        end_time = datetime.now(timezone.utc)
        interval_minutes = 1 if interval == '1m' else 5
        dates = [end_time - timedelta(minutes=interval_minutes * i) for i in range(num_candles)]
        dates.reverse()
        
        # Générer prix avec volatilité réaliste
        np.random.seed(42)
        prices = [base_price]
        
        for i in range(1, num_candles):
            # Mouvement aléatoire avec tendance
            change_pct = np.random.normal(0, 0.002)  # 0.2% volatilité
            trend = np.sin(i / 50) * 0.0005  # Tendance sinusoïdale
            
            new_price = prices[-1] * (1 + change_pct + trend)
            prices.append(new_price)
        
        # Créer OHLC
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            volatility = close * 0.001
            high = close + np.random.uniform(0, volatility)
            low = close - np.random.uniform(0, volatility)
            open_price = prices[i-1] if i > 0 else close
            
            data.append({
                'datetime': date.strftime('%Y-%m-%d %H:%M:%S'),
                'open': round(open_price, 5),
                'high': round(high, 5),
                'low': round(low, 5),
                'close': round(close, 5),
                'volume': int(np.random.uniform(1000, 5000))
            })
        
        df = pd.DataFrame(data)
        df.index = pd.to_datetime(df['datetime'])
        
        print(f"   ✅ {len(df)} bougies synthétiques générées")
        
        return df
    
    def get_otc_data(self, pair, interval='1m', limit=500):
        """
        Méthode principale - Essaie plusieurs sources dans l'ordre
        
        1. Binance (meilleur - gratuit, illimité)
        2. CoinGecko (backup)
        3. Synthétique (dernier recours)
        
        Args:
            pair: Paire OTC
            interval: Timeframe
            limit: Nombre de bougies
        
        Returns:
            DataFrame avec OHLCV
        """
        print(f"\n🔍 Récupération données OTC: {pair} {interval}")
        
        # 1. Essayer Binance (meilleur)
        df = self.get_otc_data_binance(pair, interval, limit)
        if df is not None and len(df) > 0:
            print(f"✅ Source: Binance")
            return df
        
        # 2. Essayer CoinGecko
        days = 1 if limit <= 288 else 7  # 288 bougies M5 = 1 jour
        df = self.get_otc_data_coingecko(pair, days)
        if df is not None and len(df) > 0:
            print(f"✅ Source: CoinGecko")
            return df
        
        # 3. Fallback synthétique
        print(f"⚠️ APIs indisponibles - Mode synthétique")
        df = self.get_synthetic_data(pair, interval, limit)
        return df
    
    def get_available_pairs(self):
        """Retourne les paires OTC disponibles"""
        return list(self.otc_pairs.keys())


# === Intégration dans signal_bot.py ===

def get_otc_or_forex_data(pair, interval, outputsize=300):
    """
    Fonction unifiée qui choisit automatiquement entre Forex et OTC
    
    Usage dans signal_bot.py:
        df = get_otc_or_forex_data(pair, TIMEFRAME_M1, 400)
    """
    from auto_verifier import AutoResultVerifier
    
    # Créer verifier juste pour check weekend
    verifier = AutoResultVerifier(None, None)
    
    if verifier._is_weekend(datetime.now(timezone.utc)):
        print("🏖️ Week-end détecté - Mode OTC Crypto")
        
        # Utiliser OTC Provider
        otc = OTCDataProvider()
        
        # Convertir paire Forex vers Crypto si nécessaire
        forex_to_crypto = {
            'EUR/USD': 'BTC/USD',
            'GBP/USD': 'ETH/USD',
            'USD/JPY': 'XRP/USD',
            'AUD/USD': 'LTC/USD'
        }
        
        otc_pair = forex_to_crypto.get(pair, 'BTC/USD')
        print(f"   🔄 {pair} → {otc_pair} (OTC)")
        
        return otc.get_otc_data(otc_pair, interval, outputsize)
    
    else:
        print("📈 Marché ouvert - Mode Forex standard")
        # Utiliser TwelveData comme d'habitude
        from signal_bot import fetch_ohlc_td
        return fetch_ohlc_td(pair, interval, outputsize)


# === Configuration OTC dans config.py ===

# Ajouter ces lignes dans config.py:
"""
# OTC Configuration (week-end)
OTC_ENABLED = os.getenv('OTC_ENABLED', 'true').lower() == 'true'
OTC_PAIRS = [p.strip() for p in os.getenv('OTC_PAIRS', 'BTC/USD,ETH/USD,XRP/USD').split(',')]
OTC_PROVIDER = os.getenv('OTC_PROVIDER', 'binance')  # binance, coingecko, synthetic
"""


# === Exemple d'utilisation ===

if __name__ == '__main__':
    # Test du provider OTC
    print("="*60)
    print("TEST OTC DATA PROVIDER")
    print("="*60)
    
    otc = OTCDataProvider()
    
    # Test 1: Binance
    print("\n1️⃣ Test Binance M1:")
    df = otc.get_otc_data_binance('BTC/USD', '1m', 100)
    if df is not None:
        print(f"\nDernières 5 bougies:")
        print(df.tail())
        print(f"\nPrix actuel BTC: ${df['close'].iloc[-1]:.2f}")
    
    # Test 2: CoinGecko
    print("\n\n2️⃣ Test CoinGecko:")
    df = otc.get_otc_data_coingecko('ETH/USD', days=1)
    if df is not None:
        print(f"\nDernières 3 bougies:")
        print(df.tail(3))
    
    # Test 3: Synthétique
    print("\n\n3️⃣ Test Synthétique:")
    df = otc.get_synthetic_data('TRX/USD', '1m', 50)
    print(f"\nDernières 5 bougies:")
    print(df.tail())
    
    # Test 4: Auto (essaie toutes les sources)
    print("\n\n4️⃣ Test Auto (cascade):")
    df = otc.get_otc_data('BTC/USD', '1m', 100)
    print(f"\nRésultat final: {len(df)} bougies")
    
    # Test 5: Paires disponibles
    print("\n\n5️⃣ Paires OTC disponibles:")
    for pair in otc.get_available_pairs():
        print(f"   • {pair}")
    
    print("\n" + "="*60)
    print("✅ Tests terminés")
    print("="*60)
