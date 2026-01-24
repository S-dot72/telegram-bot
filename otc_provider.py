import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from typing import Optional

class OTCDataProvider:
    """Fournisseur de données OTC utilisant TwelveData pour les cryptos"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        
    def is_weekend(self) -> bool:
        """Vérifie si c'est le week-end (UTC)"""
        utc_now = datetime.utcnow()
        return utc_now.weekday() >= 5  # Samedi=5, Dimanche=6
    
    def get_available_pairs(self) -> list:
        """Retourne la liste des paires OTC disponibles sur TwelveData"""
        return ['BTC/USD', 'ETH/USD', 'XRP/USD', 'LTC/USD']
    
    def get_twelvedata_crypto(self, pair: str, interval: str = '1min', outputsize: int = 300) -> Optional[pd.DataFrame]:
        """Récupère les données crypto depuis TwelveData"""
        try:
            # TwelveData accepte les paires crypto comme 'BTC/USD'
            params = {
                'symbol': pair,
                'interval': interval,
                'outputsize': outputsize,
                'apikey': self.api_key,
                'format': 'JSON'
            }
            
            url = f"{self.base_url}/time_series"
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'code' in data and data['code'] == 429:
                    print(f"⚠️ Limite API TwelveData atteinte pour {pair}")
                    return None
                
                if 'values' in data and len(data['values']) > 0:
                    # Convertir en DataFrame
                    df = pd.DataFrame(data['values'])[::-1].reset_index(drop=True)
                    
                    # Convertir les types
                    for col in ['open', 'high', 'low', 'close']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    if 'volume' in df.columns:
                        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                    
                    df.index = pd.to_datetime(df['datetime'])
                    
                    print(f"✅ TwelveData Crypto: {len(df)} bougies pour {pair}")
                    return df
                else:
                    print(f"⚠️ Pas de données dans la réponse TwelveData pour {pair}")
                    return None
            else:
                print(f"❌ Erreur HTTP {response.status_code} pour {pair}")
                return None
                
        except Exception as e:
            print(f"❌ Erreur TwelveData Crypto: {e}")
            return None
    
    def generate_synthetic_data(self, pair: str, interval: str = '1min', outputsize: int = 300) -> pd.DataFrame:
        """Génère des données synthétiques en cas d'indisponibilité"""
        print(f"🔧 Génération données synthétiques pour {pair}")
        
        # Prix de base réalistes
        base_prices = {
            'BTC/USD': 45000.0,
            'ETH/USD': 2500.0,
            'XRP/USD': 0.60,
            'LTC/USD': 70.0,
            'EUR/USD': 1.08,
            'GBP/USD': 1.26,
            'USD/JPY': 148.0,
            'AUD/USD': 0.66
        }
        
        base_price = base_prices.get(pair, 100.0)
        
        # Générer des timestamps
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=outputsize)
        timestamps = pd.date_range(start=start_time, end=end_time, periods=outputsize)
        
        # Générer des prix réalistes
        prices = []
        current_price = base_price
        
        for i in range(outputsize):
            # Volatilité réaliste
            volatility = 0.005 if 'BTC' in pair or 'ETH' in pair else 0.002
            
            # Mouvement aléatoire
            change = np.random.normal(0, volatility)
            current_price = current_price * (1 + change)
            
            # Générer OHLC
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
        
        print(f"✅ Données synthétiques générées: {len(df)} bougies")
        return df
