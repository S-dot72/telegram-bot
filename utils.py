import pandas as pd
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange


def compute_indicators(df, ema_fast=8, ema_slow=21, rsi_len=14, bb_len=20):
    df = df.copy()
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    
    # EMA
    df['ema_fast'] = EMAIndicator(close=df['close'], window=ema_fast).ema_indicator()
    df['ema_slow'] = EMAIndicator(close=df['close'], window=ema_slow).ema_indicator()
    
    # MACD
    macd = MACD(close=df['close'])
    df['MACD_12_26_9'] = macd.macd()
    df['MACDs_12_26_9'] = macd.macd_signal()
    df['MACDh_12_26_9'] = macd.macd_diff()
    
    # RSI
    df['rsi'] = RSIIndicator(close=df['close'], window=rsi_len).rsi()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['close'], window=bb_len)
    df['BBL_20_2.0'] = bb.bollinger_lband()
    df['BBM_20_2.0'] = bb.bollinger_mavg()
    df['BBU_20_2.0'] = bb.bollinger_hband()
    
    # ATR
    df['atr'] = AverageTrueRange(
        high=df['high'], 
        low=df['low'], 
        close=df['close'], 
        window=14
    ).average_true_range()
    
    return df


def rule_signal(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    bullish = (prev['ema_fast'] <= prev['ema_slow']) and (last['ema_fast'] > last['ema_slow'])
    bearish = (prev['ema_fast'] >= prev['ema_slow']) and (last['ema_fast'] < last['ema_slow'])
    
    macd_bull = (last['MACD_12_26_9'] > last['MACDs_12_26_9']) and (last['MACDh_12_26_9'] > 0)
    macd_bear = (last['MACD_12_26_9'] < last['MACDs_12_26_9']) and (last['MACDh_12_26_9'] < 0)
    
    rsi = last.get('rsi', None)
    
    if bullish and macd_bull and (rsi is not None and 45 <= rsi <= 65):
        return 'CALL'
    if bearish and macd_bear and (rsi is not None and 35 <= rsi <= 55):
        return 'PUT'
    return None
