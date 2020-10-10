import csv

import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
import talib

from config import ALPACA_KEY_ID, ALPACA_SECRET_KEY, ALPACA_BASE_URL


def vwap(df):
    # https://stackoverflow.com/questions/44854512/how-to-calculate-vwap-volume-weighted-average-price-using-groupby-and-apply
    v = df.volume.values
    p = df.typical_price.values
    return df.assign(vwap=(p * v).cumsum() / v.cumsum())


api = tradeapi.REST(
    key_id=ALPACA_KEY_ID,
    secret_key=ALPACA_SECRET_KEY,
    base_url=ALPACA_BASE_URL,
)
tickers = []
file = open('NYSE_only_symbols.txt', 'r')
ticker_names = file.readlines()

for ticker_name in ticker_names:
    tickers.append(ticker_name.strip())

file = open('NASDAQ_only_symbols.txt', 'r')
ticker_names = file.readlines()

for ticker_name in ticker_names:
    tickers.append(ticker_name.strip())


start = 0
while start < len(tickers):
    end = min(len(tickers), start + 100)
    barset = api.get_barset(
        symbols=','.join(tickers[start:end]),
        timeframe='1Min',
        start='2019-01-01T03:00:00-05:00',
        end='2020-10-05T15:00:00-05:00'
    )

    data = []
    for ticker in barset:
        for bar in barset[ticker]:
            data.append([ticker, bar.t, bar.o, bar.c, bar.l, bar.h, bar.v])

    data_np = np.array(data)
    df = pd.DataFrame(data_np, columns=['ticker', 'time', 'open', 'close', 'low', 'high', 'volume'])

    df.to_csv(path_or_buf='./stock_price.csv', mode='a')
    start += 100
