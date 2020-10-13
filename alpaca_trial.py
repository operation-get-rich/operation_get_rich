import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
import os

from config import ALPACA_KEY_ID, ALPACA_SECRET_KEY, ALPACA_BASE_URL

if os.path.isfile("./stock_price.csv"):
    os.remove("./stock_price.csv")

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

print("Starting Download:", flush=True)

start = 0
start_date = '2019-01-01T00:00:00-05:00'
end_date = '2020-10-11T23:59:00-05:00'
while start < len(tickers):
    end = min(len(tickers), start + 100)
    print("Current Iteration: ", start, flush=True)
    print("Downloaded Tickers: ", tickers[start:end], flush=True)

    barset = api.get_barset(
        symbols=','.join(tickers[start:end]),
        timeframe='1Min',
        start=start_date,
        end=end_date
    )

    data = []
    for ticker in barset:
        for bar in barset[ticker]:
            data.append([ticker, bar.t, bar.o, bar.c, bar.l, bar.h, bar.v])

    data_np = np.array(data)
    df = pd.DataFrame(data_np, columns=['ticker', 'time', 'open', 'close', 'low', 'high', 'volume'])

    df.to_csv(path_or_buf='./raw_data/stock_price' + str(start) + '.csv')
    start += 100
