import json
import os

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volume import VolumeWeightedAveragePrice

state_file = './state_polygon_01_03_minute_chart_bars.json'
with open('%s' % state_file) as f:
    state = json.load(f)

root_data_dir = '../datas/polygon_early_day_gap_segmenter_parallel'
performance = state['performance']

stocks_to_delete = []



for the_date, stocks in performance.items():
    for symbol, node in stocks.items():
        buy_time = node['buy_time']
        stock_file_path = os.path.join(
            root_data_dir,
            the_date,
            f'{symbol}_{the_date}.csv'
        )
        print(f'Adding TA to {stock_file_path}')
        try:
            df = pd.read_csv(stock_file_path,
                             parse_dates=['time'],
                             )
        except FileNotFoundError:
            stocks_to_delete.append((symbol, the_date))
            # print(f'Deleting {symbol} @ {the_date}')
            # del stocks[symbol]
            continue
        df['vwap'] = VolumeWeightedAveragePrice(
            high=df.high,
            low=df.low,
            close=df.close,
            volume=df.volume
        ).vwap
        df['ema'] = EMAIndicator(
            close=df.close,
        ).ema_indicator()

        df['rsi'] = RSIIndicator(
            close=df.close
        ).rsi()

        for i in range(len(df)):
            row = df.iloc[i]
            in_the_same_hour_minute = (pd.to_datetime(buy_time) - pd.to_datetime(row.time)).total_seconds() < 60
            if in_the_same_hour_minute:
                break
            rsi_1m_before = row.rsi
        node['ema'] = row.ema
        node['vwap'] = row.vwap
        node['rsi_1m_before'] = rsi_1m_before

for symbol, the_date in stocks_to_delete:
    del performance[the_date][symbol]

with open(state_file, 'w') as f:
    json.dump(state, f)
