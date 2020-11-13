import csv
import os
from statistics import mean

import pandas as pd
import torch
from ta.trend import EMAIndicator
from ta.volume import VolumeWeightedAveragePrice

from TraderGRU import TraderGRU

model = TraderGRU(
    input_size=7,
    hidden_size=5 * 7
)
model.load_state_dict(
    torch.load(
        './Train_Trader_Early_Gap_Up_10AM-20201109-105754/best_model.pt',
        map_location=torch.device('cpu')
    )
)
model.eval()

gap_stocks_by_date_dir = './fron_unchecked_code/fron_gapped_up_stocks'


def compute_vwap(i, price_volume_products, volumes, ta_period=14):
    start_pos = i - (ta_period - 1)
    end_pos = i + 1
    sum_pv = sum(price_volume_products[start_pos:end_pos])
    sum_v = sum(volumes[start_pos:end_pos])
    vwap = sum_pv / sum_v
    return vwap


def compute_ema(close_prices, ta_period):
    close_prices_series = pd.Series(close_prices)
    ewm = close_prices_series.ewm(
        span=ta_period,
        min_periods=ta_period,
        adjust=False
    ).mean()  # ewm = exponential moving window
    return ewm.values[-1]


def main():
    for date in sorted(os.listdir(gap_stocks_by_date_dir)):
        date_dir = f'{gap_stocks_by_date_dir}/{date}'
        for stock_file in sorted(os.listdir(date_dir)):
            stock_df = pd.read_csv(f'{date_dir}/{stock_file}')
            stock_df = stock_df.drop(labels=['ticker', 'time'], axis=1)

            ta_period = 14

            volumes = []
            price_volume_products = []
            close_prices = []

            # TODO: Remove once debugging is done
            vwaps = []
            emas = []
            for i in range(len(stock_df)):
                (
                    open_price,
                    close_price,
                    low_price,
                    high_price,
                    volume
                ) = stock_df.loc[i]
                typical_price = mean([close_price, low_price, high_price])

                volumes.append(volume)
                price_volume_products.append(volume * typical_price)
                close_prices.append(close_price)

                able_to_compute_ta = i >= (ta_period - 1)
                if able_to_compute_ta:
                    vwap = compute_vwap(i, price_volume_products, volumes, ta_period)
                    vwaps.append(vwap)

                    ema = compute_ema(close_prices, ta_period)
                    emas.append(ema)

            stock_df['vwap'] = VolumeWeightedAveragePrice(
                high=stock_df.high,
                low=stock_df.low,
                close=stock_df.close,
                volume=stock_df.volume,
                n=14
            ).vwap
            stock_df['ema'] = EMAIndicator(
                close=stock_df.close,
                n=14
            ).ema_indicator()
            stock_np = stock_df[14:].to_numpy()
main()
