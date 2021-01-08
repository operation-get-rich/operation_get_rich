import os
from datetime import timedelta

import pandas
from pandas import DataFrame

from directories import DATA_DIR
from utils import create_dir

SEGMENTER_NAME = os.path.basename(__file__).split('.')[0]  # remove .py
read_dir = f'polygon_stock_prices'
write_dir = f'polygon_{SEGMENTER_NAME}'

RAW_STOCK_PRICE_DIR = os.path.join(DATA_DIR, read_dir)
SEGMENTER_SAVE_DIR = os.path.join(DATA_DIR, write_dir)
create_dir(SEGMENTER_SAVE_DIR)

SEGMENTER_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def _write_segment_to_csv(index, stock_price_df, ticker, time):
    date_dir = os.path.join(SEGMENTER_SAVE_DIR, f'{time.date()}')

    create_dir(date_dir)
    stock_file_path = os.path.join(date_dir, f'{ticker}_{time.date()}.csv')

    selected_df = stock_price_df[stock_price_df['ticker'] == ticker]

    start_time = time.replace(hour=0, minute=0)
    end_time = time.replace(hour=16, minute=0)
    date_mask = (selected_df['time'] > start_time) & (selected_df['time'] <= end_time)
    selected_df = selected_df.loc[date_mask]

    selected_df.to_csv(stock_file_path)


def early_day_gap_parallel(
        gap_up_threshold=0.10,
        volume_threshold=1e06,
):
    stock_price_df = (
        pandas.read_csv(
            f'{RAW_STOCK_PRICE_DIR}/stocks.csv',
            parse_dates=['time'],
            keep_default_na=False,
            index_col=0
        )
    )  # type: DataFrame

    prev_day_close_price = None
    cumulative_volume = 0
    for index in range(len(stock_price_df.index)):
        (ticker,
         time,
         open_price,
         close_price,
         low_price,
         high_price,
         volume) = stock_price_df.loc[index]

        if index + 1 < len(stock_price_df.index):
            (next_ticker,
             next_time,
             next_open_price,
             next_close_price,
             next_low_price,
             next_high_price,
             next_volume) = stock_price_df.loc[index + 1]

            if next_ticker != ticker:
                prev_day_close_price = None
                cumulative_volume = 0
                continue
            if next_time.date() > time.date():
                pre_market_index = index + 1
                prev_day_close_price = close_price
                cumulative_volume = 0
                continue

        if prev_day_close_price is not None:
            cumulative_volume += volume

        if prev_day_close_price is not None and (time.hour, time.minute) == (9, 29):
            is_price_gapped_up = (open_price / prev_day_close_price) - 1 > gap_up_threshold

            if is_price_gapped_up and cumulative_volume > volume_threshold:
                print("Gapped Up: ", ticker, time.date(), cumulative_volume, flush=True)
                _write_segment_to_csv(pre_market_index, stock_price_df, ticker, time)
            prev_day_close_price = None
            cumulative_volume = 0


early_day_gap_parallel()
