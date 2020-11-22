import os

import pandas

from segmenters.early_day_gap_segmenter_parallel.early_day_gap_segmenter_parallel import RAW_STOCK_PRICE_DIR, _write_segment_to_csv
from utils import timeit, create_dir


@timeit
def early_day_gap_refactor(
        gap_up_threshold=0.10,
        volume_threshold=0
):
    save_dir_name = './{}_{}_{}'.format('early_day_gap_refactor', gap_up_threshold, volume_threshold)
    create_dir(save_dir_name)
    raw_stock_list = os.listdir(RAW_STOCK_PRICE_DIR)
    for stock_file in sorted(raw_stock_list):
        segment_stock(stock_file, save_dir_name, gap_up_threshold, volume_threshold)


def segment_stock(stock_file, save_dir_name, gap_up_threshold, volume_threshold):
    stock_path = os.path.join(RAW_STOCK_PRICE_DIR, stock_file)
    print("Loading Data: {}".format(stock_path), flush=True)
    stock_price_df = pandas.read_csv(stock_path, parse_dates=['time'], index_col=0)  # type: DataFrame
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
                prev_day_close_price = close_price
                continue

        if prev_day_close_price is not None:
            cumulative_volume += volume

        if prev_day_close_price is not None and (time.hour, time.minute) == (9, 30):
            is_price_gapped_up = (open_price / prev_day_close_price) - 1 > gap_up_threshold

            if is_price_gapped_up and cumulative_volume > volume_threshold:
                print("Gapped Up: ", ticker, time.date(), cumulative_volume, flush=True)
                _write_segment_to_csv(index, save_dir_name, stock_price_df, ticker, time)
            prev_day_close_price = None
            cumulative_volume = 0