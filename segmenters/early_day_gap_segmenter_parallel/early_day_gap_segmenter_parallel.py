import csv
import os
import sys
from multiprocessing.pool import Pool

import pandas
from pandas import DataFrame

from utils import create_dir, get_current_filename, get_current_directory


def _write_segment_to_csv(index, root_save_dir_name, stock_price_df, ticker, time):
    gaped_up_date = time.date()
    date_dir = f'{root_save_dir_name}/{time.date()}'
    create_dir(date_dir)
    filename = f'{date_dir}/{ticker}_{time.date()}.csv'
    with open(filename, 'w') as the_file:
        writer = csv.writer(the_file)

        while index < len(stock_price_df) and stock_price_df.loc[index]['time'].date() == gaped_up_date:
            writer.writerow(stock_price_df.loc[index])
            index += 1


def _segment_stock(stock_path, root_save_dir_name, gap_up_threshold, volume_threshold):
    stock_price_df = (
        pandas.read_csv(stock_path, parse_dates=['time'], keep_default_na=False, index_col=0)
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
                continue

        if prev_day_close_price is not None:
            cumulative_volume += volume

        if prev_day_close_price is not None and (time.hour, time.minute) == (9, 30):
            is_price_gapped_up = (open_price / prev_day_close_price) - 1 > gap_up_threshold

            if is_price_gapped_up and cumulative_volume > volume_threshold:
                print("Gapped Up: ", ticker, time.date(), cumulative_volume, flush=True)
                _write_segment_to_csv(pre_market_index, root_save_dir_name, stock_price_df, ticker, time)
            prev_day_close_price = None
            cumulative_volume = 0


def segment_stock_parallel(args):
    current_filename = get_current_filename(__file__).split('.')[0]  # remove .py extension
    current_directory = get_current_directory(__file__)
    orig_stdout = sys.stdout

    with open(f'{current_directory}/{current_filename}_console_output.txt', 'a') as console_output_file:
        sys.stdout = console_output_file
        stock_file, root_save_dir_name, gap_up_threshold, volume_threshold = args
        stock_path = os.path.join(RAW_STOCK_PRICE_DIR, stock_file)

        print("Loading Data: {}".format(stock_path), flush=True)
        try:
            _segment_stock(stock_path, root_save_dir_name, gap_up_threshold, volume_threshold)
            print(f'Finished Segmenting {stock_path}')
        except Exception as exc:
            with open('segmenters/early_day_gap_segmenter_parallel_exception.txt', 'a') as f:
                print('failed_to_segment_stock',
                      dict(
                          stock_path=stock_path,
                          exception=exc,
                      ),
                      file=f
                      )
            sys.stdout = orig_stdout
            return
        sys.stdout = orig_stdout


def early_day_gap_parallel(
        gap_up_threshold=0.10,
        volume_threshold=1e06,
):
    raw_stock_list = os.listdir(RAW_STOCK_PRICE_DIR)
    sorted_stock_files = sorted(raw_stock_list)

    save_dir_name = './{}'.format(
        early_day_gap_parallel.__name__,
    )
    create_dir(save_dir_name)

    args = []

    for stock_file in sorted_stock_files:
        args.append(
            (stock_file,
             save_dir_name,
             gap_up_threshold,
             volume_threshold)
        )

    with Pool() as pool:
        pool.map(segment_stock_parallel, args)


RAW_STOCK_PRICE_DIR = 'stock_prices_polygon'
early_day_gap_parallel()
