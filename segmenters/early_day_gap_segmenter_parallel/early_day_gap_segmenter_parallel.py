import csv
import datetime
import json
import os
import sys
from multiprocessing.pool import Pool

import pandas
from pandas import DataFrame

from definitions import PROJECT_ROOT_DIR
from utils import create_dir, DATETIME_FORMAT

SEGMENTER_NAME = os.path.basename(__file__).split('.')[0]  # remove .py
read_dir = f'polygon_stock_prices'
write_dir = f'polygon_{SEGMENTER_NAME}'

# read_dir = 'test_stock_prices'
# write_dir = f'test_stock_prices_{SEGMENTER_NAME}'

RAW_STOCK_PRICE_DIR = os.path.join(PROJECT_ROOT_DIR, 'datas', read_dir)
SEGMENTER_SAVE_DIR = os.path.join(PROJECT_ROOT_DIR, 'datas', write_dir)
create_dir(SEGMENTER_SAVE_DIR)

SEGMENTER_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE_LOCATION = os.path.join(SEGMENTER_ROOT_DIR, f'{SEGMENTER_NAME}_state.json')
CONSOLE_OUTPUT_LOCATION = os.path.join(SEGMENTER_ROOT_DIR, f'{SEGMENTER_NAME}_console_output.txt')
EXCEPTION_OUTPUT_LOCATION = os.path.join(SEGMENTER_ROOT_DIR, f'{SEGMENTER_NAME}_exception.txt')

FINISHED_FILEPATHS_KEY = 'finished_filepaths'


def _write_segment_to_csv(index, stock_price_df, ticker, time):
    gaped_up_date = time.date()
    date_dir = os.path.join(SEGMENTER_SAVE_DIR, f'{time.date()}')
    create_dir(date_dir)
    stock_file_path = os.path.join(date_dir, f'{ticker}_{time.date()}.csv')
    with open(stock_file_path, 'w') as the_file:
        writer = csv.writer(the_file)
        while index < len(stock_price_df) and stock_price_df.loc[index]['time'].date() == gaped_up_date:
            writer.writerow(stock_price_df.loc[index])
            index += 1


def _segment_stock(stock_path, gap_up_threshold, volume_threshold):
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
                _write_segment_to_csv(pre_market_index, stock_price_df, ticker, time)
            prev_day_close_price = None
            cumulative_volume = 0


def segment_stock_parallel(args):
    orig_stdout = sys.stdout
    with open(CONSOLE_OUTPUT_LOCATION, 'a') as console_output_file:
        sys.stdout = console_output_file
        stock_path, gap_up_threshold, volume_threshold = args

        print("Loading Data: {}".format(stock_path), flush=True)
        try:
            _segment_stock(stock_path, gap_up_threshold, volume_threshold)
            print(f'Finished Segmenting {stock_path}')
            update_state(update_finished_filepaths, stock_path)
        except Exception as exc:
            with open(EXCEPTION_OUTPUT_LOCATION, 'a') as f:
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


def update_finished_filepaths(state, stock_path):
    if FINISHED_FILEPATHS_KEY in state:
        state[FINISHED_FILEPATHS_KEY].append(stock_path)
    else:
        state[FINISHED_FILEPATHS_KEY] = [stock_path]
    state[FINISHED_FILEPATHS_KEY] = sorted(state[FINISHED_FILEPATHS_KEY])


def update_state(update_func=None, *args, **kwargs):
    with open(STATE_FILE_LOCATION, 'r') as state_file:
        state = json.load(state_file)

    if update_func:
        update_func(state, *args, **kwargs)

    if '_meta' in state:
        if 'first_state_update_call' not in state['_meta']:
            state['_meta']['first_state_update_call'] = datetime.datetime.now().strftime(DATETIME_FORMAT)
        state['_meta']['latest_state_update_call'] = datetime.datetime.now().strftime(DATETIME_FORMAT)

    with open(STATE_FILE_LOCATION, 'w') as state_file:
        json.dump(state, state_file)


def early_day_gap_parallel(
        gap_up_threshold=0.10,
        volume_threshold=1e06,
):
    raw_stock_list = os.listdir(RAW_STOCK_PRICE_DIR)
    stock_paths = set([os.path.join(RAW_STOCK_PRICE_DIR, s) for s in raw_stock_list])

    args = []

    state_file_location = STATE_FILE_LOCATION
    with open(state_file_location) as state_file:
        state = json.load(state_file)

    stock_paths -= set(state[FINISHED_FILEPATHS_KEY])

    for stock_path in stock_paths:
        args.append(
            (stock_path,
             gap_up_threshold,
             volume_threshold)
        )

    with Pool() as pool:
        pool.map(segment_stock_parallel, args)


early_day_gap_parallel()
