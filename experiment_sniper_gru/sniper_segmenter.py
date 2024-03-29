import argparse
import datetime
import os
import random

import pandas
from pandas import DataFrame

from directories import DATA_DIR
from utils import create_dir

parser = argparse.ArgumentParser('Sniper Segmenter')

parser.add_argument('--sd', required=True, type=str, help='Save Directory')
parser.add_argument('--ds', default='polygon_early_day_gap_segmenter_parallel', type=str, help='Data Source')
parser.add_argument('--profit', default=.01, type=float, help='Profit Goal')
parser.add_argument('--balance', default=True, type=bool, help='Whether we balanced sampling')
args = parser.parse_args()

#
ROOT_DATA_DIR = f'{DATA_DIR}/{args.ds}'
SNIPER_TRAINING_DATA_SAVE_DIR = f'{DATA_DIR}/{args.sd}'

PROFIT_DELTA = args.profit
BALANCED_SAMPLING = args.balance

for the_date in os.listdir(ROOT_DATA_DIR):
    for stock_file in os.listdir(os.path.join(ROOT_DATA_DIR, the_date)):
        stock_df = pandas.read_csv(
            os.path.join(ROOT_DATA_DIR, the_date, stock_file),
            parse_dates=['time']
        )
        if len(stock_df) == 0:
            continue
        market_open = stock_df.iloc[0].time.replace(hour=9, minute=30)
        profittable_indexes = stock_df[stock_df.time >= market_open][
            stock_df.close >= stock_df.open * (1 + PROFIT_DELTA)].index

        unprofittable_indexes = set(stock_df[stock_df.time >= market_open].index) - set(profittable_indexes)

        if BALANCED_SAMPLING:
            unprofittable_indexes = random.sample(unprofittable_indexes, len(profittable_indexes))

        create_dir(SNIPER_TRAINING_DATA_SAVE_DIR)
        for index in profittable_indexes:
            profittable_df = stock_df.loc[:index - 1]  # type: DataFrame
            profittable_df['label'] = 1
            stock_file_name_wout_csv = stock_file.split('.')[0]
            profittable_df.to_csv(
                f'{SNIPER_TRAINING_DATA_SAVE_DIR}/{stock_file_name_wout_csv}_{index}_1.csv',
                index=False
            )

        for index in unprofittable_indexes:
            unprofittable_df = stock_df.loc[:index - 1]  # type: DataFrame
            unprofittable_df['label'] = 0
            stock_file_name_wout_csv = stock_file.split('.')[0]
            unprofittable_df.to_csv(
                f'{SNIPER_TRAINING_DATA_SAVE_DIR}/{stock_file_name_wout_csv}_{index}_0.csv',
                index=False
            )
