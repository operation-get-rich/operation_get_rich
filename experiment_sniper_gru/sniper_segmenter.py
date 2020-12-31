import datetime
import os
import random

import pandas
from pandas import DataFrame

from directories import DATA_DIR
from utils import create_dir

root_data_dir = f'{DATA_DIR}/polygon_early_day_gap_segmenter_parallel'

profit_delta = .01

for the_date in os.listdir(root_data_dir):
    for stock_file in os.listdir(os.path.join(root_data_dir, the_date)):
        stock_df = pandas.read_csv(
            os.path.join(root_data_dir, the_date, stock_file),
            parse_dates=['time']
        )
        market_open = stock_df.iloc[0].time.replace(hour=9, minute=30)
        profittable_indexes = stock_df[stock_df.time >= market_open][
            stock_df.close >= stock_df.open * (1 + profit_delta)].index

        unprofittable_indexes = set(stock_df[stock_df.time >= market_open].index) - set(profittable_indexes)
        unprofittable_indexes = random.sample(unprofittable_indexes, len(profittable_indexes))

        sniper_training_data_dir = f'{DATA_DIR}/sniper_training_data'
        create_dir(sniper_training_data_dir)
        for index in profittable_indexes:
            profittable_df = stock_df.loc[:index - 1]  # type: DataFrame
            profittable_df['label'] = 1
            stock_file_name_wout_csv = stock_file.split('.')[0]
            profittable_df.to_csv(
                f'{sniper_training_data_dir}/{stock_file_name_wout_csv}_{index}_1.csv',
                index=False
            )

        for index in unprofittable_indexes:
            unprofittable_df = stock_df.loc[:index - 1]  # type: DataFrame
            unprofittable_df['label'] = 0
            stock_file_name_wout_csv = stock_file.split('.')[0]
            unprofittable_df.to_csv(
                f'{sniper_training_data_dir}/{stock_file_name_wout_csv}_{index}_0.csv',
                index=False
            )
