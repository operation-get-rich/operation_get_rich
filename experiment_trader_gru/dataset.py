import datetime

import numpy as np
import os
import random

import pandas
import torch.utils.data
from ta.momentum import RSIIndicator

from ta.trend import EMAIndicator
from ta.volume import VolumeWeightedAveragePrice

STOCK_DATA_MEAN = 0.0
STOCK_DATA_STD = 0.0
TECHNICAL_INDICATOR_PERIOD = 14
SEQUENCE_LENGTH = 390

# From gaped_up_stocks_early_volume_1e5_gap_10_statistics.csv
VOLUME_MEAN = 5236.079151848999
VOLUME_STD = 17451.76183378195

OPEN_COLUMN_INDEX = 0
CLOSE_COLUMN_INDEX = 1
LOW_COLUMN_INDEX = 2
HIGH_COLUMN_INDEX = 3
VOLUME_COLUMN_INDEX = 4
VWAP_COLUMN_INDEX = 5
EMA_COLUMN_INDEX = 6

MARKET_OPEN = datetime.time(hour=10, minute=0, second=0)


class TraderGRUDataSet(torch.utils.data.Dataset):
    def __init__(self, data_folder, split, should_add_technical_indicator=False):
        self.segment_list = []
        company_directories = os.listdir(data_folder)
        for company in company_directories:
            segments = os.listdir(os.path.join(data_folder, company))
            for segment in segments:
                self.segment_list.append(os.path.join(data_folder, company, segment))

        random.seed(69420)
        random.shuffle(self.segment_list)

        train_proportion_split = .90
        if split == 'train':
            self.segment_list = self.segment_list[:int(len(self.segment_list) * train_proportion_split)]
        elif split == 'valid':
            self.segment_list = self.segment_list[int(len(self.segment_list) * train_proportion_split):]

        self.should_add_technical_indicator = should_add_technical_indicator

    def __getitem__(self, index):
        selected_segment = self.segment_list[index]
        selected_segment_df = pandas.read_csv(filepath_or_buffer=selected_segment)

        # Get time
        time = pandas.to_datetime(selected_segment_df['time'])
        is_premarket = pandas.Series([val.time() < MARKET_OPEN for val in time])
        is_premarket = is_premarket[TECHNICAL_INDICATOR_PERIOD:].to_numpy().astype(float)

        selected_segment_df = selected_segment_df.drop(labels=['ticker', 'time'], axis=1)

        if self.should_add_technical_indicator:
            selected_segment_df['vwap'] = VolumeWeightedAveragePrice(
                high=selected_segment_df.high,
                low=selected_segment_df.low,
                close=selected_segment_df.close,
                volume=selected_segment_df.volume,
                n=TECHNICAL_INDICATOR_PERIOD
            ).vwap
            selected_segment_df['ema'] = EMAIndicator(
                close=selected_segment_df.close,
                n=TECHNICAL_INDICATOR_PERIOD
            ).ema_indicator()

            selected_segment_df['rsi'] = RSIIndicator(
                close=selected_segment_df.close,
                n=TECHNICAL_INDICATOR_PERIOD
            ).rsi()

        selected_segment_np = selected_segment_df[TECHNICAL_INDICATOR_PERIOD:].to_numpy()

        selected_segment_length = selected_segment_np.shape[0]
        if selected_segment_length < SEQUENCE_LENGTH:
            selected_segment_np = np.vstack(
                [selected_segment_np,
                 np.zeros((SEQUENCE_LENGTH - selected_segment_length,
                           selected_segment_np.shape[1]))]
            )

            is_premarket = np.hstack(
                [is_premarket,
                 np.zeros((SEQUENCE_LENGTH - selected_segment_length))]
            )

            return selected_segment_np, selected_segment_length, is_premarket

        else:
            selected_segment_np = selected_segment_np[0: SEQUENCE_LENGTH, :]
            is_premarket = is_premarket[0: SEQUENCE_LENGTH]

            return selected_segment_np, SEQUENCE_LENGTH, is_premarket

    def __len__(self):
        return len(self.segment_list)
