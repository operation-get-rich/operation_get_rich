import json
import os
import random
from datetime import time

import numpy as np
import pandas
import torch.utils.data
from pandas.errors import ParserError
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volume import VolumeWeightedAveragePrice

from directories import PROJECT_ROOT_DIR, DATA_DIR

SEQUENCE_LENGTH = 390

MARKET_OPEN = time(hour=9, minute=30, second=0)

OPEN_COLUMN_INDEX = 0
CLOSE_COLUMN_INDEX = 1
LOW_COLUMN_INDEX = 2
HIGH_COLUMN_INDEX = 3
VOLUME_COLUMN_INDEX = 4
IS_MARKET_OPEN_INDEX = 5
VWAP_COLUMN_INDEX = 6
EMA_COLUMN_INDEX = 7
RSI_COLUMN_INDEX = 8


class WaveNetDataset(torch.utils.data.Dataset):
    TECHNICAL_INDICATOR_PERIOD = 14

    def __init__(self, dataset_name, split):
        self.dataset_name = dataset_name
        self.segment_list = []

        date_directories = os.listdir(os.path.join(DATA_DIR, dataset_name))

        for date in date_directories:
            try:
                date_dir = os.path.join(DATA_DIR, dataset_name, date)
                segments = os.listdir(date_dir)
            except NotADirectoryError:
                continue

            for segment in segments:
                # sometimes '.DS_STORE' is randomly created
                if segment[0] == '.':
                    continue
                self.segment_list.append(os.path.join(date_dir, segment)
                                         )

        random.seed(69420)
        random.shuffle(self.segment_list)

        train_proportion_split = .90
        if split == 'train':
            self.segment_list = self.segment_list[:int(len(self.segment_list) * train_proportion_split)]
        elif split == 'valid':
            self.segment_list = self.segment_list[int(len(self.segment_list) * train_proportion_split):]

    def __getitem__(self, index):
        selected_segment = self.segment_list[index]
        selected_segment_df = pandas.read_csv(
            filepath_or_buffer=selected_segment,
            parse_dates=['time']
        )

        anchor_datetime = selected_segment_df.iloc[0]['time']
        market_open_datetime = anchor_datetime.replace(hour=9, minute=30)

        selected_segment_df['is_market_open'] = selected_segment_df['time'] >= market_open_datetime
        selected_segment_df['is_market_open'] = selected_segment_df['is_market_open'].astype(float)

        selected_segment_df = selected_segment_df.drop(labels=['ticker', 'time'], axis=1)

        selected_segment_df['vwap'] = VolumeWeightedAveragePrice(
            high=selected_segment_df.high,
            low=selected_segment_df.low,
            close=selected_segment_df.close,
            volume=selected_segment_df.volume,
            n=self.TECHNICAL_INDICATOR_PERIOD
        ).vwap
        selected_segment_df['ema'] = EMAIndicator(
            close=selected_segment_df.close,
            n=self.TECHNICAL_INDICATOR_PERIOD
        ).ema_indicator()

        selected_segment_df['rsi'] = RSIIndicator(
            close=selected_segment_df.close,
            n=self.TECHNICAL_INDICATOR_PERIOD
        ).rsi()

        selected_segment_np = selected_segment_df[self.TECHNICAL_INDICATOR_PERIOD:].to_numpy()

        # The convolution based model accept data with the features in the first axis
        selected_segment_np = np.transpose(selected_segment_np)  # shape: 9, ~390

        selected_segment_length = selected_segment_np.shape[1]
        if selected_segment_length < SEQUENCE_LENGTH:
            selected_segment_np = np.hstack(
                [
                    selected_segment_np,
                    np.zeros(
                        (
                            selected_segment_np.shape[0],
                            SEQUENCE_LENGTH - selected_segment_length
                        )
                    )
                ]
            )

            return selected_segment_np, selected_segment_length

        else:
            selected_segment_np = selected_segment_np[:, 0:SEQUENCE_LENGTH]

            return selected_segment_np, SEQUENCE_LENGTH

    def __len__(self):
        return len(self.segment_list)


class PercentChangeNormalizer:
    @classmethod
    def normalize_volume(cls, data, dataset_name):
        with open(os.path.join(PROJECT_ROOT_DIR, 'dataset_statistics.json'), 'r') as f:
            stats = json.load(f)

        assert dataset_name in stats, f"{dataset_name} statistics is not found yet"

        data = np.copy(data)
        data[:, VOLUME_COLUMN_INDEX] -= stats[dataset_name]['volume']['mean']
        data[:, VOLUME_COLUMN_INDEX] /= stats[dataset_name]['volume']['std']
        return torch.Tensor(data)

    @classmethod
    def normalize_price_into_percent_change(cls, data):
        data = np.copy(data)
        anchor_open_price = data[0, OPEN_COLUMN_INDEX]
        anchor_close_price = data[0, CLOSE_COLUMN_INDEX]
        anchor_low_price = data[0, LOW_COLUMN_INDEX]
        anchor_high_price = data[0, HIGH_COLUMN_INDEX]
        anchor_vwap = data[0, VWAP_COLUMN_INDEX]
        anchor_ema = data[0, EMA_COLUMN_INDEX]

        for i in range(0, data.shape[0]):
            data[i, OPEN_COLUMN_INDEX] = cls._compute_percent_change(
                anchor_open_price,
                data[i, OPEN_COLUMN_INDEX])

            data[i, CLOSE_COLUMN_INDEX] = cls._compute_percent_change(
                anchor_close_price,
                data[i, CLOSE_COLUMN_INDEX])

            data[i, LOW_COLUMN_INDEX] = cls._compute_percent_change(
                anchor_low_price,
                data[i, LOW_COLUMN_INDEX])

            data[i, HIGH_COLUMN_INDEX] = cls._compute_percent_change(
                anchor_high_price,
                data[i, HIGH_COLUMN_INDEX])

            data[i, VWAP_COLUMN_INDEX] = cls._compute_percent_change(
                anchor_vwap,
                data[i, VWAP_COLUMN_INDEX])

            data[i, EMA_COLUMN_INDEX] = cls._compute_percent_change(
                anchor_ema,
                data[i, EMA_COLUMN_INDEX])
        return torch.Tensor(data)

    @classmethod
    def _compute_percent_change(cls, p1, p2):
        return (p2 / p1) - 1
