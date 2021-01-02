import json
import os
import random

import numpy as np
import pandas
import torch.utils.data
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volume import VolumeWeightedAveragePrice

from directories import PROJECT_ROOT_DIR

TECHNICAL_INDICATOR_PERIOD = 14
SEQUENCE_LENGTH = 390

OPEN_COLUMN_INDEX = 0
CLOSE_COLUMN_INDEX = 1
LOW_COLUMN_INDEX = 2
HIGH_COLUMN_INDEX = 3
VOLUME_COLUMN_INDEX = 4
VWAP_COLUMN_INDEX = 5
EMA_COLUMN_INDEX = 6
RSI_COLUMN_INDEX = 7


class SniperDataset(torch.utils.data.Dataset):
    TECHNICAL_INDICATOR_PERIOD = 14
    SEQUENCE_LENGTH = 390
    TRUE_DATASET_NAME = 'polygon_early_day_gap_segmenter_parallel'

    def __init__(self, split, segment_data_dir):
        self.segment_data_dir = segment_data_dir
        self.segment_list = os.listdir(self.segment_data_dir)

        random.seed(69420)
        random.shuffle(self.segment_list)

        train_proportion_split = .90
        if split == 'train':
            self.segment_list = self.segment_list[:int(len(self.segment_list) * train_proportion_split)]
        elif split == 'valid':
            self.segment_list = self.segment_list[int(len(self.segment_list) * train_proportion_split):]

    def __getitem__(self, index):
        selected_segment = self.segment_list[index]
        selected_segment_full_path = f'{self.segment_data_dir}/{selected_segment}'

        selected_segment_df = pandas.read_csv(filepath_or_buffer=selected_segment_full_path)
        selected_segment_df = selected_segment_df.drop(labels=['ticker', 'time'], axis=1)

        selected_label = selected_segment_df.label.iloc[0]

        # TODO: Perhaps remove the label from the raw data.. it's kind of scary
        selected_segment_df = selected_segment_df.drop(labels=['label'], axis=1)

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

        selected_segment_np = PercentChangeNormalizer.normalize_price_into_percent_change(selected_segment_np)
        # TODO: Should we change the way we normalize volume?
        #  Find technique of normalization that keeps updating whenever new data comes in
        selected_segment_np = PercentChangeNormalizer.normalize_volume(
            selected_segment_np,
            dataset_name=self.TRUE_DATASET_NAME
        )

        selected_segment_length = selected_segment_np.shape[0]
        if selected_segment_length < self.SEQUENCE_LENGTH:
            selected_segment_np = np.vstack(
                [selected_segment_np,
                 np.zeros((self.SEQUENCE_LENGTH - selected_segment_length,
                           selected_segment_np.shape[1]))]
            )
            return selected_segment_np, selected_label, selected_segment_length
        else:
            max_start = selected_segment_length - self.SEQUENCE_LENGTH
            start = random.randint(0, max_start)
            selected_segment_np = selected_segment_np[start: start + self.SEQUENCE_LENGTH, :]

            return selected_segment_np, selected_label, self.SEQUENCE_LENGTH

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
        return data

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
        return data

    @classmethod
    def _compute_percent_change(cls, p1, p2):
        return (p2 / p1) - 1
