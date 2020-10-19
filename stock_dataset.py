import numpy as np
import os
import random

import pandas
import torch.utils.data

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


class StockDataset(torch.utils.data.Dataset):
    """
    Responsible to return batches of `gaped_up_stocks_early_volume_1e5_gap_10` stock data segments.
    """

    def __init__(self, data_folder, split, should_add_technical_indicator=False):
        self.segment_list = []
        company_directories = os.listdir(data_folder)
        for company in company_directories:
            segments = os.listdir(os.path.join(data_folder, company))
            for segment in segments:
                self.segment_list.append(os.path.join(data_folder, company, segment))
        random.seed(69420)
        random.shuffle(self.segment_list)

        if split == 'train':
            self.segment_list = self.segment_list[:int(len(self.segment_list) * .90)]
        elif split == 'test':
            self.segment_list = self.segment_list[int(len(self.segment_list) * .90):]

        self.should_add_technical_indicator = should_add_technical_indicator

    def __getitem__(self, index):
        selected_segment = self.segment_list[index]
        selected_segment_df = pandas.read_csv(filepath_or_buffer=selected_segment)
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

        selected_segment_np = selected_segment_df[TECHNICAL_INDICATOR_PERIOD:].to_numpy()

        self._normalize_price_data_using_percent_change(selected_segment_np)
        self._normalize_volume(selected_segment_np)

        selected_segment_length = selected_segment_np.shape[0]
        if selected_segment_length < SEQUENCE_LENGTH:
            selected_segment_np = np.vstack(
                [selected_segment_np,
                 np.zeros((SEQUENCE_LENGTH - selected_segment_length,
                           selected_segment_np.shape[1]))]
            )
            return selected_segment_np, selected_segment_length
        else:
            max_start = selected_segment_length - SEQUENCE_LENGTH
            start = random.randint(0, max_start)
            selected_segment_np = selected_segment_np[start: start + SEQUENCE_LENGTH, :]

            return selected_segment_np, SEQUENCE_LENGTH

    def _normalize_volume(self, selected_segment_np):
        selected_segment_np[:, VOLUME_COLUMN_INDEX] -= VOLUME_MEAN
        selected_segment_np[:, VOLUME_COLUMN_INDEX] /= VOLUME_STD

    def _normalize_price_data_using_percent_change(self, selected_segment_np):
        anchor_open_price = selected_segment_np[0, OPEN_COLUMN_INDEX]
        anchor_close_price = selected_segment_np[0, CLOSE_COLUMN_INDEX]
        anchor_low_price = selected_segment_np[0, LOW_COLUMN_INDEX]
        anchor_high_price = selected_segment_np[0, HIGH_COLUMN_INDEX]
        anchor_vwap = selected_segment_np[0, VWAP_COLUMN_INDEX]
        anchor_ema = selected_segment_np[0, EMA_COLUMN_INDEX]

        for i in range(1, selected_segment_np.shape[0]):
            selected_segment_np[i, OPEN_COLUMN_INDEX] = self._compute_percent_change(
                anchor_open_price,
                selected_segment_np[i, OPEN_COLUMN_INDEX])

            selected_segment_np[i, CLOSE_COLUMN_INDEX] = self._compute_percent_change(
                anchor_close_price,
                selected_segment_np[i, CLOSE_COLUMN_INDEX])

            selected_segment_np[i, LOW_COLUMN_INDEX] = self._compute_percent_change(
                anchor_low_price,
                selected_segment_np[i, LOW_COLUMN_INDEX])

            selected_segment_np[i, HIGH_COLUMN_INDEX] = self._compute_percent_change(
                anchor_high_price,
                selected_segment_np[i, HIGH_COLUMN_INDEX])

            selected_segment_np[i, VWAP_COLUMN_INDEX] = self._compute_percent_change(
                anchor_vwap,
                selected_segment_np[i, VWAP_COLUMN_INDEX])

            selected_segment_np[i, EMA_COLUMN_INDEX] = self._compute_percent_change(
                anchor_ema,
                selected_segment_np[i, EMA_COLUMN_INDEX])

    def _compute_percent_change(self, p1, p2):
        return (p2 / p1) - 1

    def __len__(self):
        return len(self.segment_list)
