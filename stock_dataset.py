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


class PercentChangeNormalizer:
    @classmethod
    def normalize_volume(cls, data):
        data = np.copy(data)
        for batch_index in range(len(data)):
            current_batch_data = data[batch_index]
            current_batch_data[:, VOLUME_COLUMN_INDEX] -= VOLUME_MEAN
            current_batch_data[:, VOLUME_COLUMN_INDEX] /= VOLUME_STD
        return torch.Tensor(data)

    @classmethod
    def normalize_price_into_percent_change(cls, data):
        data = np.copy(data)
        for batch_index in range(len(data)):
            current_batch_data = data[batch_index]
            anchor_open_price = current_batch_data[0, OPEN_COLUMN_INDEX]
            anchor_close_price = current_batch_data[0, CLOSE_COLUMN_INDEX]
            anchor_low_price = current_batch_data[0, LOW_COLUMN_INDEX]
            anchor_high_price = current_batch_data[0, HIGH_COLUMN_INDEX]
            anchor_vwap = current_batch_data[0, VWAP_COLUMN_INDEX]
            anchor_ema = current_batch_data[0, EMA_COLUMN_INDEX]

            for i in range(0, current_batch_data.shape[0]):
                current_batch_data[i, OPEN_COLUMN_INDEX] = cls._compute_percent_change(
                    anchor_open_price,
                    current_batch_data[i, OPEN_COLUMN_INDEX])

                current_batch_data[i, CLOSE_COLUMN_INDEX] = cls._compute_percent_change(
                    anchor_close_price,
                    current_batch_data[i, CLOSE_COLUMN_INDEX])

                current_batch_data[i, LOW_COLUMN_INDEX] = cls._compute_percent_change(
                    anchor_low_price,
                    current_batch_data[i, LOW_COLUMN_INDEX])

                current_batch_data[i, HIGH_COLUMN_INDEX] = cls._compute_percent_change(
                    anchor_high_price,
                    current_batch_data[i, HIGH_COLUMN_INDEX])

                current_batch_data[i, VWAP_COLUMN_INDEX] = cls._compute_percent_change(
                    anchor_vwap,
                    current_batch_data[i, VWAP_COLUMN_INDEX])

                current_batch_data[i, EMA_COLUMN_INDEX] = cls._compute_percent_change(
                    anchor_ema,
                    current_batch_data[i, EMA_COLUMN_INDEX])
        return torch.Tensor(data)

    @classmethod
    def _compute_percent_change(cls, p1, p2):
        return (p2 / p1) - 1


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

        if split == 'test':
            pass
            # TODO: Pick the last month stock data whatever
            return

        random.shuffle(self.segment_list)

        if split == 'train':
            self.segment_list = self.segment_list[:int(len(self.segment_list) * .90)]
        elif split == 'valid':
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

    def __len__(self):
        return len(self.segment_list)
