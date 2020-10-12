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


class StockDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, data_folder, split='train', should_add_technical_indicator=False):
        self.segment_list = []
        company_directories = os.listdir(data_folder)
        for company in company_directories:
            segments = os.listdir(os.path.join(data_folder, company))
            self.segment_list += [os.path.join(data_folder, company, segment) for segment in segments]
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

        selected_segment_length = selected_segment_np.shape[0]
        if selected_segment_length < SEQUENCE_LENGTH:
            selected_segment_np = np.vstack(
                [selected_segment_np,
                 np.zeros((SEQUENCE_LENGTH - selected_segment_length,
                           selected_segment_np.shape[1]))]
            )

        return selected_segment_np, selected_segment_length

        # TODO: Add back once we have the mean and std
        # return (selected_segment_df - STOCK_DATA_MEAN) / STOCK_DATA_STD

    def __len__(self):
        return len(self.segment_list)
