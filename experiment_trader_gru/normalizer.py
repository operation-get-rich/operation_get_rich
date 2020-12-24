import numpy as np
import torch

from experiment_trader_gru.dataset import VOLUME_COLUMN_INDEX, VOLUME_MEAN, VOLUME_STD, OPEN_COLUMN_INDEX, \
    CLOSE_COLUMN_INDEX, LOW_COLUMN_INDEX, HIGH_COLUMN_INDEX, VWAP_COLUMN_INDEX, EMA_COLUMN_INDEX


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