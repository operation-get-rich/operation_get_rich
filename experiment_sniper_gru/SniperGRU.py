from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SniperGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=1):
        super(SniperGRU, self).__init__()

        self.input_size = input_size  # 8 (open, close, low, high, volume, vwap, ema, rsi)
        self.hidden_size = hidden_size  # 35
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(
            self,
            input,  # type: Union[PackedSequence, Tensor]
    ):
        """
        During training input is a `PackedSequence` instance.
        PackedSequence will ensure the model only perform computation on non-padded data for each batch

        During evaluation input is a tensor with size of 1 x sequence_length x feature_length
        """

        _, hidden = self.gru(input)
        # hidden (the last `num_layers` layers of each batch): (num_layers, batch_size, hidden_size)

        fc_output = self.fc(hidden[-1])
        sigmoid_output = self.sigmoid(fc_output).squeeze(-1)

        return sigmoid_output
