import torch
import torch.nn as nn


# wavenet
# from https://www.kaggle.com/hanjoonchoe/wavenet-lstm-pytorch-ignite-ver
class Wave_Block(nn.Module):

    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate * (kernel_size - 1)) / 2), dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate * (kernel_size - 1)) / 2), dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res


# detail
class WaveNetModel(nn.Module):
    def __init__(self, feature_length=8, kernel_size=3):
        super().__init__()
        # self.LSTM = nn.GRU(input_size=input_size, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.wave_block1 = Wave_Block(in_channels=feature_length, out_channels=16, dilation_rates=12, kernel_size=kernel_size)
        self.wave_block2 = Wave_Block(in_channels=16, out_channels=32, dilation_rates=8, kernel_size=kernel_size)
        self.wave_block3 = Wave_Block(in_channels=32, out_channels=64, dilation_rates=4, kernel_size=kernel_size)
        self.wave_block4 = Wave_Block(in_channels=64, out_channels=128, dilation_rates=1, kernel_size=kernel_size)
        self.fc = nn.Linear(in_features=128, out_features=11)

    def forward(self, x):
        x = self.wave_block1(x)
        x = self.wave_block2(x)
        x = self.wave_block3(x)

        x = self.wave_block4(x)
        # x, _ = self.LSTM(x)
        x = self.fc(x)
        return x