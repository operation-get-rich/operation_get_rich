import torch
import torch.nn as nn


# wavenet
# from https://www.kaggle.com/hanjoonchoe/wavenet-lstm-pytorch-ignite-ver
from torch.autograd.grad_mode import F


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


class WaveNetModel(nn.Module):
    def __init__(self, feature_length=8, kernel_size=3, hard=False):
        super().__init__()
        self.wave_block1 = Wave_Block(in_channels=feature_length, out_channels=16, dilation_rates=12,
                                      kernel_size=kernel_size)
        self.wave_block2 = Wave_Block(in_channels=16, out_channels=32, dilation_rates=8, kernel_size=kernel_size)
        self.wave_block3 = Wave_Block(in_channels=32, out_channels=64, dilation_rates=4, kernel_size=kernel_size)
        self.wave_block4 = Wave_Block(in_channels=64, out_channels=128, dilation_rates=1, kernel_size=kernel_size)
        self.action_layer = nn.Linear(
            in_features=128,
            out_features=1
        )
        self.fc = nn.Linear(in_features=128, out_features=1)
        self.hard = hard

    def forward(self, x):
        x = self.wave_block1(x)
        x = self.wave_block2(x)
        x = self.wave_block3(x)

        x = self.wave_block4(x)
        x = x.permute(0, 2, 1)

        curr_out = self.fc(x)
        curr_out = torch.tanh(curr_out)

        action_logit = self.action_layer(x)
        temperature = 0.05
        action = self._gumbel_softmax_sample(action_logit, temperature, hard=self.hard)
        curr_out = curr_out * action

        return curr_out

    def _gumbel_softmax_sample(self, logits, temperature, hard=False, deterministic=False, eps=1e-20):

        if deterministic:
            if logits.shape[-1] == 1:
                return F.sigmoid(logits)
            else:
                return F.softmax(logits, dim=-1)

        # Stochastic
        if logits.shape[-1] == 1:
            noise = torch.rand_like(logits)
            y = (logits + torch.log(noise + eps) - torch.log(1 - noise + eps))
            y = torch.sigmoid(y / temperature)
            if hard:
                return (y > 0.5).float()
            else:
                return y
        else:
            y = logits + self.sample_gumbel(logits.size())
            y = F.softmax(y / temperature, dim=-1)
            if hard:
                return (y > 0.5).float()
            else:
                return y


def ProfitLoss(trade_sequence, market_sequence, is_premarket=None):
    trade_sequence = trade_sequence[:-1]
    market_sequence = market_sequence[1:]

    assert trade_sequence.shape[0] == market_sequence.shape[0]

    time_length = trade_sequence.shape[0]
    capital = 1
    shares = 0
    for t in range(time_length):
        if is_premarket != None:
            if is_premarket[t]:
                continue

        current_trade = trade_sequence[t]
        current_price = market_sequence[t]
        if current_trade > 0:  # buying
            bought_shares = (current_trade * capital) / current_price
            shares = shares + bought_shares
            capital = capital - (bought_shares * current_price)
        elif current_trade < 0:  # selling
            sold_shares = shares * current_trade  # sold_shares < 0
            shares = shares + sold_shares
            capital = capital - (sold_shares * current_price)

    final_price = market_sequence[-1]
    capital = capital + (shares * final_price)

    # Return negative reward
    return -(capital - 1)
