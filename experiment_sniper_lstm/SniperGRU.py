import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class SniperGRU(nn.Module):
    def __init__(self, input_size, hidden_size, hard=False):
        super(SniperGRU, self).__init__()

        self.input_size = input_size  # 7 (open, close, low, high, volume, vwap, ema)
        self.hidden_size = hidden_size  # 35

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.identity = torch.eye(input_size).cuda()
            self.zeros = Variable(torch.zeros(input_size).cuda())
        else:
            self.identity = torch.eye(input_size)
            self.zeros = Variable(torch.zeros(input_size))

        self.gru_cell = nn.GRUCell(
            input_size=self.input_size,  # 7
            hidden_size=self.hidden_size  # 35
        )
        self.output_layer = nn.Linear(
            in_features=self.hidden_size,  # 35
            out_features=1
        )

    def forward(
            self,
            input  # shape: batch_size, sequence_length, feature_length
    ):
        batch_size = input.shape[0]
        seq_length = input.shape[1]
        feature_length = input.shape[2]

        hidden_state = self.init_hidden(batch_size)  # shape: batch_size x hidden_size

        outputs = []
        for i in range(seq_length):
            curr_in = torch.squeeze(input[:, i, :], dim=1)  # shape: batch x feature_length
            hidden_state = self.gru_cell(curr_in, hidden_state)  # shape: batch x hidden_size

            # Action mask to decide trade or not to trade
            temperature = 0.05
            curr_out = self.output_layer(hidden_state)  # shape: batch x 1

            curr_out = torch.tanh(curr_out)
            outputs.append(curr_out)

        outputs = torch.stack(outputs).squeeze(-1)
        outputs = outputs.permute(1, 0)  # batch_size x seq_len

        return outputs  # shape: sequence_length, batch_size

    def init_hidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            hidden_state = Variable(torch.zeros(batch_size, self.hidden_size).cuda())  # shape: 10, 35
            hidden_state = hidden_state.float()
            return hidden_state  # shape: 10, 35
        else:
            hidden_state = Variable(torch.zeros(batch_size, self.hidden_size))  # shape: 10, 35
            hidden_state = hidden_state.float()
            return hidden_state  # shape: 10, 35
