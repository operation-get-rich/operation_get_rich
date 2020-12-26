import torch
import torch.nn as nn
from torch.autograd import Variable


class VanillaGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VanillaGRU, self).__init__()

        self.input_size = input_size  # 7 (open, close, low, high, volume, vwap, ema)
        self.hidden_size = hidden_size  # 35
        self.output_size = output_size  # 4 (open, close, low, high)

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.identity = torch.eye(input_size).cuda()
            self.zeros = Variable(torch.zeros(input_size).cuda())
        else:
            self.identity = torch.eye(input_size)
            self.zeros = Variable(torch.zeros(input_size))

        self.cell = nn.GRUCell(
            input_size=self.input_size,  # 7
            hidden_size=self.hidden_size  # 35
        )
        self.output_layer = nn.Linear(
            in_features=self.hidden_size,  # 35
            out_features=self.output_size  # 4
        )

    def forward(
            self,
            input  # shape: 10, 390, 7
    ):
        batch_size = input.shape[0]  # 10
        # TODO: If the seqeuence length varied, this won't be a constant
        seq_length = input.shape[1]  # 389
        num_features = input.shape[2]  # 7

        hidden_state = self.init_hidden(batch_size)  # shape: 10, 35

        outputs = []
        for i in range(seq_length):
            curr_in = torch.squeeze(input[:, i, :], dim=1)  # shape: 10, 7
            hidden_state = self.cell(curr_in, hidden_state)  # shape: 10, 35
            curr_out = self.output_layer(hidden_state)  # shape: 10, 4

            outputs.append(curr_out)

        return outputs  # shape: (up to) 389*, 10, 4

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
