import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class UCIVanillaGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(UCIVanillaGRU, self).__init__()

        self.input_size = input_size  # 561
        self.hidden_size = hidden_size  # 2805
        self.output_size = output_size  # 6

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.identity = torch.eye(input_size).cuda()
            self.zeros = Variable(torch.zeros(input_size).cuda())
        else:
            self.identity = torch.eye(input_size)
            self.zeros = Variable(torch.zeros(input_size))

        self.cell = nn.GRUCell(
            input_size=self.input_size,  # 561
            hidden_size=self.hidden_size  # 2805
        )
        self.output_layer = nn.Linear(
            in_features=self.hidden_size,  # 2805
            out_features=self.output_size  # 6
        )

    def forward(
            self,
            input  # shape: 10, 200, 561
    ):
        batch_size = input.shape[0]
        seq_length = input.shape[1]
        num_features = input.shape[2]

        hidden_state = self.init_hidden(batch_size)  # shape: 10, 2805

        outputs = []
        for i in range(seq_length):
            curr_in = torch.squeeze(input[:, i, :], dim=1)  # shape: 10, 561
            # I'm guessing it's doing forward prop for 10 rows at once
            hidden_state = self.cell(curr_in, hidden_state)  # shape: 10, 2805
            curr_out = self.output_layer(hidden_state)  # shape: 10,6

            outputs.append(curr_out)

        return outputs  # shape: 200*, 10, 6

    def init_hidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            hidden_state = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            hidden_state = hidden_state.float()
            return hidden_state
        else:
            hidden_state = Variable(torch.zeros(batch_size, self.hidden_size))  # shape: 10, 2805
            hidden_state = hidden_state.float()
            return hidden_state
