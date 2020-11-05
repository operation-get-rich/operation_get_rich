import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class TraderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, hard=False):
        super(TraderGRU, self).__init__()

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
        self.action_layer = nn.Linear(
            in_features=self.hidden_size,  # 35
            out_features=1
        )

        self.hard = hard

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).cuda()
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature, hard=False, deterministic=False, eps=1e-20):

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

    def forward(
            self,
            input  # shape: 10, 390, 7
    ):
        batch_size = input.shape[0]  # 10
        seq_length = input.shape[1]  # 389
        num_features = input.shape[2]  # 7

        hidden_state = self.init_hidden(batch_size)  # shape: 10, 35

        outputs = []
        for i in range(seq_length):
            curr_in = torch.squeeze(input[:, i, :], dim=1)  # shape: 10, 7
            hidden_state = self.gru_cell(curr_in, hidden_state)  # shape: 10, 35
            curr_out = self.output_layer(hidden_state)  # shape: 10, 1

            # To trade or not to trade
            temperature = 0.05
            action_logit = self.action_layer(hidden_state)  # batch x 1
            action = self.gumbel_softmax_sample(action_logit, temperature, hard=self.hard)

            curr_out = F.tanh(curr_out) * action
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
