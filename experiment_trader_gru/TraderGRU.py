import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class TraderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, hard=False, sparse=True):
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
        self.action_layer = nn.Linear(
            in_features=self.hidden_size,  # 35
            out_features=1
        )
        self.output_layer = nn.Linear(
            in_features=self.hidden_size,  # 35
            out_features=1
        )

        self.hard = hard
        self.sparse = sparse

        self.action_penalties = []

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
            input  # shape: batch_size, sequence_length, feature_length
    ):
        self.action_penalties = []  # action_logit_sigmoids is being reset every forward run

        batch_size = input.shape[0]
        seq_length = input.shape[1]
        feature_length = input.shape[2]

        hidden_state = self.init_hidden(batch_size)  # shape: batch_size x hidden_size

        outputs = []

        for i in range(seq_length):
            curr_in = torch.squeeze(input[:, i, :], dim=1)  # shape: batch x feature_length
            hidden_state = self.gru_cell(curr_in, hidden_state)  # shape: batch x hidden_size

            # Action mask to decide trade or not to trade
            curr_out = self.output_layer(hidden_state)  # shape: batch x 1
            curr_out = torch.tanh(curr_out)

            if self.sparse:
                temperature = 0.05
                action_logit = self.action_layer(hidden_state)  # batch x 1
                action_penalty = F.sigmoid(action_logit)  # batch x 1
                action = self.gumbel_softmax_sample(action_logit, temperature, hard=self.hard)
                curr_out = curr_out * action
                self.action_penalties.append(action_penalty)
            outputs.append(curr_out)

        if self.action_penalties:
            self.action_penalties = torch.stack(self.action_penalties).squeeze(-1)
            self.action_penalties = self.action_penalties.permute(1, 0)  # batch_size x seq_len¬
            self.action_penalties = self.action_penalties.sum(axis=1)  # batch_size x 1

        outputs = torch.stack(outputs).squeeze(-1)
        outputs = outputs.permute(1, 0)  # batch_size x seq_len

        return outputs  # shape: batch_size x seq_len

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


def ProfitLoss(
        trade_sequence,
        market_sequence,
        penalty,
        is_premarket=None,
        next_trade=False
):
    if next_trade:
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
    return -(capital - 1) + penalty


def load_trader_gru_model(model_location):
    model = TraderGRU(
        input_size=7,
        hidden_size=5 * 7,
        hard=True
    )
    model.load_state_dict(
        torch.load(
            model_location,
            map_location=torch.device('cpu')
        )
    )
    model.eval()
    return model
