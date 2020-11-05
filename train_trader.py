import argparse
import os
import time

import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from TraderGRU import TraderGRU
from stock_dataset import StockDataset, PercentChangeNormalizer
from objectives import ProfitReward

import multiprocessing

multiprocessing.set_start_method("spawn", True)

# Arguments
parser = argparse.ArgumentParser(description='TraderGRU Train')

# General Settings
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--save', type=str, default='Train', help='experiment name')
args = parser.parse_args()


def compute_loss(
        trades,  # batch_size x seq_len
        open_prices,  # batch_size x seq_len
        original_seq_length,  # batch_size
        loss_function
):
    loss_train = 0
    for i, osl in enumerate(original_seq_length):
        current_outputs = trades[i, 0: osl].float()
        current_prices = open_prices[i, 0: osl].float()

        loss_train -= loss_function(
            current_outputs,
            current_prices
        )
    loss_train /= len(original_seq_length)

    return loss_train


def get_trades_from_model(features_osl_tuples, model):
    use_gpu = torch.cuda.is_available()
    features, original_sequence_lengths = features_osl_tuples
    features = features.float()

    normalized_features = PercentChangeNormalizer.normalize_volume(features)
    normalized_features = PercentChangeNormalizer.normalize_price_into_percent_change(normalized_features)

    # labels contains open, close, low, high
    normalized_features = normalized_features[:, :-1, :].float()  # batch_size x  seq_len-1 x input_size

    if use_gpu:
        normalized_features = Variable(normalized_features.cuda())
    else:
        normalized_features = Variable(normalized_features)

    # TODO: Run and check shapes want batch_size x seq_len
    trades = model(normalized_features)  # seq_len-1 x batch_size x output_size x 1
    trades = torch.stack(trades)
    trades = trades.permute(1, 0, 2).squeeze(-1)  # batch_size x seq_len-1 x 1

    return trades


def train(
        trader_gru_model,  # type: TraderGRU
        train_loader,  # type: DataLoader
        valid_loader,  # type: DataLoader
        num_epochs=30000,  # type: int
        patience=30000,  # type: int
        min_delta=0.00001  # type: int
):
    # type: (...) -> NotImplemented
    """
    TODO: Hyperparameter to experiemnts:
        - The loss function (MSELoss, L1, etc)
        - The optimizer (RMS, Adams...)
    """
    print('Model Structure: ', trader_gru_model)
    print('Start Training ... ')

    # loss = torch.nn.MSELoss()
    loss_function = ProfitReward

    learning_rate = 0.0001
    optimizer = torch.optim.RMSprop(trader_gru_model.parameters(), lr=learning_rate, alpha=0.99)

    interval = 100
    losses_train = []
    losses_valid = []
    losses_epochs_train = []
    losses_epochs_valid = []

    cur_time = time.time()
    pre_time = time.time()

    prices, original_sequence_lengths = next(iter(train_loader))
    [batch_size, seq_length, input_size] = prices.size()

    # Variables for Early Stopping
    is_best_model = 0
    patient_epoch = 0
    min_loss_epoch_valid = 10000.0
    for epoch in range(num_epochs):
        losses_epoch_train = []
        losses_epoch_valid = []

        for features_osl_tuples in train_loader:
            prices, original_sequence_lengths = features_osl_tuples
            prices  # shape: batch_size x sequence_length x feature_length
            if prices.shape[0] != batch_size:
                continue

            trades = get_trades_from_model(features_osl_tuples, trader_gru_model)

            open_prices = prices[:, :, 0]
            loss_train = compute_loss(trades, open_prices, original_sequence_lengths, loss_function)

            losses_train.append(loss_train)
            losses_epoch_train.append(loss_train)

            trader_gru_model.zero_grad()
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        for features_osl_tuples in valid_loader:
            prices_val, original_sequence_lengths_val = features_osl_tuples

            if prices_val.shape[0] != batch_size:
                continue

            trades = get_trades_from_model(features_osl_tuples=features_osl_tuples, model=trader_gru_model)

            open_prices = prices_val[:, :, 0]
            loss_val = compute_loss(trades=trades, open_prices=open_prices,
                                    original_seq_length=original_sequence_lengths,
                                    loss_function=loss_function)
            losses_valid.append(loss_val)
            losses_epoch_valid.append(loss_val)

        torch.save(trader_gru_model.state_dict(), args.save + "/latest_model.pt")

        avg_losses_epoch_train = sum(losses_epoch_train).cpu().detach().numpy() / float(len(losses_epoch_train))
        avg_losses_epoch_valid = sum(losses_epoch_valid).cpu().detach().numpy() / float(len(losses_epoch_valid))
        losses_epochs_train.append(avg_losses_epoch_train)
        losses_epochs_valid.append(avg_losses_epoch_valid)

        # Early Stopping
        if epoch == 0:
            is_best_model = 1
            best_model = trader_gru_model
            if avg_losses_epoch_valid < min_loss_epoch_valid:
                min_loss_epoch_valid = avg_losses_epoch_valid
        else:
            if min_loss_epoch_valid - avg_losses_epoch_valid > min_delta:
                is_best_model = 1
                best_model = trader_gru_model
                min_loss_epoch_valid = avg_losses_epoch_valid
                patient_epoch = 0

                torch.save(trader_gru_model.state_dict(), args.save + "/best_model.pt")
            else:
                is_best_model = 0
                patient_epoch += 1
                if patient_epoch >= patience:
                    print('Early Stopped at Epoch:', epoch)
                    break

        # Print training parameters
        cur_time = time.time()
        print(
            'Epoch: {}, train_loss: {}, valid_loss: {}, time: {}, best model: {}'.format(
                epoch,
                np.around(avg_losses_epoch_train, decimals=8),
                np.around(avg_losses_epoch_valid, decimals=8),
                np.around([cur_time - pre_time], decimals=2),
                is_best_model)
        )
        pre_time = cur_time

    return best_model, [losses_train, losses_valid, losses_epochs_train, losses_epochs_valid]


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == "__main__":
    # Create directories
    args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    create_dir(args.save)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    train_data = StockDataset(
        data_folder='./gaped_up_stocks_early_volume_1e5_gap_10',
        split='train',
        should_add_technical_indicator=True
    )

    test_data = StockDataset(
        data_folder='./gaped_up_stocks_early_volume_1e5_gap_10',
        split='test',
        should_add_technical_indicator=True
    )

    train_loader = DataLoader(train_data, num_workers=1, shuffle=True, batch_size=10)
    test_loader = DataLoader(test_data, num_workers=1, shuffle=True, batch_size=10)

    inputs, sequence_length = next(iter(train_loader))

    inputs, original_sequence_lengths = next(iter(train_loader))
    inputs  # shape: 10, 390, 7
    [batch_size, seq_length, num_features] = inputs.size()

    model = TraderGRU(
        input_size=num_features,
        hidden_size=5 * num_features
    )
    if torch.cuda.is_available():
        model = model.cuda()

    best_grud, losses_grud = train(model, train_loader, test_loader, num_epochs=1)
