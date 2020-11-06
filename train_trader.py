import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd

import torch
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader

from TraderGRU import TraderGRU
from stock_dataset import StockDataset, PercentChangeNormalizer, OPEN_COLUMN_INDEX
from objectives import ProfitReward

import multiprocessing

from utils import create_dir

multiprocessing.set_start_method("spawn", True)

# Arguments
parser = argparse.ArgumentParser(description='TraderGRU Train')

# General Settings
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--save', type=str, default='Train', help='experiment name')
args = parser.parse_args()

BATCH_SIZE = 10


def train(
        trader_gru_model,  # type: TraderGRU
        train_loader,  # type: DataLoader
        valid_loader,  # type: DataLoader
        batch_size=BATCH_SIZE,  # type: int
        num_epochs=30000,  # type: int
        patience=30000,  # type: int
        min_delta=0.00001,  # type: float
        learning_rate=0.0001  # type: float
):
    # type: (...) -> Tuple[TraderGRU, List[Tensor]]
    loss_function = ProfitReward
    optimizer = torch.optim.RMSprop(trader_gru_model.parameters(), lr=learning_rate)

    print('Model Structure: ', trader_gru_model)
    print('Start Training ... ')

    average_epoch_losses_train = []
    average_epoch_losses_valid = []

    cur_time = time.time()
    pre_time = time.time()

    # Variables for Early Stopping
    is_best_model = 0
    patient_epoch = 0
    min_loss_epoch_valid = 10000.0
    for epoch in range(num_epochs):
        losses_epoch_train = []
        losses_epoch_valid = []

        for features, original_sequence_lengths in train_loader:
            features  # shape: batch_size x sequence_length x feature_length
            original_sequence_lengths  # shape: batch_size

            if features.shape[0] != batch_size:
                continue

            trades = get_trades_from_model(
                features=features,
                model=trader_gru_model
            )  # shape: batch_size x sequence_length

            open_prices = features[:, :, OPEN_COLUMN_INDEX]

            loss_train = compute_loss(
                trades=trades,
                open_prices=open_prices,
                original_sequence_lengths=original_sequence_lengths,
                loss_function=loss_function
            )

            losses_epoch_train.append(loss_train)

            trader_gru_model.zero_grad()
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        for features_val, original_sequence_lengths_val in valid_loader:
            if features_val.shape[0] != batch_size:
                continue

            trades = get_trades_from_model(
                features=features_val,
                model=trader_gru_model
            )

            open_prices = features_val[:, :, 0]
            loss_val = compute_loss(
                trades=trades,
                open_prices=open_prices,
                original_sequence_lengths=original_sequence_lengths_val,
                loss_function=loss_function
            )
            losses_epoch_valid.append(loss_val)

        torch.save(trader_gru_model.state_dict(), args.save + "/latest_model.pt")

        avg_losses_epoch_train = sum(losses_epoch_train).cpu().detach().numpy() / float(len(losses_epoch_train))
        avg_losses_epoch_valid = sum(losses_epoch_valid).cpu().detach().numpy() / float(len(losses_epoch_valid))
        average_epoch_losses_train.append(avg_losses_epoch_train)
        average_epoch_losses_valid.append(avg_losses_epoch_valid)

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

    return best_model, [average_epoch_losses_train, average_epoch_losses_valid]


def compute_loss(
        trades,  # batch_size x seq_len
        open_prices,  # batch_size x seq_len
        original_sequence_lengths,  # batch_size
        loss_function
):
    loss_train = 0
    for i, osl in enumerate(original_sequence_lengths):
        current_outputs = trades[i, 0: osl].float()
        current_prices = open_prices[i, 0: osl].float()

        loss_train -= loss_function(
            current_outputs,
            current_prices
        )
    loss_train /= len(original_sequence_lengths)

    return loss_train


def get_trades_from_model(
        features,  # size: batch_size x sequence_length x feature_length
        model  # type: TraderGRU
):
    use_gpu = torch.cuda.is_available()
    features = features.float()

    normalized_features = PercentChangeNormalizer.normalize_volume(
        features)  # size: batch_size x sequence_length x feature_length
    normalized_features = PercentChangeNormalizer.normalize_price_into_percent_change(normalized_features)

    if use_gpu:
        normalized_features = Variable(normalized_features.cuda())
    else:
        normalized_features = Variable(normalized_features)

    trades = model(normalized_features)  # batch_size x sequence_length

    return trades


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

    train_loader = DataLoader(train_data, num_workers=1, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, num_workers=1, shuffle=True, batch_size=BATCH_SIZE)

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
