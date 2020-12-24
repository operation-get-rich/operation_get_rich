import argparse
import time
from typing import List, Tuple

import numpy as np
import pandas

import torch
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader

from experiment_sniper_gru.SniperGRU import SniperGRU
from experiment_sniper_gru.sniper_dataset import OPEN_COLUMN_INDEX, Normalizer, StockDataset

import multiprocessing

from experiment_vanilla_gru.dataset_vanilla_gru import CLOSE_COLUMN_INDEX
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
        trader_gru_model,  # type: SniperGRU
        train_loader,  # type: DataLoader
        valid_loader,  # type: DataLoader
        batch_size=BATCH_SIZE,  # type: int
        num_epochs=30000,  # type: int
        patience=30000,  # type: int
        min_delta=0.00001,  # type: float
        learning_rate=0.0001  # type: float
):
    # type: (...) -> Tuple[SniperGRU, List[Tensor]]

    loss_function = torch.nn.MSELoss(reduction='sum')

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
        losses_epoch_train = _train(
            train_loader,
            trader_gru_model,
            loss_function,
            optimizer,
            batch_size
        )

        losses_epoch_valid = _validate(
            valid_loader,
            trader_gru_model,
            loss_function,
            batch_size
        )

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


def _train(train_loader, trader_gru_model, loss_function, optimizer, batch_size):
    losses_epoch_train = []
    for features, original_sequence_lengths in train_loader:
        features  # shape: batch_size x sequence_length x feature_length
        original_sequence_lengths  # shape: batch_size

        if features.shape[0] != batch_size:
            continue

        pandas.DataFrame(features[0].numpy())

        model_outputs = get_model_output(
            features=features,
            model=trader_gru_model
        )  # shape: batch_size x sequence_length

        targets = get_targets(
            features=features
        )  # shape: batch_size x sequence_length

        loss = compute_loss(
            model_outputs,
            targets,
            original_sequence_lengths,
            loss_function
        )

        losses_epoch_train.append(loss)

        trader_gru_model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses_epoch_train


def _validate(valid_loader, trader_gru_model, loss_function, batch_size):
    losses_epoch_valid = []
    for features, original_sequence_lengths_val in valid_loader:
        if features.shape[0] != batch_size:
            continue

        model_outputs = get_model_output(
            features=features,
            model=trader_gru_model
        )

        targets = get_targets(
            features=features
        )  # shape: batch_size x sequence_length

        loss = compute_loss(
            model_outputs=model_outputs,
            target=targets
        )

        losses_epoch_valid.append(loss)

    return losses_epoch_valid


def get_model_output(
        features,  # size: batch_size x sequence_length x feature_length
        model  # type: SniperGRU
):
    use_gpu = torch.cuda.is_available()
    features = features.float()

    # TODO: Should we change the way we normalize volume?
    #  Find technique of normalization that keeps updating whenever new data comes in
    normalized_features = Normalizer.normalize_volume(
        features)  # size: batch_size x sequence_length x feature_length
    normalized_features = Normalizer.normalize_price_into_percent_change(normalized_features)

    if use_gpu:
        normalized_features = Variable(normalized_features.cuda())
    else:
        normalized_features = Variable(normalized_features)

    outputs = model(normalized_features)  # batch_size x sequence_length

    return outputs


def get_targets(
        features,  # shape: batch x sequence_length x feature_length
        profit_target=.02
):
    pandas.DataFrame(features[0].numpy())
    sequence_length = features.shape[1]
    targets = []
    for i in range(sequence_length - 1):
        close_prices = features[:, i + 1, CLOSE_COLUMN_INDEX]
        open_prices = features[:, i + 1, OPEN_COLUMN_INDEX]
        flags = torch.from_numpy(np.array(
            [float(close >= open * (1 + profit_target)) for close, open in zip(close_prices, open_prices)]
        ))  # shape: batch_size
        targets.append(flags)
    targets = torch.stack(targets).squeeze(-1)  # shape: sequence_length x batch_size
    targets = torch.transpose(targets, 0, 1)  # shape: batch_size x sequence_length
    return targets


def compute_loss(
        model_outputs,  # shape: batch_size x sequence_length
        targets,  # shape: batch_size x sequence_length
        original_sequence_lengths,  # shape: batch_size
        loss_function
):
    batch_size = len(original_sequence_lengths)
    loss = 0
    for batch_index, original_sequence_lengths in enumerate(original_sequence_lengths):
        output = model_outputs[batch_index, :]
        target = model_outputs[batch_index, :]

        loss += loss_function(output, target)
    loss /= batch_size
    return loss


if __name__ == "__main__":
    # Create directories
    args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    create_dir(args.save)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    root_data_dir = '../datas/polygon_early_day_gap_segmenter_parallel'
    train_data = StockDataset(
        data_folder=root_data_dir,
        split='train',
        should_add_technical_indicator=True
    )

    test_data = StockDataset(
        data_folder=root_data_dir,
        split='test',
        should_add_technical_indicator=True
    )

    train_loader = DataLoader(train_data, num_workers=1, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, num_workers=1, shuffle=True, batch_size=BATCH_SIZE)

    inputs, sequence_length = next(iter(train_loader))

    inputs, original_sequence_lengths = next(iter(train_loader))
    inputs  # shape: 10, 390, 7
    [batch_size, seq_length, num_features] = inputs.size()

    model = SniperGRU(
        input_size=num_features,
        hidden_size=5 * num_features
    )
    if torch.cuda.is_available():
        model = model.cuda()

    best_grud, losses_grud = train(model, train_loader, test_loader)