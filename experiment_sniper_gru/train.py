import argparse
import time
from typing import List, Tuple

import numpy as np
import pandas

import torch
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader

from directories import DATA_DIR
from experiment_sniper_gru.SniperGRU import SniperGRU
from experiment_sniper_gru.dataset import OPEN_COLUMN_INDEX, SniperDataset
from experiment_sniper_gru.normalizer import PercentChangeNormalizer

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

# check whether cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    loss_function = torch.nn.BCELoss().to(device)

    optimizer = torch.optim.RMSprop(trader_gru_model.parameters(), lr=learning_rate)

    print('Model Structure: ', trader_gru_model)
    print('Start Training ... ')

    average_epoch_losses_train = []
    average_epoch_losses_valid = []

    average_epoch_accuracies_train = []
    average_epoch_accuracies_valid = []

    cur_time = time.time()
    pre_time = time.time()

    # Variables for Early Stopping
    is_best_model = 0
    patient_epoch = 0
    min_loss_epoch_valid = 10000.0
    for epoch in range(num_epochs):
        losses_epoch_train, accuracies_epoch_train = _train(
            train_loader,
            trader_gru_model,
            optimizer,
            loss_function,
            batch_size
        )

        losses_epoch_valid, accuracies_epoch_valid = _validate(
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

        avg_accuracies_epoch_train = (sum(accuracies_epoch_train).cpu().detach().numpy()
                                      / float(len(losses_epoch_train)))
        avg_accuracies_epoch_valid = (sum(accuracies_epoch_valid).cpu().detach().numpy()
                                      / float(len(losses_epoch_valid)))
        average_epoch_accuracies_train.append(avg_accuracies_epoch_train)
        average_epoch_accuracies_valid.append(avg_accuracies_epoch_valid)

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
            'Epoch: {}, train_loss: {}, valid_loss: {}, train_acc: {}, valid_acc: {}, time: {}, best model: {}'.format(
                epoch,
                np.around(avg_losses_epoch_train, decimals=8),
                np.around(avg_losses_epoch_valid, decimals=8),
                np.around(avg_accuracies_epoch_train, decimals=8),
                np.around(avg_accuracies_epoch_valid, decimals=8),
                np.around([cur_time - pre_time], decimals=2),
                is_best_model)
        )
        pre_time = cur_time

    return best_model, [average_epoch_losses_train, average_epoch_losses_valid]


def _train(train_loader, trader_gru_model, optimizer, loss_function, batch_size):
    losses_epoch_train = []
    accuracies_epoch_train = []
    for features, labels, original_sequence_lengths in train_loader:
        features  # shape: batch_size x sequence_length x feature_length
        labels  # shape: batch_size
        original_sequence_lengths  # shape: batch_size

        if features.shape[0] != batch_size:
            continue

        outputs = get_model_output(
            features=features,
            model=trader_gru_model,
            original_sequence_lengths=original_sequence_lengths
        )  # shape: batch_size x 1

        # TODO: Calling float here is kind of hacky. Fix this
        loss = loss_function(outputs, labels.float())
        accuracy = _binary_accuracy(outputs, labels)

        losses_epoch_train.append(loss)
        accuracies_epoch_train.append(accuracy)

        trader_gru_model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses_epoch_train, accuracies_epoch_train


def _validate(valid_loader, trader_gru_model, loss_function, batch_size):
    losses_epoch_valid = []
    accuracies_epoch_valid = []
    for features, labels, original_sequence_lengths_val in valid_loader:
        if features.shape[0] != batch_size:
            continue

        outputs = get_model_output(
            features=features,
            model=trader_gru_model,
            original_sequence_lengths=original_sequence_lengths
        )

        loss = loss_function(outputs, labels)
        accuracy = _binary_accuracy(outputs, labels)

        losses_epoch_valid.append(loss)
        accuracies_epoch_valid.append(accuracy)

    return losses_epoch_valid, accuracies_epoch_valid


def get_model_output(
        features,  # size: batch_size x sequence_length x feature_length
        model,  # type: SniperGRU
        original_sequence_lengths,
):
    use_gpu = torch.cuda.is_available()
    features = features.float()

    # TODO: Should we change the way we normalize volume?
    #  Find technique of normalization that keeps updating whenever new data comes in
    normalized_features = PercentChangeNormalizer.normalize_volume(
        features)  # size: batch_size x sequence_length x feature_length
    normalized_features = PercentChangeNormalizer.normalize_price_into_percent_change(normalized_features)

    if use_gpu:
        normalized_features = Variable(normalized_features.cuda())
    else:
        normalized_features = Variable(normalized_features)

    outputs = model(normalized_features, original_sequence_lengths)  # batch_size x 1

    return outputs


def _binary_accuracy(outputs, y):
    # round predictions to the closest integer
    rounded_preds = torch.where(outputs > 0.8, 1, 0)  # if output > 0.8 outputs 1 else 0

    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


if __name__ == "__main__":
    # Hyper-parameters
    num_classes = 1
    num_epochs = 30000
    # batch_size = 100
    # learning_rate = 0.001
    #
    # input_size = 28
    # sequence_length = 28
    # hidden_size = 128
    num_layers = 2

    # Create directories
    args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    create_dir(args.save)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    train_data = SniperDataset(
        split='train',
        segment_data_dir=f'{DATA_DIR}/sniper_training_data_3'
    )

    test_data = SniperDataset(
        split='test',
        segment_data_dir=f'{DATA_DIR}/sniper_training_data_3'
    )

    train_loader = DataLoader(train_data, num_workers=1, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, num_workers=1, shuffle=True, batch_size=BATCH_SIZE)

    inputs, label, original_sequence_lengths = next(iter(train_loader))
    inputs  # shape: batch_size, seq_length, feature_length
    [batch_size, seq_length, num_features] = inputs.size()

    model = SniperGRU(
        input_size=num_features,
        hidden_size=5 * num_features,
        num_layers=num_layers,
        num_classes=num_classes
    )
    if torch.cuda.is_available():
        model = model.cuda()

    best_grud, losses_grud = train(
        model,
        train_loader,
        test_loader,
        num_epochs=num_epochs
    )
