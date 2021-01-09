import argparse
import time

import numpy as np

import matplotlib.pyplot as plt
import pandas
import torch
from sklearn.metrics import recall_score, precision_score, f1_score
from torch.autograd import Variable
from torch.utils.data import DataLoader

from experiment_sniper_gru.SniperGRU import SniperGRU
from experiment_sniper_gru.dataset import SniperDataset

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

# check whether cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IS_TEST_MODE = False
TEST_MODE_ITERATION_LIMIT = 100


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
    loss_function = torch.nn.BCELoss().to(device)
    optimizer = torch.optim.RMSprop(trader_gru_model.parameters(), lr=learning_rate)

    print('Model Structure: ', trader_gru_model)
    print('Start Training ... ')

    cur_time = time.time()
    pre_time = time.time()

    # Variables for Early Stopping
    is_least_error_model = 0
    patient_epoch = 0
    min_loss_epoch_valid = 10000.0
    avg_losses_train = []
    avg_losses_valid = []

    is_max_recall_model = 1
    max_recall_valid = 0
    avg_recalls_train = []
    avg_recalls_valid = []

    is_max_precision_model = 1
    max_precision_valid = 0
    avg_precisions_train = []
    avg_precisions_valid = []

    is_max_f1_model = 1
    max_f1_valid = 0
    avg_f1s_train = []
    avg_f1s_valid = []

    for epoch in range(num_epochs):
        (
            avg_loss_train,
            avg_recall_train,
            avg_precision_train,
            avg_f1_train
        ) = _train(
            train_loader,
            trader_gru_model,
            optimizer,
            loss_function,
            batch_size
        )
        avg_losses_train.append(avg_loss_train)
        avg_recalls_train.append(avg_recall_train)
        avg_precisions_train.append(avg_precision_train)
        avg_f1s_train.append(avg_f1_train)

        (
            avg_loss_valid,
            avg_recall_valid,
            avg_precision_valid,
            avg_f1_valid
        ) = _validate(
            valid_loader,
            trader_gru_model,
            loss_function,
            batch_size
        )
        avg_losses_train.append(avg_loss_valid)
        avg_recalls_train.append(avg_recall_valid)
        avg_precisions_train.append(avg_precision_valid)
        avg_f1s_train.append(avg_f1_valid)

        torch.save(trader_gru_model.state_dict(), args.save + "/latest_model.pt")

        # Early Stopping
        if epoch == 0:
            is_least_error_model = 1
            if avg_loss_valid < min_loss_epoch_valid:
                min_loss_epoch_valid = avg_loss_valid
        else:
            if min_loss_epoch_valid - avg_loss_valid > min_delta:
                is_least_error_model = 1
                min_loss_epoch_valid = avg_loss_valid
                patient_epoch = 0
                torch.save(trader_gru_model.state_dict(), args.save + "/least_error_model.pt")

            if avg_recall_valid > max_recall_valid:
                is_max_recall_model = 1
                max_recall_valid = avg_recall_valid
                patient_epoch = 0
                torch.save(trader_gru_model.state_dict(), args.save + "/max_recall_model.pt")

            if avg_precision_valid > max_precision_valid:
                is_max_precision_model = 1
                max_precision_valid = avg_precision_valid
                patient_epoch = 0
                torch.save(trader_gru_model.state_dict(), args.save + "/max_precision_model.pt")

            if avg_f1_valid > max_f1_valid:
                is_max_f1_model = 1
                max_f1_valid = avg_f1_valid
                patient_epoch = 0
                torch.save(trader_gru_model.state_dict(), args.save + "/max_f1_model.pt")
            else:
                is_least_error_model = 0
                is_max_recall_model = 0
                is_max_precision_model = 0
                is_max_f1_model = 0
                patient_epoch += 1
                if patient_epoch >= patience:
                    print('Early Stopped at Epoch:', epoch)
                    break

        # Print training parameters
        cur_time = time.time()
        print(
            'Epoch: {}, '
            'train_loss: {}, '
            'valid_loss: {}, '
            'train_precision: {}, '
            'valid_precision: {}, '
            'train_recall: {}, '
            'valid_recall: {}, '
            'train_f1: {}, '
            'valid_f1: {}, '
            'time: {}, '
            'least error model: {}, '
            'max_precision_model: {}, '
            'max_recall_model: {} '
            'max_f1_model: {}'.format(
                epoch,
                np.around(avg_loss_train, decimals=8),
                np.around(avg_loss_valid, decimals=8),
                np.around(avg_precision_train, decimals=8),
                np.around(avg_precision_valid, decimals=8),
                np.around(avg_recall_train, decimals=8),
                np.around(avg_recall_valid, decimals=8),
                np.around(avg_f1_train, decimals=8),
                np.around(avg_f1_valid, decimals=8),
                np.around([cur_time - pre_time], decimals=2),
                is_least_error_model,
                is_max_precision_model,
                is_max_recall_model,
                is_max_f1_model,
            )
        )
        pre_time = cur_time

    plt.title('Training')
    plt.plot(avg_losses_train, label='loss')
    plt.plot(avg_recalls_train, label='recall')
    plt.plot(avg_precisions_train, label='precision')
    plt.plot(avg_f1s_train, label='f1')

    plt.legend()
    plt.savefig(f'{args.save}/training_metrics.png')

    plt.clf()
    plt.title('Validation')
    plt.plot(avg_losses_valid, label='loss')
    plt.plot(avg_recalls_valid, label='recall')
    plt.plot(avg_precisions_valid, label='precision')
    plt.plot(avg_f1s_valid, label='f1')

    plt.legend()
    plt.savefig(f'{args.save}/validation_metrics.png')


def _train(train_loader, model, optimizer, loss_function, batch_size):
    loss_sum = torch.tensor(0).float()
    recall_sum = torch.tensor(0).float()
    precision_sum = torch.tensor(0).float()
    f1_sum = torch.tensor(0).float()

    test_mode_index = 0  # used just for confirming training are running properly

    for features, labels, original_sequence_lengths in train_loader:

        features = features.float().to(device)  # shape: batch_size x sequence_length x feature_length
        labels = labels.float().to(device)  # shape: batch_size
        original_sequence_lengths = original_sequence_lengths.to(device)  # shape: batch_size

        if features.shape[0] != batch_size:
            continue

        normalized_features = Variable(features).to(device)
        outputs = model(normalized_features, original_sequence_lengths)  # batch_size x 1

        loss = loss_function(outputs, labels)
        recall, precision, f1 = _get_evaluation_metric(outputs, labels)

        loss_sum += loss
        recall_sum += recall
        precision_sum += precision
        f1_sum += f1

        model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if IS_TEST_MODE:
            print(f'iteration {test_mode_index}')
            test_mode_index += 1
            if test_mode_index == TEST_MODE_ITERATION_LIMIT:
                break
    return (
        loss_sum.detach().numpy() / len(train_loader),
        recall_sum.detach().numpy() / len(train_loader),
        precision_sum.detach().numpy() / len(train_loader),
        f1_sum.detach().numpy() / len(train_loader)
    )


def _validate(valid_loader, trader_gru_model, loss_function, batch_size):
    loss_sum = torch.tensor(0).float()
    recall_sum = torch.tensor(0).float()
    precision_sum = torch.tensor(0).float()
    f1_sum = torch.tensor(0).float()
    test_mode_index = 0
    for features, labels, original_sequence_lengths in valid_loader:
        features = features.float().to(device)  # shape: batch_size x sequence_length x feature_length
        labels = labels.float().to(device)  # shape: batch_size
        original_sequence_lengths = original_sequence_lengths.to(device)  # shape: batch_size

        if features.shape[0] != batch_size:
            continue

        normalized_features = Variable(features).to(device)
        outputs = trader_gru_model(normalized_features, original_sequence_lengths)  # batch_size x 1

        loss = loss_function(outputs, labels)
        recall, precision, f1 = _get_evaluation_metric(outputs, labels)

        loss_sum += loss
        recall_sum += recall
        precision_sum += precision
        f1_sum += f1
        if IS_TEST_MODE:
            print(f'Valid iteration {test_mode_index}')
            test_mode_index += 1
            if test_mode_index == TEST_MODE_ITERATION_LIMIT:
                break
    return (
        loss_sum.detach().numpy() / len(train_loader),
        recall_sum.detach().numpy() / len(train_loader),
        precision_sum.detach().numpy() / len(train_loader),
        f1_sum.detach().numpy() / len(train_loader)
    )


def _get_evaluation_metric(
        outputs,  # batch_size x 1
        y,  # batch_size x 1
):
    # round predictions to the closest integer
    rounded_preds = torch.where(outputs > 0.5, 1, 0)  # if output > 0.8 outputs 1 else 0

    recall = recall_score(y_true=y, y_pred=rounded_preds)
    precision = precision_score(y_true=y, y_pred=rounded_preds)
    f1 = f1_score(y_true=y, y_pred=rounded_preds)

    return recall, precision, f1


if __name__ == "__main__":
    # Hyper-parameters
    num_classes = 1
    num_epochs = 30000
    batch_size = 10
    learning_rate = 0.0001
    num_layers = 2
    hidden_size_multipier = 5

    # Create directories
    args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    create_dir(args.save)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    train_data = SniperDataset(
        split='train',
        dataset_name=f'sniper_training_data_3',
    )

    test_data = SniperDataset(
        split='test',
        dataset_name=f'sniper_training_data_3'
    )

    train_loader = DataLoader(train_data, num_workers=1, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, num_workers=1, shuffle=True, batch_size=BATCH_SIZE)

    inputs, label, original_sequence_lengths = next(iter(train_loader))
    # inputs: batch_size, seq_length, feature_length

    [batch_size, seq_length, num_features] = inputs.size()

    model = SniperGRU(
        input_size=num_features,
        hidden_size=hidden_size_multipier * num_features,
        num_layers=num_layers,
        num_classes=num_classes
    ).to(device)

    train(
        model,
        train_loader,
        test_loader,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
