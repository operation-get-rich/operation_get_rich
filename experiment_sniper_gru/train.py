import argparse
import time

import numpy as np

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
from torch.utils.data import DataLoader

from decorators import timeit
from experiment_sniper_gru.SniperGRU import SniperGRU
from experiment_sniper_gru.dataset import SniperDataset

import multiprocessing

from utils import create_dir

multiprocessing.set_start_method("spawn", True)

# Arguments
parser = argparse.ArgumentParser(description='Sniper GRU Train')

# General Settings
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--save', type=str, default='Train', help='experiment name')
parser.add_argument('--dataset-name', required=True, type=str)
args = parser.parse_args()

BATCH_SIZE = 50

# check whether cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IS_TEST_MODE = False
TEST_MODE_ITERATION_LIMIT = 10


def train(
        model,  # type: SniperGRU
        train_loader,  # type: DataLoader
        valid_loader,  # type: DataLoader
        batch_size=BATCH_SIZE,  # type: int
        num_epochs=30000,  # type: int
        patience=30000,  # type: int
        min_delta=0.00001,  # type: float
        learning_rate=0.0001  # type: float
):
    loss_function = torch.nn.BCELoss().to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    print('Model Structure: ', model)
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
            model,
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
            model,
            loss_function,
            batch_size
        )

        avg_losses_valid.append(avg_loss_valid)
        avg_recalls_valid.append(avg_recall_valid)
        avg_precisions_valid.append(avg_precision_valid)
        avg_f1s_valid.append(avg_f1_valid)

        torch.save(model.state_dict(), args.save + "/latest_model.pt")

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
                torch.save(model.state_dict(), args.save + "/least_error_model.pt")

            if avg_recall_valid > max_recall_valid:
                is_max_recall_model = 1
                max_recall_valid = avg_recall_valid
                patient_epoch = 0
                torch.save(model.state_dict(), args.save + "/max_recall_model.pt")

            if avg_precision_valid > max_precision_valid:
                is_max_precision_model = 1
                max_precision_valid = avg_precision_valid
                patient_epoch = 0
                torch.save(model.state_dict(), args.save + "/max_precision_model.pt")

            if avg_f1_valid > max_f1_valid:
                is_max_f1_model = 1
                max_f1_valid = avg_f1_valid
                patient_epoch = 0
                torch.save(model.state_dict(), args.save + "/max_f1_model.pt")
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


@timeit
def _train(train_loader, model, optimizer, loss_function, batch_size):
    print("\n=== TRAINING ===\n")
    loss_sum, precision_sum, recall_sum, f1_sum = instantiate_metric_variables()
    test_mode_index = 0  # used just for confirming training are running properly

    for i, loaded_data in enumerate(train_loader):
        a = time.time()
        features, labels, original_sequence_lengths = loaded_data

        features = features.float()  # shape: batch_size x sequence_length x feature_length
        original_sequence_lengths  # shape: batch_size

        if features.shape[0] != batch_size:
            continue

        outputs = _get_predictions_from_model(model, features, original_sequence_lengths)

        labels = labels.float().to(device)  # shape: batch_size
        loss = loss_function(outputs, labels)

        model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        recall, precision, f1 = _get_evaluation_metric(outputs, labels)

        loss_sum += loss
        recall_sum += recall
        precision_sum += precision
        f1_sum += f1

        if IS_TEST_MODE:
            print(f'iteration {test_mode_index}')
            test_mode_index += 1
            if test_mode_index == TEST_MODE_ITERATION_LIMIT:
                break
        b = time.time()
        loop_time_taken = np.around(b - a, decimals=2)
        print(f'{i}th loop, time taken: {loop_time_taken}s')
    return (
        loss_sum.cpu().detach().numpy() / len(train_loader),
        recall_sum.cpu().detach().numpy() / len(train_loader),
        precision_sum.cpu().detach().numpy() / len(train_loader),
        f1_sum.cpu().detach().numpy() / len(train_loader)
    )


@timeit
def _validate(valid_loader, model, loss_function, batch_size):
    print("\n=== Validation ===\n")
    loss_sum, precision_sum, recall_sum, f1_sum = instantiate_metric_variables()
    test_mode_index = 0  # used just for confirming training are running properly
    for i, loaded_data in valid_loader:
        a = time.time()
        features, labels, original_sequence_lengths = loaded_data

        features = features.float()  # shape: batch_size x sequence_length x feature_length
        original_sequence_lengths  # shape: batch_size

        if features.shape[0] != batch_size:
            continue

        outputs = _get_predictions_from_model(model, features, original_sequence_lengths)

        labels = labels.float().to(device)  # shape: batch_size
        loss = loss_function(outputs, labels)

        loss_sum, precision_sum, recall_sum, f1_sum = update_evaluation_metric(
            outputs, labels, loss, loss_sum,
            precision_sum, recall_sum, f1_sum
        )

        if IS_TEST_MODE:
            print(f'Valid iteration {test_mode_index}')
            test_mode_index += 1
            if test_mode_index == TEST_MODE_ITERATION_LIMIT:
                break
        b = time.time()
        loop_time_taken = np.around(b - a, decimals=2)
        print(f'{i}th loop, time taken: {loop_time_taken}s')
    return (
        loss_sum.cpu().detach().numpy() / len(train_loader),
        recall_sum.cpu().detach().numpy() / len(train_loader),
        precision_sum.cpu().detach().numpy() / len(train_loader),
        f1_sum.detach().cpu().numpy() / len(train_loader)
    )


def update_evaluation_metric(outputs, labels, loss, loss_sum, precision_sum, recall_sum, f1_sum):
    recall, precision, f1 = _get_evaluation_metric(outputs, labels)
    loss_sum += loss
    recall_sum += recall
    precision_sum += precision
    f1_sum += f1
    return loss_sum, precision_sum, recall_sum, f1_sum


def instantiate_metric_variables():
    loss_sum = torch.tensor(0).float().to(device)
    recall_sum = torch.tensor(0).float().to(device)
    precision_sum = torch.tensor(0).float().to(device)
    f1_sum = torch.tensor(0).float().to(device)
    return loss_sum, precision_sum, recall_sum, f1_sum


def _get_predictions_from_model(model, features, original_sequence_lengths):
    """
    Wraps the features with `rnn.pack_padded_sequence`.
    `rnn.pack_padded_sequence` returns a PackedSequence instance that helps the rnn to
    compute only on unpadded data
    """
    features = Variable(features)
    input = torch.nn.utils.rnn.pack_padded_sequence(
        features,
        original_sequence_lengths,
        enforce_sorted=False,
        batch_first=True).to(device)  # type: PackedSequence
    outputs = model(input)  # batch_size
    return outputs  # batch_size


def _get_evaluation_metric(
        outputs,  # batch_size x 1
        labels,  # batch_size x 1
):
    rounded_preds = torch.where(outputs > 0.5, 1, 0)

    fn = torch.tensor(0).to(device)
    fp = torch.tensor(0).to(device)
    tp = torch.tensor(0).to(device)

    for i in range(len(rounded_preds)):
        if rounded_preds[i] == 0 and labels[i] == 1:
            fn += 1
        elif rounded_preds[i] == 1 and labels[i] == 0:
            fp += 1
        elif rounded_preds[i] == 1 and labels[i] == 1:
            tp += 1

    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    return recall, precision, f1


if __name__ == "__main__":
    # Hyper-parameters
    num_classes = 1
    num_epochs = 30000
    batch_size = BATCH_SIZE
    learning_rate = 0.0001
    num_layers = 2
    hidden_size_multipier = 5

    # Create directories
    args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    create_dir(args.save)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    dataset_name = args.dataset_name
    train_data = SniperDataset(
        split='train',
        dataset_name=dataset_name,
    )

    valid_data = SniperDataset(
        split='valid',
        dataset_name=dataset_name
    )

    train_loader = DataLoader(train_data, num_workers=10, shuffle=True, batch_size=BATCH_SIZE)
    valid_loader = DataLoader(valid_data, num_workers=10, shuffle=True, batch_size=BATCH_SIZE)

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
        valid_loader,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
