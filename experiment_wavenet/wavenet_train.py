import argparse
import multiprocessing
import os
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from experiment_wavenet.wavenet_dataset import WaveNetDataset, OPEN_COLUMN_INDEX, IS_MARKET_OPEN_INDEX
from experiment_wavenet.wavenet_directories import RUNS_DIR
from experiment_wavenet.wavenet_model import WaveNetModel, ProfitLoss
from utils import create_dir

multiprocessing.set_start_method("spawn", True)

# Arguments
parser = argparse.ArgumentParser(description='Wavenet Train')

# General Settings
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--load', type=str, default='', help='experiment name')
parser.add_argument('--save', type=str, default='Debug', help='experiment name')

args = parser.parse_args()

BATCH_SIZE = 10


def train(
        wavenet_model,  # type: WaveNetModel
        train_loader,  # type: DataLoader
        valid_loader,  # type: DataLoader
        batch_size=BATCH_SIZE,  # type: int
        num_epochs=30000,  # type: int
        patience=30000,  # type: int
        min_delta=0.00001,  # type: float
        learning_rate=0.0001  # type: float
):
    loss_function = ProfitLoss
    optimizer = torch.optim.RMSprop(
        wavenet_model.parameters(),
        lr=learning_rate
    )

    print('Model Structure: ', wavenet_model)
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
            wavenet_model,
            loss_function,
            optimizer,
            batch_size
        )

        losses_epoch_valid = _validate(
            valid_loader,
            wavenet_model,
            loss_function,
            batch_size
        )

        torch.save(wavenet_model.state_dict(), args.save + '/latest_model.pt')

        avg_losses_epoch_train = sum(losses_epoch_train).cpu().detach().numpy() / float(len(losses_epoch_train))
        avg_losses_epoch_valid = sum(losses_epoch_valid).cpu().detach().numpy() / float(len(losses_epoch_valid))
        average_epoch_losses_train.append(avg_losses_epoch_train)
        average_epoch_losses_valid.append(avg_losses_epoch_valid)

        # Early Stopping
        if epoch == 0:
            is_best_model = 1
            best_model = wavenet_model
            if avg_losses_epoch_valid < min_loss_epoch_valid:
                min_loss_epoch_valid = avg_losses_epoch_valid
        else:
            if min_loss_epoch_valid - avg_losses_epoch_valid > min_delta:
                is_best_model = 1
                best_model = wavenet_model
                min_loss_epoch_valid = avg_losses_epoch_valid
                patient_epoch = 0

                torch.save(wavenet_model.state_dict(), args.save + "/best_model.pt")
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


def _train(train_loader, wavenet_model, loss_function, optimizer, batch_size):
    losses_epoch_train = []
    for features, original_sequence_lengths in train_loader:
        features  # shape: batch_size x feature_length x sequence_length
        original_sequence_lengths  # shape: batch_size

        if features.shape[0] != batch_size:
            continue

        trades = get_trades_from_model(
            features=features,
            model=wavenet_model
        )

        open_prices = features[:, :, OPEN_COLUMN_INDEX]
        is_premarket = features[:, :, IS_MARKET_OPEN_INDEX]

        loss_train = compute_loss(
            loss_function=loss_function,
            trades=trades,
            open_prices=open_prices,
            original_sequence_lengths=original_sequence_lengths,
            is_premarket=is_premarket,
            multiply=args.multiply
        )

        wavenet_model.zero_grad()
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        if args.multiply:
            loss_train = -(torch.exp(-loss_train) - 1)
        losses_epoch_train.append(loss_train)
    return losses_epoch_train


def compute_loss(
        loss_function,
        trades,  # batch_size x seq_len
        open_prices,  # batch_size x seq_len
        original_sequence_lengths,  # batch_size
        is_premarket=None,
        multiply=False
):
    num_sequences = 0
    losses = []
    for batch_index, osl in enumerate(original_sequence_lengths):
        # BUG: Sometimes osl is 0
        if osl <= 1:
            continue
        current_outputs = trades[batch_index, 0: osl].float()
        current_prices = open_prices[batch_index, 0: osl].float()
        current_is_premarket = is_premarket[batch_index, 0: osl].float()

        current_loss = loss_function(
            current_outputs,
            current_prices,
            current_is_premarket,
            args.next_trade
        )

        losses.append(current_loss)
        num_sequences += 1
    if multiply:
        eps = 1e-15
        logsum = 0
        for loss in losses:
            change = -loss + 1  # change is in range [0, inf]
            log_change = torch.log(change + eps)
            logsum += log_change
        total_loss = -logsum
    else:
        total_loss = 0
        for loss in losses:
            total_loss += loss
        total_loss /= float(num_sequences)
    return total_loss


def get_trades_from_model(
        features,  # size: batch_size x feature_length x sequence_length
        model  # type: WaveNetModel
):
    use_gpu = torch.cuda.is_available()
    features = features.float()

    if use_gpu:
        features = Variable(features.cuda())
    else:
        features = Variable(features)

    trades = model(features)  # batch_size x sequence_length

    return trades


def _validate(valid_loader, trader_gru_model, loss_function, batch_size):
    losses_epoch_valid = []
    for features, original_sequence_lengths_val in valid_loader:
        if features.shape[0] != batch_size:
            continue

        trades = get_trades_from_model(
            features=features,
            model=trader_gru_model
        )

        open_prices = features[:, :, 0]
        is_premarket = features[:, :, IS_MARKET_OPEN_INDEX]

        loss_val = compute_loss(
            loss_function=loss_function,
            trades=trades,
            open_prices=open_prices,
            original_sequence_lengths=original_sequence_lengths_val,
            is_premarket=is_premarket,
            multiply=args.multiply
        )

        if args.multiply:
            loss_val = -(torch.exp(-loss_val) - 1)
        losses_epoch_valid.append(loss_val)
    return losses_epoch_valid


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    """
    Flow: 
        - The dataset outputs 10 segments of 9, 390
        - The model outputs 390 transactions 
        - Use ProfitLoss to compute the loss 
    """
    train_dataset = WaveNetDataset(
        dataset_name='polygon_early_day_gap_segmenter_parallel',
        split='train'
    )
    test_dataset = WaveNetDataset(
        dataset_name='polygon_early_day_gap_segmenter_parallel',
        split='valid'
    )

    train_loader = DataLoader(train_dataset, num_workers=1, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, num_workers=1, shuffle=True, batch_size=BATCH_SIZE)

    inputs, original_sequence_lengths = next(iter(train_loader))

    inputs  # 10, 9, 390 -> 10 samples of (feature_length, sequence_length)

    model = WaveNetModel(
        feature_length=inputs.shape[1],
    )

    # Create directories
    if args.load:
        model_path = os.path.join('runs', args.load, 'best_model.pt')
        model.load_state_dict(torch.load(model_path,
                                         map_location=lambda storage, loc: storage))

    create_dir(RUNS_DIR)
    args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    args.save = os.path.join(RUNS_DIR, args.save)
    create_dir(args.save)

    if torch.cuda.is_available():
        model = model.cuda()

    train(model, train_loader, test_loader)
