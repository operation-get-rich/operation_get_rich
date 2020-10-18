import argparse
import os
import time

import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from UCI_example.VanillaGRU import VanillaGRU
from stock_dataset import StockDataset

import multiprocessing

multiprocessing.set_start_method("spawn", True)

# Arguments
parser = argparse.ArgumentParser(description='VanillaGRU Train')

# General Settings
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--save', type=str, default='Train', help='experiment name')
args = parser.parse_args()


def train(model,
          train_loader,
          valid_loader,
          num_epochs=30000,
          patience=30000,
          min_delta=0.00001):
    print('Model Structure: ', model)
    print('Start Training ... ')

    # model.cuda()

    loss = torch.nn.CrossEntropyLoss()

    learning_rate = 0.0001
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99)
    use_gpu = torch.cuda.is_available()

    interval = 100
    losses_train = []
    losses_valid = []
    losses_epochs_train = []
    losses_epochs_valid = []

    cur_time = time.time()
    pre_time = time.time()

    inputs, labels = next(iter(train_loader))
    [batch_size, seq_length, input_size] = inputs.size()

    # Variables for Early Stopping
    is_best_model = 0
    patient_epoch = 0
    for epoch in range(num_epochs):

        trained_number = 0

        valid_dataloader_iter = iter(valid_loader)

        losses_epoch_train = []
        losses_epoch_valid = []

        for data in train_loader:
            inputs, labels = data
            inputs = inputs.float()

            labels = labels.long()
            labels = labels.permute(1, 0)  # seq_length x batch
            labels = labels.flatten()  # seq_length * batch

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            model.zero_grad()

            outputs = model(inputs)
            outputs = torch.cat(outputs)  # seq_length * batch x num_classes

            loss_train = loss(outputs, labels)
            loss_train = loss_train

            losses_train.append(loss_train.data)
            losses_epoch_train.append(loss_train.data)

            optimizer.zero_grad()

            loss_train.backward()

            optimizer.step()

            # Validation
            try:
                inputs_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_loader)
                inputs_val, labels_val = next(valid_dataloader_iter)

            inputs_val = inputs_val.float()
            labels_val = labels_val.long()
            labels_val = labels_val.permute(1, 0)  # seq_length x batch
            labels_val = labels_val.flatten()  # seq_length * batch

            if use_gpu:
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else:
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)

            model.zero_grad()

            outputs_val = model(inputs_val)
            outputs_val = torch.cat(outputs_val)  # seq_length * batch x num_classes

            loss_valid = loss(outputs_val, labels_val)
            loss_valid = loss_valid

            losses_valid.append(loss_valid.data)
            losses_epoch_valid.append(loss_valid.data)

            # output
            trained_number += 1

        torch.save(model.state_dict(), args.save + "/latest_model.pt")

        avg_losses_epoch_train = sum(losses_epoch_train).cpu().numpy() / float(len(losses_epoch_train))
        avg_losses_epoch_valid = sum(losses_epoch_valid).cpu().numpy() / float(len(losses_epoch_valid))
        losses_epochs_train.append(avg_losses_epoch_train)
        losses_epochs_valid.append(avg_losses_epoch_valid)

        # Early Stopping
        if epoch == 0:
            is_best_model = 1
            best_model = model
            min_loss_epoch_valid = 10000.0
            if avg_losses_epoch_valid < min_loss_epoch_valid:
                min_loss_epoch_valid = avg_losses_epoch_valid
        else:
            if min_loss_epoch_valid - avg_losses_epoch_valid > min_delta:
                is_best_model = 1
                best_model = model
                min_loss_epoch_valid = avg_losses_epoch_valid
                patient_epoch = 0

                torch.save(model.state_dict(), args.save + "/best_model.pt")
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
    torch.cuda.set_device(args.gpu)
    torch.cuda.set_device(-1)

    train_data = StockDataset(
        data_folder='./gapped_up_stocks',
        split='train',
        should_add_technical_indicator=True
    )

    test_data = StockDataset(
        data_folder='./gapped_up_stocks',
        split='test',
        should_add_technical_indicator=True
    )

    train_loader = DataLoader(train_data, num_workers=1, shuffle=True, batch_size=10)
    test_loader = DataLoader(test_data, num_workers=1, shuffle=True, batch_size=20)

    inputs, labels = next(iter(train_loader))
    [batch_size, seq_length, num_features] = inputs.size()

    inputs, labels = next(iter(train_loader))
    [batch_size, seq_length, num_features] = inputs.size()

    model = VanillaGRU(input_size=num_features, hidden_size=5 * num_features, output_size=num_classes)

    best_grud, losses_grud = train(model, train_loader, test_loader)
