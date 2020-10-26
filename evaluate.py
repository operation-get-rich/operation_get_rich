import argparse
import os
import time

import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from VanillaGRU import VanillaGRU
from stock_dataset import StockDataset

import multiprocessing

from plotting import plot_single_trajectory

multiprocessing.set_start_method("spawn", True)


# Arguments
parser = argparse.ArgumentParser(description='VanillaGRU Train')

# General Settings
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--load', type=str, default='Train-20201021-105752', help='experiment name')
args = parser.parse_args()

MODEL_OUTPUT_SIZE = 4  # The model predicts: open, close, high, & low of the next bar

def feed_data(data, model):
    use_gpu = torch.cuda.is_available()
    inputs, original_sequence_lengths = data
    inputs = inputs.float()

    # labels contains open, close, low, high
    targets = inputs[:, 1:, :4]  # batch_size x  seq_len-1 x output_size
    inputs = inputs[:, :-1, :]  # batch_size x  seq_len-1 x input_size

    if use_gpu:
        inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
    else:
        inputs, targets = Variable(inputs), Variable(targets)

    outputs = model(inputs)  # seq_len-1 x batch_size x output_size
    outputs = torch.stack(outputs)
    outputs = outputs.permute(1, 0, 2) # batch_size x seq_len-1 x output_size

    return outputs, targets

def evaluate(
        model,  # type: VanillaGRU
        valid_loader,  # type: DataLoader
):
    print('Model Structure: ', model)
    print('Start Training ... ')

    inputs, original_sequence_lengths = next(iter(valid_loader))

    # Variables for Early Stopping

    for index, data_val in enumerate(valid_loader):
        inputs_val, original_sequence_lengths_val = data_val
        outputs_val, targets_val = feed_data(data_val, model)

        save_dir = os.path.join(args.load, 'plots', str(index) + '.pdf')
        plot_single_trajectory(outputs_val, targets_val, save_dir=save_dir)

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == "__main__":
    # Create directories

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)


    test_data = StockDataset(
        data_folder='./gaped_up_stocks_early_volume_1e5_gap_10',
        split='test',
        should_add_technical_indicator=True
    )

    test_loader = DataLoader(test_data, num_workers=1, shuffle=True, batch_size=1)

    inputs, original_sequence_lengths = next(iter(test_loader))
    [batch_size, seq_length, num_features] = inputs.size()

    model = VanillaGRU(
        input_size=num_features,
        hidden_size=5 * num_features,
        output_size=MODEL_OUTPUT_SIZE
    )
    model.load_state_dict(torch.load(args.load + "/best_model.pt", 
                            map_location="cuda:{}".format(args.gpu)))
    model = model.cuda()

    evaluate(model, test_loader)
