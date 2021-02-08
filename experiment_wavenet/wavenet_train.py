import argparse
import multiprocessing

import torch

multiprocessing.set_start_method("spawn", True)

# Arguments
parser = argparse.ArgumentParser(description='Wavenet Train')

# General Settings
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--load', type=str, default='', help='experiment name')
parser.add_argument('--save', type=str, default='Debug', help='experiment name')

args = parser.parse_args()

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    train_dataset = NotImplemented