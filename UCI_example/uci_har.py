import numpy as np
import argparse
import json
import os
import random
import torch
import torch.utils.data
import sys


class UciHarDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, data_folder, split='train', segment_length=20, normalize=True):
        if split == 'train':
            features = np.loadtxt(data_folder + "train/X_train.txt")  # shape: 7352, 561
            labels = np.loadtxt(data_folder + "train/y_train.txt")  # shape: 7352, 1
            subjects = np.loadtxt(data_folder + "train/subject_train.txt")  # shape: 7352, 1

            features = features[subjects < max(subjects) - 2, :]  # shape: 6243, 561
            labels = labels[subjects < max(subjects) - 2]  # shape: 6243, 1
            subjects = subjects[subjects < max(subjects) - 2]  # shape: 6243, 1

            if normalize:
                mean_train = np.mean(features, axis=0)
                std_train = np.std(features, axis=0)

                features = (features - mean_train) / std_train
        elif split == 'valid':
            features = np.loadtxt(data_folder + "train/X_train.txt")
            labels = np.loadtxt(data_folder + "train/y_train.txt")
            subjects = np.loadtxt(data_folder + "train/subject_train.txt")

            features = features[subjects >= max(subjects) - 2, :]
            labels = labels[subjects >= max(subjects) - 2]
            subjects = subjects[subjects >= max(subjects) - 2]

            if normalize:
                features_train = np.loadtxt(data_folder + "train/X_train.txt")
                mean_train = np.mean(features_train, axis=0)
                std_train = np.std(features_train, axis=0)

                features = (features - mean_train) / std_train
        elif split == 'test':
            features = np.loadtxt(data_folder + "test/X_test.txt")
            labels = np.loadtxt(data_folder + "test/y_test.txt")
            subjects = np.loadtxt(data_folder + "test/subject_test.txt")

            if normalize:
                features_train = np.loadtxt(data_folder + "train/X_train.txt")
                mean_train = np.mean(features_train, axis=0)
                std_train = np.std(features_train, axis=0)

                features = (features - mean_train) / std_train

        self.num_classes = int(np.max(labels))
        labels = labels - 1  # convert from 1 -- C to 0 -- C-1

        # Divide based on subjects
        features_divided = []
        labels_divided = []
        for i in np.unique(subjects):
            features_i = features[subjects == i, :]  # shape: ~347, 561
            labels_i = labels[subjects == i]  # shape: ~347
            features_divided.append(features_i)
            labels_divided.append(labels_i)

        # Time is not the same length
        features_divided = np.array(features_divided)  # shape: 18*, ~347, 561

        labels_divided = np.array(labels_divided)  # shape: 18*, ~347, 1

        self.segment_length = segment_length

        self.features = features_divided  # shape: 18*, ~347, 561
        self.labels = labels_divided  # shape: 18*, ~347, 1

        # *(not really like this, its actually an array of size 18)

    def __getitem__(self, index):
        feature = self.features[index]  # shape: ~347, 561
        label = self.labels[index]  # shape: ~347, 1

        # Take segment
        if len(feature) >= self.segment_length:
            max_start = len(feature) - self.segment_length
            start = random.randint(0, max_start)
            feature = (
                feature[start: start + self.segment_length, :]  # shape: self.segment_length=200, 561
            )
            label = (
                label[start: start + self.segment_length]  # shape: self.segment_length=200, 1
            )
        else:
            raise ValueError('Segment length too large')
            # feature = torch.nn.functional.pad(feature, (0, self.segment_length - len(feature)), 'constant').data
            # label = torch.nn.functional.pad(label, (0, self.segment_length - len(feature)), 'constant').data

        return (feature, label)

    def __len__(self):
        return self.features.shape[0]
