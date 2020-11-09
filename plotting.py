import os

import pandas as pd
import mplfinance as mpf

import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt


def plot_single_trajectory(predictions, targets, save_dir):
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    predictions = predictions.squeeze()
    targets = targets.squeeze()

    pred_df = pd.DataFrame(predictions, columns=['Open', 'Close', 'Low', 'High'])
    target_df = pd.DataFrame(targets, columns=['Open', 'Close', 'Low', 'High'])

    mpf.plot(pred_df, type='candle', style='yahoo', volume=True)

    time = range(len(predictions))

    fig = plt.figure()
    plt.plot(time, targets, label="True")
    plt.plot(time, predictions, label="Prediction")
    plt.xlim([0, len(time)])
    plt.xlabel('Time (Minutes)')
    plt.xticks([i for i in time])
    plt.grid()
    plt.legend()

    save_dir = os.path.join(save_dir)
    plt.savefig(save_dir)
    plt.close()
