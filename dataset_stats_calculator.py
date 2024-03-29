import argparse
import json
import os

import pandas
import pandas as pd

from directories import DATA_DIR

parser = argparse.ArgumentParser(description='Dataset Stats Calculator')

parser.add_argument('--dataset-name', required=True, type=str)
args = parser.parse_args()


def company_segment_organized_parser(root_data_dir):
    """
    directory structure:
        root_data_dir/stock_a/stock_a_date_a.csv
        root_data_dir/stock_a/stock_a_date_b.csv
        root_data_dir/stock_b/stock_b_date_a.csv
        root_data_dir/stock_b/stock_b_date_b.csv
    """
    entire_segment_stats = []
    for company in os.listdir(root_data_dir):
        for segment in os.listdir(os.path.join(root_data_dir, company)):
            segment_file_path = os.path.join(root_data_dir, company, segment)
            print(f'Parsing {segment_file_path}')
            df = pd.read_csv(segment_file_path)
            entire_segment_stats.append(df.describe().T)
    entire_segment_stats_df = pandas.concat(entire_segment_stats)
    entire_segment_stats_df = entire_segment_stats_df.groupby(
        entire_segment_stats_df.index).agg(
        {
            'mean': 'mean',
            'std': 'std',
            'min': 'min',
            '25%': 'mean',
            '50%': 'mean',
            '75%': 'mean',
            'max': 'max'
        }
    )
    return entire_segment_stats_df.T


def date_segment_organized_parser(root_data_dir):
    """
    directory structure:
        root_data_dir/date_a/stock_a.csv
        root_data_dir/date_b/stock_b.csv
    """
    entire_segment_stats = []
    for the_date in sorted(os.listdir(root_data_dir)):
        for segment in sorted(os.listdir(os.path.join(root_data_dir, the_date))):
            segment_file_path = os.path.join(root_data_dir, the_date, segment)
            print(f'Parsing {segment_file_path}')
            df = pd.read_csv(segment_file_path)
            entire_segment_stats.append(df.describe().T)
    entire_segment_stats_df = pandas.concat(entire_segment_stats)
    entire_segment_stats_df = entire_segment_stats_df.groupby(
        entire_segment_stats_df.index).agg(
        {
            'mean': 'mean',
            'std': 'std',
            'min': 'min',
            '25%': 'mean',
            '50%': 'mean',
            '75%': 'mean',
            'max': 'max'
        }
    )

    return entire_segment_stats_df.T


def segment_organized_parser(root_data_dir):
    """
    directory structure:
        root_data_dir/stock_a.csv
        root_data_dir/stock_b.csv
        ...
    """
    entire_segment_stats = []
    for segment in os.listdir(os.path.join(root_data_dir)):
        segment_file_path = os.path.join(root_data_dir, segment)
        print(f'Parsing {segment_file_path}')

        df = pd.read_csv(segment_file_path)

        entire_segment_stats.append(df.describe().T)

    entire_segment_stats_df = pandas.concat(entire_segment_stats)
    entire_segment_stats_df = entire_segment_stats_df.groupby(
        entire_segment_stats_df.index).agg(
        {
            'mean': 'mean',
            'std': 'std',
            'min': 'min',
            '25%': 'mean',
            '50%': 'mean',
            '75%': 'mean',
            'max': 'max'
        }
    )

    return entire_segment_stats_df.T


dataset_name_to_parser = {
    'gaped_up_stocks_early_volume_1e5_gap_10': company_segment_organized_parser,
    'polygon_early_day_gap_segmenter_parallel': date_segment_organized_parser,
    'sniper_training_data_3': segment_organized_parser,
    'sniper_training_data_balanced_1': segment_organized_parser,
}


def compute_stats(dataset_name):
    root_dir = os.path.join(DATA_DIR, dataset_name)

    entire_segments_descriptive_stats = dataset_name_to_parser[dataset_name](
        root_dir)  # parsing the directory to get the statistics of the dataset

    stats_dict = {}
    for key in entire_segments_descriptive_stats.keys():
        if key not in stats_dict:
            stats_dict[key] = {}
        for inner_key in entire_segments_descriptive_stats[key].keys():
            stats_dict[key][inner_key] = entire_segments_descriptive_stats[key][inner_key]

    try:
        with open('dataset_statistics.json', 'r') as f:
            statistics_dataset = json.load(f)
    except FileNotFoundError as e:
        statistics_dataset = {}

    statistics_dataset[dataset_name] = stats_dict

    with open('dataset_statistics.json', 'w') as f:
        json.dump(statistics_dataset, f)


compute_stats(args.dataset_name)
