import json
import os

import pandas as pd

from directories import DATA_DIR


def company_segment_organized_parser(root_data_dir):
    entire_segments_df = pd.DataFrame()
    for company in os.listdir(root_data_dir):
        for segment in os.listdir(os.path.join(root_data_dir, company)):
            segment_file_path = os.path.join(root_data_dir, company, segment)
            entire_segments_df = entire_segments_df.append(pd.read_csv(segment_file_path))
    return entire_segments_df


def date_segment_organized_parser(root_data_dir):
    entire_segments_df = pd.DataFrame()
    for the_date in sorted(os.listdir(root_data_dir)):
        for segment in sorted(os.listdir(os.path.join(root_data_dir, the_date))):
            segment_file_path = os.path.join(root_data_dir, the_date, segment)
            entire_segments_df = entire_segments_df.append(pd.read_csv(segment_file_path))
    return entire_segments_df


def segment_organized_parser(root_data_dir):
    entire_segments_df = pd.DataFrame()
    for segment in os.listdir(os.path.join(root_data_dir)):
        segment_file_path = os.path.join(root_data_dir, segment)
        entire_segments_df = entire_segments_df.append(pd.read_csv(segment_file_path))
    return entire_segments_df


dataset_name_to_parser = {
    'gaped_up_stocks_early_volume_1e5_gap_10': company_segment_organized_parser,
    'polygon_early_day_gap_segmenter_parallel': date_segment_organized_parser,
    'sniper_training_data_3': segment_organized_parser,
}


def compute_stats(dataset_name):
    root_dir = os.path.join(DATA_DIR, dataset_name)
    entire_segments_df = dataset_name_to_parser[dataset_name](root_dir)

    entire_segments_descriptive_stats = entire_segments_df.describe()

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


compute_stats('polygon_early_day_gap_segmenter_parallel')
