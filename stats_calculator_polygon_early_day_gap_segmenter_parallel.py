import json
import os

import pandas as pd

from directories import DATA_DIR

DATASET_NAME = 'polygon_early_day_gap_segmenter_parallel'
ROOT_DATA_DIR = f'{DATA_DIR}/{DATASET_NAME}'

entire_segments_df = pd.DataFrame()

for the_date in sorted(os.listdir(ROOT_DATA_DIR)):
    for segment in sorted(os.listdir(os.path.join(ROOT_DATA_DIR, the_date))):
        segment_file_path = os.path.join(ROOT_DATA_DIR, the_date, segment)
        entire_segments_df = entire_segments_df.append(pd.read_csv(segment_file_path))

entire_segments_descriptive_stats = entire_segments_df.describe()

stats_dict = {}
for key in entire_segments_descriptive_stats.keys():
    if key not in stats_dict:
        stats_dict[key] = {}
    for inner_key in entire_segments_descriptive_stats[key].keys():
        stats_dict[key][inner_key] = entire_segments_descriptive_stats[key][inner_key]

try:
    with open('./statistics_dataset.json', 'r') as f:
        statistics_dataset = json.load(f)
except FileNotFoundError as e:
    statistics_dataset = {}

statistics_dataset[DATASET_NAME] = stats_dict

with open('./statistics_dataset.json', 'w') as f:
    json.dump(statistics_dataset, f)
