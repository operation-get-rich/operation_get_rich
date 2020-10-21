import os

import pandas as pd

ROOT_DATA_DIR = './gaped_up_stocks_early_volume_1e5_gap_10'

entire_segments_df = pd.DataFrame()

for company in os.listdir(ROOT_DATA_DIR):
    for segment in os.listdir(os.path.join(ROOT_DATA_DIR, company)):
        segment_file_path = os.path.join(ROOT_DATA_DIR, company, segment)
        entire_segments_df = entire_segments_df.append(pd.read_csv(segment_file_path))

entire_segments_descriptive_stats = entire_segments_df.describe()

entire_segments_descriptive_stats.to_csv(f'{ROOT_DATA_DIR}_statistics.csv')