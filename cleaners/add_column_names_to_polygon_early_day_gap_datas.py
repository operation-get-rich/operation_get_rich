import os

import pandas

from directories import DATA_DIR

data_source = 'polygon_early_day_gap_segmenter_parallel'
data_source_dir = f'{DATA_DIR}/{data_source}'

for the_date in sorted(os.listdir(data_source_dir)):
    for stock_file in sorted(os.listdir(os.path.join(data_source_dir, the_date))):
        df = pandas.read_csv(os.path.join(data_source_dir, the_date, stock_file),
                             index_col=None)
        df.columns = ['ticker', 'time', 'open', 'close', 'low', 'high', 'volume']
        df.to_csv(os.path.join(data_source_dir, the_date, stock_file), index=None)
