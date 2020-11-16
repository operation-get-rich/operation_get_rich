import os
from shutil import copyfile

from utils import create_dir

root_dir = './gaped_up_stocks_early_volume_1e5_gap_10'
dest_root_dir = './gaped_up_stocks_early_volume_1e5_gap_10_by_date'

for stock_dir in sorted(os.listdir(root_dir)):
    for stock_file in sorted(os.listdir(f'{root_dir}/{stock_dir}')):
        date = stock_file[-25:-15]
        create_dir(f'{dest_root_dir}/{date}')
        copyfile(f'{root_dir}/{stock_dir}/{stock_file}',
                 f'{dest_root_dir}/{date}/{stock_file}')
