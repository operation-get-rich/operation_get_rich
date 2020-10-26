import os
from shutil import copyfile, copy

root_dir = '../gaped_up_stocks_early_volume_1e5_gap_10'

by_date_dir = '../gaped_up_stocks_early_volume_1e5_gap_10_by_date'
if not os.path.exists(by_date_dir):
    os.mkdir(by_date_dir)

for company in os.listdir('%s' % root_dir):
    company_dir = f'{root_dir}/{company}/'
    for stock_file in os.listdir(company_dir):
        stock_date = stock_file.split('_')[1].split()[0]
        date_dir = os.path.join(by_date_dir, stock_date)
        if not os.path.exists(date_dir):
            os.mkdir(date_dir)
        copy(
            src=os.path.join(company_dir, stock_file),
            dst=os.path.join(date_dir)
        )
