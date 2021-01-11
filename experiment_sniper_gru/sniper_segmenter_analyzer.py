import os

from directories import DATA_DIR

root_dir = f'{DATA_DIR}/sniper_training_data_1'
zero_count = 0
one_count = 0
for file in os.listdir(root_dir):
    if '0' in file.split('.')[0][-2:]:
        zero_count += 1
    else:
        one_count += 1

print(f'Zero Count {zero_count}')
print(f'One Count {one_count}')
