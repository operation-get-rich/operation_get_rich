exchange_file = open('NASDAQ.txt')

lines = exchange_file.readlines()
stock_names = []
for line in lines:
    stock_names.append(line.strip().split()[0])

with open('NASDAQ_only_symbols.txt', 'a') as the_file:
    for name in stock_names:
        the_file.write(name + '\n')

exchange_file = open('NYSE.txt')

lines = exchange_file.readlines()
stock_names = []
for line in lines:
    stock_names.append(line.strip().split()[0])

with open('NYSE_only_symbols.txt', 'a') as the_file:
    for name in stock_names:
        the_file.write(name + '\n')

