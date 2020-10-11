def get_all_ticker_names():
    ticker_names = []
    ticker_names += get_ticker_names('NASDAQ_only_symbols.txt')
    ticker_names += get_ticker_names('NYSE_only_symbols.txt')

    return ticker_names


def get_ticker_names(file_name):
    ticker_names = []
    exchange_file = open(file_name)
    lines = exchange_file.readlines()
    for line in lines:
        ticker_names.append(line)
    return ticker_names




