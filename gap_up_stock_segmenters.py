import pandas
from pandas import DataFrame

from utils import create_dir, get_date, get_date_time, get_date_string

GAP_UP_THRESHOLD = 0.15
VOLUME_THRESHOLD = 1e+06
RAW_STOCK_PRICE_SOURCE_FILENAME = 'stock_price_small.csv'
GAPED_UP_STOCKS_DIR_NAME = 'gaped_up_stocks'


def is_gapped_up(open_price, close_price):
    return (open_price - close_price) / close_price > GAP_UP_THRESHOLD


def find_segments():
    print("Loading Data:", flush=True)

    stock_price_df = pandas.read_csv(RAW_STOCK_PRICE_SOURCE_FILENAME)  # type: DataFrame
    stock_price_df['open'] = pandas.to_numeric(stock_price_df['open'])
    stock_price_df['close'] = pandas.to_numeric(stock_price_df['close'])
    stock_price_df['low'] = pandas.to_numeric(stock_price_df['low'])
    stock_price_df['high'] = pandas.to_numeric(stock_price_df['high'])
    stock_price_df['volume'] = pandas.to_numeric(stock_price_df['volume'])

    print("Finding Segments:", flush=True)

    segments = []  # [('AAPL', '01/17/20'), ('AAPL', '01/18/20')]

    current_ticker = None
    previous_ticker = None
    open_price = None
    close_price = None
    cummulative_volume = 0

    for index in reversed(range(len(stock_price_df.index))):
        row = stock_price_df.loc[index]
        # Reset everything when new ticker arrives
        if current_ticker is None or row.ticker != previous_ticker:
            current_ticker = row.ticker
            open_price = None
            close_price = None
            cummulative_volume = 0

        print("Current Ticker: ", current_ticker, flush=True)
        print("Current Index: ", index, flush=True)

        if get_date_time(row.time).hour < 10 and open_price is None:
            open_price = row.open

        if open_price is not None:
            cummulative_volume += row.volume

        if index - 1 > 0:
            if open_price is not None and get_date(row.time) > get_date(stock_price_df.loc[index - 1].time):
                close_price = stock_price_df.loc[index - 1].open
                if is_gapped_up(open_price, close_price) and cummulative_volume >= VOLUME_THRESHOLD:
                    segments.append((current_ticker, get_date(row.time)))
                    print("Gapped Up: ", current_ticker, get_date(row.time), flush=True)

                open_price = None
                close_price = None
                cummulative_volume = 0
        previous_ticker = current_ticker
    return segments


def save_segments(segments):
    create_dir(GAPED_UP_STOCKS_DIR_NAME)
    stock_price_df['just_date'] = stock_price_df.apply(lambda row: get_date(row.time), axis=1)
    for ticker_segment, date_segment in segments:
        print("Writing Ticker: ", ticker_segment, flush=True)
        get_date_string(date_segment)
        the_segment = stock_price_df[
            (stock_price_df.ticker == ticker_segment) &
            (stock_price_df.just_date == date_segment)
            ]  # type: DataFrame
        the_segment = the_segment.drop('just_date', axis=1)
        the_segment = the_segment.drop(the_segment.columns[0], axis=1)

        the_dir = create_dir('./{}/{}'.format(GAPED_UP_STOCKS_DIR_NAME, ticker_segment))

        the_segment.to_csv(path_or_buf='{}/{}_{}'.format(the_dir, ticker_segment, date_segment), index=False)


segments = find_segments()
save_segments(segments)
