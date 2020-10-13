import os
import pandas
from pandas import DataFrame

from utils import get_date_time, get_date, create_dir, get_date_string

RAW_STOCK_PRICE_DIR = 'stock_prices'
def total_day_volume_and_high_of_day_gap(
        gap_up_threshold=0.35,
        volume_threshold=1e+05
):
    """
    Look for stocks that has its high of day price gapped up relative to the closing price of the day before.
    It also only looks if the total volume within the day is at least `volume_threshold`
    """
    raw_stock_list = os.listdir(RAW_STOCK_PRICE_DIR)
    for stock_file in raw_stock_list:
        stock_path = os.path.join(RAW_STOCK_PRICE_DIR, stock_file)

        print("Loading Data:", flush=True)
        stock_price_df = pandas.read_csv(stock_path)  # type: DataFrame

        current_ticker = None
        previous_ticker = None
        open_price = 0
        close_price = None
        cummulative_volume = 0

        segments = []  # [('AAPL', '01/17/20'), ('AAPL', '01/18/20')]
        print("Starting Segmentation:", flush=True)
        for index in reversed(range(len(stock_price_df.index))):
            row = stock_price_df.loc[index]

            # Reset everything when new ticker arrives
            if current_ticker is None or row.ticker != previous_ticker:
                current_ticker = row.ticker
                open_price = 0
                close_price = None
                cummulative_volume = 0
                print("Current Ticker: ", current_ticker, flush=True)
                print("Current Index: ", index, flush=True)

            open_price = max(row.close, open_price)

            cummulative_volume += row.volume

            if index - 1 > 0:
                if get_date(row.time) > get_date(stock_price_df.loc[index - 1].time):
                    close_price = stock_price_df.loc[index - 1].open
                    is_gapping_up = (open_price - close_price) / close_price > gap_up_threshold
                    print(f'Cummulative Volume = {cummulative_volume}')
                    if is_gapping_up and cummulative_volume >= volume_threshold:
                        segments.append((current_ticker, get_date(row.time)))
                        print("Gapped Up: ", current_ticker, get_date(row.time), flush=True)

                    open_price = 0
                    close_price = None
                    cummulative_volume = 0

            previous_ticker = current_ticker

        stock_price_df['just_date'] = stock_price_df.apply(lambda row: get_date(row.time), axis=1)

        create_dir('./%s' % GAPED_UP_STOCKS_DIR_NAME)
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


GAPED_UP_STOCKS_DIR_NAME = 'gaped_up_stocks_early_volume_1e5_gap_10'


def early_day_gap(
        gap_up_threshold=0.10,
        volume_threshold=1e+05,
):
    """
    Look for stocks that gapped up by `gap_up_threshold` and have before 10am volume of `volume_threshold`
    """
    raw_stock_list = os.listdir(RAW_STOCK_PRICE_DIR)
    for stock_file in raw_stock_list:
        stock_path = os.path.join(RAW_STOCK_PRICE_DIR, stock_file)

        print("Loading Data:", flush=True)
        stock_price_df = pandas.read_csv(stock_path)  # type: DataFrame

        current_ticker = None
        previous_ticker = None
        open_price = None
        close_price = None
        cummulative_volume = 0

        segments = []  # [('AAPL', '01/17/20'), ('AAPL', '01/18/20')]
        print("Starting Segmentation:", flush=True)
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
                    is_gapping_up = (open_price - close_price) / close_price > gap_up_threshold
                    if is_gapping_up and cummulative_volume >= volume_threshold:
                        segments.append((current_ticker, get_date(row.time)))
                        print("Gapped Up: ", current_ticker, get_date(row.time), flush=True)

                    open_price = None
                    close_price = None
                    cummulative_volume = 0

            previous_ticker = current_ticker

        stock_price_df['just_date'] = stock_price_df.apply(lambda row: get_date(row.time), axis=1)

        create_dir('./%s' % GAPED_UP_STOCKS_DIR_NAME)
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


total_day_volume_and_high_of_day_gap()
