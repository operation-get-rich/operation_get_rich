import os
import pandas
from pandas import DataFrame

from utils import get_date_time, get_date, create_dir, get_date_string

RAW_STOCK_PRICE_DIR = 'stock_prices'
GAPED_UP_STOCKS_DIR_NAME = 'gaped_up_stocks_early_volume_1e5_gap_10'


def total_day_volume_and_high_of_day_gap(
        gap_up_threshold=0.35,
        volume_threshold=1e+05,
        save_dir_name='./gaped_up_stocks_early_volume_1e5_gap_10'
):
    """
    Look for stocks that has its high of day price gapped up relative to the closing price of the day before.
    It also only looks if the total volume within the day is at least `volume_threshold`
    """
    raw_stock_list = os.listdir(RAW_STOCK_PRICE_DIR)
    for stock_file in raw_stock_list:
        stock_path = os.path.join(RAW_STOCK_PRICE_DIR, stock_file)
        df = pandas.read_csv(stock_path)

        tickers = df['ticker']

        for ticker in tickers:
            print(f'Scanning {ticker}')
            ticker_df = df[df['ticker'] == ticker]
            dates = ticker_df['time']
            dates = dates.apply(lambda time: get_date(time).strftime('%Y-%m-%d')).unique()
            dates = sorted(dates)
            for index in range(1, len(dates)):
                todays_date = dates[index]
                yesterdays_date = dates[index - 1]
                print(f'today\'s date: {todays_date}')
                print(f'yesterday\'s date: {yesterdays_date}')

                ticker_today_df = ticker_df[ticker_df['time'].str.contains(todays_date)]
                ticker_yesterday_df = ticker_df[ticker_df['time'].str.contains(yesterdays_date)]

                today_high_of_day = max(ticker_today_df['close'])
                yesterday_high_of_day = max(ticker_yesterday_df['close'])
                today_volume = ticker_today_df.sum().volume

                print(f'{ticker}\'s high of day: {today_high_of_day}')
                print(f'{ticker}\'s yesterday\'s high of day: {yesterday_high_of_day}')
                print(f'{ticker}\'s today\'s volume: {today_volume}')

                is_gapping_up = (today_high_of_day - yesterday_high_of_day) / yesterday_high_of_day > gap_up_threshold
                if today_volume >= volume_threshold and is_gapping_up:
                    print(f'Found Gap Up: {ticker} at {todays_date}')
                    ticker_today_df.to_csv(path_or_buf=f'{save_dir_name}/{ticker}/{ticker}_{todays_date}.csv')


def early_day_gap_refactor(
        gap_up_threshold=0.10,
        volume_threshold=0
):
    segments = []
    raw_stock_list = os.listdir(RAW_STOCK_PRICE_DIR)
    for stock_file in sorted(raw_stock_list):
        stock_path = os.path.join(RAW_STOCK_PRICE_DIR, stock_file)

        print("Loading Data: {}".format(stock_path), flush=True)
        stock_price_df = pandas.read_csv(stock_path, parse_dates=['time'], index_col=0)  # type: DataFrame

        prev_day_close_price = None
        cumulative_volume = 0

        for index in range(len(stock_price_df.index)):
            (ticker,
             time,
             open_price,
             close_price,
             low_price,
             high_price,
             volume) = stock_price_df.loc[index]

            if index + 1 < len(stock_price_df.index):
                (next_ticker,
                 next_time,
                 next_open_price,
                 next_close_price,
                 next_low_price,
                 next_high_price,
                 next_volume) = stock_price_df.loc[index + 1]

                if next_ticker != ticker:
                    prev_day_close_price = None
                    cumulative_volume = 0
                    continue

                if next_time.date() > time.date():
                    prev_day_close_price = close_price
                    continue

            if prev_day_close_price is not None:
                cumulative_volume += volume

            if prev_day_close_price is not None and (time.hour, time.minute) == (9, 30):
                is_price_gapped_up = (open_price / prev_day_close_price) - 1 > gap_up_threshold
                if is_price_gapped_up and cumulative_volume > volume_threshold:
                    segments.append((ticker, time.date))
                    print("Gapped Up: ", ticker, time.date(), cumulative_volume flush=True)
                prev_day_close_price = None
                cumulative_volume = 0


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


early_day_gap_refactor()
