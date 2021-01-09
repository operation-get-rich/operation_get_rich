import csv
import json
import os

import alpaca_trade_api as tradeapi

from config import ALPACA_KEY_ID, ALPACA_SECRET_KEY, ALPACA_BASE_URL
from utils import create_dir, get_current_directory, \
    update_state, DATETIME_FORMAT
from decorators import retry_download

api = tradeapi.REST(
    key_id=ALPACA_KEY_ID,
    secret_key=ALPACA_SECRET_KEY,
    base_url=ALPACA_BASE_URL,
)

STATE_FILE_LOCATION = f'{get_current_directory(__file__)}/polygon_tick_by_tick_downloader_state.json'
raw_gaped_up_dir = '../datas/polygon_early_day_gap_segmenter_parallel'

save_root_dir = '../datas/polygon_gapped_up_tick_by_tick'
create_dir(save_root_dir)


def _download_quotes(symbol, the_date):
    quotes = []
    lowerbound_timestamp = None
    lowerbound_timestamp_obj = None  # For logging only
    lowerbound_timestamps = []
    quotes.append([
        'sip_timestamp',
        'bid_price',
        'bid_size',
        'bid_exchange',
        'ask_price',
        'ask_size',
        'ask_exchange'
    ])  # Column names
    while True:
        if lowerbound_timestamp in lowerbound_timestamps:
            # market close early, lowerbound_timestamp repeats indefinitely
            return quotes
        print(f'Downloading {symbol}, {the_date}, lowerbound_timestamp={lowerbound_timestamp_obj}')
        symbol_quotes = _history_quote_v2_wrapper(lowerbound_timestamp, symbol, the_date)

        for quote in symbol_quotes:
            quotes.append([
                quote.sip_timestamp.strftime(DATETIME_FORMAT),
                quote.bid_price,
                quote.bid_size,
                quote.bid_exchange,
                quote.ask_price,
                quote.ask_size,
                quote.ask_exchange
            ])

        if symbol_quotes[-1].sip_timestamp.hour < 16:
            lowerbound_timestamps.append(lowerbound_timestamp)
            lowerbound_timestamp = symbol_quotes[-1]._raw['sip_timestamp']
            lowerbound_timestamp_obj = symbol_quotes[-1].sip_timestamp  # for logging only
        else:
            break
    return quotes

@retry_download
def _history_quote_v2_wrapper(lowerbound_timestamp, symbol, the_date):
    symbol_quotes = api.polygon.historic_quotes_v2(
        symbol=symbol,
        date=the_date,
        timestamp=lowerbound_timestamp
    )
    return symbol_quotes


def update_finished_file_state(state, the_date, symbol):
    if not 'finished' in state:
        state['finished'] = []
    state['finished'].append(f'{the_date}/{symbol}')


def main():
    with open('./polygon_tick_by_tick_downloader_state.json') as state_file:
        state = json.load(state_file)

    finished_segments = set(state['finished']) if 'finished' in state else {}

    for date_dir in sorted(os.listdir('%s' % raw_gaped_up_dir)):
        for stock_filename in sorted(os.listdir(f'{raw_gaped_up_dir}/{date_dir}')):
            symbol = stock_filename[0:-15]
            the_date = date_dir

            if f'{the_date}/{symbol}' in finished_segments:
                continue

            print(f'Downloading {symbol} @ {the_date}')

            quotes = _download_quotes(symbol, the_date)

            create_dir(f'{save_root_dir}/{the_date}')
            stock_tick_by_tick_csv = f'{save_root_dir}/{the_date}/{symbol}.csv'
            with open(stock_tick_by_tick_csv, 'w') as f:
                print(f'Writing into {stock_tick_by_tick_csv}')
                writer = csv.writer(f)
                for quote in quotes:
                    writer.writerow(quote)
            update_state(STATE_FILE_LOCATION, update_finished_file_state, the_date, symbol)
            print(f'Finished Writing into {stock_tick_by_tick_csv}')


main()
