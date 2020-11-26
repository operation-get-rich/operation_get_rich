import csv
import os

import alpaca_trade_api as tradeapi

from config import ALPACA_KEY_ID, ALPACA_SECRET_KEY, ALPACA_BASE_URL
from utils import create_dir, convert_pandas_timestamp_to_formatted_string, retry_download

api = tradeapi.REST(
    key_id=ALPACA_KEY_ID,
    secret_key=ALPACA_SECRET_KEY,
    base_url=ALPACA_BASE_URL,
)

raw_gaped_up_dir = '../datas/alpaca_gaped_up_stocks_early_volume_1e5_gap_10_by_date'

save_root_dir = '../datas/alpaca_gapped_up_tick_by_tick'
create_dir(save_root_dir)


def _download_quotes(symbol, the_date):
    quotes = []
    lowerbound_timestamp = None
    quotes.append([
        'sip_timestamp',
        'bid_price',
        'bid_size',
        'bid_exchange',
        'ask_price',
        'ask_size',
        'ask_exchange'
    ])
    while True:
        symbol_quotes = _history_quote_v2_wrapper(lowerbound_timestamp, symbol, the_date)

        for quote in symbol_quotes:
            quotes.append([
                convert_pandas_timestamp_to_formatted_string(quote.sip_timestamp),
                quote.bid_price,
                quote.bid_size,
                quote.bid_exchange,
                quote.ask_price,
                quote.ask_size,
                quote.ask_exchange
            ])

        if symbol_quotes[-1].sip_timestamp.hour < 16:
            lowerbound_timestamp = symbol_quotes[-1]._raw['sip_timestamp']
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


def main():
    for date_dir in sorted(os.listdir('%s' % raw_gaped_up_dir)):
        for stock_filename in sorted(os.listdir(f'{raw_gaped_up_dir}/{date_dir}')):
            symbol = stock_filename[0:-26]
            the_date = date_dir

            print(f'Downloading {symbol} @ {the_date}')

            quotes = _download_quotes(symbol, the_date)

            create_dir(f'{save_root_dir}/{the_date}')
            stock_tick_by_tick_csv = f'{save_root_dir}/{the_date}/{symbol}.csv'
            with open(stock_tick_by_tick_csv, 'w') as f:
                print(f'Writing into {stock_tick_by_tick_csv}')
                writer = csv.writer(f)
                for quote in quotes:
                    writer.writerow(quote)
            print(f'Finished Writing into {stock_tick_by_tick_csv}')


main()
