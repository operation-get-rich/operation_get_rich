import logging
import time
from datetime import timedelta

import alpaca_trade_api as tradeapi
from alpaca_trade_api import StreamConn

from experiment_trader_gru.directories import RUNS_DIR, ALPACA_PAPER_TRADE_LOGS_DIR
from experiment_trader_gru.TraderGRU import load_trader_gru_model
from config import PAPER_ALPACA_API_KEY, PAPER_ALPACA_SECRET_KEY, PAPER_ALPACA_BASE_URL
from experiment_trader_gru.stock_trader import GappedUpStockFinder, StockTraderManager
from utils import get_current_datetime, get_today_market_open, get_alpaca_time_str_format, get_previous_market_open

BEST_MODEL_LOCATION = f'{RUNS_DIR}/Trader_Load_NextTrade_PreMarket_Multiply-20201121-222444/best_model.pt'

api = tradeapi.REST(
    key_id=PAPER_ALPACA_API_KEY,
    secret_key=PAPER_ALPACA_SECRET_KEY,
    base_url=PAPER_ALPACA_BASE_URL
)


def set_up_logging():
    log_filename = f'{ALPACA_PAPER_TRADE_LOGS_DIR}/alpaca_paper_trade{get_current_datetime().date()}.log'
    with open(log_filename, 'w'):
        pass
    logging.basicConfig(
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ],
        format='%(asctime)s %(levelname)s %(message)s',
        level=logging.INFO,
    )


def warm_up_stock_traders(stock_traders_by_symbol):
    market_open_hour = get_today_market_open()
    while get_current_datetime() < market_open_hour:
        now_time = get_current_datetime()

        end_time = now_time
        start_time = end_time - timedelta(minutes=1)

        end_time_str = get_alpaca_time_str_format(end_time)
        start_time_str = get_alpaca_time_str_format(start_time)

        barset = api.get_barset(
            symbols=','.join([s for s in stock_traders_by_symbol.keys()]),
            timeframe='1Min',
            end=end_time_str,
            start=start_time_str,
        )  # TODO: Use Polygon barset api when using Polygon trained model

        StockTraderManager.update_multiple_from_barset(
            stock_traders_by_symbol=stock_traders_by_symbol,
            barset=barset
        )

        time.sleep(
            min((market_open_hour - end_time).seconds, 60)
        )


def instantiate_stock_traders(model, symbols):
    end_time = get_current_datetime()
    start_time = get_previous_market_open(anchor_time=end_time)
    end_time_str = get_alpaca_time_str_format(end_time)
    start_time_str = get_alpaca_time_str_format(start_time)
    barset = api.get_barset(
        symbols=','.join([s for s in symbols]),
        timeframe='1Min',
        end=end_time_str,
        start=start_time_str,
    )  # TODO: Use Polygon barset api when using Polygon trained model

    stock_traders_by_symbol = StockTraderManager.create_multiple_from_barset(
        model=model,
        barset=barset,
        capital=100_000  # TODO: Figure out how to get capital from our account
    )
    return stock_traders_by_symbol


def main():
    set_up_logging()

    # symbol_volume_gaps = GappedUpStockFinder.find_gapped_up_stock()  # type: SymbolVolumeGap
    # gapped_up_symbols = [s[0] for s in symbol_volume_gaps]

    gapped_up_symbols = ['TSLA']

    model = load_trader_gru_model(model_location=BEST_MODEL_LOCATION)

    stock_traders_by_symbol = instantiate_stock_traders(model, gapped_up_symbols)
    warm_up_stock_traders(stock_traders_by_symbol)

    stream = StreamConn(
        key_id=PAPER_ALPACA_API_KEY,
        secret_key=PAPER_ALPACA_SECRET_KEY,
        base_url=PAPER_ALPACA_BASE_URL
    )

    @stream.on(r'^AM')
    async def on_minute_bars(conn, channel, bar):
        """
        {
            'symbol': 'MSFT',
            'volume': 3841,
            'totalvolume': 210280,
            'vwap': 216.905,
            'open': 216.89,
            'close': 216.83,
            'high': 216.94,
            'low': 216.83,
            'average': 216.576,
            'start': 1605628500000,
            'end': 1605628560000,
            'timestamp': 1605628500000
        }
        """
        logging.info(
            dict(
                type='minute_bars_call',
                bar=bar
            )
        )
        if bar.symbol in stock_traders_by_symbol:
            await stock_traders_by_symbol[bar.symbol].handle_bar(bar)
        else:
            logging.warning(
                dict(
                    msg=f"Can't find symbol of {bar.symbol} in stock_traders_by_symbol",
                    type='cant_find_symbol_stock_traders_minute_bars',
                    payload=dict(symbol=bar.symbol)
                )
            )

    @stream.on('trade_updates')
    async def on_trade_updates(conn, channel, account):
        """
            {
                'event': 'fill',
                'order': {
                    'asset_class': 'us_equity',
                    'asset_id': 'b0b6dd9d-8b9b-48a9-ba46-b9d54906e415',
                    'canceled_at': None,
                    'client_order_id': '9548909c-bb9a-402d-b916-95a13e917ce6',
                    'created_at': '2020-11-19T19:57:02.542513Z',
                    'expired_at': None,
                    'extended_hours': False,
                    'failed_at': None,
                    'filled_at': '2020-11-19T19:57:02.695986Z',
                    'filled_avg_price': '118.42',
                    'filled_qty': '84',
                    'hwm': None,
                    'id': '88c7f734-f2e2-4d46-96e8-5cfb9a2fce86',
                    'legs': None,
                    'limit_price': '118.42',
                    'order_class': '',
                    'order_type': 'limit',
                    'qty': '84',
                    'replaced_at': None,
                    'replaced_by': None,
                    'replaces': None,
                    'side': 'buy',
                    'status': 'filled',
                    'stop_price': None,
                    'submitted_at': '2020-11-19T19:57:02.537808Z',
                    'symbol': 'AAPL',
                    'time_in_force': 'gtc',
                    'trail_percent': None,
                    'trail_price': None,
                    'type': 'limit',
                    'updated_at': '2020-11-19T19:57:02.706647Z'
                },
                'position_qty': '84',
                'price': '118.42',
                'qty': '84',
                'timestamp': '2020-11-19T19:57:02.695986612Z'
            }
            """
        if account.order['symbol'] in stock_traders_by_symbol:
            stock_traders_by_symbol[account.order['symbol']].handle_trade_updates(account)
        else:
            logging.warning(
                dict(
                    msg=f"Can't find symbol of {symbol} in stock_traders_by_symbol",
                    type='cant_find_symbol_stock_traders_trade_updates',
                    payload=dict(symbol=symbol)
                )
            )

    channels = []
    for symbol in stock_traders_by_symbol.keys():
        assert symbol in stock_traders_by_symbol
        channels.append(f'AM.{symbol}')

    # channels.append('trade_updates')

    stream.run(channels)


main()
