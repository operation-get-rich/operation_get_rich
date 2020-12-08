import asyncio
import datetime
import json
import math
import os
from datetime import time
from statistics import mean
from typing import Dict

import alpaca_trade_api as tradeapi
import pandas as pd
import torch
from alpaca_trade_api import StreamConn

from TraderGRU import TraderGRU
from config import PAPER_ALPACA_API_KEY, PAPER_ALPACA_SECRET_KEY, PAPER_ALPACA_BASE_URL
from stock_dataset import PercentChangeNormalizer as PCN

model = TraderGRU(
    input_size=7,
    hidden_size=5 * 7,
    hard=True
)
model.load_state_dict(
    torch.load(
        './TraderGRU_NextTrade/best_model.pt',
        map_location=torch.device('cpu')
    )
)
model.eval()

gap_stocks_by_date_dir = 'gaped_up_stocks_early_volume_1e5_gap_10_by_date'


def get_trade_from_model(inputs):
    inputs_tensor = torch.FloatTensor([inputs])
    normalized_inputs_tensor = PCN.normalize_volume(inputs_tensor)
    normalized_inputs_tensor = PCN.normalize_price_into_percent_change(normalized_inputs_tensor)
    trades = model(normalized_inputs_tensor)
    trade = trades.detach().numpy()[0][-1]
    return trade


def sell(symbol, shares_owned, trade, bid_price, capital):
    shares_to_sell = math.floor(abs(trade) * shares_owned)
    print(f'Selling {shares_to_sell} of {symbol} @ {format_usd(bid_price)}')
    print('bar_state_sell:', bar_state)
    if shares_to_sell > 0:
        api.submit_order(
            symbol=symbol,
            qty=shares_to_sell,
            side=SIDE_SELL,
            type=ORDER_TYPE_LIMIT,
            time_in_force=TIME_IN_FORCE_GTC,
            limit_price=bid_price
        )


def buy(symbol, trade, ask_price, capital):
    capital_to_use = capital * trade
    shares_to_buy = math.floor(capital_to_use / ask_price)
    print(f'Buying {shares_to_buy} of {symbol} @ {format_usd(ask_price)}')
    print('bar_state_buy:', bar_state)
    if shares_to_buy > 0:
        api.submit_order(
            symbol=symbol,
            qty=shares_to_buy,
            side='buy',
            type='limit',
            time_in_force='gtc',
            limit_price=ask_price
        )


def format_usd(capital):
    capital_formatted = '${:,.2f}'.format(capital)
    return capital_formatted


def compute_vwap(price_volume_products, volumes, ta_period=14):
    sum_pv = sum(price_volume_products[-ta_period:])
    sum_v = sum(volumes[-ta_period:])
    vwap = sum_pv / sum_v
    return vwap


def compute_ema(close_prices, ta_period=14):
    close_prices_series = pd.Series(close_prices)
    ewm = close_prices_series.ewm(
        span=ta_period,
        min_periods=ta_period,
        adjust=False
    ).mean()  # ewm = exponential moving window
    return ewm.values[-1]


conn = StreamConn(
    key_id=PAPER_ALPACA_API_KEY,
    secret_key=PAPER_ALPACA_SECRET_KEY,
    base_url=PAPER_ALPACA_BASE_URL
)

api = tradeapi.REST(
    key_id=PAPER_ALPACA_API_KEY,
    secret_key=PAPER_ALPACA_SECRET_KEY,
    base_url=PAPER_ALPACA_BASE_URL
)

stocks_bar_history = {}

bar_state = {}
TA_PERIOD = 14

RAW_FEATURES_KEY = 'raw_features'
MODEL_INPUTS_KEY = 'model_inputs'
VOLUMES_KEY = 'volumes'
PRICE_VOLUME_PRODUCTS_KEY = 'price_volume_products'
CLOSE_PRICES_KEY = 'close_prices'
TRADES_KEY = 'trades'
TRADES_BY_TIME_KEY = 'trades_by_time'
CAPITAL_KEY = 'capital'
SHARES_OWNED_KEY = 'shares'
TOTAL_ASSETS_PROGRESSION_KEY = 'total_assets_progression'

ASK_KEY = 'ask'
BID_KEY = 'bid'

SIDE_SELL = 'sell'
SIDE_BUY = 'buy'
ORDER_TYPE_LIMIT = 'limit'
TIME_IN_FORCE_GTC = 'gtc'


async def handle_bar(
        bar,  # type: Dict
        symbol,
        last_hour_minute,
        ask_price,
        bid_price,
):
    print('fancy_bar', bar)
    if symbol not in bar_state:
        bar_state[symbol] = {
            RAW_FEATURES_KEY: [],
            MODEL_INPUTS_KEY: [],
            VOLUMES_KEY: [],
            PRICE_VOLUME_PRODUCTS_KEY: [],
            CLOSE_PRICES_KEY: [],
            TRADES_KEY: [],
            CAPITAL_KEY: 10000,  # TODO: Figure out how to divide the capital
            SHARES_OWNED_KEY: 0,
            TOTAL_ASSETS_PROGRESSION_KEY: {},
            TRADES_BY_TIME_KEY: {},
        }
    bar_state[symbol][RAW_FEATURES_KEY].append(
        [
            bar['o'],
            bar['c'],
            bar['l'],
            bar['h'],
            bar['v']  # TODO: Implement Volume...
        ]
    )

    symbol_bar_state = bar_state[symbol]
    last_raw_feature = symbol_bar_state[RAW_FEATURES_KEY][-1]
    (
        last_open_price,
        last_close_price,
        last_low_price,
        last_high_price,
        last_volume
    ) = last_raw_feature

    last_typical_price = mean([last_close_price, last_low_price, last_high_price])

    volumes = symbol_bar_state[VOLUMES_KEY]
    volumes.append(last_volume)

    price_volume_products = symbol_bar_state[PRICE_VOLUME_PRODUCTS_KEY]
    price_volume_products.append(last_volume * last_typical_price)

    close_prices = symbol_bar_state[CLOSE_PRICES_KEY]
    close_prices.append(last_close_price)

    capital = symbol_bar_state[CAPITAL_KEY]
    shares_owned = symbol_bar_state[SHARES_OWNED_KEY]
    total_assets_progression = symbol_bar_state[TOTAL_ASSETS_PROGRESSION_KEY]

    able_to_compute_ta = len(symbol_bar_state[RAW_FEATURES_KEY]) >= TA_PERIOD

    if able_to_compute_ta:
        current_total_asset = bid_price * shares_owned + capital
        total_assets_progression[last_hour_minute] = current_total_asset
        print(f'\nCurrent Total Asset (liquid + shares_owned) {format_usd(current_total_asset)}')

        last_vwap = compute_vwap(price_volume_products, volumes, ta_period=TA_PERIOD)
        last_ema = compute_ema(close_prices, ta_period=TA_PERIOD)

        last_input = [last_open_price, last_close_price, last_low_price, last_high_price, last_volume, last_vwap,
                      last_ema]
        symbol_bar_state[MODEL_INPUTS_KEY].append(last_input)

        trade = get_trade_from_model(symbol_bar_state[MODEL_INPUTS_KEY])

        symbol_bar_state[TRADES_KEY].append(trade)

        if trade > 0:
            buy(symbol, trade, ask_price, capital)
        if trade < 0:
            sell(symbol, shares_owned, trade, bid_price, capital)


@conn.on(r'^trade_updates$')
async def on_account_updates(conn, channel, account):
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
    if account.event == 'fill':
        filled_quantity = int(account.order['filled_qty'])
        filled_avg_price = float(account.order['filled_avg_price'])
        if account.order['side'] == 'buy':
            bar_state[account.order['symbol']][SHARES_OWNED_KEY] += filled_quantity
            bar_state[account.order['symbol']][CAPITAL_KEY] -= filled_quantity * filled_avg_price
            print('bar_state_buy_filled', bar_state)
        if account.order['side'] == 'sell':
            bar_state[account.order['symbol']][SHARES_OWNED_KEY] -= filled_quantity
            bar_state[account.order['symbol']][CAPITAL_KEY] += filled_quantity * filled_avg_price
            print('bar_state_sell_filled', bar_state)
        return


@conn.on(r'^AM.*')
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
    print('regular_bar', bar)


def _get_previous_hour_minute(hour_minute):
    if hour_minute[1] - 1 > 0:
        return hour_minute[0], hour_minute[1] - 1
    return hour_minute[0] - 1, 59


initial_quote_symbol_state = {
    'exchange_prices': {
        'ask': {},
        'bid': {}
    },
    'ask_bars': {

    }
}
quote_state = {}


@conn.on(r'^Q.*')
async def on_quote(conn, channel, quote):
    """
    Quote({'askexchange': 2,
           'askprice': 119.63,
           'asksize': 2,
           'bidexchange': 17,
           'bidprice': 119.48,
           'bidsize': 1,
           'conditions': [0],
           'symbol': 'AAPL',
           'timestamp': 1605730275616000000})
    """
    # print('quote_state: ', quote_state)
    if quote.symbol not in quote_state:
        quote_state[quote.symbol] = initial_quote_symbol_state

    symbol_state = quote_state[quote.symbol]

    symbol_state['exchange_prices']['ask'][quote.askexchange] = {
        'price': quote.askprice,
        'size': quote.asksize,
    }
    symbol_state['exchange_prices']['bid'][quote.askexchange] = {
        'price': quote.bidprice,
        'size': quote.bidsize,
    }

    ask_price = 2 ** 32 - 1
    chosen_ask_size = None
    for exchange_id, exchange_node in symbol_state['exchange_prices']['ask'].items():
        current_ask_price, current_ask_size = exchange_node['price'], exchange_node['size']
        if current_ask_price < ask_price:
            ask_price = current_ask_price
            chosen_ask_exchange_id = exchange_id

            if 'chosen_exchanges' in symbol_state:
                if symbol_state['chosen_exchanges']['size'] != curre

            chosen_ask_size = current_ask_size

    bid_price = 0
    chosen_bid_size = None
    for exchange_id, exchange_node in symbol_state['exchange_prices']['bid'].items():
        current_bid_price, current_bid_size = exchange_node['price'], exchange_node['size']
        if current_bid_price > bid_price:
            bid_price = current_bid_price
            chosen_bid_exchange_id = exchange_id
            chosen_bid_size = current_bid_size
    symbol_state['chosen_exchanges'] = {
        'ask': {
            'exchange_id': chosen_ask_exchange_id,
            'price': ask_price,
            'size': chosen_ask_size,
        },
        'bid': {
            'exchange_id': chosen_bid_exchange_id,
            'price': bid_price,
            'size': chosen_bid_size
        }
    }

    # ask_price = min(symbol_state['exchange_prices']['ask'].values()['price'])
    # bid_price = max(symbol_state['exchange_prices']['bid'].values()['price'])

    # ask
    bars = symbol_state['ask_bars']

    hour_minute = (quote.timestamp.hour, quote.timestamp.minute)
    if hour_minute not in bars:
        previous_hour_minute = _get_previous_hour_minute(hour_minute)
        if previous_hour_minute in bars:
            bars[previous_hour_minute]['c'] = ask_price
            bars[previous_hour_minute]['v'] += chosen_ask_size

            await handle_bar(
                bars[previous_hour_minute],
                quote.symbol,
                previous_hour_minute,
                ask_price,
                bid_price)

        bars[hour_minute] = {
            'o': ask_price,
        }

    if 'l' not in bars[hour_minute]:
        bars[hour_minute]['l'] = ask_price
    else:
        bars[hour_minute]['l'] = min(ask_price, bars[hour_minute]['l'])

    if 'h' not in bars[hour_minute]:
        bars[hour_minute]['h'] = ask_price
    else:
        bars[hour_minute]['h'] = max(ask_price, bars[hour_minute]['h'])

    if 'v' not in bars[hour_minute]:
        bars[hour_minute]['v'] = chosen_ask_size + chosen_bid_size
    else:
        bars[hour_minute]['v'] += chosen_ask_size + chosen_bid_size

    # print(f'ask_exchange {quote.askexchange}', quote.timestamp.strftime(DATETIME_FORMAT))
    # await handle_quote(quote)


conn.run([
    'Q.AAPL', 'AM.AAPL',
    # 'Q.LGVW', 'AM.LGVW',
    # 'Q.AMRN', 'AM.AMRN'
    'trade_updates'])