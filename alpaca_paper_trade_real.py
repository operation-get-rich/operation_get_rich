import asyncio
import json
import math
import os
from statistics import mean

import alpaca_trade_api as tradeapi
import matplotlib.pyplot as plt
import pandas as pd
import torch
import websockets
from alpaca_trade_api import StreamConn

from TraderGRU import TraderGRU
from config import PAPER_ALPACA_API_KEY, PAPER_ALPACA_SECRET_KEY, PAPER_ALPACA_BASE_URL
from stock_dataset import PercentChangeNormalizer as PCN
from utils import timeit

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


def compute_ema(close_prices, ta_period=14):
    close_prices_series = pd.Series(close_prices)
    ewm = close_prices_series.ewm(
        span=ta_period,
        min_periods=ta_period,
        adjust=False
    ).mean()  # ewm = exponential moving window
    return ewm.values[-1]


def append_total_assets_db(trade_date, ticker, total_assets_progression):
    json_filename = 'total_assets_progression_by_time.json'
    with open(json_filename, 'r') as json_file:
        current_total_assets_db = json.load(json_file)

    if trade_date in current_total_assets_db:
        current_total_assets_db[trade_date][ticker] = total_assets_progression
    else:
        current_total_assets_db[trade_date] = {
            ticker: total_assets_progression
        }

    with open(json_filename, 'w') as json_file:
        json.dump(current_total_assets_db, json_file)


def append_trades_db(trade_date, ticker, trades):
    json_filename = 'trades_by_time.json'
    with open(json_filename, 'r') as json_file:
        db_json = json.load(json_file)

    if trade_date in db_json:
        db_json[trade_date][ticker] = trades
    else:
        db_json[trade_date] = {
            ticker: trades
        }

    with open(json_filename, 'w') as json_file:
        json.dump(db_json, json_file)


@timeit
def _main():
    capital = 10000
    capital_progression = []
    dates = []
    for date in sorted(os.listdir(gap_stocks_by_date_dir)):
        dates.append(date)
        print(f'\nTrading date: {date}')
        start_of_day_capital = capital
        print(f'Capital Start of day : {format_usd(start_of_day_capital)}')
        date_dir = f'{gap_stocks_by_date_dir}/{date}'

        stock_files = sorted(os.listdir(date_dir))
        capital_per_company = float(capital) / len(stock_files)
        capital = 0
        for stock_file in stock_files:
            (
                trade_date,
                ticker,
                total_assets_progression,
                trades,
                stock_capital
            ) = day_trade(stock_file, date_dir, capital_per_company)
            capital += stock_capital
            # append_total_assets_db(trade_date, ticker, total_assets_progression)
            # append_trades_db(trade_date, ticker, trades)
        capital_progression.append(capital)
    plt.plot(dates, capital_progression)
    plt.xticks(rotation=90)
    plt.show()
    return capital


@timeit
def day_trade(stock_file, date_dir, capital):
    ticker_name = stock_file[:-26]
    trade_date = date_dir[-10:]

    print(f'\nTrading ticker: {ticker_name}')
    stock_df = pd.read_csv(f'{date_dir}/{stock_file}')
    stock_df = stock_df.drop(labels=['ticker'], axis=1)

    ta_period = 14
    volumes = []
    price_volume_products = []
    close_prices = []
    inputs = []

    trade = None
    shares = 0

    total_assets_progression = {}
    trades = {}
    for i in range(len(stock_df)):
        (
            time,
            open_price,
            close_price,
            low_price,
            high_price,
            volume
        ) = stock_df.loc[i]
        typical_price = mean([close_price, low_price, high_price])

        volumes.append(volume)
        price_volume_products.append(volume * typical_price)
        close_prices.append(close_price)

        able_to_compute_ta = i >= (ta_period - 1)
        if able_to_compute_ta:
            current_total_asset = open_price * shares + capital
            total_assets_progression[time] = current_total_asset
            print(f'\nCurrent Total Asset (liquid + shares) {format_usd(current_total_asset)}')
            if trade:
                trades[time] = float(trade)
                if trade > 0:
                    capital, shares = buy(trade, open_price, capital, shares, ticker_name)
                if trade < 0:
                    capital, shares = sell(trade, open_price, capital, shares, ticker_name)

            vwap = compute_vwap(i, price_volume_products, volumes, ta_period)
            ema = compute_ema(close_prices, ta_period)

            input = [open_price, close_price, low_price, high_price, volume, vwap, ema]
            inputs.append(input)

            trade = get_trade_from_model(inputs, trade)
    if shares > 0:
        capital += shares * close_price
        total_assets_progression[time] = capital
    return (trade_date, ticker_name, total_assets_progression, trades, capital)


def get_trade_from_model(inputs):
    inputs_tensor = torch.FloatTensor([inputs])
    normalized_inputs_tensor = PCN.normalize_volume(inputs_tensor)
    normalized_inputs_tensor = PCN.normalize_price_into_percent_change(normalized_inputs_tensor)
    trades = model(normalized_inputs_tensor)
    trade = trades.detach().numpy()[0][-1]
    return trade


def sell_legacy(trade, open_price, capital, shares, ticker_name):
    shares_to_sell = math.floor(abs(trade) * shares)
    capital += shares_to_sell * open_price
    shares -= shares_to_sell
    print(f'\nSelling {shares_to_sell} shares of {ticker_name} at {format_usd(open_price)}')
    print(f'Delta: {format_usd(shares_to_sell * open_price)}')
    print(f'Capital: {format_usd(capital)}')
    print(f'Shares: {shares}')
    return capital, shares


def buy_legacy(trade, open_price, capital, shares, ticker_name):
    capital_to_use = capital * trade
    shares_to_buy = math.floor(capital_to_use / open_price)
    capital_to_use = shares_to_buy * open_price
    capital -= capital_to_use
    shares += shares_to_buy
    print(f'\nBuying {shares_to_buy} shares of {ticker_name} at {format_usd(open_price)}')
    print(f'Buying using {format_usd(capital_to_use)} worth of capital')
    print(f'Capital: {format_usd(capital)}')
    print(f'Shares: {shares}')
    return capital, shares


def sell_all(symbol):
    # TODO: Implement sell all shares at market price
    pass


# TODO: Update capital and shares owned in the call back of trade updates
def sell(symbol, shares_owned, trade, capital):
    shares_to_sell = math.floor(abs(trade) * shares_owned)
    api.submit_order(
        symbol=symbol,
        qty=shares_to_sell,

        side=SIDE_SELL,
        type=ORDER_TYPE_LIMIT,
        time_in_force=TIME_IN_FORCE_GTC,
        limit_price=get_best_bid_price(symbol)
    )


# TODO: Update capital and shares owned in the call back of trade updates
def buy(symbol, trade, capital):
    capital_to_use = capital * trade
    shares_to_buy = math.floor(capital_to_use / get_best_ask_price(symbol))
    api.submit_order(
        symbol=symbol,
        qty=shares_to_buy,
        side='buy',
        type='limit',
        time_in_force='gtc',
        limit_price=get_best_ask_price(symbol)
    )


def format_usd(capital):
    capital_formatted = '${:,.2f}'.format(capital)
    return capital_formatted


def compute_vwap(price_volume_products, volumes, ta_period=14):
    sum_pv = sum(price_volume_products[-ta_period:])
    sum_v = sum(volumes[-ta_period:])
    vwap = sum_pv / sum_v
    return vwap


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
quote_state = {}
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


def get_best_bid_price(symbol):
    return max(quote_state[symbol][BID_KEY].values())


def get_best_ask_price(symbol):
    return min(quote_state[symbol][ASK_KEY].values())


async def handle_bar(bar):
    symbol = bar.symbol
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
            TOTAL_ASSETS_PROGRESSION_KEY: [],
            TRADES_BY_TIME_KEY: {},
        }
    else:
        bar_state[symbol][RAW_FEATURES_KEY].append(
            bar.start,
            bar.open,
            bar.close,
            bar.low,
            bar.high,
            bar.totalvolume
        )

    symbol_bar_state = bar_state[symbol]
    last_raw_feature = symbol_bar_state[RAW_FEATURES_KEY][-1]
    (
        last_time,
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

    able_to_compute_ta = len(bar_state[RAW_FEATURES_KEY]) >= TA_PERIOD

    if able_to_compute_ta:
        current_total_asset = get_best_bid_price(symbol) * shares_owned + capital
        total_assets_progression[last_time] = current_total_asset
        print(f'\nCurrent Total Asset (liquid + shares_owned) {format_usd(current_total_asset)}')

        last_vwap = compute_vwap(price_volume_products, volumes)
        last_ema = compute_ema(close_prices)

        last_input = [last_open_price, last_close_price, last_low_price, last_high_price, last_volume, last_vwap,
                      last_ema]
        symbol_bar_state[MODEL_INPUTS_KEY].append(last_input)

        trade = get_trade_from_model(symbol_bar_state[MODEL_INPUTS_KEY])

        if trade > 0:
            buy(symbol, trade, capital)
        if trade < 0:
            sell(symbol, shares_owned, trade, capital)


async def handle_quote(quote):
    symbol = quote.symbol
    if symbol not in quote_state:
        quote_state[symbol] = {
            ASK_KEY: {},
            BID_KEY: {}
        }

    quote_state[symbol][ASK_KEY][quote.askexchange] = quote.askprice
    quote_state[symbol][BID_KEY][quote.bidexchange] = quote.bidprice


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
    print(bar.symbol)


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
    print(quote)


conn.run(['Q.AAPL'])
