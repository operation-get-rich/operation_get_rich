import ast
import math
from statistics import mean

import pandas as pd
import torch
from dateutil.parser import parse

from experiment_trader_gru.TraderGRU import TraderGRU
from experiment_vanilla_gru.dataset_vanilla_gru import PercentChangeNormalizer as PCN
from decorators import timeit

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

gap_stocks_by_date_dir = 'fron_gapped_up_stocks'


def compute_vwap(i, price_volume_products, volumes, ta_period=14):
    start_pos = i - (ta_period - 1)
    end_pos = i + 1
    sum_pv = sum(price_volume_products[start_pos:end_pos])
    sum_v = sum(volumes[start_pos:end_pos])
    vwap = sum_pv / sum_v
    return vwap


def compute_ema(close_prices, ta_period):
    close_prices_series = pd.Series(close_prices)
    ewm = close_prices_series.ewm(
        span=ta_period,
        min_periods=ta_period,
        adjust=False
    ).mean()  # ewm = exponential moving window
    return ewm.values[-1]


@timeit
def main():
    capital = 25000
    stock_file = '../datas/alpaca_cross_check/TKAT_2020-12-17.csv'
    trading_state = day_trade(stock_file, capital)

    trading_state_from_logs = {}
    with open('alpaca_paper_trade_logs/alpaca_paper_trade2020-12-17.log') as f:
        for line in f:
            line_splitted = line.split()

            if not line_splitted:
                break

            date_str = line_splitted[0]
            hour_str = line_splitted[1]
            log_level = line_splitted[2]

            parsed_datetime = parse(date_str + ' ' + hour_str)

            parsed_datetime = parsed_datetime.replace(
                hour=parsed_datetime.hour + 1,  # To make EST timezone
                second=0,
                microsecond=0
            )

            payload_start_index = len(date_str) + len(hour_str) + len(log_level) + 3
            try:
                log_payload = ast.literal_eval(line[payload_start_index:])
            except SyntaxError:
                print(f'Payload is not a dict {line}')

            payload = log_payload['payload']
            if log_payload['type'] in {'no_trade_sell', 'no_trade_buy'}:
                trading_state_from_logs[parsed_datetime] = {
                    'trade': payload['trade'],
                    'capital': payload['capital'],
                    'shares_owned': payload['shares_owned']
                }
            elif log_payload['type'] == 'trade_buy':
                trading_state_from_logs[parsed_datetime] = {
                    'trade': payload['trade'],
                    'capital': payload['capital'],
                    'shares_owned': payload['shares_owned'],
                    'shares_to_buy': payload['shares_to_buy'],
                    'price': payload['price'],
                    'price_with_slippage': payload['price_with_slippage']
                }
            elif log_payload['type'] == 'trade_sell':
                trading_state_from_logs[parsed_datetime] = {
                    'trade': payload['trade'],
                    'capital': payload['capital'],
                    'shares_owned': payload['shares_owned'],
                    'shares_to_sell': payload['shares_to_sell'],
                    'price': payload['price'],
                    'price_with_slippage': payload['price_with_slippage']
                }

    return capital


@timeit
def day_trade(stock_file, capital):
    ticker_name = stock_file[-19:-15]
    print(f'\nTrading ticker: {ticker_name}')
    stock_df = pd.read_csv(f'{stock_file}')
    stock_df = stock_df.drop(labels=['ticker'], axis=1)

    ta_period = 14
    volumes = []
    price_volume_products = []
    close_prices = []
    inputs = []

    trade = None
    shares = 0

    trading_state = {}
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
        parsed_time = parse(time)
        typical_price = mean([close_price, low_price, high_price])

        volumes.append(volume)
        price_volume_products.append(volume * typical_price)
        close_prices.append(close_price)

        able_to_compute_ta = i >= (ta_period - 1)
        if able_to_compute_ta:
            current_total_asset = open_price * shares + capital
            print(f'\nCurrent Total Asset (liquid + shares) {format_usd(current_total_asset)}')
            if trade and parsed_time >= parsed_time.replace(hour=10, minute=13):
                trades[time] = float(trade)
                if trade > 0:
                    capital, shares = buy(trade, parsed_time, open_price, capital, shares, ticker_name, trading_state)
                if trade < 0:
                    capital, shares = sell(trade, parsed_time, open_price, capital, shares, ticker_name, trading_state)

            vwap = compute_vwap(i, price_volume_products, volumes, ta_period)
            ema = compute_ema(close_prices, ta_period)

            input = [open_price, close_price, low_price, high_price, volume, vwap, ema]
            inputs.append(input)

            trade = get_next_trade_from_model(inputs, trade)
    if shares > 0:
        capital += shares * close_price
    return trading_state


def get_next_trade_from_model(inputs, trade):
    inputs_tensor = torch.FloatTensor([inputs])
    normalized_inputs_tensor = PCN.normalize_volume(inputs_tensor)
    normalized_inputs_tensor = PCN.normalize_price_into_percent_change(normalized_inputs_tensor)
    trades = model(normalized_inputs_tensor)
    trade = trades.detach().numpy()[0][-1]
    return trade


def sell(trade, time, open_price, capital, shares, ticker_name, trading_state):
    shares_to_sell = math.floor(abs(trade) * shares)
    capital += shares_to_sell * open_price
    shares -= shares_to_sell
    print(f'\nTime: {time}')
    print(f'Trade: {trade}')
    print(f'Capital: {format_usd(capital)}')
    print(f'Shares: {shares}')
    print(f'Selling {shares_to_sell} shares of {ticker_name} at {format_usd(open_price)}')
    print(f'Delta: {format_usd(shares_to_sell * open_price)}')
    trading_state[time] = {
        "trade": trade,
        "capital": capital,
        "shares_owned": shares,
    }
    if shares_to_sell:
        trading_state[time]['shares_to_sell'] = shares_to_sell
        trading_state[time]['price'] = open_price
    return capital, shares


def buy(trade, time, open_price, capital, shares, ticker_name, trading_state):
    capital_to_use = capital * trade
    shares_to_buy = math.floor(capital_to_use / open_price)
    capital_to_use = shares_to_buy * open_price
    capital -= capital_to_use
    shares += shares_to_buy
    print(f'\nTime: {time}')
    print(f'Trade: {trade}')
    print(f'Capital: {format_usd(capital)}')
    print(f'Shares: {shares}')
    print(f'Buying {shares_to_buy} shares of {ticker_name} at {format_usd(open_price)}')
    print(f'Buying using {format_usd(capital_to_use)} worth of capital')
    trading_state[time] = {
        "trade": trade,
        "capital": capital,
        "shares_owned": shares,
    }
    if shares_to_buy:
        trading_state[time]['shares_to_buy'] = shares_to_buy
        trading_state[time]['price'] = open_price

    return capital, shares


def format_usd(capital):
    capital_formatted = '${:,.2f}'.format(capital)
    return capital_formatted


capital = main()
print(capital)
