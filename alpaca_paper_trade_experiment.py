import json
import math
import os
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd
import torch

from TraderGRU import TraderGRU
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
def main():
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

            trade = get_next_trade_from_model(inputs, trade)
    if shares > 0:
        capital += shares * close_price
        total_assets_progression[time] = capital
    return (trade_date, ticker_name, total_assets_progression, trades, capital)


def get_next_trade_from_model(inputs, trade):
    inputs_tensor = torch.FloatTensor([inputs])
    normalized_inputs_tensor = PCN.normalize_volume(inputs_tensor)
    normalized_inputs_tensor = PCN.normalize_price_into_percent_change(normalized_inputs_tensor)

    trades = model(normalized_inputs_tensor)
    trade = trades.detach().numpy()[0][-1]
    return trade


def sell(trade, open_price, capital, shares, ticker_name):
    shares_to_sell = math.floor(abs(trade) * shares)
    capital += shares_to_sell * open_price
    shares -= shares_to_sell
    print(f'\nSelling {shares_to_sell} shares of {ticker_name} at {format_usd(open_price)}')
    print(f'Delta: {format_usd(shares_to_sell * open_price)}')
    print(f'Capital: {format_usd(capital)}')
    print(f'Shares: {shares}')
    return capital, shares


def buy(trade, open_price, capital, shares, ticker_name):
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


def format_usd(capital):
    capital_formatted = '${:,.2f}'.format(capital)
    return capital_formatted


capital = main()
print(capital)
