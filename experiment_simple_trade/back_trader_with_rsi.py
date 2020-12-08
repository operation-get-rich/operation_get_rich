import datetime
import json
import os

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volume import VolumeWeightedAveragePrice

gapped_up_data_root_dir_ticks = '../datas/polygon_gapped_up_tick_by_tick'
gapped_up_data_root_dir_minutes = '../datas/polygon_early_day_gap_segmenter_parallel'

STATE_FILE_LOCATION = 'state_polygon_01_03_high_rsi.json'

from utils import format_usd, DATETIME_FORMAT


def main():
    profit_goal = .01
    loss_tolerance = .03
    _back_trade(loss_tolerance, profit_goal)


def _back_trade(loss_tolerance, profit_goal):
    total_delta = 0
    trade_counter = 0
    average_delta = 0
    compounding_delta = 1
    for the_date in sorted(os.listdir(gapped_up_data_root_dir_ticks)):
        if '.' in the_date:
            continue
        for stock_file in sorted(os.listdir(f'{gapped_up_data_root_dir_ticks}/{the_date}')):
            stock_symbol = stock_file.split('.')[0]
            print(f'\nTrading {stock_symbol} on {the_date}')
            try:
                delta = day_trade(
                    stock_symbol,
                    the_date,
                    profit_goal,
                    loss_tolerance
                )

                if delta:
                    trade_counter += 1
                    total_delta += delta
                    compounding_delta *= (1 + delta)
                if trade_counter:
                    average_delta = total_delta / trade_counter

                # TODO: Delta stats should be continous, not reset on each run

                update_state(
                    update_delta_stats_state,
                    average_delta=average_delta,
                    compounding_delta=compounding_delta,
                    total_delta=total_delta
                )
            except Exception as exc:
                print(f'Exception: {exc}')


def day_trade(stock_symbol, the_date, profit_goal, loss_tolerance):
    delta = None
    stock_df_ticks = pd.read_csv(
        f'{gapped_up_data_root_dir_ticks}/{the_date}/{stock_symbol}.csv',
        parse_dates=['sip_timestamp'],
    )

    stock_df_minutes = pd.read_csv(
        f'{gapped_up_data_root_dir_minutes}/{the_date}/{stock_symbol}_{the_date}.csv',
        parse_dates=['time'],
        index_col=0
    )
    stock_df_minutes['vwap'] = VolumeWeightedAveragePrice(
        high=stock_df_minutes.high,
        low=stock_df_minutes.low,
        close=stock_df_minutes.close,
        volume=stock_df_minutes.volume
    ).vwap

    stock_df_minutes['ema'] = EMAIndicator(
        close=stock_df_minutes.close,
    ).ema_indicator()

    stock_df_minutes['rsi'] = RSIIndicator(
        close=stock_df_minutes.close,
    ).rsi()

    for i in range(len(stock_df_minutes)):
        (
            ticker,
            time,
            open,
            close,
            low,
            high,
            volume,
            vwap,
            ema,
            rsi
        ) = stock_df_minutes.iloc[i]

        if time.hour < 9:
            continue

        if time.hour == 9 and time.minute < 30:
            continue

        if time.hour > 9:
            break

        if i + 2 < len(stock_df_minutes):
            (
                next_ticker,
                next_time,
                next_open,
                next_close,
                next_low,
                next_high,
                next_volume,
                next_vwap,
                next_ema,
                next_rsi
            ) = stock_df_minutes.iloc[i + 1]

            (
                next_ticker_2,
                next_time_2,
                next_open_2,
                next_close_2,
                next_low_2,
                next_high_2,
                next_volume_2,
                next_vwap_2,
                next_ema_2,
                next_rsi_2,
            ) = stock_df_minutes.iloc[i + 2]

            if close < open and next_close > next_open and next_high_2 > next_high:
                if rsi < 60:
                    continue
                buy_price = None
                buy_time = None
                for j in range(len(stock_df_ticks)):
                    # TODO: Implement the `quote_state` approach to get better price
                    (
                        sip_timestamp,
                        bid_price,
                        bid_size,
                        bid_exchange,
                        ask_price,
                        ask_size,
                        ask_exchange,
                    ) = stock_df_ticks.iloc[j]

                    if sip_timestamp.hour < next_time_2.hour:
                        continue

                    if sip_timestamp.hour == next_time_2.hour and sip_timestamp.minute < next_time_2.minute:
                        continue

                    if not buy_price and ask_price > next_high:
                        buy_time = sip_timestamp
                        buy_price = ask_price
                        print(f'Buying {stock_symbol} @ {buy_time} {format_usd(buy_price)}')

                    if buy_price:
                        if (
                                bid_price >= buy_price * (1 + profit_goal) or
                                bid_price <= buy_price * (1 - loss_tolerance)
                        ):
                            sell_time = sip_timestamp
                            sell_price = bid_price
                            delta = (sell_price / buy_price) - 1

                            print(f'Selling {stock_symbol} @ {sell_time} {format_usd(sell_price)}')
                            print(f'Delta: {delta * 100}%')

                            update_state(
                                update_performance_state,
                                delta=delta,
                                buy_time=buy_time,
                                buy_price=buy_price,
                                sell_price=sell_price,
                                sell_time=sell_time,
                                stock_symbol=stock_symbol,
                                the_date=the_date
                            )
                            return delta
                return None


def restart_state_meta():
    try:
        with open(STATE_FILE_LOCATION, 'r') as state_file:
            state = json.load(state_file)
    except FileNotFoundError:
        state = {}

    state['_meta'] = {}

    with open(STATE_FILE_LOCATION, 'w') as state_file:
        json.dump(state, state_file)


def update_state(update_func=None, *args, **kwargs):
    try:
        with open(STATE_FILE_LOCATION, 'r') as state_file:
            state = json.load(state_file)
    except FileNotFoundError:
        state = {}

    if update_func:
        update_func(state, *args, **kwargs)

    if '_meta' in state:
        if 'first_state_update_call' not in state['_meta']:
            state['_meta']['first_state_update_call'] = datetime.datetime.now().strftime(DATETIME_FORMAT)
        state['_meta']['latest_state_update_call'] = datetime.datetime.now().strftime(DATETIME_FORMAT)

    with open(STATE_FILE_LOCATION, 'w') as state_file:
        json.dump(state, state_file)


def update_delta_stats_state(state, average_delta, compounding_delta, total_delta):
    state['delta_stats'] = {
        'total_delta': f'{total_delta * 100}%',
        'average_delta': f'{average_delta * 100}%',
        'compounding_delta': f'{(compounding_delta - 1) * 100}%'
    }


def update_performance_state(state, delta, buy_time, buy_price, sell_price, sell_time,
                             stock_symbol, the_date):
    if 'performance' not in state:
        state['performance'] = {}
    node = {
        'delta': f'{delta * 100}%',
        'buy_price': buy_price,
        'sell_price': sell_price,
        'buy_time': buy_time.strftime(DATETIME_FORMAT),
        'sell_time': sell_time.strftime(DATETIME_FORMAT),
    }
    if the_date in state['performance']:
        state['performance'][the_date][stock_symbol] = node
    else:
        state['performance'][the_date] = {
            stock_symbol: node
        }


restart_state_meta()
main()
