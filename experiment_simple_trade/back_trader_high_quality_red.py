import datetime
import json
import os

import pandas as pd

gapped_up_data_root_dir = '../datas/polygon_gapped_up_tick_by_tick'

STATE_FILE_LOCATION = 'state_01_03_high_high_quality_green.json'

from utils import format_usd, DATETIME_FORMAT


def main():
    profit_goal = .01
    loss_tolerance = .03
    max_high_close_delta = 0
    _back_trade(loss_tolerance, profit_goal, max_high_close_delta)


def _back_trade(loss_tolerance, profit_goal, max_high_close_delta):
    total_delta = 0
    trade_counter = 0
    compounding_delta = 1
    for the_date in sorted(os.listdir(gapped_up_data_root_dir)):
        if '.' in the_date:
            continue
        for stock_file in sorted(os.listdir(f'{gapped_up_data_root_dir}/{the_date}')):
            stock_symbol = stock_file.split('.')[0]
            print(f'\nTrading {stock_symbol} on {the_date}')

            if stock_symbol == 'BAC' and the_date == '2019-01-22':
                # hack, its not trading this stock anyway
                continue

            if stock_symbol == 'EBAY' and the_date == '2019-01-22':
                # hack, its not trading this stock anyway
                continue

            try:
                delta = day_trade(
                    stock_symbol,
                    the_date,
                    profit_goal,
                    loss_tolerance,
                    max_high_close_delta
                )

                if delta:
                    trade_counter += 1
                    total_delta += delta
                    compounding_delta *= (1 + delta)
                average_delta = total_delta / trade_counter

                # TODO: Delta stats should be continous, not reset on each run

                update_state(
                    update_delta_stats_state,
                    average_delta=average_delta,
                    compounding_delta=compounding_delta,
                    total_delta=total_delta
                )
            except Exception:
                print('Exception')
                continue


def day_trade(stock_symbol, the_date, profit_goal, loss_tolerance, max_high_close_delta):
    delta = None
    stock_df = pd.read_csv(
        f'{gapped_up_data_root_dir}/{the_date}/{stock_symbol}.csv',
        parse_dates=['sip_timestamp'],
    )
    bars = {}
    bars_list = []
    is_green_red_pair_found = False
    last_green_bar = None
    buy_price = None
    for i in range(len(stock_df)):
        (
            sip_timestamp,
            bid_price,
            bid_size,
            bid_exchange,
            ask_price,
            ask_size,
            ask_exchange
        ) = stock_df.iloc[i]

        bid_price = float(bid_price)
        bid_size = float(bid_size)
        ask_price = float(ask_price)
        ask_size = float(ask_size)

        if sip_timestamp.hour >= 9:
            if sip_timestamp.hour == 9 and sip_timestamp.minute < 30:
                continue
        else:
            continue

        if not buy_price and is_green_red_pair_found:
            if ask_price > last_green_bar['h']:
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

        hour_minute = (sip_timestamp.hour, sip_timestamp.minute)

        if hour_minute not in bars:
            bars[hour_minute] = {
                'o': ask_price
            }

        if 'l' not in bars[hour_minute]:
            bars[hour_minute]['l'] = ask_price
        else:
            bars[hour_minute]['l'] = min(ask_price, bars[hour_minute]['l'])

        if 'h' not in bars[hour_minute]:
            bars[hour_minute]['h'] = ask_price
        else:
            bars[hour_minute]['h'] = max(ask_price, bars[hour_minute]['h'])

        if i + 1 >= len(stock_df):
            break

        next_stock_df = stock_df.iloc[i + 1]
        next_hour_minute = (next_stock_df.sip_timestamp.hour, next_stock_df.sip_timestamp.minute)
        if next_hour_minute != hour_minute:
            bars[hour_minute]['c'] = ask_price
            bars_list.append(bars[hour_minute])

            if not buy_price and len(bars_list) >= 2:
                previous_bar = bars_list[-2]
                current_bar = bars_list[-1]

                is_previous_bar_red = previous_bar['o'] > previous_bar['c']
                is_current_bar_green = current_bar['o'] < current_bar['c']
                is_high_quality_red = (previous_bar['c'] / previous_bar['l'] - 1) <= max_high_close_delta
                is_high_quality_green = (current_bar['h'] / current_bar['c'] - 1) <= max_high_close_delta
                is_green_red_pair_found = is_previous_bar_red and is_current_bar_green and is_high_quality_green
                if is_green_red_pair_found:
                    last_green_bar = current_bar

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
    with open(STATE_FILE_LOCATION, 'r') as state_file:
        state = json.load(state_file)

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


def update_performance_state(state, delta, buy_time, buy_price, sell_price, sell_time, stock_symbol, the_date):
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
