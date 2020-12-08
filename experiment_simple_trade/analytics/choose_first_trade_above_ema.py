import datetime
import json

import matplotlib as mpl
from matplotlib import pyplot as plt
from dateutil.parser import parse

from utils import format_usd, DATETIME_FORMAT

mpl.use('macosx')

STATE_FILE_LOCATION = 'backtrader_state_analysis.json'
BACK_TRADER_STATE_FILE = '../state_polygon_01_03_minute_chart_bars.json'
STRATEGY_NAME = __file__.split('.')[0].split('/')[-1]


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


def main():
    with open(BACK_TRADER_STATE_FILE) as f:
        backtrader_state = json.load(f)
    capital = 25000
    capital_progressions = []
    deltas = []
    dates = []
    for the_date in backtrader_state['performance']:
        earliest_stock_trade = None
        earliest_symbol = None
        for stock_symbol in backtrader_state['performance'][the_date]:
            stock_trade = backtrader_state['performance'][the_date][stock_symbol]

            # if stock_trade['buy_price'] < stock_trade['ema'] or stock_trade['buy_price'] > stock_trade['vwap']:
            #     continue

            if stock_trade['ema'] < stock_trade['vwap']:
                continue

            # if stock_trade['buy_price'] < 1:
            #     continue


            if earliest_stock_trade is None:
                earliest_stock_trade = stock_trade
                earliest_symbol = stock_symbol
            buy_time = parse(stock_trade['buy_time'])
            earliest_buy_time = parse(earliest_stock_trade['buy_time'])
            if buy_time < earliest_buy_time:
                earliest_stock_trade = stock_trade
                earliest_symbol = stock_symbol

        if not earliest_stock_trade:
            continue

        print(f'\nEarliest Stock Trade @ {the_date}: {earliest_symbol}, {earliest_stock_trade["buy_time"]}')
        print(f'Delta: {earliest_stock_trade["delta"]}')
        buy_price = earliest_stock_trade['buy_price']
        sell_price = earliest_stock_trade['sell_price']
        deltas.append(((sell_price / buy_price) - 1) * 100)
        capital *= ((sell_price / buy_price))
        print(f'Capital {format_usd(capital)}')
        capital_progressions.append(capital)
        dates.append(the_date)
    print(f'\nAverage Deltas {sum(deltas) / len(deltas)}%')
    accuracy = (sum([1 for d in deltas if d > 0]) / len(deltas))
    print(f'Accuracy: {accuracy * 100}%')
    print(f'Max delta: {max(deltas)}')
    print(f'Min delta: {min(deltas)}')
    plt.plot(dates, capital_progressions)
    plt.xticks(rotation=90)
    plt.show()

    update_state(update_strategy_state, capital_progressions, accuracy, deltas, dates)


def update_strategy_state(state, capital_progressions, accuracy, deltas, dates):
    if BACK_TRADER_STATE_FILE in state:
        state[BACK_TRADER_STATE_FILE][STRATEGY_NAME] = {
            'end_capital': capital_progressions[-1],
            'accuracy': accuracy,
            'max_delta': max(deltas),
            'min_delta': min(deltas),
            'average_delta': sum(deltas) / len(deltas),
            'dates': dates,
            'capital_progressions': capital_progressions,
            'deltas': deltas
        }
    else:
        state[BACK_TRADER_STATE_FILE] = {
            STRATEGY_NAME: {
                'end_capital': capital_progressions[-1],
                'accuracy': accuracy,
                'max_delta': max(deltas),
                'min_delta': min(deltas),
                'average_delta': sum(deltas) / len(deltas),
                'dates': dates,
                'capital_progressions': capital_progressions,
                'deltas': deltas
            }
        }


main()
