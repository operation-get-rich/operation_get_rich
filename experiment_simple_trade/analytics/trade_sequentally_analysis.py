import datetime
import json

import matplotlib as mpl
from matplotlib import pyplot as plt
from dateutil.parser import parse

from utils import format_usd, DATETIME_FORMAT

mpl.use('macosx')

STATE_FILE_LOCATION = 'backtrader_state_analysis.json'
BACK_TRADER_STATE_FILE = '../state_02_03_high.json'
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
        symbol_trade_pairs = [
            (key, value) for key, value in backtrader_state['performance'][the_date].items()
        ]

        symbol_trade_pairs.sort(
            key=lambda x: x[1]['buy_time']
        )

        open_trade = None
        for symbol, trade in symbol_trade_pairs:
            if not open_trade or parse(trade['buy_time']) > parse(open_trade['sell_time']):
                print(f'\nTrading Stock Trade @ {the_date}: {symbol}, {trade["buy_time"]}')
                buy_price = trade['buy_price']
                sell_price = trade['sell_price']
                delta = ((sell_price / buy_price) - 1) * 100
                deltas.append(delta)
                print(f'Delta: {delta}')
                capital *= ((sell_price / buy_price))
                print(f'Capital {format_usd(capital)}')
                open_trade = trade
        capital_progressions.append(capital)
        dates.append(the_date)
    print(f'\nAverage Deltas {sum(deltas) / len(deltas)}%')
    accuracy = (sum([1 for d in deltas if d > 0]) / len(deltas))
    print(f'Accuracy: {accuracy * 100}%')
    print(f'Max delta: {max(deltas)}')
    print(f'Min delta: {min(deltas)}')
    plt.plot(backtrader_state['performance'].keys(), capital_progressions)
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
                'dates': dates,
                'capital_progressions': capital_progressions,
                'deltas': deltas
            }
        }


main()
