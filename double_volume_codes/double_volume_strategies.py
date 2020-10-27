import csv
from datetime import datetime

from dateutil.parser import parse
import itertools

from constants import TIME_FORMAT

PROFIT_GOAL = .02
RISK_TOLERANCE = .01
VOLUME_MULTIPLIER = 2
PRICE_THRESHOLD = 20
TIME_THRESHOLD = '10:30'
USE_VOLUME_MULTIPLIER = True


def peek(it):
    try:
        first = next(it)
    except StopIteration:
        return None, it
    return first, itertools.chain([first], it)


class TradeInfo:
    def __init__(
            self,
            ticker,
            buy_time,
            sell_time,
            buy_price,
            sell_price,
            profit=0,
            loss=0,
            max_loss=0
    ):
        self.ticker = ticker
        self.buy_time = buy_time
        if isinstance(self.buy_time, str):
            self.buy_time = parse(self.buy_time)

        self.sell_time = sell_time
        if isinstance(self.sell_time, str):
            self.sell_time = parse(self.sell_time)

        self.buy_price = buy_price
        self.sell_price = sell_price
        self.profit = profit
        self.loss = loss
        self.max_loss = max_loss


def should_consider_trade_given_time(time):
    time_parsed = time.split()[1].split('-')[0].split(':')

    hour = int(time_parsed[0])
    minute = int(time_parsed[1])

    d1 = datetime.strptime(f'{hour}:{minute}', '%H:%M')
    d2 = datetime.strptime(TIME_THRESHOLD, '%H:%M')

    return d1 > d2

    # return hour >= 9 and minute >= 30


def find_first_trade(stock_file_path):
    total_gains = 0
    total_loss = 0
    total_max_loss = 0

    with open(stock_file_path) as stock_file:
        print(f'\nEvaluating {stock_file_path}')
        reader = csv.reader(stock_file)

        buy_price, last_red_volume, buy_time = None, None, None

        pointer = next(reader)
        while (pointer):
            try:
                pointer = next(reader)
            except StopIteration:
                break

            ticker, time, open_price, close, low, high, volume = pointer

            open_price = float(open_price)
            close = float(close)
            low = float(low)
            high = float(high)
            volume = int(volume)

            if not should_consider_trade_given_time(time):
                continue

            if open_price > PRICE_THRESHOLD:
                continue

            if buy_price:
                trade_info = evaluate_position(
                    buy_price, buy_time, close, ticker, time, total_gains, total_loss,
                    total_max_loss
                )
                if trade_info:
                    return trade_info
                continue

            if close < open_price:
                last_red_volume = volume
                continue

            if pass_volume_gate(last_red_volume, volume):
                next_pointer, reader = peek(reader)

                if not next_pointer:
                    continue

                _, next_time, _, _, _, next_high, _ = next_pointer
                next_high = float(next_high)

                if next_high > high:
                    # It means that at some point, the tick price goes above the current candle's high
                    buy_price = high  # TODO: We need ticker data to be more accurate
                    print(f'\nBuying at {buy_price}')
                    print(f'Buying at time = {next_time}')
                    buy_time = next_time
                continue

        if buy_price:
            trade_info = evaluate_position(
                buy_price, buy_time, close, ticker, time, total_gains, total_loss,
                total_max_loss,
                force_sell=True
            )
            return trade_info


def pass_volume_gate(last_red_volume, volume):
    if USE_VOLUME_MULTIPLIER:
        return last_red_volume and volume >= VOLUME_MULTIPLIER * last_red_volume
    else:
        return last_red_volume


def evaluate_position(buy_price, buy_time, close, ticker, time, total_gains, total_loss, total_max_loss,
                      force_sell=False):
    price_change = close / buy_price - 1
    trade_info = None
    if force_sell:
        print('FORCE_SELL')
    if price_change > 0 and (price_change >= PROFIT_GOAL or force_sell):
        print(f'Selling at {close}')
        print(f'Selling at time = {time}')
        print(f'Profit = {price_change * 100}%')
        total_gains += price_change
        print(f'total_gains = {total_gains * 100}%')
        print(f'total_loss = {total_loss * 100}%')
        print(f'total_max_loss = {total_max_loss * 100}%')

        trade_info = TradeInfo(
            ticker=ticker,
            buy_time=buy_time,
            sell_time=time,
            buy_price=buy_price,
            sell_price=close,
            profit=price_change,
        )

    elif price_change < 0 and (price_change * -1 >= RISK_TOLERANCE or force_sell):
        loss = price_change * -1
        print(f'Selling at {close}')
        print(f'Selling at time = {time}')
        print(f'*** Max Loss = {loss * 100}% ***')
        total_loss += RISK_TOLERANCE if not force_sell else loss
        total_max_loss += loss
        print(f'total_profit = {total_gains * 100}%')
        print(f'total_loss = {total_loss * 100}%')
        print(f'total_max_loss = {total_max_loss * 100}%')

        trade_info = TradeInfo(
            ticker=ticker,
            buy_time=buy_time,
            sell_time=time,
            buy_price=buy_price,
            sell_price=close,
            loss=RISK_TOLERANCE if not force_sell else loss,
            max_loss=loss
        )
    return trade_info
