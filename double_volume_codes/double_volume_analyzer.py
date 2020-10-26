import csv
import itertools

the_date = '2020-10-22'
ticker = 'EVK'


PROFIT_GOAL = .01
RISK_TOLERANCE = .03
VOLUME_MULTIPLIER = 4

def peek(it):
    first = next(it)
    return first, itertools.chain([first], it)

total_gains = 0
total_loss = 0
total_max_loss = 0

with open(f'./fron_double_volume_stocks/{the_date}_{ticker}.csv') as stock_file:
    reader = csv.reader(stock_file)

    last_red_volume = None
    buy_price = None
    buy_time = None

    pointer = next(reader)
    while (pointer):
        try:
            pointer = next(reader)
        except StopIteration:
            break

        time, open, close, low, high, volume = pointer

        open = float(open)
        close = float(close)
        low = float(low)
        high = float(high)
        volume = int(volume)

        if buy_price:
            gains = close / buy_price - 1
            loss = gains * -1
            if gains >= PROFIT_GOAL:
                print(f'Selling at {close}')
                print(f'Selling at time = {time}')
                print(f'Profit = {gains * 100}%')

                total_gains += gains

                print(f'total_profit = {total_gains * 100}%')
                print(f'total_loss = {total_loss * 100}%')
                print(f'total_max_loss = {total_max_loss * 100}%')

                last_red_volume = None
                buy_price = None
                buy_time = None

            elif loss >= RISK_TOLERANCE:
                print(f'Selling at {close}')
                print(f'Selling at time = {time}')
                print(f'*** Max Loss = {loss * 100}% ***')

                total_loss += RISK_TOLERANCE
                total_max_loss += loss

                print(f'total_profit = {total_gains * 100}%')
                print(f'total_loss = {total_loss * 100}%')
                print(f'total_max_loss = {total_max_loss * 100}%')

                last_red_volume = None
                buy_price = None
                buy_time = None
            continue

        if close < open:
            print(f'\nLast Red Volume = {volume}')
            print(f'Last Red time = {time}')
            last_red_volume = volume
            continue

        if last_red_volume and volume >= VOLUME_MULTIPLIER * last_red_volume:
            next_pointer, reader = peek(reader)

            next_high = float(next_pointer[4])
            next_time = next_pointer[0]
            if next_high > high:
                buy_price = high # TODO: We need ticker data to be more accurate
                buy_time = time
                print(f'\nBuying at {buy_price}')
                print(f'Buying at time = {next_time}')
            continue

print(f'total_gains = {total_gains * 100}%')
print(f'total_loss = {total_loss * 100}%')
print(f'total_max_loss = {total_max_loss * 100}%')
print(f'profit = {(total_gains - total_loss)*100}%')

