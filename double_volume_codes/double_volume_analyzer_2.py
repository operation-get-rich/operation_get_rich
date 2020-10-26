import os
from collections import defaultdict
import datetime

from double_volume_codes.fron_strategies import find_first_trade

root_dir = 'gaped_up_stocks_early_volume_1e5_gap_10_by_date'

STARTING_CAPITAL = 25000
STOP_COMPOUNDING_CAPITAL_TARGET = 25000

total_profit = 0
total_loss = 0
total_max_loss = 0

max_profit_in_the_day = 0
max_loss_in_the_day = 0

trade_count = 0
winning_trade = 0
longest_consecutive_green_days = 0
consecutive_green_days_counter = 0
longest_consecutive_red_days = 0
consecutive_red_days_counter = 0

keeps_compounding = False
current_capital = STARTING_CAPITAL

max_capital = 0

has_reached_compounding_target = False

monthly_capital_datas = defaultdict(int)
daily_capital_datas = []

buy_times = []
total_holding_time = datetime.timedelta()
for the_date in sorted(os.listdir(f'../{root_dir}')):
    max_capital = max(max_capital, current_capital)
    year = int(the_date.split('-')[0])

    if year != 2019:
        continue

    monthly_capital_datas[the_date[0:-3]] = current_capital
    daily_capital_datas.append(current_capital)

    print(f'\nDate = {the_date}')

    date_dir = f'../{root_dir}/{the_date}'

    trade_infos = []

    current_earliest_buy_time = None
    current_latest_buy_time = None
    trade_of_the_day = None

    for stock_file in os.listdir(date_dir):
        trade_info = find_first_trade(stock_file_path=os.path.join(date_dir, stock_file))
        if trade_info:
            if current_earliest_buy_time is None or trade_info.buy_time < current_earliest_buy_time:
                trade_of_the_day = trade_info
                current_earliest_buy_time = trade_info.buy_time
            if current_latest_buy_time is None or trade_of_the_day.buy_time > current_latest_buy_time:
                current_latest_buy_time = trade_of_the_day.buy_time

    if trade_of_the_day:

        if not has_reached_compounding_target and current_capital >= STOP_COMPOUNDING_CAPITAL_TARGET:
            has_reached_compounding_target = True
            capital_to_print = '${:,.2f}'.format(STOP_COMPOUNDING_CAPITAL_TARGET)
            current_capital_to_print = '${:,.2f}'.format(current_capital)
            print(f'\n**** REACHED {capital_to_print}. Current: {current_capital_to_print} ***')
            print(f'Date = {the_date}')

        print(f'\nChosen Trade stats for {the_date}')

        print(f'Ticker = {trade_of_the_day.ticker}')
        print(f'Buy time = {trade_of_the_day.buy_time}')
        print(f'Sell time = {trade_of_the_day.sell_time}')

        total_holding_time += trade_of_the_day.sell_time - trade_of_the_day.buy_time

        if trade_of_the_day.profit:
            print(f'Profit: +{trade_of_the_day.profit * 100}%')

            total_profit += trade_of_the_day.profit

            winning_trade += 1

            if trade_of_the_day.profit > max_profit_in_the_day:
                max_profit_in_the_day = trade_of_the_day.profit

            consecutive_green_days_counter += 1
            longest_consecutive_green_days = max(longest_consecutive_green_days, consecutive_green_days_counter)

            consecutive_red_days_counter = 0

            if keeps_compounding:
                current_capital *= 1 + trade_of_the_day.profit
            else:
                if current_capital >= STOP_COMPOUNDING_CAPITAL_TARGET:
                    current_capital += STOP_COMPOUNDING_CAPITAL_TARGET * trade_of_the_day.profit
                else:
                    current_capital *= 1 + trade_of_the_day.profit

        if trade_of_the_day.loss:
            print(f'Loss: -{trade_of_the_day.loss * 100}%')
            print(f'Max Loss: -{trade_of_the_day.max_loss * 100}%')

            total_loss += trade_of_the_day.loss
            total_max_loss += trade_of_the_day.max_loss

            if trade_of_the_day.max_loss > max_loss_in_the_day:
                max_loss_in_the_day = trade_of_the_day.max_loss

            consecutive_red_days_counter += 1
            longest_consecutive_red_days = max(longest_consecutive_red_days, consecutive_red_days_counter)

            consecutive_green_days_counter = 0

            if keeps_compounding:
                current_capital *= 1 - trade_of_the_day.loss
            else:
                if current_capital >= STOP_COMPOUNDING_CAPITAL_TARGET:
                    current_capital -= STOP_COMPOUNDING_CAPITAL_TARGET * trade_of_the_day.loss
                else:
                    current_capital *= 1 - trade_of_the_day.loss

        trade_count += 1

print(f'\ntotal_profit: {total_profit * 100}%')
print(f'total_loss: {total_loss * 100}%')
print(f'total_max_loss: {total_max_loss * 100}%')
print(f'\nNET PROFIT: {(total_profit - total_loss) * 100}%')

end_capital = '${:,.2f}'.format(current_capital)
print(f'\nEnd Capital = {end_capital}')
max_capital = '${:,.2f}'.format(max_capital)
print(f'Max Capital = {max_capital}')

print(f'\nTrade Count = {trade_count}')
print(f'Winning Trade Count = {winning_trade}')
print(f'Losing Trade Count = {trade_count - winning_trade}')
print(f'Accuracy: {(float(winning_trade) / trade_count) * 100}%')

print(f'\n Average holding time: {total_holding_time/trade_count}')

print(f'\nAverage Profit: {(total_profit / winning_trade) * 100}%')
print(f'Average Maxloss: {(total_max_loss / (trade_count - winning_trade)) * 100}%')

print(f'\nMax profit of the day: +{max_profit_in_the_day * 100}%')
print(f'Max loss of the day: -{max_loss_in_the_day * 100}%')
print(f'Longet consecutive green days = {longest_consecutive_green_days}')
print(f'Longet consecutive red days = {longest_consecutive_red_days}')

print(f'\nLatest buy time: {current_latest_buy_time}')

import matplotlib.pyplot as plt
import numpy as np

#
plt.plot(monthly_capital_datas.values())
plt.xticks(np.arange(0, len(monthly_capital_datas), 1.0))
plt.show()

# plt.plot(daily_capital_datas)
# plt.show()
