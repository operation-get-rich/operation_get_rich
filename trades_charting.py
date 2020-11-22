import json

from dateutil.parser import parse
from matplotlib import pyplot as plt
from matplotlib import dates as md

from utils import create_dir

root_chart_dir = './trades_chart'

with open('./trades_by_time.json') as trades_file:
    trades_db = json.load(trades_file)

plt.rcParams['timezone'] = 'US/Eastern'

for date in trades_db:
    create_dir(f'{root_chart_dir}/{date}')
    for ticker in trades_db[date]:
        print(f'Charting {date}/{ticker}')
        trades_by_timestamp = trades_db[date][ticker]

        dates = trades_by_timestamp.keys()
        dates = [parse(time) for time in dates]
        trades = trades_by_timestamp.values()

        plt.title(f'{date} {ticker}')
        plt.xlabel('Time')
        plt.ylabel('Trades')

        datenums = md.date2num(dates)
        plt.subplots_adjust(bottom=0.2)
        plt.xticks(rotation=45)
        ax = plt.gca()
        xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
        ax.xaxis.set_major_formatter(xfmt)

        plt.bar(datenums, trades, label=ticker)
        plt.show()
        # plt.savefig(f'{root_chart_dir}/{date}/{ticker}.png')
    plt.clf()
