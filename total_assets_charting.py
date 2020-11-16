import json

from dateutil.parser import parse
from matplotlib import pyplot as plt
from matplotlib import dates as md

root_chart_dir = './total_assets_progression_charts'

with open('./total_assets_progression_by_time.json') as total_assets_file:
    total_assets_db = json.load(total_assets_file)

plt.rcParams['timezone'] = 'US/Eastern'

for date in total_assets_db:
    for ticker in total_assets_db[date]:
        print(f'Charting {date}/{ticker}')
        assets_progression_by_timestamp = total_assets_db[date][ticker]

        dates = assets_progression_by_timestamp.keys()
        dates = [parse(time) for time in dates]
        assets = assets_progression_by_timestamp.values()

        plt.title(f'{date} {ticker}')
        plt.xlabel('Time')
        plt.ylabel('Assets (USD)')

        datenums = md.date2num(dates)
        plt.subplots_adjust(bottom=0.2)
        plt.xticks(rotation=45)
        ax = plt.gca()
        xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
        ax.xaxis.set_major_formatter(xfmt)

        plt.plot(datenums, assets, label=ticker, marker='.')
    plt.legend()
    plt.show()
    # plt.savefig(f'{root_chart_dir}/{date}.png')
    plt.clf()
