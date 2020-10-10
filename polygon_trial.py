from datetime import datetime

from polygon import RESTClient

from config import POLYGON_API_KEY

def main():
    with RESTClient(POLYGON_API_KEY) as client:
        resp = client.stocks_equities_daily_open_close("AAPL", "2018-03-02")
        resp = client.stocks_equities_historic_trades(
            symbol='AAPL',
            date='2020-09-11',
            timestamp=int(datetime(year=2020, month=9, day=11, hour=14, minute=30, second=5).timestamp()),
        )

main()