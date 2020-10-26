import mplfinance as fplt
import numpy as np
import pandas as pd

stock_date = '2020-10-23'
stock_ticker = 'BIMI'

stock_df = pd.read_csv(
    f'./fron_double_volume_analysis/{stock_date}_{stock_ticker}.csv',
    index_col=0,
    parse_dates=True,
)

fplt.plot(
    data=stock_df,
    type='candle',
    ylabel='Price ($)',
    volume=True,
)