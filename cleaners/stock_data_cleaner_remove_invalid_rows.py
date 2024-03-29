"""
Some rows has a row like
"open close low high volume"
This cleaner cleans it
"""

import pandas
from pandas import DataFrame

FILENAME = '../stock_price_small.csv'
stock_price_df = pandas.read_csv(FILENAME)  # type: DataFrame

stock_price_df.drop(
    stock_price_df[stock_price_df['open'] == 'open'].index,
    inplace=True
)

stock_price_df.to_csv('../stock_price_small.csv')
