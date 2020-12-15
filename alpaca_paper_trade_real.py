import asyncio
import datetime
import json
import logging
import math
import time
from datetime import timedelta
from statistics import mean

import alpaca_trade_api as tradeapi
import pandas as pd
import pytz
import torch
from alpaca_trade_api import StreamConn

from TraderGRU import TraderGRU
from config import PAPER_ALPACA_API_KEY, PAPER_ALPACA_SECRET_KEY, PAPER_ALPACA_BASE_URL
from stock_dataset import PercentChangeNormalizer as PCN
from utils import DATETIME_FORMAT, get_all_ticker_names, US_CENTRAL_TIMEZONE, DATE_FORMAT, format_usd

logging.basicConfig(
    handlers=[
        logging.FileHandler("alpaca_paper_trade.log"),
        logging.StreamHandler()
    ],
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO,
)

model = TraderGRU(
    input_size=7,
    hidden_size=5 * 7,
    hard=True
)
model.load_state_dict(
    torch.load(
        './TraderGRU_NextTrade/best_model.pt',
        map_location=torch.device('cpu')
    )
)
model.eval()

gap_stocks_by_date_dir = 'gaped_up_stocks_early_volume_1e5_gap_10_by_date'


def get_trade_from_model(inputs):
    inputs_tensor = torch.FloatTensor([inputs])
    normalized_inputs_tensor = PCN.normalize_volume(inputs_tensor)
    normalized_inputs_tensor = PCN.normalize_price_into_percent_change(normalized_inputs_tensor)
    trades = model(normalized_inputs_tensor)
    trade = trades.detach().numpy()[0][-1]
    return trade


def sell(symbol, shares_owned, trade, capital, slippage=.005):
    current_price = api.polygon.last_trade(symbol)
    shares_to_sell = math.ceil(abs(trade) * shares_owned)
    logging.info(dict(
        type='trade_sell',
        message=f'Selling {shares_to_sell} of {symbol} @ ~{format_usd(current_price)}',
        payload=dict(
            trade=trade,
            capital=capital,
            shares_owned=shares_owned,
            shares_to_sell=shares_to_sell,
            price=current_price
        )
    ))
    if shares_to_sell > 0:
        # TODO: Temporary now that we know on_account_updates is unreliable update the shares owned here
        bar_state[symbol][SHARES_OWNED_KEY] -= shares_to_sell
        bar_state[symbol][CAPITAL_KEY] += shares_to_sell * current_price
        api.submit_order(
            symbol=symbol,
            qty=shares_to_sell,
            side=SIDE_SELL,
            type='limit',
            time_in_force=TIME_IN_FORCE_GTC,
            limit_price=str(current_price * (1 - slippage))
        )


def buy(symbol, trade, capital, slippage=.005):
    capital_to_use = capital * trade
    current_price = float(api.polygon.last_trade(symbol).price)
    shares_to_buy = math.floor(capital_to_use / current_price)
    logging.info(dict(
        type='trade_buy',
        message=f'Buying {shares_to_buy} of {symbol} @ ~{format_usd(current_price)}',
        payload=dict(
            trade=trade,
            capital=capital,
            capital_to_use=capital_to_use,
            shares_to_buy=shares_to_buy,
            price=current_price
        )
    ))
    if shares_to_buy > 0:
        # TODO: Temporary now that we know on_account_updates is unreliable update the shares owned here
        bar_state[symbol][SHARES_OWNED_KEY] += shares_to_buy
        bar_state[symbol][CAPITAL_KEY] -= shares_to_buy * current_price
        api.submit_order(
            symbol=symbol,
            qty=shares_to_buy,
            side='buy',
            type='limit',
            time_in_force='gtc',
            limit_price=str(current_price * (1 + slippage))
        )


def compute_vwap(price_volume_products, volumes, ta_period=14):
    sum_pv = sum(price_volume_products[-ta_period:])
    sum_v = sum(volumes[-ta_period:])
    vwap = sum_pv / sum_v
    return vwap


def compute_ema(close_prices, ta_period=14):
    close_prices_series = pd.Series(close_prices)
    ewm = close_prices_series.ewm(
        span=ta_period,
        min_periods=ta_period,
        adjust=False
    ).mean()  # ewm = exponential moving window
    return ewm.values[-1]


api = tradeapi.REST(
    key_id=PAPER_ALPACA_API_KEY,
    secret_key=PAPER_ALPACA_SECRET_KEY,
    base_url=PAPER_ALPACA_BASE_URL
)

stream = StreamConn(
    key_id=PAPER_ALPACA_API_KEY,
    secret_key=PAPER_ALPACA_SECRET_KEY,
    base_url=PAPER_ALPACA_BASE_URL
)

stocks_bar_history = {}

bar_state = {}
quote_state = {}
TA_PERIOD = 14

RAW_FEATURES_KEY = 'raw_features'
MODEL_INPUTS_KEY = 'model_inputs'
VOLUMES_KEY = 'volumes'
PRICE_VOLUME_PRODUCTS_KEY = 'price_volume_products'
CLOSE_PRICES_KEY = 'close_prices'
TRADES_KEY = 'trades'
TRADES_BY_TIME_KEY = 'trades_by_time'
CAPITAL_KEY = 'capital'
SHARES_OWNED_KEY = 'shares'
TOTAL_ASSETS_PROGRESSION_KEY = 'total_assets_progression'

ASK_KEY = 'ask'
BID_KEY = 'bid'

SIDE_SELL = 'sell'
SIDE_BUY = 'buy'
ORDER_TYPE_LIMIT = 'limit'
TIME_IN_FORCE_GTC = 'gtc'

GAPPED_UP_CACHE_LOCATION = './alpaca_paper_trade_cache.json'


def get_best_bid_price(symbol):
    return max(quote_state[symbol][BID_KEY].values())


def get_best_ask_price(symbol):
    return min(quote_state[symbol][ASK_KEY].values())


async def handle_bar(bar):
    symbol = bar.symbol
    bar_state[symbol][RAW_FEATURES_KEY].append(
        [
            bar.start.timestamp(),
            bar.open,
            bar.close,
            bar.low,
            bar.high,
            bar.volume
        ]
    )

    symbol_bar_state = bar_state[symbol]
    last_raw_feature = symbol_bar_state[RAW_FEATURES_KEY][-1]
    (
        last_time,
        last_open_price,
        last_close_price,
        last_low_price,
        last_high_price,
        last_volume
    ) = last_raw_feature

    last_typical_price = mean([last_close_price, last_low_price, last_high_price])

    volumes = symbol_bar_state[VOLUMES_KEY]
    volumes.append(last_volume)

    price_volume_products = symbol_bar_state[PRICE_VOLUME_PRODUCTS_KEY]
    price_volume_products.append(last_volume * last_typical_price)

    close_prices = symbol_bar_state[CLOSE_PRICES_KEY]
    close_prices.append(last_close_price)

    capital = symbol_bar_state[CAPITAL_KEY]
    shares_owned = symbol_bar_state[SHARES_OWNED_KEY]

    able_to_compute_ta = len(symbol_bar_state[RAW_FEATURES_KEY]) >= TA_PERIOD

    if able_to_compute_ta:
        last_vwap = compute_vwap(price_volume_products, volumes, ta_period=TA_PERIOD)
        last_ema = compute_ema(close_prices, ta_period=TA_PERIOD)

        last_input = [last_open_price, last_close_price, last_low_price, last_high_price, last_volume, last_vwap,
                      last_ema]
        symbol_bar_state[MODEL_INPUTS_KEY].append(last_input)

        trade = get_trade_from_model(symbol_bar_state[MODEL_INPUTS_KEY])

        symbol_bar_state[TRADES_KEY].append(trade)

        if trade > 0:
            buy(symbol, trade, capital)
        if trade < 0:
            sell(symbol, shares_owned, trade, capital)
    save_bar_state()
    logging.info(dict(
        type='bar_state_update',
        payload=dict(quote_state=quote_state)
    ))


async def handle_quote(quote):
    symbol = quote.symbol
    if symbol not in quote_state:
        quote_state[symbol] = {
            ASK_KEY: {},
            BID_KEY: {}
        }

    quote_state[symbol][ASK_KEY][quote.askexchange] = quote.askprice
    quote_state[symbol][BID_KEY][quote.bidexchange] = quote.bidprice


@stream.on(r'trade_updates')
async def on_trade_updates(conn, channel, account):
    """
    {
        'event': 'fill',
        'order': {
            'asset_class': 'us_equity',
            'asset_id': 'b0b6dd9d-8b9b-48a9-ba46-b9d54906e415',
            'canceled_at': None,
            'client_order_id': '9548909c-bb9a-402d-b916-95a13e917ce6',
            'created_at': '2020-11-19T19:57:02.542513Z',
            'expired_at': None,
            'extended_hours': False,
            'failed_at': None,
            'filled_at': '2020-11-19T19:57:02.695986Z',
            'filled_avg_price': '118.42',
            'filled_qty': '84',
            'hwm': None,
            'id': '88c7f734-f2e2-4d46-96e8-5cfb9a2fce86',
            'legs': None,
            'limit_price': '118.42',
            'order_class': '',
            'order_type': 'limit',
            'qty': '84',
            'replaced_at': None,
            'replaced_by': None,
            'replaces': None,
            'side': 'buy',
            'status': 'filled',
            'stop_price': None,
            'submitted_at': '2020-11-19T19:57:02.537808Z',
            'symbol': 'AAPL',
            'time_in_force': 'gtc',
            'trail_percent': None,
            'trail_price': None,
            'type': 'limit',
            'updated_at': '2020-11-19T19:57:02.706647Z'
        },
        'position_qty': '84',
        'price': '118.42',
        'qty': '84',
        'timestamp': '2020-11-19T19:57:02.695986612Z'
    }
    """
    if account.event == 'fill':
        filled_quantity = int(account.order['filled_qty'])
        filled_avg_price = float(account.order['filled_avg_price'])
        if account.order['side'] == 'buy':
            bar_state[account.order['symbol']][SHARES_OWNED_KEY] += filled_quantity
            bar_state[account.order['symbol']][CAPITAL_KEY] -= filled_quantity * filled_avg_price
            logging.info(dict(
                type='buy_filled',
                message=f'Bought {account.order["symbol"]} @ {format_usd(filled_avg_price)}',
                payload=dict(
                    symbol=account.order["symbol"],
                    price=filled_avg_price,
                    shares_owned=bar_state[account.order['symbol']][SHARES_OWNED_KEY],
                    capital=bar_state[account.order['symbol']][CAPITAL_KEY]
                )
            ))
        if account.order['side'] == 'sell':
            bar_state[account.order['symbol']][SHARES_OWNED_KEY] -= filled_quantity
            bar_state[account.order['symbol']][CAPITAL_KEY] += filled_quantity * filled_avg_price
            logging.info(dict(
                type='sell_filled',
                message=f'Sold {account.order["symbol"]} @ {format_usd(filled_avg_price)}',
                payload=dict(
                    symbol=account.order["symbol"],
                    price=filled_avg_price,
                    shares_owned=bar_state[account.order['symbol']][SHARES_OWNED_KEY],
                    capital=bar_state[account.order['symbol']][CAPITAL_KEY]
                )
            ))
        return


@stream.on(r'^AM')
async def on_minute_bars(conn, channel, bar):
    """
    {
        'symbol': 'MSFT',
        'volume': 3841,
        'totalvolume': 210280,
        'vwap': 216.905,
        'open': 216.89,
        'close': 216.83,
        'high': 216.94,
        'low': 216.83,
        'average': 216.576,
        'start': 1605628500000,
        'end': 1605628560000,
        'timestamp': 1605628500000
    }
    """
    await handle_bar(bar)


GAP_UP_THRESHOLD = .10
VOLUME_THRESHOLD = 1e05
COMPANY_STEPS = 200


def update_barstate(barset):
    for symbol in barset:
        for bar in barset[symbol]:
            if symbol not in bar_state:
                bar_state[symbol] = {
                    RAW_FEATURES_KEY: [],
                    MODEL_INPUTS_KEY: [],
                    VOLUMES_KEY: [],
                    PRICE_VOLUME_PRODUCTS_KEY: [],
                    CLOSE_PRICES_KEY: [],
                    TRADES_KEY: [],
                    CAPITAL_KEY: 0,
                    SHARES_OWNED_KEY: 0,
                    TOTAL_ASSETS_PROGRESSION_KEY: {},
                    TRADES_BY_TIME_KEY: {},
                }

            bar_state[symbol][RAW_FEATURES_KEY].append(
                [
                    bar.t.timestamp(),
                    bar.o,
                    bar.c,
                    bar.l,
                    bar.h,
                    bar.v
                ]
            )

            symbol_bar_state = bar_state[symbol]
            last_raw_feature = symbol_bar_state[RAW_FEATURES_KEY][-1]
            (
                last_time,
                last_open_price,
                last_close_price,
                last_low_price,
                last_high_price,
                last_volume
            ) = last_raw_feature

            last_typical_price = mean([last_close_price, last_low_price, last_high_price])

            volumes = symbol_bar_state[VOLUMES_KEY]
            volumes.append(last_volume)

            price_volume_products = symbol_bar_state[PRICE_VOLUME_PRODUCTS_KEY]
            price_volume_products.append(last_volume * last_typical_price)

            close_prices = symbol_bar_state[CLOSE_PRICES_KEY]
            close_prices.append(last_close_price)

            able_to_compute_ta = len(symbol_bar_state[RAW_FEATURES_KEY]) >= TA_PERIOD

            if able_to_compute_ta:
                last_vwap = compute_vwap(price_volume_products, volumes, ta_period=TA_PERIOD)
                last_ema = compute_ema(close_prices, ta_period=TA_PERIOD)

                last_input = [last_open_price, last_close_price, last_low_price, last_high_price, last_volume,
                              last_vwap,
                              last_ema]
                symbol_bar_state[MODEL_INPUTS_KEY].append(last_input)
        logging.info(dict(
            type='instantiate_bar_state_finish',
            message=f'Finished Instantiating {symbol}',
            payload=dict(bar_state=bar_state)
        ))


def get_current_datetime():
    return datetime.datetime.now(pytz.timezone(US_CENTRAL_TIMEZONE))


def save_bar_state(bar_state_save_file='./alpaca_paper_trade_real_bar_state.json'):
    with open(bar_state_save_file, 'w') as f:
        f.seek(0)
        json.dump(bar_state, f)
        f.truncate()


def find_gapped_up_symbols():
    today = datetime.datetime.now(pytz.timezone(US_CENTRAL_TIMEZONE))

    cache = read_gapped_up_symbols_cache()

    cache_date_key = today.date().strftime(DATE_FORMAT)
    if cache_date_key in cache and cache[cache_date_key]:
        return cache[cache_date_key]

    eight_am = today.replace(hour=8, minute=30, second=0, microsecond=0)
    yesterday = eight_am - timedelta(days=3)  # TODO: Get the previous market open, not yesterday
    eight_am_str = get_alpaca_time_str_format(eight_am)
    yesterday_str = get_alpaca_time_str_format(yesterday)
    gapped_up_symbols = []
    start = 0
    tickers = get_all_ticker_names()
    while start < len(tickers):
        end = min(len(tickers), start + COMPANY_STEPS)

        print(f'Downloading tickers: {tickers[start:end]}')
        barset = api.get_barset(
            symbols=','.join(tickers[start:end]),
            timeframe='15Min',
            start=yesterday_str,
            end=eight_am_str,
        )  # TODO: Use Polygon barset api when using Polygon trained model

        for symbol in barset:
            open_index = None
            for i in range(len(barset[symbol])):
                if barset[symbol][i].t.date() == today.date():
                    open_index = i
                    break
            if not open_index:
                continue
            cummulative_volume = sum([barset[symbol][i].v for i in range(open_index, len(barset[symbol]))])
            open_price = barset[symbol][open_index].o
            prev_day_close_price = barset[symbol][0].c
            gap = (open_price / prev_day_close_price) - 1
            is_price_gapped_up = gap > GAP_UP_THRESHOLD
            if is_price_gapped_up and cummulative_volume > VOLUME_THRESHOLD:
                gapped_up_symbols.append((symbol, gap, cummulative_volume))
        start += COMPANY_STEPS

    cache[cache_date_key] = gapped_up_symbols
    write_gapped_up_cache(cache)

    return gapped_up_symbols


def write_gapped_up_cache(cache):
    with open(GAPPED_UP_CACHE_LOCATION, 'w') as f:
        json.dump(cache, f)


def read_gapped_up_symbols_cache():
    try:
        with open(GAPPED_UP_CACHE_LOCATION) as f:
            cache = json.load(f)
    except FileNotFoundError:
        cache = {}
    return cache


def get_alpaca_time_str_format(
        the_datetime  # type: datetime
):
    alpaca_time_str = 'T'.join(the_datetime.strftime(DATETIME_FORMAT).split())
    alpaca_time_str = alpaca_time_str[0:-2] + ':' + alpaca_time_str[-2:]
    return alpaca_time_str


def main():
    gapped_up_symbols = find_gapped_up_symbols()

    gapped_up_symbols = [gapped_up_symbols[2]]

    end_time = get_current_datetime()
    start_time = end_time - timedelta(days=1)  # TODO: Get the previous market open, not yesterday
    # start_time = end_time - timedelta(minutes=1)

    end_time_str = get_alpaca_time_str_format(end_time)
    start_time_str = get_alpaca_time_str_format(start_time)

    barset = api.get_barset(
        symbols=','.join([s[0] for s in gapped_up_symbols]),
        timeframe='1Min',
        end=end_time_str,
        start=start_time_str,
    )  # TODO: Use Polygon barset api when using Polygon trained model

    update_barstate(barset)
    save_bar_state()

    market_open_hour = get_current_datetime().replace(
        hour=8,
        minute=30
    )
    while get_current_datetime() < market_open_hour:
        now_time = get_current_datetime()
        # now_time = datetime.datetime.now(pytz.timezone(US_CENTRAL_TIMEZONE)).replace(hour=8, minute=29, second=1)

        end_time = now_time
        start_time = end_time - timedelta(minutes=1)

        end_time_str = get_alpaca_time_str_format(end_time)
        start_time_str = get_alpaca_time_str_format(start_time)

        barset = api.get_barset(
            symbols=','.join([s[0] for s in gapped_up_symbols]),
            timeframe='1Min',
            end=end_time_str,
            start=start_time_str,
        )  # TODO: Use Polygon barset api when using Polygon trained model

        update_barstate(barset)
        save_bar_state()

        time.sleep(
            min((market_open_hour - end_time).seconds, 60)
        )
    capital = 25000  # TODO: Figure out how to get current account's capital
    capital_per_symbol = math.floor(capital / len(gapped_up_symbols))
    channels = []
    for symbol, gap, volume in gapped_up_symbols:
        assert symbol in bar_state
        bar_state[symbol][CAPITAL_KEY] = capital_per_symbol
        channels.append(f'AM.{symbol}')
    channels.append('trade_updates')

    loop = stream.loop
    loop.run_until_complete(
        asyncio.gather(
            stream.subscribe(channels)
        )
    )


main()
