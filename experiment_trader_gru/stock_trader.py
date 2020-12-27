import json
import logging
import math
from collections import namedtuple
from statistics import mean

import pandas as pd
import torch
import alpaca_trade_api as tradeapi

from config import PAPER_ALPACA_API_KEY, PAPER_ALPACA_SECRET_KEY, PAPER_ALPACA_BASE_URL
from experiment_trader_gru.normalizer import PercentChangeNormalizer as PCN
from utils import get_current_datetime, DATE_FORMAT, get_previous_market_open, get_alpaca_time_str_format, \
    get_all_ticker_names, format_usd, DATETIME_FORMAT

api = tradeapi.REST(
    key_id=PAPER_ALPACA_API_KEY,
    secret_key=PAPER_ALPACA_SECRET_KEY,
    base_url=PAPER_ALPACA_BASE_URL
)

SymbolVolumeGap = namedtuple('SymbolVolumeGap', field_names=['symbol', 'volume', 'gap'])


class GappedUpStockFinder:
    GAPPED_UP_CACHE_LOCATION = './alpaca_paper_trade_symbol_volume_gap_cache.json'
    GAP_UP_THRESHOLD = .10
    VOLUME_THRESHOLD = 100_000

    @classmethod
    def find_gapped_up_stock(cls):
        current_datetime = get_current_datetime()

        cache = cls._read_gapped_up_stock_cache()

        cache_date_key = current_datetime.date().strftime(DATE_FORMAT)

        if cache_date_key in cache and cache[cache_date_key]:
            return cache[cache_date_key]

        eight_am = current_datetime.replace(hour=8, minute=00, second=0, microsecond=0)
        previous_market_open = get_previous_market_open(eight_am)

        eight_am_str = get_alpaca_time_str_format(eight_am)
        yesterday_str = get_alpaca_time_str_format(previous_market_open)

        symbol_volume_gaps = []
        start = 0

        tickers = get_all_ticker_names()
        company_steps = 200
        while start < len(tickers):
            end = min(len(tickers), start + company_steps)

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
                    if barset[symbol][i].t.date() == current_datetime.date():
                        open_index = i
                        break

                if not open_index:
                    continue

                cummulative_volume = sum([barset[symbol][i].v for i in range(open_index, len(barset[symbol]))])

                open_price = barset[symbol][open_index].o
                prev_day_close_price = barset[symbol][open_index - 1].c

                gap = (open_price / prev_day_close_price) - 1
                is_price_gapped_up = gap > cls.GAP_UP_THRESHOLD

                if is_price_gapped_up and cummulative_volume > cls.VOLUME_THRESHOLD:
                    symbol_volume_gaps.append(
                        SymbolVolumeGap(symbol, cummulative_volume, gap)
                    )
            start += company_steps

        cache[cache_date_key] = symbol_volume_gaps
        cls._write_gapped_up_cache(cache)

        return symbol_volume_gaps

    @staticmethod
    def _read_gapped_up_stock_cache():
        try:
            with open(GappedUpStockFinder.GAPPED_UP_CACHE_LOCATION) as f:
                cache = json.load(f)
        except FileNotFoundError:
            cache = {}
        return cache

    @staticmethod
    def _write_gapped_up_cache(cache):
        with open(GappedUpStockFinder.GAPPED_UP_CACHE_LOCATION, 'w') as f:
            json.dump(cache, f)


class RawFeature:
    def __init__(
            self,
            time,  # type: float
            open,  # type: float
            close,  # type: float
            low,  # type: float
            high,  # type: float
            volume,  # type: float
    ):
        self.time = time
        self.open = open
        self.close = close
        self.low = low
        self.high = high
        self.volume = volume

    def __repr__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class StockState:
    DEFAULT_TA_PERIOD = 14

    def __init__(
            self,
            capital=0,  # type: float
            shares_owned=0,  # type: int

    ):
        self.capital = capital
        self.shares_owned = shares_owned
        self.raw_features = []  # type: List[RawFeature]

        self.volumes = []  # type: List[float]
        self.price_volume_products = []  # type: List[float]
        self.close_prices = []  # type: List[float]
        self.unnormalized_model_inputs = []  # type: List[float]

    def update_from_realtime_bars(
            self,
            # TODO: Update to find the real type of "Bar" here
            bars  # type: List[Bar]
    ):
        for bar in bars:
            self.raw_features.append(
                RawFeature(
                    time=bar.start.timestamp(),
                    open=bar.open,
                    close=bar.close,
                    low=bar.low,
                    high=bar.high,
                    volume=bar.volume
                )
            )
            last_raw_feature = self._last_raw_feature
            last_typical_price = mean([
                last_raw_feature.close,
                last_raw_feature.low,
                last_raw_feature.high
            ])

            self.volumes.append(last_raw_feature.volume)
            self.price_volume_products.append(last_typical_price * last_raw_feature.volume)
            self.close_prices.append(last_raw_feature.close)

            if self._able_to_compute_ta:
                last_input = [last_raw_feature.open,
                              last_raw_feature.close,
                              last_raw_feature.low,
                              last_raw_feature.high,
                              last_raw_feature.volume,
                              self._last_vwap,
                              self._last_ema]
                self.unnormalized_model_inputs.append(last_input)

    def update_from_bars(
            self,
            bars  # type: List[Bar]
    ):
        for bar in bars:
            self.raw_features.append(
                RawFeature(
                    time=bar.t.strftime(DATETIME_FORMAT),
                    open=bar.o,
                    close=bar.c,
                    low=bar.l,
                    high=bar.h,
                    volume=bar.v
                )
            )
            last_raw_feature = self._last_raw_feature
            last_typical_price = mean([
                last_raw_feature.close,
                last_raw_feature.low,
                last_raw_feature.high
            ])

            self.volumes.append(last_raw_feature.volume)
            self.price_volume_products.append(last_typical_price * last_raw_feature.volume)
            self.close_prices.append(last_raw_feature.close)

            if self._able_to_compute_ta:
                last_input = [last_raw_feature.open,
                              last_raw_feature.close,
                              last_raw_feature.low,
                              last_raw_feature.high,
                              last_raw_feature.volume,
                              self._last_vwap,
                              self._last_ema]
                self.unnormalized_model_inputs.append(last_input)

    @property
    def _last_vwap(self, ta_period=DEFAULT_TA_PERIOD):
        assert self._able_to_compute_ta, "Invalid call, TA cannot be computed yet"
        sum_pv = sum(self.price_volume_products[-ta_period:])
        sum_v = sum(self.volumes[-ta_period:])
        vwap = sum_pv / sum_v
        return vwap

    @property
    def _last_ema(self, ta_period=DEFAULT_TA_PERIOD):
        assert self._able_to_compute_ta, "Invalid call, TA cannot be computed yet"
        close_prices_series = pd.Series(self.close_prices)
        ewm = close_prices_series.ewm(
            span=ta_period,
            min_periods=ta_period,
            adjust=False
        ).mean()  # ewm = exponential moving window
        return ewm.values[-1]

    @property
    def _last_raw_feature(self):
        # type: (...) -> RawFeature
        assert len(self.raw_features) > 0, "Invalid call, raw_features is not yet populated"
        return self.raw_features[-1]

    @property
    def _able_to_compute_ta(self, ta_period=DEFAULT_TA_PERIOD):
        return len(self.raw_features) >= ta_period

    def __repr__(self):
        return str(self.__dict__)


class StockTraderManager:
    @classmethod
    def update_multiple_from_barset(
            cls,
            stock_traders_by_symbol,  # type: Dict[AnyStr, StockTrader]
            barset,  # type: BarSet
    ):
        for symbol in barset:
            assert symbol in stock_traders_by_symbol
            stock_trader = stock_traders_by_symbol[symbol]
            stock_trader._state.update_from_bars(barset[symbol])

    @classmethod
    def create_multiple_from_barset(
            cls,
            model,  # type: TraderGRU
            barset,  # type: BarSet
            capital,  # type: float
    ):
        # type: (...) -> Dict[AnyStr, StockTrader]
        """Returns StockTrader dictionary keyed by symbol"""
        stock_traders = {}
        capital_per_symbol = capital / len(barset)
        for symbol in barset:
            stock_trader = StockTrader(
                model=model,
                symbol=symbol,
                capital=capital_per_symbol
            )
            stock_trader._state.update_from_bars(
                barset[symbol]
            )
            stock_traders[symbol] = stock_trader
            logging.info(
                dict(
                    msg=f'Finished Instantiating {symbol}',
                    type='instantiate_stock_trader_finish'
                )
            )
        return stock_traders


class StockTrader:
    def __init__(
            self,
            model,  # type: TraderGRU
            symbol,  # type: AnyStr
            capital=0,  # type: float
            shares_owned=0,  # type: int
    ):
        self.symbol = symbol
        self._state = StockState(
            capital=capital,
            shares_owned=shares_owned
        )
        self._model = model

    async def handle_bar(self, bar):
        logging.info(
            dict(
                type='handle_bar',
                state=self._state.__dict__
            )
        )
        self._state.update_from_realtime_bars([bar])

        if len(self._state.unnormalized_model_inputs) == 0:
            return

        trade = self._get_trade_from_model(self._state.unnormalized_model_inputs)

        if trade > 0:
            try:
                self._buy(trade)
            except Exception as exc:
                logging.error(
                    dict(
                        msg='Error when buying',
                        type='error_when_buying',
                        exc=exc
                    )
                )

        if trade < 0:
            try:
                self._sell(trade)
            except Exception as exc:
                logging.error(
                    dict(
                        msg='Error when selling',
                        exc=exc
                    )
                )

    async def handle_trade_updates(
            self,
            account  # type: Account
    ):
        if account.event == 'fill':
            filled_quantity = int(account.order['filled_qty'])
            filled_avg_price = float(account.order['filled_avg_price'])

            if account.order['side'] == 'buy':
                self._state.shares_owned += filled_quantity
                self._state.capital -= filled_quantity * filled_avg_price
                logging.info(
                    dict(
                        msg=f'Bought {account.order["symbol"]} @ {format_usd(filled_avg_price)}',
                        type='buy_filled',
                        symbol=account.order["symbol"],
                        price=filled_avg_price,
                        shares_owned=self._state.shares_owned,
                        capital=self._state.capital
                    )
                )
            elif account.order['side'] == 'sell':
                self._state.shares_owned -= filled_quantity
                self._state.capital += filled_quantity * filled_avg_price
                logging.info(
                    dict(
                        msg=f'Sold {account.order["symbol"]} @ {format_usd(filled_avg_price)}',
                        type='sell_filled',
                        symbol=account.order["symbol"],
                        price=filled_avg_price,
                        shares_owned=self._state.shares_owned,
                        capital=self._state.capital
                    )
                )

    def _get_trade_from_model(self, unnormalized_inputs):
        inputs_tensor = torch.FloatTensor([unnormalized_inputs])
        normalized_inputs_tensor = PCN.normalize_volume(inputs_tensor)
        normalized_inputs_tensor = PCN.normalize_price_into_percent_change(normalized_inputs_tensor)
        trades = self._model(normalized_inputs_tensor)
        trade = trades.detach().numpy()[0][-1]
        return trade

    def _buy(self, trade, slippage=.005):
        """
        Buy the stocks given the trade output from the model

        Side effects:
         updates:
            - self._state.shares_owned
            - self._state.capital
        """
        capital_to_use = self._state.capital * trade
        current_price = float(api.polygon.last_trade(self.symbol).price)
        price_with_slippage = current_price * (1 + slippage)
        shares_to_buy = math.floor(capital_to_use / current_price)
        if shares_to_buy > 0:
            # TODO: Temporary now that we know on_account_updates is unreliable update the shares owned here
            self._state.shares_owned += shares_to_buy
            self._state.capital -= shares_to_buy * current_price
            logging.info(
                dict(
                    msg=f'Buying {shares_to_buy} of {self.symbol} @ ~{format_usd(current_price)}',
                    trade=trade,
                    type='trade_buy',
                    capital=self._state.capital,
                    shares_owned=self._state.shares_owned,
                    capital_to_use=capital_to_use,
                    shares_to_buy=shares_to_buy,
                    price=current_price,
                    price_with_slippage=price_with_slippage
                )
            )
            api.submit_order(
                symbol=self.symbol,
                qty=shares_to_buy,
                side='buy',
                type='limit',
                time_in_force='gtc',
                limit_price=str(price_with_slippage)
            )
        else:
            logging.info(
                dict(
                    type='no_trade_buy',
                    msg=f'Buying 0 shares of {self.symbol}',
                    trade=trade,
                    capital=self._state.capital,
                    shares_owned=self._state.shares_owned,
                )
            )

    def _sell(self, trade, slippage=.005):
        """
        Sell the stocks shares_owned  given the trade output from the model

        Side effects:
         updates:
            - self._state.shares_owned
            - self._state.capital
        """
        current_price = float(api.polygon.last_trade(self.symbol).price)
        price_with_slippage = current_price * (1 - slippage)
        shares_to_sell = math.ceil(abs(trade) * self._state.shares_owned)

        if shares_to_sell > 0:
            # TODO: Temporary now that we know on_account_updates is unreliable update the shares owned here
            self._state.shares_owned -= shares_to_sell
            self._state.capital += shares_to_sell * current_price

            logging.info(
                dict(
                    msg=f'Selling {shares_to_sell} of {self.symbol} @ ~{format_usd(current_price)}',
                    type='trade_sell',
                    trade=trade,
                    capital=self._state.capital,
                    shares_owned=self._state.shares_owned,
                    shares_to_sell=shares_to_sell,
                    price=current_price,
                    price_with_slippage=price_with_slippage
                )
            )
            api.submit_order(
                symbol=self.symbol,
                qty=shares_to_sell,
                side='sell',
                type='limit',
                time_in_force='gtc',
                limit_price=str(price_with_slippage)
            )
        else:
            logging.info(
                dict(
                    msg=f'Selling 0 shares of {self.symbol}',
                    type='no_trade_sell',
                    trade=trade,
                    capital=self._state.capital,
                    shares_owned=self._state.shares_owned,
                )
            )
