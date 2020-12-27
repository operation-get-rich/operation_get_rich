import asyncio
import json
import os
import pickle
from datetime import datetime
from typing import AnyStr

import pandas as pd
import pytest
from numpy import mean
from ta.volume import VolumeWeightedAveragePrice

from directories import DATA_DIR
from experiment_trader_gru.experiment_trader_gru_directories import RUNS_DIR
from experiment_trader_gru.stock_trader import StockTrader
from experiment_trader_gru.TraderGRU import load_trader_gru_model


class RealTimeBarFake:
    def __init__(
            self,
            symbol,  # type: AnyStr
            volume,  # type: float
            total_volume,  # type: float
            vwap,  # type: float
            open,  # type: float
            close,  # type: float
            high,  # type: float
            low,  # type: float
            average,  # type: float
            # TODO: Confirm whether the type for the time are correct
            start,  # type: datetime
            end,  # type: datetime
            timestamp,  # type: float
    ):
        self.symbol = symbol
        self.volume = volume
        self.total_volume = total_volume
        self.vwap = vwap
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.average = average
        self.start = start
        self.end = end
        self.timestamp = timestamp


class TradeFake:
    def __init__(
            self,
            price
    ):
        self.price = price


@pytest.mark.asyncio
async def test_stock_trader_handle_bar_update_states_correctly_when_buying_or_selling(mocker):
    mocker.patch(
        'experiment_trader_gru.entities.api'
    )

    model = load_trader_gru_model(
        model_location=f'{RUNS_DIR}/Trader_Load_NextTrade_PreMarket_Multiply-20201121-222444/best_model.pt'
    )

    stock_trader = StockTrader(
        model=model,
        symbol='CBAT',
        capital=25_000,
    )

    cbat_df = pd.read_csv(
        os.path.join(DATA_DIR, 'test_stock_prices', 'CBAT_2020-11-13.csv'),
        parse_dates=['time']
    )
    cbat_df['vwap'] = VolumeWeightedAveragePrice(
        high=cbat_df.high,
        low=cbat_df.low,
        close=cbat_df.close,
        volume=cbat_df.volume,
    ).vwap

    total_volume = 0
    for _, bar in cbat_df.iterrows():
        mocker.patch(
            'experiment_trader_gru.entities.api.polygon.last_trade',
            return_value=TradeFake(price=bar.open)
        )
        total_volume += cbat_df.volume
        await stock_trader.handle_bar(
            RealTimeBarFake(
                symbol=bar.ticker,
                volume=bar.volume,
                total_volume=total_volume,
                vwap=bar.vwap,
                open=bar.open,
                close=bar.close,
                high=bar.high,
                low=bar.low,
                average=mean([bar.close, bar.low, bar.high]),
                start=bar.time,
                end=bar.time,
                timestamp=bar.time.timestamp()
            )
        )

    with open('./expected_stock_trader_state.pickle', 'rb') as f:
        expected_state = pickle.load(f)

    assert expected_state.capital == stock_trader._state.capital
    assert expected_state.close_prices == stock_trader._state.close_prices
    assert expected_state.price_volume_products == stock_trader._state.price_volume_products
    assert expected_state.raw_features == stock_trader._state.raw_features
    assert expected_state.unnormalized_model_inputs == stock_trader._state.unnormalized_model_inputs
    assert expected_state.volumes == stock_trader._state.volumes
    assert expected_state.__dict__ == stock_trader._state.__dict__
