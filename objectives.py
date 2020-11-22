def ProfitLoss(trade_sequence, market_sequence, is_premarket=None, next_trade=False):
    if next_trade:
        trade_sequence = trade_sequence[:-1]
        market_sequence = market_sequence[1:]

    assert trade_sequence.shape[0] == market_sequence.shape[0]

    time_length = trade_sequence.shape[0]
    capital = 1
    shares = 0
    for t in range(time_length):
        if is_premarket != None:
            if is_premarket[t]:
                continue

        current_trade = trade_sequence[t]
        current_price = market_sequence[t]
        if current_trade > 0:  # buying
            bought_shares = (current_trade * capital) / current_price
            shares = shares + bought_shares
            capital = capital - (bought_shares * current_price)
        elif current_trade < 0:  # selling
            sold_shares = shares * current_trade  # sold_shares < 0
            shares = shares + sold_shares
            capital = capital - (sold_shares * current_price)

    final_price = market_sequence[-1]
    capital = capital + (shares * final_price)

    # Return negative reward
    return -(capital - 1)
