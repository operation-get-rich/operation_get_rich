def ProfitReward(trade_sequence, market_sequence):
    time_length = trade_sequence.shape[0]
    capital = 1
    shares = 0
    for t in range(time_length):
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

        # print("Current Price: ", current_price)
        # print("Current Trade: ", current_trade)
        # print("Current Capital: ", capital)
        # print("Current Shares: ", shares)

    final_price = market_sequence[-1]
    capital = capital + (shares * final_price)

    return capital - 1
