

def ProfitReward(trade_sequence, market_sequence):
    time_length = trade_sequence.shape[0]
    capital = 10000
    shares = 0
    for t in range(time_length):
        current_trade = trade_sequence[t]
        current_price = market_sequence[t]
        if current_trade > 0: # buying
            shares += current_trade * capital / current_price
            capital -= current_trade * capital
        else: # selling
            capital -= shares * current_price * current_trade
            shares += shares * current_trade
        # print("Current Price: ", current_price)
        # print("Current Trade: ", current_trade)
        # print("Current Capital: ", capital)
        # print("Current Shares: ", shares)


    final_price = market_sequence[-1]
    if shares > 0:
        capital += shares * final_price
    
    return capital - 10000
