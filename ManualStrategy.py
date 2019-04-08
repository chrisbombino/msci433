import pandas as pd
import numpy as np

data = pd.read_csv('test.csv')

numShares = 10 # shares bought in batches of 10
startingCash = 100000

# df_trades = df_trades.append({'Shares': 10, 'Position': 'Enter Long', 'Return': 0.01, 'Portfolio Value': 100100}, ignore_index=True)

def longBuyAction(portfolio, price, shares, df_trades):
    # TODO: check if portfolio has enough cash
    portfolio['long-stock'] += shares
    portfolio['cash'] -= price * shares
    portfolio['value'] = portfolio['cash'] + price * (portfolio['long-stock'] - portfolio['short-stock'])

    trade = {'Shares': shares, 'Position': 'Enter Long', 'Return': np.nan, 'Portfolio Value': portfolio['value']}
    df_trades = df_trades.append(trade, ignore_index=True)
    return portfolio, df_trades

def longSellAction(portfolio, price, shares, df_trades):
    # TODO: check if portfolio has enough cash
    portfolio['long-stock'] -= shares
    portfolio['cash'] += price * shares
    portfolio['value'] = portfolio['cash'] + price * (portfolio['long-stock'] - portfolio['short-stock'])

    ret = (portfolio['value'] / df_trades.iloc[-1, 3]) - 1
    trade = {'Shares': -shares, 'Position': 'Exit Long', 'Return': ret, 'Portfolio Value': portfolio['value']}
    df_trades = df_trades.append(trade, ignore_index=True)
    return portfolio, df_trades

def shortBuyAction(portfolio, price, shares, df_trades):
    # TODO: check if portfolio has enough cash
    portfolio['short-stock'] -= shares
    portfolio['cash'] -= price * shares
    portfolio['value'] = portfolio['cash'] + price * (portfolio['long-stock'] - portfolio['short-stock'])

    ret = (portfolio['value'] / df_trades.iloc[-1, 3]) - 1
    trade = {'Shares': shares, 'Position': 'Exit Short', 'Return': ret, 'Portfolio Value': portfolio['value']}
    df_trades = df_trades.append(trade, ignore_index=True)
    return portfolio, df_trades

def shortSellAction(portfolio, price, shares, df_trades):
    # TODO: check if portfolio should short
    portfolio['short-stock'] += shares
    portfolio['cash'] += price * shares
    portfolio['value'] = portfolio['cash'] + price * (portfolio['long-stock'] - portfolio['short-stock'])

    trade = {'Shares': -shares, 'Position': 'Enter Short', 'Return': np.nan, 'Portfolio Value': portfolio['value']}
    df_trades = df_trades.append(trade, ignore_index=True)
    return portfolio, df_trades

def simulateTrading(df):
    portfolio = {
        'cash': startingCash,
        'value': startingCash,
        'long-stock': 0,
        'short-stock': 0
    }

    df_trades = pd.DataFrame(columns=['Shares', 'Position', 'Return', 'Portfolio Value'])
    # start out not holding any position
    state = ''

    for i, row in df.iterrows():
        if state == '':
            if not pd.isna(row.BuySignal):
                # enter long position (buy stock)
                portfolio, df_trades = longBuyAction(portfolio, row.Price, numShares, df_trades)
                state = 'long'
                continue
            if not pd.isna(row.SellSignal):
                # enter short position (short stock)
                portfolio, df_trades = shortSellAction(portfolio, row.Price, numShares, df_trades)
                state = 'short'
                continue
            if not pd.isna(row.ExitSignal):
                # not possible
                continue

        # in a long position
        elif state == 'long':
            if not pd.isna(row.BuySignal):
                # not possible
                continue
            if not pd.isna(row.SellSignal):
                #  exit long position (sell the stock)
                portfolio, df_trades = longSellAction(portfolio, row.Price, numShares, df_trades)
                state = ''
                continue
            if not pd.isna(row.ExitSignal):
                # exit long position (sell the stock)
                portfolio, df_trades = longSellAction(portfolio, row.Price, numShares, df_trades)
                state = ''
                continue

        # in a short position
        elif state == 'short':
            if not pd.isna(row.BuySignal):
                # exit short position (buy stock)
                portfolio, df_trades = shortBuyAction(portfolio, row.Price, numShares, df_trades)
                state = ''
                continue
            if not pd.isna(row.SellSignal):
                # not possible
                continue
            if not pd.isna(row.ExitSignal):
                # exit short position (buy stock)
                portfolio, df_trades = shortBuyAction(portfolio, row.Price, numShares, df_trades)
                state = ''
                continue

    print 'ROI = {}%'.format(((portfolio['cash']/startingCash - 1))*100)
    print ''
    print df_trades

simulateTrading(data)
