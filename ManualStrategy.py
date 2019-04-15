'''
Code for MSCI 433 Assignment 3 Part 2, Winter 2019

This code reads a csv file that contains trading logic/strategy
The file must have historical stock data (date, price) as well as 3 columns
that are generated by the trading strategy. The 3 columns are BuySignal,
SellSignal and ExitSignal. These values are binary, and each period can only
have a maximum of 1 signal set to True.

Based on the trading strategy, this code simulates trades and details how the
strategy performs.

Please see README.md for more details.
'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# There are 4 actions regarding the purchase/sale of stock
# long buy, long sell, short buy, short sell
# the actions are all similar, but have slight differences

def longBuyAction(portfolio, price, shares, df_trades, date):
    # TODO: check if portfolio has enough cash
    # update amount of long stock, cash and portfolio value
    portfolio['long-stock'] += shares
    portfolio['cash'] -= price * shares
    portfolio['value'] = portfolio['cash'] + price * (portfolio['long-stock'] - portfolio['short-stock'])

    # add trade to df_trades
    trade = {'Date': date, 'Shares': shares, 'Position': 'Enter Long', 'Return': np.nan, 'Portfolio Value': portfolio['value']}
    df_trades = df_trades.append(trade, ignore_index=True)

    return portfolio, df_trades

def longSellAction(portfolio, price, shares, df_trades, date):
    # TODO: check if portfolio has enough cash
    # update amount of long stock, cash and portfolio value
    portfolio['long-stock'] -= shares
    portfolio['cash'] += price * shares
    portfolio['value'] = portfolio['cash'] + price * (portfolio['long-stock'] - portfolio['short-stock'])

    # calculate ROI and add trade to df_trades
    ret = (portfolio['value'] / df_trades.iloc[-1, 3]) - 1
    trade = {'Date': date, 'Shares': -shares, 'Position': 'Exit Long', 'Return': ret * 100, 'Portfolio Value': portfolio['value']}
    df_trades = df_trades.append(trade, ignore_index=True)

    return portfolio, df_trades

def shortBuyAction(portfolio, price, shares, df_trades, date):
    # TODO: check if portfolio has enough cash
    # update amount of short stock, cash and portfolio value
    portfolio['short-stock'] -= shares
    portfolio['cash'] -= price * shares
    portfolio['value'] = portfolio['cash'] + price * (portfolio['long-stock'] - portfolio['short-stock'])

    # calculate ROI and add trade to df_trades
    ret = (portfolio['value'] / df_trades.iloc[-1, 3]) - 1
    trade = {'Date': date, 'Shares': shares, 'Position': 'Exit Short', 'Return': ret * 100, 'Portfolio Value': portfolio['value']}
    df_trades = df_trades.append(trade, ignore_index=True)

    return portfolio, df_trades

def shortSellAction(portfolio, price, shares, df_trades, date):
    # TODO: check if portfolio should short
    # update amount of short stock, cash and portfolio value
    portfolio['short-stock'] += shares
    portfolio['cash'] += price * shares
    portfolio['value'] = portfolio['cash'] + price * (portfolio['long-stock'] - portfolio['short-stock'])

    # add trade to df_trades
    trade = {'Date': date, 'Shares': -shares, 'Position': 'Enter Short', 'Return': np.nan, 'Portfolio Value': portfolio['value']}
    df_trades = df_trades.append(trade, ignore_index=True)

    return portfolio, df_trades

def simulateTrading(df):
    portfolio = {
        'cash': startingCash,
        'value': startingCash,
        'long-stock': 0,
        'short-stock': 0
    }

    # initialize df_trades
    df_trades = pd.DataFrame(columns=['Shares', 'Position', 'Return', 'Portfolio Value'])

    state = '' # start out not holding any position

    for i, row in df.iterrows():
        if state == '':
            if not pd.isna(row.BuySignal):
                # enter long position (buy stock)
                portfolio, df_trades = longBuyAction(portfolio, row.Price, numShares, df_trades, row.Date)
                state = 'long'
                continue
            if not pd.isna(row.SellSignal):
                # enter short position (short stock)
                portfolio, df_trades = shortSellAction(portfolio, row.Price, numShares, df_trades, row.Date)
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
                portfolio, df_trades = longSellAction(portfolio, row.Price, numShares, df_trades, row.Date)
                state = ''
                continue
            if not pd.isna(row.ExitSignal):
                # exit long position (sell the stock)
                portfolio, df_trades = longSellAction(portfolio, row.Price, numShares, df_trades, row.Date)
                state = ''
                continue

        # in a short position
        elif state == 'short':
            if not pd.isna(row.BuySignal):
                # exit short position (buy stock)
                portfolio, df_trades = shortBuyAction(portfolio, row.Price, numShares, df_trades, row.Date)
                state = ''
                continue
            if not pd.isna(row.SellSignal):
                # not possible
                continue
            if not pd.isna(row.ExitSignal):
                # exit short position (buy stock)
                portfolio, df_trades = shortBuyAction(portfolio, row.Price, numShares, df_trades, row.Date)
                state = ''
                continue

    print '\nTrade history:\n'
    print df_trades
    print '\nROI = {}%'.format(((portfolio['cash']/startingCash - 1))*100)

    y_pos = np.arange(len(df.loc[:, 'Date']))


    heights = []
    colors = []

    counter = 0
    for d1 in df.loc[:, 'Date']:
        if counter < df_trades.shape[0]:
            d2 = df_trades.loc[counter, 'Date']
            position = df_trades.loc[counter, 'Position']
            if d1 == d2:
                if position == 'Enter Long':
                    heights.append(1)
                    color = 'green'
                    colors.append(color)
                elif position == 'Enter Short':
                    heights.append(1)
                    color = 'red'
                    colors.append(color)
                else:
                    heights.append(0)
                    colors.append('black')
                counter += 1
            else:
                heights.append(0)
                colors.append('black')
        else:
            heights.append(0)
            colors.append('black')

    plt.bar(y_pos, heights, color=colors)
    plt.xticks(y_pos, df.loc[:, 'Date'].tolist())
    plt.show()

if __name__ == '__main__':
    data = pd.read_csv('test.csv')
    numShares = 10 # shares bought in batches of 10
    startingCash = 1000

    simulateTrading(data)
