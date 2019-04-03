import pandas as pd

data = pd.read_csv('test.csv')

numShares = 10 # shares bought in batches of 10
startingCash = 100000

def longBuyAction(portfolio, price, shares):
    # TODO: check if portfolio has enough cash
    portfolio['long-stock'] += shares
    portfolio['cash'] -= price * shares
    return portfolio

def longSellAction(portfolio, price, shares):
    # TODO: check if portfolio has enough cash
    portfolio['long-stock'] -= shares
    portfolio['cash'] += price * shares
    return portfolio

def shortBuyAction(portfolio, price, shares):
    # TODO: check if portfolio has enough cash
    portfolio['short-stock'] -= shares
    portfolio['cash'] -= price * shares
    return portfolio

def shortSellAction(portfolio, price, shares):
    # TODO: check if portfolio should short
    portfolio['short-stock'] += shares
    portfolio['cash'] += price * shares
    return portfolio

def simulateTrading(df):
    portfolio = {
        'cash': startingCash,
        'long-stock': 0,
        'short-stock': 0
    }

    # start out not holding any position
    state = ''

    for i, row in df.iterrows():
        if state == '':
            if not pd.isna(row.BuySignal):
                # enter long position (buy stock)
                portfolio = longBuyAction(portfolio, row.Price, numShares)
                state = 'long'
                print portfolio
                continue
            if not pd.isna(row.SellSignal):
                # enter short position (short stock)
                portfolio = shortSellAction(portfolio, row.Price, numShares)
                state = 'short'
                print portfolio
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
                portfolio = longSellAction(portfolio, row.Price, numShares)
                state = ''
                print portfolio
                continue
            if not pd.isna(row.ExitSignal):
                # exit long position (sell the stock)
                portfolio = longSellAction(portfolio, row.Price, numShares)
                state = ''
                print portfolio
                continue

        # in a short position
        elif state == 'short':
            if not pd.isna(row.BuySignal):
                # exit short position (buy stock)
                portfolio = shortBuyAction(portfolio, row.Price, numShares)
                print portfolio
                state = ''
                continue
            if not pd.isna(row.SellSignal):
                # not possible
                continue
            if not pd.isna(row.ExitSignal):
                # exit short position (buy stock)
                portfolio = shortBuyAction(portfolio, row.Price, numShares)
                print portfolio
                state = ''
                continue
    print '\nFinal Portfolio:\n{}'.format(portfolio)
    print '\nROI = {}%'.format(((portfolio['cash']/startingCash - 1))*100)

simulateTrading(data)
