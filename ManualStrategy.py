import pandas as pd

data = pd.read_csv('test.csv')

#print data.head()

def simulateTrading(df):
    state = ''

    for i, row in df.iterrows():
        # start out not holding any position
        if state == '':
            if row.BuySignal:
                pass
                # enter long position (buy stock)
            if row.SellSignal:
                pass
                # enter short position (short stock)
            if row.ExitSignal:
                pass
                # not possible

        # in a long position
        elif state == 'long':
            if row.BuySignal:
                pass
                # not possible
            if row.SellSignal:
                pass
                #  exit long position (sell the stock)
            if row.ExitSignal:
                pass
                # exit long position (sell the stock)

        # in a short position
        elif state == 'short':
            if row.BuySignal:
                pass
                # exit short position (buy stock)
            if row.SellSignal:
                pass
                # not possible
            if row.ExitSignal:
                pass
                # exit short position (buy stock)

simulateTrading(data)
