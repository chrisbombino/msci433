import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

symbols = ['AAPL', 'DJI', 'NASDAQ']
test_symbols = ['AAPL']

indicators = ['Lower Bound', 'Upper Bound', 'Middle Bound', 'RSI', 'SMA5',
            'SMA10', 'SMA15', 'EMA5', 'EMA10', 'EMA15']

def main(symbols):
    for s in symbols:
        df = pd.read_csv('{}ML.csv'.format(s))

        df['ROI_binary'] = [1 if roi > 0 else 0 for roi in df.loc[:, 'ROI']]

        #print df.head()

        X = df.loc[:, ['SMA10', 'SMA15']]
        Y = df.loc[:, 'ROI_binary']

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        

main(test_symbols)
