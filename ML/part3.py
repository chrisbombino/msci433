import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# used the following article as a starting point for this script
# https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75

symbols = ['AAPL', 'DJI', 'NASDAQ']
test_symbols = ['AAPL']

indicators = ['Lower Bound', 'Upper Bound', 'Middle Bound', 'RSI', 'SMA5',
            'SMA10', 'SMA15', 'EMA5', 'EMA10', 'EMA15']

def main(symbols, k=5):
    for s in symbols:
        # read csv
        df = pd.read_csv('{}ML.csv'.format(s))

        # create new column based on ROI at period t
        # if ROI > 0, set value to 1; else set value to 0
        df['ROI_binary'] = [1 if roi > 0 else 0 for roi in df.loc[:, 'ROI']]

        # create X and Y values
        X = df.loc[14:, ['SMA10', 'SMA15']]
        Y = df.loc[14:, 'ROI_binary']

        # split the data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        # fit the training data
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, Y_train)

        # predict the test values
        Y_pred = knn.predict(X_test)
        print metrics.accuracy_score(Y_test, Y_pred)

main(test_symbols)
