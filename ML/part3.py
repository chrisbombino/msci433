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

    ML_DF = pd.DataFrame(columns = ['Indicator' , 'AccuracyScore'])
    Indicator = []
    AccuracyScore = []
    for symbol in symbols : 
            # create X and Y values
            X = df.loc[14:, ['RSI', 'SMA5']]
            Y = df.loc[14:, 'ROI_binary']
    
            # split the data
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
            # fit the training data
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, Y_train)
            # predict the test values
            Y_pred = knn.predict(X_test)
            print('The accuracy score for ' + symbol + ' RSI & SMA5 is ')
            print metrics.accuracy_score(Y_test, Y_pred)
            RSI_SMA5 = metrics.accuracy_score(Y_test, Y_pred)
            AccuracyScore.append(RSI_SMA5)
            Indicator.append('RSI & SMA5')
            
            
            
#--------------------------------------------------------------------------------------------------------
        
            # create X and Y values
            X = df.loc[14:, ['RSI', 'SMA10']]
            Y = df.loc[14:, 'ROI_binary']
    
            # split the data
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
            # fit the training data
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, Y_train)
    
            # predict the test values
            Y_pred = knn.predict(X_test)
            print('The accuracy score for ' + symbol + ' RSI & SMA10 is ')
            print metrics.accuracy_score(Y_test, Y_pred)
            RSI_SMA10 = metrics.accuracy_score(Y_test, Y_pred)
            AccuracyScore.append(RSI_SMA10)
            Indicator.append('RSI & SMA10')
    
#--------------------------------------------------------------------------------------------------------    
    
            # create X and Y values
            X = df.loc[14:, ['RSI', 'SMA15']]
            Y = df.loc[14:, 'ROI_binary']
    
            # split the data
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
            # fit the training data
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, Y_train)
    
            # predict the test values
            Y_pred = knn.predict(X_test)
            print('The accuracy score for ' + symbol + ' RSI & SMA15 is ')
            print metrics.accuracy_score(Y_test, Y_pred)
            RSI_SMA15 = metrics.accuracy_score(Y_test, Y_pred)
            AccuracyScore.append(RSI_SMA15)
            Indicator.append('RSI & SMA15')            
    
#--------------------------------------------------------------------------------------------------------    
               
            df['SMA5/Close'] = df['SMA5'] / df['Close']    
            # create X and Y values
            X = df.loc[14:, ['RSI', 'SMA5/Close']]
            Y = df.loc[14:, 'ROI_binary']
    
            # split the data
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
            # fit the training data
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, Y_train)
    
            # predict the test values
            Y_pred = knn.predict(X_test)
            print('The accuracy score for ' + symbol + ' RSI & SMA5/Close(t) is ')
            print metrics.accuracy_score(Y_test, Y_pred)
            RSI_SMA5_Close = metrics.accuracy_score(Y_test, Y_pred)
            AccuracyScore.append(RSI_SMA5_Close)
            Indicator.append('RSI & SMA5/Close(t)')
        
#--------------------------------------------------------------------------------------------------------        
        
            df['SMA10/Close'] = df['SMA10'] / df['Close']    
            # create X and Y values
            X = df.loc[14:, ['RSI', 'SMA10/Close']]
            Y = df.loc[14:, 'ROI_binary']
    
            # split the data
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
            # fit the training data
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, Y_train)
    
            # predict the test values
            Y_pred = knn.predict(X_test)
            print('The accuracy score for ' + symbol + ' RSI & SMA10/Close(t) is ')
            print metrics.accuracy_score(Y_test, Y_pred)
            RSI_SMA10_Close = metrics.accuracy_score(Y_test, Y_pred)        
            AccuracyScore.append(RSI_SMA10_Close)
            Indicator.append('RSI & SMA10/Close(t)')
            
#--------------------------------------------------------------------------------------------------------        
    
            df['SMA15/Close'] = df['SMA15'] / df['Close']    
            # create X and Y values
            X = df.loc[14:, ['RSI', 'SMA15/Close']]
            Y = df.loc[14:, 'ROI_binary']
    
            # split the data
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
            # fit the training data
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, Y_train)
    
            # predict the test values
            Y_pred = knn.predict(X_test)
            print('The accuracy score for ' + symbol + ' RSI & SMA15/Close(t) is ')
            print metrics.accuracy_score(Y_test, Y_pred)
            RSI_SMA15_Close = metrics.accuracy_score(Y_test, Y_pred)
            AccuracyScore.append(RSI_SMA15_Close)
            Indicator.append('RSI & SMA15/Close(t)')
        
#--------------------------------------------------------------------------------------------------------
            
            df['SMA10/SMA5'] = df['SMA10'] / df['SMA5']    
            # create X and Y values
            X = df.loc[14:, ['RSI', 'SMA10/SMA5']]
            Y = df.loc[14:, 'ROI_binary']
    
            # split the data
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
            # fit the training data
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, Y_train)
    
            # predict the test values
            Y_pred = knn.predict(X_test)
            print('The accuracy score for ' + symbol + ' RSI & SMA10/SMA5 is ')
            print metrics.accuracy_score(Y_test, Y_pred)
            RSI_SMA10_SMA5 = metrics.accuracy_score(Y_test, Y_pred)  
            AccuracyScore.append(RSI_SMA10_SMA5)
            Indicator.append('RSI & SMA10/SMA5')
        
#--------------------------------------------------------------------------------------------------------
            
            df['SMA15/SMA10'] = df['SMA15'] / df['SMA10']    
            # create X and Y values
            X = df.loc[14:, ['RSI', 'SMA15/SMA10']]
            Y = df.loc[14:, 'ROI_binary']
    
            # split the data
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
            # fit the training data
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, Y_train)
    
            # predict the test values
            Y_pred = knn.predict(X_test)
            print('The accuracy score for ' + symbol + ' RSI & SMA15/SMA10 is ')
            print metrics.accuracy_score(Y_test, Y_pred)
            RSI_SMA15_SMA10 = metrics.accuracy_score(Y_test, Y_pred)         
            AccuracyScore.append(RSI_SMA15_SMA10)
            Indicator.append('RSI & SMA15/SMA10')
            
#--------------------------------------------------------------------------------------------------------
            
            df['RSI/Close'] = df['RSI'] / df['Close']    
            # create X and Y values
            X = df.loc[14:, ['RSI/Close', 'SMA5']]
            Y = df.loc[14:, 'ROI_binary']
    
            # split the data
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
            # fit the training data
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, Y_train)
    
            # predict the test values
            Y_pred = knn.predict(X_test)
            print('The accuracy score for ' + symbol + ' RSI/Close & SMA5 is ')
            print metrics.accuracy_score(Y_test, Y_pred)
            RSI_Close_SMA5 = metrics.accuracy_score(Y_test, Y_pred)
            AccuracyScore.append(RSI_Close_SMA5)
            Indicator.append('RSI/Close & SMA5')            

#--------------------------------------------------------------------------------------------------------

            df['RSI/Close'] = df['RSI'] / df['Close']    
            # create X and Y values
            X = df.loc[14:, ['RSI/Close', 'SMA10']]
            Y = df.loc[14:, 'ROI_binary']
    
            # split the data
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
            # fit the training data
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, Y_train)
    
            # predict the test values
            Y_pred = knn.predict(X_test)
            print('The accuracy score for ' + symbol + ' RSI/Close & SMA10 is ')
            print metrics.accuracy_score(Y_test, Y_pred)
            RSI_Close_SMA10 = metrics.accuracy_score(Y_test, Y_pred)                
            AccuracyScore.append(RSI_Close_SMA10)
            Indicator.append('RSI/Close & SMA10')        
#--------------------------------------------------------------------------------------------------------
        
            ML_DF['AccuracyScore'] = AccuracyScore
            ML_DF['Indicator'] = Indicator
            
            print(ML_DF)
        
        

main(test_symbols)
