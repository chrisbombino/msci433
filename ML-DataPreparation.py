import pandas as pd

def load_data(s,directory):
    data = pd.read_csv("{}/{}.csv".format(directory, s))
    return data

if __name__== "__main__":
    #-----------Data Input-------------------
    s = 'DJI-RSI' 
    directory ='DJI30'
    DJIRSI = load_data(s,directory)
    
    s = 'NASDAQ-RSI'
    directory ='NASDAQ30'
    NASDAQRSI = load_data(s,directory)
    s = 'NASDAQBollinger'
    NASDAQBollinger = load_data(s,directory)
    s = 'NASDAQPriceChannel'
    NASDAQPriceChannel = load_data(s,directory)
    




    s = 'AAPL-RSI'
    directory= 'AAPL'
    AAPLRSI = load_data(s,directory)
    

