import pandas as pd

def load_data(s,directory):
    data = pd.read_csv("{}/{}.csv".format(directory, s))
    return data

if __name__== "__main__":
    #-----------Data Input-------------------
    s = 'DJI-RSI' 
    directory ='DJI30'
    DJIRSI = load_data(s,directory)
    s = 'DJIBollinger' 
    DJIBollinger = load_data(s,directory)
    
    DJIBollinger=DJIBollinger.assign(RSI=DJIRSI['RSI'])
    del DJIBollinger['sellSignal']
    del DJIBollinger['buySignal']
    
    DJIBollinger.to_csv('ML/DJIML.csv',index=False)


    s = 'NASDAQ-RSI'
    directory ='NASDAQ30'
    NASDAQRSI = load_data(s,directory)
    s = 'NASDAQBollinger'
    NASDAQBollinger = load_data(s,directory)
    del NASDAQBollinger['sellSignal']
    del NASDAQBollinger['buySignal']
    NASDAQBollinger=NASDAQBollinger.assign(RSI=NASDAQRSI['RSI'])
    NASDAQBollinger.to_csv('ML/NADSAQML.csv',index=False)


    s = 'AAPL-RSI'
    directory= 'AAPL'
    AAPLRSI = load_data(s,directory)
    s = 'AAPLBollinger'
    AAPLBollinger = load_data(s,directory)
    del AAPLBollinger['sellSignal']
    del AAPLBollinger['buySignal']
    AAPLBollinger=AAPLBollinger.assign(RSI=AAPLRSI['RSI'])
    AAPLBollinger.to_csv('ML/AAPLML.csv',index=False)
    

