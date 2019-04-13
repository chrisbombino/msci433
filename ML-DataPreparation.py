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
    s= 'SMA_DJI'
    DJIsma = load_data(s,directory)
    s= 'EMA_DJI'
    DJIema = load_data(s,directory)
    DJIBollinger=DJIBollinger.assign(SMA5=DJIsma['SMA 5'])
    DJIBollinger=DJIBollinger.assign(SMA10=DJIsma['SMA 10'])
    DJIBollinger=DJIBollinger.assign(SMA15=DJIsma['SMA 15'])
    DJIBollinger=DJIBollinger.assign(EMA5=DJIema['EMA 5'])
    DJIBollinger=DJIBollinger.assign(EMA10=DJIema['EMA 10'])
    DJIBollinger=DJIBollinger.assign(EMA15=DJIema['EMA 15'])

    DJIBollinger.to_csv('ML/DJIML.csv',index=False)


    s = 'NASDAQ-RSI'
    directory ='NASDAQ30'
    NASDAQRSI = load_data(s,directory)
    s = 'NASDAQBollinger'
    NASDAQBollinger = load_data(s,directory)
    del NASDAQBollinger['sellSignal']
    del NASDAQBollinger['buySignal']
    s= 'SMA_NASDAQ'
    naddaqsma = load_data(s,directory)
    s= 'EMA_NASDAQ'
    nasdaqema = load_data(s,directory)

    NASDAQBollinger=NASDAQBollinger.assign(RSI=NASDAQRSI['RSI'])
    NASDAQBollinger=NASDAQBollinger.assign(SMA5=naddaqsma['SMA 5'])
    NASDAQBollinger=NASDAQBollinger.assign(SMA10=naddaqsma['SMA 10'])
    NASDAQBollinger=NASDAQBollinger.assign(SMA15=naddaqsma['SMA 15'])
    NASDAQBollinger=NASDAQBollinger.assign(EMA5=nasdaqema['EMA 5'])
    NASDAQBollinger=NASDAQBollinger.assign(EMA10=nasdaqema['EMA 10'])
    NASDAQBollinger=NASDAQBollinger.assign(EMA15=nasdaqema['EMA 15'])
    NASDAQBollinger.to_csv('ML/NADSAQML.csv',index=False)


    s = 'AAPL-RSI'
    directory= 'AAPL'
    AAPLRSI = load_data(s,directory)
    s = 'AAPLBollinger'
    AAPLBollinger = load_data(s,directory)
    del AAPLBollinger['sellSignal']
    del AAPLBollinger['buySignal']

    s= 'SMA_AAPL'
    AAPLsma = load_data(s,directory)
    s= 'EMA_AAPL'
    AAPLema = load_data(s,directory)
    AAPLBollinger=AAPLBollinger.assign(RSI=AAPLRSI['RSI'])
    AAPLBollinger=AAPLBollinger.assign(SMA5=AAPLsma['SMA 5'])
    AAPLBollinger=AAPLBollinger.assign(SMA10=AAPLsma['SMA 10'])
    AAPLBollinger=AAPLBollinger.assign(SMA15=AAPLsma['SMA 15'])
    AAPLBollinger=AAPLBollinger.assign(EMA5=AAPLema['EMA 5'])
    AAPLBollinger=AAPLBollinger.assign(EMA10=AAPLema['EMA 10'])
    AAPLBollinger=AAPLBollinger.assign(EMA15=AAPLema['EMA 15'])
    AAPLBollinger.to_csv('ML/AAPLML.csv',index=False)
    

