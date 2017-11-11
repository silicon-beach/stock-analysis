import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import csv

def downloadHistory_stocks(symbolStock):
    ts = TimeSeries(key='055UMQXJRDY71RG3', output_format='pandas')
    data, meta_data = ts.get_intraday(symbol=symbolStock,interval='1min', outputsize='full')
    pd.set_option('display.max_rows',5000)
    dataCovert=str(pd.DataFrame(data))
    f=open('output.txt',"wr")
    f.write(dataCovert)
    f.close()
    DataTemp=["Date,time,volume,close,high,open,low\n"]
    Data1=[]
    f1=open('output.txt')
    line=f1.readline()
    line=f1.readline()
    while 1:
        line=f1.readline()
        if not line:
            break
        else:
            Data1.append(line)
    f1.close()
    for line in Data1:
        DataTemp.append(",".join(line.split())+"\n")
    f2=open(symbolStock+".csv","w")
    f2.writelines(DataTemp)
    f2.close()
if __name__ == '__main__':
    apple_data = downloadHistory_stocks('AAPL')
    mbi_data=downloadHistory_stocks('MBI')
    google_data=downloadHistory_stocks('GOOGL')


    
