import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import csv
from datetime import datetime


# Get last 7 days worth of data
def downloadHistory_stocks(symbol, interval='1min'):
    ts = TimeSeries(key='055UMQXJRDY71RG3', output_format='pandas')
    data, meta_data = ts.get_intraday(
        symbol=symbol, interval=interval, outputsize='full')
    pd.set_option('display.max_rows', 5000)
    dataCovert = str(pd.DataFrame(data))
    f = open('data/output.txt', "w")
    f.write(dataCovert)
    f.close()
    DataTemp = ["timestamp,open,high,low,close,volume,vwap\n"]
    Data1 = []
    f1 = open('data/output.txt')
    line = f1.readline()
    line = f1.readline()
    while 1:
        line = f1.readline()
        if not line:
            break
        else:
            Data1.append(line.split())
    f1.close()
    cumulative_total = 0
    cumulative_volume = 0
    for line in Data1:
        # 2017-10-30,09:30:00
        date = line.pop(0)
        date += ' ' + line.pop(0)
        typical_price = (float(line[0]) + float(line[1]) + float(line[2])) / 3
        cumulative_total += (typical_price * float(line[3]))
        cumulative_volume += float(line[3])
        DataTemp.append(
            ",".join([date] + line + [str(cumulative_total / cumulative_volume) ])+ "\n")
    f2 = open("data/" + symbol + ".csv", "w")
    f2.writelines(DataTemp)
    f2.close()


# get list of symbols
def get_symbols():
    # write a function to save all stock symbols to a csv
    return []


if __name__ == '__main__':
    apple_data = downloadHistory_stocks('SLV')
    #mbi_data = downloadHistory_stocks('MBI')
    #google_data = downloadHistory_stocks('GOOGL')
