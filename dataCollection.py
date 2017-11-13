from alpha_vantage.timeseries import TimeSeries
from datetime import datetime
import csv
import pandas as pd
import requests
import os
import glob

SYMBOL_URL = "http://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange={}&render=download"
STOCK_EXCHANGES = ["nasdaq", "nyse"]


# Get last 7 days worth of data
def downloadHistory_stocks(symbol, interval='1min'):
    try:
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
            typical_price = (float(line[0]) +
                             float(line[1]) + float(line[2])) / 3
            cumulative_total += (typical_price * float(line[3]))
            cumulative_volume += float(line[3])
            DataTemp.append(
                ",".join([date] + line + [str(cumulative_total / cumulative_volume)]) + "\n")
        write_csv(file_name="data/" + symbol + ".csv", data=DataTemp)
    except ValueError:
        pass


# get list of symbols automatically
def get_symbols(directory_name):
    for se in STOCK_EXCHANGES:
        with requests.Session() as s:
            download = s.get(SYMBOL_URL.format(se))
            decoded_content = download.content.decode('utf-8')
            cr = csv.reader(decoded_content.splitlines(), delimiter=',')
            data_list = []
            for d in list(cr):
                # print(d)
                data_list.append(';'.join(d[:8]) + '\n')
            write_csv(os.path.join(directory_name, se + ".csv"), data_list)


# Get data for all stocks below some price
def get_data():
    get_symbols("data/symbols/")
    for filename in glob.glob(os.path.join("data/symbols/", '*.csv')):
        df = read_csv(file_name=filename, names=[
                      "Symbol", "Name", "LastSale", "MarketCap", "IPOyear", "Sector", "industry", "Summary Quote"], sep=";")
        for chunk in df:
            symbols = chunk["Symbol"].values.tolist()
            for s in symbols:
                print("Downloading data for ", s)
                downloadHistory_stocks(s)

    return


def read_csv(file_name, names=["timestamp", "open", "high", "low", "close", "volume", "vwap"], sep=',', chunksize=29):
    df = pd.read_csv(file_name, names=names, sep=sep,
                     header=0, chunksize=chunksize)
    return df


def write_csv(file_name="result.csv", data=[]):
    file = open(file_name, "w")
    file.writelines(data)
    file.close()


if __name__ == '__main__':
    apple_data = downloadHistory_stocks('SLV')
    #mbi_data = downloadHistory_stocks('MBI')
    #google_data = downloadHistory_stocks('GOOGL')
