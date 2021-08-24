# @author wangjinzhao on 2021/8/23
import glob

import pandas as pd

alldata = pd.DataFrame()
for file in glob.glob("data/spot/monthly/klines/BTCUSDT/1d/2021-01-01_2021-07-01/*.zip"):
    temp = pd.read_csv(file, names=["Open time", "Open", "High", "Low", "Close", "Volume", "Close time", "Quote asset volume", "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"], compression='zip')
    temp = temp.drop(['Close time',
                      'Quote asset volume',
                      'Number of trades',
                      'Taker buy base asset volume',
                      'Taker buy quote asset volume',
                      'Ignore'], axis=1)
    alldata = alldata.append(temp, ignore_index=True)
alldata.to_csv("data/btcusdt.csv")
print(alldata.head())
