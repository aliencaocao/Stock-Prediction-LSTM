import datetime
import yfinance as yf
import pandas as pd
import os
import sys
import ta
from ta.utils import dropna

print(f'Running on Python {sys.version}.')
print('Libraries successfully imported. Initializing variables...')

Stock = 'TSLA'  # Stock to predict
TODAY = pd.to_datetime('today').to_pydatetime()
offset = max(1, (TODAY.weekday() + 6) % 7 - 3)
timedelta = datetime.timedelta(offset)
LAST_WORKING_DAY = TODAY - timedelta
interval = '15m'
period = '60d'

initial_capital = 2500
indicator = 'All'  # 'RSI','ROC','CCI','All'
fast = 14
slow = 28
filterByVol = True
k_value = 50
atrStopLoss = 1
buy, sell, hold = int(1), int(-1), int(0)
print('All variables initialized.')

# 3 pairs of predictor indicators, long and short each
# rs = rsi(close, slow), rf = rsi(close, fast)
# cs = cci(close, slow), cf = cci(close, fast)
# os = roc(close, slow), of = roc(close, fast)

# f1 = ind=='RSI' ? rs : ind=='ROC' ? os : ind=='CCI' ? cs : avg(rs, os, cs)
# f2 = ind=='RSI' ? rf : ind=='ROC' ? of : ind=='CCI' ? cf : avg(rf, of, cf)


print(f'Stock to predict: {Stock}. The last working day is {LAST_WORKING_DAY}.)')
# Read data
data = yf.download(tickers=f'{Stock}', period=period, interval=interval)
last_close_price = float("{:.2f}".format(data.iloc[-1]["Close"]))
sec_last_close_price = float("{:.2f}".format(data.iloc[-2]["Close"]))
print(
    f'{Stock} Stock data at interval of {interval} from past {period} received from Yahoo Finance. Latest closing price is ${last_close_price}.')

# Separating Data into training and validating.
print(f'Gathered {len(data)} timestamps of data. Processing...')

data = ta.add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume")
print()