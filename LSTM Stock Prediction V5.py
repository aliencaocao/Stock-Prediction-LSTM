print('LSTM (Long Short-Term Memory) Stock Prediction Version 2.9 - FULL')
print('Importing libraries...')
import tensorflow as tf
from tensorflow.keras import Sequential, initializers
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime
import sys
import os
policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
mixed_precision.set_policy(policy)
print(f'Running on Tensor Flow {tf.__version__} and Python {sys.version}.')
print('Libraries successfully imported. Initializing variables...')

# Define variables
Stock = 'TSLA'  # Stock to predict
checkpoint_path = f"V5models/{Stock} Check point"
model_path = f'V5models/{Stock}'
load_path = model_path  # CHANGE THIS TO TOGGLE
checkpoint_dir = os.path.dirname(checkpoint_path)
tolerant_rate = 1.0  # Threshold for holding stock instead of buy/sell
ACTIVATION = 'tanh'
RECURRENT_ACTIVATION = 'sigmoid'
DROPOUT = [0.002, 0.002, 0.002, 0.003]
UNITS = [256, 512, 256, 256]
BATCH_SIZE = 32
EPOCH = 90  # how many times to run at once
SEQ_LEN = 40  # how many days back used to predict, optimal 30/40 for NVDA
initializer = initializers.GlorotNormal()  # Xavier Normal initializer
LEARNING_RATE = 5e-4
LEARNING_DECAY = 1e-6
train_data_pct = 0.80  # percentage of data used for training
opt = tf.keras.optimizers.Adam(lr=LEARNING_RATE)  # decay=LEARNING_DECAY
# opt = tf.keras.optimizers.Adam()
scaler = MinMaxScaler()
Interval = "1h"
Period = '730d'
TODAY = pd.to_datetime('today').to_pydatetime()
LAST_WORKING_DAY = TODAY - datetime.timedelta(max(1, (TODAY.weekday() + 6) % 7 - 3))
print('All variables initialized.')

# Read data
data = yf.download(tickers=f'{Stock}', period=Period, interval=Interval)
data = data.drop(['Adj Close'], axis=1)  # Ditch adjusted close
last_close_price = float("{:.2f}".format(data.iloc[-1]["Close"]))
sec_last_close_price = float("{:.2f}".format(data.iloc[-2]["Close"]))
print(f'{Stock} Stock data on interval of {Interval} from past {Period} to {LAST_WORKING_DAY} received from Yahoo Finance. Latest closing price is ${last_close_price}.')

# Separating Data into training and validating and scale them between 0 and 1.
print(f'Gathered {len(data)} timestamps of data. Processing...')
scaled_data = scaler.fit_transform(data)  # Normalise to between 0 and 1
norm_scale = 1 / scaler.scale_[3]  # Save the normalisation factor for later use
train_data_len = np.math.ceil(len(scaled_data) * train_data_pct)
training_data = scaled_data[:train_data_len]
val_data = scaled_data[train_data_len:]
x_train = []
y_train = []
seq_length = 1
for i in range(0, train_data_len - seq_length, 1):
    seq_in = training_data[i:i + seq_length]
    seq_out = training_data[i + seq_length]
    x_train.append(seq_in)
    y_train.append(seq_out)
    print(seq_in * norm_scale, "->", seq_out * norm_scale)

x_train, y_train = np.array(x_train), np.array(y_train)
print()