print('LSTM (Long Short-Term Memory) Stock Prediction Version 2.9 - PREDICTION ONLY')
print('Importing libraries...')
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from pandas_datareader import data
import pandas as pd
import numpy as np
import datetime as datetime
import sys
import os

print(f'Running on Tensor Flow {tf.__version__} and Python {sys.version}.')
print('Libraries successfully imported. Initializing variables...')

# Define some variables for ease of change in the model
Stock = 'NVDA'  # Stock to predict
checkpoint_path = f"V2.9models/{Stock} Check point.ckpt"
model_path = f'V2.9models/{Stock}.tf'
load_path = model_path  # CHANGE THIS
checkpoint_dir = os.path.dirname(checkpoint_path)
tolerant_rate = 2.0  # Threshold for holding stock instead of buy/sell in %
SEQ_LEN = 40  # how many days back used to predict, optimal 30/40 for NVDA
train_data_pct = 0.80  # percentage of data used for training
TODAY = pd.to_datetime('today').to_pydatetime()
offset = max(1, (TODAY.weekday() + 6) % 7 - 3)
timedelta = datetime.timedelta(offset)
START_DATE = '2015-1-1'
LAST_WORKING_DAY = TODAY - timedelta
print('All variables initialized.')

print(
    f'Stock to predict: {Stock}. The last working day is {LAST_WORKING_DAY}, getting data from {START_DATE} to {LAST_WORKING_DAY}.')
# Read data
data = data.DataReader(f'{Stock}',
                       start=f'{START_DATE}',
                       end=f'{LAST_WORKING_DAY}',
                       data_source='yahoo')

last_close_price = float("{:.2f}".format(data.iloc[-1]["Close"]))
sec_last_close_price = float("{:.2f}".format(data.iloc[-2]["Close"]))
print(
    f'{Stock} Stock data from {START_DATE} to {LAST_WORKING_DAY} received from Yahoo Finance. Latest closing price is ${last_close_price}.')

# Separating Data into training and validating.
# Data from 1 year ago to now is used for validation, rest is for training. Data range is 2015-05-08 to 2020-05-
print(f'Gathered {len(data)} days of data. Processing...')
# number of columns used to train
train_data_len = np.math.ceil(len(data) * train_data_pct)
data_training = data[0:train_data_len]
data_test = data[train_data_len:]

training_data = data_training.drop(
    ['Adj Close'], axis=1)  # Ditch adjusted close
scaler = MinMaxScaler()
training_data = scaler.fit_transform(
    training_data)  # Normalise to between 0 and 1

X_train = []
y_train = []

for i in range(SEQ_LEN, training_data.shape[0]):
    X_train.append(training_data[i - SEQ_LEN:i])
    y_train.append(training_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
# print(X_train.shape, y_train.shape)

# Preparing dataset for training
past_days = data_training.tail(SEQ_LEN)
df = past_days.append(data_test, ignore_index=True)
df = df.drop(['Adj Close'], axis=1)  # Ditch useless columns of data
inputs = scaler.transform(df)  # Normalise to between 0 and 1

X_test = []
y_test = []

for i in range(SEQ_LEN, inputs.shape[0]):
    X_test.append(inputs[i - SEQ_LEN:i])
    y_test.append(inputs[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
print(f'Processed and scaled data according to ratio {scaler.scale_[0]}.')

# Building LSTM Model
print('Loading LSTM Model...')
model = tf.keras.models.load_model(load_path)  # uncomment this if not training for first time
print(f'Model loaded successfully from {load_path}.')
model.summary()
# model.load_weights(checkpoint_path)
# print(f'Weights loaded from {checkpoint_path}')

print('Start prediction...')
# Prediction
y_predict = model.predict(X_test)

# Normalisation back to USD
Norm_scale = 1 / scaler.scale_[0]
y_predict_normalized = y_predict * Norm_scale
y_test_normalized = y_test * Norm_scale

# Evaluate model by Root Mean Squared Error (RMSE)
rmse = np.sqrt(((y_predict - y_test) ** 2).mean())
print(f'Root Mean Squared Error is {rmse}. 0 is perfect.')

# Prediction of actual price based on predicted movement %
predicted_change_pct = float(
    y_predict_normalized[-1] / y_predict_normalized[-2])
last_predicted_change_pct = float(
    y_predict_normalized[-2] / y_predict_normalized[-3])
last_predicted_price = float("{:.2f}".format(
    sec_last_close_price * last_predicted_change_pct))
accuracy = last_predicted_price / last_close_price * 100
predicted_price = float("{:.2f}".format(
    last_close_price * predicted_change_pct))
predicted_price_corrected = float("{:.2f}".format(predicted_price)) / float(accuracy) * float(
    100)  # divide by last prediction accuracy
predicted_price_change_pct = (predicted_price - last_close_price) / last_close_price * 100
predicted_corrected_price_change_pct = (predicted_price_corrected - last_close_price) / last_close_price * 100
avg_prediction = (predicted_price + predicted_price_corrected) / 2
avg_prediction_change_pct = (avg_prediction - last_close_price) / last_close_price * 100
print()
print(
    f'Last predicted price: ${last_predicted_price}, actual price: ${last_close_price}')
print(f'Last prediction accuracy: {accuracy}%')


def signal():
    if predicted_price_change_pct <= tolerant_rate and predicted_price_change_pct >= - tolerant_rate:  # if predicted change is less than 2.5%, hold.
        action = 'Hold'
    elif predicted_price_change_pct < - tolerant_rate:
        action = 'Sell'
    elif predicted_price_change_pct > tolerant_rate:
        action = 'Buy'
    return action


action = signal()
print()
print(f"Predicted Price: ${predicted_price} ({predicted_price_change_pct}%)"
      f"\nCorrected predicted price according to last prediction error: ${predicted_price_corrected} ({predicted_corrected_price_change_pct}%)"
      f"\nAverage prediction: ${avg_prediction} ({avg_prediction_change_pct}%)"
      f"\nPrevious day close price: ${last_close_price}"
      f"\nAction: {action}")
