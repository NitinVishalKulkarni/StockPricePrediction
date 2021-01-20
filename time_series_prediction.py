#!/usr/bin/env python
# coding: utf-8

# Imports
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, BatchNormalization, LeakyReLU
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.vector_ar.var_model import VAR
import tensorflow as tf
import time


start = time.time()


# Don't run this block of code if you aren't using Tensorflow-GPU.
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Reading the data set.
walmart = pd.read_csv('WMT.csv')


# First 5 entries in the data set.
print('First 5 entries in the data set:\n', walmart.head())


# Shape of the data set.
print('\nData set Shape:\n', walmart.shape)


# Description of the data set.
print('\nData set description:\n', walmart.describe())


# Data set info.
print('\nData set info:\n', walmart.info())


# Visualization of the data set.
fig, axes = plt.subplots(4, 1, figsize=(15, 10))
axes[0].plot(walmart['Open'], alpha=0.5)
axes[0].title.set_text('Open')
axes[1].plot(walmart['Close'], alpha=0.5)
axes[1].title.set_text('Close')
axes[2].plot(walmart['Low'], alpha=0.5)
axes[2].title.set_text('Low')
axes[3].plot(walmart['High'], alpha=0.5)
axes[3].title.set_text('High')
plt.subplots_adjust(top=1, wspace=0, hspace=0.5)
plt.show()


# Splitting the dataset into training dataset and testing dataset.
training_data, testing_data = walmart[0:int(len(walmart)*0.8)], walmart[int(len(walmart)*0.8):]
print('\nTraining Dataset Shape:', training_data.shape)
print('\nTesting Dataset Shape:', testing_data.shape)


# Plot showing the training and testing data.
plt.figure(figsize=(15, 5))
plt.plot(training_data['Close'], 'blue', label='Training Data')
plt.plot(testing_data['Close'], 'red', label='Testing Data')
plt.legend()
plt.title('Plot Showing the Training Data and Testing Data')
plt.show()


# # ARIMA Model

train_arima = training_data['Close'].values
test_arima = testing_data['Close'].values
history = [x for x in train_arima]
predictions = list()

for i in range(len(test_arima)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    predictions.append(output[0])
    obs = test_arima[i]
    history.append(obs)

# Plotting the Actual Stock Closing Price vs Predicted Stock Closing Price.
plt.figure(figsize=(15, 10))
plt.plot(test_arima, label='Actual Stock Closing Price')
plt.plot(predictions, color='red', label='Predicted Stock Closing Price')
plt.legend()
plt.title('ARIMA Model: Actual Stock Closing Price Vs Predicted Stock Closing Price')
plt.show()

# Mean Squared Error
mse_arima = mean_squared_error(test_arima, predictions)
print('\nMSE of ARIMA Model is:', mse_arima)


# # Auto Regression Model

# Making the data stationary by first differencing.
train_ar = training_data['Close']
train_ar = train_ar.diff().dropna()
test_ar = testing_data['Close'].values
history = [x for x in train_ar.values]
predictions = list()

for i in range(len(test_ar)):
    model = AutoReg(history, lags=2)
    model_fit = model.fit()
    output = model_fit.predict(start=len(train_ar), end=len(train_ar) + len(test_ar))
    predictions.append(output[0] + test_ar[i])
    obs = test_ar[i]
    history.append(obs)

# Plotting the Actual Stock Closing Price vs Predicted Stock Closing Price.
plt.figure(figsize=(15, 10))
plt.plot(test_ar, label='Actual Stock Closing Price')
plt.plot(predictions, color='red', label='Predicted Stock Closing Price')
plt.legend()
plt.title('AR Model: Actual Stock Closing Price Vs Predicted Stock Price')
plt.show()

# Mean Squared Error
mse_ar = mean_squared_error(test_ar, predictions)
print('\nMSE of AR Model is:', mse_ar)


# # VAR Model.

# Making the data stationary by first differencing.
train_var = training_data[['Open', 'Close']]
train_var = train_var.diff().dropna()
test_var = testing_data[['Open', 'Close']]
history = [x for x in train_var[['Open', 'Close']].values]
predictions = list()

for i in range(len(test_var)):
    model = VAR(endog=history)
    model_fit = model.fit()
    output = model_fit.forecast(model_fit.y, steps=len(test_var))
    predictions.append(output[i, 1])
    obs = test_var.iloc[i]
    history.append(obs)

# Plotting the Actual Stock Closing Price vs Predicted Stock Closing Price.
plt.figure(figsize=(15, 10))
plt.plot(test_var['Close'].values, label='Actual Stock Closing Price')
plt.plot(predictions, color='red', label='Predicted Stock Closing Price')
plt.legend()
plt.title('VAR Model: Actual Stock Closing Price Vs Predicted Stock Price')
plt.show()

# Calculate Mean Squared Error:
mse = mean_squared_error(test_var['Close'], predictions)
print('\nMSE of VAR Model is:', mse)


# # MLP

data = pd.read_csv('WMT.csv')[::-1]
data = data.loc[:, 'Close'].tolist()


# Shuffling in unison.
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def create_xt_yt(x_, y_, percentage=0.8):
    p = int(len(x_) * percentage)
    x__train = x_[0:p]
    y__train = y_[0:p]

    x__train, y__train = shuffle_in_unison(x__train, y__train)

    x__test = x_[p:]
    y__test = y_[p:]

    return x__train, x__test, y__train, y__test


# Parameters.
WINDOW = 30
EMB_SIZE = 1
STEP = 1
FORECAST = 1


X, Y = [], []
for i in range(0, len(data), STEP):
    try:
        x_i = data[i:i + WINDOW]
        y_i = data[i + WINDOW + FORECAST]

        last_close = x_i[WINDOW - 1]
        next_close = y_i

        if last_close < next_close:
            y_i = [1, 0]
        else:
            y_i = [0, 1]

    except Exception as e:
        print(e)
        break
        
    X.append(x_i)
    Y.append(y_i)
    
X = [(np.array(x) - np.mean(x)) / np.std(x) for x in X]
X, Y = np.array(X), np.array(Y)

X_train, X_test, Y_train, Y_test = create_xt_yt(X, Y)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# # Setup 1

# Instantiating the model.
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=30))
model.add(Dense(2))
model.add(Activation('softmax'))
optimizer = Adam()

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=100, batch_size=128, verbose=0, validation_data=(X_test, Y_test),
                    shuffle=True)

# Plot Accuracy.
plt.figure(figsize=(15, 7))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

# Plot Loss.
plt.figure(figsize=(15, 7))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()


# # Setup 2

# Instantiating the model.
model = Sequential()
model.add(Dense(512, input_dim=30, kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(128, activity_regularizer=regularizers.l2(0.01)))
model.add(Dense(2))
model.add(Activation('softmax'))
optimizer = Adam()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.0001, verbose=0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=150, batch_size=128, verbose=0, validation_data=(X_test, Y_test),
                    shuffle=True, callbacks=[reduce_lr])

# Plot Accuracy.
plt.figure(figsize=(15, 7))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

# Plot Loss.
plt.figure(figsize=(15, 7))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()


# # Setup 3

# Instantiating the model.
model = Sequential()
model.add(Dense(1024, input_dim=30, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))) 
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dense(1024, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dense(1024, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dense(1024, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dense(2))
model.add(Activation('softmax'))
optimizer = Adam()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.0001, verbose=0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=100, batch_size=64, verbose=0, validation_data=(X_test, Y_test),
                    shuffle=True, callbacks=[reduce_lr])

# Plot Accuracy.
plt.figure(figsize=(15, 7))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

# Plot Loss.
plt.figure(figsize=(15, 7))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()


# # LSTM

scaler = MinMaxScaler(feature_range=(0, 1))


df = pd.read_csv('WMT.csv', usecols=['Date', 'Close'])

df.set_index('Date', inplace=True)

walmart = df.values

transformedData = scaler.fit_transform(walmart)

timesteps = 60

x, y = [], []

for i in range(timesteps, len(walmart)):
    x.append(transformedData[i-timesteps:i, 0])
    y.append(transformedData[i, 0])

x = np.array(x)
y = np.array(y)

x_train = x[:4000]
y_train = y[:4000]

x_test = x[4000:]
y_test = y[4000:]


x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=50, batch_size=128, verbose=0)

predictions = (model.predict(x_test))

# Plotting the Actual Stock Closing Price vs Predicted Stock Closing Price.
plt.figure(figsize=(20, 10))
plt.plot(y_test, label='Actual Stock Closing Price')
plt.plot(predictions, label='Predicted Stock Closing Price')
plt.legend()
plt.title('LSTM: Actual Stock Closing Price Vs Predicted Stock Price')
plt.show()

end = time.time()
print('Runtime:', end - start)
