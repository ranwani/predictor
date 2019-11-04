import os
import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib
import talib

from tensorflow_core.python.keras.models import Sequential
from tensorflow_core.python.keras.layers import Dense
from tensorflow_core.python.keras.layers import Dropout

from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    print("Prediction START")
    random.seed(42)
    file_path = os.path.join(pathlib.Path(__file__).parent,  'resources','AAPL.csv') #'RELIANCE.NS.csv'
    dataset = pd.read_csv(file_path)
    print("File has {} rows".format(len(dataset)))
    dataset.dropna(inplace=True)
    dataset = dataset[['Open', 'High', 'Low', 'Close']]
    dataset['H_L_Diff'] = dataset['High'] - dataset['Low']
    dataset['O_C_Diff'] = dataset['Close'] - dataset['Open']
    dataset['D1Close'] = dataset['Close'].shift(1)
    dataset['D2Close'] = dataset['Close'].shift(2)
    dataset['D3Close'] = dataset['Close'].shift(3)
    dataset['D4Close'] = dataset['Close'].shift(4)
    dataset['3_day_MA'] = dataset['Close'].shift(1).rolling(window=3).mean()
    dataset['10_day_MA'] = dataset['Close'].shift(1).rolling(window=10).mean()
    dataset['30_day_MA'] = dataset['Close'].shift(1).rolling(window=30).mean()
    dataset['Std_dev']= dataset['Close'].rolling(5).std()
    dataset['RSI'] = talib.RSI(dataset['Close'].values, timeperiod=9)
    dataset['Williams % R'] = talib.WILLR(dataset['High'].values, dataset['Low'].values, dataset['Close'].values, 7)

    dataset['Price_Rise'] = np.where(dataset['Close'].shift(-1) > dataset['Close'], 1, 0)
    dataset.dropna(inplace=True)
    X = dataset.iloc[:, 4:-1]
    y = dataset.iloc[:, -1]
    split = int(len(dataset) * 0.8)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    y_train = y_train.to_numpy()

    classifier = Sequential()
    classifier.add(Dense(units=64, activation = 'relu', input_dim = X.shape[1]))
    classifier.add(Dense(units=64, activation='relu'))
    classifier.add(Dense(units=1, activation = 'sigmoid'))
    classifier.compile(optimizer='adam', loss='mean_squared_error', metrics= ['accuracy'])
    classifier.fit(X_train, y_train, batch_size=10, epochs=50)

    y_pred = classifier.predict(X_test)
    #y_pred = (y_pred > 0.5)
    #y_pred = (np.round(y_pred * 2) - 1 )
    y_pred = (2*np.round(y_pred) - 1 )

    dataset['y_pred'] = np.NaN
    dataset.iloc[(len(dataset) - len(y_pred)):, -1:] = y_pred
    trade_dataset = dataset.dropna()

    trade_dataset['Tomorrows Returns'] = 0.
    trade_dataset['Tomorrows Returns'] = np.log(trade_dataset['Close'] / trade_dataset['Close'].shift(1))
    trade_dataset['Tomorrows Returns'] = trade_dataset['Tomorrows Returns'].shift(-1)

    trade_dataset['Strategy Returns'] = 0.
    # trade_dataset['Strategy Returns'] = np.where(trade_dataset['y_pred'] == True, trade_dataset['Tomorrows Returns'], - trade_dataset['Tomorrows Returns'])
    trade_dataset['Strategy Returns'] = trade_dataset['Tomorrows Returns'] * trade_dataset['y_pred']
    trade_dataset['Cumulative Market Returns'] = np.cumsum(trade_dataset['Tomorrows Returns'])
    trade_dataset['Cumulative Strategy Returns'] = np.cumsum(trade_dataset['Strategy Returns'])

    plt.figure(figsize=(10, 5))
    plt.plot(trade_dataset['Cumulative Market Returns'], color ='r', label ='Market Returns')
    plt.plot(trade_dataset['Cumulative Strategy Returns'], color ='g', label ='Strategy Returns')
    plt.legend()
    plt.show()