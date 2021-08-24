import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras import Sequential, regularizers
from tensorflow.python.keras.layers import GRU, Dense, Dropout
import tensorflow as tf

df = pd.read_csv("./data/bitcoin.csv", usecols=['Close','Volume'])
df = df.iloc[::-1]
df = df.applymap(lambda p: p.replace("$", "").replace(",", "").replace(" ", ""))
df['Close'] = df['Close'].astype('float64')
df['Volume'] = df['Volume'].astype('float64')
# mms = MinMaxScaler(feature_range=(1, 10))
# df[['Open','High','Low','Close','Volume']] = mms.fit_transform(df[['Close','Volume']])

x_data = []
y_data = []

pastDay = 10
futureDay = 1

for i in range(df.shape[0] - pastDay - futureDay):
    x_data.append(np.array(df.iloc[i:i + pastDay]))
    y_data.append(df.iloc[i + pastDay + futureDay - 1]['Close'] / df.iloc[i + pastDay - 1]['Close'] - 1)

x_data = np.array(x_data)
y_data = np.array(y_data)

print(x_data.shape)

split = int(y_data.shape[0] * 0.8)
x_train, x_test = x_data[:split], x_data[split:]
y_train, y_test = y_data[:split], y_data[split:]

model = Sequential()
model.add(GRU(128, kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='linear'))

model.compile(optimizer="adam",
              loss="mae",  # binary_crossentropy
              metrics=['mae'])  # accuracy,mae,mse,binary_crossentropy,binary_accuracy

log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

history = model.fit(x_train,
                    y_train,
                    epochs=50,
                    verbose=2,
                    callbacks=[tensorboard_callback])

y_pred = model.predict(x_test)
model.evaluate(x_test, y_test)
x = []
for i in range(1, 20):
    x.append(x_test[-i])
print(model.predict(np.array(x)))