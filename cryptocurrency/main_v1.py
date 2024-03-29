import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from cryptocurrency.model_v1 import MyModelV1

df = pd.read_csv("./data/btcusdt.csv", usecols=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume'])
# df = df[df["Open time"] > 1589414400000]
df = df.drop(['Open time'], axis=1)
mms = MinMaxScaler()
df[['Open', 'High', 'Low', 'Close', 'Volume']] = mms.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])
df['Middle'] = (df['High'] + df['Low']) / 2

x_data = []
x_data_implicit = []
y_data = []

TIME_RANGE = 10
pastDay = TIME_RANGE
futureDay = TIME_RANGE

for i in range(df.shape[0] - pastDay - futureDay):
    x_data.append(np.array(df.iloc[i:i + pastDay][['Open', 'High', 'Low', 'Close', 'Volume']]))
    current = df.iloc[i + pastDay]['Middle']
    rate = 0
    for j in range(0, pastDay):
        hl2 = df.iloc[i + j]['Middle']
        if hl2 / current > 1.01:
            rate = rate + 1 / pastDay
    x_data_implicit.append(int(rate * TIME_RANGE))

    rate = 0
    for j in range(1, futureDay):
        hl2 = df.iloc[i + pastDay + j]['Middle']
        if hl2 / current > 1.01:
            rate = rate + 1 / futureDay

    if rate > 0.2:
        y_data.append(1)
    else:
        y_data.append(0)

x_data = np.array(x_data)
y_data = np.array(y_data)
x_data_implicit = np.array(x_data_implicit)

print(x_data.shape)

split = int(y_data.shape[0] * 0.7)
x_train, x_test = x_data[:split], x_data[split:]
y_train, y_test = y_data[:split], y_data[split:]
x_train_implicit, x_test_implicit = x_data_implicit[:split], x_data_implicit[split:]

model = MyModelV1(TIME_RANGE, input1_size=x_data.shape[1:])
model.compile(optimizer="adam",
              loss="binary_crossentropy",  # binary_crossentropy
              metrics=['binary_accuracy'])  # accuracy,mae,mse,binary_crossentropy,binary_accuracy

log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

history = model.fit([x_train, x_train_implicit],
                    y_train,
                    epochs=100,
                    verbose=2,
                    callbacks=[tensorboard_callback])

y_pred = model.predict([x_test, x_test_implicit])
model.evaluate([x_test, x_test_implicit], y_test)

LENGTH = 100
y_pred = np.squeeze(y_pred[-LENGTH:])
df = df[-LENGTH:]
y_actual = df['Middle'].to_numpy()
y_actual = np.expand_dims(y_actual, axis=1)
y_actual = MinMaxScaler().fit_transform(y_actual)
y_actual = np.squeeze(y_actual)

epochs = range(LENGTH)
plt.figure()
plt.plot(epochs, y_pred, 'b', label='prediction')
plt.plot(epochs, y_actual, 'r', label='actual')
plt.title("Prediction for BTC")
plt.legend()
plt.show()
