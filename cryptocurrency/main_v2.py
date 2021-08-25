import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from cryptocurrency.model_v1 import MyModelV1
from cryptocurrency.model_v2 import MyModelV2

df = pd.read_csv("./data/btcusdt.csv", usecols=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume'])
# df = df[df["Open time"] > 1589414400000]
df = df.drop(['Open time'], axis=1)
mms = MinMaxScaler()
train_data = mms.fit_transform(df)

x_data = []
x_data_implicit = []
y_data = []

TIME_RANGE = 60
RECIPROCAL_TIME_RANGE = 1 / TIME_RANGE - 1e-4
pastDay = TIME_RANGE
FUTURE_DAY = 0

for i in range(pastDay, train_data.shape[0] - FUTURE_DAY):
    x_data.append(train_data[i - pastDay:i])
    current = train_data[i + FUTURE_DAY, 3]
    y_data.append(current)

    rate = 0
    for j in range(pastDay):
        if train_data[i - j - 1, 3] / current > 1.01:
            rate = rate + RECIPROCAL_TIME_RANGE
    x_data_implicit.append(int(rate * TIME_RANGE))

x_data = np.array(x_data)
y_data = np.array(y_data)
x_data_implicit = np.array(x_data_implicit)

print(x_data.shape)

split = int(y_data.shape[0] * 0.7)
x_train, x_test = x_data[:split], x_data[split:]
y_train, y_test = y_data[:split], y_data[split:]
x_train_implicit, x_test_implicit = x_data_implicit[:split], x_data_implicit[split:]

model = MyModelV2(embedding_size=TIME_RANGE, input1_size=x_data.shape[1:])
model.compile(optimizer="adam",
              loss="mse",  # binary_crossentropy
              metrics=['mse'])  # accuracy,mae,mse,binary_crossentropy,binary_accuracy

log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

history = model.fit([x_train, x_train_implicit],
                    y_train,
                    epochs=20,
                    verbose=2)  # callbacks=[tensorboard_callback]

y_pred = model.predict([x_test, x_test_implicit])
model.evaluate([x_test, x_test_implicit], y_test)

scale = 1 / mms.scale_[3]
y_pred = y_pred * scale
y_test = y_test * scale

epochs = range(y_pred.shape[0])
plt.figure(figsize=(14, 5))
plt.plot(epochs, y_pred, 'b', label='prediction')
plt.plot(epochs, y_test, 'r', label='actual')
plt.title("Prediction for BTC")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
