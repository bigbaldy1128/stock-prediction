import datetime

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from cryptocurrency.model import MyModel

df = pd.read_csv("./data/btcusdt.csv", usecols=['Open', 'High', 'Low', 'Close', 'Volume'])
mms = MinMaxScaler()
df[['Open', 'High', 'Low', 'Close', 'Volume']] = mms.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])

x_data = []
x_data_implicit = []
y_data = []

pastDay = 30
futureDay = 30

for i in range(df.shape[0] - pastDay - futureDay):
    x_data.append(np.array(df.iloc[i:i + pastDay]))
    current = df.iloc[i + pastDay]['Close']
    rate = 0
    for j in range(0, pastDay):
        if df.iloc[i + pastDay]['Close'] - current > 0:
            rate = rate + 1 / futureDay
    x_data_implicit.append(int(rate * 10))

    rate = 0
    for j in range(1, futureDay):
        if df.iloc[i + pastDay + j]['Close'] - current > 0:
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

model = MyModel()
model.compile(optimizer="adam",
              loss="binary_crossentropy",  # binary_crossentropy
              metrics=['binary_accuracy'])  # accuracy,mae,mse,binary_crossentropy,binary_accuracy

log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

history = model.fit([x_train, x_train_implicit],
                    y_train,
                    epochs=10,
                    verbose=2,
                    callbacks=[tensorboard_callback])

y_pred = model.predict([x_test, x_test_implicit])
model.evaluate([x_test, x_test_implicit], y_test)

x = []
x_implicit = []
for i in range(1, 50):
    x.append(x_test[-i])
    x_implicit.append(x_test_implicit[-i])
print(model.predict([np.array(x), np.array(x_implicit)]))
