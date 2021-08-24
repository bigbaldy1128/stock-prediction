import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras import Sequential, regularizers
from tensorflow.python.keras.layers import GRU, Dense, Dropout

df = pd.read_csv('./data/999999.csv', usecols=['收盘'])

x_data = []
y_data = []

pastDay = 2
futureDay = 1

for i in range(df.shape[0] - futureDay - pastDay):
    x_data.append(np.array(df.iloc[i:i + pastDay]))
    y_data.append(1 if df.iloc[i + pastDay + futureDay]['收盘'] - df.iloc[i + pastDay]['收盘'] > 0 else 0)

x_data = np.array(x_data)
y_data = np.array(y_data)

print(x_data.shape)

# if x_data.shape[0] > y_data.shape[0]:
#     x_data = x_data[:y_data.shape[0] - x_data.shape[0]]

split = int(y_data.shape[0] * 0.8)
x_train, x_test = x_data[:split], x_data[split:]
y_train, y_test = y_data[:split], y_data[split:]

model = Sequential()
model.add(GRU(128, kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=['binary_accuracy'])  # accuracy,mse,binary_crossentropy

history = model.fit(x_train,
                    y_train,
                    epochs=10,
                    verbose=2)

y_pred = model.predict(x_test)
model.evaluate(x_test, y_test)
last_data = np.expand_dims(x_test[-1], axis=0)
print(model.predict(last_data))
# print("test AUC", round(roc_auc_score(y_test, y_pred), 4))
