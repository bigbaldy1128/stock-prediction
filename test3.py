import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                  columns=['a', 'b', 'c'])
f = ((df['a'] + df['c']) / 2).to_numpy()
f = np.expand_dims(f, axis=1)
f = MinMaxScaler().fit_transform(f)
f = np.squeeze(f)
print(f)
