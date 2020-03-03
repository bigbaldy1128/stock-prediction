# import pandas as pd
import tensorflow as tf

# df = pd.DataFrame([[2, 1], [2, 2], [5, 3]], columns=['A', 'B'])
# df2 = pd.DataFrame([[1, 4], [2, 5], [3, 6], [4, 7]], columns=['A', 'D'])
# df = df.join(df2.set_index('A'), on='A')
# print(df)
from tensorflow.python.keras import losses

x = tf.random.normal([4, 3, 5])
xt = tf.unstack(x, axis=1)
with tf.Session() as sess:
    print(sess.run(xt))
