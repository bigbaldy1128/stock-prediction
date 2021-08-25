# @author wangjinzhao on 2021/8/23
# reference https://www.analyticsvidhya.com/blog/2021/05/bitcoin-price-prediction-using-recurrent-neural-networks-and-lstm/
from tensorflow.python.keras import regularizers, Model, Input
from tensorflow.python.keras.layers import GRU, Dropout, Embedding, Concatenate, Dense, Flatten
from tensorflow.python.ops.init_ops_v2 import RandomNormal


def MyModel():
    input1 = Input(shape=(30, 5))
    input2 = Input(shape=(1,))
    inputs_list = [input1, input2]

    output = GRU(units=50, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.001))(input1)
    output = Dropout(0.2)(output)
    output = GRU(units=60, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.001))(output)
    output = Dropout(0.3)(output)
    output = GRU(units=80, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.001))(output)
    output = Dropout(0.4)(output)
    output = GRU(units=120, activation='relu', kernel_regularizer=regularizers.l2(0.001))(output)
    output = Dropout(0.5)(output)
    output = Dense(units=5, activation='relu')(output)
    embedding = Embedding(10, 20, embeddings_initializer=RandomNormal(mean=0.5, stddev=0.0001, seed=1024))(input2)
    embedding = Flatten()(embedding)
    output = Concatenate(axis=-1)([output, embedding])
    output = Dense(units=1, activation='sigmoid')(output)
    model = Model(inputs=inputs_list, outputs=output)
    return model
