# @author wangjinzhao on 2021/8/23
# reference https://www.analyticsvidhya.com/blog/2021/05/bitcoin-price-prediction-using-recurrent-neural-networks-and-lstm/
from tensorflow.python.keras import regularizers, Model, Input
from tensorflow.python.keras.layers import GRU, Dropout, Embedding, Concatenate, Dense, Flatten
from tensorflow.python.ops.init_ops_v2 import RandomNormal


def MyModelV2(embedding_size, input1_size):
    input1 = Input(input1_size)
    input2 = Input(shape=(1,))
    inputs_list = [input1, input2]

    output = GRU(units=50, activation='relu', return_sequences=True)(input1)
    output = Dropout(0.2)(output)
    output = GRU(units=60, activation='relu', return_sequences=True)(output)
    output = Dropout(0.3)(output)
    output = GRU(units=80, activation='relu', return_sequences=True)(output)
    output = Dropout(0.4)(output)
    output = GRU(units=120, activation='relu')(output)
    output = Dropout(0.5)(output)
    # output = Dense(units=5, activation='relu')(output)
    # embedding = Embedding(embedding_size, 20, embeddings_initializer=RandomNormal(mean=0.5, stddev=0.0001, seed=1024))(
    #     input2)
    # embedding = Flatten()(embedding)
    # output = Concatenate(axis=-1)([output, embedding])
    output = Dense(units=10)(output)
    output = Dense(units=5)(output)
    output = Dense(units=1)(output)
    model = Model(inputs=inputs_list, outputs=output)
    return model
