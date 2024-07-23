# libraries
from keras import Sequential, layers, models
import tensorflow as tf

from common.constants import *

def max_pooling(feature):
    return tf.reduce_max(feature, axis=1)

def get_lstm(input_shape:tuple):
    return Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(units=16, activation="relu"),
        layers.Bidirectional(
            # hidden state is a 32D vector
            # the output is returned when the last x is passed
            # output shape: 64D vector
            layer=layers.LSTM(units=32, return_sequences=False),
            merge_mode="concat"
        ),
    ])

def get_cnn(input_shape:tuple):
    return Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(units=16, activation="relu"),
        layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        layers.MaxPool1D(pool_size=2),
        layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        layers.MaxPool1D(pool_size=2),
        layers.Flatten(),
    ])

def get_lstm_cnn(input_shape:tuple):
    input_layer = layers.Input(shape=input_shape)
    dense1 = layers.Dense(units=32, activation='relu')(input_layer)

    conv1 = layers.Conv1D(16, 3, activation='relu')(dense1)
    max_pool1 = layers.Lambda(max_pooling)(conv1)

    conv2 = layers.Conv1D(32, 5, activation='relu')(dense1)
    max_pool2 = layers.Lambda(max_pooling)(conv2)
    cnn_output = layers.Concatenate()([max_pool1, max_pool2])

    lstm_output = layers.Bidirectional(
        layer=layers.LSTM(32, return_sequences=False),
        merge_mode='concat'
    )(dense1)

    mix_feature = layers.Concatenate()([cnn_output, lstm_output])
    return models.Model(inputs=input_layer, outputs=mix_feature)

def get_model(model_name:str, input_shape:tuple, mode:str):
    if model_name == LSTM:
        model:models.Model = get_lstm(input_shape)
    elif model_name == CNN:
        model:models.Model = get_cnn(input_shape)
    elif model_name == LSTM_CNN:
        model:models.Model = get_lstm_cnn(input_shape)
    elif model_name == ATT:
        pass
        # input_layer = layers.Input(shape=input_shape)
        # dense1 = layers.Dense(units=16, activation="relu")(input_layer)
        # lstm_out = layers.LSTM(64, return_sequences=True)(dense1)

        # # attention_out = AttentionLayer()(lstm_out)
        # attention = layers.TimeDistributed(Dense(1, activation = 'tanh'))(lstm_out)
        # attention = layers.Flatten()(attention)
        # attention = lstm_out * attention.reshape((-1,1))
        # attention = layers.Activation('softmax')(attention)
        # attention = layers.Permute([2, 1])(attention)
        # attention = layers.Flatten()(attention)

        # output = layers.Dense(1, activation='sigmoid')(attention)
        # model = models.Model(input_layer, output)
    else:
        raise ValueError('wrong model name')

    if mode == REG:
        model.add(layer=layers.Dense(units=1))
    elif mode == CLF:
        model.add(layer=layers.Dense(units=1, activation='sigmoid'))
    return model

# bla = get_model(CNN, (20,4), CLF)
# bla.summary()
