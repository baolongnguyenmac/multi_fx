# libraries
from keras import Sequential, layers, models
import tensorflow as tf

from common.constants import *

def _get_lstm(input_layer):
    x = layers.Dense(units=16, activation="relu")(input_layer)
    return layers.Bidirectional(
        # hidden state is a 32D vector
        # the output is returned when the last x is passed
        # output shape: 64D vector
        layer=layers.LSTM(units=32, return_sequences=False),
        merge_mode="concat"
    )(x)

def _get_cnn(input_layer):
    x = layers.Dense(units=16, activation="relu")(input_layer)
    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu')(x)
    x = layers.MaxPool1D(pool_size=2)(x)
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = layers.MaxPool1D(pool_size=2)(x)
    return layers.Flatten()(x)

def _get_lstm_cnn(input_layer):
    cnn_output = _get_cnn(input_layer)

    dense1 = layers.Dense(units=16, activation="relu")(input_layer)
    lstm_output = layers.Bidirectional(
        layer=layers.LSTM(32, return_sequences=False),
        merge_mode='concat'
    )(dense1)

    return layers.Concatenate()([cnn_output, lstm_output])

def get_model(model_name:str, input_shape:tuple, mode:str):
    input_ = layers.Input(shape=input_shape)
    if model_name == LSTM:
        x = _get_lstm(input_)
    elif model_name == CNN:
        x = _get_cnn(input_)
    elif model_name == LSTM_CNN:
        x = _get_lstm_cnn(input_)
    elif model_name == ATT: # I was planning to implement attention model
        pass
    else:
        raise ValueError('wrong model name')

    if mode == REG:
        output_ = layers.Dense(units=1)(x)
    elif mode == CLF:
        output_ = layers.Dense(units=1, activation='sigmoid')(x)
    return models.Model(inputs=input_, outputs=output_)

# bla = get_model(LSTM, (20,4), CLF)
# bla.summary()
