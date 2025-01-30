from keras import layers, models

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

def _get_attention_(input_layer):
    x = layers.Dense(units=16, activation="relu")(input_layer)
    hidden_states = layers.Bidirectional(
        # hidden state is a 32D vector
        # output shape: 64D vectorS --> shape: (batch_size, days, 64)
        layer=layers.LSTM(units=32, return_sequences=True),
        merge_mode="concat"
    )(x)
    # att_out_ = layers.Attention(use_scale=False, score_mode='dot')([hidden_states, hidden_states], return_attention_scores=False)
    att_out_ = layers.MultiHeadAttention(num_heads=3, key_dim=32)(hidden_states, hidden_states, return_attention_scores=False)
    return att_out_[:,-1,:]

def transformer_encoder(inputs, head_size, num_heads, n_filters, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=0.001)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=n_filters, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=0.001)(x)
    return x + res

def _get_attention(input_layer, head_size=256, num_heads=3, n_filters=4, num_transformer_blocks=2, mlp_units=[64], dropout=0, mlp_dropout=0):
    # x = input_layer
    x = layers.Permute((2,1))(input_layer)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, n_filters, dropout)

    # x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    x = layers.Flatten()(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    return x

def get_model(model_name:str, input_shape:tuple, output_shape:int, mode:str) -> models.Model:
    input_ = layers.Input(shape=input_shape)
    if model_name == LSTM:
        x = _get_lstm(input_)
    elif model_name == CNN:
        x = _get_cnn(input_)
    elif model_name == LSTM_CNN:
        x = _get_lstm_cnn(input_)
    elif model_name == ATT:
        x = _get_attention(input_)
    else:
        raise ValueError('wrong model name')

    if mode == REG:
        output_ = layers.Dense(units=1)(x)
    elif mode == CLF:
        output_ = layers.Dense(units=output_shape, activation='softmax')(x)
    return models.Model(inputs=input_, outputs=output_)

# bla = get_model(ATT, (20,4), 2, CLF)
# bla.summary()
