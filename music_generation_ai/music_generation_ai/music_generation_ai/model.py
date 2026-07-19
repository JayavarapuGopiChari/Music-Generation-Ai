"""
model.py
--------
Defines the LSTM network architecture used to learn and generate music.

The model takes a sequence of previous notes (as integers) and predicts
the probability distribution over the next note in the vocabulary.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.optimizers import RMSprop

import config


def build_model(n_vocab, sequence_length=config.SEQUENCE_LENGTH):
    """
    Build and compile the LSTM model.

    n_vocab          : size of the note vocabulary (number of unique notes/chords)
    sequence_length  : number of timesteps fed into the network
    """
    model = Sequential()

    model.add(LSTM(
        config.LSTM_UNITS,
        input_shape=(sequence_length, 1),
        return_sequences=True
    ))
    model.add(Dropout(config.DROPOUT_RATE))

    model.add(LSTM(config.LSTM_UNITS, return_sequences=True))
    model.add(Dropout(config.DROPOUT_RATE))

    model.add(LSTM(config.LSTM_UNITS))
    model.add(BatchNormalization())
    model.add(Dropout(config.DROPOUT_RATE))

    model.add(Dense(config.DENSE_UNITS))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(config.DROPOUT_RATE))

    model.add(Dense(n_vocab))
    model.add(Activation("softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=RMSprop(learning_rate=0.001)
    )

    return model
