"""
train.py
--------
Step 2 of the pipeline.

Loads notes/notes.pkl (produced by preprocess.py), converts notes into
integer-encoded input/output sequences, builds the LSTM model, trains it,
and saves the trained weights to model/music_model.h5.

Run:
    python train.py
"""

import json
import pickle

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical

import config
from model import build_model


def prepare_sequences(notes, sequence_length=config.SEQUENCE_LENGTH):
    """Turn the raw note list into normalized (X, y) training tensors."""
    pitch_names = sorted(set(notes))
    note_to_int = {note: i for i, note in enumerate(pitch_names)}

    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    n_vocab = len(pitch_names)

    # reshape + normalize for LSTM: (samples, timesteps, features)
    X = np.reshape(network_input, (n_patterns, sequence_length, 1))
    X = X / float(n_vocab)
    y = to_categorical(network_output, num_classes=n_vocab)

    return X, y, note_to_int, n_vocab


def main():
    print(f"Loading notes from: {config.NOTES_FILE}")
    with open(config.NOTES_FILE, "rb") as f:
        notes = pickle.load(f)

    print("Preparing training sequences ...")
    X, y, note_to_int, n_vocab = prepare_sequences(notes)
    print(f"Training samples: {X.shape[0]}  |  Vocabulary size: {n_vocab}")

    # Save the note<->int mapping so generate.py can decode predictions later
    mapping_path = config.NOTES_FILE.replace("notes.pkl", "note_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump({"note_to_int": note_to_int, "n_vocab": n_vocab}, f)
    print(f"Saved vocabulary mapping to: {mapping_path}")

    print("Building model ...")
    model = build_model(n_vocab)
    model.summary()

    checkpoint = ModelCheckpoint(
        config.MODEL_WEIGHTS_FILE,
        monitor="loss",
        verbose=1,
        save_best_only=True,
        mode="min"
    )
    early_stop = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)

    print(f"Training for up to {config.EPOCHS} epochs (batch size {config.BATCH_SIZE}) ...")
    model.fit(
        X, y,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=[checkpoint, early_stop]
    )

    model.save(config.MODEL_WEIGHTS_FILE)
    print(f"Training complete. Model saved to: {config.MODEL_WEIGHTS_FILE}")
    print("Next step -> python generate.py")


if __name__ == "__main__":
    main()
