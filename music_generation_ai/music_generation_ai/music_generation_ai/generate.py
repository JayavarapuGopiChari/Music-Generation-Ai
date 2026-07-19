"""
generate.py
-----------
Step 3 of the pipeline.

Loads the trained model + vocabulary mapping, seeds the network with a random
sequence from the training data, and autoregressively generates new notes,
which are then converted into a playable .mid file.

Run:
    python generate.py
"""

import json
import pickle

import numpy as np
from tensorflow.keras.models import load_model

import config
from utils.midi_utils import save_midi, set_output_instrument
from utils.helper import timestamped_filename


def prepare_generation_input(notes, note_to_int, sequence_length=config.SEQUENCE_LENGTH):
    """Build all possible seed windows so we can pick a random starting point."""
    network_input = []
    for i in range(len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[n] for n in seq_in])
    return network_input


def generate_notes(model, network_input, int_to_note, n_vocab,
                    num_notes=config.NOTES_TO_GENERATE,
                    temperature=config.GENERATION_TEMPERATURE):
    """Autoregressively predict `num_notes` new notes starting from a random seed."""
    start = np.random.randint(0, len(network_input) - 1)
    pattern = list(network_input[start])

    prediction_output = []

    for _ in range(num_notes):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)[0]

        # temperature-based sampling for more/less randomness
        prediction = np.log(prediction + 1e-9) / temperature
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        index = np.random.choice(len(prediction), p=prediction)

        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:]

    return prediction_output


def main(output_filename=None, num_notes=None, temperature=None, instrument_name=None,
         style=None, emotion=None):
    """
    num_notes       : how many notes to generate (default: config.NOTES_TO_GENERATE)
    temperature     : creativity/randomness, e.g. 0.5 (safe) - 1.5 (wild) (default: config.GENERATION_TEMPERATURE)
    instrument_name : e.g. "Piano", "Violin", "Flute", "Guitar" (default: Piano)
    style           : one of config.STYLE_PRESETS keys (e.g. "Pop", "Rock", "Metal") -
                      sets instrument + temperature unless explicitly overridden above
    emotion         : one of config.EMOTION_PRESETS keys (e.g. "Happy", "Sad") -
                      sets tempo (BPM) + octave shift for the rendered MIDI
    """
    style_preset = config.STYLE_PRESETS.get(style, {})
    emotion_preset = config.EMOTION_PRESETS.get(emotion, {})

    num_notes = num_notes or config.NOTES_TO_GENERATE
    temperature = temperature or style_preset.get("temperature") or config.GENERATION_TEMPERATURE
    instrument_name = instrument_name or style_preset.get("instrument") or "Piano"
    tempo_bpm = emotion_preset.get("tempo_bpm")
    octave_shift = emotion_preset.get("octave_shift", 0)

    print(f"Loading notes from: {config.NOTES_FILE}")
    with open(config.NOTES_FILE, "rb") as f:
        notes = pickle.load(f)

    mapping_path = config.NOTES_FILE.replace("notes.pkl", "note_mapping.json")
    print(f"Loading vocabulary mapping from: {mapping_path}")
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    note_to_int = mapping["note_to_int"]
    n_vocab = mapping["n_vocab"]
    int_to_note = {i: n for n, i in note_to_int.items()}

    print(f"Loading trained model from: {config.MODEL_WEIGHTS_FILE}")
    model = load_model(config.MODEL_WEIGHTS_FILE)

    print("Preparing seed sequences ...")
    network_input = prepare_generation_input(notes, note_to_int)

    print(f"Generating {num_notes} notes (style={style}, emotion={emotion}, temperature={temperature}) ...")
    prediction_output = generate_notes(
        model, network_input, int_to_note, n_vocab,
        num_notes=num_notes, temperature=temperature
    )

    if output_filename is None:
        output_filename = timestamped_filename()

    set_output_instrument(instrument_name)

    output_path = f"{config.OUTPUT_DIR}/{output_filename}"
    save_midi(prediction_output, output_path, tempo_bpm=tempo_bpm, octave_shift=octave_shift)
    print(f"Generated MIDI saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    main()
