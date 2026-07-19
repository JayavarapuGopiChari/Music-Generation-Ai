"""
preprocess.py
-------------
Step 1 of the pipeline.

Scans data/midi_songs/ for .mid files, extracts notes & chords using music21,
and pickles the resulting note list to notes/notes.pkl for train.py to consume.

Run:
    python preprocess.py
"""

import pickle

import config
from utils.midi_utils import extract_notes_from_folder


def main():
    print(f"Scanning MIDI files in: {config.DATA_DIR}")
    notes = extract_notes_from_folder(config.DATA_DIR)

    print(f"Extracted {len(notes)} note/chord events total.")
    print(f"Unique pitch classes: {len(set(notes))}")

    with open(config.NOTES_FILE, "wb") as f:
        pickle.dump(notes, f)

    print(f"Saved notes to: {config.NOTES_FILE}")
    print("Preprocessing complete. Next step -> python train.py")


if __name__ == "__main__":
    main()
