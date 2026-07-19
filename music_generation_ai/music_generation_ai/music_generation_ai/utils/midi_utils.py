"""
utils/midi_utils.py
--------------------
Low-level helpers that talk to music21 / MIDI files.
Kept separate from preprocess.py and generate.py so both can reuse the same logic.
"""

import glob
import os
from music21 import converter, instrument, note, chord, stream, tempo

# Instruments a user can pick from the web UI / CLI for the generated output.
INSTRUMENT_MAP = {
    "Piano": instrument.Piano,
    "Violin": instrument.Violin,
    "Flute": instrument.Flute,
    "Guitar": instrument.AcousticGuitar,
    "Cello": instrument.Violoncello,
    "Trumpet": instrument.Trumpet,
    "Saxophone": instrument.Saxophone,
}

_current_instrument = instrument.Piano


def set_output_instrument(name):
    """Set which instrument the next save_midi() call should render with."""
    global _current_instrument
    _current_instrument = INSTRUMENT_MAP.get(name, instrument.Piano)


def extract_notes_from_file(midi_path):
    """
    Parse a single MIDI file and return a flat list of note/chord strings.
    Notes are represented as their pitch name (e.g. 'C4').
    Chords are represented as dot-separated pitch ids (e.g. '4.8.11').
    """
    notes = []
    midi = converter.parse(midi_path)

    parts = instrument.partitionByInstrument(midi)
    if parts:  # file has instrument parts
        notes_to_parse = parts.parts[0].recurse()
    else:      # file has notes in a flat structure
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append(".".join(str(n) for n in element.normalOrder))

    return notes


def extract_notes_from_folder(folder_path):
    """
    Walk every .mid/.midi file in folder_path and combine all extracted notes
    into a single list (used to build the training corpus).
    """
    all_notes = []
    midi_files = glob.glob(os.path.join(folder_path, "*.mid")) + \
                 glob.glob(os.path.join(folder_path, "*.midi"))

    if not midi_files:
        raise FileNotFoundError(
            f"No .mid/.midi files found in '{folder_path}'. "
            "Add some training files there before running preprocess.py."
        )

    for i, file_path in enumerate(midi_files, start=1):
        print(f"[{i}/{len(midi_files)}] Parsing {os.path.basename(file_path)} ...")
        try:
            all_notes.extend(extract_notes_from_file(file_path))
        except Exception as exc:
            print(f"  -> Skipped ({exc})")

    return all_notes


def notes_to_midi_stream(prediction_output, octave_shift=0):
    """
    Convert a list of predicted note/chord strings back into a music21 Stream
    that can be written to a .mid file.

    octave_shift: transpose every note up/down by N octaves (12 semitones
    each) to give the same melody a brighter or darker feel.
    """
    offset = 0
    output_notes = []
    semitone_shift = 12 * octave_shift

    for pattern in prediction_output:
        # Chord
        if ("." in pattern) or pattern.isdigit():
            chord_notes = [note.Note(int(n)) for n in pattern.split(".")]
            for n in chord_notes:
                if semitone_shift:
                    n.pitch.midi += semitone_shift
                n.storedInstrument = _current_instrument()
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # Note
        else:
            new_note = note.Note(pattern)
            if semitone_shift:
                new_note.pitch.midi += semitone_shift
            new_note.offset = offset
            new_note.storedInstrument = _current_instrument()
            output_notes.append(new_note)

        offset += 0.5  # fixed spacing between notes; tweak for different rhythms

    return stream.Stream(output_notes)


def save_midi(prediction_output, output_path, tempo_bpm=None, octave_shift=0):
    """Convert predicted notes to a MIDI stream and write them to disk."""
    midi_stream = notes_to_midi_stream(prediction_output, octave_shift=octave_shift)
    if tempo_bpm:
        midi_stream.insert(0, tempo.MetronomeMark(number=tempo_bpm))
    midi_stream.write("midi", fp=output_path)
    return output_path
