"""
config.py
----------
Central place for all paths and hyperparameters used across the project.
Change values here instead of hunting through every script.
"""

import os

# ---------- Base paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data", "midi_songs")   # put your training .mid files here
MODEL_DIR = os.path.join(BASE_DIR, "model")                # trained model weights (.h5) saved here
NOTES_DIR = os.path.join(BASE_DIR, "notes")                # preprocessed notes (pickle) saved here
OUTPUT_DIR = os.path.join(BASE_DIR, "output")               # generated .mid files saved here

NOTES_FILE = os.path.join(NOTES_DIR, "notes.pkl")
MODEL_WEIGHTS_FILE = os.path.join(MODEL_DIR, "music_model.h5")
MODEL_JSON_FILE = os.path.join(MODEL_DIR, "music_model.json")

# Make sure required folders exist
for _dir in (DATA_DIR, MODEL_DIR, NOTES_DIR, OUTPUT_DIR):
    os.makedirs(_dir, exist_ok=True)

# ---------- Preprocessing / sequence settings ----------
SEQUENCE_LENGTH = 100     # how many previous notes the model looks at to predict the next one

# ---------- Model hyperparameters ----------
LSTM_UNITS = 256
DROPOUT_RATE = 0.3
DENSE_UNITS = 256

# ---------- Training hyperparameters ----------
EPOCHS = 100
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1

# ---------- Generation settings ----------
NOTES_TO_GENERATE = 300   # length of the newly generated piece
GENERATION_TEMPERATURE = 1.0   # >1 = more random/creative, <1 = more conservative

# ---------- Flask settings ----------
FLASK_SECRET_KEY = "change-this-secret-key-in-production"
ALLOWED_UPLOAD_EXTENSIONS = {"mid", "midi"}

# ---------- Generated images ----------
IMAGES_DIR = os.path.join(BASE_DIR, "static", "images", "generated")
os.makedirs(IMAGES_DIR, exist_ok=True)

# ---------- Style presets ----------
# Maps a genre/style tag (shown as pill buttons in the UI) to concrete
# generation parameters for the melody model. These give each style a
# distinct "flavor" (instrument voice + how adventurous the note choices
# are) - the model itself stays the same, only its rendering changes.
STYLE_PRESETS = {
    "Pop":     {"instrument": "Piano",  "temperature": 0.9},
    "Rap":     {"instrument": "Piano",  "temperature": 1.3},
    "Classic": {"instrument": "Piano",  "temperature": 0.5},
    "Folk":    {"instrument": "Guitar", "temperature": 0.7},
    "Rock":    {"instrument": "Guitar", "temperature": 1.1},
    "Diss":    {"instrument": "Trumpet","temperature": 1.4},
    "Metal":   {"instrument": "Guitar", "temperature": 1.5},
    "Country": {"instrument": "Guitar", "temperature": 0.8},
    "Drill":   {"instrument": "Piano",  "temperature": 1.3},
}

# ---------- Emotion presets ----------
# Maps a mood tag to tempo (BPM) and an octave shift, giving the same
# melody a faster/brighter or slower/darker feel.
EMOTION_PRESETS = {
    "Happy":      {"tempo_bpm": 140, "octave_shift": 1},
    "Passionate": {"tempo_bpm": 112, "octave_shift": 0},
    "Warm":       {"tempo_bpm": 92,  "octave_shift": 0},
    "Excited":    {"tempo_bpm": 150, "octave_shift": 1},
    "Thrilled":   {"tempo_bpm": 145, "octave_shift": 1},
    "Healing":    {"tempo_bpm": 70,  "octave_shift": -1},
    "Sad":        {"tempo_bpm": 65,  "octave_shift": -1},
    "Romantic":   {"tempo_bpm": 85,  "octave_shift": 0},
}
