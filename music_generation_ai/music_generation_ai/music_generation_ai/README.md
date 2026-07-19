# 🎵 Music Generation with AI

Generate original music using an LSTM deep learning model trained on MIDI files
(classical, jazz, etc.), pick a **Style** (Pop/Rap/Classic/Folk/Rock/Diss/Metal/Country/Drill)
and **Emotion** (Happy/Sad/etc.), get matching **AI-generated cover art**, and
play everything back right in the browser — behind a simple login/signup wall.

## Project structure

```
music_generation_ai/
├── app.py               # Flask web app (routes, auth, generation)
├── auth.py              # SQLite-backed signup/login helpers
├── image_gen.py         # calls the free Pollinations.ai text-to-image API
├── config.py            # paths, hyperparameters, style/emotion presets
├── preprocess.py         # MIDI -> note sequences (music21)
├── model.py              # LSTM architecture
├── train.py               # trains the model
├── generate.py             # generates new music from the trained model
├── requirements.txt
├── .gitignore
├── data/midi_songs/       # put your training .mid files here
├── model/                 # trained weights saved here
├── notes/                 # preprocessed note data saved here
├── output/                # generated .mid files saved here
├── templates/             # login.html, signup.html, create.html, result.html, about.html
├── static/css, js, images/generated (AI cover art saved here)
└── utils/
    ├── midi_utils.py
    └── helper.py
```

## 1. Setup (run once)

Open a terminal in VS Code inside the `music_generation_ai` folder, then:

```bash
# create & activate a virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

> If you're on **Windows ARM64** and `pip install tensorflow` fails with
> "No matching distribution found", use WSL2 (native Linux ARM64 wheels) —
> see the troubleshooting note at the bottom.

## 2. Add training data

Copy some `.mid` / `.midi` files (classical, jazz, whatever style you want to learn)
into:

```
data/midi_songs/
```

A good free source: search for "public domain MIDI classical piano dataset"
(e.g. the classic Nottingham or Lakh MIDI datasets, or your own MIDI collection).

## 3. Run the pipeline (terminal commands, in order)

```bash
# Step 1: extract notes/chords from all MIDI files
python preprocess.py

# Step 2: train the LSTM model (can take a while depending on data size/CPU/GPU)
python train.py

# Step 3 (optional, CLI-only test): generate music without the web UI
python generate.py
```

After training, `model/music_model.h5` and `notes/notes.pkl` +
`notes/note_mapping.json` will exist — the web app checks for these
before allowing you to click "Generate".

## 4. Run the web app

```bash
python app.py
```

Then open your browser at:

```
http://127.0.0.1:5000
```

You'll land on the **login page** first.
- Click **Sign up** to create an account (username + password, stored locally
  in `users.db`, passwords are hashed — never stored in plain text).
- Log in, and you'll be taken to the **Create** page.

## 5. Using the app

On the Create page:
- Type a **description** of the song you want (used to generate matching cover art).
- Optionally tap one **Style** pill (Pop, Rap, Classic, Folk, Rock, Diss, Metal,
  Country, Drill) — this picks the instrument voice and how adventurous the
  melody's note choices are.
- Optionally tap one **Emotion** pill (Happy, Sad, Excited, Healing, etc.) —
  this sets the tempo (BPM) and shifts the melody up/down an octave for a
  brighter or darker feel.
- Click **✨ Generate**. This:
  1. Generates a new melody with the trained LSTM, flavored by your Style/Emotion choice
  2. Calls the free Pollinations.ai API to generate real AI cover art from your description
- On the **Result** page: listen in-browser (piano-roll visualizer), see the
  cover art, or download the `.mid` file. Requires internet access for the
  cover-art step (Pollinations.ai) and for the in-browser MIDI player's soundfont.

> **Honest scope note:** the melody model responds to your Style/Emotion tags,
> not the literal words in your description — true text-conditioned music
> generation (like Suno) needs far larger pretrained models than an LSTM
> trained from scratch on a small MIDI dataset. The **cover art**, however,
> is genuine text-to-image AI generation from your full description.

## Notes on tuning

All key knobs live in `config.py`:
- `SEQUENCE_LENGTH` — how much context the model uses per prediction
- `LSTM_UNITS`, `DROPOUT_RATE` — model capacity/regularization
- `EPOCHS`, `BATCH_SIZE` — training duration
- `NOTES_TO_GENERATE`, `GENERATION_TEMPERATURE` — default length & randomness
- `STYLE_PRESETS` — instrument + temperature per style tag
- `EMOTION_PRESETS` — tempo (BPM) + octave shift per emotion tag

## Troubleshooting

- **"No .mid/.midi files found"** → add files to `data/midi_songs/` first.
- **Training is slow** → reduce `EPOCHS`, use fewer MIDI files, or run on a
  machine with a GPU + `tensorflow` GPU support.
- **`tensorflow` install issues on Windows ARM64** → TensorFlow has no native
  Windows ARM64 wheels. Use WSL2 instead:
  ```bash
  wsl --install
  ```
  then inside the Ubuntu terminal, copy the project into the Linux filesystem
  and repeat steps 1–4 there — `pip` will find proper `manylinux_aarch64` wheels.
- **Cover art doesn't appear** → check your internet connection; the app
  calls out to `image.pollinations.ai` at generation time.
- **Forgot your password** → delete `users.db` and sign up again (this is a
  local single-file database, fine for personal/dev use).
