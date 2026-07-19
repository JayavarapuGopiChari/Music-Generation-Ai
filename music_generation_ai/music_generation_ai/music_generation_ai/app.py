"""
app.py
------
Flask web front-end for the Music Generation with AI project.

Routes:
    /            -> redirects to /create (if logged in) or /login
    /signup      -> create an account
    /login       -> log in
    /logout      -> clear session
    /create      -> home page: describe a song, pick style/emotion, generate music + cover art
    /generate    -> POST: runs generate.py logic, redirects to /result
    /generate_image -> POST: calls the free Pollinations text-to-image API
    /result      -> shows the most recently generated MIDI file (playable + downloadable)
    /about       -> project info page
    /download/<filename> -> serve a generated MIDI file for download
    /upload      -> POST: accept new .mid training files into data/midi_songs/

Run:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import traceback
from functools import wraps

from flask import (
    Flask, render_template, request, redirect,
    url_for, send_from_directory, flash, session
)
from werkzeug.utils import secure_filename

import config
import auth
from utils.helper import allowed_file, list_output_files, model_files_exist, notes_file_exists

app = Flask(__name__)
app.secret_key = config.FLASK_SECRET_KEY
auth.init_db()


def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if "username" not in session:
            flash("Please log in to continue.", "error")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapped


@app.route("/")
def root():
    if "username" in session:
        return redirect(url_for("create_page"))
    return redirect(url_for("login"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template("signup.html")

        ok, message = auth.create_user(username, password)
        if ok:
            flash("Account created! Please log in.", "success")
            return redirect(url_for("login"))
        flash(message, "error")

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        if auth.verify_user(username, password):
            session["username"] = username.strip()
            return redirect(url_for("create_page"))
        flash("Invalid username or password.", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))


@app.route("/create")
@login_required
def create_page():
    has_model = model_files_exist(config.MODEL_WEIGHTS_FILE)
    has_notes = notes_file_exists(config.NOTES_FILE)
    training_files = [
        f for f in os.listdir(config.DATA_DIR)
        if f.lower().endswith((".mid", ".midi"))
    ]
    return render_template(
        "create.html",
        has_model=has_model,
        has_notes=has_notes,
        training_file_count=len(training_files),
        styles=list(config.STYLE_PRESETS.keys()),
        emotions=list(config.EMOTION_PRESETS.keys()),
        username=session.get("username")
    )


@app.route("/upload", methods=["POST"])
@login_required
def upload():
    if "midi_files" not in request.files:
        flash("No file part in request.", "error")
        return redirect(url_for("create_page"))

    files = request.files.getlist("midi_files")
    saved = 0
    for file in files:
        if file and file.filename and allowed_file(file.filename, config.ALLOWED_UPLOAD_EXTENSIONS):
            filename = secure_filename(file.filename)
            file.save(os.path.join(config.DATA_DIR, filename))
            saved += 1

    if saved:
        flash(f"Uploaded {saved} MIDI file(s) to data/midi_songs/.", "success")
    else:
        flash("No valid .mid/.midi files were uploaded.", "error")

    return redirect(url_for("create_page"))


@app.route("/generate", methods=["POST"])
@login_required
def generate_route():
    if not model_files_exist(config.MODEL_WEIGHTS_FILE):
        flash("No trained model found. Run 'python train.py' from the terminal first.", "error")
        return redirect(url_for("create_page"))

    try:
        description = request.form.get("description", "").strip()
        style = request.form.get("style") or None
        emotion = request.form.get("emotion") or None
        num_notes = request.form.get("num_notes", type=int) or config.NOTES_TO_GENERATE

        # Imported lazily so the Flask app starts instantly even before
        # TensorFlow / a trained model exist.
        import generate as generate_module
        output_path = generate_module.main(
            num_notes=num_notes,
            style=style,
            emotion=emotion
        )
        filename = os.path.basename(output_path)

        # Also generate matching cover art from the same description/style/emotion.
        image_filename = None
        try:
            import image_gen
            image_filename = image_gen.generate_image(description, style=style, emotion=emotion)
        except Exception:
            traceback.print_exc()
            flash("Music generated, but cover art generation failed (check internet connection).", "error")

        return redirect(url_for("result", filename=filename, image=image_filename))
    except Exception as exc:
        traceback.print_exc()
        flash(f"Generation failed: {exc}", "error")
        return redirect(url_for("create_page"))


@app.route("/generate_image", methods=["POST"])
@login_required
def generate_image_route():
    description = request.form.get("description", "").strip()
    style = request.form.get("style") or None
    emotion = request.form.get("emotion") or None

    try:
        import image_gen
        image_filename = image_gen.generate_image(description, style=style, emotion=emotion)
        return redirect(url_for("result", image=image_filename))
    except Exception as exc:
        traceback.print_exc()
        flash(f"Image generation failed: {exc}", "error")
        return redirect(url_for("create_page"))


@app.route("/result")
@login_required
def result():
    filename = request.args.get("filename")
    image_filename = request.args.get("image")
    all_outputs = list_output_files(config.OUTPUT_DIR)

    if not filename and all_outputs:
        filename = all_outputs[0]["name"]

    return render_template(
        "result.html",
        filename=filename,
        image_filename=image_filename,
        all_outputs=all_outputs
    )


@app.route("/download/<path:filename>")
@login_required
def download(filename):
    return send_from_directory(config.OUTPUT_DIR, filename, as_attachment=True)


@app.route("/play/<path:filename>")
@login_required
def play(filename):
    # Served without as_attachment so <audio>/midi players can stream it
    return send_from_directory(config.OUTPUT_DIR, filename)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
