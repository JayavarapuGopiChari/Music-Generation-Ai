"""
utils/helper.py
----------------
Small, generic helper functions used by app.py (not MIDI-specific).
"""

import os
import time


def allowed_file(filename, allowed_extensions):
    """Return True if filename ends with one of allowed_extensions."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in allowed_extensions
    )


def timestamped_filename(prefix="generated", ext="mid"):
    """Build a unique filename like generated_20260717_153000.mid"""
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{ext}"


def list_output_files(output_dir):
    """Return generated MIDI files sorted newest-first, with basic metadata."""
    if not os.path.isdir(output_dir):
        return []

    files = []
    for fname in os.listdir(output_dir):
        if fname.lower().endswith((".mid", ".midi")):
            fpath = os.path.join(output_dir, fname)
            files.append({
                "name": fname,
                "size_kb": round(os.path.getsize(fpath) / 1024, 1),
                "modified": time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(fpath))
                ),
            })
    files.sort(key=lambda f: f["modified"], reverse=True)
    return files


def model_files_exist(model_weights_path):
    """Check whether a trained model already exists on disk."""
    return os.path.isfile(model_weights_path)


def notes_file_exists(notes_path):
    return os.path.isfile(notes_path)
