"""
image_gen.py
------------
Real text-to-image generation using Pollinations.ai's free public API
(no API key required). The description + selected style/emotion are
combined into a prompt, sent to the service, and the returned image is
saved into static/images/generated/.

Requires internet access at runtime (this calls out over HTTPS).
"""

import os
import time
import urllib.parse

import requests

import config

POLLINATIONS_URL = "https://image.pollinations.ai/prompt/{prompt}"

# A little extra flavor text per style so the album-art image visually
# matches the chosen genre, even if the user's description is short.
STYLE_IMAGE_HINTS = {
    "Pop":     "vibrant colorful pop album cover art",
    "Rap":     "urban street style hip hop album cover art",
    "Classic": "elegant classical concert hall album cover art",
    "Folk":    "rustic acoustic folk album cover art",
    "Rock":    "gritty energetic rock band album cover art",
    "Diss":    "bold intense confrontational album cover art",
    "Metal":   "dark heavy metal album cover art",
    "Country": "warm rustic country music album cover art",
    "Drill":   "moody dark urban drill music album cover art",
}

EMOTION_IMAGE_HINTS = {
    "Happy": "bright cheerful colors",
    "Passionate": "warm intense fiery tones",
    "Warm": "soft golden lighting",
    "Excited": "dynamic energetic composition",
    "Thrilled": "vivid high energy visuals",
    "Healing": "calm peaceful serene atmosphere",
    "Sad": "moody blue melancholic tones",
    "Romantic": "soft dreamy romantic lighting",
}


def build_prompt(description, style=None, emotion=None):
    parts = [description.strip()] if description and description.strip() else []
    if style in STYLE_IMAGE_HINTS:
        parts.append(STYLE_IMAGE_HINTS[style])
    if emotion in EMOTION_IMAGE_HINTS:
        parts.append(EMOTION_IMAGE_HINTS[emotion])
    if not parts:
        parts = ["abstract music album cover art"]
    return ", ".join(parts)


def generate_image(description, style=None, emotion=None, output_filename=None):
    """
    Calls the free Pollinations.ai image API and saves the result locally.
    Returns the filename (relative to static/images/generated/).
    """
    prompt = build_prompt(description, style, emotion)
    encoded_prompt = urllib.parse.quote(prompt)
    url = POLLINATIONS_URL.format(prompt=encoded_prompt) + "?width=768&height=768&nologo=true"

    response = requests.get(url, timeout=90)
    response.raise_for_status()

    if output_filename is None:
        output_filename = f"cover_{int(time.time())}.png"

    output_path = os.path.join(config.IMAGES_DIR, output_filename)
    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_filename
