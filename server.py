"""Flask server for the emotion detector application."""

from __future__ import annotations

from flask import Flask, render_template, request
from requests import exceptions as requests_exceptions

from EmotionDetection import emotion_detector

app = Flask("Emotion Detector")


@app.route("/")
def render_index_page() -> str:
    """Render the main application page."""
    return render_template("index.html")


@app.route("/emotionDetector")
def sent_analyzer() -> str:
    """Analyze user text and return a Coursera-compatible response."""
    text_to_analyze = request.args.get("textToAnalyze", "")
    if not text_to_analyze.strip():
        return "Invalid text! Please try again!"

    try:
        response = emotion_detector(text_to_analyze)
    except requests_exceptions.RequestException:
        return "Unable to process the request right now. Please try again later."

    if response is None:
        return "Invalid text! Please try again!"

    return (
        "For the given statement, the system response is "
        f"'anger': {response['anger']}, "
        f"'disgust': {response['disgust']}, "
        f"'fear': {response['fear']}, "
        f"'joy': {response['joy']} and "
        f"'sadness': {response['sadness']}. "
        f"The dominant emotion is {response['dominant_emotion']}."
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
