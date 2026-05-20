"""Utilities for calling the Watson emotion detection service."""

from __future__ import annotations

from typing import Any

import requests

URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock",
}


def _extract_emotions(response_json: dict[str, Any]) -> dict[str, float]:
    """Extract the emotion scores from the service response."""
    return response_json["emotionPredictions"][0]["emotion"]


def emotion_detector(text_to_analyze: str) -> dict[str, float | str] | None:
    """Analyze text and return emotion scores plus the dominant emotion.

    If the remote service returns status code 400, ``None`` is returned.
    """
    payload = {"raw_document": {"text": text_to_analyze}}
    response = requests.post(
        url=URL,
        headers=HEADERS,
        json=payload,
        timeout=10,
    )

    if response.status_code == 400:
        return None

    response.raise_for_status()
    emotions = _extract_emotions(response.json())
    dominant_emotion = max(emotions, key=emotions.get)

    return {
        "anger": emotions["anger"],
        "disgust": emotions["disgust"],
        "fear": emotions["fear"],
        "joy": emotions["joy"],
        "sadness": emotions["sadness"],
        "dominant_emotion": dominant_emotion,
    }
