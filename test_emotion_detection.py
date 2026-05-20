"""Unit tests for the emotion detection package."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from EmotionDetection import emotion_detector


def _mock_response(
    scores: dict[str, float],
    status_code: int = 200,
) -> Mock:
    """Create a mocked requests response object."""
    response = Mock()
    response.status_code = status_code
    response.json.return_value = {
        "emotionPredictions": [
            {
                "emotion": scores
            }
        ]
    }
    response.raise_for_status.return_value = None
    return response


class TestEmotionDetector(unittest.TestCase):
    """Test suite for emotion detection behavior."""

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_joy_dominant(self, mock_post: Mock) -> None:
        """Joy should be detected as dominant."""
        mock_post.return_value = _mock_response(
            {"anger": 0.01, "disgust": 0.02, "fear": 0.03, "joy": 0.91, "sadness": 0.03}
        )
        result = emotion_detector("I am glad this happened")
        self.assertIsNotNone(result)
        self.assertEqual(result["dominant_emotion"], "joy")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_anger_dominant(self, mock_post: Mock) -> None:
        """Anger should be detected as dominant."""
        mock_post.return_value = _mock_response(
            {"anger": 0.85, "disgust": 0.03, "fear": 0.04, "joy": 0.03, "sadness": 0.05}
        )
        result = emotion_detector("I am really mad about this")
        self.assertIsNotNone(result)
        self.assertEqual(result["dominant_emotion"], "anger")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_disgust_dominant(self, mock_post: Mock) -> None:
        """Disgust should be detected as dominant."""
        mock_post.return_value = _mock_response(
            {"anger": 0.04, "disgust": 0.87, "fear": 0.03, "joy": 0.02, "sadness": 0.04}
        )
        result = emotion_detector("I feel disgusted just hearing about this")
        self.assertIsNotNone(result)
        self.assertEqual(result["dominant_emotion"], "disgust")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_fear_dominant(self, mock_post: Mock) -> None:
        """Fear should be detected as dominant."""
        mock_post.return_value = _mock_response(
            {"anger": 0.05, "disgust": 0.04, "fear": 0.82, "joy": 0.03, "sadness": 0.06}
        )
        result = emotion_detector("I am worried something bad will happen")
        self.assertIsNotNone(result)
        self.assertEqual(result["dominant_emotion"], "fear")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_sadness_dominant(self, mock_post: Mock) -> None:
        """Sadness should be detected as dominant."""
        mock_post.return_value = _mock_response(
            {"anger": 0.05, "disgust": 0.03, "fear": 0.04, "joy": 0.02, "sadness": 0.86}
        )
        result = emotion_detector("I am feeling very sad today")
        self.assertIsNotNone(result)
        self.assertEqual(result["dominant_emotion"], "sadness")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_blank_text_returns_none_on_400(self, mock_post: Mock) -> None:
        """A 400 response should map to None."""
        mock_post.return_value = _mock_response(
            {"anger": 0.0, "disgust": 0.0, "fear": 0.0, "joy": 0.0, "sadness": 0.0},
            status_code=400,
        )
        result = emotion_detector("")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
