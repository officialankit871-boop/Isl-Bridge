"""Speech trigger tests; no audio device or speech engine is needed."""

from __future__ import annotations

import unittest

from backend.services.speech_service import SpeechTrigger


class RecordingSpeechService:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)


class TestSpeechTrigger(unittest.TestCase):
    def test_speaks_only_once_for_repeated_stable_class(self) -> None:
        speech = RecordingSpeechService()
        trigger = SpeechTrigger(speech)  # type: ignore[arg-type]

        self.assertTrue(trigger.on_stable_prediction("class_001", "Hello"))
        self.assertFalse(trigger.on_stable_prediction("class_001", "Hello"))
        self.assertFalse(trigger.on_stable_prediction("class_001", "Hello"))
        self.assertEqual(speech.spoken, ["Hello"])

    def test_new_class_can_speak_and_prior_class_can_repeat_after_transition(self) -> None:
        speech = RecordingSpeechService()
        trigger = SpeechTrigger(speech)  # type: ignore[arg-type]

        trigger.on_stable_prediction("class_001", "Hello")
        trigger.on_stable_prediction("class_002", "Thank you")
        trigger.on_stable_prediction("class_001", "Hello")

        self.assertEqual(speech.spoken, ["Hello", "Thank you", "Hello"])

    def test_reset_allows_same_class_to_speak_again(self) -> None:
        speech = RecordingSpeechService()
        trigger = SpeechTrigger(speech)  # type: ignore[arg-type]

        trigger.on_stable_prediction("class_001", "Hello")
        trigger.reset()

        self.assertTrue(trigger.on_stable_prediction("class_001", "Hello"))
        self.assertEqual(speech.spoken, ["Hello", "Hello"])


if __name__ == "__main__":
    unittest.main()
