"""Optional, local Windows text-to-speech for stabilized predictions."""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading

logger = logging.getLogger(__name__)


class SpeechService:
    """Speak queued text with Windows SAPI, without blocking video inference."""

    def __init__(self) -> None:
        executable = shutil.which("powershell") or shutil.which("pwsh")
        if executable is None:
            raise RuntimeError("Offline speech requires Windows PowerShell and SAPI.")
        check = subprocess.run(
            [
                executable,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "$ErrorActionPreference = 'Stop'; "
                "try { Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "if ($s.GetInstalledVoices().Count -eq 0) { exit 2 }; "
                "$s.Dispose() } catch { exit 3 }",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if check.returncode != 0:
            raise RuntimeError(
                "Windows SAPI is available, but no local speech voice could be initialized."
            )
        self._executable = executable
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True, name="isl-tts")
        self._worker.start()

    def speak(self, text: str) -> None:
        """Queue non-empty text for local speech."""
        sentence = text.strip()
        if sentence:
            self._queue.put(sentence)

    def close(self) -> None:
        """Drain queued speech and stop the worker."""
        if self._worker.is_alive():
            self._queue.put(None)
            self._worker.join(timeout=5)

    def _run(self) -> None:
        while True:
            sentence = self._queue.get()
            try:
                if sentence is None:
                    return
                # Pass text as an argument instead of interpolating it into PowerShell.
                script = (
                    "Add-Type -AssemblyName System.Speech; "
                    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    "$s.Speak($args[0]); $s.Dispose()"
                )
                subprocess.run(
                    [
                        self._executable,
                        "-NoProfile",
                        "-NonInteractive",
                        "-Command",
                        script,
                        sentence,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                logger.warning("Local speech failed: %s", exc)
            finally:
                self._queue.task_done()


class SpeechTrigger:
    """Emit speech once per newly stabilized class until another class stabilizes."""

    def __init__(self, speech_service: SpeechService) -> None:
        self._speech_service = speech_service
        self._last_class_id: str | None = None

    def on_stable_prediction(self, class_id: str, sentence: str) -> bool:
        """Speak a new stable class and return whether speech was queued."""
        if class_id == self._last_class_id:
            return False
        self._last_class_id = class_id
        self._speech_service.speak(sentence)
        return True

    def reset(self) -> None:
        """Allow the current class to be spoken again after a user reset."""
        self._last_class_id = None
