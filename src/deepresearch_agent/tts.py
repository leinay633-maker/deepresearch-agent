from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class TtsProvider(Protocol):
    name: str

    def synthesize_to_wav(self, text: str, path: Path) -> None:
        raise NotImplementedError


@dataclass
class WindowsSapiTtsProvider:
    voice: str = ""
    rate: int = 0
    volume: int = 100
    timeout_seconds: float = 60.0

    name: str = "windows_sapi"

    def synthesize_to_wav(self, text: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "text": text,
                "path": str(path),
                "voice": self.voice,
                "rate": max(-10, min(10, self.rate)),
                "volume": max(0, min(100, self.volume)),
            }
        )
        script = "\n".join(
            [
                "$payload = [Console]::In.ReadToEnd() | ConvertFrom-Json",
                "Add-Type -AssemblyName System.Speech",
                "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer",
                "if ($payload.voice) { $synth.SelectVoice($payload.voice) }",
                "$synth.Rate = [int]$payload.rate",
                "$synth.Volume = [int]$payload.volume",
                "$synth.SetOutputToWaveFile($payload.path)",
                "$synth.Speak($payload.text)",
                "$synth.Dispose()",
            ]
        )
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ],
            input=payload,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if result.returncode != 0:
            error = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"Windows SAPI TTS failed: {error}")
