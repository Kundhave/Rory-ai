"""Device I/O for voice: record to WAV bytes, play WAV bytes.

WAV encode/decode and resampling are kept as pure functions, separate from
the device calls, so the format logic is testable without a real audio
device.
"""
from __future__ import annotations

import io

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000  # what Sarvam STT expects: 16kHz mono PCM
CHANNELS = 1
DTYPE = "int16"


def to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, subtype="PCM_16", format="WAV")
    return buffer.getvalue()


def from_wav_bytes(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    samples, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="int16")
    return samples, sample_rate


def resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Linear interpolation. There's no scipy in this project's dependency
    list — linear resampling is lower quality than a sinc-based resampler,
    but it needs nothing beyond numpy and is more than sufficient for speech
    going into an STT model."""
    if from_rate == to_rate or len(samples) == 0:
        return samples
    duration = len(samples) / from_rate
    old_times = np.linspace(0, duration, num=len(samples), endpoint=False)
    new_len = max(1, int(round(duration * to_rate)))
    new_times = np.linspace(0, duration, num=new_len, endpoint=False)
    resampled = np.interp(new_times, old_times, samples.astype(np.float64))
    return resampled.astype(np.int16)


def play(wav_bytes: bytes) -> None:
    samples, sample_rate = from_wav_bytes(wav_bytes)
    sd.play(samples, sample_rate)
    sd.wait()


class Recorder:
    """Click-to-start, click-to-stop capture. No VAD, no endpointing, no
    streaming — the caller decides exactly when to stop by calling stop().

    Opens the input stream at the device's own native sample rate rather
    than assuming 16kHz — some ALSA hardware devices reject arbitrary rates
    outright — then resamples the captured audio down to 16kHz afterward.
    """

    def __init__(self, device: int | None = None) -> None:
        self._device = device
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._native_rate = int(sd.query_devices(device, "input")["default_samplerate"])

    def start(self) -> None:
        self._frames = []

        def callback(indata, frames, time_info, status) -> None:
            self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self._native_rate,
            channels=CHANNELS,
            dtype=DTYPE,
            device=self._device,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> bytes:
        self._stream.stop()
        self._stream.close()
        self._stream = None

        if self._frames:
            captured = np.concatenate(self._frames).reshape(-1)
        else:
            captured = np.zeros((0,), dtype=np.int16)

        resampled = resample(captured, self._native_rate, SAMPLE_RATE)
        return to_wav_bytes(resampled, SAMPLE_RATE)
