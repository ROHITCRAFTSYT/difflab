"""Audio pillar: Mel <-> waveform conversion is shape-correct and reversible.

Uses a synthetic sine wave so the test is fast, hermetic (no dataset download),
and CPU-only. Griffin-Lim reconstruction is lossy, so we check that re-encoding
the reconstructed audio yields a spectrogram strongly correlated with the
original rather than demanding bit-exact recovery.
"""

from __future__ import annotations

import numpy as np

from difflab.data.audio import MelConverter


def _sine(seconds: float, sr: int, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)


def _converter() -> MelConverter:
    return MelConverter(sample_rate=22050, n_fft=1024, hop_length=256, n_mels=64, sample_size=64)


def test_audio_to_image_shape_and_range():
    conv = _converter()
    wav = _sine(conv.slice_samples / conv.sample_rate, conv.sample_rate)
    img = conv.audio_to_image(wav)
    assert img.shape == (1, 64, 64)
    assert img.min() >= -1.0 and img.max() <= 1.0


def test_image_to_audio_returns_waveform():
    conv = _converter()
    wav = _sine(conv.slice_samples / conv.sample_rate, conv.sample_rate)
    img = conv.audio_to_image(wav)
    recon = conv.image_to_audio(img, n_iter=8)
    assert recon.ndim == 1 and len(recon) > 0


def test_roundtrip_spectrogram_correlates():
    conv = _converter()
    wav = _sine(conv.slice_samples / conv.sample_rate, conv.sample_rate)
    img = conv.audio_to_image(wav)
    recon = conv.image_to_audio(img, n_iter=32)
    img2 = conv.audio_to_image(recon)
    a = img.flatten().numpy()
    b = img2.flatten().numpy()
    corr = np.corrcoef(a, b)[0, 1]
    assert corr > 0.8  # spectrogram content preserved through the round-trip


def test_slice_samples_matches_sample_size():
    conv = _converter()
    # hop_length * sample_size samples -> exactly sample_size mel frames.
    assert conv.slice_samples == 256 * 64
