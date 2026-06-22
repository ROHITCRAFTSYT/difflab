"""Audio <-> Mel-spectrogram conversion and audio dataset loading.

Spectrogram-domain audio diffusion treats a fixed-size **Mel spectrogram** as a
single-channel image. We train an ordinary image diffusion UNet on these
spectrogram images; to listen to a sample we invert the spectrogram back to a
waveform with the Griffin-Lim algorithm.

``MelConverter`` is the bridge:

    waveform --audio_to_image--> mel image in [-1, 1] (1, S, S)
    mel image --image_to_audio--> waveform (Griffin-Lim)

The converter is sized so a slice of ``hop_length * sample_size`` audio samples
maps to exactly ``sample_size`` spectrogram frames, giving a square image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from difflab.config import DataConfig

_TOP_DB = 80.0  # dynamic range used for dB normalization


@dataclass
class MelConverter:
    """Reversible waveform <-> normalized Mel-spectrogram-image converter."""

    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    sample_size: int = 128  # output image is (1, sample_size, sample_size)

    @property
    def slice_samples(self) -> int:
        """Number of audio samples that map to ``sample_size`` mel frames."""
        return self.hop_length * self.sample_size

    def audio_to_image(self, waveform: np.ndarray) -> torch.Tensor:
        """Convert a 1-D waveform to a (1, S, S) mel image in [-1, 1]."""
        import librosa

        waveform = np.asarray(waveform, dtype=np.float32)
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )
        db = librosa.power_to_db(mel, ref=np.max, top_db=_TOP_DB)  # [-_TOP_DB, 0]
        img = (db + _TOP_DB / 2) / (_TOP_DB / 2)  # -> roughly [-1, 1]
        img = self._fit(img)
        return torch.from_numpy(img).unsqueeze(0).float().clamp(-1, 1)

    def image_to_audio(self, image: torch.Tensor, n_iter: int = 32) -> np.ndarray:
        """Invert a (1, S, S) or (S, S) mel image back to a waveform."""
        import librosa

        arr = image.detach().cpu().squeeze().numpy().astype(np.float32)
        db = arr * (_TOP_DB / 2) - _TOP_DB / 2  # undo normalization -> [-_TOP_DB, 0]
        power = librosa.db_to_power(db)
        waveform = librosa.feature.inverse.mel_to_audio(
            power,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_iter=n_iter,
        )
        return waveform

    def _fit(self, mel: np.ndarray) -> np.ndarray:
        """Pad/crop a (n_mels, T) array to (sample_size, sample_size)."""
        n_mels, t = mel.shape
        target = self.sample_size
        # Frequency axis.
        if n_mels < target:
            mel = np.pad(mel, ((0, target - n_mels), (0, 0)), constant_values=-1.0)
        else:
            mel = mel[:target]
        # Time axis.
        if t < target:
            mel = np.pad(mel, ((0, 0), (0, target - t)), constant_values=-1.0)
        else:
            mel = mel[:, :target]
        return mel


def make_mel_converter(cfg: DataConfig, sample_size: int) -> MelConverter:
    return MelConverter(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        sample_size=sample_size,
    )


def build_audio_dataloader(
    cfg: DataConfig,
    sample_size: int,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
):
    """Build a ``DataLoader`` yielding ``{"images": (B, 1, S, S)}`` mel images.

    The audio column is loaded *undecoded* (raw bytes) and decoded here with
    ``soundfile``. This avoids the ``torchcodec`` runtime dependency that recent
    ``datasets`` versions require for automatic audio decoding, and works with
    any parquet/file-based audio dataset on the Hub.
    """
    import io

    import librosa
    import soundfile as sf
    from datasets import Audio, load_dataset
    from torch.utils.data import DataLoader

    if cfg.streaming:
        if cfg.max_samples is None:
            raise ValueError("data.streaming=true requires data.max_samples to be set.")
        # Pull exactly max_samples examples without downloading the whole dataset.
        stream = load_dataset(cfg.dataset, split=cfg.split, streaming=True)
        stream = stream.cast_column(cfg.audio_column, Audio(decode=False))
        ds = [ex for _, ex in zip(range(cfg.max_samples), stream, strict=False)]
    else:
        ds = load_dataset(cfg.dataset, split=cfg.split)
        if cfg.max_samples is not None:
            ds = ds.select(range(min(cfg.max_samples, len(ds))))
        # Disable automatic decoding so we receive {"bytes": ..., "path": ...}.
        ds = ds.cast_column(cfg.audio_column, Audio(decode=False))

    converter = make_mel_converter(cfg, sample_size)
    audio_col = cfg.audio_column
    slice_len = converter.slice_samples

    def _decode(entry) -> tuple[np.ndarray, int]:
        if entry.get("bytes"):
            wav, sr = sf.read(io.BytesIO(entry["bytes"]), dtype="float32", always_2d=False)
        else:  # fall back to a file path
            wav, sr = sf.read(entry["path"], dtype="float32", always_2d=False)
        if wav.ndim > 1:  # stereo -> mono
            wav = wav.mean(axis=1)
        return wav.astype(np.float32), sr

    def to_waveform(item) -> np.ndarray:
        wav, sr = _decode(item[audio_col])
        if sr != cfg.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=cfg.sample_rate)
        if len(wav) < slice_len:
            wav = np.pad(wav, (0, slice_len - len(wav)))
        else:
            wav = wav[:slice_len]
        return wav

    def collate(batch):
        images = torch.stack([converter.audio_to_image(to_waveform(it)) for it in batch])
        return {"images": images}

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=True,
    )
