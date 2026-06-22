# Audio diffusion

Audio diffusion here is **spectrogram-domain**: we render a fixed-size Mel
spectrogram as a single-channel image, train an ordinary image diffusion UNet on
it, and invert generated spectrograms back to audio with Griffin-Lim.

## The bridge: `MelConverter`

```
waveform --audio_to_image--> mel image in [-1, 1] (1, S, S)
mel image --image_to_audio--> waveform (Griffin-Lim)
```

The converter is sized so `hop_length * sample_size` audio samples map to exactly
`sample_size` spectrogram frames, producing a square image. dB values are
normalized to $[-1, 1]$ for the diffusion model and de-normalized on the way out.

```python
from difflab.data.audio import MelConverter

conv = MelConverter(sample_rate=22050, n_fft=2048, hop_length=512,
                    n_mels=128, sample_size=128)
img = conv.audio_to_image(waveform)     # (1, 128, 128) in [-1, 1]
audio = conv.image_to_audio(img)        # 1-D waveform
```

## Run it

Smoke test (CPU):

```bash
difflab train -c configs/audio_diffusion_smoke.yaml
```

Full run (GPU) on a music dataset:

```bash
difflab train -c configs/audio_diffusion.yaml
```

The audio column is loaded undecoded and decoded with `soundfile`, so no
`torchcodec` runtime dependency is needed. Large datasets can be streamed
(`data.streaming: true` with `data.max_samples`) to avoid a full download.

Sampling produces spectrogram images (via the shared `Trainer`); convert them to
audio with `MelConverter.image_to_audio`.

## Notes & limitations

- Griffin-Lim reconstructs phase from magnitude only, so audio is intelligible
  but not hi-fi. A neural vocoder would improve fidelity — out of scope here.
- Quality scales with mel resolution (`sample_size`, `n_mels`) and training
  budget. Start at 128×128.

See the notebook [`04_audio_diffusion.ipynb`](https://github.com/ROHITCRAFTSYT/difflab/blob/main/notebooks/04_audio_diffusion.ipynb).
