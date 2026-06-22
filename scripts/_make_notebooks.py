"""Generate the four demo notebooks as valid .ipynb files.

Run once from the repo root:  python scripts/_make_notebooks.py
This keeps the notebook *source* in reviewable Python rather than raw JSON.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = "ROHITCRAFTSYT/difflab"
NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def badge(nb_name: str) -> str:
    url = f"https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{nb_name}"
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


INSTALL = (
    "# === difflab bootstrap — makes `import difflab` and the configs work anywhere ===\n"
    "# Works on Colab/Kaggle (no checkout) and on a local clone. Three install paths,\n"
    "# tried in order: already-installed -> GitHub clone -> uploaded difflab.zip.\n"
    "import glob, os, pathlib, subprocess, sys\n"
    "\n"
    "def _have_difflab():\n"
    "    try:\n"
    "        import difflab  # noqa: F401\n"
    "        return True\n"
    "    except ModuleNotFoundError:\n"
    "        return False\n"
    "\n"
    "ROOT = None\n"
    "if not _have_difflab():\n"
    "    # (1) If you pushed difflab to GitHub, set DIFFLAB_REPO and it clones+installs:\n"
    "    REPO_URL = os.environ.get('DIFFLAB_REPO', '')  # e.g. 'https://github.com/you/difflab.git'\n"
    "    if REPO_URL:\n"
    "        subprocess.run(['git', 'clone', '-q', REPO_URL, '/content/difflab'], check=False)\n"
    "        ROOT = '/content/difflab'\n"
    "    else:\n"
    "        # (2) Else upload difflab.zip via the Files panel (left sidebar), then run this.\n"
    "        hits = glob.glob('/content/difflab.zip') or glob.glob('difflab.zip')\n"
    "        if not hits:\n"
    "            from google.colab import files  # type: ignore\n"
    "            print('Select difflab.zip to upload...')\n"
    "            files.upload()\n"
    "            hits = glob.glob('difflab.zip') or glob.glob('/content/difflab.zip')\n"
    "        subprocess.run(['unzip', '-q', '-o', hits[0], '-d', '/content/difflab_pkg'], check=False)\n"
    "        found = glob.glob('/content/difflab_pkg/**/pyproject.toml', recursive=True)\n"
    "        ROOT = str(pathlib.Path(found[0]).parent) if found else '/content/difflab_pkg'\n"
    "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', ROOT], check=False)\n"
    "\n"
    "# Resolve the repo root so we can find configs/ whether installed remotely or locally.\n"
    "if ROOT is None:\n"
    "    here = pathlib.Path.cwd()\n"
    "    ROOT = str(here.parent if here.name == 'notebooks' else here)\n"
    "CONFIGS = str(pathlib.Path(ROOT) / 'configs')\n"
    "\n"
    "import difflab, torch\n"
    "print('difflab', difflab.__version__, '| configs:', CONFIGS)\n"
    "print('CUDA available:', torch.cuda.is_available())"
)

SMOKE_NOTE = (
    "> **Compute note.** The cell below runs the *smoke* config — a tiny model and "
    "2 optimizer steps — so it finishes in seconds on CPU and proves the pipeline. "
    "For real results, switch to the full config on a GPU runtime (Runtime → Change "
    "runtime type → GPU)."
)


def viz_cell(checkpoint: str, conditioned: bool) -> str:
    labels = (
        "labels = torch.arange(8, device=device) % cfg.model.num_classes\n"
        if conditioned
        else "labels = None\n"
    )
    return (
        "import torch\n"
        "from diffusers import UNet2DModel\n"
        "from difflab.config import load_config\n"
        "from difflab.models import build_scheduler\n"
        "from difflab.sampling import ddim_sample\n"
        "from difflab.utils import get_device, make_image_grid, tensor_to_pil\n\n"
        f"cfg = load_config(f'{{CONFIGS}}/{checkpoint}')\n"
        "device = get_device()\n"
        "model = UNet2DModel.from_pretrained(result.checkpoint_dir).to(device)\n"
        "scheduler = build_scheduler(cfg.scheduler, kind='ddim')\n"
        f"{labels}"
        "imgs = ddim_sample(model, scheduler, num_samples=8, num_inference_steps=20,\n"
        "                   class_labels=labels, device=device)\n"
        "grid = make_image_grid(tensor_to_pil(imgs))\n"
        "grid"
    )


def save(nb, name):
    nbf.write(nb, NB_DIR / name)
    print("wrote", name)


def nb_finetune():
    name = "01_finetuning.ipynb"
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(f"# 01 · Fine-tuning a pretrained diffusion model\n\n{badge(name)}\n\n"
                          "Fine-tune a pretrained DDPM UNet on a new image dataset (butterflies)."),
        new_code_cell(INSTALL),
        new_markdown_cell(SMOKE_NOTE),
        new_code_cell(
            "from difflab.config import load_config\n"
            "from difflab.training import finetune\n\n"
            "cfg = load_config(f'{CONFIGS}/finetune_butterflies_smoke.yaml')  # -> _butterflies.yaml on GPU\n"
            "result = finetune.run(cfg)\n"
            "result"
        ),
        new_markdown_cell("## Sample from the fine-tuned model"),
        new_code_cell(viz_cell("finetune_butterflies_smoke.yaml", conditioned=False)),
        new_markdown_cell(
            "## Publish to the Hub (optional)\n"
            "Set `hub.push_to_hub: true` and `hub.repo_id` in the config and export "
            "`HF_TOKEN`, then:"),
        new_code_cell(
            "import os\n"
            "from difflab.hub import push_model_to_hub\n"
            "# os.environ['HF_TOKEN'] = '...'\n"
            "# cfg.hub.push_to_hub = True; cfg.hub.repo_id = 'your-name/difflab-butterflies'\n"
            "push_model_to_hub(result.checkpoint_dir, cfg)"
        ),
    ]
    save(nb, name)


def nb_class_conditioned():
    name = "02_class_conditioned.ipynb"
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(f"# 02 · Class-conditioned generation\n\n{badge(name)}\n\n"
                          "Train a label-conditioned UNet on Fashion-MNIST and sample any class."),
        new_code_cell(INSTALL),
        new_markdown_cell(SMOKE_NOTE),
        new_code_cell(
            "from difflab.config import load_config\n"
            "from difflab.training import class_conditioned\n\n"
            "cfg = load_config(f'{CONFIGS}/class_conditioned_fashionmnist_smoke.yaml')\n"
            "result = class_conditioned.run(cfg)\n"
            "result"
        ),
        new_markdown_cell("## Sample one image per class (0-9)"),
        new_code_cell(viz_cell("class_conditioned_fashionmnist_smoke.yaml", conditioned=True)),
    ]
    save(nb, name)


def nb_inversion():
    name = "03_ddim_inversion.ipynb"
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(f"# 03 · DDIM inversion & prompt editing\n\n{badge(name)}\n\n"
                          "Invert a real image to its latent with Stable Diffusion, then edit it "
                          "by changing the prompt. **Needs a GPU runtime** to download/run SD."),
        new_code_cell(INSTALL),
        new_markdown_cell(
            "## The inversion math (CPU, no download)\n"
            "First verify the deterministic invert→sample round-trip on a tiny model — the same "
            "code path used for Stable Diffusion."),
        new_code_cell(
            "import torch\n"
            "from difflab.config import ModelConfig, SchedulerConfig\n"
            "from difflab.models import build_unet, build_scheduler\n"
            "from difflab.inversion import ddim_invert, ddim_sample_latents\n\n"
            "unet = build_unet(ModelConfig(sample_size=8, in_channels=1, out_channels=1,\n"
            "    layers_per_block=1, norm_num_groups=8, block_out_channels=(8,16),\n"
            "    down_block_types=('DownBlock2D','DownBlock2D'),\n"
            "    up_block_types=('UpBlock2D','UpBlock2D')))\n"
            "sched = build_scheduler(SchedulerConfig(num_train_timesteps=50), 'ddim',\n"
            "    clip_sample=False, set_alpha_to_one=False)\n"
            "x0 = torch.randn(1,1,8,8)\n"
            "const = torch.randn(1,1,8,8)\n"
            "eps = lambda l,t: const.expand_as(l)\n"
            "noise = ddim_invert(eps, sched, x0, 50)\n"
            "recon = ddim_sample_latents(eps, sched, noise, 50)\n"
            "print('round-trip rel error:', (recon-x0).norm().item()/x0.norm().item())"
        ),
        new_markdown_cell(
            "## Real-image editing with Stable Diffusion (GPU)\n"
            "Uncomment to invert a real image and swap its content via the target prompt."),
        new_code_cell(
            "# from PIL import Image\n"
            "# from difflab.inversion import DDIMInverter\n"
            "# inv = DDIMInverter.from_pretrained('runwayml/stable-diffusion-v1-5', device='cuda',\n"
            "#                                    num_inference_steps=50)\n"
            "# image = Image.open('cat.png').convert('RGB').resize((512,512))\n"
            "# latents = inv.invert(image, prompt='a photo of a cat')\n"
            "# edited  = inv.sample(latents, prompt='a photo of a dog')\n"
            "# edited[0]"
        ),
    ]
    save(nb, name)


def nb_audio():
    name = "04_audio_diffusion.ipynb"
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(f"# 04 · Audio diffusion (Mel spectrograms)\n\n{badge(name)}\n\n"
                          "Diffuse Mel-spectrogram images, then reconstruct audio with Griffin-Lim."),
        new_code_cell(INSTALL + "\n# Audio extras:\n# !pip install -q librosa soundfile"),
        new_markdown_cell("## Inspect the audio↔mel bridge (CPU, synthetic sine)"),
        new_code_cell(
            "import numpy as np\n"
            "from difflab.data.audio import MelConverter\n\n"
            "conv = MelConverter(sample_rate=22050, n_fft=1024, hop_length=256, n_mels=64, sample_size=64)\n"
            "t = np.linspace(0, conv.slice_samples/conv.sample_rate, conv.slice_samples, endpoint=False)\n"
            "wav = 0.5*np.sin(2*np.pi*440*t).astype('float32')\n"
            "img = conv.audio_to_image(wav)\n"
            "print('mel image:', tuple(img.shape), 'range', float(img.min()), float(img.max()))\n"
            "recon = conv.image_to_audio(img, n_iter=32)\n"
            "print('reconstructed waveform samples:', recon.shape)"
        ),
        new_markdown_cell(SMOKE_NOTE + "\n\n## Train the audio diffusion model"),
        new_code_cell(
            "from difflab.config import load_config\n"
            "from difflab.training import audio\n\n"
            "cfg = load_config(f'{CONFIGS}/audio_diffusion_smoke.yaml')  # -> audio_diffusion.yaml on GPU\n"
            "result = audio.run(cfg)\n"
            "result"
        ),
        new_markdown_cell("## Sample a spectrogram and turn it into audio"),
        new_code_cell(
            "import torch\n"
            "from diffusers import UNet2DModel\n"
            "from difflab.models import build_scheduler\n"
            "from difflab.sampling import ddim_sample\n"
            "from difflab.data.audio import make_mel_converter\n"
            "from difflab.utils import get_device\n\n"
            "device = get_device()\n"
            "model = UNet2DModel.from_pretrained(result.checkpoint_dir).to(device)\n"
            "sched = build_scheduler(cfg.scheduler, 'ddim')\n"
            "mel = ddim_sample(model, sched, 1, num_inference_steps=10, device=device)\n"
            "conv = make_mel_converter(cfg.data, cfg.model.sample_size)\n"
            "wav = conv.image_to_audio(mel[0], n_iter=16)\n"
            "print('generated waveform samples:', wav.shape)\n"
            "# from IPython.display import Audio; Audio(wav, rate=cfg.data.sample_rate)"
        ),
    ]
    save(nb, name)


if __name__ == "__main__":
    NB_DIR.mkdir(exist_ok=True)
    nb_finetune()
    nb_class_conditioned()
    nb_inversion()
    nb_audio()
