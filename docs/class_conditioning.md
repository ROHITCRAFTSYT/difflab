# Class-conditioned generation

A class-conditioned model learns $\epsilon_\theta(x_t, t, y)$ — the noise
prediction depends on a class label $y$ — so you can generate a chosen category
on demand.

## How it works

Setting `model.num_classes > 0` gives the `UNet2DModel` a learned class-embedding
table (`num_class_embeds`). The label embedding is added to the timestep
embedding inside the UNet, so conditioning costs almost nothing. The dataset must
expose an integer label column (`data.label_column`).

Entry point: `difflab.training.class_conditioned.run`.

## Run it

Smoke test (CPU):

```bash
difflab train -c configs/class_conditioned_fashionmnist_smoke.yaml
```

Full run (GPU) on Fashion-MNIST (10 classes):

```bash
difflab train -c configs/class_conditioned_fashionmnist.yaml
```

Sample specific classes (one image per requested label):

```bash
difflab sample -c configs/class_conditioned_fashionmnist.yaml \
    --checkpoint outputs/class_cond_fashionmnist/final \
    --labels 0,1,2,3,4,5,6,7,8,9
```

## Programmatic sampling

```python
import torch
from diffusers import UNet2DModel, DDIMScheduler
from difflab.sampling import ddim_sample

model = UNet2DModel.from_pretrained("outputs/class_cond_fashionmnist/final")
scheduler = DDIMScheduler(num_train_timesteps=1000)
labels = torch.tensor([7, 7, 7, 7])           # four "sneaker" samples
imgs = ddim_sample(model, scheduler, 4, class_labels=labels, num_inference_steps=50)
```

See the notebook [`02_class_conditioned.ipynb`](https://github.com/ROHITCRAFTSYT/difflab/blob/main/notebooks/02_class_conditioned.ipynb).
