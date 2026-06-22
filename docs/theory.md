# Theory

This page derives the equations the code implements. It is deliberately
self-contained.

## Forward (noising) process

A diffusion model defines a fixed Markov chain that gradually adds Gaussian
noise to data $x_0$ over $T$ steps, with a variance schedule $\beta_1, \dots, \beta_T$:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t; \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t I\right).$$

Defining $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod_{s=1}^{t} \alpha_s$,
the chain admits a closed form for sampling $x_t$ directly from $x_0$:

$$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I).$$

This is exactly what `DDPMScheduler.add_noise` computes, and it is the line that
produces training targets in
[`Trainer._training_step`](https://github.com/ROHITCRAFTSYT/difflab).

## Training objective

A neural network $\epsilon_\theta(x_t, t)$ is trained to predict the noise that
was added. The simplified DDPM loss is a plain MSE:

$$\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t}\left[\,\lVert \epsilon - \epsilon_\theta(x_t, t)\rVert^2\,\right].$$

The toolkit also supports `v_prediction` and `sample` targets, selected by
`scheduler.prediction_type`. For class-conditioned models the network additionally
takes a label embedding: $\epsilon_\theta(x_t, t, y)$.

## Reverse (sampling) process — DDPM

Ancestral DDPM sampling iterates from $x_T \sim \mathcal{N}(0, I)$ down to $x_0$,
at each step computing the posterior mean from the predicted noise and adding
fresh noise. This is `ddpm_sample`.

## DDIM sampling

DDIM (Song et al., 2021) defines a **non-Markovian** process with the same
marginals but a controllable stochasticity parameter $\eta$. Its update is

$$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\, \underbrace{\left(\frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta}{\sqrt{\bar\alpha_t}}\right)}_{\hat x_0}
+ \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta + \sigma_t z,$$

with $\sigma_t = 0$ giving a **deterministic** mapping from noise to image. This
determinism (used in `ddim_sample`, `eta=0`) enables far fewer sampling steps
and, crucially, inversion.

## DDIM inversion

Because the $\eta = 0$ map is a deterministic ODE, we can run it *backwards* to
recover the noise $x_T$ that generates a given $x_0$. Dropping the noise term and
swapping $t-1 \to t+1$, each inversion step estimates $\hat x_0$ from $x_t$ and
re-noises to the next (higher) timestep:

$$x_{t+1} = \sqrt{\bar\alpha_{t+1}}\,\hat x_0 + \sqrt{1-\bar\alpha_{t+1}}\,\epsilon_\theta(x_t, t).$$

This is `ddim_invert`. The equality is exact only in the small-step limit (it
assumes $\epsilon_\theta(x_t, t) \approx \epsilon_\theta(x_{t+1}, t+1)$), which is
why inversion uses many steps and **`clip_sample=False`** — clipping $\hat x_0$
would break the invertibility, as our tests demonstrate.

Editing then works by inverting under a source prompt and re-sampling under a
target prompt; the shared noise keeps structure while the new prompt changes
content. See [DDIM inversion](ddim_inversion.md).
