### Mixture Density Network

Last update: December 2022.

---

Lightweight implementation of a mixture density network [1] in PyTorch.

Suppose we want to regress response $y \in \mathbb{R}^{d}$ using covariates $x \in \mathbb{R}^n$.

We model the conditional distribution as a mixture of Gaussians
```math
p_\theta(y|x) = \sum_{k=1}^K \pi^{(k)} N(\mu^{(k)}, {\Sigma}^{(k)}),
```
where the mixture distribution parameters listed below are output by a neural network dependent on $x$.
```math
\begin{align*}
( \pi & \in\Delta^{K-1} & \mu^{(k)}&\in\mathbb{R}^{d} &\Sigma^{(k)}&\in \mathrm{S}_+^d) = f_\theta(x)
\end{align*}
```
The training objective is to maximize log-likelihood. The objective is clearly non-convex.
```math
\begin{align*}
\log p_\theta(y|x)
& \propto\log \sum_{k}\left(\pi^{(k)}\exp\left(-\frac{1}{2}\left(y-\mu^{(k)}\right)^\top {\Sigma^{(k)}}^{-1}\left(y-\mu^{(k)}\right) -\frac{1}{2}\log\det \Sigma^{(k)}\right)\right)\\
& = \mathrm{logsumexp}_k\left(\log\pi^{(k)} - \frac{1}{2}\left(y-\mu^{(k)}\right)^\top {\Sigma^{(k)}}^{-1}\left(y-\mu^{(k)}\right) -\frac{1}{2}\log\det \Sigma^{(k)}\right)\\
\end{align*}
```
Importantly, we need to use `torch.log_softmax(...)` to compute logits of $\pi^{(k)}$ for numerical stability,
```math
\begin{align*}
\log\pi^{(k)} & = \pi_\mathrm{raw}^{(k)} - \mathrm{logsumexp}_k\pi_\mathrm{raw}^{(k)} & \implies \sum_k\exp\log\pi^{(k)} &= 1.
\end{align*}
```

**Noise Model**

To simplify the training objective there are assumptions we can make on the noise model $\Sigma^{(k)}$.

1. No assumptions, $\Sigma^{(k)} \in \mathrm{S}_+^d$.
2. Fully factored, let $\Sigma^{(k)} = \mathrm{diag}({\sigma^2}^{(k)}), {\sigma^2}^{(k)}\in\mathbb{R}_+^d$ where the noise level for each dimension is predicted separately.
3. Isotrotopic, let $\Sigma^{(k)} = \sigma^2I, \sigma^2\in\mathbb{R}_+$ which assumes the same noise level for each dimension over $d$.
4. Fixed isotropic, same as above but do not learn $\sigma^2$.

Thse correspond to the following objectives.
```math
\begin{align*}
\log p_\theta(y|x) & = \mathrm{logsumexp}_k\left(\log\pi^{(k)} - \frac{1}{2}\left(y-\mu^{(k)}\right)^\top {\Sigma^{(k)}}^{-1}\left(y-\mu^{(k)}\right) -\frac{1}{2}\log\det \Sigma^{(k)}\right)  \tag{1}\\
& = \mathrm{logsumexp}_k \left(\log\pi^{(k)} - \frac{1}{2}\left\|\frac{y-\mu^{(k)}}{\sigma^{(k)}}\right\|^2-\frac{1}{2}\log\|\sigma^{(k)}\|^2\right) \tag{2}\\
& = \mathrm{logsumexp}_k \left(\log\pi^{(k)} - \frac{1}{2}\left\|\frac{y-\mu^{(k)}}{\sigma^{(k)}}\right\|^2-\frac{1}{2}\log({\sigma^{(k)}}^2)\right) \tag{3}\\
& = \mathrm{logsumexp}_k\left(\log \pi^{(k)} - \frac{1}{2}\|y-\mu^{(k)}\|^2\right) \tag{4}
\end{align*}
```
In this repository we implement options (2, 3, 4). One way to employ option (1) might be a generative modeling style network such as in PixelRNN [3], but that's not in scope here.

**Miscellaneous**

Recall that the objective is clearly non-convex. For example, one local minimum is to ignore all modes except one and place a single diffuse Gaussian distribution on the marginal outcome (i.e. high ${\sigma^2}^{(k)}$).

For this reason it's often preferable to over-parameterize the model and specify `n_components` higher than the true hypothesized number of modes.

#### Usage

```python
import torch
from src.blocks import MixtureDensityNetwork

x = torch.randn(5, 1)
y = torch.randn(5, 1)

# 1D input, 1D output, 3 mixture components
model = MixtureDensityNetwork(1, 1, n_components=3, hidden_dim=50)
pred_parameters = model(x)

# use this to backprop
loss = model.loss(x, y)

# use this to sample a trained model
samples = model.sample(x)
```

For further details see the `examples/` folder. Below is a model fit with 3 components in `ex_1d.py`.

![ex_model](examples/ex_1d.png "Example model output")



#### References

[1] Bishop, C. M. Mixture density networks. (1994).

[2] Ha, D. & Schmidhuber, J. Recurrent World Models Facilitate Policy Evolution. in *Advances in Neural Information Processing Systems 31* (eds. Bengio, S. et al.) 2450–2462 (Curran Associates, Inc., 2018).

[3] Van Den Oord, A., Kalchbrenner, N. & Kavukcuoglu, K. Pixel Recurrent Neural Networks. in *Proceedings of the 33rd International Conference on International Conference on Machine Learning - Volume 48* 1747–1756.

#### License

This code is available under the MIT License.
