### Mixture Density Networks in PyTorch

Last update: March 2019

---

Lightweight implementation of a mixture density network [1] in PyTorch.

An MDN models the conditional distribution over a scalar response as a mixture of Gaussians.
<p align="center"><img alt="$$&#10;p_\theta(y|x) = \sum_{k=1}^K \pi^{(k)} \mathcal{N}(\mu^{(k)}, {\sigma^2}^{(k)}),&#10;$$" src="svgs/17870bed581ed5d53c0b24e84ca488a6.svg" align="middle" width="232.54644105pt" height="48.18280005pt"/></p>
where the mixture distribution parameters <img alt="$\{\pi^{(k)}, \mu^{(k)}, {\sigma^2}^{(k)}\}_{k=1}^K$" src="svgs/90212d5e9d97a01e1581e42fe459a8b5.svg" align="middle" width="147.26836739999996pt" height="35.89423859999999pt"/> are output by a neural network with parameters <img alt="$\theta$" src="svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg" align="middle" width="8.17352744999999pt" height="22.831056599999986pt"/>, trained to maximize overall log-likelihood.

In order to predict the response as a multivariate Gaussian distribution (for example, in [2]), we assume a fully factored distribution (i.e. a diagonal covariance matrix) and predict each dimesion separately. Another possible approach would be to use an auto-regressive method like in [3], but we leave that implementation for future work.

#### Usage

```python
import torch
from mdn.model import MixtureDensityNetwork
from mdn.utils import mixture_density_loss, sample_gaussian_mixture

x = torch.randn(5, 1)
y = torch.randn(5, 1)

# 1D input, 1D output, 3 mixture components
model = MixtureDensityNetwork(1, 1, 3)
pred_parameters = model(x)

# use this to backprop
loss = mixture_density_loss(y, pred_parameters)

# use this to sample a trained model
samples = sample_gaussian_mixture(pred_parameters)
```

For further details see the `examples/` folder. 

#### References

[1] Bishop, C. M. Mixture density networks. (1994).

[2] Ha, D. & Schmidhuber, J. World Models. *arXiv:1803.10122 [cs, stat]* (2018).

[3] Van Den Oord, A., Kalchbrenner, N. & Kavukcuoglu, K. Pixel Recurrent Neural Networks. in *Proceedings of the 33rd International Conference on International Conference on Machine Learning - Volume 48* 1747–1756.

#### License

This code is available under the MIT License.
