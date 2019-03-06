### Mixture Density Networks in PyTorch

Last update: March 2019

---

Lightweight implementation of a mixture density network [1] in PyTorch.

A mixture density network parameterizes the distribution over a scalar response variable of interest as a mixture of Gaussian distributions.
<p align="center"><img alt="$$&#10;p_\theta(y|x) = \sum_{k=1}^K \pi^{(k)} \mathcal{N}(\mu^{(k)}, {\sigma^2}^{(k)}),&#10;$$" src="svgs/17870bed581ed5d53c0b24e84ca488a6.svg" align="middle" width="232.54644105pt" height="48.18280005pt"/></p>
where
<p align="center"><img alt="$$&#10;\{\pi^{(k)}, \mu^{(k)}, {\sigma^2}^{(k)}\}_{k=1}^K&#10;$$" src="svgs/89d606a285fc8c10fba5542b37dae2c4.svg" align="middle" width="147.2683674pt" height="22.87862775pt"/></p>

are output by parameters of a neural network.

In order to predict the response as a multivariate Gaussian distribution (for example, in [2]), we assume a fully factored distribution (i.e. a diagonal covariance matrix) and predict each dimesion separately. Another possible approach would be to use an auto-regressive method like in [3], but we leave that implementation for future work.

#### Usage

Todo.

For further details the `examples/` folder.

#### References

[1] Bishop, C. M. Mixture density networks. (1994).

[2] Ha, D. & Schmidhuber, J. World Models. *arXiv:1803.10122 [cs, stat]* (2018).

[3] Van Den Oord, A., Kalchbrenner, N. & Kavukcuoglu, K. Pixel Recurrent Neural Networks. in *Proceedings of the 33rd International Conference on International Conference on Machine Learning - Volume 48* 1747–1756.

#### License

This code is available under the MIT License.
