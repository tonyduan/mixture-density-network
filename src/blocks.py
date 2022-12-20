from enum import Enum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights_xavier(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_normal_(module.weight)
        torch.nn.init.constant_(module.bias, 0.)


class NoiseType(Enum):
    DIAGONAL = auto()
    ISOTROPIC = auto()
    ISOTROPIC_ACROSS_CLUSTERS = auto()
    FIXED = auto()


class MixtureDensityNetwork(nn.Module):
    """
    Mixture density network.

    [ Bishop, 1994 ]

    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """
    def __init__(self, dim_in, dim_out, n_components, hidden_dim, noise_type=NoiseType.DIAGONAL, fixed_noise_level=None):
        super().__init__()
        assert (fixed_noise_level is not None) == (noise_type is NoiseType.FIXED)
        num_sigma_channels = {
            NoiseType.DIAGONAL: dim_out * n_components,
            NoiseType.ISOTROPIC: n_components,
            NoiseType.ISOTROPIC_ACROSS_CLUSTERS: 1,
            NoiseType.FIXED: 0,
        }[noise_type]
        self.dim_in, self.dim_out, self.n_components = dim_in, dim_out, n_components
        self.noise_type, self.fixed_noise_level = noise_type, fixed_noise_level
        self.pi_network = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_components),
        )
        self.normal_network = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_out * n_components + num_sigma_channels)
        )
        self.apply(init_weights_xavier)

    def forward(self, x, eps=1e-6):
        #
        # Returns
        # -------
        # log_pi: (bsz, n_components)
        # mu: (bsz, n_components, dim_out)
        # sigma: (bsz, n_components, dim_out)
        #
        log_pi = torch.log_softmax(self.pi_network(x), dim=-1)
        normal_params = self.normal_network(x)
        mu = normal_params[..., :self.dim_out * self.n_components]
        sigma = normal_params[..., self.dim_out * self.n_components:]
        if self.noise_type is NoiseType.DIAGONAL:
            sigma = torch.exp(sigma + eps)
        if self.noise_type is NoiseType.ISOTROPIC:
            sigma = torch.exp(sigma + eps).repeat(1, self.dim_out)
        if self.noise_type is NoiseType.ISOTROPIC_ACROSS_CLUSTERS:
            sigma = torch.exp(sigma + eps).repeat(1, self.n_components * self.dim_out)
        if self.noise_type is NoiseType.FIXED:
            sigma = torch.full_like(mu, fill_value=self.fixed_noise_level)
        mu = mu.reshape(-1, self.n_components, self.dim_out)
        sigma = sigma.reshape(-1, self.n_components, self.dim_out)
        return log_pi, mu, sigma

    def loss(self, x, y):
        log_pi, mu, sigma = self.forward(x)
        z_score = (y.unsqueeze(1) - mu) / sigma
        normal_loglik = (
            -0.5 * torch.einsum("bij,bik->bi", z_score, z_score)
            -0.5 * torch.log(torch.einsum("bij,bik->bi", sigma, sigma))
        )
        loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1)
        return -loglik

    def sample(self, x):
        log_pi, mu, sigma = self.forward(x)
        cum_pi = torch.cumsum(torch.exp(log_pi), dim=-1)
        rvs = torch.rand(len(x), 1).to(x)
        rand_pi = torch.searchsorted(cum_pi, rvs)
        rand_normal = torch.randn_like(mu) * sigma + mu
        samples = torch.gather(rand_normal, index=rand_pi.unsqueeze(-1), dim=1).squeeze(dim=1)
        return samples
