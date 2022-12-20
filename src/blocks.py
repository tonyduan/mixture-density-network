import torch
import torch.nn as nn
import torch.nn.functional as F


class MixtureDensityNetwork(nn.Module):
    """
    Mixture density network.

    [ Bishop, 1994 ]

    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    fixed_sigma: float; if specified it fixes the homoskedastic noise model
    """
    def __init__(self, dim_in, dim_out, n_components, hidden_dim=None, fixed_sigma=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim_in
        self.dim_in, self.dim_out, self.n_components = dim_in, dim_out, n_components
        self.fixed_sigma = fixed_sigma
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
            nn.Linear(hidden_dim, 2 * dim_out * n_components),
        )

    def forward(self, x, eps=1e-6):
        #
        # Returns
        # -------
        # log_pi: (bsz, n_components)
        # mu: (bsz, n_components, dim_out)
        # sigma: (bsz, n_components, dim_out)
        #
        log_pi = torch.log_softmax(self.pi_network(x), dim=-1)
        mu, sigma = torch.split(self.normal_network(x), self.dim_out * self.n_components, dim=1)
        mu = mu.reshape(-1, self.n_components, self.dim_out)
        sigma = sigma.reshape(-1, self.n_components, self.dim_out)
        sigma = F.elu(sigma + 1 + eps)
        if self.fixed_sigma is not None:
            sigma.fill_(self.fixed_sigma)
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
