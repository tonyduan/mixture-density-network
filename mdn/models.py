import torch
import torch.nn as nn
from torch.distributions import Normal, OneHotCategorical


class MixtureDensity(nn.Module):
    """
    Mixture density network.
    [ Bishop, 1994 ]

    Take as input a tensor of shape [N, *, dim_in] and produce
    a gaussian mixture distribution over tensors of shape
    [N, *, dim_out].

    Args:
        dim_in (int): dimensionality of the covariates
        dim_out (int): dimensionality of the response variable
        n_components (int): number of components in the mixture model
    """
    def __init__(self, dim_in, dim_out, n_components):
        super().__init__()
        self.pi_network = Categorical(dim_in, n_components)
        self.normal_network = MixtureDiagNormal(dim_in, dim_out,
                                                       n_components)

    def forward(self, x):
        return self.pi_network(x), self.normal_network(x)

    def loss(self, x, y):
        pi, normal = self.forward(x)
        loglik = normal.log_prob(y.unsqueeze(-1).expand_as(normal.loc))
        loglik = torch.sum(loglik, dim=-1)
        loss = -torch.logsumexp(torch.log(pi.probs) + loglik, dim=-1)
        return loss

    def sample(self, x):
        pi, normal = self.forward(x)
        samples = torch.sum(pi.sample().unsqueeze(-1) * normal.sample(), dim=1)
        return samples


class MixtureDensityNetwork(MixtureDensity):
    """
    Mixture density network.
    [ Bishop, 1994 ]
    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """
    def __init__(self, dim_in, dim_out, n_components):
        super().__init__(dim_in, dim_out, n_components)

        self.hidden_layer = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ELU()
        )

    def forward(self, x):
        return super().forward(self.hidden_layer(x))

    def loss(self, x, y):
        pi, normal = self.forward(x)
        loglik = normal.log_prob(y.unsqueeze(-1).expand_as(normal.loc))
        loglik = torch.sum(loglik, dim=-1)
        loss = -torch.logsumexp(torch.log(pi.probs) + loglik, dim=-1)
        return loss

    def sample(self, x):
        pi, normal = self.forward(x)
        samples = torch.sum(pi.sample().unsqueeze(-1) * normal.sample(), dim=1)
        return samples


class MixtureDiagNormal(nn.Module):

    def __init__(self, in_dim, out_dim, n_components):
        super().__init__()
        self.n_components = n_components
        self.out_dim = out_dim
        self.in_dim = in_dim

        self.mean_net = nn.Linear(in_dim, out_dim * n_components)
        self.std_net = nn.Linear(in_dim, out_dim * n_components)

    def forward(self, x):
        mean, std = self.mean_net(x), self.std_net(x)
        mean = mean.view(*mean.shape[:-1], self.out_dim, self.n_components)
        std = std.view(*std.shape[:-1], self.out_dim, self.n_components)
        return Normal(mean, torch.exp(std))


class Categorical(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim

        self.categories_net = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        logits = self.categories_net(x)
        return OneHotCategorical(logits=logits)
