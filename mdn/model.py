import torch
import torch.nn as nn
from mdn.utils import gen_network


class MixtureDensityNetwork(nn.Module):
    """
    Mixture density network.

    Uses 2-layer linear neural networks (with one tanh non-linearity).

    [ Bishop, 1994 ]

    Parameters
    ----------
    dim_in: int
        dimensionality of the covariates

    dim_out: int
        dimensionality of the response variable

    n_components: int
        number of components in the mixture model
    """
    def __init__(self, dim_in, dim_out, n_components):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_components = n_components
        self.pi_network = gen_network(dim_in, n_components * dim_out)
        self.mu_network = gen_network(dim_in, n_components * dim_out)
        self.logsigma_network = gen_network(dim_in, n_components * dim_out)

    def forward(self, x):
        """
        Parameters
        ----------
        x: Tensor

        Returns
        -------
        parameters: list of tuples
            each tuple contains (π, μ, σ) for a scalar dimension of output,
            where each variable is a vector of same length as the number of
            mixture components
        """
        parameters = []
        pi = torch.softmax(self.pi_network(x), dim=1)
        mu = self.mu_network(x)
        sigma = torch.exp(self.logsigma_network(x))
        for dim in range(self.dim_out):
            i, j = dim * self.n_components, (dim + 1) * self.n_components
            parameters.append((pi[:,i:j], mu[:,i:j], sigma[:,i:j]))
        return parameters
