import torch
import torch.nn as nn
from torch.distributions import Normal, OneHotCategorical


def mixture_density_loss(y, pred_parameters):
    """
    Calculate negative log-likelihood of observations y under pred params.

    Parameters
    ---------
    """
    loglik = torch.zeros(y.shape[0])
    for dim, dim_pred_parameters in enumerate(pred_parameters):
        pi, mu, sigma = dim_pred_parameters
        n_components = pi.shape[1]
        y_mixture = torch.unsqueeze(y[:,dim], 1).expand(-1, n_components)
        mixture_logprobs = Normal(mu, sigma).log_prob(y_mixture)
        loglik += torch.logsumexp(torch.log(pi) + mixture_logprobs, dim=1)
    return -torch.mean(loglik, dim=0)


def sample_gaussian_mixture(pred_parameters):
    """
    Generates one
    """
    out_dim = len(pred_parameters)
    batch_size = len(pred_parameters[0][0])
    gen_samples = torch.zeros((batch_size, out_dim))
    for dim, dim_pred_parameters in enumerate(pred_parameters):
        pi, mu, sigma = dim_pred_parameters
        batch_size = pi.shape[0]
        samples = Normal(mu, sigma).sample((1,)).squeeze()
        components = OneHotCategorical(pi).sample()
        gen_samples[:, dim] = torch.sum(samples * components, 1)
    return gen_samples


def gen_network(dim_in, dim_out):
    """
    Return a simple two-layer neural network with a tanh non-linearity.
    """
    return nn.Sequential(nn.Linear(dim_in, dim_in),
                         nn.Tanh(),
                         nn.Linear(dim_in, dim_out))
