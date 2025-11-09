"""MCMC for Probabilistic Material Balance (PMB).

This package provides tools to build priors, evaluate forward models, compute likelihoods,
and run a Metropolis-Hastings sampler for reservoir material-balance analysis.
"""

from .priors import PriorParameters, build_prior_distribution, log_prior
from .forward_model import MaterialBalanceModel
from .likelihood import log_likelihood
from .mcmc import MetropolisHastingsConfig, run_metropolis_hastings
from .synthetic import generate_synthetic_dataset
from .analysis import summarize_chain, make_diagnostics

__all__ = [
    "PriorParameters",
    "build_prior_distribution",
    "log_prior",
    "MaterialBalanceModel",
    "log_likelihood",
    "MetropolisHastingsConfig",
    "run_metropolis_hastings",
    "generate_synthetic_dataset",
    "summarize_chain",
    "make_diagnostics",
]
