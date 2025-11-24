"""MCMC for Probabilistic Material Balance (PMB).

This package provides tools to build priors, evaluate forward models, compute likelihoods,
and run a Metropolis-Hastings sampler for reservoir material-balance analysis.
"""

from .priors import PriorParameters
from .forward_model import MaterialBalanceModel, ProductionDataset, prepare_production_dataset
from .likelihood import log_likelihood
from .mcmc import MetropolisHastingsConfig
from .synthetic import generate_synthetic_dataset
from .analysis import summarize_chain, make_diagnostics, plot_posterior_predictive

__all__ = [
    "PriorParameters",
    "MaterialBalanceModel",
    "ProductionDataset",
    "prepare_production_dataset",
    "log_likelihood",
    "MetropolisHastingsConfig",
    "generate_synthetic_dataset",
    "summarize_chain",
    "make_diagnostics",
    "plot_posterior_predictive",
]
