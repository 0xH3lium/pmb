"""PyMC sampler implementation for the material balance model."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor.graph.op import Op
from scipy.stats import multivariate_normal

from .likelihood import log_likelihood
from .mcmc import MetropolisHastingsConfig, MetropolisHastingsResult
from .priors import PriorParameters


class LogLikelihoodOp(Op):
    """
    Custom PyTensor Op to wrap the black-box log-likelihood function.
    
    This allows us to use the existing forward model (which uses scipy.optimize)
    within a PyMC model. Since the forward model is not differentiable by PyTensor,
    we cannot use gradient-based samplers like NUTS directly on this Op without
    defining a gradient (which is difficult here). We will use gradient-free
    samplers like Metropolis or Slice.
    """
    
    itypes = [pt.dvector]  # Input: vector of parameters [N, m]
    otypes = [pt.dscalar]  # Output: scalar log-likelihood

    def __init__(
        self,
        production_data: pd.DataFrame,
        model: Any,
        pressure_uncertainty: float,
    ):
        self.production_data = production_data
        self.model = model
        self.pressure_uncertainty = pressure_uncertainty

    def perform(self, node, inputs, outputs):
        (theta,) = inputs
        # theta is a numpy array [N, m]
        logl = log_likelihood(
            parameters=theta,
            production_data=self.production_data,
            model=self.model,
            pressure_uncertainty=self.pressure_uncertainty,
        )
        outputs[0][0] = np.array(logl)


def run_pymc_sampler(
    config: MetropolisHastingsConfig,
    prior_params: PriorParameters,
    production_data: pd.DataFrame,
    model: Any,
    pressure_uncertainty: float,
) -> MetropolisHastingsResult:
    """
    Run the PyMC sampler.
    
    Args:
        config: Configuration for the sampler (iterations, burn-in, etc.)
        prior_params: Hyperparameters for the priors.
        production_data: Observed production data.
        model: The forward model instance.
        pressure_uncertainty: Standard deviation of pressure measurement noise.
        
    Returns:
        MetropolisHastingsResult: A result object compatible with the existing pipeline.
    """
    
    # Create the Op
    logl_op = LogLikelihoodOp(production_data, model, pressure_uncertainty)

    with pm.Model() as pm_model:
        # Priors
        # Construct mean and covariance for MvNormal to match manual implementation exactly
        mu = np.array([prior_params.mean_N, prior_params.mean_m])
        cov = np.array([
            [prior_params.std_N**2, prior_params.correlation * prior_params.std_N * prior_params.std_m],
            [prior_params.correlation * prior_params.std_N * prior_params.std_m, prior_params.std_m**2]
        ])
        
        # MvNormal expects a covariance matrix
        theta = pm.MvNormal("theta", mu=mu, cov=cov, shape=2)
        
        # Likelihood
        # We use pm.Potential to add an arbitrary factor to the log-probability
        pm.Potential("likelihood", logl_op(theta))
        
        # Sampling
        # We use Metropolis because we don't have gradients
        
        # Configure proposal scaling
        # pm.Metropolis expects S to be the proposal covariance matrix
        proposal_std = np.array(config.proposal_std)
        S = np.diag(proposal_std**2)
        
        step = pm.Metropolis(S=S)
        
        # Handle tuning and burn-in
        # If adaptive, we use PyMC's tuning.
        # We want the total number of samples to be config.n_iterations.
        # If we use tuning, those samples are usually discarded, but we can keep them
        # to match the manual sampler's behavior of returning the full chain.
        
        if config.use_adaptive:
            tune = config.burn_in
            draws = config.n_iterations - tune
            if draws < 0:
                raise ValueError("n_iterations must be greater than burn_in")
        else:
            tune = 0
            draws = config.n_iterations
            
        idata = pm.sample(
            draws=draws,
            tune=tune,
            step=step,
            chains=1,
            return_inferencedata=True,
            progressbar=True,
            random_seed=config.random_seed,
            compute_convergence_checks=False,
            discard_tuned_samples=False
        )

    # Extract samples
    # Extract samples
    # idata.posterior["theta"] has shape (chains, draws, 2)
    chain = idata.posterior["theta"].values[0] # shape (draws, 2)
    
    # If we have warmup samples (tune > 0 and discard_tuned_samples=False), include them
    if hasattr(idata, "warmup_posterior") and idata.warmup_posterior is not None:
        # Check if "theta" is in warmup_posterior (it should be)
        if "theta" in idata.warmup_posterior:
            warmup_chain = idata.warmup_posterior["theta"].values[0] # shape (tune, 2)
            chain = np.concatenate([warmup_chain, chain], axis=0)
    
    # Re-compute log_posteriors for consistency with the result object
    log_posteriors = np.zeros(len(chain))
    
    for i, theta_val in enumerate(chain):
        # Prior
        lp = multivariate_normal.logpdf(theta_val, mean=mu, cov=cov)
        
        # Likelihood
        ll = log_likelihood(theta_val, production_data, model, pressure_uncertainty)
        
        log_posteriors[i] = lp + ll

    # Infer accepted steps
    accepted = np.concatenate(([True], np.any(chain[1:] != chain[:-1], axis=1)))
    
    return MetropolisHastingsResult(
        raw_chain=chain,
        log_posteriors=log_posteriors,
        accepted=accepted,
        burn_in=config.burn_in,
        thinning=config.thinning,
    )
