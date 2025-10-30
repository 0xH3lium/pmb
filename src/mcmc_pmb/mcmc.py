"""Metropolis-Hastings sampler utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .likelihood import log_likelihood as compute_log_likelihood
from .priors import log_prior


@dataclass(frozen=True)
class MetropolisHastingsConfig:
    """Configuration for the Metropolis-Hastings sampler."""

    n_iterations: int = 50000
    burn_in: int = 10000
    thinning: int = 1
    proposal_std: Sequence[float] = (5.0, 0.05)
    random_seed: Optional[int] = 42

    def validate(self) -> None:
        if self.n_iterations <= 0:
            raise ValueError("n_iterations must be positive")
        if self.burn_in < 0:
            raise ValueError("burn_in must be non-negative")
        if self.thinning <= 0:
            raise ValueError("thinning must be positive")
        if len(self.proposal_std) != 2:
            raise ValueError("proposal_std must contain two elements")


@dataclass
class MetropolisHastingsResult:
    """Container for the samples and diagnostics."""

    raw_chain: np.ndarray
    log_posteriors: np.ndarray
    accepted: np.ndarray
    burn_in: int
    thinning: int

    @property
    def acceptance_rate(self) -> float:
        return float(np.mean(self.accepted))

    @property
    def chain(self) -> np.ndarray:
        post_burn = self.raw_chain[self.burn_in :]
        if self.thinning > 1:
            post_burn = post_burn[:: self.thinning]
        return post_burn


def run_metropolis_hastings(
    config: MetropolisHastingsConfig,
    prior_dist: Any,
    production_data: pd.DataFrame,
    model: Any,
    pressure_uncertainty: float,
    initial_parameters: Optional[Iterable[float]] = None,
) -> MetropolisHastingsResult:
    """Run the Metropolis-Hastings sampler and return the collected chain."""

    config.validate()

    rng = np.random.default_rng(config.random_seed)
    proposal_std = np.asarray(config.proposal_std, dtype=float)

    if initial_parameters is None:
        if hasattr(prior_dist, "mean"):
            current_params = np.asarray(prior_dist.mean, dtype=float)
        else:
            raise ValueError("initial_parameters must be provided when prior mean is unavailable")
    else:
        current_params = np.asarray(tuple(initial_parameters), dtype=float)

    raw_chain = np.zeros((config.n_iterations, 2), dtype=float)
    log_posteriors = np.full(config.n_iterations, -np.inf, dtype=float)
    accepted = np.zeros(config.n_iterations, dtype=bool)

    current_log_prior = log_prior(current_params, prior_dist)
    if not np.isfinite(current_log_prior):
        raise ValueError("Initial parameters lie outside the prior support")

    current_log_likelihood = compute_log_likelihood(
        current_params, production_data, model, pressure_uncertainty
    )
    current_log_posterior = current_log_prior + current_log_likelihood

    for idx in range(config.n_iterations):
        proposal = current_params + rng.normal(scale=proposal_std, size=2)
        proposal_log_prior = log_prior(proposal, prior_dist)
        if np.isfinite(proposal_log_prior):
            proposal_log_likelihood = compute_log_likelihood(
                proposal, production_data, model, pressure_uncertainty
            )
        else:
            proposal_log_likelihood = -np.inf

        proposal_log_posterior = proposal_log_prior + proposal_log_likelihood
        log_accept_ratio = proposal_log_posterior - current_log_posterior

        if np.log(rng.uniform()) < log_accept_ratio:
            current_params = proposal
            current_log_posterior = proposal_log_posterior
            accepted[idx] = True

        raw_chain[idx] = current_params
        log_posteriors[idx] = current_log_posterior

    return MetropolisHastingsResult(
        raw_chain=raw_chain,
        log_posteriors=log_posteriors,
        accepted=accepted,
        burn_in=config.burn_in,
        thinning=config.thinning,
    )


__all__ = ["MetropolisHastingsConfig", "MetropolisHastingsResult", "run_metropolis_hastings"]
