"""Prior distribution utilities for the probabilistic material balance model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple

import numpy as np
from scipy.stats import multivariate_normal


@dataclass(frozen=True)
class PriorParameters:
    """Container for the bivariate normal prior hyperparameters."""

    mean_N: float
    mean_m: float
    std_N: float
    std_m: float
    correlation: float

    def to_mean_vector(self) -> np.ndarray:
        """Return the mean vector ordered as (N, m)."""
        return np.asarray([self.mean_N, self.mean_m], dtype=float)

    def to_covariance_matrix(self) -> np.ndarray:
        """Return the 2x2 covariance matrix implied by the hyperparameters."""
        rho = float(self.correlation)
        cov_nm = rho * self.std_N * self.std_m
        cov = np.asarray(
            [[self.std_N**2, cov_nm], [cov_nm, self.std_m**2]],
            dtype=float,
        )
        return cov


def build_prior_distribution(hyperparameters: PriorParameters) -> Any:
    """Build the SciPy multivariate normal object for the prior."""

    return multivariate_normal(
        mean=hyperparameters.to_mean_vector(),
        cov=hyperparameters.to_covariance_matrix(),
    )


def log_prior(parameters: Iterable[float], prior_dist: Any) -> float:
    """Return the log-prior probability for a proposed (N, m) pair."""

    N, m = parameters
    if N <= 0.0 or m < 0.0:
        return -np.inf
    return float(prior_dist.logpdf(np.asarray([N, m], dtype=float)))


__all__: Tuple[str, ...] = ("PriorParameters", "build_prior_distribution", "log_prior")
