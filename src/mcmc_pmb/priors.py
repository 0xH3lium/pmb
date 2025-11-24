"""Prior utilities for the probabilistic material balance model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


__all__ = ["PriorParameters"]
