"""Common configuration and result containers for samplers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class MetropolisHastingsConfig:
    """Configuration for the Metropolis-Hastings sampler."""

    n_iterations: int = 50000
    burn_in: int = 10000
    thinning: int = 1
    proposal_std: Sequence[float] = (5.0, 0.05, 0.1)
    random_seed: Optional[int] = 42
    use_adaptive: bool = False
    initial_covariance: Optional[Sequence[Sequence[float]]] = None
    adaptation_start: int = 100
    adaptation_interval: int = 100

    def validate(self) -> None:
        if self.n_iterations <= 0:
            raise ValueError("n_iterations must be positive")
        if self.burn_in < 0:
            raise ValueError("burn_in must be non-negative")
        if self.thinning <= 0:
            raise ValueError("thinning must be positive")
        if len(self.proposal_std) != 3:
            raise ValueError("proposal_std must contain three elements")
        if self.use_adaptive:
            if self.adaptation_start < 0:
                raise ValueError("adaptation_start must be non-negative")
            if self.adaptation_interval <= 0:
                raise ValueError("adaptation_interval must be positive")


@dataclass
class MetropolisHastingsResult:
    """Container for sampled chains and diagnostics."""

    raw_chain: np.ndarray
    log_posteriors: np.ndarray
    accepted: np.ndarray
    burn_in: int
    thinning: int
    idata: Any | None = None

    @property
    def acceptance_rate(self) -> float:
        return float(np.mean(self.accepted))

    @property
    def chain(self) -> np.ndarray:
        post_burn = self.raw_chain[self.burn_in :]
        if self.thinning > 1:
            post_burn = post_burn[:: self.thinning]
        return post_burn

    @property
    def log_posteriors_chain(self) -> np.ndarray:
        post_burn = self.log_posteriors[self.burn_in :]
        if self.thinning > 1:
            post_burn = post_burn[:: self.thinning]
        return post_burn


@dataclass(frozen=True)
class NUTSConfig:
    """Configuration for the NUTS sampler."""

    n_samples: int = 2000
    n_tune: int = 1000
    target_accept: float = 0.8
    max_treedepth: int = 10
    random_seed: Optional[int] = 42
    n_chains: int = 4
    
    def validate(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.n_tune < 0:
            raise ValueError("n_tune must be non-negative")
        if not (0 < self.target_accept < 1):
            raise ValueError("target_accept must be between 0 and 1")
        if self.max_treedepth <= 0:
            raise ValueError("max_treedepth must be positive")
        if self.n_chains <= 0:
            raise ValueError("n_chains must be positive")


__all__ = ["MetropolisHastingsConfig", "MetropolisHastingsResult", "NUTSConfig"]
