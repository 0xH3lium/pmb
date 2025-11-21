"""Likelihood evaluation for the probabilistic material balance model."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from .forward_model import MaterialBalanceModel


def log_likelihood(
    parameters: Sequence[float],
    production_data: pd.DataFrame,
    model: MaterialBalanceModel,
    pressure_uncertainty: float,
) -> float:
    """Return the Gaussian log-likelihood of measured pressures."""

    if pressure_uncertainty <= 0.0:
        raise ValueError("pressure_uncertainty must be positive")

    try:
        predicted = model.predict_pressures(parameters, production_data)
    except RuntimeError:
        return -np.inf


    if np.any(np.isnan(predicted)):
        return -np.inf
    
    measured = production_data["Pressure_measured"].to_numpy(dtype=float)
    if predicted.shape != measured.shape:
        raise ValueError("Predicted and measured pressures mismatch")

    return float(np.sum(norm.logpdf(measured, loc=predicted, scale=pressure_uncertainty)))


__all__ = ["log_likelihood"]
