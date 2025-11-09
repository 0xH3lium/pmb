"""Utilities for generating synthetic datasets for validation and demos."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .forward_model import MaterialBalanceModel
from .pvt import MaterialBalancePVT


def generate_synthetic_dataset(
    output_csv: Optional[Path] = None,
    n_steps: int = 24,
    true_parameters: tuple[float, float] = (110.0, 0.35),
    measurement_noise: float = 100.0,
    random_seed: int = 2026,
) -> pd.DataFrame:
    """Generate a synthetic production history with noisy pressure data."""

    rng = np.random.default_rng(random_seed)

    pvt = MaterialBalancePVT()
    model = MaterialBalanceModel(pvt=pvt)

    time_days = np.linspace(0.0, 2400.0, num=n_steps)
    cum_oil = np.linspace(0.0, 48.0, num=n_steps)  # million STB
    cumulative_gor = pvt.solution_gor_initial + 35.0 * np.log1p(cum_oil / 5.0)

    production_data = pd.DataFrame(
        {
            "time_days": time_days,
            "Np": cum_oil,
            "Rp": cumulative_gor,
        }
    )

    true_pressures = model.predict_pressures(true_parameters, production_data)
    measured_pressures = true_pressures + rng.normal(scale=measurement_noise, size=n_steps)

    production_data["Pressure_truth"] = true_pressures
    production_data["Pressure_measured"] = measured_pressures

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        production_data.to_csv(output_csv, index=False)

    return production_data


__all__ = ["generate_synthetic_dataset"]
