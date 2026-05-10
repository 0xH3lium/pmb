"""Utilities for generating physically-consistent synthetic datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .forward_model import MaterialBalanceModel
from .pvt import MaterialBalancePVT, PVTEngine


def _calculate_instantaneous_gor(
    pressure: float,
    params: MaterialBalancePVT,
    engine: PVTEngine,
    gas_prod_coefficient: float,
) -> float:
    """Calculate the instantaneous producing GOR based on reservoir pressure."""

    rs_at_p, _ = engine.solution_gor_numpy(pressure)
    rs_at_p = float(rs_at_p)

    if pressure >= params.bubble_point:
        return rs_at_p

    drawdown = params.bubble_point - pressure
    excess_gor = gas_prod_coefficient * drawdown
    return rs_at_p + excess_gor


def generate_synthetic_dataset(
    output_csv: Optional[Path] = None,
    n_steps: int = 30,
    true_parameters: tuple[float, float] = (110.0, 0.6),
    measurement_noise: float = 20.0,
    gas_prod_coefficient: float = 0.6,
    random_seed: int = 2026,
    max_prod: float = 12.0e6,  # CHANGED: 12 Million STB (approx 10% recovery)
    aquifer_J: float = 15.0,  # CHANGED: Stronger aquifer for a bigger field
    aquifer_C: float = 1.0e6,  # Aquifer capacity in rb/psi
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    params = MaterialBalancePVT()
    model = MaterialBalanceModel(
        pvt_params=params, aquifer_index=aquifer_J, aquifer_capacity=aquifer_C
    )
    engine = model.pvt_engine

    time_days = np.linspace(0.0, 2400.0, num=n_steps)
    cum_oil_schedule = np.linspace(0.0, max_prod, num=n_steps)

    true_pressures = np.zeros(n_steps)
    cumulative_gors = np.zeros(n_steps)
    cumulative_wp = np.zeros(n_steps)

    current_pressure = params.initial_pressure
    true_pressures[0] = current_pressure
    cumulative_gors[0] = params.solution_gor_initial
    cum_gas_produced = 0.0
    cumulative_wp[0] = 0.0
    we_cumulative = 0.0

    # Prescribe a smooth water-oil ratio curve to ensure rising water production.
    transition = 0.6 * time_days[-1]
    wor_max = 1.2
    wor_curve = wor_max / (
        1.0 + np.exp(-(time_days - transition) / max(transition * 0.15, 1e-6))
    )
    wor_curve = np.clip(wor_curve, 0.0, wor_max)
    wp_schedule = np.maximum.accumulate(cum_oil_schedule * wor_curve)
    wp_schedule[0] = 0.0

    for i in range(1, n_steps):
        delta_np = cum_oil_schedule[i] - cum_oil_schedule[i - 1]
        dt = time_days[i] - time_days[i - 1]

        instant_gor = _calculate_instantaneous_gor(
            current_pressure, params, engine, gas_prod_coefficient
        )

        delta_gp = instant_gor * delta_np
        cum_gas_produced += delta_gp

        current_np = cum_oil_schedule[i]
        current_rp = (
            cum_gas_produced / current_np
            if current_np > 0
            else params.solution_gor_initial
        )
        cumulative_gors[i] = current_rp
        current_wp = wp_schedule[i]

        current_pressure, we_cumulative = model.solve_single_step(
            initial_pressure=current_pressure,
            we_prev=we_cumulative,
            dt=dt,
            N=true_parameters[0],
            m=true_parameters[1],
            Np=current_np,
            Rp=current_rp,
            Wp=current_wp,
        )
        true_pressures[i] = current_pressure
        cumulative_wp[i] = current_wp

    production_data = pd.DataFrame(
        {
            "time_days": time_days,
            "Np": cum_oil_schedule,
            "Rp": cumulative_gors,
            "Wp": cumulative_wp,
        }
    )

    noisy_pressures = true_pressures + rng.normal(scale=measurement_noise, size=n_steps)
    production_data["Pressure_truth"] = true_pressures
    production_data["Pressure_measured"] = noisy_pressures

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        production_data.to_csv(output_csv, index=False)

    return production_data


__all__ = ["generate_synthetic_dataset"]
