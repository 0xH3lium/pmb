"""Utilities for generating physically-consistent synthetic datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .forward_model import MaterialBalanceModel
from .pvt import MaterialBalancePVT


def _calculate_instantaneous_gor(
    pressure: float,
    pvt: MaterialBalancePVT,
    gas_prod_coefficient: float,
) -> float:
    """
    Calculate the instantaneous producing GOR based on reservoir pressure.
    """
    rs_at_p = pvt.solution_gor(pressure)
    if pressure >= pvt.bubble_point:
        return rs_at_p
    else:
        # Excess gas production is proportional to drawdown below bubble point
        drawdown = pvt.bubble_point - pressure
        excess_gor = gas_prod_coefficient * drawdown
        return rs_at_p + excess_gor


def generate_synthetic_dataset(
    output_csv: Optional[Path] = None,
    n_steps: int = 24,
    true_parameters: tuple[float, float] = (110.0, 0.35),
    measurement_noise: float = 50.0,  # Slightly reduced noise for clearer diagnostics
    gas_prod_coefficient: float = 0.6,
    random_seed: int = 2026,
) -> pd.DataFrame:
    """
    Generate a physically-consistent synthetic production history.
    """
    rng = np.random.default_rng(random_seed)

    pvt = MaterialBalancePVT()
    model = MaterialBalanceModel(pvt=pvt)

    # Define the oil production schedule
    time_days = np.linspace(0.0, 2400.0, num=n_steps)
    
    # --- FIX: Reduced max production from 48.0 to 18.0 MMSTB ---
    # 48 MMSTB (44% recovery) is physically impossible for primary depletion,
    # causing pressure to crash to the lower bound and stalling the MCMC.
    # 18 MMSTB (~16% recovery) provides a realistic pressure decline curve.
    cum_oil_schedule = np.linspace(0.0, 18.0, num=n_steps)

    # --- Time-stepping simulation loop ---
    true_pressures = np.zeros(n_steps)
    cumulative_gors = np.zeros(n_steps)
    
    # Initialize conditions at time t=0
    current_pressure = pvt.initial_pressure
    true_pressures[0] = current_pressure
    cumulative_gors[0] = pvt.solution_gor_initial
    cum_gas_produced = 0.0

    for i in range(1, n_steps):
        # 1. Determine incremental oil production for this step
        delta_np = cum_oil_schedule[i] - cum_oil_schedule[i-1]
        
        # 2. Calculate instantaneous GOR based on the pressure at the START of the step
        instant_gor = _calculate_instantaneous_gor(
            current_pressure, pvt, gas_prod_coefficient
        )
        
        # 3. Calculate incremental and cumulative gas produced
        delta_gp = instant_gor * delta_np
        cum_gas_produced += delta_gp
        
        # 4. Calculate the new cumulative GOR (Rp)
        current_np = cum_oil_schedule[i]
        current_rp = cum_gas_produced / current_np if current_np > 0 else pvt.solution_gor_initial
        cumulative_gors[i] = current_rp
        
        # 5. Solve for the new reservoir pressure
        current_pressure = model._solve_pressure(
            N=true_parameters[0], m=true_parameters[1], Np=current_np, Rp=current_rp
        )
        true_pressures[i] = current_pressure
    
    # Assemble the final DataFrame
    production_data = pd.DataFrame(
        {
            "time_days": time_days,
            "Np": cum_oil_schedule,
            "Rp": cumulative_gors,
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