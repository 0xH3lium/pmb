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


def _solve_pressure_numpy(
    model: MaterialBalanceModel,
    N: float, # MMSTB
    m: float,
    Np: float,
    Rp: float,
    Wp: float,
    We_prev: float,
    dt: float,
    initial_guess: float,
) -> tuple[float, float]:
    context = model._context
    engine = context.engine

    p = float(initial_guess)
    lower, upper = context.pressure_bounds
    
    # SCALE N
    N_stb = N * 1.0e6

    for _ in range(context.newton_steps):
        # ... (PVT evaluations remain the same)
        bo, dbo = engine.oil_fvf_numpy(p)
        bg, dbg = engine.gas_fvf_numpy(p)
        rs, drs = engine.solution_gor_numpy(p)
        
        # Cast to float (same as before)
        bo, bg, rs = float(bo), float(bg), float(rs)
        dbo, dbg, drs = float(dbo), float(dbg), float(drs)

        delta_p = context.initial_pressure - p
        
        # Use N_stb here
        comp_coeff = N_stb * context.Boi * context.eff_compressibility
        if delta_p > 0.0:
            comp_term = comp_coeff * delta_p
            dcomp = -comp_coeff
        else:
            comp_term = 0.0
            dcomp = 0.0

        lhs = Np * (bo + (Rp - rs) * bg) + Wp * context.Bw
        dL_dp = Np * (dbo + (Rp - rs) * dbg - drs * bg)

        # Use N_stb here
        rhs_fluid = N_stb * ((bo - context.Boi) + (context.Rsi - rs) * bg)
        drhs_fluid = N_stb * (dbo - drs * bg + (context.Rsi - rs) * dbg)

        rhs_gas = N_stb * m * context.Boi * ((bg / context.Bgi) - 1.0)
        drhs_gas = N_stb * m * context.Boi * (dbg / context.Bgi)

        delta_we = model.aquifer_index * (context.initial_pressure - p) * dt
        we_total = We_prev + delta_we

        residual = lhs - (rhs_fluid + rhs_gas + comp_term + we_total)
        
        # ... (Derivative and Newton update remain the same)
        dWe_dp = -model.aquifer_index * dt
        derivative = dL_dp - (drhs_fluid + drhs_gas + dcomp + dWe_dp)

        if abs(derivative) < model.jacobian_epsilon:
            derivative = np.copysign(model.jacobian_epsilon, derivative or 1.0)

        p_candidate = p - model.newton_damping * (residual / derivative)
        p_candidate = float(np.clip(p_candidate, lower, upper))
        
        # ... (Convergence check logic remains same)
        if abs(residual) < model.newton_tol:
            delta_we_candidate = model.aquifer_index * (context.initial_pressure - p_candidate) * dt
            we_total_candidate = We_prev + delta_we_candidate
            return p_candidate, we_total_candidate

        p = p_candidate

    delta_we = model.aquifer_index * (context.initial_pressure - p) * dt
    we_total = We_prev + delta_we
    return p, we_total


def generate_synthetic_dataset(
    output_csv: Optional[Path] = None,
    n_steps: int = 24,
    true_parameters: tuple[float, float] = (110.0, 0.35),
    measurement_noise: float = 50.0,
    gas_prod_coefficient: float = 0.6,
    random_seed: int = 2026,
    max_prod: float = 12.0e6,  # CHANGED: 12 Million STB (approx 10% recovery)
    aquifer_J: float = 15.0,   # CHANGED: Stronger aquifer for a bigger field
) -> pd.DataFrame:
    
    rng = np.random.default_rng(random_seed)

    params = MaterialBalancePVT()
    model = MaterialBalanceModel(pvt_params=params, aquifer_index=aquifer_J)
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
    wor_curve = wor_max / (1.0 + np.exp(-(time_days - transition) / max(transition * 0.15, 1e-6)))
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
            cum_gas_produced / current_np if current_np > 0 else params.solution_gor_initial
        )
        cumulative_gors[i] = current_rp
        current_wp = wp_schedule[i]

        current_pressure, we_cumulative = _solve_pressure_numpy(
            model,
            N=true_parameters[0],
            m=true_parameters[1],
            Np=current_np,
            Rp=current_rp,
            Wp=current_wp,
            We_prev=we_cumulative,
            dt=dt,
            initial_guess=current_pressure,
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