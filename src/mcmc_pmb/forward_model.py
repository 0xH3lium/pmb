"""Forward material-balance model for pressure prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.optimize as opt

from .pvt import MaterialBalancePVT

@dataclass(frozen=True, slots=True)
class ProductionDataset:
    time_days: np.ndarray
    Np: np.ndarray
    Rp: np.ndarray
    pressure_measured: np.ndarray
    pressure_truth: Optional[np.ndarray] = None

    @property
    def n_steps(self) -> int:
        return self.time_days.size

def prepare_production_dataset(data: Union[pd.DataFrame, ProductionDataset]) -> ProductionDataset:
    if isinstance(data, ProductionDataset):
        return data

    required = {"time_days", "Np", "Rp", "Pressure_measured"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing columns: {required - set(data.columns)}")

    df = data.sort_values("time_days") if not data["time_days"].is_monotonic_increasing else data
    
    return ProductionDataset(
        time_days=df["time_days"].to_numpy(dtype=float),
        Np=df["Np"].to_numpy(dtype=float),
        Rp=df["Rp"].to_numpy(dtype=float),
        pressure_measured=df["Pressure_measured"].to_numpy(dtype=float),
        pressure_truth=df["Pressure_truth"].to_numpy(dtype=float) if "Pressure_truth" in df else None,
    )

@dataclass(frozen=True, slots=True)
class MaterialBalanceModel:
    pvt: MaterialBalancePVT
    connate_water_saturation: float = 0.20
    pore_compressibility: float = 4.0e-6
    water_compressibility: float = 3.0e-6
    pressure_bounds: tuple[float, float] = (100.0, 5000.0)
    root_tol: float = 1e-4
    
    # Internal effective compressibility term, not passed to init
    _eff_comp_term: float = field(init=False)

    def __init__(
        self,
        pvt: MaterialBalancePVT,
        connate_water_saturation: float = 0.20,
        pore_compressibility: float = 4.0e-6,
        water_compressibility: float = 3.0e-6,
        pressure_bounds: tuple[float, float] = (100.0, 5000.0),
        root_tol: float = 1e-4,
    ):
        object.__setattr__(self, "pvt", pvt)
        object.__setattr__(self, "connate_water_saturation", connate_water_saturation)
        object.__setattr__(self, "pore_compressibility", pore_compressibility)
        object.__setattr__(self, "water_compressibility", water_compressibility)
        object.__setattr__(self, "pressure_bounds", pressure_bounds)
        object.__setattr__(self, "root_tol", root_tol)
        
        # Precompute effective compressibility factor
        swi = connate_water_saturation
        ceff_num = (water_compressibility * swi) + pore_compressibility
        ceff_den = max(1.0 - swi, 1e-9)
        object.__setattr__(self, "_eff_comp_term", ceff_num / ceff_den)

    def _residual(self, pressure: float, N: float, m: float, Np: float, Rp: float) -> float:
        # PVT Lookups
        Bo = self.pvt.oil_fvf(pressure)
        Bg = self.pvt.gas_fvf(pressure)
        Rs = self.pvt.solution_gor(pressure)

        # Initial conditions
        Pi = self.pvt.initial_pressure
        Boi = self.pvt.oil_fvf(Pi)
        Bgi = self.pvt.gas_fvf(Pi)
        Rsi = self.pvt.solution_gor(Pi)

        # Expansion terms
        delta_p = Pi - pressure
        
        # Compressibility expansion (only active if depleted)
        comp_expansion = 0.0
        if delta_p > 0:
            comp_expansion = N * Boi * self._eff_comp_term * delta_p

        # Material Balance Equation
        lhs = Np * (Bo + (Rp - Rs) * Bg)
        rhs = (N * ((Bo - Boi) + (Rsi - Rs) * Bg)) + \
              (N * m * Boi * ((Bg / max(Bgi, 1e-12)) - 1.0)) + \
              comp_expansion

        return lhs - rhs

    def _solve_pressure(
        self, N: float, m: float, Np: float, Rp: float, upper_hint: float
    ) -> float:
        if N <= 0 or m < 0 or Np == 0:
            return self.pvt.initial_pressure

        # Closure for the solver
        def func(p): 
            return self._residual(p, N, m, Np, Rp)

        lower, default_upper = self.pressure_bounds
        # Ensure upper hint is valid
        upper = upper_hint if lower < upper_hint <= default_upper else default_upper

        try:
            return float(opt.brentq(func, lower, upper, xtol=self.root_tol))
        except ValueError:
            # Fallback attempts
            try:
                if upper != default_upper:
                    return float(opt.brentq(func, lower, default_upper, xtol=self.root_tol))
            except ValueError:
                pass
            return lower

    def predict_pressures(
        self, parameters: Sequence[float], production_data: ProductionDataset
    ) -> np.ndarray:
        N, m = parameters
        n_steps = production_data.n_steps
        pressures = np.empty(n_steps, dtype=np.float64)
        
        prev_p = self.pvt.initial_pressure
        
        for i in range(n_steps):
            p = self._solve_pressure(N, m, production_data.Np[i], production_data.Rp[i], prev_p)
            pressures[i] = p
            prev_p = p
            
        return pressures