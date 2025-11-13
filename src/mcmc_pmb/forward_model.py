"""Forward material-balance model for pressure prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.optimize import OptimizeResult

from .pvt import MaterialBalancePVT


@dataclass(frozen=True)
class _NoWaterDrive:  # placeholder removed: water drive is no longer supported
    pass


@dataclass(frozen=True)
class MaterialBalanceModel:
    """Forward solver that predicts pressure history for a given (N, m)."""

    pvt: MaterialBalancePVT
    connate_water_saturation: float = 0.20   # Swi, fraction
    pore_compressibility: float = 4.0e-6     # cf, in 1/psi
    water_compressibility: float = 3.0e-6    # cw, in 1/psi
    pressure_bounds: Tuple[float, float] = (200.0, 5000.0)
    drive_scale: float = 1.1  # Used by the analytic approximation
    root_tol: float = 1e-4

    def _material_balance_residual(self, pressure: float, N: float, m: float, Np: float, Rp: float) -> float:
        """Return residual of the general material balance equation."""
        if N <= 0.0 or m < 0.0:
            return pressure - self.pressure_bounds[0]

        pressure = float(np.clip(pressure, *self.pressure_bounds))

        pvt = self.pvt
        Bo = pvt.oil_fvf(pressure)
        Bg = pvt.gas_fvf(pressure)
        Rs = pvt.solution_gor(pressure)

        Boi = pvt.oil_fvf(pvt.initial_pressure)
        Bgi = pvt.gas_fvf(pvt.initial_pressure)
        Rsi = pvt.solution_gor(pvt.initial_pressure)

        # No water drive: effective voidage equals produced pore volume
        fraction_depleted = float(np.clip(Np / max(N, 1e-12), 0.0, 1.0))
        effective_voidage = Np

        # --- MODIFICATION START ---
        # Calculate the expansion from connate water and pore volume reduction.
        # This term provides additional reservoir drive energy.
        delta_p = pvt.initial_pressure - pressure
        compressibility_expansion = 0.0
        if delta_p > 0:  # Effect only occurs during depletion
            swi = self.connate_water_saturation
            ceff_numerator = (self.water_compressibility * swi) + self.pore_compressibility
            ceff_denominator = 1.0 - swi
            # Effective compressibility referenced to hydrocarbon pore volume
            effective_compressibility = ceff_numerator / max(ceff_denominator, 1e-9)
            
            # Total expansion volume, expressed as an equivalent surface oil volume
            compressibility_expansion = N * Boi * effective_compressibility * delta_p
        # --- MODIFICATION END ---
        
        # Left-hand side: Cumulative fluid withdrawal from the reservoir
        lhs = effective_voidage * (Bo + (Rp - Rs) * Bg)

        # Right-hand side: Expansion of original fluids in place
        oil_and_gas_expansion = N * ((Bo - Boi) + (Rsi - Rs) * Bg)
        gas_cap_expansion = N * m * Boi * ((Bg / max(Bgi, 1e-12)) - 1.0)

        # Add the new compressibility term to the expansion side
        rhs = oil_and_gas_expansion + gas_cap_expansion + compressibility_expansion

        return lhs - rhs

    def _analytic_pressure(self, N: float, m: float, Np: float, Rp: float) -> float:
        """
        Empirical closed-form approximation of material-balance depletion.
        NOTE: This simple approximation does not include compressibility effects,
        but it serves as a reasonable starting guess for the numerical solver.
        """
        if N <= 0.0 or m < 0.0:
            return self.pressure_bounds[0]

        initial_pressure = self.pvt.initial_pressure
        rs_initial = self.pvt.solution_gor_initial
        fraction_depleted = np.clip(Np / max(N, 1e-6), 0.0, 5.0)
        gas_drive_term = np.clip(Rp - rs_initial, 0.0, None) / max(rs_initial, 1e-6)

        effective_drive = fraction_depleted * (1.0 + 0.6 * m) + 0.4 * m * gas_drive_term
        
        exponent = -effective_drive / max(self.drive_scale, 1e-6)
        pressure = initial_pressure * np.exp(exponent)
        
        return float(np.clip(pressure, *self.pressure_bounds))

    def _solve_pressure(self, N: float, m: float, Np: float, Rp: float) -> float:
        """
        Solve the general MBE implicitly for reservoir pressure using SciPy.
        
        This method uses a robust bracketing solver (brentq) and falls back
        to minimizing the residual if no root is found in the interval.
        """
        if N <= 0.0 or m < 0.0 or Np == 0:
            return self.pvt.initial_pressure

        residual_func = lambda p: self._material_balance_residual(p, N=N, m=m, Np=Np, Rp=Rp)
        lower, upper = self.pressure_bounds


        try:
            # Brent's method: fast, robust, and guaranteed to find a root if one exists
            root = opt.brentq(residual_func, lower, upper, xtol=self.root_tol)
            return float(root)
        except ValueError:

            res: OptimizeResult = opt.minimize_scalar(
                lambda p: abs(residual_func(p)),
                bounds=self.pressure_bounds,
                method="bounded",
            )
            return float(res.x)

    def predict_pressures(
        self,
        parameters: Sequence[float],
        production_data: pd.DataFrame,
    ) -> np.ndarray:
        """Predict reservoir pressures across the production history."""
        N, m = map(float, parameters)

        # Ensure data is sorted by time for a valid history prediction
        sorted_data = production_data.sort_values("time_days", kind="stable")

        pressures = []
        
        # Use the first data point to get a good analytic first guess
        first_row = next(sorted_data.itertuples(), None)
        if first_row is None:
            return np.array([], dtype=float)

        current_guess = self._analytic_pressure(N=N, m=m, Np=first_row.Np, Rp=first_row.Rp)

        # Use itertuples() for a major performance improvement over iterrows()
        for row in sorted_data.itertuples(index=False):
            # The previous step's solution is an excellent guess for the current step
            pressure = self._solve_pressure(N=N, m=m, Np=row.Np, Rp=row.Rp)
            pressures.append(pressure)

        return np.asarray(pressures, dtype=float)


__all__ = ["MaterialBalanceModel"]