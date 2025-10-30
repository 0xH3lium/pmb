"""Forward material-balance model for pressure prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MaterialBalancePVT:
    """Simple PVT correlations used by the forward model."""

    initial_pressure: float = 3500.0
    bubble_point: float = 3200.0
    oil_fvf_initial: float = 1.20
    gas_fvf_initial: float = 0.0045
    solution_gor_initial: float = 650.0
    oil_compress_above_pb: float = 12e-6
    oil_compress_below_pb: float = 28e-6
    gas_compressibility: float = 1.2e-3
    rs_exponent_below_pb: float = 0.85

    def oil_fvf(self, pressure: float) -> float:
        """Return the oil formation volume factor at the specified pressure."""

        pressure = float(np.clip(pressure, 50.0, self.initial_pressure * 1.2))
        if pressure >= self.bubble_point:
            delta_p = self.initial_pressure - pressure
            return self.oil_fvf_initial * np.exp(self.oil_compress_above_pb * delta_p)

        # compress to bubble point then apply higher compressibility below bubble
        delta_p_above = self.initial_pressure - self.bubble_point
        bo_at_pb = self.oil_fvf_initial * np.exp(self.oil_compress_above_pb * delta_p_above)
        delta_p_below = self.bubble_point - pressure
        return bo_at_pb * np.exp(self.oil_compress_below_pb * delta_p_below)

    def gas_fvf(self, pressure: float) -> float:
        """Return the gas formation volume factor at the specified pressure."""

        pressure = float(np.clip(pressure, 50.0, self.initial_pressure * 1.2))
        z_ratio = np.exp(self.gas_compressibility * (pressure - self.initial_pressure))
        return self.gas_fvf_initial * (self.initial_pressure / pressure) * z_ratio

    def solution_gor(self, pressure: float) -> float:
        """Return the solution gas-oil ratio at the specified pressure."""

        pressure = float(np.clip(pressure, 50.0, self.initial_pressure * 1.2))
        if pressure >= self.bubble_point:
            return self.solution_gor_initial
        fraction = max(pressure, 1.0) / self.bubble_point
        return self.solution_gor_initial * fraction**self.rs_exponent_below_pb


@dataclass(frozen=True)
class WaterDriveModel:
    """Simple empirical representation of water-drive support."""

    strength: float = 0.0
    exponent: float = 1.1
    max_support_fraction: float = 0.9

    def support_fraction(self, depletion_fraction: float) -> float:
        """Return the fractional pressure support provided by the aquifer."""

        depletion_fraction = float(np.clip(depletion_fraction, 0.0, 1.0))
        raw_support = self.strength * depletion_fraction**self.exponent
        return float(np.clip(raw_support, 0.0, self.max_support_fraction))


@dataclass(frozen=True)
class MaterialBalanceModel:
    """Forward solver that predicts pressure history for a given (N, m)."""

    pvt: MaterialBalancePVT
    pressure_bounds: Tuple[float, float] = (200.0, 5000.0)
    drive_scale: float = 1.1  # empirical factor controlling depletion strength
    water_drive: Optional[WaterDriveModel] = None

    def _residual(self, pressure: float, N: float, m: float, Np: float, Rp: float) -> float:
        """Residual between trial pressure and analytic depletion model."""

        target_pressure = self._analytic_pressure(N=N, m=m, Np=Np, Rp=Rp)
        return float(pressure - target_pressure)

    def _analytic_pressure(self, N: float, m: float, Np: float, Rp: float) -> float:
        """Empirical closed-form approximation of material-balance depletion."""

        if N <= 0.0 or m < 0.0:
            return self.pressure_bounds[0]

        initial_pressure = self.pvt.initial_pressure
        rs_initial = self.pvt.solution_gor_initial
        fraction_depleted = np.clip(Np / max(N, 1e-6), 0.0, 5.0)
        gas_drive_term = np.clip(Rp - rs_initial, 0.0, None) / max(rs_initial, 1e-6)

        effective_drive = fraction_depleted * (1.0 + 0.6 * m) + 0.4 * m * gas_drive_term
        if self.water_drive is not None and self.water_drive.strength > 0.0:
            support = self.water_drive.support_fraction(float(np.clip(fraction_depleted, 0.0, 1.0)))
            effective_drive = max(effective_drive - support, 1e-6)
        exponent = -effective_drive / max(self.drive_scale, 1e-6)

        pressure = initial_pressure * np.exp(exponent)
        pressure = np.clip(pressure, *self.pressure_bounds)
        return float(pressure)

    def _solve_pressure(
        self,
        N: float,
        m: float,
        Np: float,
        Rp: float,
        initial_guess: float,
    ) -> float:
        """Return the analytic pressure estimate (root-solving surrogate)."""

        return self._analytic_pressure(N=N, m=m, Np=Np, Rp=Rp)

    def predict_pressures(
        self,
        parameters: Sequence[float],
        production_data: pd.DataFrame,
        pressure_initial_guess: Optional[float] = None,
    ) -> np.ndarray:
        """Predict reservoir pressures across the production history."""

        N, m = map(float, parameters)
        if pressure_initial_guess is None:
            pressure_initial_guess = self.pvt.initial_pressure

        sorted_data = production_data.sort_values("time_days", kind="stable") if "time_days" in production_data.columns else production_data.copy()

        pressures = []
        current_guess = float(pressure_initial_guess)
        for _, row in sorted_data.iterrows():
            Np = float(row["Np"])
            Rp = float(row["Rp"])
            current_guess = self._solve_pressure(N=N, m=m, Np=Np, Rp=Rp, initial_guess=current_guess)
            pressures.append(current_guess)

        return np.asarray(pressures, dtype=float)


__all__ = ["MaterialBalanceModel", "MaterialBalancePVT", "WaterDriveModel"]
