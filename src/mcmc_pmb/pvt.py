"""
PVT correlations for material balance calculations.

This module provides a more physically-grounded yet simple set of PVT correlations
suitable for material balance modeling, improving upon the previous version's
oversimplifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class MaterialBalancePVT:
    """
    PVT correlations based on standard, simplified fluid behavior models.

    This class provides methods to calculate oil FVF (Bo), gas FVF (Bg), and
    solution GOR (Rs) as a function of pressure. The correlations are chosen
    to be a balance between physical realism and simplicity, making them suitable
    for forward modeling and inversion.

    Attributes:
        initial_pressure: Initial reservoir pressure [psi].
        bubble_point: Bubble point pressure of the oil [psi].
        oil_fvf_initial: Oil FVF at initial pressure [rb/stb].
        gas_fvf_initial: Gas FVF at initial pressure [rb/scf].
        solution_gor_initial: Solution GOR at initial pressure [scf/stb].
        oil_compress_above_pb: Oil compressibility above bubble point [1/psi].
        oil_compress_below_pb: Oil compressibility below bubble point [1/psi].
        z_factor_initial: Gas deviation factor (Z-factor) at initial pressure.
        rs_exponent_below_pb: Exponent for the power-law Rs correlation below bubble point.
    """

    # --- Fluid Properties (Inputs) ---
    initial_pressure: float = 3500.0
    bubble_point: float = 3200.0
    oil_fvf_initial: float = 1.20
    gas_fvf_initial: float = 0.0045
    solution_gor_initial: float = 650.0
    oil_compress_above_pb: float = 12e-6
    oil_compress_below_pb: float = 28e-6
    z_factor_initial: float = 0.88
    rs_exponent_below_pb: float = 0.85

    # --- Pre-calculated values for internal use ---
    _bo_at_pb: float = field(init=False, repr=False)

    def __post_init__(self):
        """Validate inputs and pre-calculate key values."""
        if self.bubble_point > self.initial_pressure:
            raise ValueError("Bubble point pressure cannot be greater than initial pressure.")

        # Pre-calculate Bo at the bubble point for efficiency and clarity.
        # This is an internal value, hence the underscore.
        delta_p = self.initial_pressure - self.bubble_point
        bo_at_pb = self.oil_fvf_initial * np.exp(self.oil_compress_above_pb * delta_p)
        
        # Use object.__setattr__ to assign to a frozen dataclass instance
        object.__setattr__(self, "_bo_at_pb", bo_at_pb)

    def oil_fvf(self, pressure: float) -> float:
        """
        Calculate the oil formation volume factor (Bo).

        - Above bubble point: Models oil expansion using a constant compressibility.
        - Below bubble point: Models oil swelling due to gas liberation, also
          using a constant compressibility relative to the bubble point.
        """
        pressure = float(np.clip(pressure, 50.0, self.initial_pressure * 1.2))

        if pressure >= self.bubble_point:
            delta_p = self.initial_pressure - pressure
            return self.oil_fvf_initial * np.exp(self.oil_compress_above_pb * delta_p)
        else:
            delta_p_below = self.bubble_point - pressure
            return self._bo_at_pb * np.exp(self.oil_compress_below_pb * delta_p_below)

    def gas_fvf(self, pressure: float) -> float:
        """
        Calculate the gas formation volume factor (Bg).

        Uses the real gas law (Bg ∝ Z/P) with a linear approximation for the
        Z-factor, assuming Z=1.0 at zero pressure. This is more representative
        of real gas behavior than a simple exponential correction.
        """
        pressure = float(np.clip(pressure, 50.0, self.initial_pressure * 1.2))

        # Linear Z-factor approximation: Z(p) = m*p + c
        # Assuming Z(p=0) = 1.0 and Z(p=pi) = z_initial
        m = (self.z_factor_initial - 1.0) / self.initial_pressure
        z_at_p = m * pressure + 1.0
        
        # Ratio of (Z/P) scaled by initial conditions
        # Bg = Bgi * (Pi/P) * (Z/Zi)
        z_ratio = z_at_p / self.z_factor_initial
        pressure_ratio = self.initial_pressure / pressure

        return self.gas_fvf_initial * pressure_ratio * z_ratio

    def solution_gor(self, pressure: float) -> float:
        """
        Calculate the solution gas-oil ratio (Rs).

        - Above bubble point: Rs is constant at its initial value.
        - Below bubble point: A simple and common power-law relationship is used
          to model the decline in dissolved gas with pressure.
        """
        pressure = float(np.clip(pressure, 50.0, self.initial_pressure * 1.2))

        if pressure >= self.bubble_point:
            return self.solution_gor_initial
        else:
            pressure_frac = pressure / self.bubble_point
            return self.solution_gor_initial * pressure_frac**self.rs_exponent_below_pb


__all__ = ["MaterialBalancePVT"]