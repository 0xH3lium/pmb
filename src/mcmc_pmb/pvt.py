"""
PVT correlations for material balance calculations.

This module provides a more physically-grounded set of PVT correlations
suitable for material balance modeling, correcting common oversimplifications.
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
        oil_compress_above_pb: Oil compressibility for undersaturated oil [1/psi].
        oil_shrinkage_below_pb: Coefficient describing oil shrinkage below the
                                bubble point due to gas liberation [rb/stb/psi].
        z_factor_initial: Gas deviation factor (Z-factor) at initial pressure.
        p_at_min_z: Pressure at which the minimum Z-factor occurs, used to shape
                    the Z-factor vs. pressure curve [psi].
        rs_exponent_below_pb: Exponent for the power-law Rs correlation below bubble point.
    """

    # --- Fluid Properties (Inputs) ---
    initial_pressure: float = 3500.0
    bubble_point: float = 3200.0
    oil_fvf_initial: float = 1.20
    gas_fvf_initial: float = 0.0045
    solution_gor_initial: float = 650.0
    oil_compress_above_pb: float = 12e-6
    oil_shrinkage_below_pb: float = 1.5e-4 # A more physically plausible value
    z_factor_initial: float = 0.88
    p_at_min_z: float = 2200.0 # Typical pressure for minimum Z
    rs_exponent_below_pb: float = 0.85

    # --- Pre-calculated values for internal use ---
    _bo_at_pb: float = field(init=False, repr=False)
    _z_factor_a: float = field(init=False, repr=False)
    _z_factor_b: float = field(init=False, repr=False)

    def __post_init__(self):
        """Validate inputs and pre-calculate key values for efficiency."""
        if self.bubble_point > self.initial_pressure:
            raise ValueError("Bubble point pressure cannot be greater than initial pressure.")

        # 1. Pre-calculate Bo at the bubble point.
        # This is the peak value for Bo.
        delta_p = self.initial_pressure - self.bubble_point
        bo_at_pb = self.oil_fvf_initial * np.exp(self.oil_compress_above_pb * delta_p)
        
        # 2. Pre-calculate coefficients for the quadratic Z-factor model: Z = aP^2 + bP + 1
        # The model is constrained by Z(0)=1, Z(initial_pressure)=z_initial, and dZ/dP=0 at p_at_min_z.
        # From dZ/dP = 2aP + b = 0  =>  b = -2 * a * p_at_min_z
        # Substitute b into Z(Pi) equation to solve for a:
        Pi, Zi, P_min = self.initial_pressure, self.z_factor_initial, self.p_at_min_z
        denominator = Pi**2 - 2 * P_min * Pi
        if abs(denominator) < 1e-9:
            raise ValueError("Initial pressure and P_at_min_z result in unstable Z-factor calculation.")
        
        z_factor_a = (Zi - 1.0) / denominator
        z_factor_b = -2 * z_factor_a * P_min
        
        # Use object.__setattr__ to assign to a frozen dataclass instance
        object.__setattr__(self, "_bo_at_pb", bo_at_pb)
        object.__setattr__(self, "_z_factor_a", z_factor_a)
        object.__setattr__(self, "_z_factor_b", z_factor_b)

    def oil_fvf(self, pressure: float) -> float:
        """
        Calculate the oil formation volume factor (Bo).

        - Above bubble point: Models oil expansion using a constant compressibility. This
          is physically correct for an undersaturated, single-phase liquid.
        - Below bubble point: Models oil shrinkage as gas is liberated. Bo decreases
          from its maximum value at the bubble point. A linear shrinkage model is used
          as a robust and physically sound approximation.
        """
        pressure = float(np.clip(pressure, 50.0, self.initial_pressure * 1.2))

        if pressure >= self.bubble_point:
            delta_p = self.initial_pressure - pressure
            return self.oil_fvf_initial * np.exp(self.oil_compress_above_pb * delta_p)
        else:
            # Linear shrinkage model below the bubble point
            delta_p_below = self.bubble_point - pressure
            bo = self._bo_at_pb - self.oil_shrinkage_below_pb * delta_p_below
            return np.maximum(1.0, bo) # Ensure Bo does not fall below 1.0

    def _z_factor(self, pressure: float) -> float:
        """Calculate Z-factor using the pre-computed quadratic model."""
        return self._z_factor_a * pressure**2 + self._z_factor_b * pressure + 1.0

    def gas_fvf(self, pressure: float) -> float:
        """
        Calculate the gas formation volume factor (Bg).

        Uses the real gas law (Bg ∝ Z/P) with a more realistic quadratic
        approximation for the Z-factor. This captures the characteristic dip in Z
        at intermediate pressures.
        """
        pressure = float(np.clip(pressure, 50.0, self.initial_pressure * 1.2))
        
        # Calculate Z-factor at the given pressure using our improved model
        z_at_p = self._z_factor(pressure)
        
        # Apply the real gas law, scaled by initial conditions:
        # Bg = Bgi * (Pi/P) * (Z/Zi)
        z_ratio = z_at_p / self.z_factor_initial
        pressure_ratio = self.initial_pressure / pressure

        return self.gas_fvf_initial * pressure_ratio * z_ratio

    def solution_gor(self, pressure: float) -> float:
        """
        Calculate the solution gas-oil ratio (Rs).

        - Above bubble point: Rs is constant at its initial value.
        - Below bubble point: A standard power-law relationship is used to model
          the decline in dissolved gas with pressure. This model is scientifically sound.
        """
        pressure = float(np.clip(pressure, 50.0, self.initial_pressure * 1.2))

        if pressure >= self.bubble_point:
            return self.solution_gor_initial
        else:
            pressure_frac = pressure / self.bubble_point
            return self.solution_gor_initial * pressure_frac**self.rs_exponent_below_pb


__all__ = ["MaterialBalancePVT"]