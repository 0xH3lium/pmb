"""
PVT correlations for material balance calculations.
Supports both Numpy (float/array) and PyTensor inputs via backend dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np
import pytensor.tensor as pt

# Type alias for inputs that can be float, array, or symbolic tensor
Numeric = Union[float, np.ndarray, pt.TensorVariable]

@dataclass(frozen=True, slots=True)
class MaterialBalancePVT:
    """
    PVT correlations suitable for material balance modeling.
    """
    initial_pressure: float = 3500.0
    bubble_point: float = 3200.0
    oil_fvf_initial: float = 1.20
    gas_fvf_initial: float = 0.0045
    solution_gor_initial: float = 650.0
    oil_compress_above_pb: float = 12e-6
    oil_shrinkage_below_pb: float = 1.5e-4
    z_factor_initial: float = 0.88
    p_at_min_z: float = 2200.0
    rs_exponent_below_pb: float = 0.85

    # Internal pre-calculated constants (not passed to init)
    _bo_at_pb: float = field(init=False, repr=False)
    _z_factor_a: float = field(init=False, repr=False)
    _z_factor_b: float = field(init=False, repr=False)

    def __init__(
        self,
        initial_pressure: float = 3500.0,
        bubble_point: float = 3200.0,
        oil_fvf_initial: float = 1.20,
        gas_fvf_initial: float = 0.0045,
        solution_gor_initial: float = 650.0,
        oil_compress_above_pb: float = 12e-6,
        oil_shrinkage_below_pb: float = 1.5e-4,
        z_factor_initial: float = 0.88,
        p_at_min_z: float = 2200.0,
        rs_exponent_below_pb: float = 0.85,
    ):
        # Manual __init__ required for validation/pre-calc with frozen=True
        object.__setattr__(self, "initial_pressure", initial_pressure)
        object.__setattr__(self, "bubble_point", bubble_point)
        object.__setattr__(self, "oil_fvf_initial", oil_fvf_initial)
        object.__setattr__(self, "gas_fvf_initial", gas_fvf_initial)
        object.__setattr__(self, "solution_gor_initial", solution_gor_initial)
        object.__setattr__(self, "oil_compress_above_pb", oil_compress_above_pb)
        object.__setattr__(self, "oil_shrinkage_below_pb", oil_shrinkage_below_pb)
        object.__setattr__(self, "z_factor_initial", z_factor_initial)
        object.__setattr__(self, "p_at_min_z", p_at_min_z)
        object.__setattr__(self, "rs_exponent_below_pb", rs_exponent_below_pb)

        if self.bubble_point > self.initial_pressure:
            raise ValueError("Bubble point pressure cannot be greater than initial pressure.")

        # --- Pre-calculations ---
        delta_p = self.initial_pressure - self.bubble_point
        bo_at_pb = self.oil_fvf_initial * np.exp(self.oil_compress_above_pb * delta_p)
        
        Pi, Zi, P_min = self.initial_pressure, self.z_factor_initial, self.p_at_min_z
        denom = Pi**2 - 2 * P_min * Pi
        if abs(denom) < 1e-9:
            raise ValueError("Unstable Z-factor parameters (singularity in quadratic fit).")
            
        z_a = (Zi - 1.0) / denom
        z_b = -2 * z_a * P_min

        object.__setattr__(self, "_bo_at_pb", bo_at_pb)
        object.__setattr__(self, "_z_factor_a", z_a)
        object.__setattr__(self, "_z_factor_b", z_b)

    def _is_tensor(self, x: Any) -> bool:
        """Check if input is a PyTensor variable."""
        return isinstance(x, (pt.TensorVariable, pt.TensorConstant))

    def _math(self, x: Any):
        """Returns the math module (numpy or pytensor.tensor) matching the input."""
        return pt if self._is_tensor(x) else np

    def _clip(self, val: Numeric, low: float, high: float) -> Numeric:
        m = self._math(val)
        return m.clip(val, low, high)

    def _exp(self, val: Numeric) -> Numeric:
        return self._math(val).exp(val)

    def _switch(self, cond: Any, if_true: Numeric, if_false: Numeric) -> Numeric:
        # If any input is a tensor, we must use pt.switch
        if self._is_tensor(cond) or self._is_tensor(if_true) or self._is_tensor(if_false):
            return pt.switch(cond, if_true, if_false)
        return np.where(cond, if_true, if_false)

    def oil_fvf(self, pressure: Numeric) -> Numeric:
        p_safe = self._clip(pressure, 50.0, self.initial_pressure * 1.2)
        
        # Above Pb
        d_above = self.initial_pressure - p_safe
        bo_above = self.oil_fvf_initial * self._exp(self.oil_compress_above_pb * d_above)
        
        # Below Pb
        d_below = self.bubble_point - p_safe
        bo_below = self._bo_at_pb - self.oil_shrinkage_below_pb * d_below
        
        # Clamp minimum Bo to 1.0 (handle tensor vs numpy max)
        if self._is_tensor(bo_below):
            bo_below = pt.maximum(1.0, bo_below)
        else:
            bo_below = np.maximum(1.0, bo_below)

        # Condition: Pressure >= Bubble Point
        # Use >= operator to support both Numpy (bool array) and PyTensor (TensorVariable)
        is_above = p_safe >= self.bubble_point
        
        return self._switch(is_above, bo_above, bo_below)

    def gas_fvf(self, pressure: Numeric) -> Numeric:
        p_safe = self._clip(pressure, 50.0, self.initial_pressure * 1.2)
        
        # Z-Factor: Z = aP^2 + bP + 1
        z = self._z_factor_a * p_safe**2 + self._z_factor_b * p_safe + 1.0
        
        # Bg = Bgi * (Pi/P) * (Z/Zi)
        z_ratio = z / self.z_factor_initial
        p_ratio = self.initial_pressure / p_safe
        return self.gas_fvf_initial * p_ratio * z_ratio

    def solution_gor(self, pressure: Numeric) -> Numeric:
        p_safe = self._clip(pressure, 50.0, self.initial_pressure * 1.2)
        
        p_frac = p_safe / self.bubble_point
        rs_below = self.solution_gor_initial * (p_frac ** self.rs_exponent_below_pb)
        
        is_above = p_safe >= self.bubble_point
        
        return self._switch(is_above, self.solution_gor_initial, rs_below)