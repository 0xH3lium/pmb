"""Differentiable PVT correlations for the material-balance model.

The classic implementation mixed NumPy and PyTensor execution paths, which
prevented aggressive graph optimisation and gradient-based inference.  This
module refactors the PVT logic into two explicit components:

* ``MaterialBalancePVT`` – an immutable container of correlation parameters.
* ``PVTEngine`` – a differentiable evaluator backed by monotonic cubic
  splines, exposing PyTensor-friendly methods for fast and smooth execution.

The construction of the spline coefficients happens eagerly with NumPy and
SciPy.  All runtime evaluations – including derivatives – use PyTensor ops
only, which keeps the computation graph differentiable end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pytensor.tensor as pt
from pytensor import config as pt_config
from scipy.interpolate import PchipInterpolator

FloatArray = np.ndarray


@dataclass(frozen=True, slots=True)
class MaterialBalancePVT:
    """Immutable container for the PVT correlation parameters."""

    initial_pressure: float = 3500.0
    bubble_point: float = 3200.0
    oil_fvf_initial: float = 1.20
    gas_fvf_initial: float = 0.0045
    solution_gor_initial: float = 650.0
    oil_compress_above_pb: float = 12e-6
    oil_shrinkage_below_pb: float = 1.5e-4
    reservoir_temperature_f: float = 180.0
    gas_specific_gravity: float = 0.65
    rs_exponent_below_pb: float = 0.85
    pressure_min: float = 50.0
    pressure_multiplier: float = 1.2
    spline_size: int = 256

    def __post_init__(self) -> None:
        if self.bubble_point > self.initial_pressure:
            raise ValueError("Bubble point pressure cannot exceed initial pressure.")
        if self.reservoir_temperature_f <= -459.67:
            raise ValueError("reservoir_temperature_f must be above absolute zero.")
        if self.gas_specific_gravity <= 0.0:
            raise ValueError("gas_specific_gravity must be positive.")
        if self.spline_size < 4:
            raise ValueError("spline_size must be at least 4 points for cubic splines.")

    @property
    def pressure_max(self) -> float:
        return self.initial_pressure * self.pressure_multiplier

    def build_engine(self, *, spline_size: int | None = None) -> "PVTEngine":
        """Create a differentiable PVT engine using monotonic cubic splines."""

        grid_size = int(spline_size or self.spline_size)
        pressure_grid = np.linspace(self.pressure_min, self.pressure_max, grid_size)

        bo_values = _oil_fvf_exact(self, pressure_grid)
        bg_values = _gas_fvf_exact(self, pressure_grid)
        rs_values = _solution_gor_exact(self, pressure_grid)

        bo_coeffs = _piecewise_cubic_coefficients(pressure_grid, bo_values)
        bg_coeffs = _piecewise_cubic_coefficients(pressure_grid, bg_values)
        rs_coeffs = _piecewise_cubic_coefficients(pressure_grid, rs_values)

        return PVTEngine(
            params=self,
            knots=pressure_grid,
            oil_coefficients=bo_coeffs,
            gas_coefficients=bg_coeffs,
            rs_coefficients=rs_coeffs,
        )


@dataclass(frozen=True, slots=True)
class PVTEngine:
    """Differentiable spline-backed evaluator for PVT properties."""

    params: MaterialBalancePVT
    knots: FloatArray
    oil_coefficients: FloatArray
    gas_coefficients: FloatArray
    rs_coefficients: FloatArray

    _knots_tensor: pt.TensorConstant = field(init=False, repr=False)
    _oil_coeffs_tensor: pt.TensorConstant = field(init=False, repr=False)
    _gas_coeffs_tensor: pt.TensorConstant = field(init=False, repr=False)
    _rs_coeffs_tensor: pt.TensorConstant = field(init=False, repr=False)
    _oil_at_initial: float = field(init=False, repr=False)
    _gas_at_initial: float = field(init=False, repr=False)
    _rs_at_initial: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        dtype = pt_config.floatX
        knots = np.asarray(self.knots, dtype=dtype)
        oil = np.asarray(self.oil_coefficients, dtype=dtype)
        gas = np.asarray(self.gas_coefficients, dtype=dtype)
        rs = np.asarray(self.rs_coefficients, dtype=dtype)

        object.__setattr__(self, "_knots_tensor", pt.as_tensor_variable(knots))
        object.__setattr__(self, "_oil_coeffs_tensor", pt.as_tensor_variable(oil))
        object.__setattr__(self, "_gas_coeffs_tensor", pt.as_tensor_variable(gas))
        object.__setattr__(self, "_rs_coeffs_tensor", pt.as_tensor_variable(rs))

        oil_init = float(self.oil_fvf_numpy(self.params.initial_pressure)[0])
        gas_init = float(self.gas_fvf_numpy(self.params.initial_pressure)[0])
        rs_init = float(self.solution_gor_numpy(self.params.initial_pressure)[0])

        object.__setattr__(self, "_oil_at_initial", oil_init)
        object.__setattr__(self, "_gas_at_initial", gas_init)
        object.__setattr__(self, "_rs_at_initial", rs_init)

    @property
    def oil_at_initial(self) -> float:
        return self._oil_at_initial

    @property
    def gas_at_initial(self) -> float:
        return self._gas_at_initial

    @property
    def rs_at_initial(self) -> float:
        return self._rs_at_initial

    def oil_fvf(self, pressure: pt.TensorVariable) -> Tuple[pt.TensorVariable, pt.TensorVariable]:
        """Return formation volume factor and derivative."""

        return _evaluate_piecewise_tensor(
            pressure, self._knots_tensor, self._oil_coeffs_tensor
        )

    def gas_fvf(self, pressure: pt.TensorVariable) -> Tuple[pt.TensorVariable, pt.TensorVariable]:
        return _evaluate_piecewise_tensor(
            pressure, self._knots_tensor, self._gas_coeffs_tensor
        )

    def solution_gor(self, pressure: pt.TensorVariable) -> Tuple[pt.TensorVariable, pt.TensorVariable]:
        return _evaluate_piecewise_tensor(
            pressure, self._knots_tensor, self._rs_coeffs_tensor
        )

    def oil_fvf_numpy(self, pressure: FloatArray | float) -> Tuple[FloatArray, FloatArray]:
        return _evaluate_piecewise_numpy(pressure, self.knots, self.oil_coefficients)

    def gas_fvf_numpy(self, pressure: FloatArray | float) -> Tuple[FloatArray, FloatArray]:
        return _evaluate_piecewise_numpy(pressure, self.knots, self.gas_coefficients)

    def solution_gor_numpy(self, pressure: FloatArray | float) -> Tuple[FloatArray, FloatArray]:
        return _evaluate_piecewise_numpy(pressure, self.knots, self.rs_coefficients)

    def evaluate_all(
        self, pressure: pt.TensorVariable
    ) -> Tuple[pt.TensorVariable, pt.TensorVariable, pt.TensorVariable, pt.TensorVariable, pt.TensorVariable, pt.TensorVariable]:
        """Return Bo, dBo/dp, Bg, dBg/dp, Rs, dRs/dp in a single call."""

        bo, dbo = self.oil_fvf(pressure)
        bg, dbg = self.gas_fvf(pressure)
        rs, drs = self.solution_gor(pressure)
        return bo, dbo, bg, dbg, rs, drs


def _oil_fvf_exact(params: MaterialBalancePVT, pressure: FloatArray) -> FloatArray:
    p_safe = np.clip(np.asarray(pressure, dtype=float), params.pressure_min, params.pressure_max)

    delta_above = params.initial_pressure - p_safe
    bo_above = params.oil_fvf_initial * np.exp(params.oil_compress_above_pb * delta_above)

    delta_below = params.bubble_point - p_safe
    bo_at_pb = params.oil_fvf_initial * np.exp(params.oil_compress_above_pb * (params.initial_pressure - params.bubble_point))
    bo_below = np.maximum(1.0, bo_at_pb - params.oil_shrinkage_below_pb * delta_below)

    is_above = p_safe >= params.bubble_point
    return np.where(is_above, bo_above, bo_below)


def _gas_fvf_exact(params: MaterialBalancePVT, pressure: FloatArray) -> FloatArray:
    p_safe = np.clip(np.asarray(pressure, dtype=float), params.pressure_min, params.pressure_max)
    pi = params.initial_pressure
    z = _z_factor_dak(
        pressure_psia=p_safe,
        reservoir_temperature_f=params.reservoir_temperature_f,
        gas_specific_gravity=params.gas_specific_gravity,
    )
    zi = float(
        _z_factor_dak(
            pressure_psia=np.asarray([params.initial_pressure], dtype=float),
            reservoir_temperature_f=params.reservoir_temperature_f,
            gas_specific_gravity=params.gas_specific_gravity,
        )[0]
    )
    z_ratio = z / zi
    p_ratio = pi / p_safe
    return params.gas_fvf_initial * p_ratio * z_ratio


def _z_factor_dak(
    *,
    pressure_psia: FloatArray,
    reservoir_temperature_f: float,
    gas_specific_gravity: float,
) -> FloatArray:
    """Compute gas z-factor with Dranchuk-Abou-Kassem (1975)."""

    # Sutton pseudo-critical correlations.
    tpc_r = 169.2 + 349.5 * gas_specific_gravity - 74.0 * gas_specific_gravity**2
    ppc_psia = 756.8 - 131.0 * gas_specific_gravity - 3.6 * gas_specific_gravity**2
    tpr = (reservoir_temperature_f + 459.67) / tpc_r
    ppr = np.asarray(pressure_psia, dtype=float) / ppc_psia

    # Dranchuk-Abou-Kassem constants.
    a1, a2, a3 = 0.3265, -1.0700, -0.5339
    a4, a5, a6 = 0.01569, -0.05165, 0.5475
    a7, a8, a9 = -0.7361, 0.1844, 0.1056
    a10, a11 = 0.6134, 0.7210

    c1 = a1 + a2 / tpr + a3 / tpr**3 + a4 / tpr**4 + a5 / tpr**5
    c2 = a6 + a7 / tpr + a8 / tpr**2
    c3 = -a9 * (a7 / tpr + a8 / tpr**2)
    c4 = a10 / tpr**3

    target = 0.27 * ppr / tpr
    rho_r = np.maximum(target, 1e-10)

    tolerance = 1e-10
    max_iter = 100
    for _ in range(max_iter):
        exp_term = np.exp(-a11 * rho_r**2)
        f = (
            rho_r
            + c1 * rho_r**2
            + c2 * rho_r**3
            + c3 * rho_r**6
            + c4 * rho_r**3 * (1.0 + a11 * rho_r**2) * exp_term
            - target
        )
        df = (
            1.0
            + 2.0 * c1 * rho_r
            + 3.0 * c2 * rho_r**2
            + 6.0 * c3 * rho_r**5
            + c4 * rho_r**2 * exp_term * (3.0 + 3.0 * a11 * rho_r**2 - 2.0 * a11**2 * rho_r**4)
        )

        # Damped Newton step for numerical stability.
        df_safe = np.where(np.abs(df) > 1e-14, df, np.where(df >= 0.0, 1e-14, -1e-14))
        delta = f / df_safe
        rho_next = np.maximum(rho_r - delta, 1e-10)
        if np.max(np.abs(rho_next - rho_r)) < tolerance:
            rho_r = rho_next
            break
        rho_r = rho_next

    return 0.27 * ppr / (rho_r * tpr)


def _solution_gor_exact(params: MaterialBalancePVT, pressure: FloatArray) -> FloatArray:
    p_safe = np.clip(np.asarray(pressure, dtype=float), params.pressure_min, params.pressure_max)
    p_frac = p_safe / params.bubble_point
    rs_below = params.solution_gor_initial * np.power(p_frac, params.rs_exponent_below_pb)
    is_above = p_safe >= params.bubble_point
    return np.where(is_above, params.solution_gor_initial, rs_below)


def _piecewise_cubic_coefficients(knots: FloatArray, values: FloatArray) -> FloatArray:
    interp = PchipInterpolator(knots, values, extrapolate=False)
    slopes = interp.derivative()(knots)

    n_segments = knots.size - 1
    coeffs = np.empty((n_segments, 4), dtype=float)

    for i in range(n_segments):
        x0 = knots[i]
        x1 = knots[i + 1]
        h = x1 - x0
        if h <= 0:
            raise ValueError("Pressure grid must be strictly increasing for spline coefficients.")

        y0 = values[i]
        y1 = values[i + 1]
        m0 = slopes[i]
        m1 = slopes[i + 1]

        a = 2.0 * y0 - 2.0 * y1 + h * (m0 + m1)
        b = -3.0 * y0 + 3.0 * y1 - 2.0 * h * m0 - h * m1
        c = h * m0
        d = y0

        coeffs[i, 0] = a / (h**3)
        coeffs[i, 1] = b / (h**2)
        coeffs[i, 2] = m0
        coeffs[i, 3] = d

    return coeffs


def _evaluate_piecewise_tensor(
    pressure: pt.TensorVariable,
    knots: pt.TensorVariable,
    coeffs: pt.TensorVariable,
) -> Tuple[pt.TensorVariable, pt.TensorVariable]:
    """Evaluate cubic spline (value, derivative) using PyTensor ops only."""

    p = pt.clip(pressure, knots[0], knots[-1])
    init_val = coeffs[0, 3] + pt.zeros_like(p)
    init_grad = coeffs[0, 2] + pt.zeros_like(p)

    def step(knot_low, coeff_low, prev_val, prev_grad):
        dx = p - knot_low
        seg_val = ((coeff_low[0] * dx + coeff_low[1]) * dx + coeff_low[2]) * dx + coeff_low[3]
        seg_grad = (3.0 * coeff_low[0] * dx + 2.0 * coeff_low[1]) * dx + coeff_low[2]
        use_seg = pt.ge(p, knot_low)
        next_val = pt.switch(use_seg, seg_val, prev_val)
        next_grad = pt.switch(use_seg, seg_grad, prev_grad)
        return next_val, next_grad

    # Broadcast over the spline segments to locate the active interval without Scan.
    p_flat = pt.reshape(p, (-1,))
    knots_low = knots[:-1]
    # Count how many knot intervals are below each pressure sample.
    ge_mask = pt.ge(pt.expand_dims(p_flat, 1), knots_low)
    idx = pt.sum(ge_mask, axis=1, dtype="int64") - 1

    max_idx = coeffs.shape[0] - 1
    idx = pt.maximum(pt.minimum(idx, max_idx), 0)

    base_pressure = knots[idx]
    coeff_sel = coeffs[idx]
    dx = p_flat - base_pressure

    vals_flat = ((coeff_sel[:, 0] * dx + coeff_sel[:, 1]) * dx + coeff_sel[:, 2]) * dx + coeff_sel[:, 3]
    grads_flat = (3.0 * coeff_sel[:, 0] * dx + 2.0 * coeff_sel[:, 1]) * dx + coeff_sel[:, 2]

    vals = pt.reshape(vals_flat, p.shape)
    grads = pt.reshape(grads_flat, p.shape)
    return vals, grads


def _evaluate_piecewise_numpy(
    pressure: FloatArray | float,
    knots: FloatArray,
    coeffs: FloatArray,
) -> Tuple[FloatArray, FloatArray]:
    p = np.clip(np.asarray(pressure, dtype=float), knots[0], knots[-1])
    idx = np.searchsorted(knots, p, side="right") - 1
    idx = np.clip(idx, 0, coeffs.shape[0] - 1)

    dx = p - knots[idx]
    vals = ((coeffs[idx, 0] * dx + coeffs[idx, 1]) * dx + coeffs[idx, 2]) * dx + coeffs[idx, 3]
    grads = (3.0 * coeffs[idx, 0] * dx + 2.0 * coeffs[idx, 1]) * dx + coeffs[idx, 2]
    return vals, grads