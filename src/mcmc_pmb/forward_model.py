"""Vectorised, differentiable forward material-balance model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import pytensor.tensor as pt
from pytensor import config as pt_config
from pytensor import function
from pytensor import scan

from .pvt import MaterialBalancePVT, PVTEngine


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
        missing = required - set(data.columns)
        raise ValueError(f"Missing columns: {missing}")

    df = data.sort_values("time_days") if not data["time_days"].is_monotonic_increasing else data

    return ProductionDataset(
        time_days=df["time_days"].to_numpy(dtype=float),
        Np=df["Np"].to_numpy(dtype=float),
        Rp=df["Rp"].to_numpy(dtype=float),
        pressure_measured=df["Pressure_measured"].to_numpy(dtype=float),
        pressure_truth=df["Pressure_truth"].to_numpy(dtype=float) if "Pressure_truth" in df else None,
    )


@dataclass(frozen=True, slots=True)
class SolverContext:
    engine: PVTEngine
    eff_compressibility: float
    initial_pressure: float
    pressure_bounds: tuple[float, float]
    jacobian_epsilon: float
    damping: float
    tolerance: float
    newton_steps: int
    Boi: float
    Bgi: float
    Rsi: float


@dataclass(frozen=True, slots=True)
class MaterialBalanceModel:
    pvt_params: MaterialBalancePVT = field(default_factory=MaterialBalancePVT)
    pvt_engine: Optional[PVTEngine] = None
    connate_water_saturation: float = 0.20
    pore_compressibility: float = 4.0e-6
    water_compressibility: float = 3.0e-6
    pressure_bounds: tuple[float, float] = (100.0, 5000.0)
    newton_steps: int = 6
    newton_damping: float = 0.8
    newton_tol: float = 1e-4
    jacobian_epsilon: float = 1e-9

    _context: SolverContext = field(init=False, repr=False)

    def __post_init__(self) -> None:
        engine = self.pvt_engine or self.pvt_params.build_engine()
        lower_bound = max(self.pvt_params.pressure_min, self.pressure_bounds[0])
        upper_bound = min(self.pvt_params.pressure_max, self.pressure_bounds[1])
        if lower_bound >= upper_bound:
            raise ValueError("Invalid pressure bounds after intersecting with PVT limits.")

        swi = self.connate_water_saturation
        ceff_num = (self.water_compressibility * swi) + self.pore_compressibility
        ceff_den = max(1.0 - swi, 1e-9)
        eff_comp = ceff_num / ceff_den

        context = SolverContext(
            engine=engine,
            eff_compressibility=eff_comp,
            initial_pressure=self.pvt_params.initial_pressure,
            pressure_bounds=(lower_bound, upper_bound),
            jacobian_epsilon=self.jacobian_epsilon,
            damping=self.newton_damping,
            tolerance=self.newton_tol,
            newton_steps=self.newton_steps,
            Boi=engine.oil_at_initial,
            Bgi=max(engine.gas_at_initial, 1e-12),
            Rsi=engine.rs_at_initial,
        )

        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "pvt_engine", engine)

    def symbolic_pressures(
        self,
        N: pt.TensorVariable,
        m: pt.TensorVariable,
        dataset: ProductionDataset,
    ) -> pt.TensorVariable:
        """Return PyTensor graph for predicted pressures given symbolic parameters."""

        dtype = pt_config.floatX
        Np = pt.as_tensor_variable(dataset.Np.astype(dtype))
        Rp = pt.as_tensor_variable(dataset.Rp.astype(dtype))

        context = self._context

        def step(np_t, rp_t, prev_pressure, N_param, m_param):
            next_pressure = _newton_solve(
                prev_pressure,
                N_param,
                m_param,
                np_t,
                rp_t,
                context,
            )
            return next_pressure

        outputs, _ = scan(
            step,
            sequences=[Np, Rp],
            outputs_info=pt.as_tensor_variable(np.array(context.initial_pressure, dtype=dtype)),
            non_sequences=[N, m],
            strict=False,
        )

        return outputs

    def make_predict_function(self, dataset: ProductionDataset):
        theta = pt.vector("theta", dtype=pt_config.floatX)
        pressures = self.symbolic_pressures(theta[0], theta[1], dataset)
        return function([theta], pressures)

    def predict_pressures(
        self, parameters: Sequence[float], production_data: Union[ProductionDataset, pd.DataFrame]
    ) -> np.ndarray:
        dataset = (
            production_data
            if isinstance(production_data, ProductionDataset)
            else prepare_production_dataset(production_data)
        )

        theta = np.asarray(parameters, dtype=float)
        predictor = self.make_predict_function(dataset)
        return predictor(theta.astype(pt_config.floatX))


def _newton_solve(
    initial_guess: pt.TensorVariable,
    N: pt.TensorVariable,
    m: pt.TensorVariable,
    Np_t: pt.TensorVariable,
    Rp_t: pt.TensorVariable,
    context: SolverContext,
) -> pt.TensorVariable:
    """Run a fixed number of Newton iterations in PyTensor graph form."""

    p = initial_guess
    min_bound, max_bound = context.pressure_bounds

    for _ in range(context.newton_steps):
        residual, derivative = _material_balance_residual(
            p, N, m, Np_t, Rp_t, context
        )

        jac_safe = pt.switch(
            pt.abs(derivative) < context.jacobian_epsilon,
            pt.switch(pt.lt(derivative, 0), -context.jacobian_epsilon, context.jacobian_epsilon),
            derivative,
        )

        delta = residual / jac_safe
        p_candidate = p - context.damping * delta
        p_candidate = pt.clip(p_candidate, min_bound, max_bound)

        converged = pt.lt(pt.abs(residual), context.tolerance)
        p = pt.switch(converged, p, p_candidate)
        p = pt.where(pt.isnan(p), initial_guess, p)

    valid = pt.and_(pt.gt(N, 0.0), pt.ge(m, 0.0))
    return pt.switch(valid, p, initial_guess)


def _material_balance_residual(
    pressure: pt.TensorVariable,
    N: pt.TensorVariable,
    m: pt.TensorVariable,
    Np_t: pt.TensorVariable,
    Rp_t: pt.TensorVariable,
    context: SolverContext,
) -> tuple[pt.TensorVariable, pt.TensorVariable]:
    engine = context.engine

    Bo, dBo, Bg, dBg, Rs, dRs = engine.evaluate_all(pressure)

    delta_p = context.initial_pressure - pressure
    comp_coeff = N * context.Boi * context.eff_compressibility
    comp_term = pt.switch(pt.gt(delta_p, 0.0), comp_coeff * delta_p, 0.0)
    dcomp_dp = pt.switch(pt.gt(delta_p, 0.0), -comp_coeff, 0.0)

    lhs = Np_t * (Bo + (Rp_t - Rs) * Bg)
    dL_dp = Np_t * (dBo + (Rp_t - Rs) * dBg - dRs * Bg)

    rhs_fluid = N * ((Bo - context.Boi) + (context.Rsi - Rs) * Bg)
    drhs_fluid = N * (dBo - dRs * Bg + (context.Rsi - Rs) * dBg)

    rhs_gas = N * m * context.Boi * ((Bg / context.Bgi) - 1.0)
    drhs_gas = N * m * context.Boi * (dBg / context.Bgi)

    residual = lhs - (rhs_fluid + rhs_gas + comp_term)
    derivative = dL_dp - (drhs_fluid + drhs_gas + dcomp_dp)

    return residual, derivative