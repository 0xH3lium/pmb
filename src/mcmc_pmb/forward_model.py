"""Vectorised, differentiable forward material-balance model."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    Wp: np.ndarray
    step_days: np.ndarray
    pressure_measured: np.ndarray
    pressure_truth: Optional[np.ndarray] = None

    @property
    def n_steps(self) -> int:
        return self.time_days.size


def prepare_production_dataset(
    data: Union[pd.DataFrame, ProductionDataset],
) -> ProductionDataset:
    if isinstance(data, ProductionDataset):
        return data

    required = {"time_days", "Np", "Rp", "Pressure_measured"}
    if not required.issubset(data.columns):
        missing = required - set(data.columns)
        raise ValueError(f"Missing columns: {missing}")

    df = (
        data.sort_values("time_days")
        if not data["time_days"].is_monotonic_increasing
        else data
    )

    time_days = df["time_days"].to_numpy(dtype=float)
    step_days = np.diff(time_days, prepend=time_days[0])
    step_days = np.maximum(step_days, 0.0)

    if "Wp" in df:
        wp = df["Wp"].to_numpy(dtype=float)
    else:
        wp = np.zeros_like(time_days, dtype=float)

    return ProductionDataset(
        time_days=time_days,
        Np=df["Np"].to_numpy(dtype=float),
        Rp=df["Rp"].to_numpy(dtype=float),
        Wp=wp,
        step_days=step_days,
        pressure_measured=df["Pressure_measured"].to_numpy(dtype=float),
        pressure_truth=df["Pressure_truth"].to_numpy(dtype=float)
        if "Pressure_truth" in df
        else None,
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
    aquifer_index: float
    aquifer_capacity: float
    Bw: float


@dataclass(frozen=True, slots=True)
class MaterialBalanceModel:
    pvt_params: MaterialBalancePVT = field(default_factory=MaterialBalancePVT)
    pvt_engine: Optional[PVTEngine] = None
    connate_water_saturation: float = 0.20
    pore_compressibility: float = 4.0e-6
    water_compressibility: float = 3.0e-6
    aquifer_index: float = 0.0
    aquifer_capacity: float = 1.0e6
    water_fvf: float = 1.0
    pressure_bounds: tuple[float, float] = (100.0, 5000.0)
    newton_steps: int = 6
    newton_damping: float = 0.8
    newton_tol: float = 1e-4
    jacobian_epsilon: float = 1e-9

    _context: SolverContext = field(init=False, repr=False)
    _single_step_fn: callable = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        engine = self.pvt_engine or self.pvt_params.build_engine()
        lower_bound = max(self.pvt_params.pressure_min, self.pressure_bounds[0])
        upper_bound = min(self.pvt_params.pressure_max, self.pressure_bounds[1])
        if lower_bound >= upper_bound:
            raise ValueError(
                "Invalid pressure bounds after intersecting with PVT limits."
            )

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
            aquifer_index=self.aquifer_index,
            aquifer_capacity=self.aquifer_capacity,
            Bw=self.water_fvf,
        )

        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "pvt_engine", engine)

    def symbolic_pressures(
        self,
        N: pt.TensorVariable,
        m: pt.TensorVariable,
        aquifer_index: pt.TensorVariable,
        aquifer_capacity: pt.TensorVariable,
        dataset: ProductionDataset,
        Np_seq: pt.TensorVariable | None = None,
        Rp_seq: pt.TensorVariable | None = None,
    ) -> pt.TensorVariable:
        """Return PyTensor graph for predicted pressures given symbolic parameters."""

        dtype = pt_config.floatX
        # Allow overriding production sequences with latent variables
        Np = (
            pt.as_tensor_variable(dataset.Np.astype(dtype))
            if Np_seq is None
            else pt.as_tensor_variable(Np_seq)
        )
        Rp = (
            pt.as_tensor_variable(dataset.Rp.astype(dtype))
            if Rp_seq is None
            else pt.as_tensor_variable(Rp_seq)
        )
        Wp = pt.as_tensor_variable(dataset.Wp.astype(dtype))
        step_days = pt.as_tensor_variable(dataset.step_days.astype(dtype))

        context = replace(
            self._context,
            aquifer_index=aquifer_index,
            aquifer_capacity=aquifer_capacity,
        )

        def step(
            np_t,
            rp_t,
            wp_t,
            dt_t,
            prev_pressure,
            prev_we,
            N_param,
            m_param,
            aquifer_J,
            aquifer_C,
        ):
            next_pressure, next_we = _newton_solve(
                prev_pressure,
                prev_we,
                dt_t,
                N_param,
                m_param,
                np_t,
                rp_t,
                wp_t,
                context,
                aquifer_J,
                aquifer_C,
            )
            return next_pressure, next_we

        initial_pressure_tensor = pt.as_tensor_variable(
            np.array(context.initial_pressure, dtype=dtype)
        )
        initial_we_tensor = pt.as_tensor_variable(np.array(0.0, dtype=dtype))

        outputs, _ = scan(
            step,
            sequences=[Np, Rp, Wp, step_days],
            outputs_info=[initial_pressure_tensor, initial_we_tensor],
            non_sequences=[N, m, aquifer_index, aquifer_capacity],
            strict=False,
        )

        pressures, _ = outputs
        return pressures

    def make_predict_function(self, dataset: ProductionDataset):
        theta = pt.vector("theta", dtype=pt_config.floatX)
        pressures = self.symbolic_pressures(
            theta[0], theta[1], theta[2], theta[3], dataset
        )
        return function([theta], pressures)

    def predict_pressures(
        self,
        parameters: Sequence[float],
        production_data: Union[ProductionDataset, pd.DataFrame],
    ) -> np.ndarray:
        dataset = (
            production_data
            if isinstance(production_data, ProductionDataset)
            else prepare_production_dataset(production_data)
        )

        theta = np.asarray(parameters, dtype=float)
        if theta.size != 4:
            raise ValueError(
                "MaterialBalanceModel expects parameter vector of length 4 (N, m, J, C)."
            )
        predictor = self.make_predict_function(dataset)
        return predictor(theta.astype(pt_config.floatX))

    def _build_single_step_fn(self):
        if self._single_step_fn is not None:
            return self._single_step_fn

        dtype = pt_config.floatX
        p_init = pt.scalar("p_init", dtype=dtype)
        we_prev = pt.scalar("we_prev", dtype=dtype)
        dt = pt.scalar("dt", dtype=dtype)
        N = pt.scalar("N", dtype=dtype)
        m = pt.scalar("m", dtype=dtype)
        Np_t = pt.scalar("Np_t", dtype=dtype)
        Rp_t = pt.scalar("Rp_t", dtype=dtype)
        Wp_t = pt.scalar("Wp_t", dtype=dtype)
        aquifer_J = pt.scalar("aquifer_J", dtype=dtype)
        aquifer_C = pt.scalar("aquifer_C", dtype=dtype)

        p_next, we_next = _newton_solve(
            p_init,
            we_prev,
            dt,
            N,
            m,
            Np_t,
            Rp_t,
            Wp_t,
            self._context,
            aquifer_J,
            aquifer_C,
        )
        fn = function(
            [p_init, we_prev, dt, N, m, Np_t, Rp_t, Wp_t, aquifer_J, aquifer_C],
            [p_next, we_next],
        )
        object.__setattr__(self, "_single_step_fn", fn)
        return fn

    def solve_single_step(
        self,
        initial_pressure: float,
        we_prev: float,
        dt: float,
        N: float,
        m: float,
        Np: float,
        Rp: float,
        Wp: float,
        aquifer_index: float | None = None,
        aquifer_capacity: float | None = None,
    ) -> tuple[float, float]:
        aquifer_J = aquifer_index if aquifer_index is not None else self.aquifer_index
        aquifer_C = (
            aquifer_capacity if aquifer_capacity is not None else self.aquifer_capacity
        )
        fn = self._build_single_step_fn()
        p_next, we_next = fn(
            float(initial_pressure),
            float(we_prev),
            float(dt),
            float(N),
            float(m),
            float(Np),
            float(Rp),
            float(Wp),
            float(aquifer_J),
            float(aquifer_C),
        )
        return float(p_next), float(we_next)


def _newton_solve(
    initial_guess: pt.TensorVariable,
    we_prev: pt.TensorVariable,
    dt: pt.TensorVariable,
    N: pt.TensorVariable,
    m: pt.TensorVariable,
    Np_t: pt.TensorVariable,
    Rp_t: pt.TensorVariable,
    Wp_t: pt.TensorVariable,
    context: SolverContext,
    aquifer_index: pt.TensorVariable,
    aquifer_capacity: pt.TensorVariable,
) -> tuple[pt.TensorVariable, pt.TensorVariable]:
    """Run a fixed number of Newton iterations in PyTensor graph form."""

    p = initial_guess
    min_bound, max_bound = context.pressure_bounds

    for _ in range(context.newton_steps):
        residual, derivative, _ = _material_balance_residual(
            p,
            N,
            m,
            Np_t,
            Rp_t,
            Wp_t,
            we_prev,
            dt,
            context,
            aquifer_index,
            aquifer_capacity,
        )

        jac_safe = pt.switch(
            pt.abs(derivative) < context.jacobian_epsilon,
            pt.switch(
                pt.lt(derivative, 0),
                -context.jacobian_epsilon,
                context.jacobian_epsilon,
            ),
            derivative,
        )

        delta = residual / jac_safe
        p_candidate = p - context.damping * delta
        p_candidate = pt.clip(p_candidate, min_bound, max_bound)

        converged = pt.lt(pt.abs(residual), context.tolerance)
        p = pt.switch(converged, p, p_candidate)
        p = pt.where(pt.isnan(p), initial_guess, p)

    valid = pt.and_(pt.gt(N, 0.0), pt.ge(m, 0.0))
    pressure_final = pt.switch(valid, p, initial_guess)
    _, _, we_total = _material_balance_residual(
        pressure_final,
        N,
        m,
        Np_t,
        Rp_t,
        Wp_t,
        we_prev,
        dt,
        context,
        aquifer_index,
        aquifer_capacity,
    )

    we_final = pt.switch(valid, we_total, we_prev)
    return pressure_final, we_final


def _material_balance_residual(
    pressure: pt.TensorVariable,
    N: pt.TensorVariable,
    m: pt.TensorVariable,
    Np_t: pt.TensorVariable,
    Rp_t: pt.TensorVariable,
    Wp_t: pt.TensorVariable,
    We_accumulated: pt.TensorVariable,
    dt: pt.TensorVariable,
    context: SolverContext,
    aquifer_index: pt.TensorVariable,
    aquifer_capacity: pt.TensorVariable,
) -> tuple[pt.TensorVariable, pt.TensorVariable, pt.TensorVariable]:
    engine = context.engine

    Bo, dBo, Bg, dBg, Rs, dRs = engine.evaluate_all(pressure)

    # Scale N from MMSTB to STB
    N_stb = N * 1.0e6

    delta_p = context.initial_pressure - pressure
    comp_coeff = N_stb * context.Boi * (1.0 + m) * context.eff_compressibility
    comp_term = pt.switch(pt.gt(delta_p, 0.0), comp_coeff * delta_p, 0.0)
    dcomp_dp = pt.switch(pt.gt(delta_p, 0.0), -comp_coeff, 0.0)

    lhs = Np_t * (Bo + (Rp_t - Rs) * Bg) + Wp_t * context.Bw
    dL_dp = Np_t * (dBo + (Rp_t - Rs) * dBg - dRs * Bg)

    rhs_fluid = N_stb * ((Bo - context.Boi) + (context.Rsi - Rs) * Bg)
    drhs_fluid = N_stb * (dBo - dRs * Bg + (context.Rsi - Rs) * dBg)

    rhs_gas = N_stb * m * context.Boi * ((Bg / context.Bgi) - 1.0)
    drhs_gas = N_stb * m * context.Boi * (dBg / context.Bgi)

    # Fetkovich aquifer model
    # p_a_prev = pi - We_prev / C
    # decline_factor = exp(-J * dt / C)
    # p_a_new = p_res + (p_a_prev - p_res) * decline_factor
    # delta_We = C * (p_a_prev - p_a_new)
    C_safe = pt.maximum(aquifer_capacity, 1e-6)
    J_safe = pt.maximum(aquifer_index, 0.0)

    p_a_prev = context.initial_pressure - We_accumulated / C_safe
    decline_factor = pt.exp(-J_safe * dt / C_safe)
    p_a_new = pressure + (p_a_prev - pressure) * decline_factor
    delta_we = C_safe * (p_a_prev - p_a_new)

    we_total = We_accumulated + delta_we
    residual = lhs - (rhs_fluid + rhs_gas + comp_term + we_total)

    # dWe_dp = -C * (1 - exp(-J * dt / C))
    dWe_dp = -C_safe * (1.0 - decline_factor)
    derivative = dL_dp - (drhs_fluid + drhs_gas + dcomp_dp + dWe_dp)

    return residual, derivative, we_total
