"""Plot drive mechanism indices using MAP parameters from a gas-cap + water-drive run."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.mcmc_pmb.forward_model import MaterialBalanceModel, prepare_production_dataset
from src.mcmc_pmb.pvt import MaterialBalancePVT


def load_map_estimates(summary_path: Path) -> tuple[float, float, float, float]:
    """Read MAP estimates (N, m, J, C) from outputs/summary.csv."""
    if not summary_path.exists():
        raise FileNotFoundError(f"MAP summary not found: {summary_path}")

    summary = pd.read_csv(summary_path)
    required_cols = {"parameter", "MAP"}
    if not required_cols.issubset(summary.columns):
        raise ValueError(
            f"{summary_path} must contain columns {sorted(required_cols)}. "
            f"Found {list(summary.columns)}"
        )

    summary = summary.set_index("parameter")
    needed_params = ["N", "m", "J", "C"]
    missing = [name for name in needed_params if name not in summary.index]
    if missing:
        raise ValueError(f"Missing MAP parameters in {summary_path}: {missing}")

    return tuple(float(summary.loc[name, "MAP"]) for name in needed_params)


def compute_drive_terms(
    data_path: Path,
    map_estimate: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run a forward pass and compute Ef, Eg, We, and F at each time step."""
    data = pd.read_csv(data_path)
    dataset = prepare_production_dataset(data)

    pvt_params = MaterialBalancePVT()
    model = MaterialBalanceModel(pvt_params=pvt_params)
    engine = model.pvt_engine

    N, m, J, C = map_estimate
    N_stb = N * 1.0e6

    Boi = model._context.Boi
    Bgi = model._context.Bgi
    Rsi = model._context.Rsi
    Bw = model._context.Bw
    pi = model._context.initial_pressure
    eff_comp = model._context.eff_compressibility

    ef = np.zeros(dataset.n_steps, dtype=float)
    eg = np.zeros(dataset.n_steps, dtype=float)
    we = np.zeros(dataset.n_steps, dtype=float)
    f_total = np.zeros(dataset.n_steps, dtype=float)

    p_prev = pi
    we_prev = 0.0

    for i in range(dataset.n_steps):
        p_next, we_next = model.solve_single_step(
            initial_pressure=p_prev,
            we_prev=we_prev,
            dt=float(dataset.step_days[i]),
            N=float(N),
            m=float(m),
            Np=float(dataset.Np[i]),
            Rp=float(dataset.Rp[i]),
            Wp=float(dataset.Wp[i]),
            aquifer_index=float(J),
            aquifer_capacity=float(C),
        )

        bo = float(engine.oil_fvf_numpy(p_next)[0])
        bg = float(engine.gas_fvf_numpy(p_next)[0])
        rs = float(engine.solution_gor_numpy(p_next)[0])

        delta_p = pi - p_next
        comp_coeff = N_stb * Boi * (1.0 + m) * eff_comp
        comp_term = comp_coeff * delta_p if delta_p > 0.0 else 0.0

        fluid_term = N_stb * ((bo - Boi) + (Rsi - rs) * bg) + comp_term
        gas_cap_term = N_stb * m * Boi * ((bg / Bgi) - 1.0)
        water_term = we_next

        underground_withdrawal = dataset.Np[i] * (bo + (dataset.Rp[i] - rs) * bg) + (
            dataset.Wp[i] * Bw
        )

        ef[i] = max(fluid_term, 0.0)
        eg[i] = max(gas_cap_term, 0.0)
        we[i] = max(water_term, 0.0)
        f_total[i] = max(underground_withdrawal, 0.0)

        p_prev = p_next
        we_prev = we_next

    return dataset.time_days, ef, eg, we, f_total


def plot_fractional_drive_indices(
    time_days: np.ndarray,
    ef: np.ndarray,
    eg: np.ndarray,
    we: np.ndarray,
    f_total: np.ndarray,
    output_path: Path,
) -> None:
    """Plot 100% stacked area chart of fractional drive contributions."""
    denom = np.where(f_total > 1e-12, f_total, np.nan)

    frac_ef = np.divide(ef, denom)
    frac_eg = np.divide(eg, denom)
    frac_we = np.divide(we, denom)

    fractions = np.vstack([frac_ef, frac_eg, frac_we])
    fractions = np.nan_to_num(fractions, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize to exactly 100% where contributions are non-zero.
    frac_sum = fractions.sum(axis=0)
    valid = frac_sum > 1e-12
    fractions[:, valid] /= frac_sum[valid]

    frac_ef, frac_eg, frac_we = fractions

    c1 = frac_ef
    c2 = frac_ef + frac_eg

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    colors = {
        "fluid": "#475569",  # slate
        "gas": "#4169E1",  # royal blue
        "water": "#F59E0B",  # amber
    }

    ax.fill_between(
        time_days[1:],
        0.0,
        c1[1:],
        color=colors["fluid"],
        alpha=0.9,
        label=r"Fluid Expansion $E_f$",
    )
    ax.fill_between(
        time_days[1:],
        c1[1:],
        c2[1:],
        color=colors["gas"],
        alpha=0.9,
        label=r"Gas Cap Expansion $E_g$",
    )
    ax.fill_between(
        time_days[1:],
        c2[1:],
        1.0,
        color=colors["water"],
        alpha=0.9,
        label=r"Water Influx $W_e$",
    )

    ax.set_title("Drive Mechanism Contributions to Underground Withdrawal", fontsize=13)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Fraction of $F$")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticklabels([f"{int(v * 100)}%" for v in np.linspace(0.0, 1.0, 6)])
    ax.legend(loc="upper left", frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    data_path = root / "data" / "production_data.csv"
    summary_path = root / "outputs" / "summary.csv"
    output_path = root / "outputs" / "drive_indices_stacked.png"

    map_estimate = load_map_estimates(summary_path)
    time_days, ef, eg, we, f_total = compute_drive_terms(data_path, map_estimate)
    plot_fractional_drive_indices(time_days, ef, eg, we, f_total, output_path)

    map_rounded = tuple(float(v) for v in np.round(map_estimate, 4))
    print("MAP estimates (N, m, J, C):", map_rounded)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
