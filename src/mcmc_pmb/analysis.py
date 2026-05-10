"""
Post-processing utilities using ArviZ for high-fidelity scientific diagnostics.
Generates publication-ready visuals with uncertainty quantification and residuals.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import arviz as az
import pytensor.tensor as pt
from pytensor import config as pt_config, function

from .forward_model import ProductionDataset, prepare_production_dataset

# --- Scientific Style Configuration ---
STYLE_CONFIG = {
    "figure.figsize": (10, 6),
    "figure.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.frameon": False,
    "legend.fontsize": 10,
    "lines.linewidth": 1.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}

COLORS = {
    "data": "#1f2937",  # Dark slate
    "model_median": "#2563eb",  # Royal Blue
    "band_50": "#60a5fa",  # Lighter Blue
    "band_95": "#bfdbfe",  # Very Light Blue
    "residual": "#ef4444",  # Red
    "map": "#d97706",  # Amber
}


def _apply_style():
    """Applies custom scientific plotting style."""
    plt.rcParams.update(STYLE_CONFIG)


def _flatten_posterior_variable(idata: Any, var_name: str) -> np.ndarray | None:
    """Return posterior variable flattened as (samples, ...), or None if unavailable."""
    if idata is None or not hasattr(idata, "posterior"):
        return None
    if var_name not in idata.posterior:
        return None

    values = np.asarray(idata.posterior[var_name].values)
    if values.ndim < 2:
        return None

    return values.reshape(-1, *values.shape[2:])


def plot_posterior_predictive(
    chain: np.ndarray,
    model: Any,
    production_data: pd.DataFrame | ProductionDataset,
    output_dir: Path,
    idata: Any | None = None,
    n_curves: int = 500,
) -> None:
    """
    Generates a dual-panel scientific plot:
    1. Upper: Measurement vs Model with Uncertainty Bands (50% and 95% CI).
    2. Lower: Residuals (Data - Model Median).
    """
    _apply_style()
    dataset = prepare_production_dataset(production_data)
    chain_for_plot = chain

    theta_from_idata = _flatten_posterior_variable(idata, "theta")
    if (
        theta_from_idata is not None
        and theta_from_idata.ndim == 2
        and theta_from_idata.shape[1] == 4
    ):
        chain_for_plot = theta_from_idata

    n_samples = chain_for_plot.shape[0]
    if n_samples == 0:
        return

    # Select random indices for the ensemble
    rng = np.random.default_rng()
    indices = rng.choice(n_samples, size=min(n_curves, n_samples), replace=False)

    Np_true_samples = _flatten_posterior_variable(idata, "Np_true")
    Rp_true_samples = _flatten_posterior_variable(idata, "Rp_true")
    use_latent_sequences = (
        Np_true_samples is not None
        and Rp_true_samples is not None
        and Np_true_samples.shape[0] == n_samples
        and Rp_true_samples.shape[0] == n_samples
        and Np_true_samples.ndim == 2
        and Rp_true_samples.ndim == 2
        and Np_true_samples.shape[1] == dataset.n_steps
        and Rp_true_samples.shape[1] == dataset.n_steps
    )

    if use_latent_sequences:
        theta = pt.vector("theta", dtype=pt_config.floatX)
        np_seq = pt.vector("Np_seq", dtype=pt_config.floatX)
        rp_seq = pt.vector("Rp_seq", dtype=pt_config.floatX)
        pressures = model.symbolic_pressures(
            theta[0],
            theta[1],
            theta[2],
            theta[3],
            dataset,
            Np_seq=np_seq,
            Rp_seq=rp_seq,
        )
        predict_fn = function([theta, np_seq, rp_seq], pressures)
    else:
        predict_fn = model.make_predict_function(dataset)

    # Generate ensemble predictions
    preds = np.empty((len(indices), dataset.n_steps))
    for i, idx in enumerate(indices):
        theta = chain_for_plot[idx].astype(pt_config.floatX)
        if use_latent_sequences:
            np_true = Np_true_samples[idx].astype(pt_config.floatX)
            rp_true = Rp_true_samples[idx].astype(pt_config.floatX)
            preds[i] = predict_fn(theta, np_true, rp_true)
        else:
            preds[i] = predict_fn(theta)

    # Calculate Statistics (Median, 50% CI, 95% CI)
    p50 = np.median(preds, axis=0)
    p2_5 = np.percentile(preds, 2.5, axis=0)
    p97_5 = np.percentile(preds, 97.5, axis=0)
    p25 = np.percentile(preds, 25, axis=0)
    p75 = np.percentile(preds, 75, axis=0)

    t = dataset.time_days
    y_obs = dataset.pressure_measured

    # Setup Grid Layout (Main plot + Residuals)
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_main = plt.subplot(gs[0])
    ax_resid = plt.subplot(gs[1], sharex=ax_main)

    # --- Main Plot ---
    # 95% Confidence Interval
    ax_main.fill_between(
        t, p2_5, p97_5, color=COLORS["band_95"], alpha=0.6, label="95% CI"
    )
    # 50% Confidence Interval
    ax_main.fill_between(
        t, p25, p75, color=COLORS["band_50"], alpha=0.8, label="50% CI"
    )
    # Median Line
    ax_main.plot(t, p50, color=COLORS["model_median"], lw=2, label="Posterior Median")
    # Observed Data
    ax_main.errorbar(
        t,
        y_obs,
        yerr=0.0,
        fmt="o",
        color=COLORS["data"],
        markersize=4,
        label="Measured Data",
        elinewidth=1,
        zorder=10,
    )

    ax_main.set_ylabel("Pressure (psi)")
    ax_main.set_title(
        "Posterior Predictive Check & Uncertainty Quantification", fontweight="bold"
    )
    ax_main.legend(loc="upper right")
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # --- Residual Plot ---
    residuals = y_obs - p50
    ax_resid.scatter(t, residuals, color=COLORS["residual"], s=15, alpha=0.7)
    ax_resid.axhline(0, color="black", lw=1, linestyle="--")
    ax_resid.set_ylabel("Residuals")
    ax_resid.set_xlabel("Time (days)")

    # Add RMSE annotation
    rmse = np.sqrt(np.mean(residuals**2))
    ax_resid.text(
        0.02, 0.05, f"RMSE: {rmse:.2f} psi", transform=ax_resid.transAxes, fontsize=10
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "ppc_scientific.png", bbox_inches="tight")
    plt.close(fig)


def summarize_chain(
    chain: np.ndarray,
    parameter_names: Sequence[str] = ("N", "m", "J", "C"),
    map_estimate: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Generates summary statistics with ArviZ."""
    data_dict = {name: chain[:, i] for i, name in enumerate(parameter_names)}
    # Convert to InferenceData for robust handling
    idata = az.from_dict(posterior=data_dict)

    summary = az.summary(idata, kind="stats", hdi_prob=0.90)

    if map_estimate is not None:
        summary["MAP"] = map_estimate

    return summary.reset_index().rename(columns={"index": "parameter"})


def estimate_map(chain: np.ndarray, log_posteriors: np.ndarray) -> np.ndarray:
    max_idx = np.argmax(log_posteriors)
    return chain[max_idx]


def make_diagnostics(
    chain: np.ndarray,
    output_dir: Path,
    log_posteriors: Optional[np.ndarray] = None,
    map_estimate: Optional[Sequence[float]] = None,
    parameter_names: Sequence[str] = ("N", "m", "J", "C"),
) -> None:
    """
    Generate advanced ArviZ diagnostic plots with customized scientific styling.
    Includes: Trace plots, Posterior densities, and Joint KDE Pair plots.
    """
    _apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert flat chain to ArviZ InferenceData
    # Reshape (N_samples, N_params) -> (1, N_samples, N_params) for single chain handling
    data_dict = {name: chain[None, :, i] for i, name in enumerate(parameter_names)}
    dataset = az.from_dict(posterior=data_dict)

    # 1. Trace Plot (History + Density)
    # We use a compact layout
    axes = az.plot_trace(
        dataset,
        compact=False,
        lines=[("mean", {}, "C1")] if map_estimate is None else None,
        plot_kwargs={"color": COLORS["model_median"], "alpha": 0.8},
        fill_kwargs={"color": COLORS["band_95"], "alpha": 0.4},
    )

    # Customize ArviZ output (which returns numpy array of axes)
    fig = axes.flatten()[0].figure
    fig.suptitle("MCMC Trace & Marginal Densities", fontsize=16, y=1.02)
    fig.savefig(output_dir / "trace_plots.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Joint Distribution (Pair Plot) with KDE Contours
    # This replaces simple scatter plots with probability mass contours
    ax = az.plot_pair(
        dataset,
        kind="kde",
        marginals=True,
        point_estimate="median",
        kde_kwargs={
            "fill_last": False,
            "contourf_kwargs": {"cmap": "Blues"},
            "hdi_probs": [0.3, 0.6, 0.9],  # Plot 30, 60, 90% probability masses
        },
        marginal_kwargs={"color": COLORS["data"]},
        textsize=12,
        figsize=(8, 8),
    )
    plt.gcf().suptitle("Joint Posterior Density", y=1.02, fontsize=14)
    plt.gcf().savefig(output_dir / "joint_density_contours.png", bbox_inches="tight")
    plt.close()

    # 3. Posterior Marginals with HDI
    ax = az.plot_posterior(
        dataset,
        hdi_prob=0.95,
        point_estimate="mean",
        color=COLORS["band_50"],
        textsize=11,
        ref_val=None,
    )
    plt.gcf().suptitle("Posterior Marginals (95% HDI)", y=1.05, fontsize=14)
    plt.gcf().savefig(output_dir / "posterior_marginals.png", bbox_inches="tight")
    plt.close()

    # 4. Log Posterior Convergence
    if log_posteriors is not None:
        fig, ax = plt.subplots(figsize=(10, 4))

        # Raw trace
        ax.plot(log_posteriors, color="gray", lw=0.5, alpha=0.4, label="Log Prob")

        # Rolling mean (to show convergence trend)
        window = min(len(log_posteriors) // 20, 500)
        if window > 1:
            rolling_mean = pd.Series(log_posteriors).rolling(window).mean()
            ax.plot(
                rolling_mean,
                color=COLORS["data"],
                lw=1.5,
                label=f"Rolling Mean (n={window})",
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Log Posterior")
        ax.set_title("Optimization / Sampling Convergence", fontweight="bold")
        ax.legend()

        fig.savefig(output_dir / "log_posterior_convergence.png", bbox_inches="tight")
        plt.close(fig)


def plot_probabilistic_drive_indices(
    chain: np.ndarray,
    model: Any,
    production_data: pd.DataFrame | ProductionDataset,
    output_dir: Path,
    n_curves: int = 500,
) -> None:
    """
    Computes Probabilistic Drive Indices (DDI, SDI, WDI, EDI)
    to visualize the uncertainty of reservoir energy sources over time.
    """
    _apply_style()
    dataset = prepare_production_dataset(production_data)
    predict_fn = model.make_predict_function(dataset)

    n_samples = chain.shape[0]
    indices = np.random.choice(n_samples, size=min(n_curves, n_samples), replace=False)

    ddi_all, sdi_all, wdi_all, edi_all = [], [], [], []

    for idx in indices:
        theta = chain[idx]
        N, m, J, C = theta[:4]
        N_stb = N * 1.0e6

        pi = model._context.initial_pressure
        boi, bgi, rsi = model._context.Boi, model._context.Bgi, model._context.Rsi
        ceff = model._context.eff_compressibility

        pressures = predict_fn(theta.astype(pt_config.floatX))
        bo, _ = model.pvt_engine.oil_fvf_numpy(pressures)
        bg, _ = model.pvt_engine.gas_fvf_numpy(pressures)
        rs, _ = model.pvt_engine.solution_gor_numpy(pressures)

        rhs_fluid = N_stb * ((bo - boi) + (rsi - rs) * bg)
        rhs_gas = N_stb * m * boi * ((bg / bgi) - 1.0)
        comp_term = N_stb * boi * (1.0 + m) * ceff * np.maximum(pi - pressures, 0.0)

        we_total = np.zeros(dataset.n_steps)
        we_prev = 0.0
        for i in range(1, dataset.n_steps):
            dt = dataset.step_days[i]
            p_a_prev = pi - we_prev / max(C, 1e-6)
            decline = np.exp(-J * dt / max(C, 1e-6))
            p_a_new = pressures[i] + (p_a_prev - pressures[i]) * decline
            we_total[i] = we_prev + C * (p_a_prev - p_a_new)
            we_prev = we_total[i]

        total_exp = np.maximum(rhs_fluid + rhs_gas + comp_term + we_total, 1e-6)

        ddi_all.append(rhs_fluid / total_exp)
        sdi_all.append(rhs_gas / total_exp)
        wdi_all.append(we_total / total_exp)
        edi_all.append(comp_term / total_exp)

    t = dataset.time_days
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    drives = [
        ("Depletion Drive (DDI)", ddi_all, "#22c55e"),
        ("Water Drive (WDI)", wdi_all, "#3b82f6"),
        ("Gas Cap Drive (SDI)", sdi_all, "#ef4444"),
        ("Compaction/Rock (EDI)", edi_all, "#64748b"),
    ]

    for ax, (title, data, color) in zip(axes, drives):
        data_arr = np.array(data)
        p50 = np.median(data_arr, axis=0)
        p05 = np.percentile(data_arr, 5, axis=0)
        p95 = np.percentile(data_arr, 95, axis=0)

        ax.fill_between(t[1:], p05[1:], p95[1:], color=color, alpha=0.3, label="90% CI")
        ax.plot(t[1:], p50[1:], color=color, lw=2, label="Median")

        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction")
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (days)")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "probabilistic_drive_indices.png", bbox_inches="tight")
    plt.close(fig)

def plot_N_m_joint(
    chain: np.ndarray,
    output_dir: Path,
    parameter_names: Sequence[str] = ("N", "m", "J", "C"),
) -> None:
    """
    Generates a high-quality 2D joint plot of parameters $N$ and $m$.
    Uses a hexbin plot to handle high-density MCMC samples effectively.
    """
    _apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract N and m assuming standard ordering
    idx_N = parameter_names.index("N")
    idx_m = parameter_names.index("m")
    
    N_samples = chain[:, idx_N]
    m_samples = chain[:, idx_m]

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Hexbin plot for density
    hb = ax.hexbin(
        N_samples, 
        m_samples, 
        gridsize=50, 
        cmap="Blues", 
        mincnt=1,
        alpha=0.9
    )
    
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Sample Density")

    ax.set_xlabel("Original Oil in Place ($N$)")
    ax.set_ylabel("Gas Cap Ratio ($m$)")
    ax.set_title("Joint Posterior Density: $N$ vs $m$", fontweight="bold")
    
    # Add median marker
    ax.scatter(
        np.median(N_samples), 
        np.median(m_samples), 
        color=COLORS["residual"], 
        marker="X", 
        s=100, 
        label="Median",
        edgecolor="white",
        linewidth=1.5
    )
    ax.legend(loc="upper right")

    fig.savefig(output_dir / "joint_N_m.png", bbox_inches="tight")
    plt.close(fig)


def plot_N_m_J_3d(
    chain: np.ndarray,
    output_dir: Path,
    parameter_names: Sequence[str] = ("N", "m", "J", "C"),
    n_points: int = 2000,
) -> None:
    """
    Generates a 3D scatter plot of the joint posterior for $N$, $m$, and $J$.
    Subsamples the chain for rendering performance.
    """
    _apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    idx_N = parameter_names.index("N")
    idx_m = parameter_names.index("m")
    idx_J = parameter_names.index("J")

    # Subsample for 3D plot clarity
    n_samples = chain.shape[0]
    if n_samples > n_points:
        indices = np.random.choice(n_samples, size=n_points, replace=False)
        chain_sub = chain[indices]
    else:
        chain_sub = chain

    N_samples = chain_sub[:, idx_N]
    m_samples = chain_sub[:, idx_m]
    J_samples = chain_sub[:, idx_J]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Color by depth (J parameter) to enhance 3D perception
    sc = ax.scatter(
        N_samples, 
        m_samples, 
        J_samples, 
        c=J_samples, 
        cmap="viridis", 
        alpha=0.5,
        s=15,
        edgecolor='none'
    )

    cb = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.7)
    cb.set_label("Productivity Index ($J$)")

    ax.set_xlabel("Original Oil in Place ($N$)")
    ax.set_ylabel("Gas Cap Ratio ($m$)")
    ax.set_zlabel("Productivity Index ($J$)")
    ax.set_title("3D Joint Posterior: $N$, $m$, and $J$", fontweight="bold")

    # Adjust viewing angle
    ax.view_init(elev=25, azim=45)

    fig.savefig(output_dir / "joint_3d_N_m_J.png", bbox_inches="tight")
    plt.close(fig)
