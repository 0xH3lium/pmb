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
from pytensor import config as pt_config

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
    "data": "#1f2937",        # Dark slate
    "model_median": "#2563eb", # Royal Blue
    "band_50": "#60a5fa",     # Lighter Blue
    "band_95": "#bfdbfe",     # Very Light Blue
    "residual": "#ef4444",    # Red
    "map": "#d97706",         # Amber
}

def _apply_style():
    """Applies custom scientific plotting style."""
    plt.rcParams.update(STYLE_CONFIG)

def plot_posterior_predictive(
    chain: np.ndarray,
    model: Any,
    production_data: pd.DataFrame | ProductionDataset,
    output_dir: Path,
    n_curves: int = 500
) -> None:
    """
    Generates a dual-panel scientific plot:
    1. Upper: Measurement vs Model with Uncertainty Bands (50% and 95% CI).
    2. Lower: Residuals (Data - Model Median).
    """
    _apply_style()
    dataset = prepare_production_dataset(production_data)
    n_samples = chain.shape[0]
    if n_samples == 0: return

    # Select random indices for the ensemble
    rng = np.random.default_rng()
    indices = rng.choice(n_samples, size=min(n_curves, n_samples), replace=False)
    
    predict_fn = model.make_predict_function(dataset)
    
    # Generate ensemble predictions
    preds = np.empty((len(indices), dataset.n_steps))
    for i, idx in enumerate(indices):
        theta = chain[idx].astype(pt_config.floatX)
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
    ax_main.fill_between(t, p2_5, p97_5, color=COLORS["band_95"], alpha=0.6, label="95% CI")
    # 50% Confidence Interval
    ax_main.fill_between(t, p25, p75, color=COLORS["band_50"], alpha=0.8, label="50% CI")
    # Median Line
    ax_main.plot(t, p50, color=COLORS["model_median"], lw=2, label="Posterior Median")
    # Observed Data
    ax_main.errorbar(
        t, y_obs, yerr=0.0, fmt='o', color=COLORS["data"], 
        markersize=4, label="Measured Data", elinewidth=1, zorder=10
    )

    ax_main.set_ylabel("Pressure (psi)")
    ax_main.set_title("Posterior Predictive Check & Uncertainty Quantification", fontweight="bold")
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
    ax_resid.text(0.02, 0.05, f"RMSE: {rmse:.2f} psi", transform=ax_resid.transAxes, fontsize=10)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "ppc_scientific.png", bbox_inches="tight")
    plt.close(fig)

def summarize_chain(
    chain: np.ndarray,
    parameter_names: Sequence[str] = ("N", "m"),
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
    parameter_names: Sequence[str] = ("N", "m"),
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
        fill_kwargs={"color": COLORS["band_95"], "alpha": 0.4}
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
            "hdi_probs": [0.3, 0.6, 0.9] # Plot 30, 60, 90% probability masses
        },
        marginal_kwargs={"color": COLORS["data"]},
        textsize=12,
        figsize=(8, 8)
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
        ref_val=None
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
            ax.plot(rolling_mean, color=COLORS["data"], lw=1.5, label=f"Rolling Mean (n={window})")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Log Posterior")
        ax.set_title("Optimization / Sampling Convergence", fontweight="bold")
        ax.legend()
        
        fig.savefig(output_dir / "log_posterior_convergence.png", bbox_inches="tight")
        plt.close(fig)