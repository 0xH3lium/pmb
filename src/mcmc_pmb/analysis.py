"""Post-processing utilities using ArviZ for standard MCMC diagnostics."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import arviz as az
from pytensor import config as pt_config

from .forward_model import ProductionDataset, prepare_production_dataset

def plot_posterior_predictive(
    chain: np.ndarray,
    model: Any,
    production_data: pd.DataFrame | ProductionDataset,
    output_dir: Path,
    n_curves: int = 100
) -> None:
    dataset = prepare_production_dataset(production_data)
    n_samples = chain.shape[0]
    if n_samples == 0: return

    # Vectorized sampling of indices
    rng = np.random.default_rng()
    indices = rng.choice(n_samples, size=min(n_curves, n_samples), replace=False)
    
    predict_fn = model.make_predict_function(dataset)

    preds = np.empty((len(indices), dataset.n_steps))
    
    for i, idx in enumerate(indices):
        theta = chain[idx].astype(pt_config.floatX)
        preds[i] = predict_fn(theta)

    fig, ax = plt.subplots(figsize=(10, 6))
    t = dataset.time_days
    
    # Plot ensemble
    ax.plot(t, preds.T, color="steelblue", alpha=0.1)
    # Plot data
    ax.scatter(t, dataset.pressure_measured, color="black", zorder=5, label="Measured", s=20)
    
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Pressure (psi)")
    ax.set_title("Posterior Predictive Check")
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(output_dir / "ppc.png", dpi=150)
    plt.close(fig)

def summarize_chain(
    chain: np.ndarray,
    parameter_names: Sequence[str] = ("N", "m"),
    map_estimate: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Uses ArviZ to generate summary statistics."""
    # Convert to dictionary for ArviZ
    data_dict = {name: chain[:, i] for i, name in enumerate(parameter_names)}
    summary = az.summary(data_dict, kind="stats", hdi_prob=0.80)
    
    if map_estimate is not None:
        summary["map"] = map_estimate
        
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
    """Generate ArviZ diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create ArviZ inference data structure from numpy chain
    # Shape required: (chains, draws, parameters) -> We have flat chain so (1, N, 2)
    data_dict = {name: chain[:, i] for i, name in enumerate(parameter_names)}
    dataset = az.convert_to_dataset(data_dict)
    
    # Trace Plot
    axes = az.plot_trace(dataset)
    plt.gcf().savefig(output_dir / "trace_plots.png", dpi=150)
    plt.close()

    # Posterior Density
    axes = az.plot_posterior(dataset, point_estimate="mean", hdi_prob=0.95)
    plt.gcf().savefig(output_dir / "posterior_marginals.png", dpi=150)
    plt.close()

    # Pair Plot (Joint density)
    axes = az.plot_pair(dataset, kind="kde", marginals=True)
    plt.gcf().savefig(output_dir / "joint_density.png", dpi=150)
    plt.close()

    if log_posteriors is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(log_posteriors, color="black", lw=0.5)
        ax.set_title("Log Posterior Trace")
        fig.tight_layout()
        fig.savefig(output_dir / "log_posterior.png", dpi=150)
        plt.close()