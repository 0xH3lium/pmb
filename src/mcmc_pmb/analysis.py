"""Post-processing utilities for MCMC diagnostics and visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .forward_model import ProductionDataset, prepare_production_dataset

sns.set_style("whitegrid")


def plot_posterior_predictive(
    chain: np.ndarray,
    model: Any,
    production_data: pd.DataFrame | ProductionDataset,
    output_dir: Path,
    n_curves: int = 100
) -> None:
    dataset = prepare_production_dataset(production_data)
    rng = np.random.default_rng()
    
    n_samples = chain.shape[0]
    if n_samples == 0: return

    indices = rng.choice(n_samples, size=min(n_curves, n_samples), replace=False)
    
    # Pre-allocate array for speed
    preds = np.empty((len(indices), dataset.n_steps))
    
    # Simple loop is fine here as it's only 100 iterations, 
    # but using the optimized model helps.
    for i, idx in enumerate(indices):
        preds[i] = model.predict_pressures(chain[idx], dataset)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot curves as a collection for better rendering performance than individual plot calls
    t = dataset.time_days
    ax.plot(t, preds.T, color="steelblue", alpha=0.1)
    
    ax.scatter(t, dataset.pressure_measured, color="black", zorder=5, label="Measured", s=20)
    
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Pressure (psi)")
    ax.set_title("Posterior Predictive Check")
    ax.legend(loc="upper right")
    
    fig.tight_layout()
    fig.savefig(output_dir / "ppc.png", dpi=200)
    plt.close(fig)
    
def summarize_chain(
    chain: np.ndarray,
    parameter_names: Iterable[str] = ("N", "m"),
    map_estimate: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Return a summary table with mean, std, and selected percentiles."""

    if chain.ndim != 2 or chain.shape[1] != 2:
        raise ValueError("Expected a chain with shape (n_samples, 2)")

    names = list(parameter_names)
    if len(names) != chain.shape[1]:
        raise ValueError("parameter_names must have length equal to number of columns")

    map_vector = None
    if map_estimate is not None:
        map_vector = np.asarray(map_estimate, dtype=float)
        if map_vector.shape != (chain.shape[1],):
            raise ValueError("map_estimate must be a 1-D sequence aligned with parameter_names")

    quantiles = [0.1, 0.5, 0.9]
    summary_rows = []
    for idx, name in enumerate(names):
        samples = chain[:, idx]
        row = {
            "parameter": name,
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples, ddof=1)),
        }
        for q in quantiles:
            row[f"p{int(q*100)}"] = float(np.quantile(samples, q))
        if map_vector is not None:
            row["map"] = float(map_vector[idx])
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def estimate_map(chain: np.ndarray, log_posteriors: np.ndarray) -> np.ndarray:
    """Return the Maximum A Posteriori (MAP) estimate from the sampled chain."""

    if chain.shape[0] != log_posteriors.shape[0]:
        raise ValueError("chain and log_posteriors must have matching lengths")
    if chain.ndim != 2:
        raise ValueError("chain must be a 2-D array of samples")
    if log_posteriors.ndim != 1:
        raise ValueError("log_posteriors must be a 1-D array")

    max_idx = int(np.argmax(log_posteriors))
    return np.asarray(chain[max_idx], dtype=float)


def make_diagnostics(
    chain: np.ndarray,
    output_dir: Path,
    log_posteriors: Optional[np.ndarray] = None,
    map_estimate: Optional[Sequence[float]] = None,
    parameter_names: Iterable[str] = ("N", "m"),
) -> None:
    """Generate standard diagnostic plots and save them to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(parameter_names)
    map_vector = None
    if map_estimate is not None:
        map_vector = np.asarray(map_estimate, dtype=float)
        if map_vector.shape != (len(names),):
            raise ValueError("map_estimate must match the number of parameters")
    log_posteriors_array: Optional[np.ndarray]
    if log_posteriors is not None:
        log_posteriors_array = np.asarray(log_posteriors, dtype=float)
        if log_posteriors_array.ndim != 1 or log_posteriors_array.shape[0] != chain.shape[0]:
            raise ValueError("log_posteriors must be a 1-D array aligned with the chain")
    else:
        log_posteriors_array = None

    # Trace plots
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 6), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, name in enumerate(names):
        axes[idx].plot(chain[:, idx], color="steelblue", linewidth=0.6)
        axes[idx].set_ylabel(name)
    axes[-1].set_xlabel("Iteration")
    fig.suptitle("Trace Plots")
    fig.tight_layout()
    fig.savefig(output_dir / "trace_plots.png", dpi=200)
    plt.close(fig)

    # Marginal histograms
    fig, axes = plt.subplots(1, len(names), figsize=(10, 4))
    axes = np.atleast_1d(axes)
    for idx, name in enumerate(names):
        sns.histplot(chain[:, idx], kde=True, ax=axes[idx], color="indianred")
        axes[idx].set_xlabel(name)
    fig.suptitle("Marginal Distributions")
    fig.tight_layout()
    fig.savefig(output_dir / "marginal_histograms.png", dpi=200)
    plt.close(fig)

    # Joint scatter / kde
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.kdeplot(
        x=chain[:, 0],
        y=chain[:, 1],
        fill=True,
        levels=20,
        cmap="viridis",
        ax=ax,
    )
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_title("Joint Posterior Density")
    fig.tight_layout()
    fig.savefig(output_dir / "joint_density.png", dpi=200)
    plt.close(fig)

    if map_vector is not None:
        fig, axes = plt.subplots(len(names), 1, figsize=(8, 3 * len(names)), sharex=False)
        axes = np.atleast_1d(axes)
        for idx, name in enumerate(names):
            sns.kdeplot(chain[:, idx], fill=True, ax=axes[idx], color="slateblue")
            axes[idx].axvline(map_vector[idx], color="darkorange", linestyle="--", linewidth=1.5, label="MAP")
            axes[idx].axvline(np.median(chain[:, idx]), color="seagreen", linestyle="-.", linewidth=1.2, label="Median")
            axes[idx].set_xlabel(name)
        axes[0].set_title("Posterior Marginals with MAP and Median")
        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            axes[-1].legend(handles, labels, loc="upper right")
        fig.tight_layout()
        fig.savefig(output_dir / "posterior_marginals_with_map.png", dpi=200)
        plt.close(fig)

    if log_posteriors_array is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(log_posteriors_array, color="black", linewidth=0.7)
        ax.set_xlabel("Iteration (post burn-in)")
        ax.set_ylabel("Log posterior")
        ax.set_title("Log Posterior Trace")
        if map_vector is not None:
            map_idx = int(np.argmax(log_posteriors_array))
            ax.axvline(map_idx, color="darkorange", linestyle="--", linewidth=1.2, label="MAP sample")
            ax.scatter(map_idx, log_posteriors_array[map_idx], color="darkorange", zorder=5)
            ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_dir / "log_posterior_trace.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 6))
        scatter = ax.scatter(
            chain[:, 0],
            chain[:, 1],
            c=log_posteriors_array,
            cmap="viridis",
            s=18,
            alpha=0.75,
        )
        if map_vector is not None:
            ax.scatter(map_vector[0], map_vector[1], color="red", marker="*", s=160, label="MAP")
            ax.legend(loc="upper right")
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        ax.set_title("Posterior Density Landscape")
        fig.colorbar(scatter, ax=ax, label="Log posterior")
        fig.tight_layout()
        fig.savefig(output_dir / "posterior_density_landscape.png", dpi=200)
        plt.close(fig)


__all__ = ["summarize_chain", "estimate_map", "make_diagnostics", "plot_posterior_predictive"]
