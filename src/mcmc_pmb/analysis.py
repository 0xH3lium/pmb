"""Post-processing utilities for MCMC diagnostics and visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


def summarize_chain(chain: np.ndarray, parameter_names: Iterable[str] = ("N", "m")) -> pd.DataFrame:
    """Return a summary table with mean, std, and selected percentiles."""

    if chain.ndim != 2 or chain.shape[1] != 2:
        raise ValueError("Expected a chain with shape (n_samples, 2)")

    names = list(parameter_names)
    if len(names) != chain.shape[1]:
        raise ValueError("parameter_names must have length equal to number of columns")

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
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def make_diagnostics(
    chain: np.ndarray,
    output_dir: Path,
    parameter_names: Iterable[str] = ("N", "m"),
) -> None:
    """Generate standard diagnostic plots and save them to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(parameter_names)

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


__all__ = ["summarize_chain", "make_diagnostics"]
