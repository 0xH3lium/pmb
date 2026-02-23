"""Master posterior predictive check (PPC) plot for three reservoir drive scenarios.

Creates a 2x3 grid:
- Columns: Depletion Drive, Gas-Cap Drive, Strong Water Drive
- Top row: observed pressure points, posterior median, and 95% HDI band
- Bottom row: residuals (observed - posterior median) over time

Residual y-axes are forced to the same scale across all three columns so
uncertainty can be compared directly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pytensor import config as pt_config

from src.mcmc_pmb.forward_model import MaterialBalanceModel, prepare_production_dataset
from src.mcmc_pmb.mcmc import NUTSConfig
from src.mcmc_pmb.priors import PriorParameters
from src.mcmc_pmb.pvt import MaterialBalancePVT
from src.mcmc_pmb.pymc_model import run_sampler
from src.mcmc_pmb.synthetic import generate_synthetic_dataset


def _scenario_slug(name: str) -> str:
    return name.lower().replace("-", "").replace(" ", "_")


def run_scenario_ppc(
    name: str,
    true_N: float,
    true_m: float,
    true_J: float,
    true_C: float,
    measurement_noise: float,
    random_seed: int,
    n_ppc_draws: int,
) -> dict[str, np.ndarray | str | float]:
    data_out = Path("data") / f"{_scenario_slug(name)}_scenario.csv"

    data = generate_synthetic_dataset(
        output_csv=data_out,
        n_steps=24,
        true_parameters=(true_N, true_m),
        measurement_noise=measurement_noise,
        random_seed=random_seed,
        max_prod=12.0e6,
        aquifer_J=true_J,
        aquifer_C=true_C,
    )

    dataset = prepare_production_dataset(data)
    model = MaterialBalanceModel(pvt_params=MaterialBalancePVT())

    prior = PriorParameters(
        mean_N=100.0,
        mean_m=0.4,
        mean_J=np.log(max(true_J, 1e-9)),
        mean_C=np.log(max(true_C, 1e-9)),
        std_N=60.0,
        std_m=0.13,
        std_J=1.0,
        std_C=0.5,
        correlation=-0.5,
    )

    nuts_cfg = NUTSConfig(
        n_samples=1000,
        n_tune=1000,
        target_accept=0.9,
        random_seed=random_seed,
        n_chains=2,
    )

    result = run_sampler(
        config=nuts_cfg,
        prior=prior,
        data=dataset,
        model=model,
        sigma=100.0,
        sampler_type="nuts",
    )

    chain = result.chain
    if chain.shape[0] == 0:
        raise RuntimeError(f"No posterior samples retained for scenario: {name}")

    rng = np.random.default_rng(random_seed + 101)
    draw_count = min(n_ppc_draws, chain.shape[0])
    draw_idx = rng.choice(chain.shape[0], size=draw_count, replace=False)

    predict_fn = model.make_predict_function(dataset)
    preds = np.empty((draw_count, dataset.n_steps), dtype=float)

    for i, idx in enumerate(draw_idx):
        theta = chain[idx].astype(pt_config.floatX)
        preds[i] = predict_fn(theta)

    p50 = np.percentile(preds, 50.0, axis=0)
    p2_5 = np.percentile(preds, 2.5, axis=0)
    p97_5 = np.percentile(preds, 97.5, axis=0)

    residuals = dataset.pressure_measured - p50

    return {
        "name": name,
        "time_days": dataset.time_days,
        "observed": dataset.pressure_measured,
        "median": p50,
        "hdi_low": p2_5,
        "hdi_high": p97_5,
        "residuals": residuals,
    }


def main() -> None:
    scenarios = [
        {
            "name": "Depletion Drive",
            "true_N": 110.0,
            "true_m": 0.05,
            "true_J": 1.0,
            "true_C": 5.0e5,
            "measurement_noise": 50.0,
            "random_seed": 2026,
        },
        {
            "name": "Gas-Cap Drive",
            "true_N": 110.0,
            "true_m": 0.70,
            "true_J": 8.0,
            "true_C": 8.0e5,
            "measurement_noise": 50.0,
            "random_seed": 2027,
        },
        {
            "name": "Strong Water Drive",
            "true_N": 110.0,
            "true_m": 0.20,
            "true_J": 40.0,
            "true_C": 2.0e6,
            "measurement_noise": 50.0,
            "random_seed": 2028,
        },
    ]

    n_ppc_draws = 500
    scenario_results = [
        run_scenario_ppc(
            name=s["name"],
            true_N=s["true_N"],
            true_m=s["true_m"],
            true_J=s["true_J"],
            true_C=s["true_C"],
            measurement_noise=s["measurement_noise"],
            random_seed=s["random_seed"],
            n_ppc_draws=n_ppc_draws,
        )
        for s in scenarios
    ]

    all_residuals = np.concatenate([r["residuals"] for r in scenario_results]).astype(float)
    max_abs_resid = float(np.nanmax(np.abs(all_residuals))) if all_residuals.size else 1.0
    resid_limit = max(1.0, 1.1 * max_abs_resid)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex="col")

    obs_color = "#111827"
    med_color = "#2563eb"
    band_color = "#bfdbfe"
    resid_color = "#dc2626"

    for col, result in enumerate(scenario_results):
        t = np.asarray(result["time_days"], dtype=float)
        y_obs = np.asarray(result["observed"], dtype=float)
        p50 = np.asarray(result["median"], dtype=float)
        lo = np.asarray(result["hdi_low"], dtype=float)
        hi = np.asarray(result["hdi_high"], dtype=float)
        resid = np.asarray(result["residuals"], dtype=float)

        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        ax_top.fill_between(t, lo, hi, color=band_color, alpha=0.75, label="95% HDI")
        ax_top.plot(t, p50, color=med_color, linewidth=2.0, label="Posterior Median")
        ax_top.scatter(t, y_obs, color=obs_color, s=22, zorder=5, label="Observed")

        ax_top.set_title(str(result["name"]))
        ax_top.set_ylabel("Pressure (psi)")
        ax_top.grid(alpha=0.3, linestyle="--")

        ax_bot.scatter(t, resid, color=resid_color, s=22, alpha=0.9)
        ax_bot.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax_bot.set_ylim(-resid_limit, resid_limit)
        ax_bot.set_ylabel("Residual (Data - Median)")
        ax_bot.set_xlabel("Time (days)")
        ax_bot.grid(alpha=0.3, linestyle="--")

        rmse = float(np.sqrt(np.mean(resid**2)))
        ax_bot.text(
            0.03,
            0.08,
            f"RMSE: {rmse:.1f} psi",
            transform=ax_bot.transAxes,
            fontsize=9,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Master Posterior Predictive Check Across Drive Mechanisms", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = Path("outputs") / "master_ppc.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved master PPC figure -> {out_path}")


if __name__ == "__main__":
    main()
