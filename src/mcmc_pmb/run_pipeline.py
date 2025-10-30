"""End-to-end pipeline to run MCMC for probabilistic material balance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import make_diagnostics, summarize_chain
from .forward_model import MaterialBalanceModel, MaterialBalancePVT
from .mcmc import MetropolisHastingsConfig, run_metropolis_hastings
from .priors import PriorParameters, build_prior_distribution
from .synthetic import generate_synthetic_dataset


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "production_data.csv"
    outputs_dir = project_root / "outputs"

    if not data_path.exists():
        print(f"No data at {data_path}. Generating synthetic dataset.")
        generate_synthetic_dataset(output_csv=data_path)

    data = pd.read_csv(data_path)

    hyperparameters = PriorParameters(
        mean_N=115.0,
        mean_m=0.4,
        std_N=35.0,
        std_m=0.13,
        correlation=-0.6,
    )
    prior_dist = build_prior_distribution(hyperparameters)

    pvt = MaterialBalancePVT()
    model = MaterialBalanceModel(pvt=pvt)
    pressure_uncertainty = 10.0

    config = MetropolisHastingsConfig(
        n_iterations=60000,
        burn_in=12000,
        thinning=5,
        proposal_std=(4.0, 0.04),
        random_seed=2025,
    )

    result = run_metropolis_hastings(
        config=config,
        prior_dist=prior_dist,
        production_data=data,
        model=model,
        pressure_uncertainty=pressure_uncertainty,
        initial_parameters=(hyperparameters.mean_N, hyperparameters.mean_m),
    )

    outputs_dir.mkdir(parents=True, exist_ok=True)
    chain = result.chain

    np.savetxt(outputs_dir / "posterior_samples.csv", chain, delimiter=",", header="N,m", comments="")

    summary_table = summarize_chain(chain)
    summary_table.to_csv(outputs_dir / "posterior_summary.csv", index=False)

    make_diagnostics(chain, outputs_dir)

    print("Acceptance rate:", f"{result.acceptance_rate:.3f}")
    print("Posterior summary:\n", summary_table)


if __name__ == "__main__":
    main()
