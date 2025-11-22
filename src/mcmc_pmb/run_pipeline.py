"""End-to-end pipeline to run MCMC for probabilistic material balance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import estimate_map, make_diagnostics, summarize_chain, plot_posterior_predictive
from .forward_model import MaterialBalanceModel
from .mcmc import MetropolisHastingsConfig, run_metropolis_hastings
from .priors import PriorParameters, build_prior_distribution
from .pvt import MaterialBalancePVT
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
        mean_N=100.0,
        mean_m=0.4,
        std_N=60.0,
        std_m=0.13,
        correlation=-0.1,
    )
    prior_dist = build_prior_distribution(hyperparameters)

    pvt = MaterialBalancePVT()
    model = MaterialBalanceModel(pvt=pvt)
    
    pressure_uncertainty = 100.0

    config = MetropolisHastingsConfig(
        n_iterations=5000,
        burn_in=500,
        thinning=3,
        proposal_std=(4.0, 0.04),
        random_seed=2025,
        use_adaptive=True,
        adaptation_start=1000,
        adaptation_interval=100,
    )

    # result = run_metropolis_hastings(
    #     config=config,
    #     prior_dist=prior_dist,
    #     production_data=data,
    #     model=model,
    #     pressure_uncertainty=pressure_uncertainty,
    #     initial_parameters=(hyperparameters.mean_N, hyperparameters.mean_m),
    # )

    from .pymc_sampler import run_pymc_sampler
    result = run_pymc_sampler(
        config=config,
        prior_params=hyperparameters,
        production_data=data,
        model=model,
        pressure_uncertainty=pressure_uncertainty,
    )

    outputs_dir.mkdir(parents=True, exist_ok=True)
    chain = result.chain
    log_posteriors = result.log_posteriors_chain
    map_estimate = estimate_map(chain, log_posteriors)

    np.savetxt(outputs_dir / "posterior_samples.csv", chain, delimiter=",", header="N,m", comments="")

    summary_table = summarize_chain(chain, map_estimate=map_estimate)
    summary_table.to_csv(outputs_dir / "posterior_summary.csv", index=False)

    make_diagnostics(
        chain,
        outputs_dir,
        log_posteriors=log_posteriors,
        map_estimate=map_estimate,
    )

    
    plot_posterior_predictive(
        chain=chain,
        model=model,
        production_data=data,
        output_dir=outputs_dir,
        n_curves=100,
    )

    print("Acceptance rate:", f"{result.acceptance_rate:.3f}")
    print("Posterior summary:\n", summary_table)
    print("MAP estimate (N, m):", tuple(f"{value:.3f}" for value in map_estimate))


if __name__ == "__main__":
    main()
