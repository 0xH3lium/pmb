"""End-to-end pipeline."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .analysis import (
    estimate_map,
    make_diagnostics,
    summarize_chain,
    plot_posterior_predictive,
)
from .forward_model import MaterialBalanceModel, prepare_production_dataset
from .mcmc import MetropolisHastingsConfig, NUTSConfig
from .priors import PriorParameters
from .pvt import MaterialBalancePVT
from .synthetic import generate_synthetic_dataset
from .pymc_model import run_sampler  # Unified import


def main(sampler: str = "metropolis") -> None:
    root = Path(__file__).resolve().parents[2]
    data_path = root / "data" / "production_data.csv"
    out_dir = root / "outputs"

    aquifer_true = 15.0  # Bbl/day/psi (productivity index J)
    aquifer_capacity_true = 1.0e6  # rb/psi (aquifer capacity C)
    max_prod_true = 12.0e6  # 12 MMSTB production

    if not data_path.exists():
        print(f"Generating synthetic data at {data_path}...")
        generate_synthetic_dataset(
            output_csv=data_path,
            max_prod=max_prod_true,
            aquifer_J=aquifer_true,
            aquifer_C=aquifer_capacity_true,
        )
    data = pd.read_csv(data_path)

    # Setup
    pvt_params = MaterialBalancePVT()
    model = MaterialBalanceModel(pvt_params=pvt_params)
    dataset = prepare_production_dataset(data)
    prior = PriorParameters(
        mean_N=100.0,  # interpreted as MMSTB
        mean_m=0.4,
        mean_J=np.log(max(aquifer_true, 1e-9)),
        mean_C=np.log(max(aquifer_capacity_true, 1e-9)),
        std_N=60.0,
        std_m=0.13,
        std_J=1.0,  # Wider prior for J
        std_C=0.5,  # Prior uncertainty on log(C)
        correlation=-0.5,  # Expect strong negative correlation now
    )
    sigma = 100.0

    print(f"Running {sampler.upper()} sampler...")

    if sampler.lower() == "nuts":
        config = NUTSConfig(n_samples=1000, n_tune=500, target_accept=0.9, n_chains=2)
    else:
        config = MetropolisHastingsConfig(
            n_iterations=10000,
            burn_in=2000,
            thinning=5,
            proposal_std=(3.0, 0.03, 0.3, 0.3),
            use_adaptive=True,
        )

    result = run_sampler(
        config, prior, dataset, model, sigma, sampler_type=sampler.lower()
    )

    # Analysis
    out_dir.mkdir(parents=True, exist_ok=True)

    if result.idata is not None:
        result.idata.to_netcdf(out_dir / "idata.nc")

    chain = result.chain

    np.savetxt(out_dir / "samples.csv", chain, delimiter=",", header="N,m,J,C")

    map_est = estimate_map(chain, result.log_posteriors_chain)
    summary = summarize_chain(chain, map_estimate=map_est)
    summary.to_csv(out_dir / "summary.csv", index=False)

    make_diagnostics(chain, out_dir, result.log_posteriors_chain, map_est)

    plot_posterior_predictive(chain, model, dataset, out_dir)

    print(f"\nSampler: {sampler}")
    print(f"Acceptance: {result.acceptance_rate:.2%}")
    print("MAP Estimate:", tuple(np.round(map_est, 3)))
    print(summary)


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "nuts")
