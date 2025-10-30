# MCMC for Probabilistic Material Balance

This project implements a complete, reproducible workflow for sampling the posterior distribution of original oil in place (`N`) and gas-cap ratio (`m`) using the Metropolis–Hastings algorithm. The implementation follows the practical guide outlined in the project brief and includes:

- a bivariate normal prior informed by volumetric analysis,
- a physics-inspired forward material-balance model with configurable gas-cap and water-drive support,
- a Gaussian likelihood comparing predicted and measured pressures,
- an efficient Metropolis–Hastings sampler with burn-in and thinning support,
- diagnostic plots and summary statistics for downstream interpretation,
- synthetic data generation to validate the workflow end-to-end.

## Project Layout

```
src/mcmc_pmb/           # Python package with priors, forward model, likelihood, sampler, and analysis tools
data/                   # Input production data (synthetic data auto-generated if missing)
outputs/                # Posterior samples, summary table, and diagnostic plots
main.py                 # Convenience entry point that calls the end-to-end pipeline
```

## Quick Start

Create and activate the virtual environment (already configured for this workspace) and install the project in editable mode if desired:

- This project uses uv from package management and virtual env !

Run the full pipeline, which will generate synthetic production data when no field data are present, execute the MCMC sampler, and produce diagnostics in `outputs/`:

```bash
python main.py
```

Key outputs include:

- `outputs/posterior_samples.csv`: retained MCMC samples after burn-in/thinning,
- `outputs/posterior_summary.csv`: summary statistics (mean, std, P10/P50/P90),
- `outputs/trace_plots.png`, `outputs/marginal_histograms.png`, `outputs/joint_density.png`: diagnostic figures.

## Re-using the Components

All core logic is exposed through the `mcmc_pmb` package so you can import and reuse individual pieces (prior construction, forward model, likelihood, sampler, or diagnostics) in custom workflows or notebooks. See `main.py` for a lightweight example of how the components fit together.
