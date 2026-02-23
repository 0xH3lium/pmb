"""Plot the 2-D log-likelihood surface over (N, m) for a synthetic Gas-Cap dataset.

The filled contour map reveals the negative correlation between OOIP (N) and the
gas-cap size ratio (m): many (N, m) combinations produce nearly identical pressure
histories, forming a diagonal ridge in the likelihood landscape.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from src.mcmc_pmb.forward_model import MaterialBalanceModel, prepare_production_dataset
from src.mcmc_pmb.pvt import MaterialBalancePVT
from src.mcmc_pmb.synthetic import generate_synthetic_dataset

# ---------------------------------------------------------------------------
# True parameters and synthetic dataset
# ---------------------------------------------------------------------------
TRUE_N = 110.0          # MMSTB  (matches generate_synthetic_dataset default)
TRUE_M = 0.35           # gas-cap ratio (matches generate_synthetic_dataset default)
AQUIFER_J = 15.0        # aquifer productivity index  (matches synthetic default)
AQUIFER_C = 1.0e6       # aquifer capacity in rb/psi (matches synthetic default)
MEASUREMENT_NOISE = 50.0  # psi  (matches synthetic default)

print("Generating synthetic Gas-Cap production dataset …")
production_data = generate_synthetic_dataset(
    true_parameters=(TRUE_N, TRUE_M),
    aquifer_J=AQUIFER_J,
    aquifer_C=AQUIFER_C,
    measurement_noise=MEASUREMENT_NOISE,
    random_seed=1244,
)

# Build the forward model with the same aquifer settings used to create the data.
pvt = MaterialBalancePVT()
model = MaterialBalanceModel(
    pvt_params=pvt,
    aquifer_index=AQUIFER_J,
    aquifer_capacity=AQUIFER_C,
)

# Pre-compile the PyTensor prediction function *once* so it is reused for every
# grid point.  Calling model.predict_pressures() inside the loop would trigger a
# full recompilation on every call (~3 600 times), which is intractably slow.
print("Compiling PyTensor prediction function (one-time) …")
dataset = prepare_production_dataset(production_data)
measured = production_data["Pressure_measured"].to_numpy(dtype=float)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    predict_fn = model.make_predict_function(dataset)

print("Done — starting grid evaluation.")

# ---------------------------------------------------------------------------
# 2-D parameter grid
# ---------------------------------------------------------------------------
N_POINTS = 100           # resolution along each axis
N_values = np.linspace(40.0, 180.0, N_POINTS)
m_values = np.linspace(0.1, 0.9, N_POINTS)

N_grid, m_grid = np.meshgrid(N_values, m_values)   # shape (N_POINTS, N_POINTS)
ll_grid = np.full(N_grid.shape, np.nan)

print(f"Evaluating log-likelihood on a {N_POINTS}×{N_POINTS} grid …")
total = N_POINTS * N_POINTS
for i, m_val in enumerate(m_values):
    for j, N_val in enumerate(N_values):
        try:
            theta = np.array([N_val, m_val, AQUIFER_J, AQUIFER_C], dtype="float32")
            predicted = predict_fn(theta)
            if np.any(np.isnan(predicted)):
                ll = np.nan
            else:
                ll = float(np.sum(norm.logpdf(measured, loc=predicted,
                                              scale=MEASUREMENT_NOISE)))
        except Exception:
            ll = np.nan
        ll_grid[i, j] = ll

    completed = (i + 1) * N_POINTS
    if (i + 1) % 10 == 0 or i == N_POINTS - 1:
        print(f"  {completed}/{total} evaluations complete …")

# Replace any remaining NaN / -inf with the minimum finite value so contourf
# renders them as the lowest colour band rather than blank patches.
finite_mask = np.isfinite(ll_grid)
if finite_mask.any():
    ll_min = ll_grid[finite_mask].min()
    ll_grid = np.where(finite_mask, ll_grid, ll_min)

# ---------------------------------------------------------------------------
# Run short MCMC chains to overlay
# ---------------------------------------------------------------------------
from src.mcmc_pmb.pymc_model import run_sampler
from src.mcmc_pmb.mcmc import MetropolisHastingsConfig, NUTSConfig
from src.mcmc_pmb.priors import PriorParameters

print("Running short MCMC chains for overlay...")
prior = PriorParameters(
    mean_N=100.0,
    mean_m=0.4,
    mean_J=np.log(AQUIFER_J),
    mean_C=np.log(AQUIFER_C),
    std_N=60.0,
    std_m=0.13,
    std_J=1.0,
    std_C=0.5,
    correlation=-0.5,
)

print("  -> Metropolis-Hastings...")
mh_config = MetropolisHastingsConfig(
    n_iterations=200,
    burn_in=0,
    thinning=1,
    proposal_std=(3.0, 0.03, 0.3, 0.3),
    use_adaptive=False,
)
mh_result = run_sampler(mh_config, prior, dataset, model, MEASUREMENT_NOISE, sampler_type="metropolis")
mh_chain = mh_result.raw_chain[mh_result.accepted]
mh_chain_50 = mh_chain[:200]

print("  -> NUTS...")
nuts_config = NUTSConfig(n_samples=200, n_tune=200, n_chains=1)
nuts_result = run_sampler(nuts_config, prior, dataset, model, MEASUREMENT_NOISE, sampler_type="nuts")
nuts_chain_50 = nuts_result.raw_chain[:500]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
# 1. Take ONLY the first 40 steps
n_plot_steps = 40
mh_path = mh_chain[:n_plot_steps] # Assuming you run MH again
nuts_path = nuts_chain_50[:n_plot_steps]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, chain, title, color in zip(
    axes, 
    [mh_path, nuts_path], 
    ["Random Walk MH (First 40 steps)", "NUTS (First 40 steps)"],
    ["white", "#FF1493"]
):
    # Plot background
    ax.contourf(N_grid, m_grid, ll_grid, levels=50, cmap="viridis")
    
    # Plot True value
    ax.plot(TRUE_N, TRUE_M, marker="*", color="red", markersize=15, linestyle="none")
    
    # Plot trajectory with dots at the states
    ax.plot(chain[:, 0], chain[:, 1], color=color, alpha=0.8, linewidth=1.5)
    ax.scatter(chain[:, 0], chain[:, 1], color=color, s=15, zorder=5) # Dots show the actual stops
    
    # Mark the start point explicitly
    ax.plot(chain[0, 0], chain[0, 1], marker="s", color="cyan", markersize=8, label="Start Point")
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("N (MMSTB)")
    if ax == axes[0]: ax.set_ylabel("m — Gas-Cap Size Ratio")
    ax.legend(loc="upper right")

plt.tight_layout()
plt.show()

import arviz as az

# Convert your chains to ArviZ InferenceData (assuming shape is [iterations, parameters])
# Here we just check parameter N (index 0)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

az.plot_autocorr(mh_chain[:, 0], ax=axes[0])
axes[0].set_title("MH Autocorrelation (Parameter N)\nSlow decay = Inefficient")

az.plot_autocorr(nuts_chain_50[:, 0], ax=axes[1])
axes[1].set_title("NUTS Autocorrelation (Parameter N)\nFast decay = Highly Efficient")

plt.tight_layout()
plt.show()