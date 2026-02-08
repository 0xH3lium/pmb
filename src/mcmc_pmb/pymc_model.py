"""Native PyMC integration for the material-balance model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.stats import multivariate_normal

from .forward_model import MaterialBalanceModel, ProductionDataset, prepare_production_dataset
from .mcmc import MetropolisHastingsConfig, MetropolisHastingsResult, NUTSConfig
from .priors import PriorParameters


def _build_pymc_model(
    prior: PriorParameters,
    dataset: ProductionDataset,
    model: MaterialBalanceModel,
    sigma: float,
) -> pm.Model:
    cov = prior.to_covariance_matrix()
    mu = prior.to_mean_vector()
    chol = np.linalg.cholesky(cov)

    chol_tensor = pt.as_tensor_variable(chol)
    mu_tensor = pt.as_tensor_variable(mu)

    with pm.Model() as pymc_model:
        theta_raw = pm.MvNormal("theta_raw", mu=mu_tensor, chol=chol_tensor, shape=3)

        N = pm.Deterministic("N", theta_raw[0])
        m_linear = pm.Deterministic("m_linear", theta_raw[1])
        J = pm.Deterministic("J", pt.exp(theta_raw[2]))

        gas_in_place = pm.Deterministic(
            "gas_in_place", N * pt.maximum(m_linear, 0.0)
        )
        m_effective = pm.Deterministic(
            "m", pt.maximum(gas_in_place / pt.maximum(N, 1e-6), 0.0)
        )
        theta = pm.Deterministic("theta", pt.stack([N, m_effective, J]))

        # Errors-in-Variables: latent true cumulative oil and GOR
        n_steps = dataset.n_steps
        # Use relative 5% noise; enforce minimum to avoid zero variance early
        Np_true = pm.Normal(
            "Np_true",
            mu=pt.as_tensor_variable(dataset.Np),
            sigma=pt.as_tensor_variable(np.maximum(dataset.Np * 0.05, 1e-6)),
            shape=n_steps,
        )
        Rp_true = pm.Normal(
            "Rp_true",
            mu=pt.as_tensor_variable(dataset.Rp),
            sigma=pt.as_tensor_variable(np.maximum(dataset.Rp * 0.05, 1e-6)),
            shape=n_steps,
        )

        pressures = model.symbolic_pressures(
            theta[0], theta[1], J, dataset, Np_seq=Np_true, Rp_seq=Rp_true
        )

        # Robust likelihood via StudentT
        nu = pm.Exponential("nu", 1 / 5)
        pm.StudentT(
            "obs",
            nu=nu,
            mu=pressures,
            sigma=sigma,
            observed=dataset.pressure_measured,
        )

    return pymc_model


def run_sampler(
    config: Any,
    prior: PriorParameters,
    data: Any,
    model: MaterialBalanceModel,
    sigma: float,
    sampler_type: str = "nuts",
) -> MetropolisHastingsResult:
    dataset = prepare_production_dataset(data)
    pm_model = _build_pymc_model(prior, dataset, model, sigma)

    with pm_model:
        if sampler_type == "nuts":
            idata = pm.sample(
                draws=config.n_samples,
                tune=config.n_tune,
                chains=config.n_chains,
                target_accept=config.target_accept,
                random_seed=config.random_seed,
                progressbar=True,
            )
            raw_chain = idata.posterior["theta"].values.reshape(-1, 3)
            theta_raw_chain = idata.posterior["theta_raw"].values.reshape(-1, 3)
            accepted = np.ones(len(raw_chain), dtype=bool)
            burn_in = config.n_tune
            thinning = 1

        else:
            proposal_std = np.array(config.proposal_std, dtype=float)
            S = np.diag(proposal_std**2)
            step = pm.Metropolis(S=S)

            draws = config.n_iterations
            tune = config.burn_in if config.use_adaptive else 0
            if config.use_adaptive:
                draws -= tune

            idata = pm.sample(
                draws=draws,
                tune=tune,
                step=step,
                chains=1,
                progressbar=True,
                random_seed=config.random_seed,
                discard_tuned_samples=False,
            )

            if "warmup_posterior" in idata and "theta" in idata.warmup_posterior:
                warm = idata.warmup_posterior["theta"].values[0]
                warm_raw = idata.warmup_posterior["theta_raw"].values[0]
                post = idata.posterior["theta"].values[0]
                post_raw = idata.posterior["theta_raw"].values[0]
                raw_chain = np.vstack([warm, post])
                theta_raw_chain = np.vstack([warm_raw, post_raw])
            else:
                raw_chain = idata.posterior["theta"].values[0]
                theta_raw_chain = idata.posterior["theta_raw"].values[0]

            accepted = np.concatenate(([True], np.any(raw_chain[1:] != raw_chain[:-1], axis=1)))
            burn_in = config.burn_in
            thinning = config.thinning

        pm.compute_log_likelihood(idata)
        ll_vals = idata.log_likelihood["obs"].sum(dim="obs_dim_0").values

        if sampler_type == "nuts":
            ll_flat = ll_vals.flatten()
        else:
            if "warmup_log_likelihood" in idata:
                ll_warm = idata.warmup_log_likelihood["obs"].sum(dim="obs_dim_0").values[0]
                ll_post = ll_vals[0]
                ll_flat = np.concatenate([ll_warm, ll_post])
            else:
                ll_flat = ll_vals[0]

        cov = prior.to_covariance_matrix()
        mu = prior.to_mean_vector()

        theta_raw_chain = theta_raw_chain.reshape(-1, 3)

        log_priors = multivariate_normal.logpdf(theta_raw_chain, mean=mu, cov=cov)
        log_posteriors = log_priors + ll_flat

    return MetropolisHastingsResult(
        raw_chain=raw_chain,
        log_posteriors=log_posteriors,
        accepted=accepted,
        burn_in=burn_in,
        thinning=thinning,
    )