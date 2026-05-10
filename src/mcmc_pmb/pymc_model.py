"""Native PyMC integration for the material-balance model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.stats import norm, truncnorm

from .forward_model import (
    MaterialBalanceModel,
    ProductionDataset,
    prepare_production_dataset,
)
from .mcmc import MetropolisHastingsResult
from .priors import PriorParameters


def _build_pymc_model(
    prior: PriorParameters,
    dataset: ProductionDataset,
    model: MaterialBalanceModel,
    sigma: float,
) -> pm.Model:
    rho = float(np.clip(prior.correlation, -0.999, 0.999))
    sigma_m_cond = float(prior.std_m * np.sqrt(1.0 - rho**2))
    if sigma_m_cond <= 0.0:
        raise ValueError(
            "Invalid prior: std_m and correlation produce non-positive conditional sigma"
        )

    with pm.Model() as pymc_model:
        N = pm.Normal("N", mu=prior.mean_N, sigma=prior.std_N)
        m_cond_mu = pm.Deterministic(
            "m_cond_mu",
            prior.mean_m + rho * (prior.std_m / prior.std_N) * (N - prior.mean_N),
        )
        m = pm.TruncatedNormal("m", mu=m_cond_mu, sigma=sigma_m_cond, lower=0.0)
        log_J = pm.Normal("log_J", mu=prior.mean_J, sigma=prior.std_J)
        J = pm.Deterministic("J", pt.exp(log_J))
        log_C = pm.Normal("log_C", mu=prior.mean_C, sigma=prior.std_C)
        C = pm.Deterministic("C", pt.exp(log_C))

        theta = pm.Deterministic("theta", pt.stack([N, m, J, C]))
        pm.Deterministic("theta_raw", pt.stack([N, m, log_J, log_C]))

        pressures = model.symbolic_pressures(theta[0], theta[1], J, C, dataset)

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
                nuts_sampler='numpyro'
            )
            raw_chain = idata.posterior["theta"].values.reshape(-1, 4)
            theta_raw_chain = idata.posterior["theta_raw"].values.reshape(-1, 4)
            accepted = np.ones(len(raw_chain), dtype=bool)
            burn_in = 0
            thinning = 1

        else:
            tune = config.burn_in if config.use_adaptive else 0
            draws = (
                config.n_iterations - tune
                if config.use_adaptive
                else config.n_iterations
            )
            discard_tuned_samples = bool(config.use_adaptive)

            idata = pm.sample(
                draws=draws,
                tune=tune,
                step=pm.Metropolis(),
                chains=1,
                progressbar=True,
                random_seed=config.random_seed,
                discard_tuned_samples=discard_tuned_samples,
                
            )

            raw_chain = idata.posterior["theta"].values.reshape(-1, 4)
            theta_raw_chain = idata.posterior["theta_raw"].values.reshape(-1, 4)

            accepted = np.concatenate(
                ([True], np.any(raw_chain[1:] != raw_chain[:-1], axis=1))
            )
            burn_in = 0 if config.use_adaptive else config.burn_in
            thinning = config.thinning

        pm.compute_log_likelihood(idata)
        ll_vals = idata.log_likelihood["obs"].sum(dim="obs_dim_0").values

        if sampler_type == "nuts":
            ll_flat = ll_vals.flatten()
        else:
            ll_flat = ll_vals.flatten()

        theta_raw_chain = theta_raw_chain.reshape(-1, 4)
        N_chain = theta_raw_chain[:, 0]
        m_chain = theta_raw_chain[:, 1]
        log_J_chain = theta_raw_chain[:, 2]
        log_C_chain = theta_raw_chain[:, 3]

        rho = float(np.clip(prior.correlation, -0.999, 0.999))
        sigma_m_cond = float(prior.std_m * np.sqrt(1.0 - rho**2))
        m_cond_mu_chain = prior.mean_m + rho * (prior.std_m / prior.std_N) * (
            N_chain - prior.mean_N
        )

        log_prior_N = norm.logpdf(N_chain, loc=prior.mean_N, scale=prior.std_N)
        a = (0.0 - m_cond_mu_chain) / sigma_m_cond
        log_prior_m = truncnorm.logpdf(
            m_chain, a=a, b=np.inf, loc=m_cond_mu_chain, scale=sigma_m_cond
        )
        log_prior_log_J = norm.logpdf(log_J_chain, loc=prior.mean_J, scale=prior.std_J)
        log_prior_log_C = norm.logpdf(log_C_chain, loc=prior.mean_C, scale=prior.std_C)
        log_priors = log_prior_N + log_prior_m + log_prior_log_J + log_prior_log_C
        log_posteriors = log_priors + ll_flat

    return MetropolisHastingsResult(
        raw_chain=raw_chain,
        log_posteriors=log_posteriors,
        accepted=accepted,
        burn_in=burn_in,
        thinning=thinning,
        idata=idata,
    )
