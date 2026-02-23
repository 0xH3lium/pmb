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

        # Errors-in-Variables: latent true cumulative oil and GOR
        n_steps = dataset.n_steps

        # Monotonic cumulative production via positive increments
        dNp_obs = np.diff(np.concatenate(([0.0], dataset.Np)))
        dNp_mu = np.maximum(dNp_obs, 1e-6)
        dNp_true = pm.LogNormal(
            "dNp_true",
            mu=pt.as_tensor_variable(np.log(dNp_mu)),
            sigma=0.05,
            shape=n_steps,
        )
        Np_true = pm.Deterministic("Np_true", pt.cumsum(dNp_true))
        pm.Normal(
            "Np_obs",
            mu=Np_true,
            sigma=pt.as_tensor_variable(np.maximum(np.abs(dataset.Np) * 0.05, 1e-6)),
            observed=dataset.Np,
        )

        # EIV for Gas-Oil Ratio (Rp)
        Rp_mu = np.maximum(dataset.Rp, 1e-6)

        # 1. Prior for the latent true Rp (LogNormal ensures Rp > 0)
        Rp_true = pm.LogNormal(
            "Rp_true",
            mu=pt.as_tensor_variable(np.log(Rp_mu)),
            sigma=0.05,  # Prior uncertainty on the true Rp
            shape=n_steps,
        )

        # 2. Likelihood of observing the measured Rp given the true Rp
        pm.Normal(
            "Rp_obs",
            mu=Rp_true,
            sigma=pt.as_tensor_variable(np.maximum(dataset.Rp * 0.05, 1e-6)),
            observed=dataset.Rp,
        )

        pressures = model.symbolic_pressures(
            theta[0], theta[1], J, C, dataset, Np_seq=Np_true, Rp_seq=Rp_true
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
            raw_chain = idata.posterior["theta"].values.reshape(-1, 4)
            theta_raw_chain = idata.posterior["theta_raw"].values.reshape(-1, 4)
            accepted = np.ones(len(raw_chain), dtype=bool)
            burn_in = 0
            thinning = 1

        else:
            proposal_std = np.array(config.proposal_std, dtype=float)
            S = np.diag(proposal_std**2)
            step = [
                pm.Metropolis(
                    vars=[
                        pm_model["N"],
                        pm_model["m"],
                        pm_model["log_J"],
                        pm_model["log_C"],
                    ],
                    S=S,
                ),
                pm.Metropolis(vars=[pm_model["dNp_true"]]),
                pm.Metropolis(vars=[pm_model["Rp_true"]]),
                pm.Metropolis(vars=[pm_model["nu"]]),
            ]

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
                step=step,
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
