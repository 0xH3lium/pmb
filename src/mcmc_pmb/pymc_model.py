"""PyTensor Op and PyMC sampling logic."""

from __future__ import annotations

from typing import List, Tuple, Any
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from scipy.stats import multivariate_normal

from .forward_model import MaterialBalanceModel, ProductionDataset, prepare_production_dataset
from .mcmc import MetropolisHastingsConfig, MetropolisHastingsResult, NUTSConfig
from .priors import PriorParameters

class MaterialBalanceOp(Op):
    """
    PyTensor Op implementing the forward model with gradients via Implicit Function Theorem.
    """
    __props__ = ("swi", "cw", "cf", "Pi")

    def __init__(self, model: MaterialBalanceModel, dataset: ProductionDataset):
        self.model = model
        self.dataset = dataset
        self.swi = model.connate_water_saturation
        self.cw = model.water_compressibility
        self.cf = model.pore_compressibility
        self.Pi = model.pvt.initial_pressure
        
        # Pre-calculated tensor constants for the graph
        self.t_Np = pt.as_tensor_variable(dataset.Np, name="Np")
        self.t_Rp = pt.as_tensor_variable(dataset.Rp, name="Rp")

    def make_node(self, theta):
        theta = pt.as_tensor_variable(theta)
        return Apply(self, [theta], [pt.dvector()])

    def perform(self, node, inputs, outputs):
        (theta,) = inputs
        try:
            p_pred = self.model.predict_pressures(theta, self.dataset)
            outputs[0][0] = p_pred
        except Exception:
            outputs[0][0] = np.full(self.dataset.n_steps, np.nan)

    def grad(self, inputs, output_gradients):
        (theta,) = inputs
        (grad_p,) = output_gradients
        p = self(theta) # Symbolic output of forward pass
        
        # Reconstruct Residual Graph Symbolically
        # R(p, theta) = LHS - RHS
        # LHS = Np * (Bo + (Rp - Rs)*Bg)
        # RHS = N*( (Bo-Boi) + (Rsi-Rs)*Bg ) + N*m*Boi*(Bg/Bgi - 1) + CompExp
        
        N = theta[0]
        m = theta[1]
        
        # PVT (Unified backend handles tensors)
        pvt = self.model.pvt
        Bo = pvt.oil_fvf(p)
        Bg = pvt.gas_fvf(p)
        Rs = pvt.solution_gor(p)
        
        Boi = pvt.oil_fvf(self.Pi)
        Bgi = pvt.gas_fvf(self.Pi)
        Rsi = pvt.solution_gor(self.Pi) # Constant scalars
        
        # Compressibility
        delta_p = self.Pi - p
        eff_comp = self.model._eff_comp_term
        comp_term = pt.switch(pt.gt(delta_p, 0), N * Boi * eff_comp * delta_p, 0.0)
        
        lhs = self.t_Np * (Bo + (self.t_Rp - Rs) * Bg)
        rhs_fluid = N * ((Bo - Boi) + (Rsi - Rs) * Bg)
        rhs_gas = N * m * Boi * ((Bg / pt.maximum(Bgi, 1e-12)) - 1.0)
        
        residual = lhs - (rhs_fluid + rhs_gas + comp_term)
        
        # IFT: dp/dtheta = - (dR/dtheta) / (dR/dp)
        # 1. Jacobian Diagonal dR/dp
        # Since R[t] depends only on p[t], Jacobian is diagonal.
        dR_dp = pt.grad(pt.sum(residual), p)
        
        # 2. Total Gradient
        # dL/dtheta = sum( (dL/dp) * (dp/dtheta) )
        #           = sum( (dL/dp) * (- (dR/dtheta) / (dR/dp)) )
        #           = sum( - (dL/dp / dR_dp) * (dR/dtheta) )
        
        w = -grad_p / (dR_dp + 1e-15)
        
        # We compute gradient of (w * R) wrt theta, treating w as constant
        weighted_res = pt.sum(w * residual)
        return [pt.grad(weighted_res, theta, consider_constant=[p, w, dR_dp])]

def _build_pymc_model(
    prior: PriorParameters, 
    dataset: ProductionDataset, 
    mb_op: MaterialBalanceOp, 
    sigma: float
) -> pm.Model:
    """Shared model construction logic."""
    with pm.Model() as model:
        # Construct covariance matrix
        cov = np.array([
            [prior.std_N**2, prior.correlation * prior.std_N * prior.std_m],
            [prior.correlation * prior.std_N * prior.std_m, prior.std_m**2]
        ])
        mu = np.array([prior.mean_N, prior.mean_m])
        
        theta = pm.MvNormal("theta", mu=mu, cov=cov, shape=2)
        
        # Forward Pass
        p_pred = mb_op(theta)
        
        # Likelihood
        pm.Normal("obs", mu=p_pred, sigma=sigma, observed=dataset.pressure_measured)
        
    return model

def run_sampler(
    config: Any,
    prior: PriorParameters,
    data: Any,
    model: MaterialBalanceModel,
    sigma: float,
    sampler_type: str = "nuts"
) -> MetropolisHastingsResult:
    
    dataset = prepare_production_dataset(data)
    mb_op = MaterialBalanceOp(model, dataset)
    
    pm_model = _build_pymc_model(prior, dataset, mb_op, sigma)
    
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
            # Combine chains
            raw_chain = idata.posterior["theta"].values.reshape(-1, 2)
            accepted = np.ones(len(raw_chain), dtype=bool)
            burn_in = config.n_tune
            thinning = 1
            
        else: # Metropolis
            # Explicit proposal covariance
            S = np.diag(np.array(config.proposal_std)**2)
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
                discard_tuned_samples=False 
            )
            
            # Handling chain extraction including warmup for consistent plotting
            if "warmup_posterior" in idata and "theta" in idata.warmup_posterior:
                 warm = idata.warmup_posterior["theta"].values[0]
                 post = idata.posterior["theta"].values[0]
                 raw_chain = np.vstack([warm, post])
            else:
                 raw_chain = idata.posterior["theta"].values[0]

            # Metropolis acceptance check (simple diff)
            accepted = np.concatenate(([True], np.any(raw_chain[1:] != raw_chain[:-1], axis=1)))
            burn_in = config.burn_in
            thinning = config.thinning

        # --- Efficient Log Posterior Extraction ---
        # Compute log likelihood using PyMC's optimized graph
        pm.compute_log_likelihood(idata)
        
        # Sum log_likelihood over observed data points (dim 2 usually)
        # shape: (chains, draws, n_obs) -> (chains, draws)
        ll_vals = idata.log_likelihood["obs"].sum(dim="obs_dim_0").values
        
        if sampler_type == "nuts":
            ll_flat = ll_vals.flatten()
        else:
            # Handle warmup/posterior split logic for Metropolis
            if "warmup_log_likelihood" in idata:
                 ll_warm = idata.warmup_log_likelihood["obs"].sum(dim="obs_dim_0").values[0]
                 ll_post = ll_vals[0]
                 ll_flat = np.concatenate([ll_warm, ll_post])
            else:
                 ll_flat = ll_vals[0]

        # Add Log Prior (computed analytically for speed as it's cheap)
        cov = np.array([
            [prior.std_N**2, prior.correlation * prior.std_N * prior.std_m],
            [prior.correlation * prior.std_N * prior.std_m, prior.std_m**2]
        ])
        mu = np.array([prior.mean_N, prior.mean_m])
        
        log_priors = multivariate_normal.logpdf(raw_chain, mean=mu, cov=cov)
        log_posteriors = log_priors + ll_flat

    return MetropolisHastingsResult(
        raw_chain=raw_chain,
        log_posteriors=log_posteriors,
        accepted=accepted,
        burn_in=burn_in,
        thinning=thinning,
    )