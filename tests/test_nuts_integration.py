
import numpy as np
import pandas as pd
import pytest
from mcmc_pmb.mcmc import NUTSConfig
from mcmc_pmb.priors import PriorParameters
from mcmc_pmb.forward_model import MaterialBalanceModel
from mcmc_pmb.pymc_sampler import run_nuts_sampler
from mcmc_pmb.pvt import MaterialBalancePVT

def test_nuts_sampler_runs():
    """Verify that NUTS sampler runs end-to-end and produces valid samples."""
    # Setup minimal data
    time = np.linspace(0, 300, 10)
    Np = np.linspace(0, 3e5, 10)
    Rp = np.linspace(0, 700, 10)
    # Create dummy pressure data that roughly follows a trend
    measured = np.linspace(3500, 3200, 10)
    
    data = pd.DataFrame({
        "time_days": time,
        "Np": Np,
        "Rp": Rp,
        "Pressure_measured": measured
    })
    
    pvt = MaterialBalancePVT()
    model = MaterialBalanceModel(pvt)
    
    prior_params = PriorParameters(
        mean_N=100.0,
        mean_m=0.4,
        std_N=20.0,
        std_m=0.1,
        correlation=0.0
    )
    
    # Use small number of samples for speed
    config = NUTSConfig(
        n_samples=20,
        n_tune=20,
        target_accept=0.8,
        n_chains=1,
        random_seed=42
    )
    
    result = run_nuts_sampler(
        config=config,
        prior_params=prior_params,
        production_data=data,
        model=model,
        pressure_uncertainty=10.0
    )
    
    # Check output shape
    # n_samples + n_tune = 20 + 20 = 40
    assert result.raw_chain.shape == (40, 2)
    assert result.log_posteriors.shape == (40,)
    assert np.all(result.accepted)
    
    # Check if samples are within reasonable bounds (positive)
    assert np.all(result.raw_chain[:, 0] > 0) # N > 0
    assert np.all(result.raw_chain[:, 1] >= 0) # m >= 0
    
    # Check if log posteriors are finite
    assert np.all(np.isfinite(result.log_posteriors))
