
import numpy as np
import pandas as pd
import pytest
import pytensor.tensor as pt
from pytensor import function

from mcmc_pmb.differentiable_model import MaterialBalanceOp
from mcmc_pmb.differentiable_pvt import DifferentiablePVT
from mcmc_pmb.forward_model import MaterialBalanceModel, ProductionDataset
from mcmc_pmb.pvt import MaterialBalancePVT


@pytest.fixture
def pvt_model():
    return MaterialBalancePVT()

@pytest.fixture
def diff_pvt(pvt_model):
    return DifferentiablePVT(pvt_model)

def test_pvt_consistency(pvt_model, diff_pvt):
    """Verify differentiable PVT matches original implementation."""
    pressures = np.linspace(100, 4000, 20)
    
    # Compile PyTensor functions
    p_var = pt.dvector("p")
    f_bo = function([p_var], diff_pvt.oil_fvf(p_var))
    f_bg = function([p_var], diff_pvt.gas_fvf(p_var))
    f_rs = function([p_var], diff_pvt.solution_gor(p_var))
    
    bo_pt = f_bo(pressures)
    bg_pt = f_bg(pressures)
    rs_pt = f_rs(pressures)
    
    bo_orig = np.array([pvt_model.oil_fvf(p) for p in pressures])
    bg_orig = np.array([pvt_model.gas_fvf(p) for p in pressures])
    rs_orig = np.array([pvt_model.solution_gor(p) for p in pressures])
    
    np.testing.assert_allclose(bo_pt, bo_orig, rtol=1e-5)
    np.testing.assert_allclose(bg_pt, bg_orig, rtol=1e-5)
    np.testing.assert_allclose(rs_pt, rs_orig, rtol=1e-5)

def test_pvt_gradients(diff_pvt):
    """Verify gradients of PVT functions using finite differences."""
    p_val = np.array([3000.0]) # Below bubble point (3200)
    epsilon = 1e-4
    
    # Define scalar functions for gradient check
    p_var = pt.dscalar("p")
    
    for func_name in ["oil_fvf", "gas_fvf", "solution_gor"]:
        func_pt = getattr(diff_pvt, func_name)(p_var)
        grad_pt = pt.grad(func_pt, p_var)
        f_grad = function([p_var], grad_pt)
        
        analytical_grad = f_grad(p_val[0])
        
        # Finite difference
        f_func = function([p_var], getattr(diff_pvt, func_name)(p_var))
        val_plus = f_func(p_val[0] + epsilon)
        val_minus = f_func(p_val[0] - epsilon)
        fd_grad = (val_plus - val_minus) / (2 * epsilon)
        
        np.testing.assert_allclose(analytical_grad, fd_grad, rtol=1e-3, err_msg=f"Gradient mismatch for {func_name}")

def test_material_balance_op_gradients():
    """Verify gradients of MaterialBalanceOp using finite differences."""
    # Setup minimal data
    time = np.array([0, 100, 200, 300], dtype=float)
    Np = np.array([0, 1e5, 2e5, 3e5], dtype=float)
    Rp = np.array([0, 500, 600, 700], dtype=float)
    measured = np.array([3500, 3400, 3300, 3200], dtype=float)
    
    dataset = ProductionDataset(time, Np, Rp, measured)
    pvt = MaterialBalancePVT()
    model = MaterialBalanceModel(pvt)
    
    op = MaterialBalanceOp(model, dataset)
    
    theta_val = np.array([100.0, 0.4]) # N=100MM, m=0.4
    
    # Compile gradient function
    theta_var = pt.dvector("theta")
    pressures = op(theta_var)
    # Scalar cost function: sum of squared pressures (arbitrary)
    cost = pt.sum(pressures**2)
    grad_theta = pt.grad(cost, theta_var)
    
    f_grad = function([theta_var], grad_theta)
    f_cost = function([theta_var], cost)
    
    analytical_grad = f_grad(theta_val)
    
    # Finite differences
    epsilon = 1e-4
    fd_grad = np.zeros_like(theta_val)
    
    for i in range(len(theta_val)):
        theta_plus = theta_val.copy()
        theta_plus[i] += epsilon
        cost_plus = f_cost(theta_plus)
        
        theta_minus = theta_val.copy()
        theta_minus[i] -= epsilon
        cost_minus = f_cost(theta_minus)
        
        fd_grad[i] = (cost_plus - cost_minus) / (2 * epsilon)
        
    print(f"Analytical grad: {analytical_grad}")
    print(f"FD grad: {fd_grad}")
    
    np.testing.assert_allclose(analytical_grad, fd_grad, rtol=1e-3)
