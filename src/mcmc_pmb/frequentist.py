import numpy as np
from scipy.optimize import minimize
from .forward_model import MaterialBalanceModel, ProductionDataset

def run_frequentist(
    dataset: ProductionDataset, 
    model: MaterialBalanceModel,
    initial_guess: list[float] = [100.0, 0.4, 5.0] # N, m, J
) -> dict:
    """
    The 'Old School' Deterministic Method.
    Minimizes the Sum of Squared Errors (SSE) of pressure.
    """
    
    def objective(theta):
        # theta = [N, m, J]
        # Constraints: N>0, m>=0, J>=0
        if theta[0] <= 0 or theta[1] < 0 or theta[2] < 0:
            return 1e9 # Penalty
            
        try:
            preds = model.predict_pressures(theta, dataset)
            # SSE
            sse = np.sum((preds - dataset.pressure_measured)**2)
            return sse
        except:
            return 1e9

    result = minimize(
        objective, 
        initial_guess, 
        method='Nelder-Mead', # Robust non-gradient method often used in industry
        options={'maxiter': 1000}
    )
    
    return {
        "N": result.x[0],
        "m": result.x[1],
        "J": result.x[2],
        "sse": result.fun,
        "success": result.success
    }