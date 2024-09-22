import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

def historical_decomposition(data, B, e, p, detrend, b):
    # data: DataFrame with 'dx' and 'dn'
    # B: Structural impact matrix
    # e: Reduced-form residuals
    # p: Number of lags
    # detrend: 1 if detrended, 0 otherwise
    # b: Coefficient matrix from varlrMilliken
    
    # Compute structural shocks
    eps = np.linalg.solve(B, e.T).T  # Shape: (T, nvar)
    
    T, nvar = e.shape
    
    # Extract VAR coefficient matrices from b
    AR_matrices = []
    for i in range(p):
        start_idx = 1 + i * nvar
        end_idx = 1 + (i + 1) * nvar
        Ai = b[start_idx:end_idx, :].T
        AR_matrices.append(Ai)
    # A_matrices is an array of shape (nvar, nvar, p)
    A_matrices = np.stack(AR_matrices, axis=2)
    
    # Initialize contributions
    contrib_tech = np.zeros((T, nvar))
    contrib_demand = np.zeros((T, nvar))
    
    for shock_idx in range(nvar):
        eps_shock = np.zeros_like(eps)
        eps_shock[:, shock_idx] = eps[:, shock_idx]
    
        contrib = np.zeros((T, nvar))
        for t in range(T):
            if t == 0:
                past_contrib = np.zeros(nvar)
            else:
                past_contrib = np.zeros(nvar)
                for lag in range(1, min(p, t) + 1):
                    A_lag = A_matrices[:, :, lag - 1]
                    past_contrib += A_lag @ contrib[t - lag, :]
    
            # Current structural shock
            eps_t = eps_shock[t, :]
    
            # Contribution from the structural shock
            contrib_shock = B @ eps_t
    
            # Total contribution at time t
            contrib[t, :] = past_contrib + contrib_shock
    
        if shock_idx == 0:
            # Technology shock
            contrib_tech = contrib
        else:
            # Demand shock
            contrib_demand = contrib
    
    # Assign contributions
    dxt = contrib_tech[:, 0]
    dnt = contrib_tech[:, 1]
    dxd = contrib_demand[:, 0]
    dnd = contrib_demand[:, 1]
    dyt = dxt + dnt
    dyd = dxd + dnd
    
    # Adjust for detrending
    if detrend == 1:
        dnt_adj = np.zeros_like(dnt)
        dnd_adj = np.zeros_like(dnd)
        dnt_adj[0] = dnt[0]
        dnd_adj[0] = dnd[0]
        dnt_adj[1:] = dnt[1:] - dnt[:-1]
        dnd_adj[1:] = dnd[1:] - dnd[:-1]
        dnt = dnt_adj
        dnd = dnd_adj
    
    # Compute cumulative sums
    yt = np.cumsum(dyt)
    xt = np.cumsum(dxt)
    nt = np.cumsum(dnt)
    yd = np.cumsum(dyd)
    xd = np.cumsum(dxd)
    nd = np.cumsum(dnd)
    
    if detrend == 1:
        nt = dnt
        nd = dnd
    
    # Compile results
    decomposition = pd.DataFrame({
        'dyt': dyt,
        'dxt': dxt,
        'dnt': dnt,
        'dyd': dyd,
        'dxd': dxd,
        'dnd': dnd,
        'yt': yt,
        'xt': xt,
        'nt': nt,
        'yd': yd,
        'xd': xd,
        'nd': nd
    }, index=data.index[-T:])
    
    return decomposition