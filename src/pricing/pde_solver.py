import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator

# 1. Market and Heston Model Parameters
S0 = 100.0     # Initial stock price
K = 100.0      # Strike price
V0 = 0.04      # Initial variance (v = r_vol^2)
r = 0.03       # Risk-free rate
kappa = 2.0    # Rate of mean reversion
theta = 0.04   # Long-term mean variance
sigma = 0.3    # Volatility of variance (vol of vol)
rho = -0.5     # Asset-variance correlation
T = 1.0        # Time to maturity

# 2. Numerical Grid Setup
Ns = 50        # Stock price grid subdivisions
Nv = 30        # Variance grid subdivisions
Nt = 50        # Time steps

S_max = 300.0  # Safe high boundary for stock price (typically 3-4x Strike)
V_max = 1.0    # High boundary for variance

ds = S_max / Ns
dv = V_max / Nv
dt = T / Nt

s_grid = np.linspace(0, S_max, Ns + 1)
v_grid = np.linspace(0, V_max, Nv + 1)
N_total = (Ns + 1) * (Nv + 1)

# Helper function to map 2D coordinates (i, j) into flat 1D vector index
def get_idx(i, j):
    return i + j * (Ns + 1)

# 3. Constructing the 2D Sparse Infinitesimal Generator Matrix (L)
row, col, data = [], [], []

for j in range(Nv + 1):
    v = v_grid[j]
    for i in range(Ns + 1):
        s = s_grid[i]
        idx = get_idx(i, j)
        
        # Boundary rows are skipped during generation (overwritten directly)
        if i == 0 or i == Ns or j == 0 or j == Nv:
            row.append(idx); col.append(idx); data.append(0.0)
            continue
            
        # --- Internal Node Discretization (Central Differences) ---
        # u term
        row.append(idx); col.append(idx); data.append(-r)
        
        # First-order s-derivative: r * s * u_s
        row.append(idx); col.append(get_idx(i+1, j)); data.append(0.5 * r * s / ds)
        row.append(idx); col.append(get_idx(i-1, j)); data.append(-0.5 * r * s / ds)
        
        # First-order v-derivative: kappa * (theta - v) * u_v
        row.append(idx); col.append(get_idx(i, j+1)); data.append(0.5 * kappa * (theta - v) / dv)
        row.append(idx); col.append(get_idx(i, j-1)); data.append(-0.5 * kappa * (theta - v) / dv)
        
        # Second-order s-derivative: 0.5 * s^2 * v * u_ss
        row.append(idx); col.append(get_idx(i+1, j)); data.append(0.5 * s**2 * v / (ds**2))
        row.append(idx); col.append(get_idx(i, j));   data.append(-s**2 * v / (ds**2))
        row.append(idx); col.append(get_idx(i-1, j)); data.append(0.5 * s**2 * v / (ds**2))
        
        # Second-order v-derivative: 0.5 * sigma^2 * v * u_vv
        row.append(idx); col.append(get_idx(i, j+1)); data.append(0.5 * sigma**2 * v / (dv**2))
        row.append(idx); col.append(get_idx(i, j));   data.append(-sigma**2 * v / (dv**2))
        row.append(idx); col.append(get_idx(i, j-1)); data.append(0.5 * sigma**2 * v / (dv**2))
        
        # Mixed-order derivative: rho * sigma * s * v * u_sv
        coeff_sv = rho * sigma * s * v / (4.0 * ds * dv)
        row.append(idx); col.append(get_idx(i+1, j+1)); data.append(coeff_sv)
        row.append(idx); col.append(get_idx(i+1, j-1)); data.append(-coeff_sv)
        row.append(idx); col.append(get_idx(i-1, j+1)); data.append(-coeff_sv)
        row.append(idx); col.append(get_idx(i-1, j-1)); data.append(coeff_sv)

L = sp.csr_matrix((data, (row, col)), shape=(N_total, N_total))
I = sp.eye(N_total, format='csr')

# Crank-Nicolson matrices: (I - 0.5 * dt * L) * U^n = (I + 0.5 * dt * L) * U^{n+1}
A_lhs = I - 0.5 * dt * L
A_rhs = I + 0.5 * dt * L

# Force LHS identity rows on boundaries to preserve explicit assignment inside loop
A_lhs_lil = A_lhs.tolil()
for j in range(Nv + 1):
    for i in range(Ns + 1):
        if i == 0 or i == Ns or j == 0 or j == Nv:
            idx = get_idx(i, j)
            A_lhs_lil.rows[idx] = [idx]
            A_lhs_lil.data[idx] = [1.0]
A_lhs = A_lhs_lil.tocsr()

# 4. Terminal Payoff Configuration (t = T)
U = np.zeros((Ns + 1, Nv + 1))
for i in range(Ns + 1):
    for j in range(Nv + 1):
        U[i, j] = max(s_grid[i] - K, 0.0)

U_flat = U.flatten()

# 5. Backward Time-Stepping Loop
for n in range(Nt):
    t_next = T - (n + 1) * dt  # Moving backward toward t = 0
    
    # Process explicit half step
    b = A_rhs.dot(U_flat)
    
    # Overwrite boundaries on RHS to enforce Dirichlet boundary conditions
    for j in range(Nv + 1):
        b[get_idx(0, j)] = 0.0                                          # S = 0
        b[get_idx(Ns, j)] = S_max - K * np.exp(-r * (T - t_next))       # S = S_max
    for i in range(Ns + 1):
        b[get_idx(i, 0)] = max(s_grid[i] - K, 0.0)                      # V = 0 proxy
        b[get_idx(i, Nv)] = max(s_grid[i] - K * np.exp(-r * (T - t_next)), 0.0) # V = V_max
        
    # Solve system using a highly optimized direct sparse solver
    U_flat = spsolve(A_lhs, b)

# Reshape result vector back into our grid matrix
U_final = U_flat.reshape((Ns + 1, Nv + 1))

# 6. Extraction via Interpolation
interp = RegularGridInterpolator((s_grid, v_grid), U_final, method='linear')
calculated_price = interp([S0, V0])[0]

print(f"Heston European Call Option Price: {calculated_price:.4f}")
