import numpy as np

def heston_monte_carlo(S0, v0, r, kappa, theta, sigma, rho, T, N, M):

    dt = T / N
    
    # Initialize arrays for Asset Price and Variance
    S = np.zeros((N + 1, M))
    v = np.zeros((N + 1, M))
    
    S[0] = S0
    v[0] = v0
    
    # Generate independent standard normal random variables
    Z1 = np.random.normal(0.0, 1.0, size=(N, M))
    Z2 = np.random.normal(0.0, 1.0, size=(N, M))
    
    # Enforce correlation using Cholesky Decomposition
    W_S = Z1
    W_v = rho * Z1 + np.sqrt(1 - rho**2) * Z2
    
    # Time-stepping simulation loop
    for t in range(1, N + 1):
        # 1. Full truncation boundary condition to prevent negative variance
        v_prev_truncated = np.maximum(v[t-1], 0)
        
        # 2. Update variance process via Euler-Maruyama discretization
        v[t] = (v[t-1] + 
                kappa * (theta - v_prev_truncated) * dt + 
                sigma * np.sqrt(v_prev_truncated * dt) * W_v[t-1])
        
        # 3. Update asset price process via log-price Euler discretization
        S[t] = S[t-1] * np.exp((r - 0.5 * v_prev_truncated) * dt + 
                               np.sqrt(v_prev_truncated * dt) * W_S[t-1])
        
    return S, v

# Example Market and Model Parameters
market_price = 100.0       # S0
initial_var = 0.04        # v0 (equivalent to 20% initial volatility)
risk_free = 0.05          # r (5% interest rate)
mean_rev_speed = 2.0      # kappa
long_term_var = 0.04      # theta
vol_of_vol = 0.3          # sigma
correlation = -0.7        # rho (negative represents asset drop when vol spikes)
maturity = 1.0            # T (1 year)
steps = 252               # N (daily intervals for a trading year)
simulations = 100000      # M (number of simulation iterations)

# Run the Engine
asset_paths, variance_paths = heston_monte_carlo(
    market_price, initial_var, risk_free, mean_rev_speed, 
    long_term_var, vol_of_vol, correlation, maturity, steps, simulations
)

# Application Example: Price a European Call Option with Strike Price K = 100
K = 100.0
terminal_prices = asset_paths[-1]
payoffs = np.maximum(terminal_prices - K, 0)
option_price = np.exp(-risk_free * maturity) * np.mean(payoffs)

print(f"Simulated European Call Option Price: {option_price:.4f}")
