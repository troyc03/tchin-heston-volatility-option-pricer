import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.interpolate import RegularGridInterpolator

class HestonModel:
    def __init__(self, S0, v0, kappa, theta, sigma, rho, r, q, K, T):
        self.S0 = S0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.r = r
        self.q = q
        self.K = K
        self.T = T

class GBMSimulator:
    def heston_gbm_simulator(self, model, n_paths=10, n_steps=252):
        self.model = model
        dt = self.model.T / n_steps
        S = np.zeros((n_steps + 1, n_paths))
        v = np.zeros((n_steps + 1, n_paths))
        
        # Correctly broadcast initial row states across all paths
        S[0, :] = self.model.S0
        v[0, :] = self.model.v0
        
        for t in range(1, n_steps + 1):
            z1 = np.random.normal(0, 1, n_paths)
            z2 = self.model.rho * z1 + np.sqrt(1 - self.model.rho**2) * np.random.normal(0, 1, n_paths)
            
            v_prev = np.maximum(v[t-1, :], 0)
            
            # Update path states sequentially over time
            v[t, :] = v_prev + self.model.kappa * (self.model.theta - v_prev) * dt + self.model.sigma * np.sqrt(v_prev * dt) * z2
            v[t, :] = np.maximum(v[t, :], 0)
            
            # Asset Price simulation formula step
            S[t, :] = S[t-1, :] * np.exp((self.model.r - self.model.q - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * z1)
            
        return S, v

def v0_placeholder_check(model):
    return getattr(model, 'v0', 0.04)

class HestonPDE:
    def __init__(self, model, NS=80, Nv=40, Nt=100):
        self.model = model
        self.NS = NS
        self.Nv = Nv
        self.Nt = Nt
        self.S_max = 3.0 * model.S0
        self.v_max = 4.0 * model.theta
        self.ds = self.S_max / self.NS
        self.dv = self.v_max / self.Nv
        self.dt = model.T / self.Nt
        self.S_vec = np.linspace(0, self.S_max, self.NS + 1)
        self.v_vec = np.linspace(0, self.v_max, self.Nv + 1)
        self.S_grid, self.v_grid = np.meshgrid(self.S_vec, self.v_vec, indexing='ij')

    def pde_solver(self, option_type='call'):
        dt, ds, dv = self.dt, self.ds, self.dv
        m = self.model
        if option_type == 'call':
            U = np.maximum(self.S_grid - m.K, 0.0)
        else:
            U = np.maximum(m.K - self.S_grid, 0.0)
            
        # Time-stepping loop (backward in time)
        for step in range(1, self.Nt + 1):
            tau = step * dt
            U_next = U.copy()
            
            # Internal nodes update (finite differences)
            for i in range(1, self.NS):
                S = self.S_vec[i]
                for j in range(1, self.Nv):
                    v = self.v_vec[j]
                    
                    # 1st Derivatives (Central Differences)
                    dU_dS = (U[i+1, j] - U[i-1, j]) / (2 * ds)
                    dU_dv = (U[i, j+1] - U[i, j-1]) / (2 * dv)
                    
                    # 2nd Derivatives (Central Differences)
                    d2U_dS2 = (U[i+1, j] - 2 * U[i, j] + U[i-1, j]) / (ds**2)
                    d2U_dv2 = (U[i, j+1] - 2 * U[i, j] + U[i, j-1]) / (dv**2)
                    
                    # Mixed Derivative
                    d2U_dSdv = (U[i+1, j+1] - U[i+1, j-1] - U[i-1, j+1] + U[i-1, j-1]) / (4 * ds * dv)
                    
                    # PDE operator valuation components
                    drift_S = (m.r - m.q) * S * dU_dS
                    drift_v = m.kappa * (m.theta - v) * dU_dv
                    diff_S = 0.5 * v * (S**2) * d2U_dS2
                    diff_v = 0.5 * (m.sigma**2) * v * d2U_dv2
                    cross_diff = m.rho * m.sigma * v * S * d2U_dSdv
                    discount = -m.r * U[i, j]
                    
                    # FTCS Explicit Update Step
                    U_next[i, j] = U[i, j] + dt * (drift_S + drift_v + diff_S + diff_v + cross_diff + discount)
            
            # Apply Boundary Conditions for the next time-step
            # 1. S = 0 Boundary
            U_next[0, :] = U_next[1, :] * np.exp(-m.r * dt)
            
            # 2. S = S_max Boundary
            if option_type == 'call':
                U_next[self.NS, :] = self.S_max * np.exp(-m.q * tau) - m.K * np.exp(-m.r * tau)
            else:
                U_next[self.NS, :] = 0.0
                
            # 3. v = 0 Boundary (Forward difference for variance drift due to zero diffusion)
            for i in range(1, self.NS):
                dU_dS = (U_next[i+1, 0] - U_next[i-1, 0]) / (2 * ds)
                dU_dv = (U_next[i, 1] - U_next[i, 0]) / dv 
                d2U_dS2 = (U_next[i+1, 0] - 2 * U_next[i, 0] + U_next[i-1, 0]) / (ds**2)
                
                drift_S = (m.r - m.q) * self.S_vec[i] * dU_dS
                drift_v = m.kappa * m.theta * dU_dv
                discount = -m.r * U_next[i, 0]
                
                U_next[i, 0] = U[i, 0] + dt * (drift_S + drift_v + discount)
                
            # 4. v = v_max Boundary (Linear approximation)
            U_next[:, self.Nv] = 2 * U_next[:, self.Nv-1] - U_next[:, self.Nv-2]
            
            U = U_next.copy()
            
        return U

class HestonOptionSurface:
    def plot_surface(self, S_vec, v_vec, U, title="Option Price Surface"):
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(projection='3d')
        SG, VG = np.meshgrid(S_vec, v_vec, indexing='ij')
        surf = ax.plot_surface(SG, VG, U, cmap='viridis', edgecolor='none', alpha=0.9)
        ax.set_xlabel('Stock Price (S)', labelpad=10)
        ax.set_ylabel('Variance (v)', labelpad=10)
        ax.set_zlabel('Option Value (U)', labelpad=10)
        ax.set_title(title, fontsize=14)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        plt.show()

def plot_simulated_paths(S_paths, v_paths, T):
    n_steps = S_paths.shape[0] - 1
    time_grid = np.linspace(0, T, n_steps + 1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    
    # Plot Asset Price Paths
    ax1.plot(time_grid, S_paths, linewidth=1.2)
    ax1.set_ylabel('Asset Price ($S_t$)', fontsize=11)
    ax1.set_title('Heston Model Simulation: Asset Price Paths', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Plot Variance Paths
    ax2.plot(time_grid, v_paths, linewidth=1.2)
    ax2.set_xlabel('Time ($t$)', fontsize=11)
    ax2.set_ylabel('Variance ($v_t$)', fontsize=11)
    ax2.set_title('Heston Model Simulation: Volatility Paths', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def main():
    model = HestonModel(
        S0=100.0, v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, r=0.03, q=0.01, K=100.0, T=0.5
    )
    
    # 1. Path Generation & Plotting
    sim = GBMSimulator()
    # Simulating 10 standalone sample paths
    S_paths, v_paths = sim.heston_gbm_simulator(model, n_paths=10, n_steps=252)
    plot_simulated_paths(S_paths, v_paths, model.T)
    
    # 2. Complete Grid PDE Calculation
    pde = HestonPDE(model, NS=60, Nv=30, Nt=1500)
    call_grid = pde.pde_solver(option_type='call')
    put_grid = pde.pde_solver(option_type='put')
    
    interp_call = RegularGridInterpolator((pde.S_vec, pde.v_vec), call_grid)
    interp_put = RegularGridInterpolator((pde.S_vec, pde.v_vec), put_grid)
    
    call_price = interp_call([model.S0, model.v0])[0]
    put_price = interp_put([model.S0, model.v0])[0]
    
    print(f'Heston Grid Interpolated Call Option Price: {call_price:.4f}')
    print(f'Heston Grid Interpolated Put Option Price: {put_price:.4f}')
    
    surface_plotter = HestonOptionSurface()
    surface_plotter.plot_surface(pde.S_vec, pde.v_vec, call_grid, title="Heston PDE European Call Option Surface")

if __name__ == '__main__':
    main()
