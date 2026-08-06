import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Generate Synthetic Heston Data
np.random.seed(42)
S0, v0, kappa, theta, sigma, rho, T, N, M = (
    100.0,
    0.04,
    2.0,
    0.04,
    0.3,
    -0.7,
    1.0,
    50,
    1000,
)
dt = T / N

time_grid = np.linspace(0, T, N + 1)
S = np.zeros((M, N + 1))
v = np.zeros((M, N + 1))
S[:, 0] = S0
v[:, 0] = v0

for t in range(1, N + 1):
  Z1 = np.random.normal(0, 1, M)
  Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, M)
  v[:, t] = np.maximum(
      0,
      v[:, t - 1]
      + kappa * (theta - v[:, t - 1]) * dt
      + sigma * np.sqrt(np.maximum(0, v[:, t - 1])) * np.sqrt(dt) * Z2,
  )
  S[:, t] = S[:, t - 1] * np.exp(
      -0.5 * v[:, t - 1] * dt
      + np.sqrt(np.maximum(0, v[:, t - 1])) * np.sqrt(dt) * Z1
  )

# Prepare training data
X_data, y_data = [], []
for i in range(M):
  for t in range(N):
    X_data.append([time_grid[t], S[i, t], v[i, t]])
    y_data.append([v[i, t + 1]])

X = torch.tensor(X_data, dtype=torch.float32)
y = torch.tensor(y_data, dtype=torch.float32)


# 2. Define Neural Network
class HestonSDE_NN(nn.Module):

  def __init__(self):
    super(HestonSDE_NN, self).__init__()
    self.net = nn.Sequential(
        nn.Linear(3, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )

  def forward(self, x):
    return self.net(x)


model = HestonSDE_NN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 3. Train Model
epochs = 150
for epoch in range(epochs):
  optimizer.zero_grad()
  predictions = model(X)
  loss = criterion(predictions, y)
  loss.backward()
  optimizer.step()

# 4. Predict Trajectory for a Single Path
# We evaluate path index 0 across all time steps
test_path_idx = 0
predicted_v = [v0]

model.eval()
with torch.no_grad():
  for t in range(N):
    # Input format: [time, price, current variance]
    input_step = torch.tensor(
        [[time_grid[t], S[test_path_idx, t], predicted_v[-1]]],
        dtype=torch.float32,
    )
    next_v_pred = model(input_step).item()
    # Force variance to be non-negative
    predicted_v.append(max(0.0, next_v_pred))

# 5. Plot Comparison
plt.figure(figsize=(10, 5))
plt.plot(
    time_grid,
    v[test_path_idx, :],
    label="Actual Heston Variance (SDE)",
    color="black",
    linewidth=2,
)
plt.plot(
    time_grid,
    predicted_v,
    label="Neural Network Predicted Variance",
    color="crimson",
    linestyle="--",
    linewidth=2,
)
plt.xlabel("Time (T)")
plt.ylabel("Variance (v)")
plt.title(f"Trajectory Comparison: Actual vs Predicted Variance (Path {test_path_idx})")
plt.legend()
plt.grid(True)
plt.show()
