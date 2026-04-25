import numpy as np
import random
import matplotlib.pyplot as plt
from utils_experiments import get_data
from robust_deconfounding.utils import cosine_basis  # Import the true basis!

SEED = 12
np.random.seed(SEED)
random.seed(SEED)

FRACTION = 0.01
N = 100

data_args = {
    "process_type": "ou",      # "ou" | "blp"
    "basis_type": "cosine",    # "cosine" | "haar"
    "fraction": FRACTION,
    "beta": np.array([[3.]]),
    "band": list(range(0, 50))
}
noise_var = 1
n = N

data = get_data(n, **data_args, noise_var=noise_var)
x, y, outlier_points = data["x"], data["y"], data["outlier_points"]
t = np.arange(n)

k_idx = np.argmax(outlier_points)

# 1. Use the exact basis matrix used in data generation
basis = cosine_basis(n)

# 2. Calculate the total noisy error
error_signal = y.squeeze() - 3 * x.squeeze()

# 3. Project the error into the frequency domain, isolate the outlier index,
# and project it back into the time domain to get the PERFECT, smooth confounder wave.
error_coeffs = basis.T @ error_signal.reshape(-1, 1) / n
true_confounder_wave = basis @ (error_coeffs * outlier_points)

# --- Plotting ---
fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# Top Plot: Just 3*x and y
axs[0].plot(t, 3 * x.squeeze(), label="$3 \cdot x$", color="tab:blue")
axs[0].plot(t, y.squeeze(), label="$y$", color="tab:orange", alpha=0.8)
axs[0].set_ylabel("Value")
axs[0].set_title("Main Signals: True Causal Effect ($3 \cdot x$) vs. Observed ($y$)")
axs[0].legend(loc="upper right")

# Bottom Plot: The noisy error AND the true isolated confounder wave
axs[1].plot(t, error_signal, label="Total Error: $y - 3 \cdot x$", color="tab:green", alpha=0.5)
axs[1].plot(t, true_confounder_wave.squeeze(), label=f"True Confounder Wave ($10 \cdot k$)", color="tab:red", linestyle="--", linewidth=2)
axs[1].set_ylabel("Amplitude / Difference")
axs[1].set_xlabel("Time ($t$)")
axs[1].set_title(f"Error Analysis (Frequency Index $k={k_idx}$)")
axs[1].legend(loc="upper right")

plt.tight_layout()

# Save
plt.savefig("synthetic_data_signals_and_error.png", dpi=300, bbox_inches="tight")
plt.close()
