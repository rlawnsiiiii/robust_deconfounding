import numpy as np
import random
from robust_deconfounding import DecoR
from robust_deconfounding.robust_regression import Torrent

import matplotlib.pyplot as plt
import seaborn as sns

from utils_experiments import get_data, plot_settings

colors, ibm_cb = plot_settings()

SEED = 1
np.random.seed(SEED)
random.seed(SEED)

# Parameters
n = 1024 * 8
a = 0.7

data_args_single = {
    "process_type": "ou_sparse_to_x",
    "basis_type": "cosine",
    "fraction": 0.25,
    "beta": np.array([[3.]]),
    "band": list(range(0, 50)),
    "noise_var": 1.0
}

# Generate one dataset
data = get_data(n, **data_args_single)
X = data["x"]
y = data["y"].ravel()
basis = data["basis"]

# Fit DecoR (Torrent)
algo = Torrent(a=a, fit_intercept=False)
decor = DecoR(algo, basis)
decor.fit(X, y)

# Get removed frequencies
inliers = list(decor.algo.inliers_) if hasattr(decor.algo, "inliers_") else []
removed_freqs = list(set(range(n)) - set(inliers))

# Plot histogram
plt.figure(figsize=(5, 5))
sns.histplot(removed_freqs, color=ibm_cb[0])
plt.xlabel("Frequency")
plt.ylabel("Count")
plt.title("Histogram of Frequencies Removed by DecoR (OU Sparse-to-X)")
plt.tight_layout()
plt.savefig("sparse_to_x_removed_freqs.pdf")
plt.show()