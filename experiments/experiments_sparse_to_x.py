import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils_experiments import get_results, plot_results, get_data, plot_settings

colors, ibm_cb = plot_settings()

SEED = 1
np.random.seed(SEED)
random.seed(SEED)

data_args = {
    "process_type": "ou_sparse_to_x",
    "basis_type": "cosine",     # "cosine" | "haar"
    "fraction": 0.1,
    "beta": np.array([[5.]]),
    "band": list(range(0, 500))  # list(range(0, 50)) | None
}

method_args = {
    "a": 0.9,
    "method": "modified_torrent",        # "torrent" | "bfs" | "modified_torrent"
}

m = 10
noise_vars = [1, 4]                                      # [0, 1, 4] | [0, 1]
num_data = [2 ** k for k in range(5, 14)]  # [4 * 2 ** k for k in range(0, 5)] + [1024] | [8, 12, 16]

# ----------------------------------
# run experiments
# ----------------------------------
sample_data = None
for i in range(len(noise_vars)):
    print("Noise Variance: ", noise_vars[i])
    res = {"DecoR": [], "ols": []}
    for n in num_data:
        print("number of data points: ", n)
        res["DecoR"].append([])
        res["ols"].append([])
        for _ in range(m):
            data_values = get_data(n, **data_args, noise_var=noise_vars[i])

            if n == num_data[-1] and i == 0:
                sample_data = data_values

            estimates_decor = get_results(x=data_values["x"],
                y=data_values["y"],
                basis=data_values["basis"],
                **method_args)
            res["DecoR"][-1].append(np.linalg.norm(estimates_decor - data_args["beta"].T, ord=1))

            estimates_ols = get_results(x=data_values["x"],
                y=data_values["y"],
                basis=data_values["basis"],
                method="ols", a=method_args["a"])
            res["ols"][-1].append(np.linalg.norm(estimates_ols - data_args["beta"].T, ord=1))

        # print mean and std of the results
        print(f"DecoR: mean: {np.mean(res['DecoR'][-1]):.2f}, std_mean: {np.std(res['DecoR'][-1]) / np.sqrt(n):.2f}")
        print(f"OLS: mean: {np.mean(res['ols'][-1]):.2f}, std_mean: {np.std(res['ols'][-1]) / np.sqrt(n):.2f}")

    res["DecoR"], res["ols"] = np.array(res["DecoR"]), np.array(res["ols"])

    plot_results(res, num_data, m, colors=colors[i])

# ----------------------------------
# plotting
# ----------------------------------

titles = {"blp": "Band-Limited", "ou": "Ornstein-Uhlenbeck", "ou_sparse_to_x": "AR(1) (sparse to x, dense to y)", "blp_sparse_to_x" : "BLP (sparse to x, dense to y)"}
titles_basis = {"cosine": "", "haar": ", Haar basis"}
titles_dim = {1: "", 2: ", 2-dimensional"}


def get_handles():
    point_1 = Line2D([0], [0], label='OLS', marker='o',
                     markeredgecolor='w', color=ibm_cb[5], linestyle='-')
    point_2 = Line2D([0], [0], label='DecoR', marker='X',
                     markeredgecolor='w', color=ibm_cb[5], linestyle='-')
    point_3 = Line2D([0], [0], label="$\sigma_{\eta}^2 = $" + str(noise_vars[0]), markersize=10,
                     color=ibm_cb[1], linestyle='-')
    point_4 = Line2D([0], [0], label="$\sigma_{\eta}^2 = $" + str(noise_vars[1]), markersize=10,
                     color=ibm_cb[4], linestyle='-')
    if len(noise_vars) == 3:
        point_5 = Line2D([0], [0], label="$\sigma_{\eta}^2 = $" + str(noise_vars[2]), markersize=10,
                         color=ibm_cb[2], linestyle='-')

        return [point_1, point_2, point_3, point_4, point_5]
    return [point_1, point_2, point_3, point_4]

plt.xlabel("number of data points")
plt.ylabel("mean absolute error")
plt.title(titles[data_args["process_type"]]
          + titles_basis[data_args["basis_type"]]
          + titles_dim[len(data_args["beta"])])
plt.xscale('log')
plt.xlim(left=num_data[0] - 2)
plt.hlines(0, num_data[0], num_data[-1], colors='black', linestyles='dashed')

plt.legend(handles=get_handles(), loc="upper right")

plt.tight_layout()
plt.show()
