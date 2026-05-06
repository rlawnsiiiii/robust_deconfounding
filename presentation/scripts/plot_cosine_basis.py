import numpy as np
import matplotlib.pyplot as plt


def phi(k, x, T=1.0):
    """
    Evaluates the cosine basis function phi_k(x) as defined in the text.
    """
    # Create an array of the same shape as x to store the output
    y = np.zeros_like(x)

    if k == 0:
        # If k=0, the function is just a constant 1
        y[:] = 1.0
    else:
        # Otherwise, it is the scaled cosine wave.
        # (Assuming the formula incorporates T to scale the domain correctly)
        y = np.sqrt(2) * np.cos(k * np.pi * x / T)

    return y


# Define the interval [0, T]
T = 1.0
x = np.linspace(0, T, 500)

# Create the plot
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-whitegrid')

# Plot for multiple frequencies k
frequencies_to_plot = [0, 2, 4, 6, 8]

for k in frequencies_to_plot:
    y = phi(k, x, T)
    plt.plot(x, y, label=f'$k = {k}$', linewidth=2)

# Formatting the plot
plt.title('Continuous Cosine Basis Functions $\phi_k(x)$', fontsize=15, fontweight='bold')
plt.xlabel('$x$', fontsize=13)
plt.ylabel('$\phi_k(x)$', fontsize=13)
plt.axhline(0, color='black', linewidth=1, linestyle='--')  # Add a zero line
plt.legend(fontsize=12, loc='lower right')
plt.grid(True, alpha=0.5)

plt.tight_layout()
plt.show()