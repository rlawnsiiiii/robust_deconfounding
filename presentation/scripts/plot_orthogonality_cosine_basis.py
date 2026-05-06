import numpy as np
import matplotlib.pyplot as plt

def phi(k, x, T=1.0):
    """Evaluates the continuous cosine basis function."""
    y = np.zeros_like(x)
    if k == 0:
        y[:] = 1.0
    else:
        y = np.sqrt(2) * np.cos(k * np.pi * x / T)
    return y

# --- 1. Set Parameters ---
T = 1.0
x = np.linspace(0, T, 1000) # 1000 points for a smooth continuous curve

# Choose your frequencies (Change these to test!)
k = 1
m = 2

# --- 2. Calculate Waves and Product ---
wave_k = phi(k, x, T)
wave_m = phi(m, x, T)
product_wave = wave_k * wave_m

# Calculate the actual Inner Product (Integral of the product wave)
inner_product = np.trapezoid(product_wave, x)

# --- 3. Create the Visualization ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top Plot: The Individual Waves
ax1.plot(x, wave_k, label=f'$\phi_{k}(x)$', color='#4A90E2', linewidth=2)
ax1.plot(x, wave_m, label=f'$\phi_{m}(x)$', color='#E74C3C', linewidth=2)
ax1.axhline(0, color='black', linewidth=1, linestyle='--')
ax1.set_title(f'Individual Basis Functions ($k={k}$ and $m={m}$)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.legend(loc='upper right', fontsize=12)

# Bottom Plot: The Product Wave & Filled Areas
ax2.plot(x, product_wave, color='purple', linewidth=2, label='Product: $\phi_k(x) \\times \phi_m(x)$')

# Fill positive area green, negative area red
ax2.fill_between(x, product_wave, 0, where=(product_wave >= 0), color='green', alpha=0.4, label='Positive Area')
ax2.fill_between(x, product_wave, 0, where=(product_wave < 0), color='red', alpha=0.4, label='Negative Area')

ax2.axhline(0, color='black', linewidth=1, linestyle='--')
ax2.set_title(f'Inner Product (Net Area) $\\approx$ {inner_product:.5f}', fontsize=14, fontweight='bold')
ax2.set_xlabel('$x$', fontsize=12)
ax2.set_ylabel('Product Amplitude', fontsize=12)
ax2.legend(loc='upper right', fontsize=11)

plt.tight_layout()
plt.show()