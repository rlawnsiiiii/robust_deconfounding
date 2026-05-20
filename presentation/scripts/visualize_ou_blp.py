import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from scipy.fft import dct, idct

# --- 1. Simulation Parameters ---
n = 500
np.random.seed(42) # For reproducible plots

# --- 2. Generate Ornstein-Uhlenbeck (OU) Process ---
# We use phi = 1 - (0.8/n) to ensure high autocorrelation (true OU behavior)
ou_phi = 1 - (0.8 / n)
ar_params = np.array([1, -ou_phi])
ma_params = np.array([1 / np.sqrt(n)])

# Generate Time Domain
ou_time = ArmaProcess(ar_params, ma_params).generate_sample(nsample=n)

# Transform to Frequency Domain using DCT
ou_freq = dct(ou_time, type=2, norm='ortho')
ou_mag = np.abs(ou_freq)


# --- 3. Generate Band-Limited Process (BLP) ---
# We define a band of size 50 (e.g., indices 20 through 69)
band_start, band_end = 20, 70
band_idx = np.array([1 if i in range(band_start, band_end) else 0 for i in range(n)])

# 3a. Create the true frequencies (random weights strictly inside the band)
weights = np.random.normal(0, 1, size=n)
blp_freq_true = weights * band_idx

# 3b. Transform to Time Domain 
# (Using idct is mathematically identical to your `basis @ (weights * band_idx)`)
blp_time = idct(blp_freq_true, type=2, norm='ortho')

# 3c. Transform back to Frequency Domain using DCT
blp_freq = dct(blp_time, type=2, norm='ortho')
blp_mag = np.abs(blp_freq)


# --- 4. Visualization ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(2, 2, figsize=(14, 8))

# ---- Top Row: Time Domain ----
axs[0, 0].plot(ou_time, color='#E66100', linewidth=1.5)
axs[0, 0].set_title('Ornstein-Uhlenbeck (Time Domain)', fontsize=14, fontweight='bold')
axs[0, 0].set_xlabel('Time Step (t)', fontsize=12)
axs[0, 0].set_ylabel('Amplitude', fontsize=12)

axs[0, 1].plot(blp_time, color='#5D3A9B', linewidth=1.5)
axs[0, 1].set_title('Band-Limited Process (Time Domain)', fontsize=14, fontweight='bold')
axs[0, 1].set_xlabel('Time Step (t)', fontsize=12)
axs[0, 1].set_ylabel('Amplitude', fontsize=12)

# ---- Bottom Row: Frequency Domain ----
markerline_ou, stemlines_ou, baseline_ou = axs[1, 0].stem(ou_mag, linefmt='#E66100', markerfmt='o', basefmt=" ")
plt.setp(markerline_ou, 'markersize', 3) # Adjust this number to make it bigger or smaller
axs[1, 0].set_title('Ornstein-Uhlenbeck (DCT Frequency Domain)', fontsize=14, fontweight='bold')
axs[1, 0].set_xlabel('Frequency Index (k)', fontsize=12)
axs[1, 0].set_ylabel('Magnitude (Log Scale)', fontsize=12)
axs[1, 0].set_yscale('log') # Log scale reveals the infinite smearing of energy

markerline_blp, stemlines_blp, baseline_blp = axs[1, 1].stem(blp_mag, linefmt='#5D3A9B', markerfmt='o', basefmt=" ")
plt.setp(markerline_blp, 'markersize', 3)
axs[1, 1].set_title('Band-Limited Process (DCT Frequency Domain)', fontsize=14, fontweight='bold')
axs[1, 1].set_xlabel('Frequency Index (k)', fontsize=12)
axs[1, 1].set_ylabel('Magnitude', fontsize=12)
axs[1, 1].set_xlim(-5, 100) # Zoom in to show the strict cutoff boundaries

# ---- Add the dynamic caption ----
caption_text = (
    f"Simulation Parameters: Number of data points (n) = {n}. "
    f"Ornstein-Uhlenbeck AR coefficient (φ) = {ou_phi:.4f}. "
    f"Band-Limited Process active frequency band = [{band_start}, {band_end})."
)
# Place the text at the bottom center of the figure
fig.text(0.5, 0.02, caption_text, ha='center', fontsize=13, style='italic',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

# Adjust layout so the plots don't overlap with the caption
plt.subplots_adjust(bottom=0.1)

plt.savefig('ou_vs_blp_dct_with_caption.pdf', format='pdf', bbox_inches='tight')
plt.show()