import matplotlib.pyplot as plt
import numpy as np

# Set up the figure and axes
# Use a wide figure to accommodate the 3 subplots and the arrows between them
fig, axs = plt.subplots(1, 3, figsize=(18, 4))
plt.subplots_adjust(wspace=0.8) # Increase horizontal space for the arrows

# --- Plot 1: time domain ---
# Approximate dummy data points
x1 = [1, 3, 3, 4, 5, 5, 7, 7.5, 8.5]
y1 = [5, 3, 5, 8, 1.5, 6, 4.5, 5.5, 4.5]
axs[0].scatter(x1, y1, s=60, color='tab:blue')
axs[0].set_title('time domain', fontsize=20)
axs[0].set_xlabel('$X$', fontsize=18)
axs[0].set_ylabel('$Y$', fontsize=18)
axs[0].set_xticks([]) # Remove x tick marks
axs[0].set_yticks([]) # Remove y tick marks
axs[0].set_xlim(0, 9.5)
axs[0].set_ylim(0, 9)

# Arrow 1 and Text (Transforming to Frequency Domain)
axs[0].annotate('', xy=(1.6, 0.5), xytext=(1.05, 0.5), xycoords='axes fraction',
                arrowprops=dict(facecolor='black', width=1.5, headwidth=8, headlength=10, shrink=0.05))
axs[0].text(1.275, 0.55, '$T$', ha='center', va='bottom', transform=axs[0].transAxes, fontsize=24)

# --- Plot 2: frequency domain ---
# Approximate dummy data points
x2 = [0.5, 1.2, 2.5, 3, 5, 5.5, 7.5, 8.5, 9.5]
y2 = [1, 6, 2, 7, 1.5, 5, 7, 5, 9]
axs[1].scatter(x2, y2, s=60, color='tab:blue')
axs[1].set_title('frequency domain', fontsize=20)
axs[1].set_xlabel('$X^\\phi$', fontsize=18)
axs[1].set_ylabel('$Y^\\phi$', fontsize=18)
axs[1].set_xticks([])
axs[1].set_yticks([])
axs[1].set_xlim(0, 10)
axs[1].set_ylim(0, 10)

# Arrow 2 and Text (Applying Robust Regression)
axs[1].annotate('', xy=(1.65, 0.5), xytext=(1.1, 0.5), xycoords='axes fraction',
                arrowprops=dict(facecolor='black', width=1.5, headwidth=8, headlength=10, shrink=0.05))
axs[1].text(1.32, 0.55, 'robust', ha='center', va='bottom', transform=axs[1].transAxes, fontsize=18, fontfamily='serif')
axs[1].text(1.32, 0.45, 'regression', ha='center', va='top', transform=axs[1].transAxes, fontsize=18, fontfamily='serif')

# --- Plot 3: frequency domain (robust regression results) ---
# Define inliers (green points)
x_in = [0.5, 1.5, 2.5, 6, 8, 9.5]
y_in = [0.5, 1.5, 2.5, 5.5, 7.5, 9.5]
# Define outliers (red points)
x_out = [1.2, 3.5, 5.5, 8.5]
y_out = [7.5, 8, 1, 5]

# Plot points
axs[2].scatter(x_in, y_in, color='green', s=60)
axs[2].scatter(x_out, y_out, color='red', s=60)

# Draw the dashed regression line
x_line = np.linspace(0, 10, 100)
y_line = x_line # A simple linear function to approximate the green dashed line
axs[2].plot(x_line, y_line, color='green', linestyle='--', linewidth=2.5)

axs[2].set_title('frequency domain', fontsize=20)
axs[2].set_xlabel('$X^\\phi$', fontsize=18)
axs[2].set_ylabel('$Y^\\phi$', fontsize=18)
axs[2].set_xticks([])
axs[2].set_yticks([])
axs[2].set_xlim(0, 10)
axs[2].set_ylim(0, 10)

# Ensure axis labels are positioned closely
for ax in axs:
    ax.tick_params(axis='both', which='both', length=0)

# Display the plot
plt.show()