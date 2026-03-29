import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality style with larger fonts
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 20

# Define data
# Chunk sizes in ms: 960ms * [1, 2, 3, 4]
chunk_sizes = [960, 1920, 2880, 3840]

# Actual measured Retriever / LLM inference time ratios
# lm=1: mean=0.156924 median=0.15 n=3583
# lm=2: mean=0.137083 median=0.1221 n=1795
# lm=3: mean=0.0997956 median=0.0851 n=1199
# lm=4: mean=0.0770423 median=0.0657 n=899
overhead_ratio = [0.156924, 0.137083, 0.0997956, 0.0770423]

# Create figure with single plot (compact size)
fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

# Define color
color = '#ff0000'  # red (matching RASST color from other plots)

# Plot ratio curve
line, = ax.plot(chunk_sizes, overhead_ratio, 
                color=color, marker='*', 
                linewidth=3, markersize=15, label='Retriever / LLM', zorder=2)

# Set labels
ax.set_xlabel('Chunk Size (ms)', fontsize=18)
ax.set_ylabel('Computation Overhead\n(Retriever / LLM Inference)', fontsize=18)

# Add grid
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

# Set axis limits
ax.set_xlim(800, 4000)
ax.set_xticks(chunk_sizes)

# Set y-axis limits based on data
y_min = min(overhead_ratio) * 0.9
y_max = max(overhead_ratio) * 1.1
ax.set_ylim(y_min, y_max)

# Format y-axis to show ratio with 3 decimal places
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

# Add annotations for specific points (optional)
# Uncomment if you want to highlight specific values
# for i, (x, y) in enumerate(zip(chunk_sizes, overhead_ratio)):
#     ax.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 10),
#                 textcoords='offset points', ha='center', fontsize=9)

# Save figure
plt.savefig('/home/siqiouya/code/infinisst-omni/plots-rag/figures/inference_efficiency.pdf', 
            dpi=300, bbox_inches='tight')
print("Figure saved as 'inference_efficiency.pdf'")

# Also display data in console
print("\nData Summary:")
print(f"{'Chunk Size (ms)':<18} {'Ratio (Retriever/LLM)':<25}")
print("-" * 45)
for cs, ratio in zip(chunk_sizes, overhead_ratio):
    print(f"{cs:<18} {ratio:.6f}")

# Display
plt.show()
