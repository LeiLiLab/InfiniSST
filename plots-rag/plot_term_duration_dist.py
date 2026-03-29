import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality style
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Define data
bin_edges = [
      0.0,
      0.05,
      0.1,
      0.15,
      0.2,
      0.25,
      0.3,
      0.35,
      0.4,
      0.45,
      0.5,
      0.55,
      0.6,
      0.65,
      0.7,
      0.75,
      0.8,
      0.85,
      0.9,
      0.95,
      1.0,
      1.05,
      1.1,
      1.15,
      1.2,
      1.25,
      1.3,
      1.35,
      1.4,
      1.45,
      1.5,
      1.55,
      1.6,
      1.65,
      1.7,
      1.75,
      1.8,
      1.85,
      1.9,
      1.95,
      2.0,
      2.05,
      2.1,
      2.15,
      2.2,
      2.25,
      2.3,
      2.35,
      2.4,
      2.45,
      2.5,
      2.55,
      2.6,
      2.65,
      2.7,
      2.75,
      2.8,
      2.85,
      2.9,
      2.95,
      3.0
    ]

counts = [
      15,
      55,
      201,
      611,
      1517,
      2343,
      3277,
      3985,
      4307,
      4498,
      3378,
      3292,
      2836,
      2345,
      2286,
      1559,
      1430,
      1266,
      1033,
      887,
      667,
      550,
      538,
      404,
      373,
      262,
      212,
      190,
      145,
      128,
      91,
      73,
      65,
      35,
      37,
      25,
      21,
      19,
      19,
      9,
      7,
      9,
      10,
      4,
      2,
      2,
      1,
      3,
      1,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
    ]

# Calculate bin centers for plotting
bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]

# Calculate statistics
total_terms = sum(counts)
weighted_durations = sum(c * bc for c, bc in zip(counts, bin_centers))
mean_duration = weighted_durations / total_terms if total_terms > 0 else 0

# P99 (99th percentile)
p99_duration = 1.497

# Create figure (compact size)
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

# Plot histogram as bars
width = bin_edges[1] - bin_edges[0]  # bin width
print(len(bin_centers), len(counts))
ax.bar(bin_centers, counts, width=width*0.95, color='#1f77b4', alpha=0.7, 
       edgecolor='black', linewidth=0.5)

# Add mean line
ax.axvline(x=mean_duration, color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {mean_duration:.3f}s')

# Add p99 line
ax.axvline(x=p99_duration, color='orange', linestyle='-.', linewidth=2, 
           label=f'P99: {p99_duration:.3f}s')

# Set labels
ax.set_xlabel('Term Duration (seconds)', fontsize=14)
ax.set_ylabel('Count', fontsize=14)

# Add grid
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, axis='y')

# Set axis limits
ax.set_xlim(0, 3.0)
ax.set_ylim(0, max(counts) * 1.1)

# Set x-axis ticks
ax.set_xticks(np.arange(0, 3.1, 0.5))

# Add legend
ax.legend(loc='upper right', framealpha=0.9, fontsize=12)

# Save figure
plt.savefig('/home/siqiouya/code/infinisst-omni/plots-rag/figures/term_duration_dist.pdf', 
            dpi=300, bbox_inches='tight')
print("Figure saved as 'term_duration_dist.pdf'")

# Display statistics
print(f"\nStatistics:")
print(f"Total terms: {total_terms:,}")
print(f"Mean duration: {mean_duration:.3f}s")
print(f"P99 duration: {p99_duration:.3f}s")
print(f"Max count: {max(counts)} at bin {bin_centers[counts.index(max(counts))]:.2f}s")

# Display
plt.show()
