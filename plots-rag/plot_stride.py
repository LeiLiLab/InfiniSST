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

# Data from table
stride_labels = ['0.96', '0.48', '0.24', '0.12']
x_pos = [0, 1, 2, 3]  # evenly spaced positions
paper_recall = [92.31, 93.08, 93.85, 93.85]
tagged_recall = [77.90, 80.76, 78.77, 77.99]

# Create figure with dual y-axes
fig, ax1 = plt.subplots(1, 1, figsize=(6, 4.5))
ax2 = ax1.twinx()

# Colors
color_paper = '#1f77b4'   # blue
color_tagged = '#ff7f0e'  # orange

# Plot paper recall on left y-axis
line1, = ax1.plot(x_pos, paper_recall,
                  color=color_paper, marker='o',
                  linewidth=3, markersize=10, label='Paper')

# Plot tagged recall on right y-axis
line2, = ax2.plot(x_pos, tagged_recall,
                  color=color_tagged, marker='s',
                  linewidth=3, markersize=10, label='Tagged')

# Set labels
ax1.set_xlabel('Stride $\\delta$ (s)', fontsize=18)
ax1.set_ylabel('Paper Recall@10 (%)', fontsize=18, color=color_paper)
ax2.set_ylabel('Tagged Recall@10 (%)', fontsize=18, color=color_tagged)

# Color the tick labels
ax1.tick_params(axis='y', labelcolor=color_paper)
ax2.tick_params(axis='y', labelcolor=color_tagged)

# Set x-axis with evenly spaced ticks
ax1.set_xticks(x_pos)
ax1.set_xticklabels(stride_labels)

# Set y-axis limits with some padding
ax1.set_ylim(91, 95)
ax2.set_ylim(76, 82)

# Add grid
ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

# Combined legend
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower center')

# Highlight best values
for i, (x, y) in enumerate(zip(x_pos, paper_recall)):
    if y == max(paper_recall):
        ax1.plot(x, y, marker='o', markersize=14, color=color_paper,
                 markerfacecolor='none', markeredgewidth=2.5, zorder=3)

for i, (x, y) in enumerate(zip(x_pos, tagged_recall)):
    if y == max(tagged_recall):
        ax2.plot(x, y, marker='s', markersize=14, color=color_tagged,
                 markerfacecolor='none', markeredgewidth=2.5, zorder=3)

# Save figure
plt.savefig('/home/siqiouya/code/infinisst-omni/plots-rag/figures/stride.pdf',
            dpi=300, bbox_inches='tight')
print("Figure saved as 'stride.pdf'")

plt.show()
