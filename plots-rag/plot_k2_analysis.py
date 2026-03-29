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

# Define data (from the table provided)
# lm, k2, BLEU, StreamLAAL, StreamLAAL_CA, Term Acc
data = [
    [1, 5, 42.45, 1212, 2192, 0.8235],
    [1, 10, 43.30, 1219, 1724, 0.8075],
    [1, 15, 44.44, 1269, 2735, 0.8476],
    [1, 20, 43.67, 1296, 2636, 0.8182],
    [2, 5, 47.11, 1818, 2809, 0.8690],
    [2, 10, 47.50, 1792, 2432, 0.8930],
    [2, 15, 47.43, 1842, 2759, 0.8610],
    [2, 20, 48.34, 1772, 3158, 0.8824],
    [3, 5, 48.89, 2317, 3811, 0.8797],
    [3, 10, 49.05, 2280, 3052, 0.8556],
    [3, 15, 48.78, 2294, 4090, 0.8743],
    [3, 20, 48.56, 2305, 3779, 0.8529],
    [4, 5, 49.46, 2700, 4272, 0.9037],
    [4, 10, 49.07, 2703, 3676, 0.8930],
    [4, 15, 50.04, 2698, 4321, 0.8717],
    [4, 20, 49.25, 2605, 3632, 0.8904],
]

# Organize data by k2 value
k2_values = [5, 10, 15, 20]
data_by_k2 = {k2: {'stremlaal': [], 'bleu': [], 'term_acc': []} for k2 in k2_values}

for row in data:
    lm, k2, bleu, stremlaal, stremlaal_ca, term_acc = row
    data_by_k2[k2]['stremlaal'].append(stremlaal)
    data_by_k2[k2]['bleu'].append(bleu)
    data_by_k2[k2]['term_acc'].append(term_acc * 100)  # Convert to percentage

# Create figure with 2 subplots (1 row, 2 columns) - compact size
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.subplots_adjust(wspace=0.25, bottom=0.2)

# Define colors and markers for different k2 values
colors = {
    5: '#e41a1c',   # blue
    10: '#377eb8',  # orange
    15: '#4daf4a',  # green
    20: '#984ea3'   # red
}

markers = {
    5: 'o',   # circle
    10: 's',  # square
    15: '^',  # triangle up
    20: '*'   # star
}

metrics = ['bleu', 'term_acc']
metric_labels = ['BLEU Score', 'Terminology Accuracy (%)']

# Plot each subplot
handles, labels = [], []
for idx, (metric, ylabel) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx]
    
    # Plot each k2 curve
    for k2 in k2_values:
        x = data_by_k2[k2]['stremlaal']
        y = data_by_k2[k2][metric]
        # Use larger marker size for star
        ms = 12 if k2 == 20 else 8
        line, = ax.plot(x, y, 
                color=colors[k2], marker=markers[k2], 
                linewidth=2, markersize=ms, label=f'K2={k2}', zorder=2)
        
        # Store handles and labels from first subplot
        if idx == 0:
            handles.append(line)
            labels.append(f'K2={k2}')
    
    # Set labels (no title)
    ax.set_xlabel('StreamLAAL (ms)', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Set axis limits based on metric
    ax.set_xlim(1100, 2800)
    ax.set_xticks([1200, 1500, 1800, 2100, 2400, 2700])
    
    if metric == 'term_acc':
        ax.set_ylim(79, 92)
    else:  # bleu
        ax.set_ylim(41, 51)
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

# Add a single legend at the bottom center of the figure
fig.legend(handles, labels, loc='lower center', ncol=4, framealpha=0.9, 
           bbox_to_anchor=(0.5, -0.02), fontsize=12)

# Save figure
plt.savefig('/home/siqiouya/code/infinisst-omni/plots-rag/figures/k2_analysis.pdf', dpi=300, bbox_inches='tight')
print("Figure saved as 'k2_analysis.pdf'")

# Display
plt.show()
