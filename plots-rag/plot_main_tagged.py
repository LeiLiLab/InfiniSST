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
# Data structure: {language_pair: {model: {metric: values, stremlaal: values}}}
# Each model now has its own StreamLAAL values

# En-Zh data
data_en_zh = {
    'Offline ST': {
        'term_acc': 79.12,  # Single value for horizontal line
        'bleu': 49.66,
        'stremlaal': None  # No latency for offline model
    },
    'InfiniSST': {
        'stremlaal': [1181, 1765, 2232, 2616],  # Model-specific latency values
        'term_acc': [74.31, 76.55, 76.75, 77.54],  # 4 points
        'bleu': [40.66, 45.82, 46.71, 47.38]
    },
    'RASST': {
        'stremlaal': [1225, 1781, 2258, 2664],  # Different latency values
        'term_acc': [82.41, 83.93, 85.77, 85.51],  # 4 points, better
        'bleu': [44.22, 48.75, 49.15, 49.68]
    }
}

# En-De data
data_en_de = {
    'Offline ST': {
        'term_acc': 70.57,
        'bleu': 35.76,
        'stremlaal': None
    },
    'InfiniSST': {
        'stremlaal': [1124, 1773, 2383, 2808],
        'term_acc': [64.96, 67.44, 68.64, 68.81],
        'bleu': [27.46, 31.63, 31.70, 32.67]
    },
    'RASST': {
        'stremlaal': [1055, 1698, 2233, 2744],
        'term_acc': [73.78, 80.03, 78.51, 80.75],
        'bleu': [27.43, 32.19, 33.19, 34.60]
    }
}

# En-Ja data
data_en_ja = {
    'Offline ST': {
        'term_acc': 66.24,
        'bleu': 32.9,
        'stremlaal': None
    },
    'InfiniSST': {
        'stremlaal': [1571, 2300, 2707, 3252],
        'term_acc': [63.31, 65.64, 67.24, 67.51],
        'bleu': [22.01, 27.87, 29.30, 30.60]
    },
    'RASST': {
        'stremlaal': [1309, 2092, 2592, 3071],
        'term_acc': [77.05, 79.19, 82.39, 82.66],
        'bleu': [20.19, 28.29, 31.80, 32.53]
    }
}

# Organize data by language pair
data_by_lang = {
    'En-Zh': data_en_zh,
    'En-De': data_en_de,
    'En-Ja': data_en_ja
}

# Create figure with 2 rows and 3 columns
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
fig.subplots_adjust(hspace=0.35, wspace=0.3)

# Define colors and markers
colors = {
    'Offline ST': '#2ca02c',  # green
    'InfiniSST': '#1f77b4',   # blue
    'RASST': '#ff0000'        # lighter red
}

markers = {
    'Offline ST': 'o',
    'InfiniSST': '^',         # triangle
    'RASST': '*'              # star
}

# Language pairs
lang_pairs = ['En-Zh', 'En-De', 'En-Ja']
metrics = ['term_acc', 'bleu']
metric_labels = ['Terminology\nAccuracy (%)', 'BLEU Score']

# Plot each subplot
handles, labels = None, None
for col, lang_pair in enumerate(lang_pairs):
    data = data_by_lang[lang_pair]
    
    for row, (metric, ylabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[row, col]
        
        # Plot Offline ST as horizontal dotted line
        offline_value = data['Offline ST'][metric]
        line1 = ax.axhline(y=offline_value, color=colors['Offline ST'], 
                   linestyle='--', linewidth=2, label='Offline ST', zorder=1)
        
        # Plot InfiniSST curve with its own StreamLAAL values
        infinisst_x = data['InfiniSST']['stremlaal']
        infinisst_y = data['InfiniSST'][metric]
        line2, = ax.plot(infinisst_x, infinisst_y, 
                color=colors['InfiniSST'], marker=markers['InfiniSST'], 
                linewidth=2, markersize=7, label='InfiniSST', zorder=2)
        
        # Plot RASST curve with its own StreamLAAL values
        rasst_x = data['RASST']['stremlaal']
        rasst_y = data['RASST'][metric]
        line3, = ax.plot(rasst_x, rasst_y, 
                color=colors['RASST'], marker=markers['RASST'], 
                linewidth=2, markersize=10, label='RASST', zorder=3)
        
        # Store handles and labels from first subplot
        if handles is None:
            handles = [line1, line2, line3]
            labels = ['Offline ST', 'InfiniSST', 'RASST']
        
        # Set labels and title
        if row == 1:  # Only bottom row
            ax.set_xlabel('StreamLAAL (ms)', fontsize=14)
        if col == 0:  # Only leftmost column
            ax.set_ylabel(ylabel, fontsize=14)
        if row == 0:  # Only top row
            ax.set_title(lang_pair, fontsize=12, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Set x-axis limits and ticks based on language pair
        if lang_pair == 'En-Zh':
            ax.set_xlim(1000, 2900)
            ax.set_xticks([1000, 1500, 2000, 2500])
        elif lang_pair == 'En-De':
            ax.set_xlim(1000, 2900)
            ax.set_xticks([1000, 1500, 2000, 2500])
        else:  # En-Ja
            ax.set_xlim(1000, 3500)
            ax.set_xticks([1000, 1500, 2000, 2500, 3000, 3500])
        
        # Set y-axis limits based on metric and language pair
        if metric == 'term_acc':
            if lang_pair == 'En-Zh':
                ax.set_ylim(72, 88)
            elif lang_pair == 'En-De':
                ax.set_ylim(63, 82)
            else:  # En-Ja
                ax.set_ylim(62, 84)
        else:  # bleu
            if lang_pair == 'En-Zh':
                ax.set_ylim(39, 51)
            elif lang_pair == 'En-De':
                ax.set_ylim(26, 37)
            else:  # En-Ja
                ax.set_ylim(19, 34)
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

# Add a single legend at the bottom center of the figure
fig.legend(handles, labels, loc='lower center', ncol=3, framealpha=0.9, 
           bbox_to_anchor=(0.5, -0.05), fontsize=14)

# # Add overall title
# fig.suptitle('Speech Translation Performance vs. Latency', 
#              fontsize=14, fontweight='bold', y=0.98)

# Save figure
plt.savefig('/home/siqiouya/code/infinisst-omni/plots-rag/figures/main_result_tagged.pdf', dpi=300, bbox_inches='tight')
print("Figure saved as 'main_result_tagged.pdf'")

# Display
plt.show()

