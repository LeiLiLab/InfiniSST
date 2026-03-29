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
        'term_acc': 79.41,  # Single value for horizontal line
        'bleu': 49.66,
        'stremlaal': None  # No latency for offline model
    },
    'InfiniSST': {
        'stremlaal': [1142, 1789, 2216, 2583],  # Model-specific latency values
        'term_acc': [73.53, 72.99, 75.40, 77.54],  # 4 points
        'bleu': [40.86, 45.66, 47.50, 47.73]
    },
    'RASST': {
        'stremlaal': [1219, 1792, 2280, 2703],  # Model-specific latency values
        'term_acc': [80.75, 89.30, 85.56, 89.30],  # 4 points
        'bleu': [43.30, 47.50, 49.05, 49.07]
    }
}

# En-De data
data_en_de = {
    'Offline ST': {
        'term_acc': 72.61,
        'bleu': 35.76,
        'stremlaal': None
    },
    'InfiniSST': {
        'stremlaal': [1132, 1829, 2373, 2826],
        'term_acc': [57.32, 57.01, 68.15, 65.61],
        'bleu': [27.29, 30.89, 31.24, 31.43]
    },
    'RASST': {
        'stremlaal': [1108, 1704, 2225, 2716],
        'term_acc': [70.38, 79.94, 75.16, 78.34],
        'bleu': [28.29, 31.84, 31.49, 31.46]
    }
}

# En-Ja data
data_en_ja = {
    'Offline ST': {
        'term_acc': 76.23,
        'bleu': 32.93,
        'stremlaal': None
    },
    'InfiniSST': {
        'stremlaal': [1554, 2287, 2693, 3330],
        'term_acc': [66.79, 64.91, 70.57, 74.72],
        'bleu': [21.35, 28.04, 28.94, 30.58]
    },
    'RASST': {
        'stremlaal': [1371, 2045, 2670, 3050],
        'term_acc': [72.088, 76.60, 80.75, 82.26],
        'bleu': [19.10, 27.38, 29.31, 30.85]
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
            ax.set_xlim(1000, 3000)
            ax.set_xticks([1000, 1500, 2000, 2500, 3000])
        else:  # En-Ja
            ax.set_xlim(1000, 3500)
            ax.set_xticks([1000, 1500, 2000, 2500, 3000, 3500])
        
        # Set y-axis limits based on metric and language pair
        if metric == 'term_acc':
            if lang_pair == 'En-Zh':
                ax.set_ylim(70, 92)
            elif lang_pair == 'En-De':
                ax.set_ylim(55, 82)
            else:  # En-Ja
                ax.set_ylim(63, 84)
        else:  # bleu
            if lang_pair == 'En-Zh':
                ax.set_ylim(39, 51)
            elif lang_pair == 'En-De':
                ax.set_ylim(26, 37)
            else:  # En-Ja
                ax.set_ylim(18, 34)
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

# Add a single legend at the bottom center of the figure
fig.legend(handles, labels, loc='lower center', ncol=3, framealpha=0.9, 
           bbox_to_anchor=(0.5, -0.05), fontsize=14)

# # Add overall title
# fig.suptitle('Speech Translation Performance vs. Latency', 
#              fontsize=14, fontweight='bold', y=0.98)

# Save figure
plt.savefig('/home/siqiouya/code/infinisst-omni/plots-rag/figures/main_result_paper.pdf', dpi=300, bbox_inches='tight')
print("Figure saved as 'main_result_paper.pdf'")

# Display
plt.show()

