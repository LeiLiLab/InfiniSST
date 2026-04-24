import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['mathtext.fontset'] = 'stix'

data_all = {
    'en-de': {
        'Baseline-Fixed': {
            'StreamLAAL': [1.70],  # TODO: fill in
            'BLEU': [15.74],
        },
        'Baseline-VAD': {
            'StreamLAAL': [1.99],
            'BLEU': [17.81],
        },
        'NAIST': {
            'StreamLAAL': [1.82],
            'BLEU': [20.85],
        },
        'OSU': {
            'StreamLAAL': [1.73],
            'BLEU': [22.04],
        },
        'CMU': {
            'StreamLAAL': [1.47],
            'BLEU': [22.63],
        },
    },
    'en-zh': {
        'Baseline-Fixed': {
            'StreamLAAL': [3.76],
            'BLEU': [20.42],
        },
        'Baseline-VAD': {
            'StreamLAAL': [1.96],
            'BLEU': [22.63],
        },
        'NAIST': {
            'StreamLAAL': [2.28],  # TODO: fill in
            'BLEU': [37.82],
        },
        'OSU': {
            'StreamLAAL': [2.20],
            'BLEU': [34.06],
        },
        'CMU': {
            'StreamLAAL': [2.15],
            'BLEU': [43.26],
        },
    },
}

langs = ['en-de', 'en-zh']
titles = {'en-de': 'En-De', 'en-zh': 'En-Zh'}

model_styles = {
    'Baseline-Fixed': {
        'marker': 'D',
        'linestyle': '--',
        'color': '#9B59B6',  # Purple
        'markersize': 7,
    },
    'Baseline-VAD': {
        'marker': 'v',
        'linestyle': '--',
        'color': '#FAC05E',  # Yellow/Orange
        'markersize': 8,
    },
    'NAIST': {
        'marker': '^',
        'linestyle': '-',
        'color': '#59CD90',  # Green
        'markersize': 9,
    },
    'OSU': {
        'marker': 's',
        'linestyle': '-',
        'color': '#3FA7D6',  # Blue
        'markersize': 7,
    },
    'CMU': {
        'marker': '*',
        'linestyle': '-',
        'color': '#EE6352',  # Red
        'markersize': 10,
    },
}

fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

marker_scale = 1.5

for idx, lang in enumerate(langs):
    ax = axes[idx]
    data = data_all[lang]

    for system in ['Baseline-Fixed', 'Baseline-VAD', 'NAIST', 'OSU', 'CMU']:
        if len(data[system]['StreamLAAL']) == 0:
            continue
        x = [val for val in data[system]['StreamLAAL']]
        y = data[system]['BLEU']
        style = model_styles[system]
        ax.plot(x, y,
                marker=style['marker'],
                linestyle='none',
                color=style['color'],
                label=system,
                markersize=style['markersize'] * marker_scale)

    # Tight axis limits with small padding around data range
    all_x = [v for s in data.values() for v in s['StreamLAAL']]
    all_y = [v for s in data.values() for v in s['BLEU']]
    x_pad = (max(all_x) - min(all_x)) * 0.25 + 0.1
    y_pad = (max(all_y) - min(all_y)) * 0.15 + 0.5
    ax.set_xlim(min(all_x) - x_pad, max(all_x) + x_pad)
    ax.set_ylim(min(all_y) - y_pad, max(all_y) + y_pad)

    ax.set_xlabel('StreamLAAL (s)', fontsize=14)
    if idx == 0:
        ax.set_ylabel('BLEU', fontsize=14)
    ax.set_title(titles[lang], fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.3)

# Shared legend at the bottom
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.02),
           ncol=5, fontsize=12)

plt.subplots_adjust(bottom=0.2)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig('figures/iwslt2025.pdf', bbox_inches='tight', dpi=300)
plt.show()
