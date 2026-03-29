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

# Data (A values only)
methods = ['Random', 'LLM', 'BGE-M3']
bleu = [28.2, 47.2, 47.5]
term_acc = [59.4, 82.3, 89.3]

x = np.arange(len(methods))
width = 0.35

fig, ax1 = plt.subplots(1, 1, figsize=(6, 4.5))
ax2 = ax1.twinx()

# Remove top, left, right spines
for spine in ['top', 'left', 'right']:
    ax1.spines[spine].set_visible(False)
    ax2.spines[spine].set_visible(False)

# Colors
color_bleu = '#1f77b4'
color_term = '#ff7f0e'

bars1 = ax1.bar(x - width/2, bleu, width, color=color_bleu, label='BLEU', zorder=2)
bars2 = ax2.bar(x + width/2, term_acc, width, color=color_term, label='Term. Acc.', zorder=2)

# Labels
ax1.set_xlabel('Negative Synth Strategy', fontsize=18)

# Hide y-axis ticks and labels
ax1.tick_params(axis='y', left=False, labelleft=False)
ax2.tick_params(axis='y', right=False, labelright=False)

ax1.set_xticks(x)
ax1.set_xticklabels(methods)

# Y-axis limits
ax1.set_ylim(0, 60)
ax2.set_ylim(0, 100)

# Combined legend (outside right)
lines = [bars1, bars2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=13)

# Value labels on bars
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=13, color=color_bleu)
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=13, color=color_term)

plt.savefig('/home/siqiouya/code/infinisst-omni/plots-rag/figures/neg_synth.pdf',
            dpi=300, bbox_inches='tight')
print("Figure saved as 'neg_synth.pdf'")

plt.show()
