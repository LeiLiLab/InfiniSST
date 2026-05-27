import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({"font.size": 13})

# ====== Configuration ======
OUTPUT_DIR = "/mnt/gemini/data2/jiaxuanluo/density_eval_maxsim_fixed/zh"
PLOT_BLEU_PATH = f"{OUTPUT_DIR}/comparison_bleu_vs_latency.png"
PLOT_TERM_ACC_PATH = f"{OUTPUT_DIR}/comparison_term_acc_vs_latency.png"
TABLE_PATH = f"{OUTPUT_DIR}/comparison_table.tsv"
# ====== End Configuration ======

METHODS = {
    "MaxSim (extracted)": {
        "lm":             [1,       2,       3,       4],
        "bleu":           [43.18,   46.91,   48.53,   48.84],
        "streamlaal_ca":  [1657,    2204,    2754,    3348],
        "term_acc":       [0.7781,  0.8583,  0.8690,  0.8957],
    },
    "MaxSim (gs1k)": {
        "lm":             [1,       2,       3,       4],
        "bleu":           [42.31,   47.24,   48.33,   48.91],
        "streamlaal_ca":  [1579,    2201,    2800,    3359],
        "term_acc":       [0.7914,  0.8717,  0.8717,  0.8957],
    },
    "MaxSim (gs10k)": {
        "lm":             [1,       2,       3,       4],
        "bleu":           [43.10,   47.27,   48.07,   48.69],
        "streamlaal_ca":  [1626,    2217,    2790,    3407],
        "term_acc":       [0.8316,  0.8717,  0.8583,  0.8850],
    },
    "MaxSim (old_slm)": {
        "lm":             [1,       2,       3,       4],
        "bleu":           [42.96,   47.69,   48.27,   50.10],
        "streamlaal_ca":  [1631,    2259,    2816,    3420],
        "term_acc":       [0.8235,  0.8717,  0.8717,  0.9091],
    },
    "Old SW RASST": {
        "lm":             [1,       2,       3,       4],
        "bleu":           [43.79,   48.59,   49.15,   49.45],
        "streamlaal_ca":  [1999,    2609,    3380,    4263],
        "term_acc":       [0.8262,  0.8850,  0.8717,  0.9118],
    },
    "Baseline InfiniSST": {
        "lm":             [1,       2,       3,       4],
        "bleu":           [40.86,   45.66,   47.50,   47.73],
        "streamlaal_ca":  [1576,    2289,    2886,    3304],
        "term_acc":       [0.7353,  0.7299,  0.7540,  0.7754],
    },
}

MARKERS = ["o", "s", "D", "^", "v", "X"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

with open(TABLE_PATH, "w") as f:
    header = "Method\tlm\tBLEU\tStreamLAAL_CA\tTerm_Acc"
    f.write(header + "\n")
    print(header)
    for name, d in METHODS.items():
        for i in range(len(d["lm"])):
            row = f"{name}\t{d['lm'][i]}\t{d['bleu'][i]:.2f}\t{d['streamlaal_ca'][i]}\t{d['term_acc'][i]:.4f}"
            f.write(row + "\n")
            print(row)

print(f"\nTable saved to {TABLE_PATH}")

fig, ax = plt.subplots(figsize=(10, 6))
for idx, (name, d) in enumerate(METHODS.items()):
    ax.plot(
        d["streamlaal_ca"], d["bleu"],
        marker=MARKERS[idx], color=COLORS[idx],
        label=name, linewidth=2, markersize=8,
    )
    for i, lm in enumerate(d["lm"]):
        ax.annotate(
            f"lm{lm}", (d["streamlaal_ca"][i], d["bleu"][i]),
            textcoords="offset points", xytext=(0, 8),
            fontsize=8, ha="center", color=COLORS[idx],
        )

ax.set_xlabel("StreamLAAL_CA (ms)")
ax.set_ylabel("BLEU")
ax.set_title("BLEU vs Latency (StreamLAAL_CA)")
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(PLOT_BLEU_PATH, dpi=150)
print(f"BLEU plot saved to {PLOT_BLEU_PATH}")

fig2, ax2 = plt.subplots(figsize=(10, 6))
for idx, (name, d) in enumerate(METHODS.items()):
    ax2.plot(
        d["streamlaal_ca"], d["term_acc"],
        marker=MARKERS[idx], color=COLORS[idx],
        label=name, linewidth=2, markersize=8,
    )
    for i, lm in enumerate(d["lm"]):
        ax2.annotate(
            f"lm{lm}", (d["streamlaal_ca"][i], d["term_acc"][i]),
            textcoords="offset points", xytext=(0, 8),
            fontsize=8, ha="center", color=COLORS[idx],
        )

ax2.set_xlabel("StreamLAAL_CA (ms)")
ax2.set_ylabel("Term Accuracy")
ax2.set_title("Term Accuracy vs Latency (StreamLAAL_CA)")
ax2.legend(loc="lower right", fontsize=10)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(PLOT_TERM_ACC_PATH, dpi=150)
print(f"Term Acc plot saved to {PLOT_TERM_ACC_PATH}")
