import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 数据
results = {
    "Wiki": {
        "RAIDAR": {"No": 0.924, "Paraphrasing": 0.8724, "Decoherence": 0.923},
        "ImBD": {"No": 0.96685, "Paraphrasing": 0.7117, "Decoherence": 0.880},
        "L2D": {"No": 0.9796, "Paraphrasing": 0.9068, "Decoherence": 0.905},
    },
    "Story": {
        "RAIDAR": {"No": 0.8736, "Paraphrasing": 0.8017, "Decoherence": 0.858},
        "ImBD": {"No": 0.8921, "Paraphrasing": 0.86555, "Decoherence": 0.893},
        "L2D": {"No": 0.9598, "Paraphrasing": 0.8926, "Decoherence": 0.949},
    },
    "News": {
        "RAIDAR": {"No": 0.9018, "Paraphrasing": 0.8304, "Decoherence": 0.914},
        "ImBD": {"No": 0.90425, "Paraphrasing": 0.90035, "Decoherence": 0.844},
        "L2D": {"No": 0.9916, "Paraphrasing": 0.9827, "Decoherence": 0.984},
    },
}

datasets = ["News", "Wiki", "Story"]
methods = ["RAIDAR", "ImBD", "L2D"]
colors = ["#F28E2B", "#76B7B2", "#E15759"]
attacks = ["Paraphrasing", "Decoherence"]

x = np.arange(len(methods))
width = 0.35

fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)

for row, attack in enumerate(attacks):
    for col, dataset in enumerate(datasets):
        ax = axes[row, col]
        for j, method in enumerate(methods):
            no_val = results[dataset][method]["No"]
            atk_val = results[dataset][method][attack]

            # No attack (浅色)
            ax.bar(j - width/2, no_val, width, color=colors[j], alpha=0.4)

            # Attack (深色) — 如果是 L2D，则加黑边框并标注数值
            if method == "L2D":
                bar = ax.bar(j + width/2, atk_val, width,
                             color=colors[j], edgecolor="black", linewidth=1.5)
                ax.text(j + width/2, atk_val + 0.001, f"{atk_val:.3f}",
                        ha="center", va="bottom", fontsize=11, weight="bold")
            else:
                ax.bar(j + width/2, atk_val, width, color=colors[j])

        if row == 0:
            ax.set_title(f"{dataset}", fontsize=15, weight="bold")
        if col == 0:
            ax.set_ylabel(f"AUC ({attack})", fontsize=15)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=13)
        ax.set_ylim(0.70, 1.01)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)

# 图例：颜色 = 方法，深浅 = No vs Attack
handles = [plt.Rectangle((0,0),1,1,color=colors[i]) for i in range(len(methods))]
labels = methods
extra_handles = [
    plt.Rectangle((0,0),1,1,color="gray",alpha=0.4),
    plt.Rectangle((0,0),1,1,color="gray",alpha=1.0)
]
extra_labels = ["No Attack", "With Attack"]

fig.legend(handles + extra_handles, labels + extra_labels,
           loc="lower center", ncol=5, fontsize=15, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0,0.05,1,1])
plt.savefig("exp_adversarial.pdf", format="pdf")
plt.show()
