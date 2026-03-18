import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

prefix = Path("results/new")
paths = {
    "Guided": "gpt_5_4_guided",
    "Guided with segmentation": "gpt_5_4_guided_segm",
    "Zero-shot": "gpt_5_4_zeroshot",
    "Zero-shot segmentation": "gpt_5_4_zeroshot_segm",
}

labels = ["Missing teeth", "Endodontic treatment", "Crown lesions", "Mesial inclination", "Implant"]

colors = {
    "Guided": "#4C9F70",
    "Guided with segmentation": "#8FD0A6",
    "Zero-shot": "#4C72B0",
    "Zero-shot segmentation": "#9FBFE7",
}

results = {"Precision": {}, "Recall": {}}

for name, folder in paths.items():
    data = json.load(open(prefix / folder / "stats.json"))

    results["Precision"][name] = [data[l]["precision"] for l in labels]
    results["Recall"][name] = [data[l]["recall"] for l in labels]

for k in results:
    print(k)
    for u, v in results[k].items():
        print(u, " " * (30 - len(u)), v)
    print()
    
# cleaner style
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False
})

x = np.arange(len(labels))
width = 0.2

for metric in results:
    fig, ax = plt.subplots(figsize=(10,5))

    ratio = -1.5
    for label in paths:
        ax.bar(x - ratio*width, results[metric][label], width, label=label, color = colors[label], edgecolor = "none")
        ratio += 1
        
    ax.set_ylabel(metric)
    # fig.suptitle(f"Report comparison — {metric}", y=0.98)
    fig.suptitle(f"Report comparison — {metric}", y=0.98, fontsize=16, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")

    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    # ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2)
    ax.legend(frameon=True, facecolor="white", framealpha=0.8, loc="upper right")

    plt.tight_layout()

    plt.savefig(f"{metric.lower()}_comparison.png", dpi=300, bbox_inches="tight")  # save PNG
    plt.show()