# Create the grouped bar plot as requested
import matplotlib.pyplot as plt
import pandas as pd

# ----- Data (you can edit this block) -----
data = [
    {"architecture": "ResNet-18",  "method": "Delete",            "forget_class": 8, "original": 97.3, "unlearned": 0.0, "revival": 90.02},
    {"architecture": "VGGNet-16 ", "method": "Random Label",      "forget_class": 8, "original": 91.0, "unlearned": 3.5, "revival": 98.0},
    {"architecture": "Swin-t",     "method": "Bad Teacher",       "forget_class": 8, "original": 94.3, "unlearned": 1.4, "revival": 93.3},
    {"architecture": "ViT-B-16",   "method": "Negative Gradient", "forget_class": 8, "original": 98.7, "unlearned": 0.4, "revival": 99.5},
]
df = pd.DataFrame(data)

# ----- Plot config -----
metrics = ["original", "unlearned", "revival"]  # order of the 3 bars
bar_width = 0.22
group_gap = 0.5  # visual gap between architectures

# Compute x positions
x_positions = []
x_arch_ticks = []
x_method_ticks = []
x = 0.0

for arch, sub in df.groupby("architecture", sort=False):
    start_x = x
    for _, row in sub.iterrows():
        center = x + bar_width
        x_positions.append(center)
        x_method_ticks.append((center, row["method"]))
        x += (bar_width * len(metrics)) + 0.05
    end_x = x - 0.05
    x_arch_ticks.append(((start_x + end_x) / 2.0, arch))
    x += group_gap

centers = [pos for pos in x_positions]
offsets = {
    "original": -bar_width,
    "unlearned": 0.0,
    "revival": bar_width,
}

fig, ax = plt.subplots(figsize=(11, 4))

# Choose colors (only unlearned is forced to red)
colors = {"original": "blue", "unlearned": "red", "revival": "green"}

# Draw bars
for metric in metrics:
    xs = [c + offsets[metric] for c in centers]
    ys = [df.loc[i, metric] for i in range(len(df))]
    bars = ax.bar(xs, ys, width=bar_width, label=metric.capitalize(),
                  color=colors[metric])  # <-- color set here
    for rect, val in zip(bars, ys):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.6,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

# Aesthetics
ax.set_ylabel("Forget Accuracy (%)")
ax.set_ylim(0, max(df[metrics].max()) + 8)
#ax.set_title("Per-architecture unlearning results: original vs unlearned vs revival")

ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
ax.set_axisbelow(True)

# X ticks: line 1 = Architecture, line 2 = Method
method_positions, method_labels = zip(*x_method_ticks)
arch_labels_in_order = [a.strip() for a in df["architecture"].tolist()]  # clean spaces

combo_labels = [f"{a}\n{m}" for a, m in zip(arch_labels_in_order, method_labels)]
ax.set_xticks(method_positions)
ax.set_xticklabels(combo_labels, ha="center", rotation=0, linespacing=1.2)
ax.tick_params(axis='x', pad=6)  # a bit more space from the axis



# Legend OUTSIDE (right side, boxed)
legend = ax.legend(
    title="Metric",
    ncol=1,
    frameon=True,          # draw box
    fancybox=True,         # rounded corners
    framealpha=0.95,       # slightly transparent
    edgecolor="0.4",
    bbox_to_anchor=(1.04, 1.0),  # outside, right of axes
    loc="upper left",
    borderaxespad=0.0
)
legend.get_frame().set_linewidth(0.8)

# Leave room on the right so it doesn't overlap/clipped
plt.tight_layout(rect=(0, 0, 0.86, 1))   # shrink plot area to make space for the legend
plt.savefig("motivation.png", dpi=300, bbox_inches="tight", pad_inches=0.2)


plt.tight_layout()
plt.savefig("unlearning_grouped_bars.png", dpi=300, bbox_inches="tight")
