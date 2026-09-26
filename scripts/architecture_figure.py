import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
BLUE, ORANGE, AQUA, YELLOW, PINK, VIOLET, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#898781"
fig, ax = plt.subplots(figsize=(15, 8.2)); fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
ax.set_xlim(0, 15); ax.set_ylim(-0.3, 8.2); ax.axis("off")
def box(x, y, w, h, text, color, fs=9.5, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=color, ec="none"))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold", linespacing=1.35)
def arrow(x1, y1, x2, y2, label=None, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color="#898781", lw=1.4,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if label: ax.text((x1+x2)/2, (y1+y2)/2 + 0.18, label, ha="center", fontsize=8, color="#52514e")
# column 1: data
box(0.3, 5.9, 2.3, 1.2, "OpenML 1597\n284,807 transactions\n492 frauds (0.17%)", GREY)
box(0.3, 3.9, 2.3, 1.2, "Time-based split\n70 / 10 / 20\nno shuffling", GREY)
arrow(1.45, 5.9, 1.45, 5.1)
# column 2: modelling
box(3.4, 6.3, 2.9, 1.5, "Supervised\nLR · RF · XGBoost · LightGBM\n× none / weights / SMOTE", BLUE)
box(3.4, 4.3, 2.9, 1.5, "Unsupervised (legit rows only)\nIsolation Forest\nautoencoder", AQUA)
box(3.4, 2.3, 2.9, 1.5, "Differential privacy\nDP-SGD MLP (Opacus)\nε = 0.5 · 1 · 3 · 8 · ∞", VIOLET)
box(3.4, 0.7, 2.9, 1.2, "K-means on AE embedding\nfraud-dense segments", AQUA)
for y in (7.05, 5.05, 3.05): arrow(2.6, 4.5, 3.4, y, rad=0.0)
arrow(4.85, 4.3, 4.85, 1.9)
# column 3: selection + evaluation
box(7.1, 5.5, 3.0, 2.3, "Select on validation PR-AUC\n\nisotonic calibration\ncost-based threshold\nrecall @ 0.1% / 1% FPR\nSHAP · 5-seed stability", ORANGE)
arrow(6.3, 7.05, 7.1, 6.9)
box(7.1, 2.8, 3.0, 1.6, "Export\nONNX fp32 · int8\nCore ML (CPU / GPU)\nlatency vs 10 ms budget", YELLOW, tc="#0b0b0b")
arrow(8.6, 5.5, 8.6, 4.4)
box(7.1, 0.5, 3.0, 1.5, "Drift monitor\nPSI on 29 features\nand on the score", PINK, tc="#0b0b0b")
arrow(8.6, 2.8, 8.6, 2.0)
# column 4: streaming
ax.add_patch(FancyBboxPatch((10.9, 0.5), 3.9, 6.0, boxstyle="round,pad=0.02,rounding_size=0.2", fc="#f0efec", ec="#c3c2b7", lw=1))
ax.text(12.85, 6.15, "Docker Compose · real-time scoring", ha="center", fontsize=10, color="#52514e", fontweight="bold")
box(11.2, 4.6, 3.3, 1.1, "producer.py\nreplays the test window", GREY)
box(11.2, 3.0, 3.3, 1.1, "Kafka (KRaft)\ntransactions · alerts", "#231F20")
box(11.2, 1.0, 3.3, 1.5, "Spark Structured Streaming\npandas UDF over the\nbroadcast ONNX model", ORANGE)
arrow(12.85, 4.6, 12.85, 4.1, "transactions"); arrow(12.85, 3.0, 12.85, 2.5); arrow(13.8, 2.5, 13.8, 3.0, rad=0.0)
ax.text(14.3, 2.75, "alerts", fontsize=8, color="#52514e")
arrow(10.1, 3.6, 11.2, 1.75, rad=0.2)
ax.text(10.15, 2.55, "best_tree.onnx", fontsize=8, color="#52514e", rotation=-42)
ax.text(0.3, 7.85, "Real-Time Fraud Detection: architecture", fontsize=14, fontweight="bold", color="#0b0b0b")
ax.text(0.3, 0.05, "CI: pytest + a 5% time-prefix run of every stage on each push · every number in the README is written to results/ by these stages", fontsize=8.5, color="#52514e")
fig.savefig("results/figures/architecture.png", dpi=150, bbox_inches="tight"); print("wrote architecture.png")
