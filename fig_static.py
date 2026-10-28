"""Static conceptual figures (own drawings): pipeline, taxonomy, entropy illustration."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np, os
O = "/home/claude/paper/figs/"
os.makedirs(O, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
C = {"a": "#E8EEF7", "b": "#DDEBDD", "c": "#FBE9D0", "d": "#EBDDEB", "e": "#F4DADA", "ln": "#33475B"}


def box(ax, x, y, w, h, t, fc, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=C["ln"], lw=1.1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else None, wrap=True)


def arr(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color=C["ln"], lw=1.2))


# ---- Fig: pipeline of the proposed EMI detector
fig, ax = plt.subplots(figsize=(10.2, 3.4)); ax.set_xlim(0, 20.5); ax.set_ylim(0, 6.8); ax.axis("off")
xs = [0.1, 3.5, 6.9, 10.3, 13.7, 17.1]; w = 3.0
labels = [("Network flows\nNSL-KDD,\nUNSW-NB15\n(1 record = 1 flow)", C["a"]),
          ("Entropy features\nH(x) per counter\ngroup (raw values,\nbefore scaling)", C["c"]),
          ("Pre-processing\none-hot (proto,\nservice, state)\nlog1p + z-score", C["b"]),
          ("MI feature\nselection\nrank I(X_j;Y),\nkeep top-k", C["d"]),
          ("Classifier\nRandom Forest /\nXGBoost", C["e"]),
          ("Decision\nnormal vs attack\n+ score s(x)", C["a"])]
for x, (t, c) in zip(xs, labels):
    box(ax, x, 2.4, w, 2.7, t, c, fs=7.8)
for i in range(5):
    arr(ax, xs[i] + w + 0.02, 3.75, xs[i + 1] - 0.02, 3.75)
ax.text(8.45, 6.2, "Training only: fit scaler, MI ranking and k on the training split (no test leakage)", ha="center", fontsize=8.5, style="italic")
ax.text(8.45, 1.3, "Ablation switches: (i) entropy block on/off, (ii) MI block on/off, (iii) selector = MI / ANOVA-F / random,\n(iv) classifier = RF / XGBoost", ha="center", fontsize=8.5)
fig.savefig(O + "fig_pipeline.pdf", bbox_inches="tight"); fig.savefig(O + "fig_pipeline.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- Fig: taxonomy of literature
fig, ax = plt.subplots(figsize=(10.2, 4.6)); ax.set_xlim(0, 20.5); ax.set_ylim(0, 9.4); ax.axis("off")
box(ax, 7.6, 8.2, 5.3, 1.0, "DDoS / intrusion detection\nliterature", C["ln"], fs=9, bold=True); ax.texts[-1].set_color("white")
cols = [("Statistical /\nentropy-based", C["c"], ["Shannon entropy of\nheader fields", "Windowed thresholds", "Entropy + SDN controller"], "[Shannon; Lakhina;\nFeinstein; Nychis]"),
        ("Classical ML\n+ feature selection", C["b"], ["RF, DT, SVM, KNN", "Boosting (XGBoost)", "IG / MI / CFS / wrappers"], "[Breiman; Chen;\nKurniabudi; Zhou]"),
        ("Deep learning", C["d"], ["CNN / RNN / GRU", "Autoencoders", "Transformers"], "[Xu; HAST-IDS;\nLUCID; Kitsune]"),
        ("Evaluation &\nmethodology", C["a"], ["Datasets & flaws", "Cross-dataset tests", "Statistical comparison"], "[Tavallaee; Engelen;\nCantone; Demsar]")]
for i, (t, c, items, refs) in enumerate(cols):
    x = 0.3 + i * 5.1
    box(ax, x, 5.9, 4.6, 1.3, t, c, fs=9, bold=True); arr(ax, 10.25, 8.2, x + 2.3, 7.2)
    for j, it in enumerate(items):
        box(ax, x + 0.2, 4.3 - j * 1.35, 4.2, 1.05, it, "white", fs=8)
    ax.text(x + 2.3, 0.3, refs, ha="center", fontsize=7.5, style="italic")
fig.savefig(O + "fig_taxonomy.pdf", bbox_inches="tight"); fig.savefig(O + "fig_taxonomy.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- Fig: entropy intuition (binary entropy curve + volumetric vs benign profile)
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.2))
p = np.linspace(1e-4, 1 - 1e-4, 400); ax[0].plot(p, -(p * np.log2(p) + (1 - p) * np.log2(1 - p)), color="#1f4e79", lw=2)
ax[0].set_xlabel("share of the first component, p"); ax[0].set_ylabel("H(p) (bits)"); ax[0].set_title("(a) Two-component entropy")
ax[0].grid(alpha=.3)
cats = ["benign\n(spread)", "flood\n(concentrated)"]
for i, (pv, col) in enumerate([([0.3, 0.25, 0.25, 0.2], "#2a7f62"), ([0.94, 0.02, 0.02, 0.02], "#b5483a")]):
    H = -(np.array(pv) * np.log2(pv)).sum() / 2
    ax[1].bar(np.arange(4) + i * 5, pv, color=col)
    ax[1].text(1.5 + i * 5, 1.0, f"H/H_max = {H:.2f}", ha="center", fontsize=9)
ax[1].set_xticks([1.5, 6.5]); ax[1].set_xticklabels(cats); ax[1].set_ylim(0, 1.15); ax[1].set_ylabel("p_i")
ax[1].set_title("(b) Concentrated traffic has low entropy")
fig.tight_layout(); fig.savefig(O + "fig_entropy_concept.pdf"); fig.savefig(O + "fig_entropy_concept.png", dpi=200); plt.close()

# ---- Fig: random forest / MI schematic (math core)
fig, ax = plt.subplots(figsize=(9.6, 2.9)); ax.set_xlim(0, 19.5); ax.set_ylim(0, 5.8); ax.axis("off")
box(ax, 0.2, 2.0, 3.6, 1.9, "Training set\nD = {(x_i, y_i)}", C["a"])
for i, y in enumerate([4.3, 2.6, 0.9]):
    box(ax, 5.3, y, 3.4, 1.2, f"Bootstrap sample {i+1}\n+ random features", C["b"], fs=8); arr(ax, 3.82, 2.95, 5.28, y + 0.6)
    box(ax, 9.6, y, 3.0, 1.2, f"Tree T_{i+1}\n(Gini splits)", C["c"], fs=8); arr(ax, 8.72, y + 0.6, 9.58, y + 0.6)
    arr(ax, 12.62, y + 0.6, 14.0, 2.95)
box(ax, 14.0, 2.0, 5.2, 1.9, "Average of tree votes\ns(x) = (1/B) sum_b 1[T_b(x)=1]", C["e"], fs=8.5)
fig.savefig(O + "fig_rf.pdf", bbox_inches="tight"); fig.savefig(O + "fig_rf.png", dpi=200, bbox_inches="tight"); plt.close()
print("ok")
