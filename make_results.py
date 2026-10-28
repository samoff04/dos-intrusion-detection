"""Turn results/*.json into LaTeX tables and figures for the paper."""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

R = "/home/claude/results/"; T = "/home/claude/paper/tables/"; F = "/home/claude/paper/figs/"
os.makedirs(T, exist_ok=True); os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .3})
J = lambda n: json.load(open(R + n + ".json"))
E1, ROC, E2, E3, E4, E5, K = J("e1_main"), J("e1_roc"), J("e2_cv"), J("e3_ablation"), J("e4_robust"), J("e5_entropy"), J("k_select")
MODELS = ["LR", "NB", "KNN", "DT", "SVM", "MLP", "RF", "ET", "XGB", "EMI-RF", "EMI-XGB"]
SC = {"nsl_all": "NSL-KDD (all attacks)", "nsl_dos": "NSL-KDD (DoS only)", "unsw_all": "UNSW-NB15 (all attacks)", "unsw_dos": "UNSW-NB15 (DoS only)"}
NAME = {"KNN": "KNN$^\\dagger$", "EMI-RF": "\\textbf{EMI-RF}", "EMI-XGB": "\\textbf{EMI-XGB}"}
nm = lambda m: NAME.get(m, m)
PAL = {"base": "#7a8a99", "emi": "#c0392b", "emi2": "#e67e22", "xgb": "#2c6e9b", "rf": "#2a7f62"}


def tex(name, s):
    open(T + name + ".tex", "w").write(s)


def fmt(x, d=4):
    return f"{x:.{d}f}"


def best_mark(vals, i, hi=True, d=4):
    b = max(vals) if hi else min(vals)
    s = fmt(vals[i], d)
    return f"\\textbf{{{s}}}" if abs(vals[i] - b) < 1e-12 else s


# ------------------------------------------------------------------ Table: main results (4 scenarios)
def main_table(scs, name, caption, label):
    cols = ["acc", "prec", "rec", "f1", "auc", "fpr"]
    head = ["Acc", "Prec", "Rec", "F1", "AUC", "FPR"]
    s = "\\begin{table}[t]\n\\caption{%s}\\label{%s}\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n" % (caption, label)
    s += "\\begin{tabular}{l" + "c" * 6 + "}\n\\toprule\n"
    for sc in scs:
        s += "\\multicolumn{7}{l}{\\textit{%s}}\\\\\n\\midrule\nModel & %s \\\\\n\\midrule\n" % (SC[sc], " & ".join(head))
        for m in MODELS:
            r = E1[f"{sc}_{m}"]; row = []
            for c in cols:
                v = [E1[f"{sc}_{mm}"][c] for mm in MODELS]
                row.append(best_mark(v, MODELS.index(m), hi=(c != "fpr")))
            s += f"{nm(m)} & " + " & ".join(row) + " \\\\\n"
        s += "\\midrule\n" if sc != scs[-1] else ""
    s += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    tex(name, s)


main_table(["nsl_all", "unsw_all"], "tab_main_all", "Detection performance on the official test splits (all attack classes versus normal). Best value per column in bold. $^\\dagger$KNN trained on a 30{,}000-record stratified subsample.", "tab:main_all")
main_table(["nsl_dos", "unsw_dos"], "tab_main_dos", "Detection performance on the DoS-only scenarios (DoS family versus normal traffic). Best value per column in bold. $^\\dagger$KNN trained on a 30{,}000-record stratified subsample.", "tab:main_dos")

# ------------------------------------------------------------------ Table: measured cost
s = "\\begin{table}[t]\n\\caption{Measured computational cost on the official training split (2 CPU cores). Train time in seconds, inference in ms per 1{,}000 records, size is the pickled pipeline.}\\label{tab:cost}\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lrrrrrrrr}\n\\toprule\n& \\multicolumn{4}{c}{NSL-KDD (all)} & \\multicolumn{4}{c}{UNSW-NB15 (all)}\\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\nModel & $d$ & Train & Infer & KB & $d$ & Train & Infer & KB\\\\\n\\midrule\n"
for m in MODELS:
    a, b = E1[f"nsl_all_{m}"], E1[f"unsw_all_{m}"]
    s += f"{nm(m)} & {a['n_feat']} & {a['t_train']:.1f} & {a['t_infer_ms_per_1k']:.1f} & {a['model_kb']:.0f} & {b['n_feat']} & {b['t_train']:.1f} & {b['t_infer_ms_per_1k']:.1f} & {b['model_kb']:.0f}\\\\\n"
s += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"; tex("tab_cost", s)

# ------------------------------------------------------------------ Table: statistical tests (CV)
def p_fmt(p):
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


def stats_table(scs, name, label, extra):
    s = "\\begin{table*}[t]\n\\caption{%sPaired comparison of EMI-RF against each baseline over 10 cross-validation folds (5-fold $\\times$ 2 repeats, 20{,}000-record subsample). $\\Delta$ = mean F1 difference (EMI-RF minus baseline); negative means the baseline is better. $p_t$: paired $t$-test, $p_{NB}$: Nadeau--Bengio corrected resampled $t$-test, $p_W$: Wilcoxon signed-rank; all Holm-adjusted within each scenario.}\\label{%s}\n\\centering\\scriptsize\n\\resizebox{\\textwidth}{!}{%%\n\\begin{tabular}{l" % (extra, label) + "rrrr" * len(scs) + "}\n\\toprule\n"
    s += "& " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{SC[k]}}}" for k in scs) + "\\\\\n"
    s += "".join(f"\\cmidrule(lr){{{2+4*i}-{5+4*i}}}" for i in range(len(scs))) + "\n"
    s += "Baseline & " + " & ".join(["$\\Delta$ & $p_t$ & $p_{NB}$ & $p_W$"] * len(scs)) + "\\\\\n\\midrule\n"
    for m in [x for x in MODELS if x != "EMI-RF"]:
        row = []
        for k in scs:
            t = E2[k]["tests"][m]
            row += [f"{t['mean_diff']:+.4f}", p_fmt(t["p_t_holm"]), p_fmt(t["p_nb_holm"]), p_fmt(t["p_w_holm"])]
        s += f"{nm(m)} & " + " & ".join(row) + "\\\\\n"
    pass  # Friedman row omitted
    s += "\\bottomrule\n\\end{tabular}}\n\\end{table*}\n"; tex(name, s)


stats_table(["nsl_all", "nsl_dos"], "tab_stats", "tab:stats", "")
stats_table(["unsw_all", "unsw_dos"], "tab_stats2", "tab:stats2", "(continued) ")

# ------------------------------------------------------------------ Table: ablation (official test, mean +- std, 3 seeds)
VAR = list(E3["nsl_all"]["official"].keys())


def abl_table(scs, name, label, extra):
    s = "\\begin{table*}[t]\n\\caption{%sAblation on the official test splits (mean $\\pm$ std over 3 seeds; F1 and AUC) and in 10-fold paired cross-validation (CV $\\Delta$: full EMI-RF minus the variant, mean F1; $p_W$: uncorrected Wilcoxon test).}\\label{%s}\n\\centering\\scriptsize\n\\resizebox{\\textwidth}{!}{%%\n\\begin{tabular}{l" % (extra, label) + "ccc" * len(scs) + "}\n\\toprule\n"
    s += "& " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{SC[k]}}}" for k in scs) + "\\\\\n" + "".join(f"\\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(len(scs))) + "\n"
    s += "Variant & " + " & ".join(["F1 & AUC & CV $\\Delta$ ($p_W$)"] * len(scs)) + "\\\\\n\\midrule\n"
    for v in VAR:
        row = []
        for k in scs:
            o = E3[k]["official"][v]; t = E3[k]["tests"].get(v)
            cv = "--" if t is None else f"{t['mean_diff']:+.4f} ({p_fmt(t['p_w'])})"
            row += [f"{o['f1'][0]:.4f}$\\pm${o['f1'][1]:.4f}", f"{o['auc'][0]:.4f}", cv]
        s += v.replace("&", "\\&").replace("+", "{+}") + " & " + " & ".join(row) + "\\\\\n"
    s += "\\bottomrule\n\\end{tabular}}\n\\end{table*}\n"; tex(name, s)


abl_table(["nsl_all", "nsl_dos"], "tab_ablation", "tab:ablation", "")
abl_table(["unsw_all", "unsw_dos"], "tab_ablation2", "tab:ablation2", "(continued) ")

# ------------------------------------------------------------------ Table: k selection
s = "\\begin{table}[t]\n\\caption{Validation F1 of EMI-RF as a function of the number of retained features $k$ (25\\% validation split of the training data; 30{,}000 records). The chosen $k$ is the smallest value within 0.5\\% of the best validation F1 (Eq.~\\ref{eq:krule}).}\\label{tab:ksweep}\n\\centering\\scriptsize\n\\begin{tabular}{lrrrrrrrrr}\n\\toprule\nScenario & 5 & 10 & 15 & 20 & 30 & 50 & 80 & all & chosen $k$\\\\\n\\midrule\n"
for k in SC:
    sw = K[k]["sweep"]; s += SC[k] + " & " + " & ".join(f"{sw[str(x)]:.4f}" if str(x) in sw else "--" for x in [5, 10, 15, 20, 30, 50, 80, "all"]) + f" & {K[k]['k']}\\\\\n"
s += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"; tex("tab_ksweep", s)

# ------------------------------------------------------------------ Table: robustness (noise)
s = "\\begin{table}[t]\n\\caption{Test F1 under training-label noise (fraction of training labels flipped; mean of 2 seeds) on the official test splits (all-attack scenario).}\\label{tab:noise}\n\\centering\\scriptsize\n\\begin{tabular}{llrrrr}\n\\toprule\nDataset & Model & 0\\% & 5\\% & 10\\% & 20\\%\\\\\n\\midrule\n"
for ds in ["nsl", "unsw"]:
    for m in ["LR", "DT", "RF", "XGB", "EMI-RF"]:
        s += f"{'NSL-KDD' if ds=='nsl' else 'UNSW-NB15'} & {nm(m)} & " + " & ".join(f"{E4['noise'][f'{ds}|{m}|{n}'][0]:.4f}" for n in [0.0, 0.05, 0.1, 0.2]) + "\\\\\n"
    s += "\\midrule\n" if ds == "nsl" else ""
s += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"; tex("tab_noise", s)

# ================================================================== FIGURES
# ROC
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for a, sc in zip(ax, ["nsl_all", "unsw_all"]):
    for m, c, ls in [("LR", "#999", "--"), ("DT", "#bbb", ":"), ("RF", PAL["rf"], "-"), ("XGB", PAL["xgb"], "-"), ("EMI-RF", PAL["emi"], "-"), ("EMI-XGB", PAL["emi2"], "-")]:
        r = ROC[f"{sc}_{m}"]; a.plot(r["fpr"], r["tpr"], color=c, ls=ls, lw=1.6, label=f"{m} (AUC {E1[f'{sc}_{m}']['auc']:.3f})")
    a.plot([0, 1], [0, 1], color="k", lw=.6); a.set_xlabel("False positive rate"); a.set_ylabel("True positive rate"); a.set_title(SC[sc]); a.legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(F + "fig_roc.pdf"); fig.savefig(F + "fig_roc.png", dpi=200); plt.close()

# F1 bars (4 scenarios)
fig, ax = plt.subplots(1, 4, figsize=(11, 3.2), sharey=False)
for a, sc in zip(ax, SC):
    v = [E1[f"{sc}_{m}"]["f1"] for m in MODELS]; cols = [PAL["emi"] if m.startswith("EMI") else PAL["base"] for m in MODELS]
    a.bar(range(len(MODELS)), v, color=cols); a.set_xticks(range(len(MODELS))); a.set_xticklabels(MODELS, rotation=70, fontsize=7); a.set_title(SC[sc], fontsize=8)
    a.set_ylim(max(0, min(v) - 0.05), min(1.0, max(v) + 0.03)); a.set_ylabel("F1" if sc == "nsl_all" else "")
fig.tight_layout(); fig.savefig(F + "fig_f1_bars.pdf"); fig.savefig(F + "fig_f1_bars.png", dpi=200); plt.close()

# Accuracy-vs-cost trade-off
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for a, sc in zip(ax, ["nsl_all", "unsw_all"]):
    for m in MODELS:
        r = E1[f"{sc}_{m}"]; c = PAL["emi"] if m.startswith("EMI") else PAL["base"]
        a.scatter(r["t_infer_ms_per_1k"], r["f1"], s=25 + 18 * np.log10(r["model_kb"] + 1), color=c, alpha=.85, edgecolor="k", lw=.4); a.annotate(m, (r["t_infer_ms_per_1k"], r["f1"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    a.set_xscale("log"); a.set_ylim(0.7 if "nsl" in sc else 0.82, None); a.set_xlabel("Inference time (ms / 1000 records)"); a.set_ylabel("Test F1"); a.set_title(SC[sc])
fig.tight_layout(); fig.savefig(F + "fig_tradeoff.pdf"); fig.savefig(F + "fig_tradeoff.png", dpi=200); plt.close()

# k sweep
fig, ax = plt.subplots(figsize=(5.2, 3.3))
for k, c in zip(SC, ["#2c6e9b", "#7fb3d5", "#c0392b", "#e6a39b"]):
    sw = K[k]["sweep"]; ks = [5, 10, 15, 20, 30, 50, 80]; ax.plot(ks, [sw[str(x)] for x in ks], marker="o", ms=3.5, color=c, label=SC[k]); ax.axvline(K[k]["k"], color=c, ls=":", lw=.8)
ax.set_xlabel("Retained features k"); ax.set_ylabel("Validation F1"); ax.legend(fontsize=6.5)
fig.tight_layout(); fig.savefig(F + "fig_ksweep.pdf"); fig.savefig(F + "fig_ksweep.png", dpi=200); plt.close()

# Ablation (official F1) bars
fig, ax = plt.subplots(1, 4, figsize=(11, 3.2))
for a, sc in zip(ax, SC):
    v = [E3[sc]["official"][x]["f1"][0] for x in VAR]; e = [E3[sc]["official"][x]["f1"][1] for x in VAR]
    cols = [PAL["emi"] if "EMI" in x else PAL["base"] for x in VAR]; a.bar(range(len(VAR)), v, yerr=e, color=cols, capsize=2)
    a.set_xticks(range(len(VAR))); a.set_xticklabels([x.replace(" (baseline)", "").replace(" (EMI-RF)", "") for x in VAR], rotation=70, fontsize=6.5); a.set_title(SC[sc], fontsize=8)
    a.set_ylim(min(v) - 0.03, max(v) + 0.015); a.set_ylabel("Test F1" if sc == "nsl_all" else "")
fig.tight_layout(); fig.savefig(F + "fig_ablation.pdf"); fig.savefig(F + "fig_ablation.png", dpi=200); plt.close()

# Robustness: noise + fraction
fig, ax = plt.subplots(2, 2, figsize=(9, 5.6))
for j, ds in enumerate(["nsl", "unsw"]):
    for m, c in [("LR", "#999"), ("DT", "#bbb"), ("RF", PAL["rf"]), ("XGB", PAL["xgb"]), ("EMI-RF", PAL["emi"])]:
        ns = [0.0, 0.05, 0.1, 0.2]; ax[0, j].errorbar([100 * x for x in ns], [E4["noise"][f"{ds}|{m}|{n}"][0] for n in ns], yerr=[E4["noise"][f"{ds}|{m}|{n}"][1] for n in ns], marker="o", ms=3.5, color=c, label=m, capsize=2)
        fs = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]; ax[1, j].errorbar([100 * x for x in fs], [E4["frac"][f"{ds}|{m}|{x}"][0] for x in fs], yerr=[E4["frac"][f"{ds}|{m}|{x}"][1] for x in fs], marker="o", ms=3.5, color=c, label=m, capsize=2)
    t = "NSL-KDD" if ds == "nsl" else "UNSW-NB15"
    ax[0, j].set_title(f"{t}: label noise"); ax[0, j].set_xlabel("Flipped training labels (%)"); ax[0, j].set_ylabel("Test F1")
    ax[1, j].set_title(f"{t}: training-set size"); ax[1, j].set_xlabel("Training data used (%)"); ax[1, j].set_xscale("log"); ax[1, j].set_ylabel("Test F1")
ax[0, 0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(F + "fig_robust.pdf"); fig.savefig(F + "fig_robust.png", dpi=200); plt.close()

# Entropy by class + MI ranking
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4))
for a, ds, t in zip(ax, ["nsl", "unsw"], ["NSL-KDD", "UNSW-NB15"]):
    d = E5[f"{ds}_entropy_by_class"]; hs = list(d); x = np.arange(len(hs))
    a.bar(x - .2, [d[h][0] for h in hs], .4, yerr=[d[h][2] for h in hs], color="#2a7f62", label="normal", capsize=1.5)
    a.bar(x + .2, [d[h][1] for h in hs], .4, yerr=[d[h][3] for h in hs], color="#b5483a", label="attack", capsize=1.5)
    a.set_xticks(x); a.set_xticklabels([h[2:] for h in hs], rotation=45, fontsize=7); a.set_title(t + ": mean normalised entropy"); a.set_ylabel("$H/H_{max}$"); a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(F + "fig_entropy_class.pdf"); fig.savefig(F + "fig_entropy_class.png", dpi=200); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
for a, ds, t in zip(ax, ["nsl", "unsw"], ["NSL-KDD", "UNSW-NB15"]):
    top = E5[f"{ds}_mi_top"][:12][::-1]; nmz = [x[0] for x in top]; v = [x[1] for x in top]
    a.barh(range(len(top)), v, color=["#e67e22" if n.startswith("H_") else "#7a8a99" for n in nmz]); a.set_yticks(range(len(top))); a.set_yticklabels([n[:22] for n in nmz], fontsize=7); a.set_xlabel("Mutual information (nats)"); a.set_title(t + ": top-12 by MI (orange = entropy)")
fig.tight_layout(); fig.savefig(F + "fig_mi_rank.pdf"); fig.savefig(F + "fig_mi_rank.png", dpi=200); plt.close()

# CV distribution (box) of F1
fig, ax = plt.subplots(1, 4, figsize=(11, 3.2))
for a, sc in zip(ax, SC):
    d = E2[sc]["folds"]; a.boxplot([d[m] for m in MODELS], tick_labels=MODELS, patch_artist=True, boxprops=dict(facecolor="#dde6ee"), medianprops=dict(color="#c0392b"))
    a.set_xticklabels(MODELS, rotation=70, fontsize=7); a.set_title(SC[sc], fontsize=8); a.set_ylabel("Fold F1" if sc == "nsl_all" else "")
    lo = min(min(d[m]) for m in MODELS if m != "NB"); a.set_ylim(lo - .005, 1.0005)
fig.tight_layout(); fig.savefig(F + "fig_cv_box.pdf"); fig.savefig(F + "fig_cv_box.png", dpi=200); plt.close()

# summary numbers for text
summ = {}
for sc in SC:
    ranks = {}
    arr = np.array([E2[sc]["folds"][m] for m in MODELS]); rk = np.array([stats.rankdata(-arr[:, i]) for i in range(arr.shape[1])]).mean(0)
    summ[sc] = dict(mean_rank={m: float(r) for m, r in zip(MODELS, rk)})
json.dump(summ, open(R + "summary_ranks.json", "w"), indent=1)
print("made tables:", sorted(os.listdir(T)))
