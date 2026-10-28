"""Condensed significance and ablation tables used in the paper."""
import json
R="/home/claude/results/"; T="/home/claude/paper/tables/"
J=lambda n: json.load(open(R+n+".json"))
E2,E3=J("e2_cv"),J("e3_ablation")
SC={"nsl_all":"NSL-KDD (all)","nsl_dos":"NSL-KDD (DoS)","unsw_all":"UNSW-NB15 (all)","unsw_dos":"UNSW-NB15 (DoS)"}
p=lambda x:"$<$0.001" if x<1e-3 else f"{x:.3f}"
# significance table
rows=["LR","DT","RF","ET","XGB"]
s="\\begin{table*}[t]\n\\caption{EMI-RF against selected baselines over 10 cross-validation folds. $\\Delta$ is the mean F1 of EMI-RF minus the baseline (negative: baseline better); $p_{NB}$ is the Nadeau--Bengio corrected $t$-test and $p_W$ the Wilcoxon test, both Holm-adjusted.}\\label{tab:stats}\n\\centering\\scriptsize\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{l"+"rrr"*4+"}\n\\toprule\n"
s+="& "+" & ".join(f"\\multicolumn{{3}}{{c}}{{{SC[k]}}}" for k in SC)+"\\\\\n"+"".join(f"\\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(4))+"\nBaseline & "+" & ".join(["$\\Delta$ & $p_{NB}$ & $p_W$"]*4)+"\\\\\n\\midrule\n"
for m in rows:
    r=[]
    for k in SC:
        t=E2[k]["tests"][m]; r+=[f"{t['mean_diff']:+.4f}",p(t["p_nb_holm"]),p(t["p_w_holm"])]
    s+=m+" & "+" & ".join(r)+"\\\\\n"
s+="\\bottomrule\n\\end{tabular}}\n\\end{table*}\n"
open(T+"tab_stats.tex","w").write(s)
# ablation table
VAR=list(E3["nsl_all"]["official"].keys())
s="\\begin{table*}[t]\n\\caption{Ablation on the official test splits: F1 (mean $\\pm$ standard deviation over 3 seeds) when parts of EMI are switched on or off. All variants use Random Forest. ``Random-k'' and ``ANOVA-k'' replace the MI ranking by random choice or by the ANOVA F-statistic.}\\label{tab:ablation}\n\\centering\\scriptsize\n\\begin{tabular}{l"+"c"*4+"}\n\\toprule\nVariant & "+" & ".join(SC.values())+"\\\\\n\\midrule\n"
for v in VAR:
    s+=v.replace("&","\\&").replace("+","{+}")+" & "+" & ".join(f"{E3[k]['official'][v]['f1'][0]:.4f}$\\pm${E3[k]['official'][v]['f1'][1]:.4f}" for k in SC)+"\\\\\n"
s+="\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
open(T+"tab_ablation.tex","w").write(s)
# cross-validation F1 table
import numpy as np
MODELS=["LR","NB","KNN","DT","SVM","MLP","RF","ET","XGB","EMI-RF","EMI-XGB"]
s="\\begin{table}[t]\n\\caption{Mean F1 over the 10 cross-validation folds (20{,}000-record subsample of each training set). Best value per column in bold.}\\label{tab:cv}\n\\centering\\scriptsize\n\\begin{tabular}{lcccc}\n\\toprule\nModel & "+" & ".join(SC.values())+"\\\\\n\\midrule\n"
vals={k:{m:float(np.mean(E2[k]["folds"][m])) for m in MODELS} for k in SC}
for m in MODELS:
    row=[]
    for k in SC:
        v=f"{vals[k][m]:.4f}"; row.append("\\textbf{"+v+"}" if vals[k][m]==max(vals[k].values()) else v)
    s+=m+" & "+" & ".join(row)+"\\\\\n"
s+="\\bottomrule\n\\end{tabular}\n\\end{table}\n"
open(T+"tab_cv.tex","w").write(s)
