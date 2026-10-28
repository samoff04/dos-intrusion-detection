"""Run every experiment; results go to ../results/*.json (cached per experiment)."""
import sys, json, os, pickle, time, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_curve
from sklearn.base import clone
from lib import *

R = "/home/claude/results/"
os.makedirs(R, exist_ok=True)
SCEN = [("nsl", "all"), ("nsl", "dos"), ("unsw", "all"), ("unsw", "dos")]
BASE = ["LR", "NB", "KNN", "DT", "SVM", "MLP", "RF", "ET", "XGB"]
KNN_SUB = 30000


def save(name, obj):
    json.dump(obj, open(R + name + ".json", "w"), indent=1, default=float)


def done(name):
    return os.path.exists(R + name + ".json")


def sub(X, y, n, seed=SEED):
    if len(y) <= n:
        return X.reset_index(drop=True), y
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    return Xs.reset_index(drop=True), ys


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------- k selection
def choose_k(ds, sc):
    """Smallest k whose validation F1 >= 0.995 * best (validation split of TRAIN only)."""
    Xtr, ytr, _, _, g, c = load(ds, sc)
    Xs, ys = sub(Xtr, ytr, 30000)
    Xa, Xv, ya, yv = train_test_split(Xs, ys, test_size=0.25, stratify=ys, random_state=SEED)
    sweep = {}
    for k in [5, 10, 15, 20, 30, 50, 80]:
        p = build("RF", c, g, True, k); p.fit(Xa, ya)
        sweep[k] = f1_score(yv, p.predict(Xv))
    p = build("RF", c, g, True, None); p.fit(Xa, ya)
    sweep["all"] = f1_score(yv, p.predict(Xv))
    best = max(sweep.values())
    k = min(int(kk) for kk, v in sweep.items() if kk != "all" and v >= 0.995 * best)
    return k, sweep


def get_k():
    if not done("k_select"):
        out = {}
        for ds, sc in SCEN:
            k, sw = choose_k(ds, sc); out[f"{ds}_{sc}"] = dict(k=k, sweep=sw); log("k", ds, sc, k, sw)
        save("k_select", out)
    return json.load(open(R + "k_select.json"))


# ------------------------------------------------------------- E1 main table
def e1(K):
    if done("e1_main"):
        return
    out, roc = {}, {}
    for ds, sc in SCEN:
        Xtr, ytr, Xte, yte, g, c = load(ds, sc)
        k = K[f"{ds}_{sc}"]["k"]
        specs = {m: (m, False, None) for m in BASE}
        specs["EMI-RF"] = ("RF", True, k)
        specs["EMI-XGB"] = ("XGB", True, k)
        for name, (cn, ent, kk) in specs.items():
            Xa, ya = (sub(Xtr, ytr, KNN_SUB) if name == "KNN" else (Xtr, ytr))
            p = build(cn, c, g, ent, kk)
            r = fit_eval(p, Xa, ya, Xte, yte)
            nfeat = (p.named_steps["mi"].keep_.size if kk else
                     p.named_steps["pre"].transform(p.named_steps["ent"].transform(Xte.head(2)) if ent else Xte.head(2)).shape[1])
            r["n_feat"] = int(nfeat); r["model_kb"] = len(pickle.dumps(p)) / 1024
            if sc == "all" or name in ("RF", "EMI-RF"):
                fp, tp, _ = roc_curve(yte, r["scores"]); ix = np.linspace(0, len(fp) - 1, 200).astype(int)
                roc[f"{ds}_{sc}_{name}"] = dict(fpr=fp[ix].tolist(), tpr=tp[ix].tolist())
            r.pop("scores"); out[f"{ds}_{sc}_{name}"] = r
            log("E1", ds, sc, name, round(r["f1"], 4), round(r["auc"], 4), r["n_feat"], round(r["t_train"], 1))
    save("e1_main", out); save("e1_roc", roc)


# ----------------------------------------------------------- E2 CV + stats
def cv_scores(ds, sc, variants, n_sub=20000):
    Xtr, ytr, _, _, g, c = load(ds, sc)
    Xs, ys = sub(Xtr, ytr, n_sub)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    res = {v: [] for v in variants}
    for tr, te in cv.split(Xs, ys):
        for v, (cn, ent, kk, sel) in variants.items():
            p = build_variant(cn, c, g, ent, kk, sel)
            p.fit(Xs.iloc[tr], ys[tr]); res[v].append(f1_score(ys[te], p.predict(Xs.iloc[te])))
    return res, len(tr), len(te)


def nb_ttest(d, n_tr, n_te):
    """Nadeau-Bengio corrected resampled paired t-test."""
    n = len(d); v = np.var(d, ddof=1)
    if v == 0:
        return 0.0, 1.0
    t = np.mean(d) / np.sqrt((1 / n + n_te / n_tr) * v)
    return t, 2 * stats.t.sf(abs(t), n - 1)


def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); m = len(p); run = 0
    for i, ix in enumerate(o):
        run = max(run, (m - i) * p[ix]); adj[ix] = min(run, 1)
    return adj


def build_variant(cn, c, g, ent, kk, sel="mi"):
    if sel == "mi":
        return build(cn, c, g, ent, kk)
    steps = make_pre(c, g, ent)
    if sel == "random":
        steps.append(("mi", RandomSelector(kk)))
    elif sel == "anova":
        steps.append(("mi", AnovaSelector(kk)))
    steps.append(("clf", clf_zoo()[cn]))
    return Pipeline(steps)


class RandomSelector(MISelector):
    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        self.keep_ = rng.choice(X.shape[1], min(self.k, X.shape[1]), replace=False); return self


class AnovaSelector(MISelector):
    def fit(self, X, y):
        from sklearn.feature_selection import f_classif
        f = np.nan_to_num(f_classif(X, y)[0]); self.keep_ = np.argsort(-f)[: self.k]; return self


def e2(K):
    if done("e2_cv"):
        return
    out = {}
    for ds, sc in SCEN:
        k = K[f"{ds}_{sc}"]["k"]
        var = {m: (m, False, None, "mi") for m in BASE}
        var["EMI-RF"] = ("RF", True, k, "mi"); var["EMI-XGB"] = ("XGB", True, k, "mi")
        res, ntr, nte = cv_scores(ds, sc, var)
        ref = np.array(res["EMI-RF"]); tests = {}
        names = [m for m in var if m != "EMI-RF"]
        pt, pw, tt = [], [], []
        for m in names:
            d = ref - np.array(res[m])
            t, p_nb = nb_ttest(d, ntr, nte)
            try:
                pwv = stats.wilcoxon(ref, np.array(res[m])).pvalue
            except ValueError:
                pwv = 1.0
            pt.append(stats.ttest_rel(ref, np.array(res[m])).pvalue); pw.append(pwv); tt.append(p_nb)
        ph, pwh, pnh = holm(pt), holm(pw), holm(tt)
        for i, m in enumerate(names):
            tests[m] = dict(mean_diff=float(np.mean(ref - np.array(res[m]))), p_t=pt[i], p_t_holm=ph[i],
                            p_w=pw[i], p_w_holm=pwh[i], p_nb=tt[i], p_nb_holm=pnh[i])
        fr = stats.friedmanchisquare(*[res[m] for m in var])
        out[f"{ds}_{sc}"] = dict(folds=res, tests=tests, friedman=dict(stat=fr.statistic, p=fr.pvalue),
                                 n_train=ntr, n_test=nte)
        log("E2", ds, sc, {m: round(np.mean(v), 4) for m, v in res.items()}, "friedman p", fr.pvalue)
    save("e2_cv", out)


# ------------------------------------------------------------- E3 ablation
def e3(K):
    if done("e3_ablation"):
        return
    out = {}
    for ds, sc in SCEN:
        k = K[f"{ds}_{sc}"]["k"]
        var = {"RF (baseline)": ("RF", False, None, "mi"),
               "+Entropy": ("RF", True, None, "mi"),
               "+MI only": ("RF", False, k, "mi"),
               "+Entropy+MI (EMI-RF)": ("RF", True, k, "mi"),
               "Entropy+Random-k": ("RF", True, k, "random"),
               "Entropy+ANOVA-k": ("RF", True, k, "anova")}
        # (a) CV with paired folds
        res, ntr, nte = cv_scores(ds, sc, var)
        ref = np.array(res["+Entropy+MI (EMI-RF)"]); tst = {}
        for m in var:
            if m == "+Entropy+MI (EMI-RF)":
                continue
            d = ref - np.array(res[m])
            try:
                pw = stats.wilcoxon(ref, np.array(res[m])).pvalue
            except ValueError:
                pw = 1.0
            tst[m] = dict(mean_diff=float(d.mean()), p_w=pw, p_nb=nb_ttest(d, ntr, nte)[1])
        # (b) official test split, 3 seeds
        Xtr, ytr, Xte, yte, g, c = load(ds, sc); official = {}
        for v, (cn, ent, kk, sel) in var.items():
            rs = []
            for sd in range(3):
                p = build_variant(cn, c, g, ent, kk, sel)
                if sel == "random":
                    p.named_steps["mi"].seed = sd
                p.set_params(clf__random_state=sd)
                p.fit(Xtr, ytr); yh = p.predict(Xte); rs.append(metrics(yte, yh, score_of(p, Xte)))
            official[v] = {m: [float(np.mean([r[m] for r in rs])), float(np.std([r[m] for r in rs]))] for m in rs[0]}
        out[f"{ds}_{sc}"] = dict(cv=res, tests=tst, official=official)
        log("E3", ds, sc, {v: round(o["f1"][0], 4) for v, o in official.items()})
    save("e3_ablation", out)


# --------------------------------------------------------- E4 robustness
def e4(K):
    if done("e4_robust"):
        return
    out = {"noise": {}, "frac": {}}
    models = {"LR": ("LR", False, None), "DT": ("DT", False, None), "RF": ("RF", False, None),
              "XGB": ("XGB", False, None)}
    for ds in ["nsl", "unsw"]:
        Xtr, ytr, Xte, yte, g, c = load(ds, "all"); k = K[f"{ds}_all"]["k"]
        mm = dict(models); mm["EMI-RF"] = ("RF", True, k)
        for name, (cn, ent, kk) in mm.items():
            for noise in [0.0, 0.05, 0.10, 0.20]:
                fs = []
                for sd in range(2):
                    rng = np.random.RandomState(sd); yn = ytr.copy()
                    fl = rng.rand(len(yn)) < noise; yn[fl] = 1 - yn[fl]
                    p = build(cn, c, g, ent, kk, seed=sd); p.fit(Xtr, yn); fs.append(f1_score(yte, p.predict(Xte)))
                out["noise"][f"{ds}|{name}|{noise}"] = [float(np.mean(fs)), float(np.std(fs))]
            for fr in [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]:
                fs = []
                for sd in range(3):
                    Xa, ya = sub(Xtr, ytr, max(int(fr * len(ytr)), 200), seed=sd) if fr < 1 else (Xtr, ytr)
                    kk2 = min(kk, 20) if kk else kk
                    p = build(cn, c, g, ent, kk2, seed=sd); p.fit(Xa, ya); fs.append(f1_score(yte, p.predict(Xte)))
                    if fr == 1.0:
                        break
                out["frac"][f"{ds}|{name}|{fr}"] = [float(np.mean(fs)), float(np.std(fs))]
            log("E4", ds, name)
    save("e4_robust", out)


# ------------------------------------------- E5 entropy / MI analysis for figs
def e5(K):
    if done("e5_entropy"):
        return
    out = {}
    for ds, sc in [("nsl", "all"), ("unsw", "all")]:
        Xtr, ytr, _, _, g, c = load(ds, sc)
        Xe = EntropyFeatures(g).transform(Xtr)
        out[f"{ds}_entropy_by_class"] = {h: [float(Xe.loc[ytr == 0, h].mean()), float(Xe.loc[ytr == 1, h].mean()),
                                              float(Xe.loc[ytr == 0, h].std()), float(Xe.loc[ytr == 1, h].std())] for h in g}
        out[f"{ds}_entropy_samples"] = {h: [Xe.loc[ytr == 0, h].sample(2000, random_state=1).tolist(),
                                            Xe.loc[ytr == 1, h].sample(2000, random_state=1).tolist()] for h in g}
        p = build("RF", c, g, True, None); pre = Pipeline(p.steps[:-1])
        Xs, ys = sub(Xtr, ytr, 30000); Z = pre.fit_transform(Xs, ys)
        names = list(pre.named_steps["pre"].ct.get_feature_names_out())
        mi = mutual_info_classif(Z[:8000], ys[:8000], random_state=SEED)
        o = np.argsort(-mi)[:20]
        out[f"{ds}_mi_top"] = [[names[i].split("__")[-1], float(mi[i])] for i in o]
        out[f"{ds}_n_features"] = [len(names) - len(g), len(names)]
    save("e5_entropy", out)


if __name__ == "__main__":
    K = get_k()
    for f in (e1, e2, e3, e4, e5):
        f(K)
    log("ALL DONE")
