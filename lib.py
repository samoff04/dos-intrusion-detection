"""Shared code: dataset loaders, entropy features, MI selector, model zoo, metrics."""
import numpy as np, pandas as pd, time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef, confusion_matrix)
from xgboost import XGBClassifier

D = "/home/claude/data/"
SEED = 42

# ---------------------------------------------------------------- datasets
NSL_COLS = ("duration,protocol_type,service,flag,src_bytes,dst_bytes,land,wrong_fragment,urgent,hot,"
            "num_failed_logins,logged_in,num_compromised,root_shell,su_attempted,num_root,"
            "num_file_creations,num_shells,num_access_files,num_outbound_cmds,is_host_login,"
            "is_guest_login,count,srv_count,serror_rate,srv_serror_rate,rerror_rate,srv_rerror_rate,"
            "same_srv_rate,diff_srv_rate,srv_diff_host_rate,dst_host_count,dst_host_srv_count,"
            "dst_host_same_srv_rate,dst_host_diff_srv_rate,dst_host_same_src_port_rate,"
            "dst_host_srv_diff_host_rate,dst_host_serror_rate,dst_host_srv_serror_rate,"
            "dst_host_rerror_rate,dst_host_srv_rerror_rate,label,difficulty").split(",")
NSL_DOS = {"neptune", "smurf", "back", "teardrop", "pod", "land", "apache2", "udpstorm",
           "processtable", "mailbomb"}

NSL_GROUPS = {
    "H_bytes": ["src_bytes", "dst_bytes"],
    "H_cnt": ["count", "srv_count", "dst_host_count", "dst_host_srv_count"],
    "H_err": ["serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate"],
    "H_svc": ["same_srv_rate", "diff_srv_rate", "srv_diff_host_rate"],
    "H_dsterr": ["dst_host_serror_rate", "dst_host_srv_serror_rate",
                 "dst_host_rerror_rate", "dst_host_srv_rerror_rate"],
    "H_dstsvc": ["dst_host_same_srv_rate", "dst_host_diff_srv_rate",
                 "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate"],
}
UNSW_GROUPS = {
    "H_pkts": ["spkts", "dpkts"], "H_bytes": ["sbytes", "dbytes"],
    "H_load": ["sload", "dload"], "H_loss": ["sloss", "dloss"],
    "H_inter": ["sinpkt", "dinpkt"], "H_jit": ["sjit", "djit"],
    "H_mean": ["smean", "dmean"], "H_ttl": ["sttl", "dttl"],
    "H_ct": ["ct_srv_src", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
             "ct_dst_src_ltm", "ct_src_ltm", "ct_srv_dst"],
}


def load(name, scenario="all"):
    """Return Xtr, ytr, Xte, yte (DataFrames/arrays), groups, categorical columns.
    scenario: 'all' (every attack vs normal) or 'dos' (DoS family vs normal)."""
    if name == "nsl":
        tr = pd.read_csv(D + "KDDTrain+.txt", header=None, names=NSL_COLS)
        te = pd.read_csv(D + "KDDTest+.txt", header=None, names=NSL_COLS)
        for d in (tr, te):
            d["y"] = (d.label != "normal").astype(int)
            d["fam_dos"] = d.label.isin(NSL_DOS)
            d.drop(columns=["difficulty"], inplace=True)
        cats, groups = ["protocol_type", "service", "flag"], NSL_GROUPS
        drop = ["label", "fam_dos", "y"]
        if scenario == "dos":
            tr = tr[(tr.y == 0) | tr.fam_dos]; te = te[(te.y == 0) | te.fam_dos]
    elif name == "unsw":
        tr = pd.read_csv(D + "unsw_train.csv"); te = pd.read_csv(D + "unsw_test.csv")
        for d in (tr, te):
            d["y"] = d["label"].astype(int)
        cats, groups = ["proto", "service", "state"], UNSW_GROUPS
        drop = ["id", "attack_cat", "label", "y"]
        if scenario == "dos":
            tr = tr[tr.attack_cat.isin(["Normal", "DoS"])]; te = te[te.attack_cat.isin(["Normal", "DoS"])]
    else:
        raise ValueError(name)
    ytr, yte = tr["y"].values, te["y"].values
    Xtr, Xte = tr.drop(columns=drop).reset_index(drop=True), te.drop(columns=drop).reset_index(drop=True)
    return Xtr, ytr, Xte, yte, groups, cats


# ---------------------------------------------------------- feature engineering
def shannon(M, normalise=True):
    """Row-wise Shannon entropy (bits) of the non-negative vectors in M (n x m)."""
    M = np.clip(np.asarray(M, float), 0, None) + 1e-9
    P = M / M.sum(1, keepdims=True)
    H = -(P * np.log2(P)).sum(1)
    return H / np.log2(M.shape[1]) if normalise else H


class EntropyFeatures(BaseEstimator, TransformerMixin):
    """Append group-wise normalised Shannon entropy features to a DataFrame."""
    def __init__(self, groups=None):
        self.groups = groups

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for name, cols in (self.groups or {}).items():
            X[name] = shannon(X[cols].values)
        return X


class MISelector(BaseEstimator, TransformerMixin):
    """Keep the k features with the highest mutual information I(X_j;Y).
    MI is estimated on a random subsample (n_sub) of the training data for speed."""
    def __init__(self, k=20, n_sub=8000, seed=SEED):
        self.k, self.n_sub, self.seed = k, n_sub, seed

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        idx = rng.choice(X.shape[0], min(self.n_sub, X.shape[0]), replace=False)
        Xs = X[idx].toarray() if hasattr(X, "toarray") else X[idx]
        self.mi_ = mutual_info_classif(Xs, np.asarray(y)[idx], random_state=self.seed)
        self.keep_ = np.argsort(-self.mi_)[: min(self.k, len(self.mi_))]
        return self

    def transform(self, X):
        return X[:, self.keep_]


def make_pre(cats, groups, entropy, scale=True):
    steps = []
    if entropy:
        steps.append(("ent", EntropyFeatures(groups)))
    ct = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats)],
        remainder=StandardScaler() if scale else "passthrough", sparse_threshold=0)
    # log1p of heavy-tailed numerics before scaling
    steps.append(("pre", _LogCT(ct, cats)))
    return steps


class _LogCT(BaseEstimator, TransformerMixin):
    """log1p on non-negative numeric columns, then the ColumnTransformer."""
    def __init__(self, ct=None, cats=None):
        self.ct, self.cats = ct, cats

    def _log(self, X):
        X = X.copy()
        for c in X.columns:
            if c not in self.cats and X[c].min() >= 0 and X[c].max() > 100:
                X[c] = np.log1p(X[c])
        return X

    def fit(self, X, y=None):
        self.ct.fit(self._log(X)); return self

    def transform(self, X):
        return self.ct.transform(self._log(X))


# ------------------------------------------------------------------ models
def clf_zoo(seed=SEED):
    return {
        "LR": LogisticRegression(max_iter=300, n_jobs=1),
        "NB": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=2),
        "DT": DecisionTreeClassifier(random_state=seed),
        "SVM": LinearSVC(C=1.0, max_iter=3000, dual=False),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=60, early_stopping=True, random_state=seed),
        "RF": RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=seed),
        "ET": ExtraTreesClassifier(n_estimators=100, n_jobs=2, random_state=seed),
        "XGB": XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.2, n_jobs=2,
                             random_state=seed, eval_metric="logloss", verbosity=0),
    }


def build(clf_name, cats, groups, entropy=False, mi_k=None, seed=SEED):
    steps = make_pre(cats, groups, entropy)
    if mi_k:
        steps.append(("mi", MISelector(k=mi_k)))
    steps.append(("clf", clf_zoo(seed)[clf_name]))
    return Pipeline(steps)


def score_of(m, X):
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)[:, 1]
    return m.decision_function(X)


def metrics(y, yhat, s):
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return dict(acc=accuracy_score(y, yhat), prec=precision_score(y, yhat, zero_division=0),
                rec=recall_score(y, yhat), f1=f1_score(y, yhat), auc=roc_auc_score(y, s),
                fpr=fp / max(fp + tn, 1), fnr=fn / max(fn + tp, 1), mcc=matthews_corrcoef(y, yhat))


def fit_eval(pipe, Xtr, ytr, Xte, yte, timing=True):
    t0 = time.perf_counter(); pipe.fit(Xtr, ytr); tf = time.perf_counter() - t0
    t0 = time.perf_counter(); yh = pipe.predict(Xte); ti = time.perf_counter() - t0
    s = score_of(pipe, Xte)
    r = metrics(yte, yh, s)
    r.update(t_train=tf, t_infer_ms_per_1k=1000 * ti / len(yte) * 1000, scores=s)
    return r
