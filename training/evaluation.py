"""Découpage par locuteurs, courbes d'apprentissage et figures (enregistrées automatiquement).

Toutes les figures utilisent matplotlib.figure.Figure (sans pyplot) : le module peut donc être
appelé sans risque depuis le serveur Flask (plusieurs requêtes en parallèle).
"""
import os
import warnings

import numpy as np
from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASSES = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
FR = {"neutral": "Neutre", "happy": "Joie", "sad": "Tristesse", "angry": "Colère",
      "fearful": "Peur", "disgust": "Dégoût", "surprised": "Surprise"}
BLUE, GREEN, RED = "#4f46e5", "#16a34a", "#dc2626"


def _save(fig, folder, name):
    os.makedirs(folder, exist_ok=True)
    fig.tight_layout()
    path = os.path.join(folder, name)
    fig.savefig(path, dpi=150)
    return path


# ------------------------------------------------------------------ données
def subsample(entries, max_n, seed=0):
    """Limite chaque base (sauf RAVDESS) à max_n fichiers."""
    if not max_n:
        return entries
    rng, out = np.random.default_rng(seed), []
    for d in dict.fromkeys(e.dataset for e in entries):
        group = [e for e in entries if e.dataset == d]
        if d != "RAVDESS" and len(group) > max_n:
            group = [group[i] for i in np.sort(rng.choice(len(group), max_n, replace=False))]
        out += group
    return out


def heldout_split(ds, spk, test_size=0.2, seed=42):
    """Dans CHAQUE base, 20 % des locuteurs sont mis de côté pour le test (jamais vus à l'entraînement).
    Pour RAVDESS, ce sont les mêmes acteurs qu'avant (même graine)."""
    test = []
    for d in dict.fromkeys(ds.tolist()):
        idx = np.where(ds == d)[0]
        if len(set(spk[idx])) < 2:
            continue
        _, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
                     .split(idx, groups=spk[idx]))
        test.append(idx[te])
    test = np.concatenate(test)
    return np.setdiff1d(np.arange(len(ds)), test), test


# ------------------------------------------------------------------ 1. matrices de confusion
def _draw_cm(ax, y, pred, title):
    labs = [c for c in CLASSES if c in set(y) | set(pred)]
    cm = confusion_matrix(y, pred, labels=labs).astype(float)
    cm /= np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ticks = range(len(labs))
    ax.set_xticks(ticks)
    ax.set_xticklabels([FR[c] for c in labs], rotation=45, ha="right")
    ax.set_yticks(ticks)
    ax.set_yticklabels([FR[c] for c in labs])
    for i in ticks:
        for j in ticks:
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Émotion prédite")
    ax.set_ylabel("Émotion réelle")
    ax.set_title(title, fontsize=10)


def plot_confusions(y, pred, ds, tag, label, folder):
    """Une matrice par base (confusion_<tag>_<BASE>.png) + une image avec toutes les bases."""
    y, pred, ds = np.asarray(y), np.asarray(pred), np.asarray(ds)
    names = list(dict.fromkeys(ds.tolist()))
    big = Figure(figsize=(6.5 * len(names), 6))
    axes = big.subplots(1, len(names), squeeze=False)[0]
    for ax, d in zip(axes, names):
        m = ds == d
        title = f"{label} – {d} (n={m.sum()}, accuracy={accuracy_score(y[m], pred[m]):.2f})"
        _draw_cm(ax, y[m], pred[m], title)
        one = Figure(figsize=(7, 6))
        _draw_cm(one.subplots(), y[m], pred[m], title)
        _save(one, folder, f"confusion_{tag}_{d}.png")
    _save(big, folder, f"confusion_{tag}_toutes_bases.png")


# ------------------------------------------------------------------ 4. catégories par base
def plot_categories(y, ds, folder):
    y, ds = np.asarray(y), np.asarray(ds)
    names = list(dict.fromkeys(ds.tolist()))

    def draw(ax, d):
        yy = y[ds == d]
        counts = [int((yy == c).sum()) for c in CLASSES]
        bars = ax.bar(range(len(CLASSES)), counts, color=BLUE)
        ax.bar_label(bars, fontsize=8)
        ax.set_xticks(range(len(CLASSES)))
        ax.set_xticklabels([FR[c] for c in CLASSES], rotation=45, ha="right")
        ax.set_ylabel("Nombre d'enregistrements")
        ax.set_title(f"{d} – {len(yy)} enregistrements")

    cols = 2
    big = Figure(figsize=(6.5 * cols, 4.5 * ((len(names) + 1) // cols)))
    axes = big.subplots((len(names) + 1) // cols, cols, squeeze=False).ravel()
    for ax, d in zip(axes, names):
        draw(ax, d)
        one = Figure(figsize=(7, 4.5))
        draw(one.subplots(), d)
        _save(one, folder, f"categories_{d}.png")
    for ax in axes[len(names):]:
        ax.axis("off")
    _save(big, folder, "categories_toutes_bases.png")


# ------------------------------------------------------------------ 2. prédictions sur des enregistrements
def plot_predictions(items, folder, name, title):
    """items : liste de dict(name, wave, sr, classes, probs, [true]). Une ligne par enregistrement :
    forme d'onde à gauche, probabilités par émotion à droite."""
    n = len(items)
    fig = Figure(figsize=(11, 1.9 * n + 0.9))
    axes = fig.subplots(n, 2, squeeze=False, gridspec_kw={"width_ratios": [3, 2]})
    for (aw, ab), it in zip(axes, items):
        w, step = it["wave"], max(1, len(it["wave"]) // 4000)
        aw.plot(np.arange(0, len(w), step) / it["sr"], w[::step], lw=0.6, color=BLUE)
        aw.set_ylim(-1, 1)
        aw.set_yticks([])
        probs = np.asarray(it["probs"])
        pred = it["classes"][int(probs.argmax())]
        true = it.get("true")
        txt, color = f"{it['name']} → prédit : {FR[pred]} ({probs.max():.0%})", "black"
        if true is not None:
            txt = f"{it['name']} | réel : {FR[true]} → prédit : {FR[pred]} ({probs.max():.0%})"
            color = GREEN if true == pred else RED
        aw.set_title(txt, fontsize=9, color=color, loc="left")
        ab.barh([FR[c] for c in it["classes"]], probs,
                color=[BLUE if c == pred else "#c7d2fe" for c in it["classes"]])
        ab.invert_yaxis()
        ab.set_xlim(0, 1)
        ab.tick_params(labelsize=8)
    axes[-1][0].set_xlabel("Temps (s)")
    axes[-1][1].set_xlabel("Probabilité")
    fig.suptitle(title)
    return _save(fig, folder, name)


# ------------------------------------------------------------------ 3. perte et précision
def _new(xlabel):
    return {"x": [], "xlabel": xlabel, "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}


def _record(h, x, clf, A, ytr, B, yva):
    ptr, pva = clf.predict_proba(A), clf.predict_proba(B)
    h["x"].append(x)
    h["train_loss"].append(log_loss(ytr, ptr, labels=clf.classes_))
    h["val_loss"].append(log_loss(yva, pva, labels=clf.classes_))
    h["train_acc"].append(accuracy_score(ytr, clf.classes_[ptr.argmax(1)]))
    h["val_acc"].append(accuracy_score(yva, clf.classes_[pva.argmax(1)]))


def mlp_history(Xtr, ytr, Xva, yva, epochs=60, seed=42):
    """Perte (log-loss) et précision à chaque époque du MLP (mêmes hyperparamètres que le modèle final)."""
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xva)
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-2, random_state=seed)
    h, classes = _new("Époque"), np.unique(ytr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ep in range(1, epochs + 1):
            clf.partial_fit(A, ytr, classes=classes)
            _record(h, ep, clf, A, ytr, B, yva)
    return h


def logreg_history(Xtr, ytr, Xva, yva, C, balanced, rounds=30, step=5):
    """Perte et précision de la régression logistique en fonction du nombre d'itérations L-BFGS."""
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xva)
    clf = LogisticRegression(C=C, max_iter=step, warm_start=True,
                             class_weight="balanced" if balanced else None)
    h = _new("Itérations (L-BFGS)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in range(1, rounds + 1):
            clf.fit(A, ytr)                     # warm_start : repart des coefficients précédents
            _record(h, r * step, clf, A, ytr, B, yva)
    return h


def svm_history(Xtr, ytr, Xva, yva, C, gamma, fractions=(0.1, 0.25, 0.5, 0.75, 1.0), cap=12000, seed=0):
    """Le SVM n'est pas itératif (pas de perte par époque) : courbe d'apprentissage de la précision
    en fonction de la taille de l'ensemble d'entraînement."""
    rng = np.random.default_rng(seed)
    if len(Xtr) > cap:
        sel = rng.choice(len(Xtr), cap, replace=False)
        Xtr, ytr = Xtr[sel], ytr[sel]
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xva)
    perm = rng.permutation(len(A))
    h = _new("Taille de l'ensemble d'entraînement")
    for f in fractions:
        idx = perm[: max(10, int(f * len(A)))]
        clf = SVC(C=C, gamma=gamma).fit(A[idx], ytr[idx])
        h["x"].append(len(idx))
        h["train_acc"].append(accuracy_score(ytr[idx], clf.predict(A[idx])))
        h["val_acc"].append(accuracy_score(yva, clf.predict(B)))
    h["train_loss"] = h["val_loss"] = None
    return h


def plot_curves(h, title, folder, name):
    has_loss = h["train_loss"] is not None
    fig = Figure(figsize=(11 if has_loss else 6, 4.3))
    axes = np.atleast_1d(fig.subplots(1, 2 if has_loss else 1))
    x = h["x"]
    if has_loss:
        axes[0].plot(x, h["train_loss"], label="entraînement", color=BLUE)
        axes[0].plot(x, h["val_loss"], label="validation (locuteurs inconnus)", color=RED)
        axes[0].set_title("Perte (log-loss)")
    ax = axes[-1]
    ax.plot(x, h["train_acc"], label="entraînement", color=BLUE, marker="o" if not has_loss else None)
    ax.plot(x, h["val_acc"], label="validation (locuteurs inconnus)", color=RED, marker="o" if not has_loss else None)
    ax.set_ylim(0, 1.02)
    ax.set_title("Précision (accuracy)")
    for a in axes:
        a.set_xlabel(h["xlabel"])
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(title)
    return _save(fig, folder, name)
