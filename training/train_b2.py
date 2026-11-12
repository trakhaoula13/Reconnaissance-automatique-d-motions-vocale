"""Entraînement du modèle classique (MFCC + SVM / MLP), configuration B2 : RAVDESS + CREMA-D, SANS TESS.

Différences avec train.py :
  - les bases par défaut sont RAVDESS + CREMAD (import de emotion_datasets_b2) ;
  - l'option --cache permet de réutiliser un cache de caractéristiques déjà calculé (même s'il contient
    TESS) : les lignes des bases non demandées sont retirées, AUCUNE extraction n'est refaite.

Usage (depuis la racine du projet) :
    mkdir models_B2
    cp models/features_v3_CREMAD-RAVDESS-TESS.npz models_B2/
    python training/train_b2.py --data data --out models_B2 --cache models_B2/features_v3_CREMAD-RAVDESS-TESS.npz

Figures enregistrées dans <out>/figures/ ; rapport dans <out>/report.txt.
"""
import os
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")  # cache l'avertissement SVC

import argparse

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from emotion_datasets_b2 import DEFAULT_DATASETS, collect_entries
from evaluation import (heldout_split, mlp_history, plot_categories, plot_confusions,
                        plot_curves, plot_predictions, subsample, svm_history)
from features import SR, augment, extract_features, load_audio


def process_file(idx, path):
    """Retourne (1 + 3, d) : caractéristiques de l'original + 3 versions augmentées."""
    try:
        y = load_audio(path)
    except ValueError:
        return None
    return np.array([extract_features(y)] + [extract_features(a) for a in augment(y, seed=idx)])


def build_dataset(entries, cache):
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return [d[k] for k in ("X", "X_aug", "y", "ds", "spk", "paths")]
    results = Parallel(n_jobs=-1)(
        delayed(process_file)(i, e.path) for i, e in tqdm(list(enumerate(entries)), desc="Extraction (+ augmentation)"))
    keep = [k for k, r in enumerate(results) if r is not None]
    F = np.stack([results[k] for k in keep])               # (n, 4, d)
    data = dict(X=F[:, 0], X_aug=F[:, 1:],
                y=np.array([entries[k].label for k in keep]),
                ds=np.array([entries[k].dataset for k in keep]),
                spk=np.array([f"{entries[k].dataset}:{entries[k].speaker}" for k in keep]),
                paths=np.array([entries[k].path for k in keep]))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, **data)
    return list(data.values())


def stack_train(X, X_aug, y, groups, idx, use_aug):
    """Rassemble les données d'entraînement (originaux + versions augmentées)."""
    if not use_aug:
        return X[idx], y[idx], groups[idx]
    k = X_aug.shape[1]
    Xt = np.vstack([X[idx], X_aug[idx].reshape(-1, X.shape[1])])
    yt = np.concatenate([y[idx], np.repeat(y[idx], k)])
    gt = np.concatenate([groups[idx], np.repeat(groups[idx], k)])
    return Xt, yt, gt


def fit_svm(Xt, yt, gt):
    # 1) recherche des hyperparamètres (sans probabilités : beaucoup plus rapide)
    grid = GridSearchCV(
        make_pipeline(StandardScaler(), SVC()),
        {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.001]},
        cv=GroupKFold(n_splits=4), scoring="f1_macro", n_jobs=-1)
    grid.fit(Xt, yt, groups=gt)
    p = grid.best_params_
    print("   meilleurs paramètres SVM :", p)
    # 2) modèle final avec probabilités (nécessaires pour l'interface)
    model = make_pipeline(StandardScaler(), SVC(
        C=p["svc__C"], gamma=p["svc__gamma"], probability=True, random_state=42))
    return model.fit(Xt, yt)


def fit_mlp(Xt, yt, gt):
    model = make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(256, 128), alpha=1e-2, max_iter=300, random_state=42))
    return model.fit(Xt, yt)


def per_dataset_text(y, pred, ds):
    return "\n".join(f"  {d:8s} accuracy={accuracy_score(y[ds == d], pred[ds == d]):.3f}  (n={(ds == d).sum()})"
                     for d in dict.fromkeys(ds.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="dossier contenant RAVDESS/, CREMAD/, ...")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--max-per-dataset", type=int, default=None)
    ap.add_argument("--out", default="models_B2")
    ap.add_argument("--cache", default=None,
                    help="cache de caractéristiques existant (peut contenir d'autres bases, qui seront retirées)")
    ap.add_argument("--no-augment", action="store_true",
                    help="n'utilise pas l'augmentation (pour comparer ; ne sauvegarde pas le modèle)")
    args = ap.parse_args()
    use_aug = not args.no_augment
    suffix = "" if use_aug else "_sans_aug"
    figs = os.path.join(args.out, "figures" + suffix)
    os.makedirs(args.out, exist_ok=True)

    print("Bases de données :")
    entries = subsample(collect_entries(args.data, args.datasets), args.max_per_dataset)
    tag = "-".join(sorted({e.dataset for e in entries})) + (f"_max{args.max_per_dataset}" if args.max_per_dataset else "")
    cache = args.cache or os.path.join(args.out, f"features_v3_{tag}.npz")
    X, X_aug, y, ds, spk, paths = build_dataset(entries, cache)

    # Retire du cache les bases non demandées (ex. TESS) : aucune extraction à refaire
    wanted = sorted({e.dataset for e in entries})
    keep = np.isin(ds, wanted)
    if not keep.all():
        print(f"Cache : {int((~keep).sum())} enregistrements d'autres bases retirés ({sorted(set(ds[~keep].tolist()))})")
    X, X_aug, y, ds, spk, paths = X[keep], X_aug[keep], y[keep], ds[keep], spk[keep], paths[keep]
    print(f"{len(X)} enregistrements, {X.shape[1]} caractéristiques | augmentation : {'OUI' if use_aug else 'NON'}")

    plot_categories(y, ds, figs)                                    # figure 4

    # Test = 20 % des locuteurs de CHAQUE base (jamais vus) ; le reste sert à l'entraînement.
    train, test = heldout_split(ds, spk)
    Xt, yt, gt = stack_train(X, X_aug, y, spk, train, use_aug)
    print(f"Entraînement sur {len(Xt)} exemples, test sur {len(test)} (originaux uniquement)")

    # Petit jeu de validation (locuteurs inconnus) pour les courbes perte / précision
    sub, va = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0).split(train, groups=spk[train]))
    sub, va = train[sub], train[va]
    Xs, ys, _ = stack_train(X, X_aug, y, spk, sub, use_aug)

    best = (None, None, None, -1)                                   # (label, modèle, prédictions, F1)
    for key, label, fit in (("svm", "SVM (RBF)", fit_svm), ("mlp", "MLP", fit_mlp)):
        model = fit(Xt, yt, gt)
        pred = model.predict(X[test])
        f1 = f1_score(y[test], pred, average="macro")
        print(f"{label:10s} accuracy={accuracy_score(y[test], pred):.3f}  F1-macro={f1:.3f}")
        print(per_dataset_text(y[test], pred, ds[test]))
        plot_confusions(y[test], pred, ds[test], key, label, figs)  # figure 1
        print("   courbes perte / précision...")
        if key == "mlp":                                            # figure 3
            h = mlp_history(Xs, ys, X[va], y[va])
        else:
            h = svm_history(Xs, ys, X[va], y[va], model[-1].C, model[-1].gamma)
        plot_curves(h, f"{label} – perte et précision", figs, f"courbes_{key}.png")
        if f1 > best[3]:
            best = (label, model, pred, f1)
    best_name, best_model, pred, _ = best

    # Figure 2 : prédictions sur quelques enregistrements de test
    rng = np.random.default_rng(0)
    sel = rng.choice(test, min(8, len(test)), replace=False)
    proba, items = best_model.predict_proba(X[sel]), []
    for i, p in zip(sel, proba):
        try:
            wave = load_audio(paths[i])
        except ValueError:
            continue
        items.append(dict(name=f"{ds[i]}/{os.path.basename(paths[i])}", wave=wave, sr=SR,
                          classes=list(best_model.classes_), probs=p, true=y[i]))
    plot_predictions(items, figs, "predictions_exemples.png",
                     f"Prédictions sur des enregistrements de test – {best_name}")

    report = classification_report(y[test], pred, digits=3)
    print(f"\nMeilleur modèle : {best_name}\n{report}")
    with open(os.path.join(args.out, f"report{suffix}.txt"), "w", encoding="utf-8") as f:
        f.write(f"Modèle : {best_name} | augmentation : {use_aug} | bases : {', '.join(wanted)}\n\nPar base :\n"
                f"{per_dataset_text(y[test], pred, ds[test])}\n\n{report}")
    print("Figures enregistrées dans", figs)

    if not use_aug:
        print("Mode comparaison : le modèle n'est pas sauvegardé.")
        return
    # Réentraînement final sur TOUTES les données des bases demandées (locuteurs de test inclus)
    allidx = np.arange(len(X))
    Xa, ya, ga = stack_train(X, X_aug, y, spk, allidx, True)
    final = fit_svm(Xa, ya, ga) if best_name.startswith("SVM") else fit_mlp(Xa, ya, ga)
    joblib.dump(final, os.path.join(args.out, "ser_model.joblib"))
    print("Modèle sauvegardé dans", os.path.join(args.out, "ser_model.joblib"))


if __name__ == "__main__":
    main()
