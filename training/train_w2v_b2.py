"""Modèle deep learning (WavLM / wav2vec2 figé + régression logistique), configuration B2 : RAVDESS + CREMA-D, SANS TESS.

Différence avec train_w2v.py : seul l'import change (emotion_datasets_b2 -> bases par défaut RAVDESS + CREMAD,
faux locuteur TESS « OA » corrigé). Le cache d'embeddings existant (models/emb_v2_wavlm-large.npz) est réutilisé tel
quel : les embeddings TESS qu'il contient sont simplement ignorés. AUCUNE extraction n'est refaite.

Usage (depuis la racine du projet) :
    mkdir models_B2
    cp models/emb_v2_wavlm-large.npz models_B2/
    python training/train_w2v_b2.py --data data --out models_B2

Évaluation : dans chaque base, 20 % des locuteurs sont mis de côté (jamais vus à l'entraînement).

Figures enregistrées dans <out>/figures/ ; rapport dans <out>/report_w2v.txt.
"""
import os
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import argparse

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from emotion_datasets_b2 import DEFAULT_DATASETS, collect_entries
from evaluation import (heldout_split, logreg_history, plot_categories, plot_confusions,
                        plot_curves, plot_predictions, subsample)
from features import SR, load_audio
from w2v_features import (BACKBONES, DEFAULT_BACKBONE, SR_W2V, embed_layers,
                          layer_configs, num_layers, select_layers)


def pool_configs(emb, cfgs):
    """(L, F) -> (nb_configs, F) : on ne garde que les moyennes de couches utiles (cache léger)."""
    return np.stack([select_layers(emb, layers) for layers in cfgs.values()]).astype(np.float32)


def save_store(cache, store, L):
    tmp = cache + ".tmp.npz"
    np.savez(tmp, keys=np.array(list(store.keys())), Z=np.stack(list(store.values())), L=L)
    os.replace(tmp, cache)                      # écriture "atomique" : pas de fichier à moitié écrit


def load_store(cache, old_cache, entries):
    store, L = {}, None
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        store, L = dict(zip(d["keys"].tolist(), d["Z"])), int(d["L"])
        print(f"Cache chargé : {len(store)} embeddings déjà calculés")
    elif os.path.exists(old_cache):             # ancien cache (RAVDESS seul) -> évite 25 min de calcul
        d = np.load(old_cache, allow_pickle=True)
        rav = [e for e in entries if e.dataset == "RAVDESS"]
        if len(rav) == len(d["E"]) and d["y"].tolist() == [e.label for e in rav]:
            L = int(d["E"].shape[1])
            cfgs = layer_configs(L)
            for e, emb in zip(rav, d["E"]):
                store[e.key] = pool_configs(emb, cfgs)
            print(f"Ancien cache RAVDESS réutilisé ({len(store)} embeddings)")
    return store, L


def extract_missing(entries, store, L, backbone, cache):
    todo = [e for e in entries if e.key not in store]
    if todo:
        if L is None:
            L = num_layers(backbone)
        cfgs = layer_configs(L)
        new = 0
        for e in tqdm(todo, desc=f"Embeddings {backbone}"):
            try:
                audio = load_audio(e.path, sr=SR_W2V)
            except Exception:                   # audio trop court, illisible...
                continue
            store[e.key] = pool_configs(embed_layers(audio, backbone), cfgs)
            new += 1
            if new % 250 == 0:
                save_store(cache, store, L)     # point de sauvegarde
    if store and (todo or not os.path.exists(cache)):
        save_store(cache, store, L)
    return store, L


def make_clf(C, balanced):
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=C, max_iter=3000, class_weight="balanced" if balanced else None))


def per_dataset_text(y, pred, ds):
    return "\n".join(f"  {d:8s} accuracy={accuracy_score(y[ds == d], pred[ds == d]):.3f}  (n={(ds == d).sum()})"
                     for d in dict.fromkeys(ds.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="dossier contenant RAVDESS/, CREMAD/, ...")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--max-per-dataset", type=int, default=None)
    ap.add_argument("--out", default="models_B2")
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE, choices=list(BACKBONES))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    figs = os.path.join(args.out, "figures")

    print("Bases de données :")
    entries = subsample(collect_entries(args.data, args.datasets), args.max_per_dataset)
    cache = os.path.join(args.out, f"emb_v2_{args.backbone}.npz")
    old_cache = os.path.join(args.out, f"emb_{args.backbone}.npz")
    store, L = load_store(cache, old_cache, entries)
    store, L = extract_missing(entries, store, L, args.backbone, cache)
    entries = [e for e in entries if e.key in store]

    Z = np.stack([store[e.key] for e in entries])                  # (n, nb_configs, F)
    y = np.array([e.label for e in entries])
    ds = np.array([e.dataset for e in entries])
    spk = np.array([f"{e.dataset}:{e.speaker}" for e in entries])  # locuteurs uniques entre bases
    cfgs = layer_configs(L)
    names = list(cfgs)
    balanced = len(set(ds.tolist())) > 1
    print("\nEnregistrements par base :", {d: int((ds == d).sum()) for d in dict.fromkeys(ds.tolist())})

    plot_categories(y, ds, figs)                                   # figure 4

    # Test = 20 % des locuteurs de chaque base (RAVDESS : mêmes acteurs qu'avant).
    train, test = heldout_split(ds, spk)
    print(f"Entraînement : {len(train)} enregistrements | Test (locuteurs jamais vus) : {len(test)}")

    best_score, best_ci, best_C = -1, None, None
    for ci, name in enumerate(names):
        for C in (0.03, 0.1, 0.3):
            score = cross_val_score(make_clf(C, balanced), Z[train, ci], y[train], groups=spk[train],
                                    cv=GroupKFold(n_splits=4), scoring="f1_macro", n_jobs=-1).mean()
            print(f"  {name:24s} C={C:<5} F1-macro (CV) = {score:.3f}")
            if score > best_score:
                best_score, best_ci, best_C = score, ci, C
    best_name, Zb = names[best_ci], Z[:, best_ci]
    print(f"\nChoix : {best_name}, C={best_C}")

    clf = make_clf(best_C, balanced).fit(Zb[train], y[train])
    pred = clf.predict(Zb[test])
    acc, f1 = accuracy_score(y[test], pred), f1_score(y[test], pred, average="macro")
    per_ds = per_dataset_text(y[test], pred, ds[test])
    print(f"TEST (locuteurs jamais vus) accuracy={acc:.3f}  F1-macro={f1:.3f}\n{per_ds}\n")
    report = classification_report(y[test], pred, digits=3)
    print(report)

    # Figure 1 : matrices de confusion par base
    label = BACKBONES[args.backbone]
    plot_confusions(y[test], pred, ds[test], args.backbone, label, figs)

    # Figure 3 : perte et précision (validation = locuteurs inconnus pris dans l'entraînement)
    sub, va = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0).split(train, groups=spk[train]))
    sub, va = train[sub], train[va]
    h = logreg_history(Zb[sub], y[sub], Zb[va], y[va], best_C, balanced)
    plot_curves(h, f"{label} + régression logistique – perte et précision", figs, f"courbes_{args.backbone}.png")

    # Figure 2 : prédictions sur quelques enregistrements de test
    sel = np.random.default_rng(0).choice(test, min(8, len(test)), replace=False)
    proba, items = clf.predict_proba(Zb[sel]), []
    for i, p in zip(sel, proba):
        try:
            wave = load_audio(entries[i].path)
        except ValueError:
            continue
        items.append(dict(name=entries[i].key, wave=wave, sr=SR, classes=list(clf.classes_), probs=p, true=y[i]))
    plot_predictions(items, figs, f"predictions_exemples_{args.backbone}.png",
                     f"Prédictions sur des enregistrements de test – {label}")

    # Validation croisée par acteurs RAVDESS (6 plis) ; les autres bases sont toujours dans l'entraînement
    rav = np.where(ds == "RAVDESS")[0]
    others = np.where(ds != "RAVDESS")[0]
    accs = []
    for a, b in GroupKFold(n_splits=6).split(rav, y[rav], spk[rav]):
        idx = np.concatenate([rav[a], others])
        m = make_clf(best_C, balanced).fit(Zb[idx], y[idx])
        accs.append(accuracy_score(y[rav[b]], m.predict(Zb[rav[b]])))
    cv_txt = f"Validation croisée par acteurs RAVDESS (6 plis) : accuracy = {np.mean(accs):.3f} ± {np.std(accs):.3f}"
    print(cv_txt)

    used = ", ".join(f"{d}={int((ds == d).sum())}" for d in dict.fromkeys(ds.tolist()))
    with open(os.path.join(args.out, "report_w2v.txt"), "w", encoding="utf-8") as f:
        f.write(f"{label} – {best_name}, C={best_C}\nBases : {used}\n"
                f"Test : accuracy={acc:.3f} F1-macro={f1:.3f}\nPar base :\n{per_ds}\n{cv_txt}\n\n{report}")
    print("Figures enregistrées dans", figs)

    final = make_clf(best_C, balanced).fit(Zb, y)               # réentraînement sur TOUTES les données demandées
    joblib.dump({"version": 3, "clf": final, "layers": cfgs[best_name], "layer_config": best_name,
                 "backbone": args.backbone, "datasets": sorted(set(ds.tolist()))},
                os.path.join(args.out, "ser_w2v.joblib"))
    print("Modèle sauvegardé dans", os.path.join(args.out, "ser_w2v.joblib"))


if __name__ == "__main__":
    main()
