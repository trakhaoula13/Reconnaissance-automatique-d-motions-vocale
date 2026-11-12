"""Analyse acoustique par émotion sur TOUTES les données (RAVDESS + CREMA-D + TESS).

Analyse DESCRIPTIVE (aucun modèle entraîné) : utiliser les trois bases ne crée donc aucune fuite.
Paramètres identiques à features.py : 22 050 Hz, STFT n_fft=2048, hop=512 (fenêtre de Hann),
64 filtres Mel jusqu'à 11 025 Hz, suppression des silences à 30 dB.

Précautions de méthode :
  - valeurs moyennées PAR LOCUTEUR, puis entre locuteurs (CREMA-D et ses 91 voix ne dominent pas) ;
  - F0 (demi-tons) et énergie (dB) exprimées RELATIVEMENT au neutre du même locuteur ;
  - énergie mesurée sur le signal NON normalisé en amplitude.

Usage (depuis la racine du projet) :
    python training/analyse_acoustique.py --data data --out figure_acoustique
    python training/analyse_acoustique.py --data data --limit 200        # test rapide

Sorties dans <out>/ : acoustique_fichiers.csv (cache), spectres.npy (cache), acoustique_par_corpus.csv,
tableau_acoustique.tex (lignes à coller dans le tableau LaTeX), acoustique_boxplots.png, spectre_moyen_emotions.png.
Supprimez les deux fichiers de cache pour recalculer.
"""
import argparse
import os

import librosa
import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emotion_datasets_b2 import collect_entries

SR, N_FFT, HOP, N_MELS, FMAX = 22050, 2048, 512, 64, 11025
ORDER = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
FR = {"neutral": "Neutre", "happy": "Joie", "sad": "Tristesse", "angry": "Colère",
      "fearful": "Peur", "disgust": "Dégoût", "surprised": "Surprise"}


def analyser(e):
    try:
        y, _ = librosa.load(e.path, sr=SR, mono=True)
        y, _ = librosa.effects.trim(y, top_db=30)               # même prétraitement que l'article
        y = y[: int(8.0 * SR)]
        if len(y) < 0.4 * SR:
            return None
        f0, _, _ = librosa.pyin(y, fmin=65, fmax=500, sr=SR, frame_length=N_FFT)   # hop = 512
        f0 = f0[~np.isnan(f0)]
        P = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2
        mel = librosa.feature.melspectrogram(S=P, sr=SR, n_mels=N_MELS, fmax=FMAX).mean(axis=1)
        ligne = dict(corpus=e.dataset, speaker=f"{e.dataset}:{e.speaker}", label=e.label,
                     f0=float(np.median(f0)) if len(f0) >= 5 else np.nan,
                     rms_db=float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12)),   # signal NON normalisé
                     zcr=float(librosa.feature.zero_crossing_rate(y).mean()),
                     dur=len(y) / SR)
        return ligne, (10 * np.log10(mel / mel.sum() + 1e-12)).astype(np.float32)   # forme spectrale
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="figure_acoustique")
    ap.add_argument("--limit", type=int, default=None, help="test rapide sur N fichiers")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    csv, npy = os.path.join(a.out, "acoustique_fichiers.csv"), os.path.join(a.out, "spectres.npy")

    if not (os.path.exists(csv) and os.path.exists(npy)):
        entries = collect_entries(a.data, ["RAVDESS", "CREMAD", "TESS"])[: a.limit]
        res = Parallel(n_jobs=-1, verbose=5)(delayed(analyser)(e) for e in entries)
        res = [r for r in res if r is not None]
        pd.DataFrame([r[0] for r in res]).to_csv(csv, index=False)
        np.save(npy, np.array([r[1] for r in res]))
    df, S = pd.read_csv(csv), np.load(npy)

    # valeurs relatives au NEUTRE du même locuteur
    neu = df[df.label == "neutral"].groupby("speaker")[["f0", "rms_db"]].median()
    df["d_f0"] = 12 * np.log2(df.f0 / df.speaker.map(neu.f0))     # demi-tons
    df["d_rms"] = df.rms_db - df.speaker.map(neu.rms_db)          # dB

    # moyenne par locuteur, puis entre locuteurs (chaque voix pèse pareil)
    cols = ["d_f0", "d_rms", "zcr", "dur"]
    loc = df.groupby(["label", "speaker"])[cols].mean().reset_index()
    tab = loc.groupby("label")[cols].agg(["mean", "std"]).reindex(ORDER)
    eff = df.groupby("label").agg(fichiers=("speaker", "size"),
                                  locuteurs=("speaker", "nunique")).reindex(ORDER)
    print(eff, "\n")
    print(tab.round(3))
    df.groupby(["corpus", "label"])[cols].mean().round(3).to_csv(os.path.join(a.out, "acoustique_par_corpus.csv"))

    def f(m, s, d=1):
        return f"{m:.{d}f} $\\pm$ {s:.{d}f}".replace(".", ",")

    with open(os.path.join(a.out, "tableau_acoustique.tex"), "w", encoding="utf-8") as fh:
        for lab in ORDER:
            r, n = tab.loc[lab], eff.loc[lab]
            fh.write(f"{FR[lab]} & {f(r[('d_f0', 'mean')], r[('d_f0', 'std')])} & "
                     f"{f(r[('d_rms', 'mean')], r[('d_rms', 'std')])} & "
                     f"{f(r[('zcr', 'mean')], r[('zcr', 'std')], 3)} & "
                     f"{f(r[('dur', 'mean')], r[('dur', 'std')], 2)} & {int(n.locuteurs)} \\\\\n")

    # Figure 1 : boîtes à moustaches (un point = un locuteur)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    titres = ["Δ F0 (demi-tons / neutre)", "Δ énergie (dB / neutre)", "ZCR", "Durée utile (s)"]
    for ax, c, t in zip(axes, cols, titres):
        ax.boxplot([loc[loc.label == l][c].dropna().values for l in ORDER], showfliers=False)
        ax.set_xticks(range(1, len(ORDER) + 1))
        ax.set_xticklabels([FR[l] for l in ORDER], rotation=45, ha="right")
        ax.set_title(t)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "acoustique_boxplots.png"), dpi=200)
    plt.close()

    # Figure 2 : spectre moyen à long terme par émotion (moyenne par locuteur puis entre locuteurs)
    freqs = librosa.mel_frequencies(n_mels=N_MELS, fmax=FMAX)
    plt.figure(figsize=(8, 4))
    for lab in ORDER:
        idx = df.index[df.label == lab].values
        par_loc = pd.DataFrame(S[idx]).groupby(df.loc[idx, "speaker"].values).mean()
        plt.plot(freqs, par_loc.mean().values, label=FR[lab])
    plt.xscale("symlog", linthresh=500)
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Niveau relatif (dB)")
    plt.legend(ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "spectre_moyen_emotions.png"), dpi=200)
    plt.close()
    print("\nRésultats dans", a.out)


if __name__ == "__main__":
    main()
