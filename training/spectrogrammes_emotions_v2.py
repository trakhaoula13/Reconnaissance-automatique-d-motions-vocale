"""Figure : forme d'onde + spectrogramme Mel pour 7 émotions (même acteur RAVDESS, même phrase),
au style de l'interface (fond blanc, colormap coolwarm, axes Time / Hz).

Paramètres identiques à features.py : 22 050 Hz, n_fft=2048, hop=512, 64 filtres Mel, fmax=11 025 Hz.
F0 : pYIN avec des bornes adaptées au sexe de l'acteur (RAVDESS : impair = homme, pair = femme), médiane.

Usage :
    python training/spectrogrammes_emotions_v2.py --data data --actor 3
Sortie : spectrogrammes_emotions.png
"""
import argparse
import glob
import os

import librosa
import librosa.display
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

SR, N_FFT, HOP, N_MELS, FMAX = 22050, 2048, 512, 64, 11025
COULEUR_ONDE = "#3274b5"          # bleu de la forme d'onde de l'interface

# (étiquette, code émotion RAVDESS, intensité). Le neutre n'existe qu'en intensité normale (01).
EMOTIONS = [("Neutre", "01", "01"), ("Joie", "03", "02"), ("Tristesse", "04", "02"),
            ("Colère", "05", "02"), ("Peur", "06", "02"), ("Dégoût", "07", "02"),
            ("Surprise", "08", "02")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/RAVDESS", help="dossier RAVDESS")
    ap.add_argument("--actor", type=int, default=3)
    ap.add_argument("--show-f0", action="store_true", help="trace le contour de F0 sur le spectrogramme (vérification)")
    ap.add_argument("--out", default="spectrogrammes_emotions.png")
    a = ap.parse_args()

    def chemin(code, inten):
        nom = f"03-01-{code}-{inten}-01-01-{a.actor:02d}.wav"
        t = glob.glob(os.path.join(a.data, "**", nom), recursive=True)
        if not t:
            raise FileNotFoundError(nom)
        return t[0]

    homme = a.actor % 2 == 1
    fmin_f0, fmax_f0 = (65, 450) if homme else (130, 700)

    donnees = []
    for nom, code, inten in EMOTIONS:
        y, _ = librosa.load(chemin(code, inten), sr=SR)
        y, _ = librosa.effects.trim(y, top_db=30)
        y = librosa.util.normalize(y)                       # amplitude dans [-1, 1], comme l'interface
        S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                           n_mels=N_MELS, fmax=FMAX)
        f0, _, _ = librosa.pyin(y, fmin=fmin_f0, fmax=fmax_f0, sr=SR, frame_length=N_FFT)
        f0_med = np.nanmedian(f0) if np.any(~np.isnan(f0)) else np.nan
        donnees.append((nom, y, S, f0, f0_med))

    ref = max(S.max() for _, _, S, _, _ in donnees)         # même référence dB pour toutes les émotions
    duree_max = max(len(y) / SR for _, y, _, _, _ in donnees)

    fig = plt.figure(figsize=(16, 8.5), facecolor="white")
    outer = GridSpec(2, 4, figure=fig, hspace=0.38, wspace=0.22)
    im = None
    for i, (nom, y, S, f0, f0_med) in enumerate(donnees):
        inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i // 4, i % 4],
                                        height_ratios=[1, 2], hspace=0.45)
        ax_w, ax_s = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])

        librosa.display.waveshow(y, sr=SR, ax=ax_w, color=COULEUR_ONDE, linewidth=0.8)
        f0_txt = f"{f0_med:.0f} Hz" if not np.isnan(f0_med) else "n.d."
        ax_w.set_title(f"{nom} (F0 méd. {f0_txt})", fontsize=10)
        ax_w.set_ylim(-1.05, 1.05)
        ax_w.set_xlim(0, duree_max)

        S_db = librosa.power_to_db(S, ref=ref, top_db=80)
        im = librosa.display.specshow(S_db, sr=SR, hop_length=HOP, x_axis="time", y_axis="mel",
                                      fmax=FMAX, cmap="coolwarm", vmin=-80, vmax=0, ax=ax_s)
        ax_s.set_xlim(0, duree_max)
        if a.show_f0:
            t = librosa.times_like(f0, sr=SR, hop_length=N_FFT // 4)    # pYIN : hop = frame_length // 4
            ax_s.plot(t, f0, color="black", lw=1)

    cax = fig.add_subplot(outer[1, 3])
    cax.axis("off")
    fig.colorbar(im, ax=cax, format="%+2.0f dB", fraction=0.5, label="Niveau (dB)")
    fig.suptitle(f"Forme d'onde et spectrogramme Mel – acteur {a.actor:02d}, même phrase, 7 émotions", fontsize=13)
    plt.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
    print("Figure enregistrée :", a.out)


if __name__ == "__main__":
    main()
