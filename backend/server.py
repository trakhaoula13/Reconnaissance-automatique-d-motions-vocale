"""Serveur de l'application : sert l'interface (frontend/) et expose POST /api/analyze.

Usage (depuis la racine du projet) :
    pip install flask
    python backend/server.py            ->  http://localhost:5000
"""
import os
import sys
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))        # réutilise features.py, w2v_features.py...

import joblib
import librosa
import numpy as np
from flask import Flask, jsonify, request

from evaluation import plot_predictions
from features import LABELS_FR, SR, extract_features, load_audio

MODELS = ROOT / "models"
app = Flask(__name__, static_folder=str(ROOT / "frontend"), static_url_path="")
_loaded = {}

# Figures "temps réel" : une image par analyse + un récapitulatif des 8 dernières
REALTIME = MODELS / "figures" / "realtime"
_history, _lock = deque(maxlen=8), Lock()
# Désactivé par défaut : écrire des fichiers dans le projet fait recharger la page avec Live Server.
# Pour enregistrer une figure à chaque analyse : python backend/server.py --save-figures
SAVE_REALTIME_FIGURES = "--save-figures" in sys.argv or os.environ.get("SAVE_REALTIME_FIGURES") == "1"


def save_realtime_figures(y, classes, proba, model_name, kind):
    """Enregistre models/figures/realtime/prediction_<date>.png et dernieres_predictions.png."""
    now = datetime.now()
    label = "WavLM" if kind == "w2v" else "SVM"
    item = dict(name=f"{now:%H:%M:%S} · {label}", wave=y, sr=SR, classes=list(classes), probs=proba)
    with _lock:
        plot_predictions([item], REALTIME, f"prediction_{now:%Y%m%d_%H%M%S}.png",
                         f"Prédiction en temps réel – {model_name}")
        _history.append(item)
        plot_predictions(list(_history), REALTIME, "dernieres_predictions.png",
                         f"{len(_history)} dernières prédictions")


def get_model(kind):
    """Charge un modèle une seule fois (au premier appel)."""
    if kind not in _loaded:
        name = "ser_w2v.joblib" if kind == "w2v" else "ser_model.joblib"
        path = MODELS / name
        if not path.exists():
            hint = "train_w2v.py" if kind == "w2v" else "train.py"
            raise FileNotFoundError(f"Modèle introuvable ({name}). Lancez d'abord training/{hint}.")
        _loaded[kind] = joblib.load(path)
    return _loaded[kind]


def mel_image(y):
    """Spectrogramme Mel réduit (64 bandes) en entiers 0-255 pour l'affichage."""
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_mels=64, hop_length=SR // 40))
    S = (S - S.min()) / (S.max() - S.min() + 1e-9)
    return (S * 255).astype(int).tolist()


@app.after_request
def allow_local_pages(resp):
    """Autorise la page à appeler l'API même si elle est ouverte depuis un autre port (ex. Live Server)."""
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Cache-Control"] = "no-store"          # toujours recharger index.html / style.css / script.js
    return resp


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.post("/api/analyze")
def analyze():
    f = request.files.get("audio")
    kind = request.form.get("model", "w2v")
    if f is None or kind not in ("w2v", "svm"):
        return jsonify(error="Requête invalide."), 400

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        f.save(tmp)
        path = tmp.name
    try:
        y = load_audio(path)                                   # 22 050 Hz, pour SVM et affichage
        if kind == "w2v":
            from w2v_features import SR_W2V, embed_layers, select_layers
            bundle = get_model("w2v")
            y16 = load_audio(path, sr=SR_W2V)
            feat = select_layers(embed_layers(y16, bundle["backbone"]), bundle["layers"])
            clf, model_name = bundle["clf"], "Deep learning (WavLM) – plus précis"
        else:
            clf, model_name = get_model("svm"), "Classique (MFCC + SVM) – rapide"
            feat = extract_features(y)
        proba = clf.predict_proba(feat.reshape(1, -1))[0]
        try:
            if SAVE_REALTIME_FIGURES:
                save_realtime_figures(y, clf.classes_, proba, model_name, kind)
        except Exception as e:                                  # une figure ratée ne doit pas bloquer l'analyse
            print("Figure temps réel non enregistrée :", e)
        return jsonify(
            probs={c: float(p) for c, p in zip(clf.classes_, proba)},
            labels=LABELS_FR, model=model_name, mel=mel_image(y))
    except ValueError as e:                                     # audio trop court / silencieux
        return jsonify(error=str(e)), 400
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 500
    except Exception as e:                                      # toute autre erreur : message clair + trace dans le terminal
        import traceback
        traceback.print_exc()
        return jsonify(error=f"Erreur interne : {type(e).__name__} : {e}"), 500
    finally:
        os.remove(path)


if __name__ == "__main__":
    print("Figures « temps réel » :", f"activées ({REALTIME})" if SAVE_REALTIME_FIGURES else "désactivées (option --save-figures)")
    app.run(host="127.0.0.1", port=5000, debug=False)