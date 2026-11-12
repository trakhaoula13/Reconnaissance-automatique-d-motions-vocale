"""Embeddings d'un modèle audio pré-entraîné (WavLM / wav2vec2), sans fine-tuning."""
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

BACKBONES = {
    "wavlm-large": "microsoft/wavlm-large",       # le plus précis, le plus lent (~1,2 Go)
    "wavlm": "microsoft/wavlm-base-plus",         # bon compromis vitesse / précision
    "wav2vec2": "facebook/wav2vec2-base",         # celui utilisé avant
}
DEFAULT_BACKBONE = "wavlm-large"
SR_W2V = 16000

_cache = {}


def get_model(backbone):
    if backbone not in _cache:
        name = BACKBONES[backbone]
        _cache[backbone] = (AutoFeatureExtractor.from_pretrained(name),
                            AutoModel.from_pretrained(name).eval())
    return _cache[backbone]


@torch.no_grad()
def embed_layers(y, backbone=DEFAULT_BACKBONE):
    """Signal 16 kHz -> (L, 2*D) : moyenne ET écart-type temporels de chaque couche."""
    fe, model = get_model(backbone)
    inputs = fe(y, sampling_rate=SR_W2V, return_tensors="pt")
    hidden = model(**inputs, output_hidden_states=True).hidden_states   # L x (1, T, D)
    H = torch.stack(hidden)[:, 0]                                       # (L, T, D)
    return torch.cat([H.mean(dim=1), H.std(dim=1)], dim=-1).numpy().astype(np.float32)


def layer_configs(n_layers_total):
    """Groupes de couches à tester (les couches intermédiaires portent le plus d'émotion)."""
    n = n_layers_total - 1

    def r(a, b):
        return list(range(int(round(a * n)), min(n, int(round(b * n))) + 1))

    return {
        "milieu 25-75 %": r(0.25, 0.75),
        "milieu 40-80 %": r(0.40, 0.80),
        "moitié haute 50-100 %": r(0.50, 1.0),
        "toutes les couches": list(range(n_layers_total)),
    }


def select_layers(E, layers):
    """E : (..., L, F) -> (..., F), moyenne des couches choisies."""
    return np.asarray(E)[..., layers, :].mean(axis=-2)


def num_layers(backbone):
    """Nombre de couches cachées (embeddings + couches du Transformer), ex. 25 pour wavlm-large."""
    return get_model(backbone)[1].config.num_hidden_layers + 1