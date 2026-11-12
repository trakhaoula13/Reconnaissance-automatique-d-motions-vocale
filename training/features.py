"""Chargement audio, augmentation de données et extraction des caractéristiques."""
import numpy as np
import librosa

SR = 22050            # fréquence d'échantillonnage pour les caractéristiques classiques
MAX_DURATION = 8.0    # on garde au maximum 8 secondes
MIN_DURATION = 0.4    # en dessous, l'audio est jugé trop court

# AMÉLIORATION 1 : "calme" (02) est regroupé avec "neutre" -> 7 classes
RAVDESS_EMOTIONS = {
    "01": "neutral", "02": "neutral", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
LABELS_FR = {
    "neutral": "Neutre / calme", "happy": "Joie", "sad": "Tristesse",
    "angry": "Colère", "fearful": "Peur", "disgust": "Dégoût",
    "surprised": "Surprise",
}


def load_audio(path, sr=SR):
    """Charge un audio, supprime les silences au début/à la fin et normalise."""
    y, _ = librosa.load(path, sr=sr, mono=True, duration=MAX_DURATION * 2)
    y, _ = librosa.effects.trim(y, top_db=30)
    y = y[: int(MAX_DURATION * sr)]
    if len(y) < MIN_DURATION * sr:
        raise ValueError("Audio trop court ou silencieux (min. 0,4 s de parole).")
    return y / (np.max(np.abs(y)) + 1e-9)


def augment(y, seed=0, sr=SR):
    """AMÉLIORATION 2 : retourne 3 versions modifiées du signal
    (bruit à 20 dB, hauteur +2 demi-tons, hauteur -2 demi-tons)."""
    rng = np.random.default_rng(seed)
    rms = np.sqrt(np.mean(y ** 2)) + 1e-9
    noisy = y + rng.normal(0, rms / 10, len(y))          # rapport signal/bruit = 20 dB
    up = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
    return [noisy, up, down]


def _stats(m):
    """Moyenne et écart-type de chaque ligne d'une matrice temps-fréquence."""
    return np.concatenate([m.mean(axis=1), m.std(axis=1)])


def extract_features(y, sr=SR):
    """MFCC (+delta), chroma, Mel, contraste spectral, ZCR, RMS (moyenne + écart-type)."""
    stft = np.abs(librosa.stft(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64))
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    return np.concatenate([_stats(m) for m in (mfcc, delta, chroma, mel, contrast, zcr, rms)])