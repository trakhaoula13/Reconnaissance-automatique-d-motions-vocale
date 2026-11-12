"""Lecture de plusieurs bases de données d'émotions vocales, regroupées en 7 classes communes :
neutral, happy, sad, angry, fearful, disgust, surprised.

VERSION B2 (par rapport à emotion_datasets.py) :
  1. DEFAULT_DATASETS = RAVDESS + CREMAD seulement (TESS et SAVEE ne sont plus dans l'entraînement).
  2. parse_tess : le fichier OA_bite_neutral.wav (locuteur « OA ») est rattaché à l'actrice OAF,
     ce qui supprime le faux troisième locuteur de TESS.

Dossiers attendus dans le dossier de données (majuscules/tirets sans importance) :
    data/RAVDESS/   data/CREMAD/ (ou CREMA-D)   data/TESS/   data/SAVEE/   data/EMODB/
"""
import glob
import os
import re
from collections import namedtuple

Entry = namedtuple("Entry", "key path dataset speaker label")

DEFAULT_DATASETS = ["RAVDESS", "CREMAD"]   # TESS et SAVEE : réservés au test externe / à l'interface
RAVDESS_MAP = {"01": "neutral", "02": "neutral", "03": "happy", "04": "sad",
               "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}
CREMAD_MAP = {"ANG": "angry", "DIS": "disgust", "FEA": "fearful",
              "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
TESS_MAP = {"angry": "angry", "disgust": "disgust", "fear": "fearful",
            "happy": "happy", "neutral": "neutral", "sad": "sad"}
SAVEE_MAP = {"a": "angry", "d": "disgust", "f": "fearful", "h": "happy",
             "n": "neutral", "sa": "sad", "su": "surprised"}
EMODB_MAP = {"W": "angry", "E": "disgust", "A": "fearful", "F": "happy",
             "T": "sad", "N": "neutral"}          # "L" (ennui) n'a pas d'équivalent -> ignoré


# Chaque parseur reçoit le nom de fichier sans extension et renvoie (locuteur, émotion) ou None.
def parse_ravdess(stem):                      # 03-01-06-01-02-01-12 : 3e champ = émotion, dernier = acteur
    p = stem.split("-")
    if len(p) != 7 or p[2] not in RAVDESS_MAP:
        return None
    return f"{int(p[6]):02d}", RAVDESS_MAP[p[2]]


def parse_cremad(stem):                       # 1001_DFA_ANG_XX : 1er champ = acteur, 3e = émotion
    p = stem.split("_")
    if len(p) < 3 or p[2].upper() not in CREMAD_MAP:
        return None
    return p[0], CREMAD_MAP[p[2].upper()]


def parse_tess(stem):                         # OAF_back_angry : OAF/YAF = actrice, dernier = émotion
    s = stem.lower()
    p = s.split("_")
    if len(p) < 3:
        return None
    spk = "OAF" if p[0] == "oa" else p[0].upper()      # CORRECTION : OA_bite_neutral = actrice OAF
    if "surprise" in s or p[-1] == "ps":
        return spk, "surprised"
    lab = TESS_MAP.get(p[-1])
    return (spk, lab) if lab else None


def parse_savee(stem):                        # DC_sa05 : DC = locuteur, "sa" = tristesse
    m = re.match(r"^([A-Za-z]{2})_(sa|su|a|d|f|h|n)\d+$", stem, re.I)
    if not m:
        return None
    return m.group(1).upper(), SAVEE_MAP[m.group(2).lower()]


def parse_emodb(stem):                        # 03a01Fa : 03 = locuteur, 6e caractère = émotion
    if len(stem) < 7 or not stem[:2].isdigit():
        return None
    lab = EMODB_MAP.get(stem[5].upper())
    return (stem[:2], lab) if lab else None


PARSERS = {"RAVDESS": parse_ravdess, "CREMAD": parse_cremad, "TESS": parse_tess,
           "SAVEE": parse_savee, "EMODB": parse_emodb}


def normalize(name):
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def find_dataset_folders(root):
    found = {}
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            full = os.path.join(root, name)
            if os.path.isdir(full) and normalize(name) in PARSERS:
                found[normalize(name)] = full
    return found


def collect_entries(root, wanted=None):
    """Liste des fichiers utilisables : Entry(key, path, dataset, speaker, label).
    RAVDESS est toujours inclus : ses acteurs "de test" servent d'évaluation commune."""
    folders = find_dataset_folders(root)
    wanted = [normalize(w) for w in (wanted or DEFAULT_DATASETS)]
    for w in wanted:
        if w not in PARSERS:
            raise SystemExit(f"Base inconnue : {w}. Bases possibles : {', '.join(PARSERS)}")
    if "RAVDESS" not in wanted:
        wanted.insert(0, "RAVDESS")
    entries, seen = [], set()
    for ds in wanted:
        if ds not in folders:
            print(f"  {ds:8s}: dossier introuvable dans '{root}' -> ignoré")
            continue
        before = len(entries)
        for f in sorted(glob.glob(os.path.join(folders[ds], "**", "*.wav"), recursive=True)):
            parsed = PARSERS[ds](os.path.splitext(os.path.basename(f))[0])
            key = f"{ds}/{os.path.basename(f)}"
            if parsed is None or key in seen:
                continue
            seen.add(key)
            entries.append(Entry(key, f, ds, parsed[0], parsed[1]))
        print(f"  {ds:8s}: {len(entries) - before} fichiers utilisables")
    if not any(e.dataset == "RAVDESS" for e in entries):
        raise SystemExit(f"RAVDESS est introuvable dans '{root}'. Placez les bases dans "
                         f"{root}/RAVDESS, {root}/CREMAD, ... et lancez avec --data {root}")
    return entries
