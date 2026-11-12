# Reconnaissance automatique des émotions vocales

## Structure
```
projet/
├── data/                  bases audio (non versionnées)
│   ├── RAVDESS/  CREMAD/  TESS/  SAVEE/
├── models/                modèles et rapports générés par l'entraînement
│   ├── ser_model.joblib        (MFCC + SVM/MLP)
│   ├── ser_w2v.joblib          (WavLM + régression logistique)
│   └── report*.txt, confusion_matrix*.png
├── training/              partie IA (Python)
│   ├── emotion_datasets.py  features.py  w2v_features.py
│   ├── train.py  train_w2v.py
│   └── requirements.txt  requirements_w2v.txt
├── backend/               API + serveur de l'interface
│   ├── server.py
│   └── requirements.txt
├── frontend/              interface (HTML / CSS / JS)
│   ├── index.html  style.css  script.js
└── README.md
```

## Installation (une fois)
```
pip install -r training/requirements.txt -r training/requirements_w2v.txt -r backend/requirements.txt
```
(gradio n'est plus nécessaire : vous pouvez le retirer de training/requirements.txt.)

## Entraînement (depuis la racine du projet)
```
python training/train.py --data data/RAVDESS --out models
python training/train_w2v.py --data data --out models
```

## Lancer l'application
```
python backend/server.py
```
Puis ouvrir http://localhost:5000
