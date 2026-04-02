# AI Forensics Pipeline v4.0

Détection d'images et vidéos générées par IA, optimisée pour les générateurs dominants sur les réseaux sociaux en 2025-2026. Composant d'analyse du projet de détection de campagnes d'influence.

---

## Sommaire

1. [Quoi de neuf en v4.0](#quoi-de-neuf-en-v40)
2. [Modèles — justification du choix](#modèles--justification-du-choix)
3. [Prérequis système](#prérequis-système)
4. [Installation](#installation)
5. [Installation de Synthbuster](#installation-de-synthbuster)
6. [SwinV2 OpenFake — fine-tuning local](#swinv2-openfake--fine-tuning-local)
7. [GPU — cas particulier Blackwell (RTX 50xx)](#gpu--cas-particulier-blackwell-rtx-50xx)
8. [Intégration dans le projet](#intégration-dans-le-projet)
9. [Utilisation rapide](#utilisation-rapide)
10. [Options en ligne de commande](#options-en-ligne-de-commande)
11. [Fichier de configuration](#fichier-de-configuration)
12. [Calibration bi-dossier FP-first](#calibration-bi-dossier-fp-first)
13. [Interprétation de la sortie de calibration](#interprétation-de-la-sortie-de-calibration)
14. [Parallélisme — option --workers](#parallélisme--option---workers)
15. [Extraction adaptative des frames vidéo](#extraction-adaptative-des-frames-vidéo)
16. [Interprétation des résultats CSV](#interprétation-des-résultats-csv)
17. [FAQ / Dépannage](#faq--dépannage)
18. [Historique des versions](#historique-des-versions)

---

## Quoi de neuf en v4.0

Refonte complète de la sélection des modèles. Les 5 modèles HuggingFace de la v3.x ont été remplacés par **3 modèles orthogonaux** couvrant les générateurs réellement utilisés sur les RS en 2025-2026.

| Retiré (v3.x) | Raison |
|---|---|
| `prithivMLmods/Deep-Fake-Detector-v2-Model` | Poids 0.00 après calibration — entraîné face-swap uniquement |
| `umm-maybe/AI-image-detector` | Spécialisé contenu artistique, inutile sur photos RS |
| `dima806/ai_vs_real_image_detection` | Trop biaisé (moy.REAL=0.71) |
| `Ateeqq/ai-vs-human-image-detector` | Remplacé par SwinV2 OpenFake, bien supérieur sur générateurs 2025 |

| Ajouté (v4.0) | Raison |
|---|---|
| `microsoft/swinv2-openfake` | Seul modèle public entraîné sur FLUX 1.1-pro, MJ v6, DALL-E 3, Grok-2 |

---

## Modèles — justification du choix

La logique de sélection est **l'orthogonalité** : chaque modèle détecte via un mécanisme différent, couvrant des générateurs différents. La calibration FP-first pondère ensuite selon la performance réelle sur votre dataset.

| # | Modèle | Mécanisme | Générateurs couverts | RAM |
|---|---|---|---|---|
| 1 | `Organika/sdxl-detector` | CNN diffusion (ViT fine-tuné Wikimedia/SDXL) | SD 1.5/2.1/XL, FLUX partiel | ~2 Go |
| 2 | `microsoft/swinv2-openfake` | SwinV2-Small fine-tuné OpenFake | FLUX 1.0/1.1-pro, MJ v6, DALL-E 3, Grok-2, Ideogram 3, SDXL LoRA/FT | ~1.5 Go |
| 3 | `synthbuster/synthbuster` | Fourier fréquentiel (sklearn) | Artefacts spectraux tous générateurs de diffusion | <100 Mo |

**Slot 4 optionnel** : prévu dans le `.cfg` pour ajouter un modèle custom après fine-tuning. Désactivé par défaut (RAM < 24 Go recommandé).

### Pourquoi SwinV2 OpenFake est critique

FLUX Dev, DALL-E 3, et Midjourney v7 atteignent seulement 18-30% de précision de détection sur les méthodes classiques. Le SwinV2 entraîné sur le dataset OpenFake (McGill/Mila, arxiv 2509.09495) atteint F1=0.99 in-distribution et la meilleure généralisation hors-distribution disponible publiquement — c'est le seul modèle qui "voit" ces générateurs de façon documentée.

**Note importante** : ComplexDataLab n'a pas encore publié les poids fine-tunés sur HuggingFace. Le pipeline bascule automatiquement sur le backbone de base non fine-tuné. Pour des performances optimales, lancer `fine_tune_swinv2.py` sur votre dataset REAL/ALT.

---

## Prérequis système

| Composant | Minimum | Recommandé |
|---|---|---|
| OS | Ubuntu 22.04 / 24.04 | Ubuntu 24.04 LTS |
| Python | 3.10 | 3.11 |
| RAM | 8 Go | 16 Go (30 Go si `--workers 2`) |
| Disque | 8 Go libres | 15 Go (modèles + médias) |
| GPU | *(optionnel)* | NVIDIA CUDA 12.x |

---

## Installation

### Étape 1 — Dépendances système

```bash
sudo apt update
sudo apt install -y ffmpeg git
ffmpeg -version
```

### Étape 2 — Environnement Conda (recommandé)

```bash
conda create -n forensics python=3.11 -y
conda activate forensics

# CPU (VM sans GPU)
conda install pytorch==2.6.0 torchvision==0.21.0 cpuonly -c pytorch -y

# GPU CUDA 12.6
conda install pytorch==2.6.0 torchvision==0.21.0 pytorch-cuda=12.6 -c pytorch -c nvidia -y
```

### Étape 3 — Dépendances Python

```bash
conda activate forensics
pip install -r requirements.txt
```

### Étape 4 — Vérification

```bash
python -c "
import torch, cv2, transformers, PIL, pandas, tqdm, sklearn, joblib, numba, imageio, timm
print('torch       :', torch.__version__)
print('transformers:', transformers.__version__)
print('CUDA dispo  :', torch.cuda.is_available())
print('OK ✓')
"
```

---

## Installation de Synthbuster

```bash
cd /chemin/vers/le/script
git clone https://github.com/qbammey/synthbuster synthbuster

# Créer les liens symboliques vers le modèle JPEG
cd synthbuster/models/
ln -s model_jpeg.joblib model.joblib
ln -s config_jpeg.json config.json
```

L'arborescence attendue :

```
ton_projet/
    detect_ai_pipeline-v4.0.py
    fine_tune_swinv2.py
    ai_forensics.cfg
    requirements.txt
    synthbuster/
        inference_common.py
        preprocess.py
        models/
            model.joblib       ← lien → model_jpeg.joblib
            config.json        ← lien → config_jpeg.json
            model_jpeg.joblib
            config_jpeg.json
            model_uncompressed.joblib
            config_uncompressed.json
```

---

## SwinV2 OpenFake — fine-tuning local

En attendant la publication officielle des poids par ComplexDataLab, fine-tuner le backbone sur votre dataset REAL/ALT :

```bash
# Fine-tuning (10 epochs, ~30-60 min sur CPU, ~5 min sur GPU)
python fine_tune_swinv2.py \
    --data-dir ./calib_dataset \
    --output-dir ./swinv2_openfake \
    --epochs 10 \
    --batch-size 16 \
    --verbose

# Mettre à jour le .cfg pour pointer vers le modèle local
# [models]
# default_models = Organika/sdxl-detector, ./swinv2_openfake, synthbuster/synthbuster
```

Le script applique la méthodologie OpenFake exacte : deux flux d'augmentation (géométrique/photométrique sur tout, dégradation JPEG légère sur les fakes uniquement) pour éviter que le modèle apprenne des raccourcis de compression.

Évaluation d'un modèle existant :

```bash
python fine_tune_swinv2.py --eval-only \
    --model-dir ./swinv2_openfake \
    --data-dir ./calib_dataset
```

---

## GPU — cas particulier Blackwell (RTX 50xx)

Les RTX 50xx (architecture Blackwell, sm_120) ne sont pas supportées par PyTorch 2.6.0 stable.

### Solution rapide — forcer le CPU

```bash
CUDA_VISIBLE_DEVICES="" python detect_ai_pipeline-v4.0.py \
    ./calib_dataset --calibrate --workers 2 --verbose
```

### Solution permanente — nightly cu128

```bash
conda create --name forensics_nightly --clone forensics
conda activate forensics_nightly
python -m pip uninstall torch torchvision -y
python -m pip install --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu128
```

En cas de problème, supprimer le clone sans toucher à `forensics` :

```bash
conda env remove --name forensics_nightly
```

---

## Intégration dans le projet

```
Scrapper → MongoDB (Change Streams) → Worker deepfake (ce script)
                                    → MongoDB enrichi (champ deepfake.*)
                                    → ETL → Neo4j + Elasticsearch
```

```python
from schema import get_db, patch_post_deepfake, patch_media_deepfake

result = {
    "final_score": 0.87,
    "prediction": "synthetic",
    "model_divergence": 0.09,
    "jpeg_artifact_score": 0.41,
    "score_sdxl-detector": 0.78,
    "score_swinv2-openfake": 0.91,
    "score_synthbuster": 0.34,
}
db = get_db()
db.posts.update_one({"_id": post_id}, patch_post_deepfake(result))
```

---

## Utilisation rapide

```bash
# Analyse standard
python detect_ai_pipeline-v4.0.py ./data --ensemble --workers 2 --verbose

# Calibration (après avoir préparé calib_dataset avec prepare_calib_dataset.py)
CUDA_VISIBLE_DEVICES="" python detect_ai_pipeline-v4.0.py \
    ./calib_dataset --calibrate --workers 2 --verbose

# Sans GPU
CUDA_VISIBLE_DEVICES="" python detect_ai_pipeline-v4.0.py ./data --ensemble --workers 2

# Sans Synthbuster (si non installé)
python detect_ai_pipeline-v4.0.py ./data --no-synthbuster --ensemble

# Diagnostic rapide — mode fast (sdxl-detector uniquement)
python detect_ai_pipeline-v4.0.py ./data --mode fast --verbose
```

---

## Options en ligne de commande

| Option | Défaut | Description |
|---|---|---|
| `folder` | *(obligatoire)* | Dossier à analyser. En mode `--calibrate` : contient `REAL/` et `ALT/` |
| `--config` | `ai_forensics.cfg` | Fichier de configuration |
| `--output` | `results.csv` | CSV de sortie (préfixe horodaté automatique) |
| `--json-output` | *(désactivé)* | JSON de sortie optionnel |
| `--models` | *(voir config)* | Liste de modèles (surcharge le `.cfg`) |
| `--fps` | `1` | Frames/s extraites des vidéos (désactive l'adaptatif) |
| `--max-frames` | *(illimité)* | Plafond de frames par vidéo |
| `--threshold-high` | `0.82` | Score → `synthetic` |
| `--threshold-low` | `0.65` | Score → `suspicious` |
| `--mode` | `balanced` | `fast` (sdxl seul) / `balanced` / `accurate` |
| `--workers` | `2` | Processus parallèles |
| `--no-adaptive` | *(flag)* | Désactive l'extraction adaptative, utilise `--fps` |
| `--ensemble` | *(flag)* | Active la fusion pondérée |
| `--no-ensemble` | *(flag)* | Désactive la fusion pondérée |
| `--require-face` | *(flag)* | Analyse uniquement les frames avec visage |
| `--no-bias-correction` | *(flag)* | Scores bruts sans correction |
| `--no-synthbuster` | *(flag)* | Désactive Synthbuster |
| `--synthbuster-dir` | `./synthbuster/` | Chemin vers le repo Synthbuster |
| `--calibrate` | *(flag)* | Mode calibration bi-dossier |
| `--skip-errors` | *(flag)* | Continue en cas d'erreur sur un fichier |
| `--verbose` | *(flag)* | Logs INFO en console |
| `--debug` | *(flag)* | Logs DEBUG en console |

---

## Fichier de configuration

### Section `[behaviour]` — paramètres notables

| Paramètre | Défaut | Description |
|---|---|---|
| `mode` | `balanced` | Preset de modèles |
| `ensemble` | `true` | Fusion pondérée par défaut |
| `bias_correction` | `true` | Correction de biais activée |
| `workers` | `2` | Workers parallèles (3 modèles × ~1.5 Go ≈ 4.5 Go par worker) |
| `divergence_alert_threshold` | `0.20` | Seuil alerte divergence |
| `synthbuster_dir` | *(vide)* | Chemin Synthbuster (vide = `./synthbuster/`) |

### Section `[video]`

| Paramètre | Défaut | Description |
|---|---|---|
| `adaptive_frames` | `true` | Extraction adaptative selon la durée |
| `adaptive_tiers` | `5:3, 15:5, 60:8, 180:12, inf:16` | Paliers durée:frames |
| `max_frames` | *(vide)* | Plafond absolu (vide = illimité) |

---

## Calibration bi-dossier FP-first

```bash
CUDA_VISIBLE_DEVICES="" python detect_ai_pipeline-v4.0.py ./calib_dataset \
    --calibrate --mode accurate --workers 2 --verbose \
    --json-output calib_report.json
```

Structure attendue :

```
calib_dataset/
    REAL/    ← images et vidéos authentiques
    ALT/     ← images et vidéos générées ou altérées
```

### Critère FP-first

```
fp_penalty    = avg(score_brut sur REAL)
fp_factor     = (1 − fp_penalty)²
detect_margin = avg(ALT) − avg(REAL)
poids_brut    = max(0, detect_margin) × fp_factor
poids         = poids_brut / Σ poids_bruts  (normalisé)
```

Un modèle qui score haut sur le contenu réel reçoit une pénalité quadratique — l'objectif est de minimiser les faux positifs avant tout.

### Datasets recommandés pour calib_dataset

Pour couvrir les générateurs 2025, enrichir le dossier `ALT/` avec :

| Dataset | Générateurs couverts | Lien |
|---|---|---|
| OpenFake | FLUX, MJ v6, DALL-E 3, Grok-2, Ideogram 3 | https://huggingface.co/datasets/ComplexDataLab/OpenFake |
| real-vs-fake (Kaggle) | StyleGAN2 | https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces |
| ArtiFact | AFHQ, BigGAN, LSUN | https://www.kaggle.com/datasets/awsaf49/artifact-dataset |

```bash
python prepare_calib_dataset.py \
    --ff ./donnees/ff \
    --celebdf ./donnees/celebdf \
    --artifact ./donnees/artifact \
    --real-vs-fake ./donnees/real_vs_fake \
    --custom-alt ./donnees/openfake/synthetic \
    --output ./calib_dataset \
    --max-real 2000 --max-alt 2000 --balance --link --verbose
```

---

## Interprétation de la sortie de calibration

| Colonne | Signification |
|---|---|
| `moy.REAL` | Score brut moyen sur contenu authentique — doit être bas (< 0.3 idéal) |
| `moy.ALT` | Score brut moyen sur contenu généré — doit être bien supérieur à moy.REAL |
| `marge` | `moy.ALT − moy.REAL` — séparation nette ; négatif = modèle éliminé |
| `fp²` | Facteur de pénalité FP : `(1 − moy.REAL)²` |
| `Poids` | Poids final normalisé écrit dans le `.cfg` |

Un poids nul n'est pas un bug — c'est le critère FP-first qui élimine un modèle inutile sur votre dataset. Voir la section SwinV2 si ses scores sont incohérents (backbone non fine-tuné → scores non interprétables → calibrer après fine-tuning).

---

## Parallélisme — option `--workers`

Avec 3 modèles (~4.5 Go par worker au lieu de ~9 Go en v3.x) :

| Workers | RAM requise | Recommandé pour |
|---|---|---|
| 1 | ~4.5 Go | Machines limitées |
| 2 | ~9 Go | Configuration standard (défaut) |
| 3 | ~13.5 Go | ≥ 20 Go RAM disponibles |
| 4 | ~18 Go | ≥ 24 Go RAM disponibles |

Configurable dans le `.cfg` :

```ini
[behaviour]
workers = 2
```

---

## Extraction adaptative des frames vidéo

Activée par défaut. `ffprobe` mesure la durée de chaque vidéo et calcule automatiquement le nombre de frames à extraire selon des paliers :

| Durée | Frames | Cas typique |
|---|---|---|
| < 5 s | 3 | Story, GIF |
| 5–15 s | 5 | Reels, TikTok courts |
| 15–60 s | 8 | Vidéos moyennes |
| 1–3 min | 12 | Vidéos longues |
| > 3 min | 16 | YouTube, interviews |

Désactiver si besoin :

```bash
python detect_ai_pipeline-v4.0.py ./data --no-adaptive --fps 1 --ensemble
```

---

## Interprétation des résultats CSV

### Colonnes images

| Colonne | Description |
|---|---|
| `source_type` | `image` |
| `source` | Nom du fichier |
| `face_found` | Visage détecté |
| `jpeg_artifact_score` | Score artefact JPEG [0–1] |
| `final_score` | Score final fusionné [0–1] |
| `model_divergence` | Écart-type inter-modèles (fiabilité du résultat) |
| `prediction` | `likely_real` / `suspicious` / `synthetic` |
| `score_sdxl-detector` | Score corrigé du modèle 1 |
| `score_swinv2-openfake` | Score corrigé du modèle 2 |
| `score_synthbuster` | Score corrigé du modèle 3 |
| `raw_sdxl-detector` | Score brut du modèle 1 (si correction biais active) |
| `raw_swinv2-openfake` | Score brut du modèle 2 |
| `raw_synthbuster` | Score brut du modèle 3 |

### Colonnes supplémentaires vidéos

| Colonne | Description |
|---|---|
| `duration_sec` | Durée en secondes (via ffprobe) |
| `frames_analyzed` | Frames effectivement analysées |
| `faces_detected` | Frames avec visage détecté |
| `avg_jpeg_artifact_score` | Score artefact JPEG moyen |

### Étiquettes de prédiction

| Étiquette | Condition |
|---|---|
| `likely_real` | `final_score ≤ threshold_low (0.65)` |
| `suspicious` | `threshold_low < score ≤ threshold_high (0.82)` |
| `synthetic` | `final_score > threshold_high (0.82)` |

### Utiliser `model_divergence`

Une divergence élevée (> 0.20) signifie que les modèles ne s'accordent pas — le résultat est peu fiable et mérite une vérification manuelle. Par exemple : sdxl-detector score 0.9 mais swinv2 score 0.1 et synthbuster score 0.5 → divergence ~0.33, résultat à vérifier.

---

## FAQ / Dépannage

**SwinV2 score 0.5 sur tout**
Normal si le backbone de base (non fine-tuné) est chargé. Les classes ImageNet ne sont pas interprétables comme real/fake → le wrapper retourne 0.5 (neutre). Lancer `fine_tune_swinv2.py` pour obtenir un modèle utilisable.

**SwinV2 poids zéro après calibration**
Même cause — backbone non fine-tuné, scores neutres, marge nulle. Fine-tuner d'abord, puis recalibrer.

**CUDA not compatible — RTX 50xx**
Voir section [GPU Blackwell](#gpu--cas-particulier-blackwell-rtx-50xx). Solution immédiate : `CUDA_VISIBLE_DEVICES=""`.

**Les modèles HuggingFace ne se chargent pas**
Vérifier la connexion Internet. Téléchargement automatique au premier lancement (quelques Go).

**`ffmpeg` introuvable**
`sudo apt install ffmpeg` — les images ne sont pas affectées.

**Synthbuster score 1.0 sur tout**
Vérifier les liens symboliques dans `synthbuster/models/` : `ls -la synthbuster/models/ | grep model.joblib`. Si absents : `ln -s model_jpeg.joblib model.joblib`.

**Analyse très lente avec 1 worker**
Comportement normal — passer à `--workers 2` ou `--workers 3`. Avec 3 modèles la consommation RAM est réduite vs v3.x.

---

## Historique des versions

| Version | Changements principaux |
|---|---|
| v4.0 | Refonte modèles : 3 modèles orthogonaux (sdxl-detector + SwinV2 OpenFake + Synthbuster), script fine_tune_swinv2.py, CSV colonnes nettoyées |
| v3.5.2 | Extraction adaptative frames vidéo (`adaptive_frames`, `adaptive_tiers`), colonne `duration_sec` |
| v3.5.1 | Option `--workers N` pour traitement parallèle |
| v3.5.0 | Intégration Synthbuster, `jpeg_artifact_score`, `--no-synthbuster` |
| v3.4.2 | Calibration bi-dossier FP-first, comparaison avant/après |
| v3.4.1 | Logs enrichis, colonne `model_divergence` |
| v3.4 | Fichier `.cfg`, README, logs dans dossier dédié |
| v3.3 | Correction de biais, seuils remontés |
| v3.0 | Support multi-modèles et analyse vidéo |
