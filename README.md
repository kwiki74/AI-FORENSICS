# AI-FORENSICS

Pipeline d'analyse forensique de médias et de détection de campagnes de désinformation coordonnées.

## Vue d'ensemble

Le pipeline ingère des contenus (images, vidéos, textes) issus de réseaux sociaux, les analyse via plusieurs workers spécialisés, et expose les résultats dans une interface web interactive.

```
DATA_IN (JSON scrappés)
    │
    ▼
[WORKER IMPORT] ──► MongoDB (influence_detection)
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
        [DEEPFAKE]     [NLP]    [NETWORK]
              │          │          │
              └──────────┴──────────┘
                         │
                    ▼         ▼
                MongoDB     Neo4j
                    │         │
                    └────┬────┘
                         ▼
                  [WWW — Streamlit]
```

### Workers

| Worker | Rôle |
|--------|------|
| **IMPORT** | Ingestion des JSON scrappés vers MongoDB |
| **DETECT_AI_PIPELINE** | Détection deepfake (SwinV2 + Synthbuster + ViT SDXL) |
| **NLP** | Sentiment, embeddings, clustering narratif (HDBSCAN) |
| **NETWORK** | ETL MongoDB → Neo4j, détection de campagnes coordonnées |
| **WWW** | Interface Streamlit de visualisation et d'exploration |

---

## Prérequis

- Ubuntu 22.04 / 24.04 LTS
- Anaconda (Python 3.11)
- MongoDB 8.0 (mode ReplicaSet)
- Neo4j (+ plugin GDS optionnel)
- 16 Go RAM recommandés (30 Go si NLP + deepfake en parallèle)
- GPU NVIDIA CUDA 12.x optionnel (CPU fonctionne)

---

## Installation

Voir [INSTALL_v3.2.md](INSTALL_v3.2.md) pour le guide complet.

### Résumé rapide

```bash
# 1. Créer l'environnement conda
conda create -n forensics python=3.11 -y
conda activate forensics
conda install pytorch==2.6.0 torchvision cpuonly -c pytorch -y

# 2. Installer les dépendances Python
pip install -r requirements.txt

# 3. Configurer les credentials
cp .env.example .env
nano .env   # renseigner les mots de passe MongoDB et Neo4j

# 4. Créer les index MongoDB
cd ~/AI-FORENSICS
python -c "from SCHEMA.schema import get_db, create_indexes; create_indexes(get_db()); print('OK')"

# 5. Initialiser le submodule Synthbuster
git submodule update --init --recursive
```

---

## Configuration

Copier `.env.example` en `.env` et renseigner les valeurs :

```env
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_USER=influence_app
MONGO_PASSWORD=...
MONGO_DB=influence_detection

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
```

> Le fichier `.env` ne doit jamais être commité dans Git.

Chaque worker possède également son propre fichier `.cfg` dans son dossier.

---

## Modèles requis

| Modèle | Emplacement | Source |
|--------|-------------|--------|
| SwinV2 OpenFake (fine-tuné) | `WORKER/DETECT_AI_PIPLINE/swinv2_openfake/` | Fine-tuning local (`fine_tune_swinv2.py`) |
| Synthbuster | `WORKER/DETECT_AI_PIPLINE/synthbuster/models/` | Submodule Git + fichiers `.joblib` |
| ViT SDXL (`Organika/sdxl-detector`) | Cache HuggingFace | Téléchargement automatique |

> Les fichiers `.pt`, `.safetensors`, `.joblib` ne sont pas versionnés (trop lourds). Les transférer manuellement depuis l'ancienne machine via `scp`.

---

## Lancement

### Avec supervisord (recommandé)

```bash
conda activate forensics
cd ~/AI-FORENSICS/SUPERVISOR
supervisord -c supervisord.conf

# Vérifier l'état des workers
supervisorctl -c supervisord.conf status
```

### Worker deepfake en one-shot

```bash
conda activate forensics
cd ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE
CUDA_VISIBLE_DEVICES="" python detect_ai_pipeline-v4.0.3.py \
    ~/AI-FORENSICS/DATA_IN \
    --ensemble --workers 2 --verbose \
    --output RESULT/results.csv
```

### Interface web

```bash
conda activate forensics
cd ~/AI-FORENSICS/WWW
streamlit run forensics_explorer.py
```

---

## Structure du projet

```
AI-FORENSICS/
├── CONTEXT/            # Documentation et contexte projet
├── DATA_IN/            # JSON scrappés (inbox worker import)
├── SCHEMA/             # Schéma MongoDB et fonctions patch
├── SUPERVISOR/         # Configuration supervisord + scripts terminal
├── WORKER/
│   ├── DETECT_AI_PIPLINE/  # Détection deepfake (3 modèles en ensemble)
│   ├── IMPORT/             # Ingestion JSON → MongoDB
│   ├── NLP/                # Analyse NLP (sentiment, embeddings, clustering)
│   └── NETWORK/            # Graphe d'influence (Neo4j + détection campagnes)
├── WWW/                # Interface Streamlit
├── requirements.txt    # Dépendances Python unifiées
├── .env.example        # Template de configuration (sans secrets)
└── INSTALL_v3.2.md     # Guide d'installation complet
```

---

## Cloner sur une nouvelle machine

```bash
# Cloner avec le submodule Synthbuster
git clone --recurse-submodules https://github.com/kwiki74/AI-FORENSICS.git
cd AI-FORENSICS

# Ou si déjà cloné sans --recurse-submodules
git submodule update --init --recursive
```

---

## Collaborateurs

- Aquilina Julien
