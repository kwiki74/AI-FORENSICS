# Contexte projet — AI Forensics Pipeline
# À lire au début du prochain chat pour reprendre le projet

---

## Projet

Détection de deepfakes et de médias générés par IA dans le cadre d'un projet de **détection de campagnes d'influence sur les réseaux sociaux** (Instagram, TikTok, Twitter/X, Telegram).

Deux personnes : toi (pipeline deepfake + MongoDB + infra) et un binôme (scrapping + interface graphique de configuration des collections).

---

## Architecture globale

```
Interface config (binôme)
    ↓
Scrapper (binôme) → DATA_IN/ (JSON + médias bruts)
    ↓
worker_import.py → MongoDB (influence_detection)
    ↓  Change Streams
    ├── Worker deepfake  ← WORKER/DETECT_AI_PIPLINE/detect_ai_pipeline-v4.0.py
    ├── Worker NLP       ← WORKER/NLP/nlp_worker.py
    └── Worker réseau    ← WORKER/NETWORK/network_worker.py
                                  ↓
                             Neo4j (via neo4j_client.py + campaign_detector.py)
    ↓  ETL Change Streams
    └── Elasticsearch    (full-text, dashboards Kibana — optionnel)
```

**MongoDB** = source de vérité (base opérationnelle).
**Neo4j** = analyse des relations et détection de campagnes coordonnées.

---

## Structure des dossiers (racine : `~/AI-FORENSICS/`)

```
~/AI-FORENSICS/
├── CONTEXT/
│   └── CONTEXT_NEXT_CHAT.md
├── DATA_IN/                              ← inbox scrapper
│   ├── converted_INSTAGRAM_crypto_2026-03-17/
│   │   ├── coincryptofr/
│   │   ├── cryptoastmedia/
│   │   ├── cryptocomofficial/
│   │   ├── cryptoedgeofficiel/
│   │   └── instacoin.crypto/
│   ├── converted_TELEGRAM_crypto_2026-03-17/
│   │   └── cryptoast_fr/
│   └── converted_TIKTOK_crypto_2026-03-17/
│       └── agama.club/ ...
├── logs/                                 ← logs centralisés
│   ├── detect_ai_pipeline_20260322.log
│   ├── errors.log
│   ├── nlp_worker.log
│   ├── warnings.log
│   └── worker_import.log
├── log.txt
├── schema.py                             ← schéma MongoDB v3 (référence)
├── SUPERVISOR/                           ← workspace de supervision
│   ├── supervisord.conf
│   ├── launch_workspace.sh
│   ├── terminator_layout.conf
│   └── INSTALL_WORKSPACE.md
├── WORKER/
│   ├── DETECT_AI_PIPLINE/
│   │   ├── detect_ai_pipeline-v4.0.py   ← version courante
│   │   ├── detect_ai_pipeline-v4.0.1.py
│   │   ├── detect_ai_pipeline-v4.0.2.py
│   │   ├── ai_forensics.cfg
│   │   ├── calib_report_v4.json         ← calibration v4 déjà faite ✅
│   │   ├── fine_tune_swinv2.py
│   │   ├── swinv2_openfake/
│   │   │   └── model.safetensors ✅
│   │   ├── synthbuster/
│   │   │   └── models/
│   │   │       ├── model.joblib → model_jpeg.joblib ✅
│   │   │       └── config.json  → config_jpeg.json  ✅
│   │   └── logs/
│   ├── NETWORK/                         ← worker réseau ✅ EXISTE
│   │   ├── network_worker.py
│   │   ├── neo4j_client.py
│   │   ├── campaign_detector.py
│   │   ├── network_pipeline.cfg
│   │   ├── network-worker.service       ← systemd NON UTILISÉ (supervisord)
│   │   ├── schema.py
│   │   └── logs/network_worker.log
│   └── NLP/                             ← worker NLP ✅ EXISTE
│       ├── nlp_worker.py
│       ├── embeddings.py
│       ├── sentiment.py
│       ├── narrative_clustering.py
│       ├── nlp_pipeline.cfg
│       ├── nlp-worker.service           ← systemd NON UTILISÉ (supervisord)
│       ├── schema.py
│       └── requirements_nlp.txt
├── worker_import.py                     ← worker import ✅ EXISTE (racine)
├── worker_import.cfg
└── worker_import.cfg.bak
```

---

## État actuel du projet (mars 2026)

### ✅ Terminé / Existe

**Pipeline deepfake v4.0** (`WORKER/DETECT_AI_PIPLINE/detect_ai_pipeline-v4.0.py`)
- 3 modèles orthogonaux, calibration FP-first, workers parallèles
- Extraction adaptative frames vidéo, score artefact JPEG, logs rotatifs 7j
- `calib_report_v4.json` présent → calibration v4 déjà lancée

**SwinV2 fine-tuné** — `model.safetensors` présent, F1=0.943, MCC=0.886

**MongoDB 8.0** — ReplicaSet `rs0`, auth activée, schema v3 (7 collections)

**Synthbuster** — liens symboliques en place (`model.joblib → model_jpeg.joblib`)

**Tous les workers codés**
- `worker_import.py` + `worker_import.cfg` (racine)
- `WORKER/NLP/nlp_worker.py` + `embeddings.py` + `sentiment.py` + `narrative_clustering.py`
- `WORKER/NETWORK/network_worker.py` + `neo4j_client.py` + `campaign_detector.py`

**Workspace de supervision** — 4 fichiers dans `SUPERVISOR/` ⚠ chemins à corriger

**Données de test** — `DATA_IN/` contient des données réelles Instagram/TikTok/Telegram

### 🔄 À vérifier / corriger

- **`supervisord.conf`** — chemins générés avec `~/Scripts/4.0/`, à corriger vers `~/AI-FORENSICS/` (voir section Workspace ci-dessous)
- **`ai_forensics.cfg`** — vérifier que les poids/biais ont été mis à jour après `calib_report_v4.json`
- **État fonctionnel des workers** NLP, NETWORK, Import — codés mais testés ?
- **`schema.py` dupliqué** — 3 copies (racine, NLP/, NETWORK/) : vérifier synchronisation

### ❌ Pas encore fait

- Env conda `nlp_pipeline` à créer (dépendances dans `WORKER/NLP/requirements_nlp.txt`)
- Réentraîner Synthbuster dans `forensics_nightly` (warning sklearn 1.7.1 vs 1.8.0)
- Enrichir `DATASET/calib_dataset/ALT/` avec OpenFake (FLUX 1.1-pro, MJ v7, GPT Image 1)

---

## Modèles v4.0

| # | Modèle | Mécanisme | Générateurs couverts |
|---|---|---|---|
| 1 | `Organika/sdxl-detector` | CNN diffusion | SD/SDXL, FLUX partiel |
| 2 | `./swinv2_openfake` | SwinV2 fine-tuné | FLUX, MJ v6, DALL-E 3, Grok-2, Ideogram 3 |
| 3 | `synthbuster/synthbuster` | Fourier/sklearn | Artefacts spectraux |

Poids/biais : voir `WORKER/DETECT_AI_PIPLINE/ai_forensics.cfg` (mis à jour post-calib v4).

---

## Infrastructure technique

**Machine** — Ubuntu 24.04, 20 cœurs, 30 Go RAM, RTX 5070 Laptop (8 Go VRAM, sm_120 Blackwell)

**Environnements conda**

| Env | Usage |
|---|---|
| `forensics` | Deepfake + worker_import, CPU uniquement (PyTorch 2.6.0 stable) |
| `forensics_nightly` | GPU RTX 5070, fine-tuning (nightly cu128) |
| `nlp_pipeline` | Worker NLP + Worker réseau — **à créer/vérifier** |

---

## Workspace de supervision

### Layout Terminator

```
┌─────────────────────────────────────────┐
│  T1 — SUPERVISION  watch supervisorctl  │
├──────────────┬──────────────┬───────────┤
│ T2 Import    │ T3 Deepfake  │ T4 NLP    │
├──────────────┴──────────────┴───────────┤
│  T5 — Worker Réseau → Neo4j             │
└─────────────────────────────────────────┘
```

### Mapping workers / chemins réels

| Terminal | Worker | Script | Env |
|---|---|---|---|
| T1 | Supervision | `supervisorctl status` | — |
| T2 | Import | `~/AI-FORENSICS/worker_import.py` | `forensics` |
| T3 | Deepfake | `~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE/detect_ai_pipeline-v4.0.py` | `forensics` |
| T4 | NLP | `~/AI-FORENSICS/WORKER/NLP/nlp_worker.py` | `nlp_pipeline` |
| T5 | Réseau | `~/AI-FORENSICS/WORKER/NETWORK/network_worker.py` | `nlp_pipeline` |

### ⚠ supervisord.conf — correction des chemins requise

```ini
[supervisord]
logfile=/home/kwiki/AI-FORENSICS/logs/supervisord.log

[program:worker_import]
command=/home/kwiki/miniconda3/envs/forensics/bin/python worker_import.py
directory=/home/kwiki/AI-FORENSICS
stdout_logfile=/home/kwiki/AI-FORENSICS/logs/worker_import.log

[program:worker_deepfake]
command=/home/kwiki/miniconda3/envs/forensics/bin/python detect_ai_pipeline-v4.0.py --mongo-watch --ensemble --workers 2
directory=/home/kwiki/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE
environment=CUDA_VISIBLE_DEVICES=""
stdout_logfile=/home/kwiki/AI-FORENSICS/logs/detect_ai_pipeline.log

[program:worker_nlp]
command=/home/kwiki/miniconda3/envs/nlp_pipeline/bin/python nlp_worker.py
directory=/home/kwiki/AI-FORENSICS/WORKER/NLP
stdout_logfile=/home/kwiki/AI-FORENSICS/logs/nlp_worker.log

[program:worker_reseau]
command=/home/kwiki/miniconda3/envs/nlp_pipeline/bin/python network_worker.py
directory=/home/kwiki/AI-FORENSICS/WORKER/NETWORK
stdout_logfile=/home/kwiki/AI-FORENSICS/logs/network_worker.log
```

### Lancer le workspace

```bash
cd ~/AI-FORENSICS/SUPERVISOR
bash launch_workspace.sh
```

### Commandes supervisorctl

```bash
cd ~/AI-FORENSICS/SUPERVISOR
supervisorctl -c supervisord.conf status
supervisorctl -c supervisord.conf restart worker_deepfake
supervisorctl -c supervisord.conf stop  all_workers:*
supervisorctl -c supervisord.conf start all_workers:*
```

---

## Commandes de référence

### Analyse deepfake standard
```bash
conda activate forensics
cd ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE
CUDA_VISIBLE_DEVICES="" python detect_ai_pipeline-v4.0.py \
    ~/AI-FORENSICS/DATA_IN --ensemble --workers 2 --verbose
```

### Calibration
```bash
cd ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE
CUDA_VISIBLE_DEVICES="" python detect_ai_pipeline-v4.0.py \
    ~/DATASET/calib_dataset \
    --calibrate --workers 2 --verbose \
    --json-output calib_report_v4.json
```

### Fine-tuning SwinV2
```bash
conda activate forensics_nightly
cd ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python fine_tune_swinv2.py \
    --data-dir ~/DATASET/calib_dataset \
    --output-dir ./swinv2_openfake \
    --epochs 10 --batch-size 16 --num-workers 8 --verbose
```

### MongoDB
```bash
sudo systemctl start mongod
mongosh -u influence_app -p AppPassword456! --authenticationDatabase influence_detection
cd ~/AI-FORENSICS
python -c "from schema import get_db, create_indexes; create_indexes(get_db())"
```

---

## Points techniques importants

**FP-first** — Calibration minimise les faux positifs. Pénalité quadratique si un modèle score haut sur du réel. Mieux vaut rater des fakes qu'accuser du contenu réel.

**Warning sklearn** — model.joblib Synthbuster entraîné avec sklearn 1.7.1, env nightly a 1.8.0. Régler en réentraînant dans `forensics_nightly`.

**HuggingFace token** — `~/.huggingface/token` (chmod 600). Jamais dans le code.

**Divergence inter-modèles** — `model_divergence` > 0.20 → vérification manuelle recommandée.

**schema.py dupliqué** — Référence = racine `~/AI-FORENSICS/schema.py`. Les copies dans NLP/ et NETWORK/ doivent être synchronisées (ou remplacées par des imports relatifs).

**Services systemd non utilisés** — `nlp-worker.service` et `network-worker.service` existent dans WORKER/. Ne pas les activer avec systemd pour éviter les conflits avec supervisord.

---

## Prochaines priorités

1. **Corriger `SUPERVISOR/supervisord.conf`** — chemins `~/AI-FORENSICS/...`
2. **Vérifier `ai_forensics.cfg`** — poids/biais mis à jour après calib_report_v4 ?
3. **Tester les workers** — NLP, NETWORK, Import : fonctionnels en l'état ?
4. **Créer/vérifier env `nlp_pipeline`** — voir `WORKER/NLP/requirements_nlp.txt`
5. **Réentraîner Synthbuster** dans `forensics_nightly`
6. **Enrichir `ALT/`** avec OpenFake (FLUX 1.1-pro, MJ v7, GPT Image 1)
