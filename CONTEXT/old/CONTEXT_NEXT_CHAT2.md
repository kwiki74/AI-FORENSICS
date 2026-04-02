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
    ├── Worker deepfake  ← WORKER/DETECT_AI_PIPLINE/detect_ai_pipeline-v4.0.2.py  ⚠ one-shot
    ├── Worker NLP       ← WORKER/NLP/nlp_worker.py
    └── Worker réseau    ← WORKER/NETWORK/network_worker.py
                                  ↓
                             Neo4j (neo4j_client.py + campaign_detector.py)
    ↓  ETL optionnel
    └── Elasticsearch    (full-text, Kibana — optionnel)
```

**MongoDB** = source de vérité. **Neo4j** = relations et campagnes coordonnées.

---

## Structure des dossiers (racine : `~/AI-FORENSICS/`)

```
~/AI-FORENSICS/
├── CONTEXT/
│   └── CONTEXT_NEXT_CHAT.md
├── DATA_IN/                              ← inbox scrapper
│   ├── converted_INSTAGRAM_crypto_2026-03-17/
│   │   ├── coincryptofr/ · cryptoastmedia/ · cryptocomofficial/ · ...
│   ├── converted_TELEGRAM_crypto_2026-03-17/
│   │   └── cryptoast_fr/
│   └── converted_TIKTOK_crypto_2026-03-17/
│       └── agama.club/ ...
├── logs/                                 ← logs centralisés (tous workers)
│   ├── detect_ai_pipeline.log
│   ├── worker_import.log
│   ├── nlp_worker.log
│   ├── network_worker.log
│   ├── supervisord.log
│   └── supervision_web.log               ← (quand serveur web actif)
├── schema.py                             ← schéma MongoDB v3 (référence)
├── SUPERVISOR/
│   ├── supervisord.conf                  ← ✅ chemins corrigés anaconda3
│   ├── launch_workspace.sh               ← ✅ chemins corrigés
│   ├── terminator_layout.conf            ← ✅ layout 5 terminaux
│   ├── t1_supervision.sh                 ← voyants workers + services
│   ├── t2_import.sh                      ← tail coloré
│   ├── t3_deepfake.sh                    ← menu one-shot
│   ├── t4_nlp.sh
│   ├── t5_reseau.sh
│   ├── supervision_server.py             ← ⏳ À DÉPLOYER (voir ci-dessous)
│   └── INSTALL_WORKSPACE.md
├── WORKER/
│   ├── DETECT_AI_PIPLINE/
│   │   ├── detect_ai_pipeline-v4.0.2.py  ← version courante ⚠ one-shot
│   │   ├── ai_forensics.cfg
│   │   ├── calib_report_v4.json          ← calibration v4 faite ✅
│   │   ├── swinv2_openfake/model.safetensors ✅
│   │   └── synthbuster/models/ (liens sym ✅)
│   ├── NETWORK/
│   │   ├── network_worker.py · neo4j_client.py · campaign_detector.py
│   │   ├── network_pipeline.cfg
│   │   └── logs/network_worker.log
│   └── NLP/
│       ├── nlp_worker.py · embeddings.py · sentiment.py · narrative_clustering.py
│       ├── nlp_pipeline.cfg
│       └── requirements_nlp.txt
├── worker_import.py                      ← racine
└── worker_import.cfg
```

---

## État actuel du projet (mars 2026)

### ✅ Opérationnel / en production

- **worker_import** — tourne via supervisord, logs dans `logs/worker_import.log`
  - Stats typiques : ~3270 posts OK · 0 erreurs · warnings BUG_ID/post_url normaux
- **Workspace Terminator** — 5 terminaux opérationnels
  - T1 : voyants supervisord + MongoDB + Neo4j (refresh 5s)
  - T2 : tail coloré worker_import (WARNING jaune, ERROR rouge)
  - T3 : menu interactif one-shot deepfake (analyse / verbose / calibration / logs)
  - T4 : tail nlp_worker
  - T5 : tail network_worker
- **MongoDB 8.0** — ReplicaSet rs0, auth, schema v3
- **Deepfake v4.0.2** — utilisable en one-shot via menu T3

### ⚠ Points importants sur le deepfake worker

`detect_ai_pipeline-v4.0.2.py` est **one-shot uniquement** — pas de mode daemon/watch.
Dans supervisord : `autostart=false`, `autorestart=false`.
Pour analyser : utiliser le menu T3 ou la commande directe.

**Un wrapper `deepfake_watcher.py` est à coder** — deux modes envisagés :
- Mode A : surveiller `DATA_IN/` pour nouveaux médias non analysés
- Mode B : interroger MongoDB (`media`, `status=pending`)
- Mode C (préféré) : les deux en séquence — choix à trancher

### ⏳ À déployer quand disponible

**Serveur web de supervision** (`SUPERVISOR/supervision_server.py`)
- Serveur Flask léger, port 5050, polling 5s
- Accessible réseau local → collègue peut l'ouvrir dans son navigateur
- Affiche : état workers + services (MongoDB/Neo4j) + stats + logs déroulants + erreurs
- Endpoint `/api/status` JSON pour intégration dans page existante du collègue
- **Prérequis** : `pip install flask` dans env `forensics`
- **Déploiement** :
  ```bash
  conda activate forensics && pip install flask
  cp supervision_server.py ~/AI-FORENSICS/SUPERVISOR/
  # Ajouter [program:supervision_web] dans supervisord.conf (déjà écrit)
  supervisorctl -c supervisord.conf reread && supervisorctl -c supervisord.conf update
  # → http://localhost:5050  ou  http://<IP-locale>:5050
  ```

### 🔄 À vérifier

- **`ai_forensics.cfg`** — poids/biais mis à jour après calib_report_v4 ?
- **Workers NLP + Réseau** — fonctionnels ? testés avec données réelles ?
- **Env `nlp_pipeline`** — existe ? voir `WORKER/NLP/requirements_nlp.txt`
- **`schema.py` dupliqué** — 3 copies (racine, NLP/, NETWORK/) à synchroniser

### ❌ Pas encore fait

- Wrapper `deepfake_watcher.py` (mode daemon pour le pipeline deepfake)
- Env `nlp_pipeline` à créer/vérifier
- Réentraîner Synthbuster dans `forensics_nightly` (warning sklearn 1.7.1 vs 1.8.0)
- Enrichir `ALT/` avec OpenFake (FLUX 1.1-pro, MJ v7, GPT Image 1)

---

## Modèles deepfake v4.0.2

| # | Modèle | Mécanisme | Générateurs couverts |
|---|---|---|---|
| 1 | `Organika/sdxl-detector` | CNN diffusion | SD/SDXL, FLUX partiel |
| 2 | `./swinv2_openfake` | SwinV2 fine-tuné | FLUX, MJ v6, DALL-E 3, Grok-2, Ideogram 3 |
| 3 | `synthbuster/synthbuster` | Fourier/sklearn | Artefacts spectraux |

Poids/biais : `WORKER/DETECT_AI_PIPLINE/ai_forensics.cfg` (à vérifier post-calib v4).

---

## Infrastructure technique

**Machine** — Ubuntu 24.04, 20 cœurs, 30 Go RAM, RTX 5070 Laptop (8 Go VRAM, sm_120 Blackwell)

**Environnements conda** (`~/anaconda3/envs/`)

| Env | Usage |
|---|---|
| `forensics` | Deepfake + worker_import + supervisord + Flask, CPU uniquement |
| `forensics_nightly` | GPU RTX 5070, fine-tuning (nightly cu128) |
| `nlp_pipeline` | Worker NLP + Worker réseau — **à créer/vérifier** |

---

## Workspace de supervision

### Layout Terminator

```
┌─────────────────────────────────────────┐
│  T1 — SUPERVISION  (voyants 5s)         │
├──────────────┬──────────────┬───────────┤
│ T2 Import    │ T3 Deepfake  │ T4 NLP    │
│ tail coloré  │ menu one-shot│ tail      │
├──────────────┴──────────────┴───────────┤
│  T5 — Worker Réseau → Neo4j  tail       │
└─────────────────────────────────────────┘
```

### Fichiers dans `SUPERVISOR/`

| Fichier | Rôle | État |
|---|---|---|
| `supervisord.conf` | Config 4 workers + supervision_web | ✅ chemins corrigés |
| `launch_workspace.sh` | Démarre MongoDB + supervisord + Terminator | ✅ |
| `terminator_layout.conf` | Layout 5 terminaux | ✅ |
| `t1_supervision.sh` | Voyants + services + erreurs | ✅ |
| `t2_import.sh` | Tail coloré import | ✅ |
| `t3_deepfake.sh` | Menu one-shot deepfake | ✅ |
| `t4_nlp.sh` | Tail NLP | ✅ |
| `t5_reseau.sh` | Tail réseau | ✅ |
| `supervision_server.py` | Serveur web Flask port 5050 | ⏳ à déployer |
| `INSTALL_WORKSPACE.md` | Guide d'installation | ✅ |

### Commandes supervisorctl

```bash
cd ~/AI-FORENSICS/SUPERVISOR
supervisorctl -c supervisord.conf status
supervisorctl -c supervisord.conf restart worker_import
supervisorctl -c supervisord.conf stop   workers_actifs:*
supervisorctl -c supervisord.conf start  workers_actifs:*
```

---

## Commandes de référence

### Lancer le workspace
```bash
cd ~/AI-FORENSICS/SUPERVISOR
bash launch_workspace.sh
```

### Analyse deepfake one-shot
```bash
conda activate forensics
cd ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE
CUDA_VISIBLE_DEVICES="" python detect_ai_pipeline-v4.0.2.py \
    ~/AI-FORENSICS/DATA_IN --ensemble --workers 2 --verbose
```

### Calibration
```bash
cd ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE
CUDA_VISIBLE_DEVICES="" python detect_ai_pipeline-v4.0.2.py \
    ~/DATASET/calib_dataset --calibrate --workers 2 --verbose \
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

**FP-first** — Calibration minimise faux positifs. Pénalité quadratique si modèle score haut sur du réel.

**Deepfake one-shot** — `detect_ai_pipeline-v4.0.2.py` n'a pas de mode daemon. `--mongo-watch` n'existe pas. Utiliser le menu T3 ou coder `deepfake_watcher.py`.

**Warning sklearn** — Synthbuster entraîné avec 1.7.1, env nightly a 1.8.0. Régler en réentraînant dans `forensics_nightly`.

**HuggingFace token** — `~/.huggingface/token` (chmod 600).

**Divergence inter-modèles** — `model_divergence` > 0.20 → vérification manuelle.

**schema.py dupliqué** — Référence = racine. Copies dans NLP/ et NETWORK/ à synchroniser.

**Services systemd** — `nlp-worker.service` et `network-worker.service` existent mais NE PAS activer (conflits supervisord).

---

## Prochaines priorités

1. **Coder `deepfake_watcher.py`** — trancher mode A (DATA_IN) / B (MongoDB) / C (les deux)
2. **Vérifier `ai_forensics.cfg`** — poids/biais post-calibration v4
3. **Tester workers NLP + Réseau** — fonctionnels avec données réelles ?
4. **Créer env `nlp_pipeline`** — `pip install -r WORKER/NLP/requirements_nlp.txt`
5. **Déployer `supervision_server.py`** — `pip install flask` + update supervisord
6. **Réentraîner Synthbuster** dans `forensics_nightly`
7. **Enrichir `ALT/`** avec OpenFake (FLUX 1.1-pro, MJ v7, GPT Image 1)
