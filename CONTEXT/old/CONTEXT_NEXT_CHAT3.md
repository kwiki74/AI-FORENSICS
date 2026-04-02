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
                                  ↓  (détection auto post-idle 30s)
                             campaign_detector.py  (6 signaux dont Louvain + PageRank)
                                  ↓
                             Neo4j + MongoDB collection campaigns
    ↓  ETL optionnel
    └── Elasticsearch    (full-text, Kibana — optionnel)
```

**MongoDB** = source de vérité. **Neo4j** = relations et campagnes coordonnées.
**GDS (Graph Data Science)** = plugin Neo4j installé ✅ — Louvain, PageRank, Betweenness, BFS.

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
│   ├── NETWORK/                          ← ✅ testé en production (23/03/2026)
│   │   ├── network_worker.py             ← v2 — détection auto post-idle
│   │   ├── neo4j_client.py               ← Louvain + PageRank + Betweenness + BFS
│   │   ├── campaign_detector.py          ← 6 signaux (4 MongoDB + 2 GDS)
│   │   ├── network_pipeline.cfg          ← section [detection] ajoutée
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

- **worker_import** — tourne via supervisord
- **Workspace Terminator** — 5 terminaux opérationnels (T1-T5)
- **MongoDB 8.0** — ReplicaSet rs0, auth, schema v3
- **Deepfake v4.0.2** — utilisable en one-shot via menu T3
- **Neo4j** — opérationnel, GDS installé ✅
- **Worker réseau v2** — testé le 23/03/2026 avec données réelles :
  - 29 comptes + 3230 posts + 49 narratifs synchronisés vers Neo4j
  - 28 campagnes détectées (scores 0.30 à 0.55)
  - Détection automatique post-idle 30s fonctionnelle (une seule passe par période d'inactivité)
  - Signaux actifs : content_reuse, cross_platform, coordinated_posting, key_amplifiers (PageRank)
  - Louvain retourne 0 lien (29 comptes sans coordination directe détectée sur ce dataset)

### Worker réseau — détail des fichiers v2

**`network_worker.py` v2** (modifié 23/03/2026)
- Détection auto déclenchée quand file Change Stream vide depuis `detection_idle_seconds` (défaut 30s)
- Une seule passe par période d'inactivité — flag `_detection_done_this_idle` reset au prochain doc
- Paramètre `--skip-detection` pour désactiver la détection auto

**`neo4j_client.py`** (modifié 23/03/2026)
- Vérification GDS via `RETURN gds.version()` (corrigé — ancienne méthode `gds.list()` ne fonctionnait pas)
- Projection GDS via `gds.graph.project.cypher` (Account↔Account via posts partagés)
- `run_louvain()` — corrigé + écrit `community_id` sur nœuds Account
- `run_pagerank()` — écrit `pagerank_score` sur nœuds Account
- `run_betweenness()` — ponctuel / investigation manuelle ⚠ coûteux
- `run_bfs(source_mongo_id)` — propagation d'un contenu / investigation manuelle
- ⚠ Warning de dépréciation GDS : `gds.graph.project.cypher` déprécié dans la version installée
  → fonctionne encore, à migrer vers `gds.graph.project` avec aggregation function (non urgent)

**`campaign_detector.py`** (modifié 23/03/2026)
- 6 signaux : content_reuse (0.25) + cross_platform (0.20) + coordinated_posting (0.20)
  + synthetic_media (0.10) + coordinated_accounts/Louvain (0.15) + key_amplifiers/PageRank (0.10)
- Paramètre `--skip-gds` pour ignorer signaux GDS
- `neo4j_client` optionnel — si absent, tourne avec les 4 signaux MongoDB uniquement

**`network_pipeline.cfg`** (modifié 23/03/2026)
- Nouvelle clé `[worker] detection_idle_seconds = 30`
- Nouvelle section `[detection] min_score = 0.30 / skip_gds = false`

### Requêtes Neo4j utiles (Browser)

```cypher
-- Vue d'ensemble campagnes
MATCH (a:Account)-[:A_PUBLIÉ]->(p:Post)-[:APPARTIENT_À]->(n:Narrative)
RETURN n.label, count(DISTINCT a) AS comptes, count(DISTINCT p) AS posts,
       collect(DISTINCT a.platform) AS plateformes
ORDER BY posts DESC

-- Comptes amplificateurs (PageRank)
MATCH (a:Account) WHERE a.pagerank_score IS NOT NULL
RETURN a.username, a.platform, a.pagerank_score, a.community_id
ORDER BY a.pagerank_score DESC LIMIT 20

-- Doublons cross-plateforme
MATCH (a1:Account)-[:A_PUBLIÉ]->(p1:Post)-[:EST_DOUBLON_DE]->(p2:Post)<-[:A_PUBLIÉ]-(a2:Account)
WHERE a1.platform <> a2.platform
RETURN a1.username, a1.platform, a2.username, a2.platform, count(*) AS doublons
ORDER BY doublons DESC
```

### ⚠ Points importants sur le deepfake worker

`detect_ai_pipeline-v4.0.2.py` est **one-shot uniquement** — pas de mode daemon/watch.
Dans supervisord : `autostart=false`, `autorestart=false`.
**Un wrapper `deepfake_watcher.py` est à coder** — modes envisagés :
- Mode A : surveiller `DATA_IN/` pour nouveaux médias
- Mode B : interroger MongoDB (`media.status=pending`)
- Mode C (préféré) : les deux en séquence

### ⏳ À déployer quand disponible

**Serveur web de supervision** (`SUPERVISOR/supervision_server.py`)
- Serveur Flask léger, port 5050, polling 5s
- Accessible réseau local → collègue peut l'ouvrir dans son navigateur
- **Déploiement** :
  ```bash
  conda activate forensics && pip install flask
  supervisorctl -c supervisord.conf reread && supervisorctl -c supervisord.conf update
  # → http://localhost:5050
  ```

### 🔄 À vérifier

- **`ai_forensics.cfg`** — poids/biais mis à jour après calib_report_v4 ?
- **Worker NLP** — fonctionnel ? testé avec données réelles ?
- **Env `nlp_pipeline`** — existe ? voir `WORKER/NLP/requirements_nlp.txt`
- **`schema.py` dupliqué** — 3 copies (racine, NLP/, NETWORK/) à synchroniser

### ❌ Pas encore fait

- Wrapper `deepfake_watcher.py` (mode daemon pour le pipeline deepfake)
- Env `nlp_pipeline` à créer/vérifier
- Réentraîner Synthbuster dans `forensics_nightly` (warning sklearn 1.7.1 vs 1.8.0)
- Enrichir `ALT/` avec OpenFake (FLUX 1.1-pro, MJ v7, GPT Image 1)
- Migrer `gds.graph.project.cypher` → nouvelle syntaxe GDS (non urgent)

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

### Worker réseau (avec backfill au démarrage)
```bash
conda activate nlp_pipeline
cd ~/AI-FORENSICS/WORKER/NETWORK
python network_worker.py --backfill
# Sans détection GDS (plus rapide)
python network_worker.py --backfill --skip-detection
```

### Détection campagnes manuelle
```bash
cd ~/AI-FORENSICS/WORKER/NETWORK
python campaign_detector.py --min-score 0.30
python campaign_detector.py --min-score 0.30 --skip-gds
python campaign_detector.py --min-score 0.30 --dry-run
```

### MongoDB
```bash
sudo systemctl start mongod
mongosh -u influence_app -p AiForens!cS1 --authenticationDatabase influence_detection
```

### Neo4j
```bash
sudo systemctl start neo4j
# Browser : http://localhost:7474
# Credentials : neo4j / influence2026!
```

---

## Points techniques importants

**FP-first** — Calibration minimise faux positifs. Pénalité quadratique si modèle score haut sur du réel.

**Deepfake one-shot** — `detect_ai_pipeline-v4.0.2.py` n'a pas de mode daemon. Coder `deepfake_watcher.py`.

**Warning sklearn** — Synthbuster entraîné avec 1.7.1, env nightly a 1.8.0. Régler en réentraînant dans `forensics_nightly`.

**HuggingFace token** — `~/.huggingface/token` (chmod 600). Jamais dans le code.

**Divergence inter-modèles** — `model_divergence` > 0.20 → vérification manuelle.

**schema.py dupliqué** — Référence = racine. Copies dans NLP/ et NETWORK/ à synchroniser.

**Services systemd** — `nlp-worker.service` et `network-worker.service` existent mais NE PAS activer (conflits supervisord).

**Louvain 0 lien** — Normal sur le dataset actuel (29 comptes, pas de retweets/forwards Telegram). Louvain sera utile quand le dataset contiendra des relations directes Account↔Account via posts partagés en masse.

**Dépréciation GDS** — `gds.graph.project.cypher` génère un warning dans les logs. Fonctionne encore. Migration vers la nouvelle syntaxe à prévoir mais non urgente.

---

## Prochaines priorités

1. **Coder `deepfake_watcher.py`** — trancher mode A (DATA_IN) / B (MongoDB) / C (les deux)
2. **Vérifier `ai_forensics.cfg`** — poids/biais post-calibration v4
3. **Tester worker NLP** — fonctionnel avec données réelles ?
4. **Créer env `nlp_pipeline`** — `pip install -r WORKER/NLP/requirements_nlp.txt`
5. **Déployer `supervision_server.py`** — `pip install flask` + update supervisord
6. **Réentraîner Synthbuster** dans `forensics_nightly`
7. **Enrichir `ALT/`** avec OpenFake (FLUX 1.1-pro, MJ v7, GPT Image 1)
8. **Migrer GDS** — `gds.graph.project.cypher` → nouvelle syntaxe (non urgent)
