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
Scrapper (binôme) → INPUT/ (JSON + médias bruts — arborescence DOSSIER_INPUT)
    ↓
worker_import.py → MongoDB (influence_detection)
    ↓  Change Streams
    ├── Worker deepfake  ← WORKER/DETECT_AI_PIPLINE/detect_ai_pipeline-v4.0.3.py  ✅ mode --mongojob
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
├── INPUT/                                ← DOSSIER_INPUT scrapper (arborescence projets/scans/users)
│   └── ProJet0/
│       ├── converted_TIKTOK_crypto_2026-03-25/
│       │   └── memecoin_clips/           ← converted_*.info.json
│       └── TIKTOK_crypto_2026-03-25/
│           └── memecoin_clips/           ← *.mp4 (médias bruts)
├── logs/                                 ← logs centralisés (tous workers)
│   ├── worker_import.log
│   ├── import_cursor.json                ← curseur worker_import (ne pas supprimer)
│   ├── archives/                         ← entrées curseur obsolètes archivées
│   ├── detect_ai_pipeline_YYYYMMDD.log
│   ├── detect_ai_pipeline_error.log
│   ├── nlp_worker.log
│   ├── network_worker.log
│   ├── supervisord.log
│   └── supervision_web.log
├── SCHEMA/
│   └── schema.py                         ← ✅ schéma MongoDB v3 — référence unique
├── SUPERVISOR/
│   ├── supervisord.conf
│   ├── launch_workspace.sh
│   ├── terminator_layout.conf
│   ├── t1_supervision.sh · t2_import.sh · t3_deepfake.sh · t4_nlp.sh · t5_reseau.sh
│   ├── supervision_server.py             ← ⏳ À DÉPLOYER
│   └── INSTALL_WORKSPACE.md
└── WORKER/
    ├── DETECT_AI_PIPLINE/
    │   ├── detect_ai_pipeline-v4.0.3.py  ← ✅ version courante — mode --mongojob multiprocessing
    │   ├── ai_forensics.cfg              ← ✅ sections [mongodb] et [mongojob] ajoutées
    │   ├── calib_report_v4.json          ← calibration v4 faite ✅
    │   ├── swinv2_openfake/              ← poids fine-tunés locaux ✅
    │   └── synthbuster/models/           ← liens symboliques ✅
    ├── IMPORT/
    │   ├── worker_import.py              ← ✅ v1.4 — curseur, multiplateforme, heartbeat
    │   └── worker_import.cfg             ← credentials MongoDB + poll/heartbeat
    ├── NETWORK/
    │   ├── network_worker.py             ← v2
    │   ├── neo4j_client.py
    │   ├── campaign_detector.py
    │   └── network_pipeline.cfg
    └── NLP/
        ├── nlp_worker.py · embeddings.py · sentiment.py · narrative_clustering.py
        ├── nlp_pipeline.cfg
        └── requirements_nlp.txt
```

---

## État actuel du projet (mars 2026)

### ✅ Opérationnel / en production

- **worker_import v1.4** — tourne via supervisord, mode `--watch` sur INPUT/
  - Curseur `import_cursor.json` dans logs/ — skip des fichiers inchangés entre cycles
  - Archives des entrées obsolètes dans logs/archives/
  - Résumé coloré : nouveaux / déjà en base / ignorés (inchangés)
  - Heartbeat ♥ sur stderr toutes les 30s en mode attente
  - Médias indexés dans collection `media` avec `source` {project, scan, user}
  - Jobs `deepfake_analysis` créés automatiquement pour chaque média

- **detect_ai_pipeline v4.0.3** — ✅ mode `--mongojob` opérationnel et testé
  - Consomme la file `jobs` MongoDB (claim atomique, pas de race condition)
  - Multiprocessing spawn — N workers indépendants (`workers = 8` dans cfg)
  - Mise à jour `media.deepfake` + `posts.deepfake` (stratégie pire cas)
  - Log dédié `detect_ai_pipeline_error.log` pour les skips et erreurs
  - Import `schema.py` depuis `../../SCHEMA/schema.py` (fallback local)
  - `ai_forensics.cfg` — sections `[mongodb]` et `[mongojob]` (poll=10s, heartbeat=60s)

- **MongoDB 8.0** — ReplicaSet rs0, auth, schema v3
  - 7 collections : accounts, posts, comments, media, narratives, campaigns, jobs
  - `media` — champ `source` {project, scan, user} ✅ (session courante)
  - Index TTL sur jobs (7 jours)

- **Neo4j** — opérationnel, GDS installé ✅

- **Worker réseau v2** — testé le 23/03/2026 avec données réelles :
  - 29 comptes + 3230 posts + 49 narratifs synchronisés vers Neo4j
  - 28 campagnes détectées (scores 0.30 à 0.55)
  - Détection automatique post-idle 30s fonctionnelle

- **Workspace Terminator** — 5 terminaux opérationnels (T1-T5)

### Scripts de maintenance ✅ (session courante)

- `purge_mongodb.py` — purge sélective ou complète de la base MongoDB
  ```bash
  python purge_mongodb.py --dry-run
  python purge_mongodb.py --all --yes
  python purge_mongodb.py --collections posts media jobs
  python purge_mongodb.py --reset-deepfake   # remet tout en pending sans re-importer
  ```
- `purge_neo4j.py` — purge Neo4j (prêt, nécessite `pip install neo4j`)

---

## Architecture DOSSIER_INPUT (scrapper binôme)

```
INPUT/
  ProjetX/
    converted_RESEAU_motcle_YYYY-MM-DD/   ← JSON converted (importés)
      UserA/
        converted_post1.info.json
    RESEAU_motcle_YYYY-MM-DD/             ← médias bruts (référencés, non déplacés)
      UserA/
        post1.mp4
        post1.jpg
```

- Les fichiers `converted_*.info.json` sont importés (double suffixe `.info` géré)
- Les médias sont référencés par chemin absolu dans `media.url_local`
- Le curseur `import_cursor.json` évite de rescanner les fichiers inchangés
- `media.source` = `{project: "ProjetX", scan: "RESEAU_motcle_YYYY-MM-DD", user: "UserA"}`

---

## Modèles deepfake v4.0.3

| # | Modèle | Mécanisme | Générateurs couverts |
|---|---|---|---|
| 1 | `Organika/sdxl-detector` | CNN diffusion | SD/SDXL, FLUX partiel |
| 2 | `./swinv2_openfake` | SwinV2 fine-tuné local | FLUX, MJ v6, DALL-E 3, Grok-2, Ideogram 3 |
| 3 | `synthbuster/synthbuster` | Fourier/sklearn | Artefacts spectraux |

Poids/biais calibrés dans `WORKER/DETECT_AI_PIPLINE/ai_forensics.cfg`.

---

## Infrastructure technique

**Machine** — Ubuntu 24.04, 20 cœurs, 30 Go RAM, RTX 5070 Laptop (8 Go VRAM, sm_120 Blackwell)
→ GPU incompatible avec PyTorch stable — fallback CPU automatique ✅

**Environnements conda**

| Env | Usage |
|---|---|
| `forensics` | Deepfake + worker_import + supervisord + Flask, CPU |
| `forensics_nightly` | GPU RTX 5070, fine-tuning (nightly cu128) |
| `nlp_pipeline` | Worker NLP + Worker réseau — **à vérifier** |

---

## Points techniques importants

**schema.py** — référence unique dans `SCHEMA/schema.py`.
Tous les scripts importent depuis `../../SCHEMA/schema.py` (fallback local).
⚠ Les copies dans NLP/ et NETWORK/ sont à synchroniser (pas encore fait).

**worker_import — curseur** — `logs/import_cursor.json` est l'état persistant des fichiers importés.
Ne pas supprimer sauf pour forcer un re-import complet.
Les entrées obsolètes (fichiers supprimés du INPUT) sont archivées dans `logs/archives/`.

**mongojob multiprocessing** — `workers = 8` dans `ai_forensics.cfg`.
Chaque worker spawn charge ses modèles et consomme la file de façon autonome.
Le claim atomique garantit qu'aucun job n'est traité deux fois.

**FP-first** — Calibration minimise faux positifs. Pénalité quadratique si modèle score haut sur du réel.

**Warning sklearn** — Synthbuster entraîné avec 1.7.1, env courant a 1.8.0. Régler en réentraînant dans `forensics_nightly`.

**HuggingFace token** — `~/.huggingface/token` (chmod 600). Jamais dans le code.

**Divergence inter-modèles** — `model_divergence` > 0.20 → vérification manuelle conseillée.

**Louvain 0 lien** — Normal sur le dataset actuel (29 comptes sans coordination directe).

**Dépréciation GDS** — `gds.graph.project.cypher` génère un warning. Fonctionne. Migration non urgente.

---

## Commandes de référence

### Worker import
```bash
conda activate nlp_pipeline
cd ~/AI-FORENSICS/WORKER/IMPORT
python worker_import.py                        # watch continu sur INPUT/
python worker_import.py --source ~/INPUT/ --dry-run  # test sans écriture
python worker_import.py --once                 # one-shot inbox
```

### Pipeline deepfake — mode MongoDB jobs
```bash
conda activate forensics
cd ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE
python detect_ai_pipeline-v4.0.3.py --mongojob           # 8 workers (cfg)
python detect_ai_pipeline-v4.0.3.py --mongojob --workers 4
python detect_ai_pipeline-v4.0.3.py --mongojob --verbose  # logs détaillés
```

### Pipeline deepfake — mode one-shot dossier (inchangé)
```bash
CUDA_VISIBLE_DEVICES="" python detect_ai_pipeline-v4.0.3.py \
    ~/AI-FORENSICS/DATA_IN --ensemble --workers 8 --verbose
```

### Maintenance base MongoDB
```bash
cd ~/AI-FORENSICS/WORKER/IMPORT   # ou tout autre dossier
python purge_mongodb.py --dry-run
python purge_mongodb.py --reset-deepfake   # relancer analyse sans re-importer
python purge_mongodb.py --all --yes        # purge complète
```

### MongoDB shell
```bash
sudo systemctl start mongod
mongosh --host localhost --port 27017 -u influence_app -p 'AiForens!cS1' \
        --authenticationDatabase influence_detection
# Dans mongosh :
use influence_detection
['accounts','posts','comments','media','jobs'].forEach(c =>
    print(c.padEnd(12) + ' : ' + db[c].countDocuments()))
```

### Neo4j
```bash
sudo systemctl start neo4j
# Browser : http://localhost:7474
# Credentials : neo4j / influence2026!
```

### Worker réseau
```bash
conda activate nlp_pipeline
cd ~/AI-FORENSICS/WORKER/NETWORK
python network_worker.py --backfill
python campaign_detector.py --min-score 0.30
```

### Supervisord
```bash
cd ~/AI-FORENSICS/SUPERVISOR
supervisorctl -c supervisord.conf status
supervisorctl -c supervisord.conf restart worker_import
```

---

## ❌ Pas encore fait / À faire

1. **Synchroniser `schema.py`** dans NLP/ et NETWORK/ avec `SCHEMA/schema.py`
   (ou mettre à jour leurs imports pour pointer sur `../../SCHEMA/schema.py`)

2. **Tester worker NLP** avec données réelles — fonctionnel ?

3. **Déployer `supervision_server.py`** :
   ```bash
   conda activate forensics && pip install flask
   supervisorctl -c supervisord.conf reread && update
   # → http://localhost:5050
   ```

4. **Réentraîner Synthbuster** dans `forensics_nightly`
   (résout le warning sklearn 1.7.1 vs 1.8.0)

5. **Enrichir `ALT/`** avec OpenFake (FLUX 1.1-pro, MJ v7, GPT Image 1)
   `huggingface.co/datasets/ComplexDataLab/OpenFake`

6. **Migrer GDS** — `gds.graph.project.cypher` → nouvelle syntaxe (non urgent)

7. **Calibration v4** — vérifier que `ai_forensics.cfg` reflète bien `calib_report_v4.json`

---

## Requêtes Neo4j utiles (Browser)

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
