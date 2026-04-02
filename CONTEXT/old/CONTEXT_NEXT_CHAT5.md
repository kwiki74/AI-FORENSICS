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
│   └── schema.py                         ← ✅ schéma MongoDB v4 — référence UNIQUE
│                                            (ne plus avoir de copies ailleurs)
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
    │   ├── worker_import.py              ← ✅ v1.5 — résolution schema centralisé + champs v4
    │   └── worker_import.cfg             ← credentials MongoDB + poll/heartbeat
    ├── NETWORK/
    │   ├── network_worker.py             ← ✅ v4 — Media + Hashtag + Deepfake nodes + --projet
    │   ├── neo4j_client.py               ← ✅ v4 — upsert_media, link_post_media, purge_all
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

- **worker_import v1.5**
  - Résolution `schema.py` centralisé (`../../SCHEMA/` → `../` → `./`)
  - Mapping nouveaux champs v4 : `music_author`, `cover_url`, `influence_score`, `is_bot_suspected`
  - Import `patch_media_sync` ajouté

- **detect_ai_pipeline v4.0.3** — mode `--mongojob` opérationnel et testé
  - Multiprocessing spawn — N workers indépendants (`workers = 8` dans cfg)
  - Import `schema.py` depuis `../../SCHEMA/schema.py` (fallback local)

- **schema.py v4** — fichier unique dans `SCHEMA/`
  - `new_post()` — `scrapper_signals` (`influence_score`, `is_bot_suspected`), `music_author`, `cover_url`
  - `new_media()` — champ `sync` ajouté (`neo4j: false`)
  - `patch_media_sync()` — nouvelle fonction
  - Index `sync.neo4j` sur collection `media`

- **MongoDB 8.0** — ReplicaSet rs0, auth, schema v4
  - 7 collections : accounts, posts, comments, media, narratives, campaigns, jobs
  - `media` — champ `sync` ajouté ⚠ voir migration ci-dessous
  - Index TTL sur jobs (7 jours)

- **Neo4j** — opérationnel, GDS installé ✅

- **neo4j_client.py v4**
  - Nouveaux nœuds : `:Hashtag`, `:Deepfake`, `:Media`
  - Nouvelles relations : `HAS_HASHTAG`, `IS_DEEPFAKE`, `A_MEDIA`
  - `upsert_media()`, `upsert_media_batch()`, `link_post_media()`
  - `purge_all()` — suppression totale en batches
  - `upsert_post()` étendu : `like_count`, `comment_count`, `influence_score`, `music_author`, `cover_url`, `source_project`

- **network_worker.py v4**
  - Écoute Change Stream sur `posts`, `comments`, `accounts`, **`media`**
  - `_sync_media_doc()`, `_backfill_media()`, `_flush_media_batch()`
  - `_media_to_node()` — convertit doc media MongoDB → nœud `:Media` Neo4j
  - Mode `--projet` / `-p` (répétable) + `--add` (sans purge)
  - Purge Neo4j + reset `sync.neo4j=False` en mode `--projet`

- **Workspace Terminator** — 5 terminaux opérationnels (T1-T5)

---

## ⚠ Migration requise avant prochain lancement

Les documents `media` existants en base **n'ont pas encore le champ `sync`**.
À exécuter dans `mongosh` avant le premier lancement du network_worker v4 :

```javascript
use influence_detection
db.media.updateMany(
  { sync: { $exists: false } },
  { $set: { "sync.neo4j": false, "sync.elasticsearch": false, "sync.synced_at": null } }
)
```

---

## Résolution schema.py — règle commune à TOUS les workers

Tous les scripts utilisent la même chaîne de recherche :
```
../../SCHEMA/schema.py   (chemin nominal — WORKER/XXX/ → racine SCHEMA/)
../schema.py             (fallback)
./schema.py              (fallback local)
```
Workers mis à jour : `worker_import`, `network_worker`, `nlp_worker` (à vérifier).
`detect_ai_pipeline` : déjà correct depuis une session précédente.
**Ne plus avoir de copies de `schema.py` dans les sous-dossiers WORKER/.**

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

Format JSON unifié (toutes plateformes) — champs clés :
- `scrappeurInfo` — métadonnées scrapper (version, platform, gdh_scrap…)
- `desc` → mappé vers `posts.text.content`
- `hashtags` → `posts.text.hashtags`
- `stats.plays` → `engagement.views` / `stats.favorites` → `engagement.saves`
- `music.author` → `platform_specific.music_author` [v4]
- `cover` → `platform_specific.cover_url` [v4]
- `influence_score` → `scrapper_signals.influence_score` [v4]
- `is_bot_suspected` → `scrapper_signals.is_bot_suspected` [v4]

---

## Modèle de graphe Neo4j (v4)

```
Nœuds :
  (:Account)   — compte sur une plateforme
  (:Post)      — publication (texte + métadonnées)
  (:Narrative) — cluster narratif NLP
  (:Hashtag)   — hashtag [v3]
  (:Deepfake)  — type de média synthétique [v3]
  (:Media)     — fichier média vidéo/image [v4]

Relations :
  (:Account)-[:A_PUBLIÉ]->(:Post)
  (:Account)-[:A_COMMENTÉ]->(:Post)
  (:Account)-[:A_FORWARDÉ {count}]->(:Post)
  (:Post)-[:EST_DOUBLON_DE]->(:Post)
  (:Post)-[:APPARTIENT_À]->(:Narrative)
  (:Post)-[:HAS_HASHTAG]->(:Hashtag)
  (:Post)-[:IS_DEEPFAKE {score}]->(:Deepfake)
  (:Post)-[:A_MEDIA]->(:Media)               [v4]
```

---

## Modèles deepfake v4.0.3

| # | Modèle | Mécanisme | Générateurs couverts |
|---|---|---|---|
| 1 | `Organika/sdxl-detector` | CNN diffusion | SD/SDXL, FLUX partiel |
| 2 | `./swinv2_openfake` | SwinV2 fine-tuné local | FLUX, MJ v6, DALL-E 3, Grok-2, Ideogram 3 |
| 3 | `synthbuster/synthbuster` | Fourier/sklearn | Artefacts spectraux |

---

## Infrastructure technique

**Machine** — Ubuntu 24.04, 20 cœurs, 30 Go RAM, RTX 5070 Laptop (8 Go VRAM, sm_120 Blackwell)
→ GPU incompatible avec PyTorch stable — fallback CPU automatique ✅

**Environnements conda**

| Env | Usage |
|---|---|
| `forensics` | Deepfake + worker_import + supervisord + Flask, CPU |
| `forensics_nightly` | GPU RTX 5070, fine-tuning (nightly cu128) |
| `nlp_pipeline` | Worker NLP + Worker réseau |

---

## Points techniques importants

**schema.py v4** — fichier UNIQUE dans `SCHEMA/schema.py`.
Tous les scripts résolvent le chemin avec fallback `../../SCHEMA/` → `../` → `./`.
Supprimer toutes les anciennes copies dans WORKER/IMPORT/, WORKER/NLP/, WORKER/NETWORK/.

**Champ `sync` sur `media`** — ajouté en v4. Migration MongoDB requise sur les docs existants
(voir section Migration ci-dessus).

**`reuse.post_ids`** — lien entre un doc `media` et ses posts associés.
Rempli par `worker_import` via `patch_post_media()` côté posts.
Le network_worker v4 lit ce champ pour créer les relations `A_MEDIA` dans Neo4j.
⚠ `patch_media_reuse()` n'est pas appelé par `worker_import` — `reuse.post_ids` peut être vide
sur les anciens docs. La relation `A_MEDIA` ne sera créée que si le champ est présent.

**mongojob multiprocessing** — `workers = 8` dans `ai_forensics.cfg`.
Le claim atomique garantit qu'aucun job n'est traité deux fois.

**FP-first** — Calibration minimise faux positifs. Pénalité quadratique si modèle score haut sur du réel.

**Warning sklearn** — Synthbuster entraîné avec 1.7.1, env courant a 1.8.0. Régler en réentraînant dans `forensics_nightly`.

**HuggingFace token** — `~/.huggingface/token` (chmod 600). Jamais dans le code.

**Louvain 0 lien** — Normal sur le dataset actuel (29 comptes sans coordination directe).

**Dépréciation GDS** — `gds.graph.project.cypher` génère un warning. Fonctionne. Migration non urgente.

---

## Commandes de référence

### Worker import
```bash
conda activate nlp_pipeline
cd ~/AI-FORENSICS/WORKER/IMPORT
python worker_import.py                              # watch continu sur INPUT/
python worker_import.py --source ~/INPUT/ --dry-run  # test sans écriture
python worker_import.py --once                       # one-shot
```

### Pipeline deepfake — mode MongoDB jobs
```bash
conda activate forensics
cd ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE
python detect_ai_pipeline-v4.0.3.py --mongojob           # 8 workers (cfg)
python detect_ai_pipeline-v4.0.3.py --mongojob --workers 4
python detect_ai_pipeline-v4.0.3.py --mongojob --verbose
```

### Maintenance base MongoDB
```bash
cd ~/AI-FORENSICS/WORKER/IMPORT
python purge_mongodb.py --dry-run
python purge_mongodb.py --reset-deepfake
python purge_mongodb.py --all --yes
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

### Worker réseau v4
```bash
conda activate nlp_pipeline
cd ~/AI-FORENSICS/WORKER/NETWORK

# Stream continu (défaut)
python network_worker.py --backfill

# Mode projet — purge Neo4j + réinjection filtrée
python network_worker.py --projet ProJet0 -p TIKTOK_crypto_2026-03-25

# Mode projet sans purge (ajout d'un nouveau projet)
python network_worker.py --projet ProJet0 --add

# Détection campagnes manuelle
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

1. **Migration MongoDB** — initialiser `sync` sur les docs `media` existants (voir section Migration)

2. **Vérifier `reuse.post_ids`** — s'assurer que `worker_import` appelle bien `patch_media_reuse()`
   pour remplir ce champ (actuellement absent de `build_post_doc`). Sans ça, les relations
   `A_MEDIA` ne seront pas créées en Neo4j pour les médias existants.

3. **Supprimer les copies de `schema.py`** dans WORKER/IMPORT/, WORKER/NLP/, WORKER/NETWORK/
   (remplacées par la résolution centralisée vers SCHEMA/)

4. **Tester network_worker v4** avec données réelles — vérifier nœuds Media + A_MEDIA dans Neo4j

5. **Tester worker NLP** avec données réelles — fonctionnel ?

6. **Déployer `supervision_server.py`** :
   ```bash
   conda activate forensics && pip install flask
   supervisorctl -c supervisord.conf reread && update
   # → http://localhost:5050
   ```

7. **Réentraîner Synthbuster** dans `forensics_nightly`
   (résout le warning sklearn 1.7.1 vs 1.8.0)

8. **Enrichir `ALT/`** avec OpenFake (FLUX 1.1-pro, MJ v7, GPT Image 1)
   `huggingface.co/datasets/ComplexDataLab/OpenFake`

9. **Migrer GDS** — `gds.graph.project.cypher` → nouvelle syntaxe (non urgent)

---

## Requêtes Neo4j utiles (Browser)

```cypher
// Vue d'ensemble campagnes
MATCH (a:Account)-[:A_PUBLIÉ]->(p:Post)-[:APPARTIENT_À]->(n:Narrative)
RETURN n.label, count(DISTINCT a) AS comptes, count(DISTINCT p) AS posts,
       collect(DISTINCT a.platform) AS plateformes
ORDER BY posts DESC

// Comptes amplificateurs (PageRank)
MATCH (a:Account) WHERE a.pagerank_score IS NOT NULL
RETURN a.username, a.platform, a.pagerank_score, a.community_id
ORDER BY a.pagerank_score DESC LIMIT 20

// Doublons cross-plateforme
MATCH (a1:Account)-[:A_PUBLIÉ]->(p1:Post)-[:EST_DOUBLON_DE]->(p2:Post)<-[:A_PUBLIÉ]-(a2:Account)
WHERE a1.platform <> a2.platform
RETURN a1.username, a1.platform, a2.username, a2.platform, count(*) AS doublons
ORDER BY doublons DESC

// Hashtags les plus utilisés [v3]
MATCH (p:Post)-[:HAS_HASHTAG]->(h:Hashtag)
RETURN h.name, count(p) AS nb_posts, collect(DISTINCT p.platform) AS plateformes
ORDER BY nb_posts DESC LIMIT 20

// Sons viraux coordonnés TikTok [v4]
MATCH (p:Post) WHERE p.music_author IS NOT NULL AND p.music_author <> ''
RETURN p.music_author, count(p) AS nb_posts, collect(DISTINCT p.source_project) AS projets
ORDER BY nb_posts DESC LIMIT 20

// Médias réutilisés sur plusieurs comptes [v4]
MATCH (p1:Post)-[:A_MEDIA]->(m:Media)<-[:A_MEDIA]-(p2:Post)
WHERE p1 <> p2
MATCH (a1:Account)-[:A_PUBLIÉ]->(p1)
MATCH (a2:Account)-[:A_PUBLIÉ]->(p2)
WHERE a1 <> a2
RETURN m.mongo_id, m.type, count(DISTINCT a1) AS comptes, m.reuse_count
ORDER BY comptes DESC LIMIT 10

// Posts deepfake par type [v3]
MATCH (p:Post)-[r:IS_DEEPFAKE]->(d:Deepfake)
RETURN d.type, count(p) AS nb_posts, avg(r.score) AS score_moyen
ORDER BY nb_posts DESC

// Comptes suspects selon scrapper (influence_score élevé + bot) [v4]
MATCH (a:Account)-[:A_PUBLIÉ]->(p:Post)
WHERE p.is_bot_suspected = true OR p.influence_score > 0.5
RETURN a.username, a.platform,
       avg(p.influence_score) AS influence_moy,
       count(p) AS nb_posts
ORDER BY influence_moy DESC LIMIT 20
```
