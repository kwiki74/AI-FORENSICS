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
    ├── Worker deepfake  ← WORKER/DETECT_AI_PIPLINE/detect_ai_pipeline-v4.0.3.py  ✅
    ├── Worker NLP       ← WORKER/NLP/nlp_worker.py  ✅ v5
    │       ↓  (auto post-idle 60s)
    │   narrative_clustering.py  (HDBSCAN + UMAP + TF-IDF — intégré dans nlp_worker)
    │       ↓
    │   MongoDB collection narratives + posts.nlp.narrative_id
    └── Worker réseau    ← WORKER/NETWORK/network_worker.py  ✅ v4
                                  ↓  (détection auto post-idle 30s)
                             campaign_detector.py  (6 signaux dont Louvain + PageRank)
                                  ↓
                             Neo4j + MongoDB collection campaigns
```

**MongoDB** = source de vérité. **Neo4j** = relations et campagnes coordonnées.
**GDS** = plugin Neo4j installé ✅ — Louvain, PageRank, Betweenness, BFS.

---

## Structure des dossiers (racine : `~/AI-FORENSICS/`)

```
~/AI-FORENSICS/
├── CONTEXT/
│   └── CONTEXT_NEXT_CHAT.md
├── INPUT/
├── logs/
│   ├── worker_import.log
│   ├── import_cursor.json                ← NE PAS SUPPRIMER
│   ├── archives/
│   ├── detect_ai_pipeline_YYYYMMDD.log
│   ├── nlp_worker.log
│   └── network_worker.log
├── SCHEMA/
│   └── schema.py                         ← ✅ v4 — référence UNIQUE (pas de copies ailleurs)
├── SUPERVISOR/
│   ├── supervisord.conf · launch_workspace.sh · terminator_layout.conf
│   ├── t1_supervision.sh · t2_import.sh · t3_deepfake.sh · t4_nlp.sh · t5_reseau.sh
│   └── supervision_server.py             ← ⏳ À DÉPLOYER (port 5050)
└── WORKER/
    ├── DETECT_AI_PIPLINE/
    │   ├── detect_ai_pipeline-v4.0.3.py  ← ✅ --mongojob multiprocessing
    │   ├── ai_forensics.cfg
    │   ├── swinv2_openfake/
    │   └── synthbuster/models/
    ├── IMPORT/
    │   ├── worker_import.py              ← ✅ v1.5
    │   ├── worker_import.cfg
    │   └── mongo_status.py               ← ✅ monitoring MongoDB
    ├── NETWORK/
    │   ├── network_worker.py             ← ✅ v4
    │   ├── neo4j_client.py               ← ✅ v4
    │   ├── campaign_detector.py
    │   └── network_pipeline.cfg
    └── NLP/
        ├── nlp_worker.py                 ← ✅ v5 — clustering auto intégré
        ├── narrative_clustering.py       ← ✅ schema centralisé corrigé
        ├── embeddings.py                 ← ✅ inchangé
        ├── sentiment.py                  ← ✅ inchangé
        ├── nlp_pipeline.cfg
        └── requirements_nlp.txt
```

---

## État actuel (mars 2026)

### ✅ Opérationnel

- **worker_import v1.5** — schema centralisé, champs v4, `posts.source` correctement rempli
- **detect_ai_pipeline v4.0.3** — mode `--mongojob` OK
- **schema.py v4** — `new_media(source=)`, `patch_post_nlp(embedding=)`, `patch_comment_nlp(embedding=)`, `patch_media_sync()`
- **network_worker v4** — écoute posts/comments/accounts/media, mode `--projet`
- **neo4j_client v4** — nœuds :Media, :Hashtag, :Deepfake
- **nlp_worker v5** — sentiment + embedding + clustering narratif auto post-idle 60s
- **narrative_clustering** — schema centralisé corrigé, utilisé par nlp_worker v5
- **mongo_status.py** — monitoring collections + pipeline statuses

### Clustering narratif (nlp_worker v5)

`narrative_clustering.py` n'a **plus besoin d'être lancé séparément**.
Le `nlp_worker` le déclenche automatiquement après 60s d'inactivité Change Stream.
- Si une passe est en cours → flag "relance unique" posé, exécutée à la fin
- `--skip-clustering` pour désactiver
- Paramètres dans `nlp_pipeline.cfg` section `[worker]`

---

## ⚠ Migrations requises avant prochain lancement

**1. Champ `sync` manquant sur media existants :**
```javascript
use influence_detection
db.media.updateMany(
  { sync: { $exists: false } },
  { $set: { "sync.neo4j": false, "sync.elasticsearch": false, "sync.synced_at": null } }
)
```

**2. `posts.source` vide sur les 279 posts existants** (bug corrigé, nouveaux imports OK) :
Options : purge + réimport propre, ou laisser tel quel si non critique.

---

## Résolution schema.py — règle commune

Tous les workers utilisent la même chaîne :
```
../../SCHEMA/schema.py  →  ../schema.py  →  ./schema.py
```
**Ne pas avoir de copies de `schema.py` dans les sous-dossiers WORKER/.**

---

## Format JSON scrapper → MongoDB

| Champ JSON | → MongoDB |
|---|---|
| `desc` | `posts.text.content` |
| `hashtags` | `posts.text.hashtags` |
| `music.author` | `posts.platform_specific.music_author` [v4] |
| `cover` | `posts.platform_specific.cover_url` [v4] |
| `influence_score` | `posts.scrapper_signals.influence_score` [v4] |
| `is_bot_suspected` | `posts.scrapper_signals.is_bot_suspected` [v4] |
| `stats.plays` | `engagement.views` |
| `stats.favorites` | `engagement.saves` |
| Arborescence INPUT/ | `posts.source.{project, scan, user}` |

---

## Modèle Neo4j v4

```
(:Account)-[:A_PUBLIÉ]->(:Post)
(:Account)-[:A_COMMENTÉ]->(:Post)
(:Account)-[:A_FORWARDÉ {count}]->(:Post)
(:Post)-[:EST_DOUBLON_DE]->(:Post)
(:Post)-[:APPARTIENT_À]->(:Narrative)
(:Post)-[:HAS_HASHTAG]->(:Hashtag)
(:Post)-[:IS_DEEPFAKE {score}]->(:Deepfake)
(:Post)-[:A_MEDIA]->(:Media)
```

---

## Commandes de référence

```bash
# Worker import
conda activate nlp_pipeline && cd ~/AI-FORENSICS/WORKER/IMPORT
python worker_import.py                    # watch continu
python mongo_status.py --verbose           # état MongoDB

# Deepfake
conda activate forensics && cd ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE
python detect_ai_pipeline-v4.0.3.py --mongojob

# NLP (clustering auto intégré)
conda activate nlp_pipeline && cd ~/AI-FORENSICS/WORKER/NLP
python nlp_worker.py --backfill            # traite existant + stream + clustering auto
python nlp_worker.py --skip-clustering     # sans clustering
# narrative_clustering.py peut toujours être lancé manuellement si besoin :
python narrative_clustering.py --min-cluster-size 5 --dry-run

# Réseau
conda activate nlp_pipeline && cd ~/AI-FORENSICS/WORKER/NETWORK
python network_worker.py --backfill
python network_worker.py --projet ProJet0 [-p ProJet1]   # purge + réinjection
python network_worker.py --projet ProJet0 --add           # ajout sans purge
python campaign_detector.py --min-score 0.30

# MongoDB
mongosh --host localhost --port 27017 -u influence_app -p 'AiForens!cS1' \
        --authenticationDatabase influence_detection

# Neo4j : http://localhost:7474  —  neo4j / influence2026!

# Supervisord
cd ~/AI-FORENSICS/SUPERVISOR
supervisorctl -c supervisord.conf status
```

---

## Configuration nlp_pipeline.cfg — paramètres clustering v5

```ini
[worker]
clustering_idle_seconds     = 60   # délai inactivité avant clustering (défaut: 60)
clustering_min_cluster_size = 5    # taille min HDBSCAN (défaut: 5)
clustering_umap_components  = 50   # dimensions UMAP (défaut: 50)
```

---

## ❌ À faire

1. **Migration MongoDB** — champ `sync` sur media existants (voir ci-dessus)
2. **Corriger `posts.source`** — 279 posts avec `source: {}` (purge + réimport ou laisser)
3. **`reuse.post_ids`** — `worker_import` ne remplit pas ce champ → relations `A_MEDIA` absentes pour médias existants. À corriger dans une prochaine session.
4. **Supprimer copies de `schema.py`** dans sous-dossiers WORKER/
5. **Tester nlp_worker v5** end-to-end : vérifier `nlp.embedding` sauvegardé + clustering déclenché
6. **Déployer `supervision_server.py`** (port 5050)
7. **Réentraîner Synthbuster** (warning sklearn 1.7.1 vs 1.8.0)
8. **Migrer GDS** `gds.graph.project.cypher` → nouvelle syntaxe (non urgent)

---

## Requêtes Neo4j utiles

```cypher
// Campagnes
MATCH (a:Account)-[:A_PUBLIÉ]->(p:Post)-[:APPARTIENT_À]->(n:Narrative)
RETURN n.label, count(DISTINCT a) AS comptes, count(DISTINCT p) AS posts
ORDER BY posts DESC

// Hashtags coordonnés
MATCH (p:Post)-[:HAS_HASHTAG]->(h:Hashtag)
RETURN h.name, count(p) AS nb_posts, collect(DISTINCT p.platform) AS plateformes
ORDER BY nb_posts DESC LIMIT 20

// Sons viraux TikTok
MATCH (p:Post) WHERE p.music_author IS NOT NULL AND p.music_author <> ''
RETURN p.music_author, count(p) AS nb_posts ORDER BY nb_posts DESC LIMIT 20

// Médias réutilisés
MATCH (p1:Post)-[:A_MEDIA]->(m:Media)<-[:A_MEDIA]-(p2:Post)
WHERE p1 <> p2
MATCH (a1:Account)-[:A_PUBLIÉ]->(p1), (a2:Account)-[:A_PUBLIÉ]->(p2)
WHERE a1 <> a2
RETURN m.type, count(DISTINCT a1) AS comptes ORDER BY comptes DESC LIMIT 10

// Deepfakes par type
MATCH (p:Post)-[r:IS_DEEPFAKE]->(d:Deepfake)
RETURN d.type, count(p) AS nb_posts, avg(r.score) AS score_moyen ORDER BY nb_posts DESC

// Comptes amplificateurs (PageRank)
MATCH (a:Account) WHERE a.pagerank_score IS NOT NULL
RETURN a.username, a.platform, a.pagerank_score ORDER BY a.pagerank_score DESC LIMIT 20

// Doublons cross-plateforme
MATCH (a1:Account)-[:A_PUBLIÉ]->(p1:Post)-[:EST_DOUBLON_DE]->(p2:Post)<-[:A_PUBLIÉ]-(a2:Account)
WHERE a1.platform <> a2.platform
RETURN a1.username, a1.platform, a2.username, a2.platform, count(*) AS doublons
ORDER BY doublons DESC
```
