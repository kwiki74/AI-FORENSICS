# AI-FORENSICS — Pipeline de détection de médias synthétiques & campagnes d'influence

> Détection automatique d'images et vidéos générées par IA sur les réseaux sociaux,  
> analyse linguistique des publications et identification de campagnes d'influence coordonnées.

**Vidéo de présentation :** https://youtu.be/95AWVQPc-ss


> Travail réalisé dans le cadre du mastère spécialisé cyberdéfense / cybersécurit de **Telecom Paris** [Telecom Paris](https://www.telecom-paris.fr/)
> Pour le cours CYBER721-Réseaux et Organisation de donénes : intelligence et sécurité

> La solution AI-FORENSICS constitue la motié d'une solution gobale, réalisée dans le cadre d'un projet de Cyber Threat Intelligence & OSIONT dont le sujet était **Deeopfakes & Opérations d'Influence**. 

> Les scripts de la solution AI-FORENSIC on été codés avec l'aide de *Claude-Code*.
---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture](#2-architecture)
3. [Choix techniques](#3-choix-techniques)
   - [MongoDB comme source de vérité](#31-mongodb-comme-source-de-vérité)
   - [Neo4j pour l'analyse relationnelle](#32-neo4j-pour-lanalyse-relationnelle)
   - [Change Streams comme bus d'événements](#33-change-streams-comme-bus-dévénements)
   - [Multiprocessing spawn](#34-multiprocessing-spawn-pas-fork)
   - [Secrets centralisés](#35-secrets-centralisés-jamais-dans-le-code)
4. [Stratégie de détection](#4-stratégie-de-détection)
   - [Pourquoi trois modèles orthogonaux](#41-pourquoi-trois-modèles-orthogonaux)
   - [Les trois modèles](#42-les-trois-modèles)
   - [Calibration FP-first](#43-calibration-fp-first)
5. [Schéma de données — schema.py](#5-schéma-de-données--schempy)
6. [Workers](#6-workers)
   - [worker_import.py](#61-worker_importpy)
   - [detect_ai_pipeline-v4.0.py](#62-detect_ai_pipeline-v40py)
   - [nlp_worker.py](#63-nlp_workerpy)
   - [network_worker.py](#64-network_workerpy)
7. [Outils annexes](#7-outils-annexes)
8. [Structure du dépôt](#8-structure-du-dépôt)
9. [Credentials](#9-credentials)

---

## 1. Vue d'ensemble

AI-FORENSICS est le composant back-end d'une solution de détection de campagnes d'influence sur les réseaux sociaux (Instagram, TikTok, Twitter/X, Telegram). Il prend en entrée des publications collectées par un scraper externe, les analyse à plusieurs niveaux, et produit des scores et des structures exploitables dans une interface d'exploration.

La chaîne de traitement complète couvre :

- **Ingestion** — normalisation et insertion en base des publications et médias collectés
- **Détection de médias synthétiques** — images et vidéos générées par IA (FLUX, Midjourney, DALL-E, SDXL, Grok, Ideogram)
- **Analyse linguistique** — sentiment, embeddings sémantiques, clustering narratif
- **Détection de campagnes** — identification de réseaux de comptes coordonnés via analyse de graphe

Le pipeline ne communique avec le scraper que par fichiers (JSON + médias). Le format attendu est décrit dans `spec_interface_ai_forensics.docx`.

---

## 2. Architecture

```
Scraper (externe)
    │
    ▼  JSON + médias
storage/inbox/
    │
    ▼
worker_import.py                    ← normalisation, insertion MongoDB, création des jobs
    │
    ▼  Change Streams (événements en temps réel)
    ├── detect_ai_pipeline-v4.0.py  ← analyse médias (images/vidéos)
    |
    ├── nlp_worker.py               ← analyse textes (sentiment, embeddings, narratives)
    |
    ▼
network_worker.py                   ← ETL MongoDB→Neo4j + détection de campagnes
    │                                 (peux également fonctionner avec les change streams)  
    ▼
Neo4j GDS                           ← graphe de relations, communautés Louvain, PageRank
```

**MongoDB** est la source de vérité opérationnelle. Tous les workers lisent et écrivent dans MongoDB. **Neo4j** est alimenté par le worker réseau pour les analyses relationnelles que MongoDB ne peut pas effectuer efficacement.

---

## 3. Choix techniques

### 3.1 MongoDB comme source de vérité

MongoDB a été retenu pour plusieurs raisons complémentaires :

**Modèle document natif JSON.** Les données issues des scrapers (publications Instagram, tweets, vidéos TikTok, messages Telegram) sont naturellement hétérogènes — chaque plateforme expose ses propres champs. MongoDB absorbe cette diversité sans transformation préalable coûteuse, là où un schéma relationnel rigide aurait nécessité de nombreuses tables de jointure ou des colonnes nullable à profusion.

**Change Streams.** La fonctionnalité Change Streams de MongoDB permet aux workers de s'abonner en temps réel aux insertions et mises à jour, sans polling ni broker de messages externe. Cela supprime une dépendance d'infrastructure (pas de Kafka, pas de RabbitMQ) tout en garantissant la réactivité du pipeline.

**File de jobs interne.** La collection `jobs` remplace un broker léger : le worker deepfake consomme les jobs via un `findOneAndUpdate` atomique, qui garantit qu'un seul worker prend en charge un job donné même en parallèle. Pas de Redis, pas de Celery.

**ReplicaSet obligatoire.** Les Change Streams nécessitent un ReplicaSet initialisé (`rs0`), même à un seul nœud. C'est une exigence de MongoDB, pas un overhead de déploiement inutile.

```
MongoDB influence_detection
├── accounts     — profils de comptes (toutes plateformes)
├── posts        — publications avec sous-docs deepfake, nlp, sync
├── comments     — commentaires et threads de réponses
├── media        — fichiers physiques (hash MD5 + perceptuel pour dédup)
├── narratives   — clusters sémantiques identifiés par le worker NLP
├── campaigns    — campagnes d'influence détectées
└── jobs         — file de traitement interne
```

### 3.2 Neo4j pour l'analyse relationnelle

MongoDB répond mal aux questions du type *« quels comptes partagent les mêmes médias sur plusieurs plateformes ? »* ou *« quelle est la densité de connexion entre ce groupe de comptes ? »* — ces questions nécessitent de traverser un graphe de relations, opération pour laquelle les bases documentaires ne sont pas conçues.

Neo4j est une base de données native graphe : ses entités sont des nœuds (`:Account`, `:Post`, `:Hashtag`, `:Media`, `:Campaign`, `:Project`) et ses relations sont des arêtes indexées et traversables efficacement. Les algorithmes de graphe — détection de communautés (Louvain), centralité (PageRank), chemins courts — s'y exécutent nativement via le plugin GDS (Graph Data Science).

**MongoDB et Neo4j sont complémentaires, pas concurrents :**

| Usage | Base |
|---|---|
| Ingestion, enrichissement, pipeline de traitement | MongoDB |
| Analyse des relations entre comptes | Neo4j |
| Détection de communautés coordonnées | Neo4j GDS (Louvain) |
| Hiérarchisation des amplificateurs | Neo4j GDS (PageRank) |

### 3.3 Change Streams comme bus d'événements

Plutôt qu'un broker de messages dédié, le pipeline exploite les Change Streams natifs de MongoDB. Chaque insertion ou mise à jour dans une collection déclenche une notification en temps réel vers les workers abonnés. Les workers deepfake et NLP réagissent ainsi immédiatement à l'arrivée de nouveaux médias ou publications, sans polling.

### 3.4 Multiprocessing spawn (pas fork)

Le worker deepfake exploite plusieurs processus parallèles (`--workers N`). Le mode `spawn` a été retenu à la place du mode `fork`, habituel sous Linux, pour une raison précise : PyTorch n'est pas thread-safe pour l'inférence en mode CPU. Lors d'un `fork`, les verrous internes de PyTorch (et de MKL/OpenMP) sont copiés dans un état potentiellement verrouillé, ce qui produit des deadlocks ou des résultats d'inférence corrompus.

Avec `spawn`, chaque processus enfant est initialisé de zéro, charge ses propres modèles en mémoire et opère de manière totalement isolée. Contrepartie : la RAM est multipliée par le nombre de workers (~4-5 Go par worker en v4.0 avec 3 modèles).

**Recommandation :** 2 workers avec 16 Go de RAM, 4 workers avec 30 Go.

### 3.5 Secrets centralisés, jamais dans le code

Tous les credentials (MongoDB, Neo4j) sont stockés dans un fichier `.env` non versionné. Chaque script charge ses variables d'environnement via `python-dotenv`. Un `.env.example` documenté est fourni dans le dépôt. 
Le token HuggingFace est stocké dans `~/.huggingface/token` (chmod 600), jamais dans le code ni dans `.bashrc`.

---

## 4. Stratégie de détection

### 4.1 Pourquoi trois modèles orthogonaux

Aucun modèle de détection ne couvre l'ensemble du spectre des générateurs d'images actuels. Les générateurs dominants sur les réseaux sociaux en 2025-2026 sont :

| Générateur | Usage principal | Difficulté de détection |
|---|---|---|
| FLUX (Black Forest Labs) | Contenu malveillant photo-réaliste | Très élevée (18-30% sur méthodes classiques) |
| Midjourney v7 | Contenu artistique et photo-réaliste | Élevée |
| GPT Image 1 / DALL-E | Intégré à ChatGPT, Meta AI | Élevée |
| Stable Diffusion / SDXL | Contenu générique | Modérée |
| Ideogram 3.0 | Affiches, fausses unes avec texte | Modérée |
| Grok-2 | Intégré nativement dans X/Twitter | Élevée |

Un modèle entraîné sur une famille de générateurs est potentiellement aveugle aux signatures d'une autre famille. La solution retenue combine trois modèles dont les **mécanismes de détection sont fondamentalement différents** — ce que l'un manque, les deux autres ont des chances de détecter.

Plusieurs autres modèles ont été évalués (Deep-Fake-Detector-v2, AI-image-detector, ai-vs-real-image-detection) mais écartés après calibration, leurs poids tombant systématiquement à zéro sur les données de réseaux sociaux.

### 4.2 Les trois modèles

| # | Modèle | Mécanisme | Générateurs couverts |
|---|---|---|---|
| 1 | `Organika/sdxl-detector` | CNN entraîné sur artefacts de diffusion | SD, SDXL, FLUX partiel |
| 2 | `swinv2_openfake` | SwinV2 fine-tuné sur dataset OpenFake (McGill/Mila) | FLUX 1.1-pro, MJ v6, DALL-E 3, Grok-2, Ideogram 3 |
| 3 | `synthbuster/synthbuster` | Analyse du spectre de Fourier (sklearn) | Artefacts spectraux — robuste aux générateurs inconnus |

**Synthbuster** ([github.com/qbammey/synthbuster](https://github.com/qbammey/synthbuster)) est développé par Quentin Bammey (Télécom Paris, LTCI). Son approche par analyse fréquentielle est totalement orthogonale aux deux CNN — elle ne nécessite pas d'avoir été entraînée sur le générateur ciblé, ce qui la rend robuste face à des générateurs nouveaux ou inconnus.

**SwinV2 OpenFake** est un transformeur de vision fine-tuné sur le dataset OpenFake (ComplexDataLab/McGill), le seul modèle public entraîné explicitement sur les générateurs de nouvelle génération. Les poids ne sont pas publiés par ComplexDataLab sur HuggingFace : un script de fine-tuning (`fine_tune_swinv2.py`) est fourni pour fine-tuner le backbone `microsoft/swinv2-small-patch4-window16-256` sur votre propre dataset.

Résultats du fine-tuning sur ~99 000 images (dataset de calibration) :

```
F1 = 0.943 | Precision = 0.938 | Recall = 0.948 | MCC = 0.886
```

### 4.3 Calibration FP-first

La calibration ajuste les poids et biais de chaque modèle sur un corpus contrôlé (`REAL/` + `ALT/`). La philosophie est **FP-first** (False Positive first) : minimiser les faux positifs avant tout autre critère.

Dans un contexte de modération et d'analyse d'influence, accuser du contenu authentique d'être généré par IA est analytiquement plus coûteux que rater un contenu synthétique. On préfère rater quelques fakes que d'accuser du contenu réel.

**Critère de calibration :**

```
fp_factor       = (1 - avg_score_REAL)²      ← pénalité quadratique sur les faux positifs
detect_margin   = avg_score_ALT - avg_score_REAL
score_modèle    = max(0, detect_margin) × fp_factor
poids_normalisé = score_modèle / Σ scores
```

Un modèle dont le score moyen sur les images réelles est trop élevé voit son poids réduit, voire annulé. La calibration met à jour automatiquement `ai_forensics.cfg` avec backup horodaté.

```bash
CUDA_VISIBLE_DEVICES="" python detect_ai_pipeline-v4.0.py \
    ./DATASET/calib_dataset --calibrate --workers 2 --verbose
```

---

## 5. Schéma de données — schema.py

`schema.py` est le **contrat de données** du projet. Tous les scripts importent ce module comme unique point d'accès à MongoDB — ce qui garantit la cohérence structurelle des documents quelle que soit leur origine.

Il expose :

**Constructeurs de documents :**

```python
new_account(platform, platform_id, ...)   → dict document accounts
new_post(platform, platform_id, ...)      → dict document posts
new_comment(post_id, ...)                 → dict document comments
new_media(media_type, url_original, ...)  → dict document media
new_narrative(label, keywords, ...)       → dict document narratives
new_campaign(name, platforms)             → dict document campaigns
new_job(job_type, payload, priority)      → dict document jobs
```

**Fonctions de mise à jour partielle (`patch_*`) :**

```python
patch_post_deepfake(result)               → $set enrichissement deepfake sur un post
patch_post_nlp(sentiment, embedding, ...) → $set enrichissement NLP sur un post
patch_post_sync(neo4j, elasticsearch)     → $set flags de synchronisation
patch_media_deepfake(result)              → $set enrichissement deepfake sur un média
patch_media_reuse(post_id, platform)      → $addToSet détection de réutilisation
```

Ces fonctions retournent des opérateurs MongoDB (`$set`, `$push`, `$addToSet`) prêts à être passés à `update_one()` — les workers n'écrivent jamais de documents entiers, ils enrichissent les documents existants.

**Connexion :**

```python
from schema import get_db
db = get_db()  # lit MONGO_HOST, MONGO_PORT, MONGO_USER, MONGO_PASSWORD depuis .env
```

**Index :**

```python
from schema import get_db, create_indexes
create_indexes(get_db())  # idempotent, à appeler au premier démarrage
```

---

## 6. Workers

### 6.1 worker_import.py

Point d'entrée unique des données dans le système. Surveille en continu le répertoire `storage/inbox/`(défini dans le fichier de configuration) et traite les fichiers JSON déposés par le scraper.

**Flux de traitement :**

```
storage/inbox/<post_id>.json
    │
    ├── Validation du format (bloc scrappeurInfo obligatoire)
    ├── Résolution du compte auteur → upsert dans accounts
    ├── Insertion du post → posts
    ├── Pour chaque média attaché :
    │     ├── Calcul hash MD5 (déduplication)
    │     ├── Upsert dans media (réutilisation détectée si hash connu)
    │     └── Création d'un job deepfake_analysis dans jobs
    ├── Création d'un job nlp_analysis dans jobs
    └── Déplacement du JSON vers storage/done/
```

**Traitement différencié post / média.** Les publications textuelles sont insérées dans `posts`. Les médias (images, vidéos) font l'objet d'un document séparé dans `media` avec le chemin local du fichier et une référence vers le post parent. Cette séparation permet de détecter qu'un même fichier physique (même hash MD5 ou perceptuel) a été publié par plusieurs comptes — signal fort de coordination.

**Format d'entrée attendu :** JSON normalisé conforme à `spec_interface_ai_forensics.docx`. Le champ `scrappeurInfo` est obligatoire. Toute valeur inconnue doit être `null` (jamais `""` ni date epoch Unix).

### 6.2 detect_ai_pipeline-v4.0.py

Worker de détection de médias générés par IA. Consomme les jobs `deepfake_analysis` depuis MongoDB.

**Mode opérationnel (`--mongojob`) :**

```
1. Claim atomique du prochain job pending (findOneAndUpdate)
2. Lecture du chemin média depuis le payload du job
3. Détermination du type (image / vidéo)
4. Si vidéo → extraction adaptative de frames
5. Inférence des 3 modèles en parallèle (processus spawn)
6. Correction des biais individuels
7. Fusion pondérée → final_score [0.0–1.0]
8. Calcul de la divergence inter-modèles
9. Écriture du résultat → patch_media_deepfake()
10. Marquage du job : status=done
11. → Job suivant
```

**Extraction adaptative de frames vidéo :**
(paramétré dans le fichier de configuration)

| Durée | Frames extraites |
|---|---|
| < 5 s | 3 (story, GIF) |
| 5–15 s | 5 (Reels, TikTok courts) |
| 15–60 s | 8 (vidéos moyennes) |
| 1–3 min | 12 |
| > 3 min | 16 |

**CPU / GPU.** Le pipeline fonctionne nativement en CPU, déployable sur toute machine sans GPU. Le mode GPU est prévu mais non validé en production (RTX 5070 Blackwell sm_120 non supporté par PyTorch stable au moment du développement).

**Commandes principales :**

```bash
# Analyse d'un dossier
CUDA_VISIBLE_DEVICES="" python detect_ai_pipeline-v4.0.py \
    ./storage --ensemble --workers 2 --verbose

# Mode piloté par MongoDB
CUDA_VISIBLE_DEVICES="" python detect_ai_pipeline-v4.0.py \
    --mongojob --workers 2 --verbose

# Calibration
CUDA_VISIBLE_DEVICES="" python detect_ai_pipeline-v4.0.py \
    ./DATASET/calib_dataset --calibrate --workers 2 --verbose

# Sans Synthbuster (si dépendances manquantes)
CUDA_VISIBLE_DEVICES="" python detect_ai_pipeline-v4.0.py \
    ./storage --no-synthbuster --ensemble
```

### 6.3 nlp_worker.py

Worker d'analyse linguistique des publications. Opère en mode stream sur la collection `posts` via Change Streams — traite chaque nouveau post textuel à mesure de son insertion, sans déclenchement manuel.

**Deux analyses produites par post :**

**Analyse de sentiment** via `cardiffnlp/twitter-roberta-base-sentiment-latest` — modèle RoBERTa fine-tuné spécifiquement sur des tweets. Classe chaque publication sur un axe positif / neutre / négatif. Ce modèle est choisi pour sa robustesse sur le langage informel des réseaux sociaux (abréviations, hashtags, ironie, argot), qui met en échec les modèles entraînés sur des corpus formels.

**Embedding sémantique** via `sentence-transformers` — vecteur de 384 flottants 32 bits (modèle `all-MiniLM-L6-v2`) représentant le sens du texte dans un espace vectoriel à haute dimension. Deux textes sémantiquement proches ont des vecteurs proches géométriquement, indépendamment de leur formulation exacte. La distance entre deux embeddings se mesure par la **similarité cosinus** — insensible à la longueur des textes, donc bien adaptée aux publications de longueurs très variables.

**Clustering narratif** (déclenché périodiquement) : l'algorithme **HDBSCAN** regroupe les publications par proximité sémantique sans hypothèse préalable sur le nombre de clusters. Chaque cluster devient une `narrative` en base, caractérisée par ses mots-clés, les plateformes impliquées et la proportion de médias synthétiques parmi les posts associés.

**Résultats écrits en base :** `patch_post_nlp()` enrichit le document post avec le sentiment score, l'embedding, et la référence narrative.

**Dépendances :**

```
sentence-transformers
transformers (cardiffnlp/twitter-roberta-base-sentiment-latest)
hdbscan
umap-learn
```

### 6.4 network_worker.py

Worker de synchronisation MongoDB → Neo4j et de détection de campagnes d'influence. Opère en mode `--projet` sur demande explicite (pas en stream continu).

**ETL MongoDB → Neo4j :**

Le worker lit les comptes, posts, médias, hashtags et narratives de MongoDB et les projette sous forme de nœuds et relations dans Neo4j :

```
:Account  -[:POSTED]->   :Post
:Post     -[:CONTAINS]-> :Media
:Post     -[:TAGGED]->   :Hashtag
:Post     -[:BELONGS_TO]-> :Narrative
:Account  -[:REPOSTED]-> :Post
:Media    -[:REUSED_BY]-> :Account
```

**Détection de campagnes (campaign_detector) :**

```
1. Algorithme de Louvain (Neo4j GDS)
   → Détection de communautés de comptes densément connectés
   → Signature topologique d'un réseau coordonné

2. PageRank (Neo4j GDS)
   → Hiérarchisation des amplificateurs au sein de chaque communauté

3. Qualification d'une campagne si convergence de :
   ├── Publications coordonnées dans le temps
   ├── Réutilisation de médias identiques sur plusieurs comptes
   ├── Narrative commune identifiée par le worker NLP
   └── Ratio significatif de médias détectés comme synthétiques

4. Matérialisation → nœud :Campaign dans Neo4j
                  → document campaigns dans MongoDB
```

**Dépendances :**

```
neo4j (driver Python)
neo4j-graph-data-science
pymongo
```

---

## 7. Outils annexes

### mongo_status.py

Outil de monitoring de l'état de la base MongoDB. Affiche en temps réel les compteurs de chaque collection, l'état de la queue de jobs (pending / processing / done / error), et les statistiques de détection (distribution des scores, ratio synthetic/real).

```bash
python mongo_status.py
python mongo_status.py --verbose
python mongo_status.py --watch   # rafraîchissement continu
```

### purge_mongodb.py

Outil de remise à zéro sélective de la base. Permet de purger une ou plusieurs collections sans supprimer les index ni la structure. Utile pour recommencer une analyse sur un nouveau corpus sans réinstaller.

```bash
python purge_mongodb.py --collections posts media jobs
python purge_mongodb.py --all          # purge complète (avec confirmation)
python purge_mongodb.py --dry-run      # aperçu sans modification
```

> ⚠️ Opération irréversible. Toujours utiliser `--dry-run` en premier.

---

## 8. Structure du dépôt

```
AI-FORENSICS/
├── WORKER/
│   ├── detect_ai_pipeline-v4.0.py    ← pipeline deepfake (script principal)
│   ├── worker_import.py              ← ingestion JSON → MongoDB
│   ├── nlp_worker.py                 ← analyse linguistique (stream)
│   ├── network_worker.py             ← ETL Neo4j + détection campagnes
│   ├── mongo_status.py               ← monitoring MongoDB
│   ├── purge_mongodb.py              ← remise à zéro sélective
│   ├── fine_tune_swinv2.py           ← fine-tuning SwinV2 (bf16 + gradient checkpointing)
│   ├── prepare_calib_dataset.py      ← préparation dataset de calibration
│   └── synthbuster/                  ← git submodule (github.com/qbammey/synthbuster)
│       └── swinv2_openfake/          ← poids SwinV2 fine-tuné (non versionné)
├── SCHEMA/
│   ├── schema.py                     ← contrat de données MongoDB
│   └── SCHEMA_MONGODB.md             ← documentation schéma
├── CONTEXT/
│   └── ai_forensics.cfg              ← configuration pipeline (modèles, poids, biais)
├── WWW/
│   └── pipeline_architecture.html    ← diagramme architecture interactif
├── .env.example                      ← template variables d'environnement
├── .gitignore
├── .gitmodules
└── requirements.txt
```

---

## 9. Credentials

Les credentials MongoDB et Neo4j sont lus exclusivement depuis `.env` :

```env
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_USER=influence_app
MONGO_PASSWORD=***
MONGO_DB=influence_detection
MONGO_AUTH_DB=influence_detection

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=***
```

---

## Références

- Synthbuster — Q. Bammey, Télécom Paris : https://github.com/qbammey/synthbuster
- Dataset OpenFake (SwinV2) — ComplexDataLab/McGill : https://huggingface.co/datasets/ComplexDataLab/OpenFake
- PEReN — Détection de contenus artificiels : https://www.peren.gouv.fr/perenlab/2025-02-11_ai_summit/
- Cardiff NLP twitter-roberta : https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
