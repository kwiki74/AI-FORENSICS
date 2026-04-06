# Guide d'installation — AI-FORENSICS Pipeline

**Version** : v5.0 · Avril 2026  
**Système cible** : Ubuntu 22.04 / 24.04 LTS  
**Dossier racine du projet** : `~/AI-FORENSICS/`

---

## Sommaire

1. [Vue d'ensemble des environnements](#1-vue-densemble-des-environnements)
2. [Prérequis système](#2-prérequis-système)
3. [Cloner le dépôt](#3-cloner-le-dépôt)
4. [Installation d'Anaconda](#4-installation-danaconda)
5. [Environnement conda `forensics`](#5-environnement-conda-forensics)
6. [Environnement conda `nlp_pipeline`](#6-environnement-conda-nlp_pipeline)
7. [Environnement conda `www`](#7-environnement-conda-www)
8. [Installation de MongoDB 8.0](#8-installation-de-mongodb-80)
9. [Installation de Neo4j](#9-installation-de-neo4j)
10. [Authentification HuggingFace](#10-authentification-huggingface)
11. [Modèles deepfake](#11-modèles-deepfake)
12. [Configuration du fichier `.env`](#12-configuration-du-fichier-env)
13. [Vérification de l'installation](#13-vérification-de-linstallation)
14. [Structure des dossiers du projet](#14-structure-des-dossiers-du-projet)
15. [Paramétrage du dossier d'entrée (worker import)](#15-paramétrage-du-dossier-dentrée-worker-import)

---

## 1. Vue d'ensemble des environnements

Le pipeline utilise **deux environnements conda distincts** pour isoler des dépendances incompatibles :

| Environnement | Workers | Python | PyTorch |
|---|---|---|---|
| `forensics` | detect_ai_pipeline | 3.11 | 2.6.0 CPU stable |
| `nlp_pipeline` | nlp_worker + network_worker + campaign_detector + worker_import + supervisord | 3.11 | 2.6.0 (CPU ou GPU) |
| `www` | forensics_explorer (Streamlit) | 3.11 | — |
| `forensics_nightly` | Fine-tuning GPU RTX 50xx uniquement | 3.11 | nightly cu128 |

**Pourquoi deux environnements séparés ?**

- `nlp_pipeline` requiert `sentence-transformers >= 5.x` et `transformers >= 5.x`, incompatibles avec les versions requises par le pipeline deepfake (`transformers >= 4.40`).
- `forensics` utilise `scikit-learn` via Synthbuster avec des contraintes de version différentes.
- `supervisord` tourne dans `nlp_pipeline` et orchestre les deux environnements — chaque worker est lancé avec le binaire Python de son propre environnement.

---

## 2. Prérequis système

```bash
sudo apt update && sudo apt upgrade -y

sudo apt install -y \
    ffmpeg \
    git \
    curl \
    wget \
    gnupg \
    ca-certificates \
    build-essential \
    libssl-dev \
    libffi-dev

# Vérification
ffmpeg -version
git --version
```

**Configuration matérielle minimale :**

| Composant | Minimum | Recommandé |
|---|---|---|
| OS | Ubuntu 22.04 | Ubuntu 24.04 LTS |
| Python | 3.10 | 3.11 |
| RAM | 8 Go | 16 Go (30 Go si workers NLP + deepfake en parallèle) |
| Disque | 20 Go libres | 40 Go (modèles + datasets + médias) |
| GPU | *(optionnel)* | NVIDIA CUDA 12.x — non compatible RTX 50xx avec PyTorch stable |

---

## 3. Cloner le dépôt

```bash
cd ~
git clone --recurse-submodules https://github.com/kwiki74/AI-FORENSICS.git
```

L'option `--recurse-submodules` est indispensable pour récupérer le sous-module `synthbuster`.

---

## 4. Installation d'Anaconda

> Si Anaconda est déjà installé, passer directement à la section 5. Vérifier avec `conda --version`.

```bash
# Télécharger Anaconda (Linux x86_64)
wget https://repo.anaconda.com/archive/Anaconda3-2025.12-2-Linux-x86_64.sh -O anaconda.sh

# Installer (chemin par défaut : ~/anaconda3)
bash anaconda.sh -b -p $HOME/anaconda3

# Initialiser le shell
$HOME/anaconda3/bin/conda init bash
source ~/.bashrc

# Vérification
conda --version
# Attendu : conda 24.x.x ou supérieur

# Acceptation des Terms of Service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

> **Anaconda vs Miniconda :** Anaconda inclut une distribution scientifique complète (numpy, scipy, pandas, matplotlib, jupyter…). Miniconda est une installation minimale. Les deux fonctionnent avec ce pipeline ; Anaconda est recommandé pour un poste de travail dédié à l'analyse.

---

## 5. Environnement conda `forensics`

Cet environnement fait tourner uniquement le **worker deepfake** (`detect_ai_pipeline-v4.0.3.py`).

### 5.1 Création et PyTorch

```bash
conda create -n forensics python=3.11 -y
conda activate forensics
```

Choisir **une** option PyTorch selon le matériel :

```bash
# Option A — CPU uniquement (recommandé pour installation initiale)
conda install pytorch==2.6.0 torchvision cpuonly -c pytorch -y

# Option B — GPU NVIDIA CUDA 12.x (hors RTX 50xx)
# conda install pytorch==2.6.0 torchvision==0.21.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

> **RTX 50xx (Blackwell sm_120) :** PyTorch stable ne supporte pas cette architecture. Utiliser `CUDA_VISIBLE_DEVICES=""` pour forcer le CPU, ou créer l'environnement `forensics_nightly` (section 5.4).

### 5.2 Installation des dépendances

```bash
cd ~/AI-FORENSICS
pip install -r requirements_forensics.txt
```

### 5.3 Vérification de l'environnement `forensics`

```bash
conda activate forensics
python -c "
checks = {
    'torch'          : 'torch',
    'torchvision'    : 'torchvision',
    'transformers'   : 'transformers',
    'timm'           : 'timm',
    'accelerate'     : 'accelerate',
    'opencv-python'  : 'cv2',
    'Pillow'         : 'PIL',
    'numpy'          : 'numpy',
    'pandas'         : 'pandas',
    'scikit-learn'   : 'sklearn',
    'joblib'         : 'joblib',
    'numba'          : 'numba',
    'imageio'        : 'imageio',
    'decord'         : 'decord',
    'pymongo'        : 'pymongo',
    'python-dotenv'  : 'dotenv',
    'supervisor'     : 'supervisor',
    'psutil'         : 'psutil',
    'rich'           : 'rich',
    'colorlog'       : 'colorlog',
}
import importlib
ok, ko = [], []
for name, mod in checks.items():
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, '__version__', '?')
        ok.append(f'  OK  {name:<30} {ver}')
    except ImportError:
        ko.append(f'  MANQUANT  {name}')
print('=== ENV FORENSICS — vérification ===')
for l in ok: print(l)
if ko:
    print()
    print('=== PACKAGES MANQUANTS ===')
    for l in ko: print(l)
    print()
    print('Relancer : pip install -r ~/AI-FORENSICS/requirements_forensics.txt')
else:
    print()
    print('Tous les packages sont installés ✓')
"
```

### 5.4 Variante GPU RTX 50xx — environnement `forensics_nightly`

> Uniquement pour le fine-tuning SwinV2 sur GPU RTX 5070/5080/5090.

```bash
conda create --name forensics_nightly --clone forensics
conda activate forensics_nightly
python -m pip uninstall torch torchvision -y
python -m pip install --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Vérification GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

---

## 6. Environnement conda `nlp_pipeline`

Cet environnement fait tourner le **worker NLP** (`nlp_worker.py`), le **worker réseau** (`network_worker.py`), le **worker import** (`worker_import.py`).

### 6.1 Création et PyTorch

```bash
conda create -n nlp_pipeline python=3.11 -y
conda activate nlp_pipeline
```

Choisir **une** option PyTorch :

```bash
# Option A — CPU uniquement
conda install pytorch==2.6.0 cpuonly -c pytorch -y

# Option B — GPU NVIDIA CUDA 12.x (hors RTX 50xx) — recommandé pour NLP
# conda install pytorch==2.6.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 6.2 Installation des dépendances

`hdbscan` doit être installé via conda **avant** pip pour éviter les erreurs de compilation :

```bash
conda install -c conda-forge hdbscan -y
pip install -r ~/AI-FORENSICS/requirements_générique.txt
```

### 6.3 Vérification de l'environnement `nlp_pipeline`

```bash
conda activate nlp_pipeline
python -c "
checks = {
    'torch'                    : 'torch',
    'transformers'             : 'transformers',
    'accelerate'               : 'accelerate',
    'sentence-transformers'    : 'sentence_transformers',
    'sentencepiece'            : 'sentencepiece',
    'tiktoken'                 : 'tiktoken',
    'lingua-language-detector' : 'lingua',
    'scikit-learn'             : 'sklearn',
    'scipy'                    : 'scipy',
    'umap-learn'               : 'umap',
    'hdbscan'                  : 'hdbscan',
    'numpy'                    : 'numpy',
    'pymongo'                  : 'pymongo',
    'neo4j'                    : 'neo4j',
    'python-dotenv'            : 'dotenv',
    'tqdm'                     : 'tqdm',
    'psutil'                   : 'psutil',
    'rich'                     : 'rich',
    'colorlog'                 : 'colorlog',
}
import importlib
ok, ko = [], []
for name, mod in checks.items():
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, '__version__', '?')
        ok.append(f'  OK  {name:<30} {ver}')
    except ImportError:
        ko.append(f'  MANQUANT  {name}')
print('=== ENV NLP_PIPELINE — vérification ===')
for l in ok: print(l)
if ko:
    print()
    print('=== PACKAGES MANQUANTS ===')
    for l in ko: print(l)
    print()
    print('Relancer : pip install -r ~/AI-FORENSICS/requirements_générique.txt')
else:
    print()
    print('Tous les packages sont installés ✓')
"
```

---

## 7. Environnement conda `www`

Environnement minimal pour l'interface Streamlit spécifique à AI-FORENSICS. cette interface web est utilisé pour le contrôle en isolé de la solution.

```bash
conda create -n www python=3.11 -y
conda activate www

pip install streamlit
pip install neo4j
pip install python-dotenv
pip install pymongo
pip install psutil
```

Une fois l'installation completement terminer, vous pouvez lancer la page de cette manière : 

```bash
#activation de l'environement
conda activate www

#lancement de la page via streamlit (attention, la commande ne rends pas la main)
streamlit run ~/AI-FORENSICS/WWW/app.py

```



---

## 8. Installation de MongoDB 8.0

MongoDB est la **base de vérité** du pipeline. Le mode ReplicaSet est obligatoire pour les Change Streams utilisés par les workers NLP et réseau.

### 8.1 Installation

```bash
curl -fsSL https://www.mongodb.org/static/pgp/server-8.0.asc | \
    sudo gpg -o /usr/share/keyrings/mongodb-server-8.0.gpg --dearmor

echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] \
https://repo.mongodb.org/apt/ubuntu noble/mongodb-org/8.0 multiverse" | \
    sudo tee /etc/apt/sources.list.d/mongodb-org-8.0.list

sudo apt update
sudo apt install -y mongodb-org
sudo systemctl start mongod
sudo systemctl enable mongod
sudo systemctl status mongod
```

### 8.2 Activation du ReplicaSet

Le **ReplicaSet** est un mécanisme de MongoDB qui maintient plusieurs copies synchronisées d'une même base de données. Dans une configuration classique, un nœud est désigné **primary** (il reçoit toutes les écritures) et les autres sont des **secondary** (ils répliquent les données en temps réel).

Dans le cadre de ce projet, on n'utilise qu'un seul nœud — donc le ReplicaSet ne sert pas à la haute disponibilité. On l'active pour une autre raison : les **Change Streams**. Cette fonctionnalité MongoDB permet à un script Python de s'abonner à un flux d'événements en temps réel ("un document vient d'être inséré dans `posts`") sans avoir à interroger la base en boucle. Les workers NLP et réseau en dépendent pour traiter les contenus au fil de l'eau, dès leur insertion par le worker import. Or MongoDB n'autorise les Change Streams **que** sur une instance en mode ReplicaSet — même à un seul membre.


> ⚠ Effectuer **avant** d'activer l'authentification.

```bash
sudo nano /etc/mongod.conf
```

Ajouter à la fin du fichier :

```yaml
replication:
  replSetName: "rs0"
```

```bash
sudo systemctl restart mongod
mongosh
```

Dans mongosh :

```javascript
rs.initiate({
  _id: "rs0",
  members: [{ _id: 0, host: "localhost:27017" }]
})
// Attendre que le prompt affiche : rs0 [direct: primary] test>
exit
```

### 8.3 Création du keyFile

```bash
sudo openssl rand -base64 756 > /tmp/mongodb-keyfile
sudo mv /tmp/mongodb-keyfile /etc/mongodb-keyfile
sudo chown mongodb:mongodb /etc/mongodb-keyfile
sudo chmod 400 /etc/mongodb-keyfile
```

### 8.4 Création des utilisateurs

Se connecter **avant** d'activer l'authentification :

```bash
mongosh
```

> ⚠ Exécuter les commandes **une par une** dans mongosh.

```javascript
// 1 — Passer sur la base admin
use admin

// 2 — Créer l'utilisateur admin
db.createUser({
  user: "admin",
  pwd: "VOTRE_MOT_DE_PASSE_ADMIN",
  roles: [{ role: "root", db: "admin" }]
})

// 3 — Passer sur la base applicative
use influence_detection

// 4 — Créer l'utilisateur applicatif
db.createUser({
  user: "influence_app",
  pwd: "VOTRE_MOT_DE_PASSE_APP",
  roles: [{ role: "readWrite", db: "influence_detection" }]
})

// 5
exit
```

### 8.5 Activation de l'authentification

```bash
sudo nano /etc/mongod.conf
```

Ajouter :

```yaml
security:
  authorization: enabled
  keyFile: /etc/mongodb-keyfile
```

> ⚠ Indentation YAML stricte — 2 espaces, pas de tabulation.

```bash
sudo systemctl restart mongod

# Test de connexion (guillemets simples obligatoires si le mot de passe contient !)
mongosh -u influence_app -p 'VOTRE_MOT_DE_PASSE_APP' \
    --authenticationDatabase influence_detection \
    --eval "db.runCommand({ping:1})"
```

### 8.6 Création des index MongoDB

> ⚠ Lancer depuis `~/AI-FORENSICS/` pour que `schema.py` trouve le `.env`.

```bash
conda activate forensics
cd ~/AI-FORENSICS
python -c "
from SCHEMA.schema import get_db, create_indexes
db = get_db()
create_indexes(db)
print('Index créés ✓')
"
```

### 8.7 Correction de l'index `hash_md5_1` (si erreur E11000)

Si le worker import remonte `E11000 duplicate key — hash_md5: null`, l'index `hash_md5_1` a été créé sans `partialFilterExpression`. Le corriger :

```bash
mongosh -u influence_app -p 'VOTRE_MOT_DE_PASSE_APP' \
    --authenticationDatabase influence_detection
```

```javascript
use influence_detection
db.media.dropIndex("hash_md5_1")
db.media.createIndex(
  { hash_md5: 1 },
  {
    unique: true,
    partialFilterExpression: { hash_md5: { $type: "string" } },
    name: "hash_md5_1"
  }
)
exit
```

> Le `schema.py` patché détecte et corrige automatiquement ce cas à chaque appel de `create_indexes()`.

---

## 9. Installation de Neo4j

Neo4j est utilisé pour l'analyse des relations entre comptes et la détection de campagnes coordonnées.

### 9.1 Installation

```bash
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo gpg --dearmor \
    -o /usr/share/keyrings/neo4j.gpg

echo "deb [signed-by=/usr/share/keyrings/neo4j.gpg] https://debian.neo4j.com stable latest" | \
    sudo tee /etc/apt/sources.list.d/neo4j.list

sudo apt update
sudo apt install -y neo4j
sudo systemctl start neo4j
sudo systemctl enable neo4j
sudo systemctl status neo4j
```

### 9.2 Configuration initiale

```bash
cypher-shell -u neo4j -p neo4j
```

Généralement, le changement de mot de passe est demandé automatiquement. Si ce n'est pas le cas : 
```cypher
ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO 'VOTRE_MOT_DE_PASSE_NEO4J';
```

Quitter avec **Ctrl+D**. Interface web disponible sur : http://localhost:7474

### 9.3 Installation de GDS (Graph Data Science)

Le plugin GDS est requis pour la détection de campagnes (Louvain, PageRank). Sans lui, positionner `skip_gds = true` dans `WORKER/NETWORK/network_pipeline.cfg`.

```bash
# Vérifier si le plugin est présent
find /var/lib/neo4j -name "neo4j-graph-data-science-*.jar" 2>/dev/null

# Copier dans le dossier plugins
sudo cp /var/lib/neo4j/products/neo4j-graph-data-science-*.jar /var/lib/neo4j/plugins/

# Autoriser les procédures GDS
sudo nano /etc/neo4j/neo4j.conf
# Ajouter :
# dbms.security.procedures.unrestricted=gds.*
# dbms.security.procedures.allowlist=gds.*

sudo systemctl restart neo4j
```

---

## 10. Authentification HuggingFace

Optionnel, mais recommandé pour éviter les avertissements de rate-limiting.

```bash
mkdir -p ~/.huggingface
echo "hf_VOTRE_TOKEN_ICI" > ~/.huggingface/token
chmod 600 ~/.huggingface/token
```

> Créer un token lecture seule sur https://huggingface.co. Ne jamais committer ce fichier.

---

## 11. Modèles deepfake

### Synthbuster

Le sous-module est cloné automatiquement avec `--recurse-submodules`. Si vous disposez de  fichiers de modèle entraîné, copiez les dans les dossier `~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE/synthbuster/models/` puis effectuez ces commandes :
```bash
# Créer les liens symboliques si absents
cd ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE/synthbuster/models/
ln -sf model_jpeg.joblib model.joblib
ln -sf config_jpeg.json config.json
```

### SwinV2 OpenFake

Le modèle est **téléchargé automatiquement** lors de la première exécution du pipeline deepfake (dans un sous-dossier d'Anaconda).

**Fine-tuning optionnel** — résultats de référence (effectué sur une dataset ~99 000 images): 
F1 = 0.943 · Precision = 0.938 · MCC = 0.886 

```bash
# CPU (recommandé hors GPU Blackwell)
conda activate forensics
cd ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE
CUDA_VISIBLE_DEVICES="" python fine_tune_swinv2.py \
    --data-dir ~/DATASET/calib_dataset \
    --output-dir ./swinv2_openfake \
    --epochs 10 --verbose

# GPU RTX 50xx
conda activate forensics_nightly
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python fine_tune_swinv2.py \
    --data-dir ~/DATASET/calib_dataset \
    --output-dir ./swinv2_openfake \
    --epochs 10 --batch-size 16 --num-workers 8 --verbose
```

---

## 12. Configuration du fichier `.env`

Le fichier `.env` à la racine du projet centralise toutes les credentials.

```bash
cd ~/AI-FORENSICS
cp .env.example .env
nano .env   # renseigner les mots de passe définis aux sections 8 et 9
```

Contenu du `.env.example` fourni :

```ini
# --- MongoDB (compte applicatif — utilisé par tous les workers) ---
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_USER=influence_app
MONGO_PASSWORD=CHANGER_MOI
MONGO_DB=influence_detection
MONGO_AUTH_DB=influence_detection

# --- MongoDB (compte admin — utilisé uniquement par schema.py create_indexes) ---
MONGO_ADMIN_USER=admin
MONGO_ADMIN_PASSWORD=CHANGER_MOI

# --- Neo4j (utilisé par le worker NETWORK) ---
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=CHANGER_MOI
```

> ⚠ Le fichier `.env` est exclu du dépôt Git via `.gitignore`. Ne jamais le committer.

---

## 13. Vérification de l'installation

### Test worker deepfake

> Lors du **premier démarrage**, le worker télécharge les modèles HuggingFace. Le temps d'exécution est donc plus long.

```bash
conda activate forensics
cd ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE

# CUDA_VISIBLE_DEVICES="" force le CPU et supprime le warning sm_120 (RTX 50xx)
CUDA_VISIBLE_DEVICES="" python detect_ai_pipeline-v4.0.3.py --mongojob --workers 1 --verbose
```

**Sortie attendue :** `Modèles prêts : 3/3, échecs : aucun`

### Test worker NLP

```bash
conda activate nlp_pipeline
cd ~/AI-FORENSICS/WORKER/NLP
python nlp_worker.py --dry-run
```

L'option `--dry-run` simule le fonctionnement sans modifier la base.

**Sortie attendue (extrait) :**

```
[INFO] nlp_worker — === Worker NLP démarré (dry_run=True) ===
[INFO] nlp_worker — MongoDB connecté : influence_detection
[INFO] nlp_worker — SentimentAnalyzer prêt (device=cpu)
[INFO] nlp_worker — EmbeddingEngine prêt
[INFO] nlp_worker — Écoute Change Streams sur : posts, comments
```

### Test MongoDB

```bash
mongosh -u influence_app -p 'VOTRE_MOT_DE_PASSE_APP' \
    --authenticationDatabase influence_detection \
    --eval "db.runCommand({ping:1})"
```

**Sortie attendue :** `{ ok: 1, ... }`
```bash
    --authenticationDatabase influence_detection \
    --eval "db.runCommand({ping:1})"
{
  ok: 1,
  '$clusterTime': {
    clusterTime: Timestamp({ t: 1775207042, i: 1 }),
    signature: {
      hash: Binary.createFromBase64('MLAAZtn4lYYv7B61A98lxPyL/0Y=', 0),
      keyId: Long('7624445898277257223')
    }
  },
  operationTime: Timestamp({ t: 1775207042, i: 1 })
}

```
### Test Neo4j

```bash
cypher-shell -u neo4j -p 'VOTRE_MOT_DE_PASSE_NEO4J' \
    "MATCH (n) RETURN count(n) AS total_noeuds"
```

**Sortie attendue :**

```
+--------------+
| total_noeuds |
+--------------+
| 0            |
+--------------+
1 row
```

### Warnings normaux et sans impact

| Warning | Cause | Action |
|---|---|---|
| `libc10_cuda.so: cannot open shared object file` | Extension CUDA absente en mode CPU | Ignorer |
| `torchvision.datapoints [...] still Beta` | APIs bêta internes à torchvision | Ignorer |
| `ViTImageProcessor is now loaded as a fast processor` | Comportement transformers 4.x/5.x | Ignorer |
| `unauthenticated requests to the HF Hub` | Token HuggingFace non configuré | Configurer (section 10) ou ignorer |
| `resource_tracker: leaked semaphore` | Artefact multiprocessing — disparaît si RESULT/ existe | Créer le dossier `RESULT/` |
| `NVIDIA RTX 50xx sm_120 not compatible` | GPU Blackwell non supporté par PyTorch stable | Utiliser `CUDA_VISIBLE_DEVICES=""` |

---

## 14. Structure des dossiers du projet

```
~/AI-FORENSICS/
├── CONTEXT/
│   ├── CONTEXT_NEXT_CHAT.md        ← contexte projet (reprise de session)
│   ├── Neo4j_Guide_Analyse.md      ← requêtes Cypher d'analyse
│   └── pipeline_architecture.html  ← schéma interactif de l'architecture
│
├── SCHEMA/
│   └── schema.py                   ← schéma MongoDB v3 (source canonique)
│
├── SUPERVISOR/
│   ├── supervisord.conf            ← configuration supervisord
│   ├── launch_workspace.sh
│   ├── terminator_layout.conf
│   └── t1_supervision.sh … t5_reseau.sh
│
├── WORKER/
│   ├── DETECT_AI_PIPLINE/          ← env : forensics
│   │   ├── detect_ai_pipeline-v4.0.3.py
│   │   ├── ai_forensics.cfg
│   │   ├── fine_tune_swinv2.py
│   │   ├── calib_report_v4.json
│   │   ├── requirements_forensics.txt
│   │   ├── swinv2_openfake/        ← modèle fine-tuné (non versionné)
│   │   ├── synthbuster/            ← sous-module git
│   │   └── RESULT/
│   │
│   ├── IMPORT/                     ← env : forensics
│   │   ├── worker_import.py
│   │   ├── worker_import.cfg
│   │   ├── mongo_status.py
│   │   └── purge_mongodb.py
│   │
│   ├── NLP/                        ← env : nlp_pipeline
│   │   ├── nlp_worker.py
│   │   ├── sentiment.py
│   │   ├── embeddings.py
│   │   ├── narrative_clustering.py
│   │   ├── nlp_pipeline.cfg
│   │   └── requirements_nlp.txt
│   │
│   └── NETWORK/                    ← env : nlp_pipeline
│       ├── network_worker.py
│       ├── neo4j_client.py
│       ├── campaign_detector.py
│       └── network_pipeline.cfg
│
├── WWW/
│   └── forensics_explorer.py       ← interface Streamlit
│
├── logs/
├── .env                            ← credentials (ne pas committer)
├── .env.example                    ← modèle fourni
└── .gitignore
```

---

## 15. Paramétrage du dossier d'entrée (worker import)

Pour configurer le dossier source du worker import, modifier `~/AI-FORENSICS/WORKER/IMPORT/worker_import.cfg` :

```ini
# Mode source directe : pointe sur DOSSIER_INPUT.
# Le worker scanne récursivement ce dossier SANS déplacer les fichiers.
# Laisser vide pour utiliser le mode inbox/ normal.
source_dir = /chemin/vers/votre/dossier_input
```
