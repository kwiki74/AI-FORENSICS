# Guide d'installation — AI Forensics Pipeline
## Environnement complet : pipeline deepfake + NLP + infrastructure

**Version** : v5.0 · Mars 2026 — deux environnements conda séparés + Anaconda  
**Système cible** : Ubuntu 22.04 / 24.04 LTS  
**Dossier racine du projet** : `~/AI-FORENSICS/`

---

## Sommaire

1. [Vue d'ensemble des environnements](#1-vue-densemble-des-environnements)
2. [Prérequis système](#2-prérequis-système)
3. [Installation d'Anaconda](#3-installation-danaconda)
4. [Environnement conda `forensics`](#4-environnement-conda-forensics)
5. [Environnement conda `nlp_pipeline`](#5-environnement-conda-nlp_pipeline)
6. [Installation de MongoDB 8.0](#6-installation-de-mongodb-80)
7. [Installation de Neo4j](#7-installation-de-neo4j)
8. [Authentification HuggingFace](#8-authentification-huggingface)
9. [Récupération du modèle Synthbuster](#9-récupération-du-modèle-synthbuster)
10. [Récupération / création du modèle SwinV2 OpenFake](#10-récupération--création-du-modèle-swinv2-openfake)
11. [Vérification de l'installation](#11-vérification-de-linstallation)
12. [Configuration des fichiers `.env`](#12-configuration-des-fichiers-env)
13. [Structure des dossiers du projet](#13-structure-des-dossiers-du-projet)

---


# 0. GIT

```
cd
git clone --recurse-submodules https://github.com/kwiki74/AI-FORENSICS.git

```


## 1. Vue d'ensemble des environnements

Le pipeline utilise **deux environnements conda distincts** pour isoler les dépendances incompatibles :

| Environnement       | Workers                                                   | Python | PyTorch            |
| ------------------- | --------------------------------------------------------- | ------ | ------------------ |
| `forensics`         | detect_ai_pipeline + worker_import + supervisord          | 3.11   | 2.6.0 CPU stable   |
| `nlp_pipeline`      | nlp_worker + network_worker (+ neo4j + campaign_detector) | 3.11   | 2.6.0 (CPU ou GPU) |
| `forensics_nightly` | Fine-tuning GPU RTX 50xx uniquement                       | 3.11   | nightly cu128      |

**Pourquoi deux environnements ?**

- `nlp_pipeline` requiert `sentence-transformers>=5.x` et `transformers>=5.x` pour les modèles de sentiment et d'embedding. Ces versions entrent en conflit avec les versions requises par le pipeline deepfake (`transformers>=4.40`).
- `forensics` utilise `scikit-learn` via Synthbuster avec des contraintes de version différentes de celles du worker NLP.
- `supervisord` tourne dans `forensics` et orchestre les deux environnements — chaque worker est lancé avec le binaire Python de son propre environnement.

---

## 2. Prérequis système

```bash
sudo apt update && sudo apt upgrade -y

# Dépendances système obligatoires
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
| GPU | *(optionnel)* | NVIDIA CUDA 12.x (non RTX 50xx avec PyTorch stable) |

---

## 3. Installation d'Anaconda

> Si Anaconda est déjà installé sur la machine, passer directement à la section 4. Vérifier avec `conda --version`.

```bash
# Télécharger Anaconda (Linux x86_64) — version distribution complète
wget https://repo.anaconda.com/archive/Anaconda3-2025.12-2-Linux-x86_64.sh -O anaconda.sh

# Installer (chemin par défaut : ~/anaconda3)
bash anaconda.sh -b -p $HOME/anaconda3

# Initialiser le shell
$HOME/anaconda3/bin/conda init bash
source ~/.bashrc

# Vérification
conda --version
# Doit afficher : conda 24.x.x ou supérieur


#Acceptation des Terms of Service
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

```

> **Différence Anaconda vs Miniconda :** Anaconda inclut une distribution scientifique complète (numpy, scipy, pandas, matplotlib, jupyter…) et l'interface graphique Anaconda Navigator. Miniconda est une installation minimale. Les deux fonctionnent avec ce pipeline ; Anaconda est recommandé pour un poste de travail dédié à l'analyse.

---

## 4. Environnement conda `forensics`

Cet environnement fait tourner le **worker deepfake** (`detect_ai_pipeline-v4.0.3.py`), le **worker import** (`worker_import.py`), et `supervisord`.

### 4.1 Création et PyTorch

```bash
conda create -n forensics python=3.11 -y
conda activate forensics
```

Choisir **une** option PyTorch selon le matériel :

```bash
# Option A — CPU uniquement (VM sans GPU) — recommandé pour installation initiale
conda install pytorch==2.6.0 torchvision cpuonly -c pytorch -y

# Option B — GPU NVIDIA CUDA 12.x (hors RTX 50xx)
# conda install pytorch==2.6.0 torchvision==0.21.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

> **RTX 50xx (Blackwell sm_120) :** PyTorch stable ne supporte pas cette architecture. Utiliser `CUDA_VISIBLE_DEVICES=""` pour forcer le CPU, ou créer l'environnement `forensics_nightly` (section 4.4).

### 4.2 Installation des dépendances

```bash
cd ~/AI-FORENSICS
pip install -r requirements_forensics.txt
```

### 4.3 Vérification de l'environnement `forensics`

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

### 4.4 Variante GPU RTX 50xx — environnement `forensics_nightly`

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

## 5. Environnement conda `nlp_pipeline`

Cet environnement fait tourner le **worker NLP** (`nlp_worker.py`) et le **worker réseau** (`network_worker.py`).

### 5.1 Création et PyTorch

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

### 5.2 Installation des dépendances

`hdbscan` doit être installé via conda avant pip pour éviter les erreurs de compilation :

```bash
conda install -c conda-forge hdbscan -y
pip install -r ~/AI-FORENSICS/requirements_générique.txt
```

### 5.3 Vérification de l'environnement `nlp_pipeline`

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
    print('Relancer : pip install -r ~/AI-FORENSICS/WORKER/NLP/requirements_nlp.txt')
else:
    print()
    print('Tous les packages sont installés ✓')
"
```

---

## 6. Installation de MongoDB 8.0

MongoDB est la **base de vérité** du pipeline. Le mode ReplicaSet est obligatoire pour les Change Streams utilisés par les workers NLP et réseau.

### 6.1 Installation

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

### 6.2 Activation du ReplicaSet

**⚠ Faire AVANT d'activer l'authentification.**

```bash
sudo nano /etc/mongod.conf
```

Ajouter à la fin :

```yaml
replication:
  replSetName: "rs0"
```

```bash
sudo systemctl restart mongod

mongosh
```

```javascript
rs.initiate({
  _id: "rs0",
  members: [{ _id: 0, host: "localhost:27017" }]
})
// Attendre que le prompt affiche : rs0 [direct: primary] test>
exit
```

### 6.3 Création du keyFile

```bash
sudo openssl rand -base64 756 > /tmp/mongodb-keyfile
sudo mv /tmp/mongodb-keyfile /etc/mongodb-keyfile
sudo chown mongodb:mongodb /etc/mongodb-keyfile
sudo chmod 400 /etc/mongodb-keyfile
```

### 6.4 Création des utilisateurs

Se connecter **avant** d'activer l'authentification :

```bash
mongosh
```

> ⚠ Exécuter les commandes **une par une** dans mongosh.

```javascript
// 1
use admin

// 2
db.createUser({
  user: "admin",
  pwd: "kikiland",
  roles: [{ role: "root", db: "admin" }]
})

// 3
use influence_detection

// 4
db.createUser({
  user: "influence_app",
  pwd: "AiForens!cS1",
  roles: [{ role: "readWrite", db: "influence_detection" }]
})

// 5
exit
```

### 6.5 Activation de l'authentification

```bash
sudo nano /etc/mongod.conf
```

```yaml
security:
  authorization: enabled
  keyFile: /etc/mongodb-keyfile
```

> ⚠ Indentation YAML stricte — 2 espaces, pas de tabulation.

```bash
sudo systemctl restart mongod

# Test — guillemets simples obligatoires (le ! est interprété par bash)
mongosh -u influence_app -p 'AiForens!cS1' \
    --authenticationDatabase influence_detection \
    --eval "db.runCommand({ping:1})"
```

### 6.6 Création des index MongoDB

> ⚠ Lancer depuis `~/AI-FORENSICS/` pour que `schema.py` trouve le `.env`.

```bash
conda activate forensics
cd ~/AI-FORENSICS
python -c "
from SCHEMA.schema import get_db, create_indexes
db = get_db(
    host='localhost',
    port=27017,
    user='influence_app',
    password='AiForens!cS1',
    db_name='influence_detection',
    auth_db='influence_detection'
)
create_indexes(db)
print('Index créés ✓')
"
```

### 6.7 Erreur `E11000 duplicate key — hash_md5: null`

Si le worker import remonte cette erreur lors de l'insertion de médias, l'index `hash_md5_1` a été créé sans `partialFilterExpression` (ancienne définition). Le corriger en base :

```bash
mongosh -u influence_app -p 'AiForens!cS1' \
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

> Le `schema.py` patché détecte automatiquement ce cas et recréé l'index correctement à chaque appel de `create_indexes()`.

---

## 7. Installation de Neo4j

Neo4j est utilisé pour l'analyse des relations entre comptes et la détection de campagnes coordonnées.

### 7.1 Installation

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

### 7.2 Configuration initiale

```bash
cypher-shell -u neo4j -p neo4j
```

```cypher
ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO 'influence2026!';
```

Quitter avec **Ctrl+D**. Interface web : http://localhost:7474

### 7.3 Installation de GDS (Graph Data Science) — optionnel mais recommandé

```bash
find /var/lib/neo4j -name "neo4j-graph-data-science-*.jar" 2>/dev/null
sudo cp /var/lib/neo4j/products/neo4j-graph-data-science-*.jar /var/lib/neo4j/plugins/

sudo nano /etc/neo4j/neo4j.conf
# Ajouter :
# dbms.security.procedures.unrestricted=gds.*
# dbms.security.procedures.allowlist=gds.*

sudo systemctl restart neo4j
sudo systemctl status neo4j

```

> Si GDS n'est pas installé, positionner `skip_gds = true` dans `WORKER/NETWORK/network_pipeline.cfg`.

---

## 8. Authentification HuggingFace (optionnel)

```bash
mkdir -p ~/.huggingface
echo "hf_VOTRE_TOKEN_ICI" > ~/.huggingface/token
chmod 600 ~/.huggingface/token
```

> Créer un token sur https://huggingface.co (lecture seule suffit). Ne jamais committer le token.

---

## 9. Récupération du modèle Synthbuster (noramlent pris en charge lors du premier GIT)

```bash
cd ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE
git clone https://github.com/qbammey/synthbuster synthbuster
```

Copier les fichiers de modèle entraîné depuis l'ancien PC :

```bash
scp USER@ANCIEN_PC:~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE/synthbuster/models/model_jpeg.joblib \
    ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE/synthbuster/models/
scp USER@ANCIEN_PC:~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE/synthbuster/models/config_jpeg.json \
    ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE/synthbuster/models/

# Créer les liens symboliques si absents
cd ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE/synthbuster/models/
ln -sf model_jpeg.joblib model.joblib
ln -sf config_jpeg.json config.json
```

---

## 10. Récupération / création du modèle SwinV2 OpenFake

**Le modèle est automatiquement télécharger lors de la première exécution sur script detect_ai_pipeline-v4.0.3.py. Il sera télécharger dans un sous dossier de anaconda3.**


Optionnel : Fine-tuner SwinV2 depuis le backbone de base

```bash
conda activate forensics
cd ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE

# CPU (forcer désactivation GPU pour éviter le warning sm_120)
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

**Résultats obtenus sur notre fine-tuning de référence :**

| Métrique | Acceptable | Bon | Notre résultat |
|---|---|---|---|
| F1 | > 0.65 | > 0.75 | **0.943** |
| Precision | > 0.70 | > 0.80 | **0.938** |
| MCC | > 0.35 | > 0.50 | **0.886** |

---
# 11. création du fichier .env

Le fichier.env, à la racine de la solution, va contenir l'ensemble des carentials.

```bash
# créer le fichier à partir du modele fourni 
cd ~/AI-FORENSICS
cp .env.example .env


#modifier le fichier.env en inscivant les mots de passe déterminés plus haut
nano .env
```

A titre d'explement, voici le contenu du fichier .env.exemple
```c
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





---
## 12. Vérification de l'installation

### Test deepfake one-shot

Pour rappel, lors du **premier** démarrage le worker va télécharger des modèles de données. Le temps d’exécution est donc plus important.
```bash
conda activate forensics
cd ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE

# CUDA_VISIBLE_DEVICES="" force le CPU et supprime le warning sm_120 (RTX 50xx)
CUDA_VISIBLE_DEVICES="" python detect_ai_pipeline-v4.0.3.py --mongojob --workers 1 --verbose
```

**Sortie attendue :** `Modèles prêts : 3/3, échecs : aucun`

il est possible que certain warning soient affichés. ceux ci-dessous ne sont pas impactant pour le fonctionnement des script : 
```bash
/home/xxx/anaconda3/envs/forensics/lib/python3.11/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: 'libc10_cuda.so: cannot open shared object file: No such file or directory'If you don't plan on using image functionality from `torchvision.io`, you can ignore this warning. Otherwise, there might be something wrong with your environment. Did you have `libjpeg` or `libpng` installed before building `torchvision` from source?
  warn(
  
/home/xxx/anaconda3/envs/forensics/lib/python3.11/site-packages/torchvision/datapoints/__init__.py:12: UserWarning: The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta. While we do not expect major breaking changes, some APIs may still change according to user feedback. Please submit any feedback you may have in this issue: https://github.com/pytorch/vision/issues/6753, and you can also check out https://github.com/pytorch/vision/issues/7319 to learn more about the APIs that we suspect might involve future changes. You can silence this warning by calling torchvision.disable_beta_transforms_warning().
  warnings.warn(_BETA_TRANSFORMS_WARNING)
  
/home/xxx/anaconda3/envs/forensics/lib/python3.11/site-packages/torchvision/transforms/v2/__init__.py:54: UserWarning: The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta. While we do not expect major breaking changes, some APIs may still change according to user feedback. Please submit any feedback you may have in this issue: https://github.com/pytorch/vision/issues/6753, and you can also check out https://github.com/pytorch/vision/issues/7319 to learn more about the APIs that we suspect might involve future changes. You can silence this warning by calling torchvision.disable_beta_transforms_warning().
  warnings.warn(_BETA_TRANSFORMS_WARNING)

```


### Test worker NLP

```bash
conda activate nlp_pipeline
cd ~/AI-FORENSICS/WORKER/NLP
python nlp_worker.py --dry-run
```

L'option `--dry-run` permet de simuler un fonctionnement normal sans rien modifier en base.
**Sortie attendue :** 
```bash
2026-04-03 11:00:58 [INFO    ] nlp_worker — Logs fichier → /home/kwiki/AI-FORENSICS/WORKER/NLP/../../logs/nlp_worker.log (max 5 Mo, 5 backups)
2026-04-03 11:00:58 [INFO    ] nlp_worker — Configuration chargée : /home/kwiki/AI-FORENSICS/WORKER/NLP/nlp_pipeline.cfg
2026-04-03 11:00:58 [INFO    ] nlp_worker — === Worker NLP démarré (dry_run=True) ===
2026-04-03 11:00:58 [INFO    ] nlp_worker — 🔤 Clustering auto activé (idle=60s, min_cluster_size=5)
2026-04-03 11:00:58 [INFO    ] nlp_worker — Worker ID : nlp_worker_25261
2026-04-03 11:00:58 [INFO    ] nlp_worker — MongoDB connecté : influence_detection
2026-04-03 11:00:59 [INFO    ] nlp_worker — SentimentAnalyzer prêt (device=cpu)
2026-04-03 11:00:59 [INFO    ] embeddings — EmbeddingEngine initialisé (model=paraphrase-multilingual-MiniLM-L12-v2, dedup_threshold=0.95, device=cpu)
2026-04-03 11:00:59 [INFO    ] nlp_worker — EmbeddingEngine prêt (model=paraphrase-multilingual-MiniLM-L12-v2, dedup_threshold=0.95, dedup_limit=5000)
2026-04-03 11:00:59 [INFO    ] nlp_worker.heartbeat — ♥ heartbeat démarré (interval=60s, worker_id=nlp_worker_25261)
2026-04-03 11:00:59 [INFO    ] nlp_worker — Écoute Change Streams sur : posts, comments
2026-04-03 11:00:59 [INFO    ] nlp_worker — Change Stream ouvert
^C2026-04-03 11:01:34 [WARNING ] nlp_worker — Signal 2 reçu — arrêt propre en cours…

```
### Test MongoDB

```bash
mongosh -u influence_app -p 'AiForens!cS1' \
    --authenticationDatabase influence_detection \
    --eval "db.runCommand({ping:1})"
```

**Sortie attendue :** 
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
cypher-shell -u neo4j -p 'VOTRE_MOT_DE_PASSE' \
    "MATCH (n) RETURN count(n) AS total_noeuds"
```

**Sortie attendue :** 

```bash
    "MATCH (n) RETURN count(n) AS total_noeuds"
+--------------+
| total_noeuds |
+--------------+
| 0            |
+--------------+

1 row
ready to start consuming query after 99 ms, results consumed after another 0 ms

```

**Warnings normaux et sans impact :**

| Warning | Cause | Action |
|---|---|---|
| `libc10_cuda.so: cannot open shared object file` | Extension CUDA absente sur VM sans GPU | Ignorer |
| `torchvision.datapoints [...] still Beta` | APIs bêta internes à torchvision | Ignorer |
| `ViTImageProcessor is now loaded as a fast processor` | Comportement transformers 4.x/5.x | Ignorer |
| `unauthenticated requests to the HF Hub` | Token HuggingFace non configuré | Configurer token (section 8) ou ignorer |
| `resource_tracker: leaked semaphore` | Artefact multiprocessing — disparaît une fois RESULT/ créé | Créer le dossier RESULT/ |
| `NVIDIA RTX 50xx sm_120 not compatible` | GPU Blackwell non supporté par PyTorch stable | Utiliser `CUDA_VISIBLE_DEVICES=""` |

---

## 13. Structure des dossiers du projet


**A METTRE A JOUR**

```
~/AI-FORENSICS/
├── CONTEXT/
│   ├── CONTEXT_NEXT_CHAT.md        ← contexte projet (pour reprise de session)
│   ├── Neo4j_Guide_Analyse.md      ← requêtes Cypher d'analyse
│   └── pipeline_architecture.html  ← schéma interactif de l'architecture
│
├── DATA_IN/                        ← JSON scrappés (inbox worker import)
│
├── SCHEMA/
│   └── schema.py                   ← schéma MongoDB v3 + patch hash_md5_1 (source canonique)
│
├── SUPERVISOR/
│   ├── supervisord.conf            ← configuration supervisord (2 envs, 4 workers)
│   ├── launch_workspace.sh
│   ├── terminator_layout.conf
│   ├── INSTALL_WORKSPACE.md
│   └── t1_supervision.sh … t5_reseau.sh
│
├── WORKER/
│   ├── DETECT_AI_PIPLINE/          ← env : forensics
│   │   ├── detect_ai_pipeline-v4.0.3.py
│   │   ├── ai_forensics.cfg
│   │   ├── fine_tune_swinv2.py
│   │   ├── calib_report_v4.json
│   │   ├── requirements_forensics.txt  ← dépendances env forensics
│   │   ├── swinv2_openfake/
│   │   ├── synthbuster/
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
│   │   └── requirements_nlp.txt    ← dépendances env nlp_pipeline
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
└── .env                            ← credentials (ne pas committer dans git)
```




## 14. environements web

### simple : 

```bash
conda create -n www python=3.11 -y
conda activate www

pip install streamlit
pip install neo4j
pip install python-dotenv
pip install pymongo


```


# 15 . paramètre worker input

afin de paramétrer le dossier d'INPUT du worker import, il faut modifier son fichier de config ~/AI-FORENSICS/WORKER/IMPORT/worker_import.cfg :

```bash
# Mode source directe : pointe sur le DOSSIER_INPUT.
# Le worker scanne récursivement ce dossier SANS déplacer les fichiers.
# Laisser vide pour utiliser le mode inbox/ normal.
source_dir = /chemin_de_votre_dossier_input
```