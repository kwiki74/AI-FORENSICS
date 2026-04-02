# Guide d'installation — AI Forensics Pipeline
## Environnement complet : pipeline deepfake + NLP + infrastructure

**Version** : v4.4 · Mars 2026 — environnement unique `forensics` + Anaconda  
**Système cible** : Ubuntu 22.04 / 24.04 LTS  
**Dossier racine du projet** : `~/AI-FORENSICS/`

---

## Sommaire

1. [Vue d'ensemble de l'environnement](#1-vue-densemble-de-lenvironnement)
2. [Prérequis système](#2-prérequis-système)
3. [Installation d'Anaconda](#3-installation-danaconda)
4. [Environnement conda `forensics`](#4-environnement-conda-forensics)
5. [Installation de MongoDB 8.0](#5-installation-de-mongodb-80)
6. [Installation de Neo4j](#6-installation-de-neo4j)
7. [Authentification HuggingFace](#7-authentification-huggingface)
8. [Récupération du modèle Synthbuster](#8-récupération-du-modèle-synthbuster)
9. [Récupération / création du modèle SwinV2 OpenFake](#9-récupération--création-du-modèle-swinv2-openfake)
10. [Vérification de l'installation](#10-vérification-de-linstallation)
11. [Configuration des fichiers `.env`](#11-configuration-des-fichiers-env)
12. [Structure des dossiers du projet](#12-structure-des-dossiers-du-projet)

---

## 1. Vue d'ensemble de l'environnement

Le pipeline utilise désormais un **environnement conda unique** `forensics` pour l'ensemble des workers :

| Environnement | Usage | Python | PyTorch |
|---|---|---|---|
| `forensics` | Tous les workers (deepfake, import, NLP, réseau, Streamlit) + supervisord | 3.11 | 2.6.0 CPU stable |
| `forensics_nightly` | Fine-tuning GPU uniquement (RTX 50xx Blackwell) | 3.11 | nightly cu128 |

> **Compatibilité `sentence-transformers` et pipeline deepfake :** `sentence-transformers>=2.7.0` est compatible avec `transformers>=4.40.0`. La contrainte qui justifiait deux environnements (ST==5.x exigeant transformers==5.x) n'existe plus avec les versions `>=` libres. Tous les workers coexistent dans un seul environnement.

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
wget https://repo.anaconda.com/archive/Anaconda3-latest-Linux-x86_64.sh -O anaconda.sh

# Installer (chemin par défaut : ~/anaconda3)
bash anaconda.sh -b -p $HOME/anaconda3

# Initialiser le shell
$HOME/anaconda3/bin/conda init bash
source ~/.bashrc

# Vérification
conda --version
# Doit afficher : conda 24.x.x ou supérieur
```

> **Différence Anaconda vs Miniconda :** Anaconda inclut une distribution scientifique complète (numpy, scipy, pandas, matplotlib, jupyter…) et l'interface graphique Anaconda Navigator. Miniconda est une installation minimale. Les deux fonctionnent avec ce pipeline ; Anaconda est recommandé pour un poste de travail dédié à l'analyse.

---

## 4. Environnement conda `forensics`

Cet environnement unique fait tourner **tous les workers** du pipeline ainsi que `supervisord`.

### 4.1 Création et PyTorch

```bash
# Créer l'environnement
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

### 4.2 Installation des dépendances Python

> ⚠ **Utiliser exclusivement le `requirements.txt` à la racine `~/AI-FORENSICS/`.**  
> Les anciens fichiers fragmentés (`WORKER/DETECT_AI_PIPLINE/requirements.txt` et `WORKER/NLP/requirements_nlp.txt`) sont désormais obsolètes — ils ne couvrent pas l'ensemble des workers et sont la cause d'erreurs `ModuleNotFoundError` sur `pymongo`, `lingua_language_detector`, `hdbscan`, `sentencepiece`, `tiktoken` et d'autres packages NLP.

```bash
cd ~/AI-FORENSICS
pip install -r requirements.txt
```

Le fichier `requirements.txt` unifié couvre l'intégralité des dépendances du pipeline :
- Deepfake : transformers, timm, opencv, imageio, scikit-learn, numba, decord
- NLP : sentence-transformers, lingua-language-detector, umap-learn, hdbscan, **sentencepiece, tiktoken**
- Infrastructure : pymongo, neo4j, supervisor, streamlit

**Si l'environnement a été installé avec les anciens requirements fragmentés**, compléter avec :

```bash
conda activate forensics
pip install \
    pymongo>=4.6.0 \
    lingua-language-detector>=2.0.0 \
    hdbscan>=0.8.33 \
    umap-learn>=0.5.6 \
    sentence-transformers>=2.7.0 \
    sentencepiece>=0.1.99 \
    tiktoken>=0.6.0 \
    neo4j>=5.0.0 \
    streamlit>=1.35.0 \
    decord>=0.6.0 \
    supervisor>=4.2.0
```

### 4.3 Dossier RESULT (worker deepfake)

```bash
mkdir -p ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE/RESULT
```

### 4.4 Vérification de l'environnement

Vérification complète package par package — toute ligne `MANQUANT` indique un `pip install` à relancer :

```bash
conda activate forensics
python -c "
checks = {
    'torch'                : 'torch',
    'torchvision'          : 'torchvision',
    'transformers'         : 'transformers',
    'sentence-transformers': 'sentence_transformers',
    'timm'                 : 'timm',
    'accelerate'           : 'accelerate',
    'opencv-python'        : 'cv2',
    'Pillow'               : 'PIL',
    'numpy'                : 'numpy',
    'pandas'               : 'pandas',
    'scikit-learn'         : 'sklearn',
    'joblib'               : 'joblib',
    'numba'                : 'numba',
    'imageio'              : 'imageio',
    'decord'               : 'decord',
    'pymongo'              : 'pymongo',
    'neo4j'                : 'neo4j',
    'lingua-language-detector': 'lingua',
    'umap-learn'           : 'umap',
    'hdbscan'              : 'hdbscan',
    'sentencepiece'        : 'sentencepiece',
    'tiktoken'             : 'tiktoken',
    'streamlit'            : 'streamlit',
    'supervisor'           : 'supervisor',
    'python-dotenv'        : 'dotenv',
    'rich'                 : 'rich',
    'colorlog'             : 'colorlog',
    'psutil'               : 'psutil',
}
import importlib
ok, ko = [], []
for name, mod in checks.items():
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, '__version__', '?')
        ok.append(f'  OK  {name:<35} {ver}')
    except ImportError:
        ko.append(f'  MANQUANT  {name}')
print('=== ENV FORENSICS — vérification ===')
for l in ok: print(l)
if ko:
    print()
    print('=== PACKAGES MANQUANTS ===')
    for l in ko: print(l)
else:
    print()
    print('Tous les packages sont installés ✓')
"
```

Sortie attendue sur VM sans GPU (installation complète) :

```
=== ENV FORENSICS — vérification ===
  OK  torch                               2.6.0
  OK  transformers                        4.x.x
  OK  sentence-transformers               2.x.x
  OK  pymongo                             4.x.x
  OK  lingua-language-detector            2.x.x
  OK  hdbscan                             0.8.x
  OK  umap-learn                          0.5.x
  ...
  Tous les packages sont installés ✓
```

Si des packages apparaissent dans `=== PACKAGES MANQUANTS ===`, relancer :

```bash
pip install -r ~/AI-FORENSICS/requirements.txt
```

Le warning suivant est **normal et sans impact** sur VM sans GPU :

```
UserWarning: Failed to load image Python extension: 'libc10_cuda.so: cannot open shared object file'
```

torchvision cherche l'extension CUDA pour l'I/O image accélérée GPU. Le pipeline utilise PIL et OpenCV — cette extension n'est pas utilisée.

### 4.5 Variante GPU RTX 50xx (Blackwell — environnement `forensics_nightly`)

> Uniquement si votre machine possède un GPU RTX 5070/5080/5090. PyTorch stable ne supporte pas l'architecture sm_120. Cet environnement est uniquement nécessaire pour le fine-tuning SwinV2.

```bash
# Cloner l'environnement forensics de base
conda create --name forensics_nightly --clone forensics
conda activate forensics_nightly

# Remplacer PyTorch par la version nightly cu128
python -m pip uninstall torch torchvision -y
python -m pip install --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Vérification GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

---

## 5. Installation de MongoDB 8.0

MongoDB est la **base de vérité** du pipeline. Le mode ReplicaSet est obligatoire pour les Change Streams (mécanisme utilisé par les workers NLP et réseau pour réagir aux nouvelles données en temps réel).

### 5.1 Installation

```bash
# Clé GPG officielle MongoDB
curl -fsSL https://www.mongodb.org/static/pgp/server-8.0.asc | \
    sudo gpg -o /usr/share/keyrings/mongodb-server-8.0.gpg --dearmor

# Repo Ubuntu 24.04
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] \
https://repo.mongodb.org/apt/ubuntu noble/mongodb-org/8.0 multiverse" | \
    sudo tee /etc/apt/sources.list.d/mongodb-org-8.0.list

sudo apt update
sudo apt install -y mongodb-org

# Démarrer et activer au boot
sudo systemctl start mongod
sudo systemctl enable mongod
sudo systemctl status mongod
```

### 5.2 Activation du ReplicaSet dans la configuration

**Qu'est-ce qu'un ReplicaSet ?**
Un ReplicaSet est un groupe de serveurs MongoDB qui maintiennent le même jeu de données. Même avec un seul serveur (notre cas), il est obligatoire pour activer les **Change Streams** — le mécanisme qui permet aux workers NLP et réseau d'être notifiés en temps réel de chaque nouvelle insertion dans MongoDB, sans avoir à interroger la base en boucle (polling).

**⚠ Cette étape doit être faite AVANT d'activer l'authentification.** Si MongoDB démarre sans la config ReplicaSet, `rs.initiate()` échouera avec `NoReplicationEnabled`.

```bash
sudo nano /etc/mongod.conf
```

Ajouter à la fin du fichier :

```yaml
replication:
  replSetName: "rs0"
```

> ⚠ L'indentation YAML est stricte — 2 espaces, pas de tabulation.

```bash
# Redémarrer pour prendre en compte la config
sudo systemctl restart mongod
sudo systemctl status mongod

# Se connecter
mongosh

# Initier le ReplicaSet
rs.initiate({
  _id: "rs0",
  members: [{ _id: 0, host: "localhost:27017" }]
})
```

Attendre quelques secondes. Le prompt mongosh doit changer de :
```
test>
```
vers :
```
rs0 [direct: primary] test>
```

**C'est la seule confirmation nécessaire** — le préfixe `rs0 [direct: primary]` indique que le ReplicaSet est actif et que ce nœud est PRIMARY.

```bash
exit
```

### 5.3 Création du keyFile (requis avec ReplicaSet + auth)

Avec un ReplicaSet, MongoDB exige un **keyFile** pour l'authentification interne entre les membres du replica. Sans lui, le service refusera de démarrer avec l'erreur `security.keyFile is required when authorization is enabled with replica sets`.

```bash
# Créer le keyFile
sudo openssl rand -base64 756 > /tmp/mongodb-keyfile
sudo mv /tmp/mongodb-keyfile /etc/mongodb-keyfile
sudo chown mongodb:mongodb /etc/mongodb-keyfile
sudo chmod 400 /etc/mongodb-keyfile
```

### 5.4 Création des utilisateurs

Se connecter **avant** d'activer l'authentification (MongoDB est encore sans auth à ce stade) :

```bash
mongosh
```

> ⚠ **Exécuter les commandes une par une** dans mongosh — ne pas coller plusieurs commandes ensemble.

```javascript
// Commande 1 — changer de base
use admin

// Commande 2 — créer l'admin (exécuter séparément)
db.createUser({
  user: "admin",
  pwd: "VOTRE_MOT_DE_PASSE_ADMIN",
  roles: [{ role: "root", db: "admin" }]
})

// Commande 3 — changer de base
use influence_detection

// Commande 4 — créer l'utilisateur applicatif (exécuter séparément)
db.createUser({
  user: "influence_app",
  pwd: "AiForens!cS1",
  roles: [{ role: "readWrite", db: "influence_detection" }]
})

// Commande 5 — quitter
exit
```

### 5.5 Activation de l'authentification

```bash
sudo nano /etc/mongod.conf
```

La section `security` doit contenir les deux lignes suivantes :

```yaml
security:
  authorization: enabled
  keyFile: /etc/mongodb-keyfile
```

> ⚠ L'indentation YAML est critique — 2 espaces devant `authorization` et `keyFile`, pas de tabulation.

```bash
sudo systemctl restart mongod
sudo systemctl status mongod
# Doit afficher : active (running)

# Test de connexion authentifiée
# ⚠ Utiliser des guillemets simples autour du mot de passe
# (le ! dans AiForens!cS1 est interprété par bash avec des guillemets doubles)
mongosh -u influence_app -p 'AiForens!cS1' \
    --authenticationDatabase influence_detection \
    --eval "db.runCommand({ping:1})"
```

### 5.6 Création des index MongoDB

> ⚠ La commande doit être lancée **depuis `~/AI-FORENSICS/`** (racine du projet) afin que `schema.py` trouve le fichier `.env` contenant les credentials. Sans credentials, MongoDB lève `OperationFailure: Command createIndexes requires authentication`.

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

Si `get_db()` n'accepte pas de paramètres positionnels et lit uniquement le `.env`, s'assurer que le fichier est bien présent avant de lancer :

```bash
cat ~/AI-FORENSICS/.env   # vérifier les credentials
cd ~/AI-FORENSICS
python -c "from SCHEMA.schema import get_db, create_indexes; create_indexes(get_db()); print('Index créés ✓')"
```

### 5.7 Erreur `E11000 duplicate key — hash_md5: null` (worker import)

Si le worker import remonte cette erreur lors de l'insertion de médias :

```
E11000 duplicate key error collection: influence_detection.media
index: hash_md5_1 dup key: { hash_md5: null }
```

**Cause :** l'index sur `hash_md5` est défini en `unique: true` sans option `sparse: true`. MongoDB ne tolère qu'un seul document avec `hash_md5: null` dans la collection — le deuxième média sans hash est rejeté.

**Correction — recréer l'index en mode sparse :**

```bash
mongosh -u influence_app -p 'AiForens!cS1' \
    --authenticationDatabase influence_detection
```

```javascript
use influence_detection

// Supprimer l'index problématique
db.media.dropIndex("hash_md5_1")

// Recréer en sparse (les documents avec hash_md5: null sont exclus de l'index)
db.media.createIndex(
  { hash_md5: 1 },
  { unique: true, sparse: true, name: "hash_md5_1" }
)

exit
```

> L'option `sparse: true` exclut de l'index tous les documents où `hash_md5` est absent ou `null`, ce qui permet d'insérer plusieurs médias sans hash tout en conservant l'unicité sur les médias qui ont un hash calculé.

---

## 6. Installation de Neo4j

Neo4j est utilisé pour l'analyse des relations entre comptes et la détection de campagnes coordonnées.

### 6.1 Installation

```bash
# Clé GPG Neo4j
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo gpg --dearmor \
    -o /usr/share/keyrings/neo4j.gpg

echo "deb [signed-by=/usr/share/keyrings/neo4j.gpg] https://debian.neo4j.com stable latest" | \
    sudo tee /etc/apt/sources.list.d/neo4j.list

sudo apt update
sudo apt install -y neo4j

# Démarrer
sudo systemctl start neo4j
sudo systemctl enable neo4j
sudo systemctl status neo4j
```

> Si le fichier `/usr/share/keyrings/neo4j.gpg` existe déjà (installation précédente), répondre `o` à la question "Faut-il réécrire par-dessus ?".

### 6.2 Configuration initiale

```bash
# Se connecter au shell Neo4j pour changer le mot de passe par défaut
cypher-shell -u neo4j -p neo4j
```

Une fois connecté, changer le mot de passe :

```cypher
ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO 'influence2026!';
```

Quitter avec **Ctrl+D**.

L'interface web est également accessible sur **http://localhost:7474** — identifiants par défaut : `neo4j` / `neo4j`, puis changer vers `influence2026!`.

### 6.3 Installation de GDS (Graph Data Science) — optionnel mais recommandé

Le plugin GDS est requis pour les algorithmes Louvain (détection de communautés) et PageRank (mesure d'influence) utilisés dans la détection automatique de campagnes.

```bash
# Vérifier si le jar GDS est disponible
find /var/lib/neo4j -name "neo4j-graph-data-science-*.jar" 2>/dev/null
# Exemple : /var/lib/neo4j/products/neo4j-graph-data-science-2.27.1.jar
```

Si le fichier est trouvé, le copier dans le dossier `plugins` :

```bash
sudo cp /var/lib/neo4j/products/neo4j-graph-data-science-*.jar /var/lib/neo4j/plugins/
```

Si le fichier est absent, le télécharger depuis https://neo4j.com/deployment-center/#gds-tab (version compatible Neo4j 5.x).

Ensuite autoriser le plugin dans la configuration :

```bash
sudo nano /etc/neo4j/neo4j.conf
# Ajouter ou décommenter :
# dbms.security.procedures.unrestricted=gds.*
# dbms.security.procedures.allowlist=gds.*

sudo systemctl restart neo4j
```

> Si GDS n'est pas installé, positionner `skip_gds = true` dans `WORKER/NETWORK/network_pipeline.cfg` pour désactiver les algorithmes de graphe dans la détection automatique.

---

## 7. Authentification HuggingFace

Les modèles HuggingFace (`Organika/sdxl-detector`) sont téléchargés automatiquement au premier lancement. Sans token, les téléchargements sont anonymes — rate limit réduit et warning au démarrage. Ce warning est sans impact fonctionnel.

```bash
# Créer un compte sur https://huggingface.co et générer un token (lecture seule suffit)
mkdir -p ~/.huggingface
echo "hf_VOTRE_TOKEN_ICI" > ~/.huggingface/token
chmod 600 ~/.huggingface/token
```

> Ne jamais mettre le token dans le code source, dans `.bashrc`, ou dans un fichier versionné.

---

## 8. Récupération du modèle Synthbuster

Synthbuster est un détecteur basé sur l'analyse spectrale de Fourier.

```bash
cd ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE

# Cloner le repo Synthbuster
git clone https://github.com/qbammey/synthbuster synthbuster
```

### Copier les fichiers de modèle entraîné

Le repo cloné contient la structure mais pas les modèles entraînés. Copier les 4 fichiers de l'archive `models-synthbuster` dans le dossier `synthbuster/models/` :

```
synthbuster/models/
    model_jpeg.joblib      ← modèle principal (entraîné)
    config_jpeg.json       ← configuration du modèle
    model.joblib           ← lien symbolique → model_jpeg.joblib
    config.json            ← lien symbolique → config_jpeg.json
```

Si les fichiers sont disponibles sur l'ancien PC, les copier via scp :

```bash
scp USER@ANCIEN_PC:~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE/synthbuster/models/model_jpeg.joblib \
    ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE/synthbuster/models/
scp USER@ANCIEN_PC:~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE/synthbuster/models/config_jpeg.json \
    ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE/synthbuster/models/
```

### Vérifier et créer les liens symboliques

```bash
ls -la synthbuster/models/ | grep "model.joblib\|config.json"
# Doit afficher des liens symboliques vers model_jpeg.joblib et config_jpeg.json

# Si les liens sont absents, les créer :
cd synthbuster/models/
ln -sf model_jpeg.joblib model.joblib
ln -sf config_jpeg.json config.json
cd ../..
```

---

## 9. Récupération / création du modèle SwinV2 OpenFake

SwinV2 OpenFake est le modèle central du pipeline deepfake (poids calibré : 0.871). ComplexDataLab (McGill/Mila) n'a pas encore publié les poids fine-tunés sur HuggingFace. Deux options :

### Option A — Copier les poids fine-tunés depuis l'ancien PC (recommandé)

```bash
# Copier le dossier complet depuis l'ancien PC
scp -r USER@ANCIEN_PC:~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE/swinv2_openfake/ \
    ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE/

# Vérifier que le modèle est bien présent
ls -lh ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE/swinv2_openfake/
# Doit contenir model.safetensors (~100 Mo) + config.json

# Copier aussi le .cfg calibré depuis l'ancien PC
scp USER@ANCIEN_PC:~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE/ai_forensics.cfg \
    ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE/
```

### Option B — Fine-tuner SwinV2 depuis le backbone de base

Si aucun modèle fine-tuné n'est disponible, lancer le fine-tuning. Nécessite un dataset REAL/ALT et idéalement un GPU (2-3h par epoch sur CPU avec 70k images).

```bash
conda activate forensics
cd ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE

# CPU
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

## 10. Vérification de l'installation

### Test environnement `forensics` — complet

```bash
conda activate forensics
python -c "
import torch, cv2, PIL, pandas, tqdm, numpy
import transformers, sentence_transformers, timm, accelerate
import sklearn, joblib, numba, imageio
import umap, hdbscan
from lingua import Language
import pymongo, neo4j, streamlit
print('=== ENV FORENSICS (complet) ===')
print('torch                :', torch.__version__)
print('transformers         :', transformers.__version__)
print('sentence-transformers:', sentence_transformers.__version__)
print('sklearn              :', sklearn.__version__)
print('pymongo              :', pymongo.__version__)
print('CUDA dispo           :', torch.cuda.is_available())
print('OK ✓')
"
```

### Test MongoDB

```bash
# ⚠ Guillemets simples obligatoires — le ! dans le mot de passe est
# interprété par bash comme un rappel d'historique avec des guillemets doubles
mongosh -u influence_app -p 'AiForens!cS1' \
    --authenticationDatabase influence_detection \
    --eval "db.runCommand({ping:1})"
```

### Test Neo4j

```bash
cypher-shell -u neo4j -p influence2026! \
    "MATCH (n) RETURN count(n) AS total_noeuds"
```

### Test analyse deepfake one-shot

```bash
conda activate forensics
cd ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE
CUDA_VISIBLE_DEVICES="" python detect_ai_pipeline-v4.0.3.py \
    ~/AI-FORENSICS/DATA_IN \
    --ensemble --workers 2 --verbose \
    --output RESULT/results.csv
```

**Sortie attendue :** les 3 modèles se chargent (`Modèles prêts : 3/3, échecs : aucun`), la barre de progression avance, et un CSV est sauvegardé dans `RESULT/`.

**Warnings normaux et sans impact :**

| Warning | Cause | Action |
|---|---|---|
| `libc10_cuda.so: cannot open shared object file` | Extension CUDA absente sur VM sans GPU | Ignorer |
| `torchvision.datapoints [...] still Beta` | APIs bêta internes à torchvision | Ignorer |
| `ViTImageProcessor is now loaded as a fast processor` | Comportement transformers 4.x/5.x | Ignorer |
| `unauthenticated requests to the HF Hub` | Token HuggingFace non configuré | Configurer token (section 7) ou ignorer |
| `resource_tracker: leaked semaphore` | Artefact multiprocessing — disparaît une fois RESULT/ créé | Créer le dossier RESULT/ |

---

## 11. Configuration des fichiers `.env`

Créer un fichier `.env` à la racine du projet (ne jamais le committer dans git) :

```bash
nano ~/AI-FORENSICS/.env
```

Contenu :

```env
# MongoDB
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_USER=influence_app
MONGO_PASSWORD=AiForens!cS1
MONGO_DB=influence_detection
MONGO_AUTH_DB=influence_detection
```

> Les workers lisent en priorité les fichiers `.cfg` de leur dossier, qui contiennent déjà les credentials. Le `.env` sert de fallback pour `schema.py`.

---

## 12. Structure des dossiers du projet

```
~/AI-FORENSICS/
├── CONTEXT/
│   ├── CONTEXT_NEXT_CHAT.md        ← contexte projet (pour reprise de session)
│   ├── Neo4j_Guide_Analyse.md      ← requêtes Cypher d'analyse
│   └── pipeline_architecture.html  ← schéma interactif de l'architecture
│
├── DATA_IN/                        ← JSON scrappés (inbox worker import)
│   ├── converted_X_crypto_*/
│   ├── converted_TIKTOK_crypto_*/
│   ├── converted_INSTAGRAM_crypto_*/
│   └── converted_TELEGRAM_crypto_*/
│
├── SCHEMA/
│   └── schema.py                   ← schéma MongoDB v3 + fonctions patch_* (source canonique)
│
├── SUPERVISOR/
│   ├── supervisord.conf            ← configuration supervisord (4 workers, env unique forensics)
│   ├── launch_workspace.sh         ← script de démarrage workspace Terminator
│   ├── terminator_layout.conf      ← layout terminal 5 panneaux
│   ├── INSTALL_WORKSPACE.md        ← guide installation supervisord
│   └── t1_supervision.sh … t5_reseau.sh   ← scripts panneaux terminal
│
├── WORKER/
│   ├── DETECT_AI_PIPLINE/
│   │   ├── detect_ai_pipeline-v4.0.3.py   ← script principal deepfake
│   │   ├── ai_forensics.cfg               ← modèles, poids, biais, seuils (calibré)
│   │   ├── fine_tune_swinv2.py            ← fine-tuning SwinV2
│   │   ├── calib_report_v4.json           ← dernier rapport de calibration
│   │   ├── swinv2_openfake/               ← modèle SwinV2 fine-tuné (model.safetensors)
│   │   ├── synthbuster/                   ← repo cloné (models/model.joblib → symlink)
│   │   └── RESULT/                        ← CSV et JSON de résultats d'analyse
│   │
│   ├── IMPORT/
│   │   ├── worker_import.py               ← ingestion JSON → MongoDB
│   │   ├── worker_import.cfg              ← configuration
│   │   ├── mongo_status.py                ← monitoring collections et pipeline
│   │   └── purge_mongodb.py               ← utilitaire de purge
│   │
│   ├── NLP/
│   │   ├── nlp_worker.py                  ← orchestrateur NLP (Change Streams)
│   │   ├── sentiment.py                   ← analyse de sentiment FR/EN
│   │   ├── embeddings.py                  ← embeddings + déduplication
│   │   ├── narrative_clustering.py        ← clustering HDBSCAN
│   │   └── nlp_pipeline.cfg              ← modèles, seuils, logging
│   │
│   └── NETWORK/
│       ├── network_worker.py              ← ETL MongoDB → Neo4j (Change Streams)
│       ├── neo4j_client.py                ← client Neo4j + requêtes Cypher
│       ├── campaign_detector.py           ← détection campagnes coordonnées
│       └── network_pipeline.cfg          ← MongoDB, Neo4j, seuils détection
│
├── WWW/
│   └── forensics_explorer.py       ← interface Streamlit (visualisation)
│
├── requirements.txt                ← dépendances Python unifiées (environnement forensics)
├── logs/                           ← logs centralisés (tous workers via supervisord)
└── .env                            ← credentials (ne pas committer dans git)
```

### Note sur `supervisord.conf` avec l'environnement unique

Tous les workers dans `supervisord.conf` pointent désormais vers le même Python :

```ini
[program:worker_import]
command=%(ENV_HOME)s/anaconda3/envs/forensics/bin/python worker_import.py
...

[program:nlp_worker]
command=%(ENV_HOME)s/anaconda3/envs/forensics/bin/python nlp_worker.py
...

[program:network_worker]
command=%(ENV_HOME)s/anaconda3/envs/forensics/bin/python network_worker.py
...
```

> Si le chemin Anaconda diffère, adapter `anaconda3` selon l'installation réelle (`conda info --base` affiche le chemin de base).
