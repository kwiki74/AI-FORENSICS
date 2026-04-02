# Guide d'installation — AI Forensics Pipeline
## Environnement complet : pipeline deepfake + NLP + infrastructure

**Version** : v4.0 · Mars 2026 — révisé après tests sur VM fraîche  
**Système cible** : Ubuntu 22.04 / 24.04 LTS  
**Dossier racine du projet** : `~/AI-FORENSICS/`

---

## Sommaire

1. [Vue d'ensemble de l'environnement](#1-vue-densemble-de-lenvironnement)
2. [Prérequis système](#2-prérequis-système)
3. [Installation de Miniconda](#3-installation-de-miniconda)
4. [Environnement conda `forensics`](#4-environnement-conda-forensics)
5. [Environnement conda `nlp_pipeline`](#5-environnement-conda-nlp_pipeline)
6. [Faut-il un seul ou deux environnements ?](#6-faut-il-un-seul-ou-deux-environnements-)
7. [Installation de MongoDB 8.0](#7-installation-de-mongodb-80)
8. [Installation de Neo4j](#8-installation-de-neo4j)
9. [Authentification HuggingFace](#9-authentification-huggingface)
10. [Récupération du modèle Synthbuster](#10-récupération-du-modèle-synthbuster)
11. [Récupération / création du modèle SwinV2 OpenFake](#11-récupération--création-du-modèle-swinv2-openfake)
12. [Vérification de l'installation](#12-vérification-de-linstallation)
13. [Configuration des fichiers `.env`](#13-configuration-des-fichiers-env)
14. [Structure des dossiers du projet](#14-structure-des-dossiers-du-projet)

---

## 1. Vue d'ensemble de l'environnement

Le pipeline utilise **deux environnements conda distincts**, pour des raisons de compatibilité des dépendances :

| Environnement | Usage | Python | PyTorch |
|---|---|---|---|
| `forensics` | Worker deepfake + worker import + supervisord | 3.11 | 2.6.0 CPU stable |
| `forensics_nightly` | Fine-tuning GPU (RTX 50xx Blackwell uniquement) | 3.11 | nightly cu128 |
| `nlp_pipeline` | Worker NLP + worker réseau/Neo4j | 3.11 | 2.10.0 + CUDA |

> **Note sur la fusion des environnements :** Un seul environnement est techniquement possible, mais déconseillé. Le worker NLP requiert `sentence-transformers==5.x` et `transformers==5.x`, en conflit potentiel avec certaines versions requises par Synthbuster (sklearn) et les modèles HuggingFace du pipeline deepfake. La séparation évite des régressions silencieuses difficiles à diagnostiquer.

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
| RAM | 8 Go | 16 Go (30 Go si `--workers 2` en parallèle avec NLP) |
| Disque | 15 Go libres | 30 Go (modèles + datasets + médias) |
| GPU | *(optionnel)* | NVIDIA CUDA 12.x (non RTX 50xx avec PyTorch stable) |

---

## 3. Installation de Miniconda

```bash
# Télécharger Miniconda (Linux x86_64)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# Installer
bash miniconda.sh -b -p $HOME/miniconda3

# Initialiser le shell
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc

# Vérification
conda --version
```

> Si Anaconda est déjà installé, adapter les chemins `miniconda3` → `anaconda3` dans les configs supervisord.

---

## 4. Environnement conda `forensics`

Cet environnement fait tourner le **worker deepfake** (`detect_ai_pipeline-v4.0.2.py`), le **worker import**, et `supervisord`.

```bash
# Créer l'environnement
conda create -n forensics python=3.11 -y
conda activate forensics

# PyTorch CPU (machines sans GPU ou GPU non compatible CUDA stable)
# ⚠ Ne pas spécifier de version pour torchvision — conda résout la version compatible
conda install pytorch==2.6.0 torchvision cpuonly -c pytorch -y

# Dépendances Python du pipeline
cd ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE
pip install -r requirements.txt

# Dépendances supplémentaires (schéma MongoDB, supervisord)
pip install pymongo python-dotenv supervisor

# Créer le dossier RESULT (nécessaire pour la sauvegarde des CSV)
mkdir -p ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE/RESULT
```

### Vérification

```bash
python -c "
import torch, cv2, transformers, PIL, pandas, tqdm, sklearn, joblib, numba, imageio, timm
print('torch       :', torch.__version__)
print('transformers:', transformers.__version__)
print('CUDA dispo  :', torch.cuda.is_available())
print('OK ✓')
"
```

Sortie attendue sur VM sans GPU :

```
torch       : 2.6.0
transformers: 5.3.0
CUDA dispo  : False
OK ✓
```

Le warning suivant est **normal et sans impact** sur VM sans GPU :

```
UserWarning: Failed to load image Python extension: 'libc10_cuda.so: cannot open shared object file'
```

torchvision cherche l'extension CUDA pour l'I/O image accélérée GPU. Elle n'existe pas sur une installation CPU-only. Le pipeline utilise PIL et OpenCV — cette extension n'est pas utilisée.

### Variante GPU RTX 50xx (Blackwell — environnement `forensics_nightly`)

> Uniquement si votre machine possède un GPU RTX 5070/5080/5090. PyTorch stable ne supporte pas l'architecture sm_120.

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

## 5. Environnement conda `nlp_pipeline`

Cet environnement fait tourner le **worker NLP** (`nlp_worker.py`) et le **worker réseau** (`network_worker.py`).

### Correction préalable du requirements_nlp.txt

Le fichier `requirements_nlp.txt` est un `pip freeze` qui contient une ligne avec un chemin de build local invalide sur toute autre machine. Il faut la corriger avant l'installation :

```bash
# Corriger la ligne packaging qui pointe vers un chemin local invalide
sed -i 's|packaging @ file:///home/task_176104885106445/conda-bld/packaging_1761049078006/work|packaging>=23.0|' \
    ~/AI-FORENSICS/WORKER/NLP/requirements_nlp.txt

# Vérifier la correction
grep "packaging" ~/AI-FORENSICS/WORKER/NLP/requirements_nlp.txt
# Doit afficher : packaging>=23.0
```

### Installation

```bash
conda create -n nlp_pipeline python=3.11 -y
conda activate nlp_pipeline

# Installer toutes les dépendances NLP
pip install -r ~/AI-FORENSICS/WORKER/NLP/requirements_nlp.txt

# Installer le driver Neo4j (absent du requirements_nlp.txt — à ajouter manuellement)
pip install neo4j

# Ajouter neo4j au requirements pour les prochaines installations
echo "neo4j>=5.0.0" >> ~/AI-FORENSICS/WORKER/NLP/requirements_nlp.txt
```

### Vérification

```bash
python -c "
import sentence_transformers, transformers, pymongo, neo4j
print('sentence-transformers:', sentence_transformers.__version__)
print('transformers         :', transformers.__version__)
print('pymongo              :', pymongo.__version__)
print('OK ✓')
"
```

---

## 6. Faut-il un seul ou deux environnements ?

**Conclusion : non, deux environnements sont nécessaires.**

Les raisons de la séparation `forensics` / `nlp_pipeline` :

- `sentence-transformers==5.x` requiert `transformers==5.x`, qui modifie des APIs internes utilisées par les modèles HuggingFace du pipeline deepfake (comportement des processors ViT/Siglip).
- Le pipeline deepfake (`forensics`) utilise `scikit-learn` via Synthbuster, avec une version figée (1.7.x). Le worker NLP requiert sklearn 1.8.x. Les deux versions coexistent mal dans un même environnement.
- `supervisord` est installé dans `forensics` et orchestre les deux environnements — chaque worker est lancé avec le binaire Python de son propre environnement (voir `supervisord.conf`).

---

## 7. Installation de MongoDB 8.0

MongoDB est la **base de vérité** du pipeline. Le mode ReplicaSet est obligatoire pour les Change Streams (mécanisme utilisé par les workers NLP et réseau pour réagir aux nouvelles données en temps réel).

### 7.1 Installation

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

### 7.2 Activation du ReplicaSet dans la configuration

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

`rs.status()` est une vérification optionnelle qui affiche un bloc détaillé. Si tu la lances, chercher `"stateStr" : "PRIMARY"` dans la section `members`. Si le prompt affiche encore `test>` sans préfixe après quelques secondes, appuyer sur Entrée — le shell se met à jour.

```bash
exit
```

### 7.3 Création du keyFile (requis avec ReplicaSet + auth)

Avec un ReplicaSet, MongoDB exige un **keyFile** pour l'authentification interne entre les membres du replica. Sans lui, le service refusera de démarrer avec l'erreur `security.keyFile is required when authorization is enabled with replica sets`.

```bash
# Créer le keyFile
sudo openssl rand -base64 756 > /tmp/mongodb-keyfile
sudo mv /tmp/mongodb-keyfile /etc/mongodb-keyfile
sudo chown mongodb:mongodb /etc/mongodb-keyfile
sudo chmod 400 /etc/mongodb-keyfile
```

### 7.4 Création des utilisateurs

Se connecter **avant** d'activer l'authentification (MongoDB est encore sans auth à ce stade) :

```bash
mongosh
```

> ⚠ **Exécuter les commandes une par une** dans mongosh — ne pas coller plusieurs commandes ensemble. Si `use admin` et `db.createUser(...)` sont collés et exécutés ensemble, seule la commande `use` sera prise en compte.

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

### 7.5 Activation de l'authentification

```bash
sudo nano /etc/mongod.conf
```

La section `security` doit contenir les deux lignes suivantes :

```yaml
security:
  authorization: enabled
  keyFile: /etc/mongodb-keyfile
```

> ⚠ L'indentation YAML est critique — 2 espaces devant `authorization` et `keyFile`, pas de tabulation. Un espace incorrect désactive l'authentification silencieusement.

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

### 7.6 Création des index MongoDB

```bash
conda activate forensics
cd ~/AI-FORENSICS
python -c "from schema import get_db, create_indexes; create_indexes(get_db())"
```

---

## 8. Installation de Neo4j

Neo4j est utilisé pour l'analyse des relations entre comptes et la détection de campagnes coordonnées.

### 8.1 Installation

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

### 8.2 Configuration initiale

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

### 8.3 Installation de GDS (Graph Data Science) — optionnel mais recommandé

Le plugin GDS est requis pour les algorithmes Louvain (détection de communautés) et PageRank (mesure d'influence) utilisés dans la détection automatique de campagnes.

Le fichier `.jar` est normalement déjà présent dans le dossier `products` de Neo4j :

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

## 9. Authentification HuggingFace

Les modèles HuggingFace (`Organika/sdxl-detector`) sont téléchargés automatiquement au premier lancement. Sans token, les téléchargements sont anonymes — rate limit réduit et warning au démarrage. Ce warning est sans impact fonctionnel.

```bash
# Créer un compte sur https://huggingface.co et générer un token (lecture seule suffit)
mkdir -p ~/.huggingface
echo "hf_VOTRE_TOKEN_ICI" > ~/.huggingface/token
chmod 600 ~/.huggingface/token
```

> Ne jamais mettre le token dans le code source, dans `.bashrc`, ou dans un fichier versionné.

---

## 10. Récupération du modèle Synthbuster

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

> **Point d'attention :** Synthbuster a été entraîné avec scikit-learn 1.7.x. L'environnement `nlp_pipeline` utilise sklearn 1.8.x — ne pas lancer le pipeline deepfake depuis `nlp_pipeline`.

---

## 11. Récupération / création du modèle SwinV2 OpenFake

SwinV2 OpenFake est le modèle central du pipeline deepfake (poids calibré : 0.871). ComplexDataLab (McGill/Mila) n'a pas encore publié les poids fine-tunés sur HuggingFace. Deux options :

### Option A — Copier les poids fine-tunés depuis l'ancien PC (recommandé)

```bash
# Copier le dossier complet depuis l'ancien PC
scp -r USER@ANCIEN_PC:~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE/swinv2_openfake/ \
    ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE/

# Vérifier que le modèle est bien présent
ls -lh ~/AI-FORENSICS/WORKER/DETECT_AI_PIPLINE/swinv2_openfake/
# Doit contenir model.safetensors (~100 Mo) + config.json
```

Vérifier également que `ai_forensics.cfg` pointe sur le modèle local et utilise les poids calibrés :

```bash
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

## 12. Vérification de l'installation

### Test environnement `forensics`

```bash
conda activate forensics
python -c "
import torch, cv2, transformers, PIL, pandas, tqdm, sklearn, joblib, numba, imageio, timm, pymongo
print('=== ENV FORENSICS ===')
print('torch       :', torch.__version__)
print('transformers:', transformers.__version__)
print('pymongo     :', pymongo.__version__)
print('CUDA dispo  :', torch.cuda.is_available())
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
CUDA_VISIBLE_DEVICES="" python detect_ai_pipeline-v4.0.2.py \
    ~/AI-FORENSICS/DATA_IN \
    --ensemble --workers 2 --verbose \
    --output RESULT/results.csv
```

**Sortie attendue :** les 3 modèles se chargent (`Modèles prêts : 3/3, échecs : aucun`), la barre de progression avance, et un CSV est sauvegardé dans `RESULT/`.

**Warnings normaux et sans impact :**

| Warning | Cause | Action |
|---|---|---|
| `libc10_cuda.so: cannot open shared object file` | Extension CUDA absente sur VM sans GPU — non utilisée par le pipeline | Ignorer |
| `torchvision.datapoints [...] still Beta` | APIs bêta internes à torchvision | Ignorer |
| `ViTImageProcessor is now loaded as a fast processor` | Changement de comportement transformers 5.x — différence de score négligeable | Ignorer |
| `unauthenticated requests to the HF Hub` | Token HuggingFace non configuré — modèles déjà en cache | Configurer token (section 9) ou ignorer |
| `resource_tracker: leaked semaphore` | Artefact multiprocessing à l'arrêt sur erreur — disparaît une fois RESULT/ créé | Créer le dossier RESULT/ |

### Test environnement `nlp_pipeline`

```bash
conda activate nlp_pipeline
python -c "
import sentence_transformers, transformers, pymongo, neo4j
print('=== ENV NLP_PIPELINE ===')
print('sentence-transformers:', sentence_transformers.__version__)
print('transformers         :', transformers.__version__)
print('pymongo              :', pymongo.__version__)
print('OK ✓')
"
```

---

## 13. Configuration des fichiers `.env`

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

## 14. Structure des dossiers du projet

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
├── SUPERVISOR/
│   ├── supervisord.conf            ← configuration supervisord (4 workers)
│   ├── launch_workspace.sh         ← script de démarrage workspace Terminator
│   ├── terminator_layout.conf      ← layout terminal 5 panneaux
│   ├── INSTALL_WORKSPACE.md        ← guide installation supervisord
│   └── t1_supervision.sh … t5_reseau.sh   ← scripts panneaux terminal
│
├── WORKER/
│   ├── DETECT_AI_PIPLINE/
│   │   ├── detect_ai_pipeline-v4.0.2.py   ← script principal (dernière version)
│   │   ├── ai_forensics.cfg               ← modèles, poids, biais, seuils (calibré)
│   │   ├── ai_forensics.cfg.bak_*         ← backup avant recalibration
│   │   ├── fine_tune_swinv2.py            ← fine-tuning SwinV2
│   │   ├── requirements.txt               ← dépendances env forensics
│   │   ├── calib_report_v4.json           ← dernier rapport de calibration
│   │   ├── swinv2_openfake/               ← modèle SwinV2 fine-tuné (model.safetensors)
│   │   ├── synthbuster/                   ← repo cloné (models/model.joblib → symlink)
│   │   └── RESULT/                        ← CSV et JSON de résultats d'analyse
│   │
│   ├── IMPORT/
│   │   ├── worker_import.py               ← ingestion JSON → MongoDB
│   │   └── worker_import.cfg              ← configuration
│   │
│   ├── NLP/
│   │   ├── nlp_worker.py                  ← orchestrateur NLP (Change Streams)
│   │   ├── sentiment.py                   ← analyse de sentiment FR/EN
│   │   ├── embeddings.py                  ← embeddings + déduplication
│   │   ├── narrative_clustering.py        ← clustering HDBSCAN
│   │   ├── nlp_pipeline.cfg               ← modèles, seuils, logging
│   │   ├── requirements_nlp.txt           ← dépendances env nlp_pipeline (figées)
│   │   └── schema.py                      ← schéma MongoDB (copie locale)
│   │
│   └── NETWORK/
│       ├── network_worker.py              ← ETL MongoDB → Neo4j (Change Streams)
│       ├── neo4j_client.py                ← client Neo4j + requêtes Cypher
│       ├── campaign_detector.py           ← détection campagnes coordonnées
│       ├── network_pipeline.cfg           ← MongoDB, Neo4j, seuils détection
│       └── schema.py                      ← schéma MongoDB (copie locale)
│
├── schema.py                       ← schéma MongoDB v3 + fonctions patch_*
├── logs/                           ← logs centralisés (tous workers via supervisord)
└── .env                            ← credentials (ne pas committer dans git)
```
