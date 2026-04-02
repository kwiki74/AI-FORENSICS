# fine_tune_swinv2.py — Guide d'utilisation

Script de fine-tuning du modèle **SwinV2-Small** sur votre dataset REAL/ALT pour créer un détecteur d'images IA spécialisé sur les générateurs dominants en 2025-2026 (FLUX, Midjourney, GPT Image 1, SDXL, Ideogram).

---

## Sommaire

1. [Pourquoi fine-tuner SwinV2 ?](#pourquoi-fine-tuner-swinv2-)
2. [Ce qu'est un epoch](#ce-quest-un-epoch)
3. [Prérequis](#prérequis)
4. [Structure du dataset attendue](#structure-du-dataset-attendue)
5. [Lancement](#lancement)
6. [Options en ligne de commande](#options-en-ligne-de-commande)
7. [Ce que produit le script](#ce-que-produit-le-script)
8. [Interpréter les résultats](#interpréter-les-résultats)
9. [Intégrer le modèle dans le pipeline](#intégrer-le-modèle-dans-le-pipeline)
10. [Évaluer un modèle existant](#évaluer-un-modèle-existant)
11. [FAQ / Dépannage](#faq--dépannage)

---

## Pourquoi fine-tuner SwinV2 ?

Le backbone `microsoft/swinv2-small-patch4-window16-256` est pré-entraîné sur ImageNet-1k — il sait reconnaître des chats, des voitures, des chaises. Ce qu'il ne sait pas encore faire, c'est distinguer une photo réelle d'une image générée par FLUX ou Midjourney v7.

Le fine-tuning lui apprend cette nouvelle tâche en réutilisant les représentations visuelles qu'il a déjà apprises (textures, bords, structures) et en ajoutant une nouvelle tête de classification binaire (real / fake) à la place des 1000 classes ImageNet.

La méthodologie appliquée est celle du papier **OpenFake** (McGill/Mila, arxiv 2509.09495), entraîné originellement sur FLUX 1.1-pro, Midjourney v6, DALL-E 3, Grok-2 et Ideogram 3. En le fine-tunant sur votre propre dataset, vous adaptez le modèle à votre contexte spécifique (médias RS, JPEG compressé, contenu hétérogène).

---

## Ce qu'est un epoch

Un **epoch** est une passe complète sur l'ensemble du dataset d'entraînement.

Concrètement avec votre configuration (69 630 images, batch_size=16) :

- Le modèle traite les images par lots de 16 (**batches**)
- Un batch = 16 images vues → erreur calculée → poids ajustés
- 69 630 / 16 = **4 352 batches** pour compléter un epoch
- Quand les 4 352 batches sont traités → 1 epoch terminé, le modèle recommence dans un ordre aléatoire différent

Avec 10 epochs, chaque image est vue **10 fois** au total, mais jamais dans le même ordre et avec des augmentations différentes à chaque passage (dégradations JPEG aléatoires).

La progression typique sur 10 epochs :

| Epochs | Ce qui se passe |
|---|---|
| 1–2 | Apprentissage des patterns basiques (textures, artefacts évidents) |
| 3–5 | Affinement rapide — le F1 monte significativement |
| 6–8 | Convergence — gains plus lents mais réels |
| 9–10 | Stabilisation — le modèle peaufine les cas difficiles |

Le script sauvegarde automatiquement le **meilleur modèle** selon le F1 sur le set de validation. Même si la dernière epoch est légèrement moins bonne que la précédente, vous récupérez toujours le pic de performance.

---

## Prérequis

### Dépendances Python

```bash
pip install torch torchvision transformers accelerate timm tqdm
```

### GPU recommandé

| VRAM disponible | batch_size recommandé |
|---|---|
| < 6 Go | 4 |
| 6–8 Go | 8–16 |
| 8–12 Go | 32 |
| > 12 Go | 64 |

Le script active automatiquement le **gradient checkpointing** et la **mixed precision bf16** sur CUDA pour minimiser l'empreinte VRAM. Si votre GPU supporte sm_120 (RTX 50xx), utiliser l'environnement `forensics_nightly` (PyTorch cu128).

### CPU (sans GPU)

Fonctionne mais lent — prévoir 2-3 heures par epoch avec 70k images. Réduire `--num-workers` à 2 ou 4 selon votre machine.

---

## Structure du dataset attendue

```
calib_dataset/
    REAL/    ← images authentiques (photos réelles)
    ALT/     ← images générées / altérées par IA
```

Le script effectue automatiquement le split **70% train / 15% val / 15% test** depuis ces deux dossiers. Pas besoin de préparer les splits manuellement.

Pour préparer ce dossier depuis les datasets publics, utiliser `prepare_calib_dataset.py` :

```bash
python prepare_calib_dataset.py \
    --ff ./donnees/ff \
    --celebdf ./donnees/celebdf \
    --artifact ./donnees/artifact \
    --real-vs-fake ./donnees/real_vs_fake \
    --output ./calib_dataset \
    --max-real 2000 --max-alt 2000 --balance --link --verbose
```

Pour couvrir les générateurs 2025 (FLUX, MJ v7, GPT Image 1), ajouter le dataset OpenFake :

```bash
python prepare_calib_dataset.py \
    ... \
    --custom-alt ./donnees/openfake/synthetic \
    --output ./calib_dataset
```

---

## Lancement

### Standard (GPU, recommandé)

```bash
conda activate forensics_nightly   # ou forensics si GPU compatible

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python fine_tune_swinv2.py \
    --data-dir ./calib_dataset \
    --output-dir ./swinv2_openfake \
    --epochs 10 \
    --batch-size 16 \
    --num-workers 8 \
    --verbose
```

### CPU uniquement (VM sans GPU)

```bash
conda activate forensics

python fine_tune_swinv2.py \
    --data-dir ./calib_dataset \
    --output-dir ./swinv2_openfake \
    --epochs 10 \
    --batch-size 16 \
    --num-workers 4 \
    --verbose
```

### RTX 50xx (Blackwell, sm_120)

```bash
conda activate forensics_nightly

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python fine_tune_swinv2.py \
    --data-dir ./calib_dataset \
    --output-dir ./swinv2_openfake \
    --epochs 10 \
    --batch-size 16 \
    --num-workers 8 \
    --verbose
```

---

## Options en ligne de commande

| Option | Défaut | Description |
|---|---|---|
| `--data-dir` | *(obligatoire)* | Dossier contenant `REAL/` et `ALT/` |
| `--output-dir` | `./swinv2_openfake` | Dossier de sauvegarde du modèle fine-tuné |
| `--epochs` | `10` | Nombre de passes sur le dataset |
| `--batch-size` | `32` | Images traitées simultanément — réduire si OOM |
| `--lr` | `5e-5` | Learning rate (taux d'apprentissage) |
| `--num-workers` | `4` | Threads de chargement des images |
| `--eval-only` | *(flag)* | Évaluer un modèle existant sans entraîner |
| `--model-dir` | *(vide)* | Dossier du modèle à évaluer (avec `--eval-only`) |
| `--verbose` | *(flag)* | Affiche la progression epoch par epoch |

---

## Ce que produit le script

À la fin de l'entraînement, le dossier `--output-dir` contient :

```
swinv2_openfake/
    config.json              ← configuration du modèle (architecture, labels)
    model.safetensors        ← poids fine-tunés
    preprocessor_config.json ← paramètres de prétraitement des images
    training_info.json       ← métriques d'entraînement et de test
```

Le fichier `training_info.json` contient toutes les métriques, les hyperparamètres utilisés, et la taille des splits — utile pour documenter et comparer plusieurs runs.

---

## Interpréter les résultats

À la fin de chaque epoch, le script affiche :

```
Epoch 05  loss=0.3241  F1=0.7834  MCC=0.5712  Prec=0.8201  Rec=0.7510
```

Et à la fin, l'évaluation sur le test set :

```
=== Évaluation finale (test set) ===
  accuracy     : 0.7823
  f1           : 0.7756
  precision    : 0.8134
  recall       : 0.7415
  mcc          : 0.5641
  tp           : 4821    ← vrais fakes détectés
  tn           : 6847    ← vrais réels correctement classés
  fp           : 1102    ← réels classés comme fakes (faux positifs)
  fn           : 1153    ← fakes ratés (faux négatifs)
```

### Lecture des métriques

**F1** — moyenne harmonique de la précision et du rappel. Valeur cible : > 0.75. C'est la métrique principale pour juger la qualité globale.

**Precision** — parmi tout ce que le modèle dit être fake, quelle fraction l'est vraiment. **C'est la métrique la plus importante** pour notre usage : une précision élevée = peu de faux positifs = on n'accuse pas à tort du contenu réel. Valeur cible : > 0.80.

**Recall** — parmi tous les vrais fakes, quelle fraction est détectée. Peut être plus faible que la précision — mieux vaut rater quelques fakes que d'accuser des contenus réels. Valeur cible : > 0.65.

**MCC** (Matthews Correlation Coefficient) — indicateur global robuste aux datasets déséquilibrés. Valeur entre -1 et 1 : > 0.5 = bon, > 0.7 = excellent.

**fp (faux positifs)** — le chiffre le plus surveillé dans notre contexte. Un faux positif = un contenu réel classé comme fake = un faux signal d'alerte dans le pipeline.

### Valeurs attendues pour un premier entraînement

| Métrique | Acceptable | Bon | Excellent |
|---|---|---|---|
| F1 | > 0.65 | > 0.75 | > 0.85 |
| Precision | > 0.70 | > 0.80 | > 0.90 |
| MCC | > 0.35 | > 0.50 | > 0.70 |

Ces valeurs sont difficiles à atteindre sur un dataset hétérogène (ArtiFact + FF++ + Celeb-DF + real-vs-fake) car les générateurs sont très variés. L'enrichissement avec OpenFake améliore significativement les résultats sur FLUX et MJ.

---

## Intégrer le modèle dans le pipeline

Après l'entraînement, mettre à jour `ai_forensics.cfg` pour pointer vers le modèle local :

```ini
[models]
default_models =
    Organika/sdxl-detector,
    ./swinv2_openfake,
    synthbuster/synthbuster
```

Puis recalibrer les poids et biais sur votre dataset :

```bash
CUDA_VISIBLE_DEVICES="" python detect_ai_pipeline-v4.0.py \
    ./calib_dataset --calibrate --workers 2 --verbose \
    --json-output calib_report.json
```

La calibration FP-first attribuera automatiquement le bon poids au modèle SwinV2 selon sa performance réelle sur votre dataset — en particulier en pénalisant les faux positifs sur le contenu réel.

---

## Évaluer un modèle existant

Pour réévaluer un modèle déjà entraîné sans relancer l'entraînement :

```bash
python fine_tune_swinv2.py --eval-only \
    --model-dir ./swinv2_openfake \
    --data-dir ./calib_dataset
```

Utile pour comparer plusieurs versions du modèle ou vérifier les performances après avoir enrichi le dataset.

---

## FAQ / Dépannage

**`CUDA out of memory`**
Réduire `--batch-size` de moitié. L'ordre à essayer : 32 → 16 → 8 → 4. Les performances finales ne sont pas significativement affectées par un batch size plus petit.

**L'entraînement est très lent**
Sur CPU avec 70k images, compter 2-3h par epoch. Soit utiliser le GPU (voir section RTX 50xx), soit réduire le dataset avec `--max-real 1000 --max-alt 1000` dans `prepare_calib_dataset.py`.

**`MISMATCH classifier.weight / classifier.bias`**
C'est normal et attendu. Le backbone ImageNet a 1000 classes, on remplace sa tête de classification par une à 2 classes (real/fake). `ignore_mismatched_sizes=True` gère ça proprement.

**F1 stagne ou baisse après quelques epochs**
Le modèle converge. Le meilleur checkpoint est sauvegardé automatiquement — l'évaluation finale utilise toujours ce meilleur modèle, pas le dernier.

**Precision faible (beaucoup de faux positifs)**
Enrichir `ALT/` avec des exemples plus représentatifs des générateurs RS actuels (OpenFake, FLUX, MJ). Ou augmenter `--epochs` à 15-20. Recalibrer ensuite avec `--calibrate` pour que le pipeline ajuste automatiquement le biais de SwinV2.

**Le modèle score 0.5 partout dans le pipeline**
Le backbone de base (non fine-tuné) a été chargé au lieu du modèle fine-tuné. Vérifier que `--output-dir` dans `ai_forensics.cfg` pointe vers le bon dossier et que `model.safetensors` existe dedans.
