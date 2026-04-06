# Spécification d'interface d'entrée — AI-FORENSICS

> Format JSON & organisation des fichiers

| | |
|---|---|
| **Version** | 1.1 |
| **Date** | Avril 2026 |
| **Projet** | AI-FORENSICS — Détection de médias synthétiques & campagnes d'influence |
| **Objet** | Contrat d'interface entre tout système producteur de données (scraper, outil tiers) et le pipeline d'analyse AI-FORENSICS |
| **Statut** | Applicable |

---

## Table des matières

1. [Objet et portée](#1-objet-et-portée)
2. [Conventions](#2-conventions)
3. [Organisation des fichiers sur disque](#3-organisation-des-fichiers-sur-disque)
4. [Format JSON des posts](#4-format-json-des-posts)
5. [Exemple JSON complet](#5-exemple-json-complet)
6. [Utilitaire de nettoyage recommandé](#6-utilitaire-de-nettoyage-recommandé)
7. [Récapitulatif des exigences](#7-récapitulatif-des-exigences)
8. [Spécificités par plateforme](#8-spécificités-par-plateforme)
9. [Notes et points ouverts](#9-notes-et-points-ouverts)

---

## 1. Objet et portée

Ce document définit le format de données que tout système d'alimentation (ci-après « le producteur ») doit respecter pour être compatible avec le pipeline d'analyse AI-FORENSICS. Il s'adresse à tout développeur souhaitant interfacer un outil de collecte — scraper, outil de veille, export de plateforme — avec le système sans connaître son fonctionnement interne.

AI-FORENSICS analyse des publications issues des réseaux sociaux (Instagram, Twitter/X, TikTok, Telegram) afin d'y détecter des médias générés par intelligence artificielle et d'identifier des campagnes d'influence coordonnées. Il ingère des fichiers JSON et des fichiers médias organisés selon la structure décrite ici.

> **PRINCIPE**
> Le producteur et le pipeline ne communiquent que par fichiers (JSON + médias).
> Toute ambiguïté dans le format produit entraîne un échec silencieux ou une perte de données côté analyse.
> Ce document est le contrat d'interface entre les deux parties. En cas de doute, il fait référence.

---

## 2. Conventions

### 2.1 Niveaux d'exigence

| Niveau | Signification |
|---|---|
| 🔴 **BLOQUANT** | L'import automatique échoue ou produit des données incorrectes si ce point n'est pas respecté. |
| 🟡 **IMPORTANT** | Fortement recommandé. Le non-respect dégrade la qualité de l'analyse ou empêche certaines fonctionnalités. |
| 🟢 **RECOMMANDÉ** | Bonne pratique. L'absence de ces éléments n'est pas bloquante mais peut limiter les possibilités d'analyse. |

### 2.2 Règles générales

- Encodage des fichiers JSON : **UTF-8 sans BOM**.
- Format des dates : **ISO 8601 avec timezone explicite**, ex. `"2026-03-18T10:00:00+00:00"`.
- Valeurs inconnues ou indisponibles : **`null`** (jamais une chaîne vide `""` ni une date epoch Unix `"1970-01-01T00:00:00+00:00"`).
- Les booléens sont `true` / `false` (minuscules, sans guillemets).
- Les listes vides sont `[]` (jamais `null` ni absent).

> ⚠️ **RÈGLE CRITIQUE**
> Une chaîne vide et `null` ne sont pas équivalentes. Les index et filtres du pipeline se comportent différemment selon le type. Toute valeur non disponible doit être strictement `null`.

---

## 3. Organisation des fichiers sur disque

### 3.1 Arborescence générale

La racine du stockage (ci-après `/storage/`) peut être un chemin local, un volume partagé ou tout système de fichiers accessible en lecture/écriture par les deux parties. L'arborescence interne est normalisée :

```
/storage/
    <platform>/
        <post_id>/
            <post_id>.json           ← fichier de métadonnées du post
            <post_id>_cover.<ext>    ← miniature (optionnel)
            <post_id>_media_1.<ext>  ← premier média
            <post_id>_media_2.<ext>  ← second média (carousel, etc.)
```

Exemple complet avec les quatre plateformes supportées :

```
/storage/
    instagram/
        DOGI4cxiRAj/
            DOGI4cxiRAj.json
            DOGI4cxiRAj_cover.jpg
            DOGI4cxiRAj_media_1.jpg
    twitter/
        1898765432100000001/
            1898765432100000001.json
            1898765432100000001_media_1.jpg
    tiktok/
        7412345678901234567/
            7412345678901234567.json
            7412345678901234567_cover.jpg
            7412345678901234567_media_1.mp4
    telegram/
        canal_crypto_fr/
            189234/
                189234.json
                189234_media_1.mp4
```

> ℹ️ **TELEGRAM**
> Telegram est structuré en canaux pouvant contenir des milliers de messages. Un niveau supplémentaire est donc ajouté : `/storage/telegram/<channel_id>/<message_id>/`

### 3.2 Règles de nommage des fichiers

| Type | Nom attendu | Exemple |
|---|---|---|
| Métadonnées JSON du post | `<post_id>.json` | `DOGI4cxiRAj.json` |
| Miniature / couverture | `<post_id>_cover.<ext>` | `DOGI4cxiRAj_cover.jpg` |
| Image ou vidéo (premier) | `<post_id>_media_1.<ext>` | `DOGI4cxiRAj_media_1.jpg` |
| Médias supplémentaires | `<post_id>_media_N.<ext>` | `DOGI4cxiRAj_media_2.jpg` |

### 3.3 Gestion des médias non téléchargés

Lorsqu'un média n'a pas pu être récupéré (erreur réseau, contenu supprimé, restriction géographique) :

- Ne pas créer de fichier vide ou de placeholder.
- Renseigner `media_telecharge: false` dans le bloc `scrappeurInfo`.
- Conserver l'URL originale dans les champs `cover` ou `video_url` du JSON (pour permettre une nouvelle tentative).
- Passer `media_path` à `null` si le dossier ne contient aucun fichier média.

### 3.4 Dépose des fichiers JSON — dossier inbox

Pour permettre la découverte automatique des nouveaux fichiers par le pipeline, le producteur dépose les JSON dans un sous-dossier `inbox/` à la racine du stockage :

```
/storage/
    inbox/       ← JSON déposés ici par le producteur
    instagram/   ← médias organisés par plateforme
    twitter/
    tiktok/
    telegram/
```

> ⚠️ Les médias ne doivent **pas** être déposés dans `inbox/`. Seuls les fichiers JSON y sont attendus. Les médias restent dans leur arborescence `<platform>/<post_id>/`.

---

## 4. Format JSON des posts

### 4.1 Structure globale

Chaque post correspond à un fichier JSON unique, structuré dans l'ordre suivant :

```json
{
    "scrappeurInfo": { ... },      // bloc de métadonnées de collecte
    "id": "<post_id>",             // identifiant natif de la plateforme
    "post_url": "<url>",           // URL canonique du post
    "media_type": "<type>",        // type de contenu média principal
    "platform": "<platform>",      // plateforme source
    "desc": "<texte>",             // description / légende
    "createTime": "<ISO8601>",     // date de publication
    "video_url": null,             // URL de la vidéo ou null
    "cover": null,                 // URL de la miniature ou null
    "author": { ... },             // informations sur le compte
    "music": { ... },              // piste audio (TikTok principalement)
    "hashtags": [],                // liste de hashtags
    "mentions": [],                // liste de mentions
    "stats": { ... },              // métriques d'engagement
    "comments": []                 // liste de commentaires
}
```

### 4.2 Bloc scrappeurInfo

> 🔴 **BLOQUANT** — Ce bloc est obligatoire. Son absence entraîne le rejet du fichier.

| Champ | Type | Description |
|---|---|---|
| `version` | string | Version du producteur — ex. `"1.2.0"` |
| `platform` | string | Plateforme : `instagram` \| `twitter` \| `tiktok` \| `telegram` |
| `content_type` | string | Type de contenu : `post` \| `story` \| `reel` \| `message` |
| `gdh_scrap` | ISO 8601 | Date et heure de la collecte avec timezone — ex. `"2026-03-18T10:00:00+00:00"` |
| `media_telecharge` | boolean | `true` si au moins un fichier média a été téléchargé localement |
| `media_path` | string \| null | Chemin relatif du dossier contenant les médias — ex. `"instagram/DOGI4cxiRAj/"`. `null` si aucun média téléchargé. |
| `scrapper_id` | string | Identifiant du worker producteur (utile en cas de parallélisation) |

```json
"scrappeurInfo": {
    "version": "1.2.0",
    "platform": "instagram",
    "content_type": "post",
    "gdh_scrap": "2026-03-18T10:00:00+00:00",
    "media_telecharge": true,
    "media_path": "instagram/DOGI4cxiRAj/",
    "scrapper_id": "worker_01"
}
```

### 4.3 Champs racine du post

| Champ | Type | Niveau | Description |
|---|---|---|---|
| `id` | string | 🔴 BLOQUANT | Identifiant natif de la plateforme. Doit être unique par plateforme. |
| `post_url` | string | 🟡 IMPORTANT | URL canonique et pérenne du post. |
| `media_type` | string | 🟡 IMPORTANT | Type du média principal : `image` \| `video` \| `carousel` \| `text` \| `audio`. `null` si aucun média. |
| `platform` | string | 🔴 BLOQUANT | Plateforme source. Doit correspondre à `scrappeurInfo.platform`. |
| `desc` | string \| null | 🟡 IMPORTANT | Texte de la publication (légende, tweet, message). `null` si absent. |
| `createTime` | ISO 8601 \| null | 🟡 IMPORTANT | Date et heure de publication avec timezone. `null` si inconnue. Ne jamais mettre la date epoch Unix. |
| `video_url` | string \| null | 🟢 RECOMMANDÉ | URL directe de la vidéo. `null` si absent ou non applicable. |
| `cover` | string \| null | 🟢 RECOMMANDÉ | URL de la miniature ou image de couverture. `null` si absent. |
| `hashtags` | array | 🟡 IMPORTANT | Liste de chaînes sans le caractère `#`. `[]` si aucun hashtag. |
| `mentions` | array | 🟢 RECOMMANDÉ | Liste d'identifiants mentionnés. `[]` si aucun. |

### 4.4 Bloc author

Les informations relatives au compte auteur sont regroupées dans un sous-objet `author`. Ne pas les exposer à plat au niveau racine.

| Champ | Type | Description |
|---|---|---|
| `name` | string \| null | Nom d'affichage du compte |
| `id` | string \| null | Identifiant numérique natif de la plateforme |
| `uniqueId` | string \| null | Nom d'utilisateur (@handle) |
| `avatar` | string \| null | URL de la photo de profil |
| `url` | string \| null | URL du profil public |

```json
"author": {
    "name": "Cryptocom",
    "id": "5474917067",
    "uniqueId": "cryptocom",
    "avatar": null,
    "url": null
}
```

### 4.5 Bloc music

Principalement utilisé pour TikTok. Toujours présent en tant qu'objet (jamais absent), avec `null` si les informations ne sont pas disponibles.

```json
"music": {
    "title": "Original Sound",
    "author": null
}
```

### 4.6 Bloc stats

Métriques d'engagement au moment de la collecte. Les champs non applicables à la plateforme sont mis à `0`, jamais à `null`.

| Champ | Type | Description |
|---|---|---|
| `likes` | integer | Nombre de j'aime / réactions |
| `comments` | integer | Nombre de commentaires |
| `shares` | integer | Nombre de partages / retweets |
| `plays` | integer | Nombre de lectures (vidéo). `0` si non applicable. |
| `favorites` | integer | Nombre d'ajouts aux favoris. `0` si non applicable. |

### 4.7 Bloc comments

Liste des commentaires collectés pour ce post. Peut être `[]`. Chaque commentaire respecte la structure suivante :

| Champ | Type | Niveau | Description |
|---|---|---|---|
| `comment_id` | string | 🟡 IMPORTANT | Identifiant unique du commentaire. Si non disponible nativement, générer un hash `MD5(post_id + author_id + timestamp)`. |
| `author` | string \| null | 🔴 BLOQUANT | Nom d'utilisateur de l'auteur du commentaire |
| `author_id` | string \| null | 🟡 IMPORTANT | Identifiant natif de l'auteur |
| `text` | string \| null | 🔴 BLOQUANT | Contenu textuel du commentaire |
| `timestamp` | ISO 8601 \| null | 🟡 IMPORTANT | Date et heure du commentaire avec timezone |
| `likes` | integer | 🟢 RECOMMANDÉ | Nombre de j'aime sur le commentaire. `0` si inconnu. |
| `reply_to_id` | string \| null | 🟡 IMPORTANT | `comment_id` du commentaire parent si réponse. `null` si premier niveau. |
| `hashtags` | array | 🟢 RECOMMANDÉ | Hashtags dans le commentaire. `[]` si aucun. |
| `mentions` | array | 🟢 RECOMMANDÉ | Mentions dans le commentaire. `[]` si aucune. |

---

## 5. Exemple JSON complet

```json
{
    "scrappeurInfo": {
        "version": "1.2.0",
        "platform": "instagram",
        "content_type": "post",
        "gdh_scrap": "2026-03-18T10:00:00+00:00",
        "media_telecharge": true,
        "media_path": "instagram/DOGI4cxiRAj/",
        "scrapper_id": "worker_01"
    },
    "id": "DOGI4cxiRAj",
    "post_url": "https://www.instagram.com/p/DOGI4cxiRAj/",
    "media_type": "image",
    "platform": "instagram",
    "desc": "Ready to take your wealth to the next level?",
    "createTime": "2025-09-02T10:07:41+00:00",
    "video_url": null,
    "cover": "https://scontent-cdg4-1.cdninstagram.com/...",
    "author": {
        "name": "Cryptocom",
        "id": "5474917067",
        "uniqueId": "cryptocom",
        "avatar": null,
        "url": null
    },
    "music": { "title": null, "author": null },
    "hashtags": ["CryptoCom", "LevelUp", "promo"],
    "mentions": [],
    "stats": {
        "likes": 2145,
        "comments": 780,
        "shares": 0,
        "plays": 0,
        "favorites": 0
    },
    "comments": [
        {
            "comment_id": "17858893269000001",
            "author": "mreuro_official",
            "author_id": "59227622622",
            "text": "Super content !",
            "timestamp": "2026-03-16T00:20:29+00:00",
            "likes": 0,
            "reply_to_id": null,
            "hashtags": [],
            "mentions": []
        }
    ]
}
```

---

## 6. Utilitaire de nettoyage recommandé

À intégrer en sortie du producteur, avant l'écriture du JSON sur disque. Remplace automatiquement les valeurs invalides par `null` :

```python
def clean_nulls(obj):
    """Remplace chaînes vides et epoch Unix par None."""
    EPOCH = "1970-01-01T00:00:00+00:00"
    if isinstance(obj, dict):
        return {k: clean_nulls(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nulls(i) for i in obj]
    if obj in ("", EPOCH):
        return None
    return obj

# Appel en fin de collecte :
output = clean_nulls(scraped_data)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
```

---

## 7. Récapitulatif des exigences

| # | Exigence | Niveau | Impact si non respecté |
|---|---|---|---|
| 1 | Bloc `scrappeurInfo` présent dans chaque JSON | 🔴 BLOQUANT | Fichier rejeté à l'import |
| 2 | Valeurs inconnues à `null` (jamais `""` ni epoch Unix) | 🔴 BLOQUANT | Filtres et index incorrects |
| 3 | Champ `platform` cohérent entre racine et `scrappeurInfo` | 🔴 BLOQUANT | Mauvais routage de l'analyse |
| 4 | Champ `id` présent et unique par plateforme | 🔴 BLOQUANT | Doublons en base |
| 5 | `post_url` renseigné (URL canonique) | 🟡 IMPORTANT | Liens de traçabilité manquants |
| 6 | `media_type` renseigné au niveau racine | 🟡 IMPORTANT | Routage vers les détecteurs impossible |
| 7 | Champs `author_*` dans sous-objet `author{}` | 🟡 IMPORTANT | Incohérence structurelle |
| 8 | Champs `music_*` dans sous-objet `music{}` | 🟡 IMPORTANT | Incohérence structurelle |
| 9 | `comment_id` présent sur chaque commentaire | 🟡 IMPORTANT | Déduplication des commentaires impossible |
| 10 | `reply_to_id` présent sur chaque commentaire | 🟡 IMPORTANT | Reconstruction des fils de réponse impossible |
| 11 | Structure `/storage/<platform>/<post_id>/` | 🟡 IMPORTANT | Médias introuvables automatiquement |
| 12 | Nommage `<post_id>_media_N.<ext>` | 🟡 IMPORTANT | Association JSON ↔ média échoue |
| 13 | Dépôt des JSON dans `inbox/` | 🟢 RECOMMANDÉ | Découverte automatique désactivée |
| 14 | Intégration de `clean_nulls()` avant écriture | 🟢 RECOMMANDÉ | Risque de valeurs invalides en base |

---

## 8. Spécificités par plateforme

### Instagram

- Le `comment_id` est disponible via l'API Graph (ex. `"17858893269000001"`).
- Les stories ont une durée de vie de 24 h : s'assurer que le traitement est suffisamment rapide.
- Les posts de type carousel contiennent plusieurs médias : les numéroter `_media_1`, `_media_2`, etc.

### Twitter / X

- L'identifiant du commentaire (réponse) est l'ID du tweet de réponse lui-même.
- Le champ `id` contient l'ID du tweet — représenter en string pour éviter les pertes de précision JSON (entier long).

### TikTok

- Le champ `music` est généralement renseigné — ne pas le mettre à `null` par défaut.
- Le `comment_id` est disponible dans la réponse de l'API TikTok.
- Les vidéos courtes (< 5 s) peuvent ne pas avoir de `cover` distincte.

### Telegram

- Un niveau de dossier supplémentaire est requis : `/storage/telegram/<channel_id>/<message_id>/`.
- Le `channel_id` doit être exposé dans le JSON du message.
- Les messages Telegram peuvent ne contenir aucun média (texte seul) : dans ce cas `media_telecharge` vaut `false` et `media_path` vaut `null`.

---

## 9. Notes et points ouverts

| Point | Description |
|---|---|
| Emplacement de `/storage/` | Chemin local, volume NFS partagé, ou objet compatible S3/MinIO. À définir selon l'infrastructure de déploiement. |
| `comment_id` non disponible | En l'absence d'identifiant natif, utiliser un hash `MD5(post_id + author_id + timestamp)` comme identifiant de repli. |
| Stories Instagram | Durée de vie 24 h. Décider si elles sont incluses et s'assurer que le traitement est suffisamment rapide. |
| `channel_id` Telegram | Vérifier que le `channel_id` est bien exposé dans tous les JSON Telegram produits. |
| Retry des médias manquants | Le pipeline peut relancer le téléchargement si `media_telecharge` est `false` et que l'URL originale est présente dans `cover` ou `video_url`. |
