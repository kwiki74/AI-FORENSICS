"""schema.py  v4
===============

Schéma MongoDB — Détection deepfake / campagnes d'influence
Plateformes : Twitter/X, TikTok, Instagram, Telegram
Contenu     : posts texte, images, vidéos, profils, commentaires/réponses

Collections :
  accounts    — profils de comptes (toutes plateformes)
  posts       — publications (texte, image, vidéo)
  comments    — commentaires et réponses à un post
  media       — fichiers médias téléchargés localement
  narratives  — clusters de messages partageant un même narratif
  campaigns   — campagnes d'influence détectées
  jobs        — file de traitement interne (alternative légère à Redis)

Utilisation :
    from schema import get_db, new_post, new_account, new_media, new_comment
    from schema import patch_post_deepfake, patch_post_nlp, patch_post_sync
    from schema import patch_media_sync                                  # [v4]
    from schema import create_indexes

Configuration de la connexion :
    Les credentials sont lus depuis un fichier .env situé dans le même dossier
    que ce script, ou depuis les variables d'environnement du système.

    Fichier .env (à créer, NE PAS committer dans git) :
        MONGO_HOST=localhost
        MONGO_PORT=27017
        MONGO_USER=influence_app
        MONGO_PASSWORD=AppPassword456!
        MONGO_DB=influence_detection
        MONGO_AUTH_DB=influence_detection

    Connexion admin (pour create_indexes) :
        MONGO_USER=admin
        MONGO_AUTH_DB=admin

Dépendances :
    pip install pymongo python-dotenv
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote_plus

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING, MongoClient, IndexModel

# Chargement optionnel du .env (ne plante pas si python-dotenv est absent)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass  # python-dotenv non installé — on utilise les variables d'environnement système



# ---------------------------------------------------------------------------
# Connexion MongoDB
# ---------------------------------------------------------------------------

def _build_uri(user: str, password: str, host: str, port: int, auth_db: str) -> str:
    """Construit l'URI de connexion en encodant les caractères spéciaux."""
    return (
        f"mongodb://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/?authSource={auth_db}"
        f"&replicaSet=rs0"
        f"&directConnection=true"
    )


def get_db(
    db_name: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    auth_db: Optional[str] = None,
):
    """
    Retourne un objet Database MongoDB prêt à l'emploi.

    Les paramètres non fournis sont lus depuis les variables d'environnement
    (ou le fichier .env) dans l'ordre suivant :
      MONGO_HOST     (défaut: localhost)
      MONGO_PORT     (défaut: 27017)
      MONGO_USER
      MONGO_PASSWORD
      MONGO_DB       (défaut: influence_detection)
      MONGO_AUTH_DB  (défaut: même valeur que MONGO_DB)

    Exemple d'utilisation dans tous les autres modules du projet :
        from schema import get_db
        db = get_db()
        db.posts.find_one({"platform": "instagram"})
    """
    _host     = host     or os.getenv("MONGO_HOST",     "localhost")
    _port     = port     or int(os.getenv("MONGO_PORT", "27017"))
    _user     = user     or os.getenv("MONGO_USER",     "")
    _password = password or os.getenv("MONGO_PASSWORD", "")
    _db_name  = db_name  or os.getenv("MONGO_DB",       "influence_detection")
    _auth_db  = auth_db  or os.getenv("MONGO_AUTH_DB",  _db_name)

    if _user and _password:
        uri = _build_uri(_user, _password, _host, _port, _auth_db)
    else:
        # Pas d'auth (dev local sans sécurité activée)
        uri = f"mongodb://{_host}:{_port}/?replicaSet=rs0&directConnection=true"

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)

    # Vérification rapide de la connexion
    try:
        client.admin.command("ping")
    except Exception as exc:
        raise ConnectionError(
            f"Impossible de se connecter à MongoDB ({_host}:{_port}) : {exc}\n"
            f"Vérifiez que mongod est démarré et que les credentials sont corrects."
        ) from exc

    return client[_db_name]


def get_admin_client(
    host: Optional[str] = None,
    port: Optional[int] = None,
    admin_user: Optional[str] = None,
    admin_password: Optional[str] = None,
) -> MongoClient:
    """
    Retourne un MongoClient connecté en tant qu'admin.
    Utilisé uniquement pour create_indexes() et les tâches d'administration.

    Variables d'environnement lues :
      MONGO_ADMIN_USER      (défaut: valeur de MONGO_USER)
      MONGO_ADMIN_PASSWORD  (défaut: valeur de MONGO_PASSWORD)
    """
    _host     = host           or os.getenv("MONGO_HOST",           "localhost")
    _port     = port           or int(os.getenv("MONGO_PORT",       "27017"))
    _user     = admin_user     or os.getenv("MONGO_ADMIN_USER",     os.getenv("MONGO_USER", ""))
    _password = admin_password or os.getenv("MONGO_ADMIN_PASSWORD", os.getenv("MONGO_PASSWORD", ""))

    if _user and _password:
        uri = _build_uri(_user, _password, _host, _port, "admin")
    else:
        uri = f"mongodb://{_host}:{_port}/?replicaSet=rs0&directConnection=true"

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except Exception as exc:
        raise ConnectionError(f"Connexion admin échouée : {exc}") from exc

    return client



def _now() -> datetime:
    return datetime.now(timezone.utc)


def _id() -> ObjectId:
    return ObjectId()


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

PLATFORMS  = {"twitter", "tiktok", "instagram", "telegram"}
MEDIA_TYPES = {"image", "video", "gif", "audio", "sticker"}
JOB_TYPES  = {"deepfake_analysis", "nlp_analysis", "etl_sync", "media_download"}


# ===========================================================================
# Collection : accounts
# ===========================================================================

def new_account(
    platform: str,
    platform_id: str,
    username: str,
    display_name: str = "",
    raw: Optional[dict] = None,
) -> dict:
    """
    Document account unifié toutes plateformes.
    Clé naturelle unique : (platform, platform_id).

    Champs platform_specific :
      twitter   → verified_type (blue/gold/grey), is_blue_verified
      tiktok    → region, is_creator
      instagram → is_business, business_category, external_url
      telegram  → is_channel, is_group, member_count, invite_link
                  (Telegram n'a pas de followers au sens RS classique)
    """
    assert platform in PLATFORMS, f"Plateforme inconnue : {platform}"
    now = _now()

    # Champs spécifiques à la plateforme
    platform_specific: dict = {
        "twitter":   {
            "verified_type":    None,   # blue | gold | grey | none
            "is_blue_verified": False,
            "protected":        False,
            "listed_count":     0,
        },
        "tiktok":    {
            "region":           None,
            "is_creator":       False,
            "is_commerce":      False,
        },
        "instagram": {
            "is_business":        False,
            "business_category":  None,
            "external_url":       None,
            "is_private":         False,
        },
        "telegram":  {
            "is_channel":     False,   # canal (broadcast, pas de followers visibles)
            "is_group":       False,   # groupe (membres peuvent poster)
            "is_bot":         False,
            "member_count":   None,    # visible seulement si public
            "invite_link":    None,
            "username":       None,    # @username (optionnel sur Telegram)
        },
    }[platform]

    return {
        "_id": _id(),

        # --- Identification ---
        "platform":     platform,
        "platform_id":  platform_id,   # user_id, channel_id, etc.
        "username":     username,       # @handle (None pour certains comptes Telegram)
        "display_name": display_name,
        "url":          None,

        # --- Profil commun ---
        "profile": {
            "bio":        None,
            "location":   None,
            "website":    None,
            "avatar_url": None,
            "banner_url": None,         # non applicable sur Telegram
            "verified":   False,        # vérification officielle plateforme
            "created_at": None,
            "language":   None,
        },

        # --- Stats snapshot ---
        # Pour Telegram : followers_count = member_count du canal/groupe
        "stats": {
            "followers_count":  0,
            "following_count":  0,      # toujours 0 pour Telegram
            "posts_count":      0,
            "likes_count":      0,      # vues/réactions sur Telegram
            "scraped_at":       now,
        },

        # --- Champs spécifiques à la plateforme ---
        "platform_specific": platform_specific,

        # --- Analyse NLP / comportementale ---
        "analysis": {
            "status":            "pending",  # pending|processing|done|error
            "bot_score":         None,       # [0–1] probabilité de compte automatisé
            "bot_signals": {
                "post_regularity":   None,   # écart-type des intervalles de post (secondes)
                "follower_ratio":    None,   # following/followers (N/A pour Telegram)
                "avg_engagement":    None,   # (likes+RT)/followers moyen
                "account_age_days":  None,
                "post_burst_score":  None,   # détection de bursts de publication coordonnés
            },
            "language_detected": None,
            "narratives":        [],         # IDs → narratives
            "analyzed_at":       None,
        },

        # --- Flags ---
        "flags": {
            "is_suspicious":    False,
            "is_confirmed_bot": False,
            "campaign_ids":     [],          # IDs → campaigns
            "notes":            "",
        },

        # --- Cycle de vie ---
        "scraped_at": now,
        "updated_at": now,
        "raw":        raw or {},
    }


def patch_account_stats(stats: dict) -> dict:
    return {"$set": {"stats": {**stats, "scraped_at": _now()}, "updated_at": _now()}}


def patch_account_analysis(
    bot_score: float,
    bot_signals: dict,
    language: str,
    narratives: list,
) -> dict:
    return {
        "$set": {
            "analysis.status":            "done",
            "analysis.bot_score":         bot_score,
            "analysis.bot_signals":       bot_signals,
            "analysis.language_detected": language,
            "analysis.narratives":        narratives,
            "analysis.analyzed_at":       _now(),
            "updated_at":                 _now(),
        }
    }


# ===========================================================================
# Collection : posts
# ===========================================================================

def new_post(
    platform: str,
    platform_id: str,
    account_id: ObjectId,
    account_platform_id: str,
    text_content: str = "",
    raw: Optional[dict] = None,
) -> dict:
    """
    Document post unifié toutes plateformes.
    Clé naturelle unique : (platform, platform_id).

    Champs platform_specific :
      twitter   → is_retweet, quoted_tweet_id, lang (fourni par l'API)
      tiktok    → music_id, music_title, effect_ids, stitch_count, duet_count
      instagram → shortcode, media_type (photo/carousel/reel), location
      telegram  → channel_id, message_id, forward_from, views, forward_count
                  (Telegram : pas de likes au sens strict, mais réactions)
    """
    assert platform in PLATFORMS, f"Plateforme inconnue : {platform}"
    now = _now()

    platform_specific: dict = {
        "twitter":   {
            "is_retweet":       False,
            "is_quote":         False,
            "quoted_tweet_id":  None,
            "lang":             None,   # langue détectée par Twitter
            "source_app":       None,   # "Twitter for iPhone", "TweetDeck", etc.
            "cover_url":        None,   # image de prévisualisation
        },
        "tiktok":    {
            "music_id":       None,
            "music_title":    None,
            "music_author":   None,   # auteur du son — signal de son viral coordonné [v4]
            "effect_ids":     [],
            "stitch_count":   0,
            "duet_count":     0,
            "is_ad":          False,
            "cover_url":      None,   # miniature de la vidéo [v4]
        },
        "instagram": {
            "shortcode":    None,
            "media_type":   None,   # photo | carousel | reel | story
            "location":     None,
            "tagged_users": [],
            "is_reel":      False,
            "cover_url":    None,   # miniature du post [v4]
        },
        "telegram":  {
            "channel_id":     None,
            "message_id":     None,
            "forward_from":   None,    # channel/user d'origine si forwardé
            "forward_count":  0,
            "views":          0,       # vues du message dans le canal
            "reactions":      {},      # {"👍": 120, "❤️": 45, ...}
            "edit_date":      None,    # date de dernière modification
            "is_forwarded":   False,
        },
    }[platform]

    return {
        "_id": _id(),

        # --- Identification ---
        "platform":            platform,
        "platform_id":         platform_id,
        "url":                 None,
        "account_id":          account_id,
        "account_platform_id": account_platform_id,

        # --- Contenu textuel ---
        "text": {
            "content":     text_content,
            "language":    None,
            "hashtags":    [],
            "mentions":    [],
            "urls":        [],
            "is_truncated": False,
        },

        # --- Médias attachés (références → collection media) ---
        "media": [],

        # --- Métriques d'engagement ---
        # Telegram : likes=total réactions, shares=forward_count, views=vues canal
        "engagement": {
            "likes":      0,
            "shares":     0,
            "comments":   0,
            "views":      0,
            "saves":      0,
            "scraped_at": now,
        },

        # --- Contexte de publication ---
        "context": {
            "published_at": None,
            "is_reply_to":  None,   # platform_id du post parent
            "is_repost_of": None,   # platform_id du post original
            "reply_count":  0,
            "thread_id":    None,
        },

        # --- Champs spécifiques à la plateforme ---
        "platform_specific": platform_specific,

        # --- Signaux scrapper [v4] ---
        # Pré-calculés par le scrapper, complémentaires aux analyses pipeline
        "scrapper_signals": {
            "influence_score":  None,   # score d'influence estimé [0–1]
            "is_bot_suspected": False,  # flag bot détecté par le scrapper
        },

        # --- Analyse deepfake (ton script) ---
        "deepfake": {
            "status":           "pending",  # pending|processing|done|skipped|error
            "has_media":        False,
            "final_score":      None,
            "prediction":       None,       # likely_real|suspicious|synthetic
            "model_divergence": None,
            "scores":           {},
            "raw_scores":       {},
            "artifact_score":   None,
            "frames_analyzed":  None,
            "pipeline_version": None,
            "processed_at":     None,
            "error":            None,
        },

        # --- Analyse NLP ---
        "nlp": {
            "status":           "pending",
            "sentiment": {
                "label":  None,    # positive|negative|neutral
                "score":  None,
                "model":  None,
            },
            "embedding":        None,   # bytes float16 (384d) — rempli par nlp_worker
            "embedding_model":  None,
            "topics":           [],
            "is_duplicate_of":  None,   # _id du post original si copie détectée
            "similarity_score": None,
            "narrative_id":     None,   # référence → narratives
            "processed_at":     None,
            "error":            None,
        },

        # --- Synchronisation ETL ---
        "sync": {
            "neo4j":         False,
            "elasticsearch": False,
            "synced_at":     None,
        },

        # --- Cycle de vie ---
        "scraped_at": now,
        "updated_at": now,
        "raw":        raw or {},
    }


def patch_post_deepfake(result: dict, pipeline_version: str = "3.4.2") -> dict:
    """
    Mise à jour deepfake à partir du dict retourné par analyze_image/video.
    Compatible avec le format de sortie de detect_ai_pipeline-v3.4.2.py.
    """
    scores     = {k[6:]: v for k, v in result.items() if k.startswith("score_")}
    raw_scores = {k[4:]: v for k, v in result.items() if k.startswith("raw_")}
    return {
        "$set": {
            "deepfake.status":           "done",
            "deepfake.final_score":      result.get("final_score"),
            "deepfake.prediction":       result.get("prediction"),
            "deepfake.model_divergence": result.get("model_divergence"),
            "deepfake.artifact_score":   result.get("artifact_score"),
            "deepfake.frames_analyzed":  result.get("frames_analyzed"),
            "deepfake.scores":           scores,
            "deepfake.raw_scores":       raw_scores,
            "deepfake.pipeline_version": pipeline_version,
            "deepfake.processed_at":     _now(),
            "deepfake.error":            None,
            "updated_at":                _now(),
        }
    }


def patch_post_deepfake_error(error: str) -> dict:
    return {
        "$set": {
            "deepfake.status":       "error",
            "deepfake.error":        error,
            "deepfake.processed_at": _now(),
            "updated_at":            _now(),
        }
    }


def patch_post_nlp(
    sentiment_label: str,
    sentiment_score: float,
    sentiment_model: str,
    embedding_model: str,
    topics: list[str],
    embedding: Optional[bytes] = None,
    narrative_id: Optional[ObjectId] = None,
    is_duplicate_of: Optional[ObjectId] = None,
    similarity_score: Optional[float] = None,
) -> dict:
    update = {
        "nlp.status":             "done",
        "nlp.sentiment.label":    sentiment_label,
        "nlp.sentiment.score":    sentiment_score,
        "nlp.sentiment.model":    sentiment_model,
        "nlp.embedding_model":    embedding_model,
        "nlp.topics":             topics,
        "nlp.narrative_id":       narrative_id,
        "nlp.is_duplicate_of":    is_duplicate_of,
        "nlp.similarity_score":   similarity_score,
        "nlp.processed_at":       _now(),
        "nlp.error":              None,
        "updated_at":             _now(),
    }
    if embedding is not None:
        update["nlp.embedding"] = embedding
    return {"$set": update}


def patch_post_sync(neo4j: bool = False, elasticsearch: bool = False) -> dict:
    update: dict[str, Any] = {"updated_at": _now(), "sync.synced_at": _now()}
    if neo4j:
        update["sync.neo4j"] = True
    if elasticsearch:
        update["sync.elasticsearch"] = True
    return {"$set": update}


def patch_post_media(media_ref: dict) -> dict:
    """Ajoute une référence média et marque has_media=True."""
    return {
        "$push": {"media": media_ref},
        "$set":  {"deepfake.has_media": True, "updated_at": _now()},
    }


# ===========================================================================
# Collection : comments
# ===========================================================================

def new_comment(
    platform: str,
    platform_id: str,
    post_id: ObjectId,
    post_platform_id: str,
    account_id: ObjectId,
    account_platform_id: str,
    text_content: str = "",
    parent_comment_id: Optional[ObjectId] = None,
    parent_comment_platform_id: Optional[str] = None,
    raw: Optional[dict] = None,
) -> dict:
    """
    Document commentaire / réponse.
    Clé naturelle unique : (platform, platform_id).

    Pourquoi une collection séparée ?
      - Un post peut avoir des milliers de commentaires
      - On veut analyser le sentiment des commentaires indépendamment
      - Les commentaires ont leur propre cycle de vie (suppression, signalement)
      - Les threads de réponses (parent_comment_id) forment un arbre navigable

    Notes par plateforme :
      twitter   → les réponses sont elles-mêmes des tweets, platform_id = tweet_id
      tiktok    → commentaires + réponses aux commentaires
      instagram → commentaires sur posts/reels, pas de commentaires sur stories
      telegram  → les réponses à des messages dans un groupe/canal
    """
    assert platform in PLATFORMS, f"Plateforme inconnue : {platform}"
    now = _now()
    return {
        "_id": _id(),

        # --- Identification ---
        "platform":                     platform,
        "platform_id":                  platform_id,
        "post_id":                      post_id,            # référence → posts
        "post_platform_id":             post_platform_id,   # dénormalisé
        "account_id":                   account_id,         # auteur du commentaire
        "account_platform_id":          account_platform_id,

        # --- Hiérarchie (thread) ---
        # None = commentaire de premier niveau
        # ObjectId = réponse à un autre commentaire
        "parent_comment_id":            parent_comment_id,
        "parent_comment_platform_id":   parent_comment_platform_id,
        "depth":                        0 if parent_comment_id is None else 1,

        # --- Contenu ---
        "text": {
            "content":    text_content,
            "language":   None,
            "mentions":   [],
            "urls":       [],
        },

        # --- Médias (rare mais possible : GIFs sur Twitter, stickers Telegram) ---
        "media": [],

        # --- Engagement ---
        "engagement": {
            "likes":      0,
            "replies":    0,
            "scraped_at": now,
        },

        # --- Contexte ---
        "published_at": None,

        # --- Analyse NLP ---
        # Note : on n'analyse pas le deepfake des commentaires (texte + petits médias)
        "nlp": {
            "status":           "pending",
            "sentiment": {
                "label":  None,
                "score":  None,
                "model":  None,
            },
            "embedding_model":  None,
            "topics":           [],
            "is_duplicate_of":  None,
            "similarity_score": None,
            "processed_at":     None,
            "error":            None,
        },

        # --- Synchronisation ETL ---
        "sync": {
            "neo4j":         False,
            "elasticsearch": False,
            "synced_at":     None,
        },

        # --- Cycle de vie ---
        "scraped_at": now,
        "updated_at": now,
        "raw":        raw or {},
    }


def patch_comment_nlp(
    sentiment_label: str,
    sentiment_score: float,
    sentiment_model: str,
    embedding_model: str,
    topics: list[str],
    embedding: Optional[bytes] = None,
    is_duplicate_of: Optional[ObjectId] = None,
    similarity_score: Optional[float] = None,
) -> dict:
    update = {
        "nlp.status":             "done",
        "nlp.sentiment.label":    sentiment_label,
        "nlp.sentiment.score":    sentiment_score,
        "nlp.sentiment.model":    sentiment_model,
        "nlp.embedding_model":    embedding_model,
        "nlp.topics":             topics,
        "nlp.is_duplicate_of":    is_duplicate_of,
        "nlp.similarity_score":   similarity_score,
        "nlp.processed_at":       _now(),
        "nlp.error":              None,
        "updated_at":             _now(),
    }
    if embedding is not None:
        update["nlp.embedding"] = embedding
    return {"$set": update}


def patch_comment_sync(neo4j: bool = False, elasticsearch: bool = False) -> dict:
    update: dict[str, Any] = {"updated_at": _now(), "sync.synced_at": _now()}
    if neo4j:
        update["sync.neo4j"] = True
    if elasticsearch:
        update["sync.elasticsearch"] = True
    return {"$set": update}


# ===========================================================================
# Collection : media
# ===========================================================================

def new_media(
    media_type: str,
    url_original: str,
    url_local: Optional[str] = None,
    source: Optional[dict] = None,
) -> dict:
    """
    Un document par fichier physique.
    Séparé de posts pour détecter la réutilisation du même média
    sur plusieurs comptes / plateformes (signal fort de campagne coordonnée).

    hash_md5          → doublons exacts
    hash_perceptual   → near-duplicates visuels (pHash)
    source            → {project, scan, user} hérité du worker_import [v4]
    """
    assert media_type in MEDIA_TYPES, f"Type inconnu : {media_type}"
    now = _now()
    return {
        "_id": _id(),

        "type":            media_type,
        "url_original":    url_original,
        "url_local":       url_local,
        "hash_md5":        None,
        "hash_perceptual": None,

        # --- Origine scrapper [v4] ---
        "source": source or {},   # {project, scan, user}

        "metadata": {
            "size_bytes":   None,
            "duration_sec": None,
            "width":        None,
            "height":       None,
            "fps":          None,
            "format":       None,
            "codec":        None,
        },

        "deepfake": {
            "status":           "pending",
            "final_score":      None,
            "prediction":       None,
            "model_divergence": None,
            "scores":           {},
            "raw_scores":       {},
            "artifact_score":   None,
            "frames_analyzed":  None,
            "faces_detected":   None,
            "pipeline_version": None,
            "processed_at":     None,
            "error":            None,
        },

        # Réutilisation : ce média a-t-il été posté par plusieurs comptes ?
        "reuse": {
            "post_ids":      [],
            "seen_count":    1,
            "first_seen_at": now,
            "platforms":     [],
        },

        "downloaded_at": now,
        "updated_at":    now,

        # --- Synchronisation ETL [v4] ---
        "sync": {
            "neo4j":         False,
            "elasticsearch": False,
            "synced_at":     None,
        },
    }


def patch_media_deepfake(result: dict, pipeline_version: str = "3.4.2") -> dict:
    scores     = {k[6:]: v for k, v in result.items() if k.startswith("score_")}
    raw_scores = {k[4:]: v for k, v in result.items() if k.startswith("raw_")}
    return {
        "$set": {
            "deepfake.status":           "done",
            "deepfake.final_score":      result.get("final_score"),
            "deepfake.prediction":       result.get("prediction"),
            "deepfake.model_divergence": result.get("model_divergence"),
            "deepfake.artifact_score":   result.get("artifact_score"),
            "deepfake.frames_analyzed":  result.get("frames_analyzed"),
            "deepfake.faces_detected":   result.get("faces_detected"),
            "deepfake.scores":           scores,
            "deepfake.raw_scores":       raw_scores,
            "deepfake.pipeline_version": pipeline_version,
            "deepfake.processed_at":     _now(),
            "deepfake.error":            None,
            "updated_at":                _now(),
        }
    }


def patch_media_reuse(post_id: ObjectId, platform: str) -> dict:
    return {
        "$addToSet": {"reuse.post_ids": post_id, "reuse.platforms": platform},
        "$inc":      {"reuse.seen_count": 1},
        "$set":      {"updated_at": _now()},
    }


def patch_media_sync(neo4j: bool = False, elasticsearch: bool = False) -> dict:
    """Marque le document media comme synchronisé vers Neo4j / Elasticsearch."""
    update: dict[str, Any] = {"updated_at": _now(), "sync.synced_at": _now()}
    if neo4j:
        update["sync.neo4j"] = True
    if elasticsearch:
        update["sync.elasticsearch"] = True
    return {"$set": update}


# ===========================================================================
# Collection : narratives
# ===========================================================================

def new_narrative(
    label: str,
    keywords: list[str],
    embedding_model: str,
    similarity_threshold: float = 0.82,
) -> dict:
    now = _now()
    return {
        "_id": _id(),

        "label":       label,
        "description": "",
        "keywords":    keywords,
        "hashtags":    [],

        "stats": {
            "post_count":       0,
            "comment_count":    0,      # ajout v2
            "account_count":    0,
            "platforms":        [],
            "first_seen_at":    now,
            "last_seen_at":     now,
            "peak_date":        None,
            "synthetic_ratio":  0.0,    # part de posts avec prediction=synthetic
        },

        "embedding_centroid":   None,
        "embedding_model":      embedding_model,
        "similarity_threshold": similarity_threshold,

        "review": {
            "status":      "pending",   # pending|reviewed|confirmed|dismissed
            "is_campaign": None,
            "reviewed_by": None,
            "reviewed_at": None,
            "notes":       "",
        },

        "created_at": now,
        "updated_at": now,
    }


# ===========================================================================
# Collection : campaigns
# ===========================================================================

def new_campaign(name: str, platforms: list[str]) -> dict:
    now = _now()
    return {
        "_id": _id(),

        "name":    name,
        "status":  "suspected",    # suspected|active|confirmed|closed

        "narrative_ids": [],
        "account_ids":   [],
        "platforms":     platforms,
        "date_range": {
            "start": now,
            "end":   None,
        },

        "signals": {
            "coordinated_posting":   False,  # pics de publication synchronisés
            "content_reuse":         False,  # même média sur plusieurs comptes
            "bot_accounts_ratio":    None,
            "synthetic_media_ratio": None,
            "cross_platform":        False,  # présent sur ≥2 plateformes
            "narrative_count":       0,
            # Signal spécifique Telegram : même message forwardé massivement
            "telegram_forward_burst": False,
        },

        "review": {
            "confidence":  None,    # [0–1] score de confiance automatique
            "confirmed":   False,
            "reviewed_by": None,
            "reviewed_at": None,
            "notes":       "",
        },

        "created_at": now,
        "updated_at": now,
    }


# ===========================================================================
# Collection : jobs
# ===========================================================================

def new_job(
    job_type: str,
    payload: dict,
    priority: int = 1,
    max_attempts: int = 3,
) -> dict:
    assert job_type in JOB_TYPES, f"Type inconnu : {job_type}"
    now = _now()
    return {
        "_id": _id(),

        "type":         job_type,
        "status":       "pending",   # pending|processing|done|error|retrying
        "priority":     priority,    # 0=haute, 1=normale, 2=basse
        "payload":      payload,

        "attempts":     0,
        "max_attempts": max_attempts,
        "last_error":   None,
        "worker_id":    None,

        "created_at":   now,
        "started_at":   None,
        "completed_at": None,
    }


def claim_job(worker_id: str) -> dict:
    """
    Paramètres pour un findOneAndUpdate atomique.
    Garantit qu'un seul worker prend un job donné, même avec plusieurs workers
    en parallèle (pas de race condition).
    """
    return {
        "filter": {"status": "pending", "attempts": {"$lt": 3}},
        "sort":   [("priority", ASCENDING), ("created_at", ASCENDING)],
        "update": {
            "$set": {"status": "processing", "worker_id": worker_id, "started_at": _now()},
            "$inc": {"attempts": 1},
        },
        "return_document": True,
    }


def complete_job() -> dict:
    return {"$set": {"status": "done", "completed_at": _now()}}


def fail_job(error: str, retry: bool = True) -> dict:
    return {
        "$set": {
            "status":     "retrying" if retry else "error",
            "last_error": error,
            "worker_id":  None,
        }
    }


# ===========================================================================
# Index MongoDB
# ===========================================================================

def create_indexes(db: Any) -> None:
    """
    Crée tous les index sur la base `db`.
    Idempotent : MongoDB ignore les index déjà existants.

    Usage :
        client = MongoClient("mongodb://localhost:27017")
        db = client["influence_detection"]
        create_indexes(db)
    """

    # ------ accounts ------
    db.accounts.create_indexes([
        IndexModel([("platform", ASCENDING), ("platform_id", ASCENDING)], unique=True),
        IndexModel([("username", ASCENDING), ("platform", ASCENDING)]),
        IndexModel([("analysis.status", ASCENDING)]),
        IndexModel([("flags.campaign_ids", ASCENDING)]),
        IndexModel([("stats.followers_count", DESCENDING)]),
        # Canaux/groupes Telegram publics
        IndexModel([("platform_specific.member_count", DESCENDING)], sparse=True),
    ])

    # ------ posts ------
    db.posts.create_indexes([
        IndexModel([("platform", ASCENDING), ("platform_id", ASCENDING)], unique=True),
        IndexModel([("account_id", ASCENDING)]),
        IndexModel([("context.published_at", DESCENDING)]),
        IndexModel([("deepfake.status", ASCENDING)]),
        IndexModel([("nlp.status", ASCENDING)]),
        IndexModel([("deepfake.prediction", ASCENDING)]),
        IndexModel([("sync.neo4j", ASCENDING), ("sync.elasticsearch", ASCENDING)]),
        IndexModel([("text.hashtags", ASCENDING)]),
        IndexModel([("nlp.narrative_id", ASCENDING)]),
        # Pipeline deepfake : requête la plus fréquente du worker
        IndexModel([
            ("deepfake.status", ASCENDING),
            ("deepfake.has_media", ASCENDING),
            ("scraped_at", ASCENDING),
        ]),
        # Détection de forwards Telegram massifs
        IndexModel([("platform_specific.forward_from", ASCENDING)], sparse=True),
        IndexModel([("platform_specific.forward_count", DESCENDING)], sparse=True),
    ])

    # ------ comments ------
    db.comments.create_indexes([
        IndexModel([("platform", ASCENDING), ("platform_id", ASCENDING)], unique=True),
        IndexModel([("post_id", ASCENDING)]),
        IndexModel([("account_id", ASCENDING)]),
        IndexModel([("parent_comment_id", ASCENDING)], sparse=True),
        IndexModel([("published_at", DESCENDING)]),
        IndexModel([("nlp.status", ASCENDING)]),
        IndexModel([("nlp.sentiment.label", ASCENDING)]),
        IndexModel([("sync.neo4j", ASCENDING), ("sync.elasticsearch", ASCENDING)]),
    ])

    # ------ media ------
    db.media.create_indexes([
        IndexModel([("hash_md5", ASCENDING)],        unique=True, sparse=True),
        IndexModel([("hash_perceptual", ASCENDING)], sparse=True),
        IndexModel([("deepfake.status", ASCENDING)]),
        IndexModel([("deepfake.prediction", ASCENDING)]),
        IndexModel([("reuse.post_ids", ASCENDING)]),
        IndexModel([("reuse.seen_count", DESCENDING)]),
        IndexModel([("sync.neo4j", ASCENDING)]),   # ETL tracking [v4]
    ])

    # ------ narratives ------
    db.narratives.create_indexes([
        IndexModel([("review.status", ASCENDING)]),
        IndexModel([("stats.last_seen_at", DESCENDING)]),
        IndexModel([("keywords", ASCENDING)]),
    ])

    # ------ campaigns ------
    db.campaigns.create_indexes([
        IndexModel([("status", ASCENDING)]),
        IndexModel([("narrative_ids", ASCENDING)]),
        IndexModel([("account_ids", ASCENDING)]),
    ])

    # ------ jobs ------
    db.jobs.create_indexes([
        IndexModel([
            ("status", ASCENDING),
            ("priority", ASCENDING),
            ("created_at", ASCENDING),
        ]),
        IndexModel([("type", ASCENDING), ("status", ASCENDING)]),
        # TTL : supprime les jobs terminés après 7 jours
        IndexModel(
            [("completed_at", ASCENDING)],
            expireAfterSeconds=7 * 24 * 3600,
            sparse=True,
        ),
    ])

    print("✓ Index MongoDB créés (7 collections).")


# ===========================================================================
# Exemple d'utilisation
# ===========================================================================

# ===========================================================================
# Exemple d'utilisation
# ===========================================================================

if __name__ == "__main__":
    """
    Lance ce script directement pour :
      1. Créer les index sur toutes les collections
      2. Insérer quelques documents de test

    Prérequis — créer le fichier .env dans le même dossier :
        MONGO_HOST=localhost
        MONGO_PORT=27017
        MONGO_USER=influence_app
        MONGO_PASSWORD=AppPassword456!
        MONGO_DB=influence_detection
        MONGO_AUTH_DB=influence_detection
        MONGO_ADMIN_USER=admin
        MONGO_ADMIN_PASSWORD=ChangeThisPassword123!

    Puis lancer :
        pip install python-dotenv
        python schema.py
    """
    import sys

    print("Connexion à MongoDB...")

    # Création des index avec le compte admin (droits étendus)
    try:
        admin_client = get_admin_client()
        admin_db = admin_client[os.getenv("MONGO_DB", "influence_detection")]
        create_indexes(admin_db)
        admin_client.close()
    except ConnectionError as e:
        print(f"\n❌ {e}")
        sys.exit(1)

    # Connexion applicative pour les insertions
    try:
        db = get_db()
        print(f"Connecté à la base : {db.name}\n")
    except ConnectionError as e:
        print(f"\n❌ {e}")
        sys.exit(1)

    # --- Compte Telegram (canal d'information) ---
    channel_doc = new_account(
        platform="telegram",
        platform_id="1234567890",
        username="canal_info_politique",
        display_name="Info Politique FR",
    )
    channel_doc["platform_specific"]["is_channel"] = True
    channel_doc["platform_specific"]["member_count"] = 45000
    channel_id = db.accounts.insert_one(channel_doc).inserted_id

    # --- Post Telegram avec vidéo ---
    post_doc = new_post(
        platform="telegram",
        platform_id="msg_99887766",
        account_id=channel_id,
        account_platform_id="1234567890",
        text_content="Regardez cette vidéo exclusive ! 🔴 #Élections2026",
    )
    post_doc["platform_specific"]["views"] = 98000
    post_doc["platform_specific"]["forward_count"] = 3400
    post_doc["platform_specific"]["is_forwarded"] = True
    post_doc["platform_specific"]["forward_from"] = "canal_etranger_suspect"
    post_id = db.posts.insert_one(post_doc).inserted_id

    # --- Commentaire sur ce post ---
    comment_doc = new_comment(
        platform="telegram",
        platform_id="reply_11223344",
        post_id=post_id,
        post_platform_id="msg_99887766",
        account_id=channel_id,
        account_platform_id="1234567890",
        text_content="Incroyable, partagez au maximum !",
    )
    comment_id = db.comments.insert_one(comment_doc).inserted_id

    # --- Média associé ---
    media_doc = new_media(
        "video",
        "https://cdn.telegram.org/file/abc123.mp4",
        "/storage/telegram/abc123.mp4",
    )
    media_id = db.media.insert_one(media_doc).inserted_id

    # Attacher le média au post
    db.posts.update_one({"_id": post_id}, patch_post_media({
        "media_id":     media_id,
        "type":         "video",
        "url_original": "https://cdn.telegram.org/file/abc123.mp4",
        "url_local":    "/storage/telegram/abc123.mp4",
        "downloaded":   True,
    }))

    # --- Résultat du script de détection deepfake ---
    result = {
        "final_score": 0.91, "prediction": "synthetic",
        "model_divergence": 0.07, "artifact_score": 0.41,
        "frames_analyzed": 52,
        "score_sdxl-detector": 0.94, "raw_sdxl-detector": 0.96,
    }
    db.posts.update_one({"_id": post_id},  patch_post_deepfake(result))
    db.media.update_one({"_id": media_id}, patch_media_deepfake(result))

    print(f"Compte  inséré : {channel_id}")
    print(f"Post    inséré : {post_id}")
    print(f"Comment inséré : {comment_id}")
    print(f"Média   inséré : {media_id}")
    print("\n✓ Schéma v3 opérationnel")

    # Vérification finale
    print(f"\nCollections présentes : {db.list_collection_names()}")
    print(f"Posts en base         : {db.posts.count_documents({})}")

