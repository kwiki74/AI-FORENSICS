"""tools.py
==========
Outils de lecture pour l'agent d'investigation AI-FORENSICS.

Chaque fonction est indépendante, testable en isolation, et retourne
un dict/list résumé — jamais le document MongoDB brut complet — pour
ne pas saturer la fenêtre de contexte du LLM.

Contrainte stricte : LECTURE SEULE. Aucune écriture MongoDB / Neo4j.

Utilisation standalone (test) :
    python tools.py --test-account instagram cryptocom

Dépendances :
    pip install pymongo neo4j python-dotenv
    (schema.py doit être dans le même dossier ou dans PYTHONPATH)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from bson import ObjectId
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Chargement .env (cherche dans le dossier parent = racine du projet)
# ---------------------------------------------------------------------------
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)

# ---------------------------------------------------------------------------
# Import schema.py depuis AI-FORENSICS/SCHEMA/
# Structure : AI-FORENSICS/AI/tools.py → AI-FORENSICS/SCHEMA/schema.py
# ---------------------------------------------------------------------------
_SCHEMA_DIR = Path(__file__).resolve().parent.parent / "SCHEMA"
if str(_SCHEMA_DIR) not in sys.path:
    sys.path.insert(0, str(_SCHEMA_DIR))

from schema import get_db  # noqa: E402

logger = logging.getLogger("ai_forensics.agent.tools")


# ===========================================================================
# Connexion MongoDB (singleton léger — une connexion par process)
# ===========================================================================

_db = None


def _get_db():
    global _db
    if _db is None:
        _db = get_db()
    return _db


# ===========================================================================
# Connexion Neo4j (singleton)
# ===========================================================================

_neo4j_driver = None


def _get_neo4j():
    global _neo4j_driver
    if _neo4j_driver is None:
        try:
            from neo4j import GraphDatabase
            uri  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER",     "neo4j")
            pwd  = os.getenv("NEO4J_PASSWORD", "")
            _neo4j_driver = GraphDatabase.driver(uri, auth=(user, pwd))
            _neo4j_driver.verify_connectivity()
        except Exception as exc:
            logger.warning("Neo4j non disponible : %s", exc)
            _neo4j_driver = None
    return _neo4j_driver


# ===========================================================================
# Helpers internes
# ===========================================================================

def _str_id(doc_id) -> str:
    """Convertit ObjectId en str pour la sérialisation JSON."""
    return str(doc_id) if doc_id else None


def _round(v, n=3):
    return round(v, n) if isinstance(v, float) else v


def _safe_run(fn_name: str, fn, *args, **kwargs):
    """Wrapper try/except — retourne un dict d'erreur si l'outil plante."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.error("[%s] Erreur : %s", fn_name, exc, exc_info=True)
        return {"error": str(exc), "tool": fn_name}


# ===========================================================================
# Outil 1 — Profil d'un compte
# ===========================================================================

def get_account_info(platform: str, unique_id: str) -> dict:
    """
    Retourne le profil, les stats et les scores agrégés d'un compte.

    Recherche par username OU platform_id (tente les deux).

    Args:
        platform  : instagram | tiktok | twitter | telegram
        unique_id : @handle ou platform_id numérique

    Returns:
        dict avec profil, stats, analyse, flags, résumé deepfake agrégé.
    """
    def _run():
        db = _get_db()

        # Recherche flexible : username ou platform_id
        uid_clean = unique_id.lstrip("@")
        query = {
            "platform": platform,
            "$or": [
                {"username":     uid_clean},
                {"platform_id":  uid_clean},
                {"display_name": uid_clean},
            ],
        }
        doc = db.accounts.find_one(query)
        if not doc:
            return {"error": f"Compte introuvable : {platform}/{unique_id}"}

        account_id = doc["_id"]

        # Agrégation deepfake sur les posts du compte
        pipeline = [
            {"$match": {
                "account_id": account_id,
                "deepfake.status": "done",
                "deepfake.final_score": {"$ne": None},
            }},
            {"$group": {
                "_id": None,
                "total_posts":        {"$sum": 1},
                "avg_score":          {"$avg": "$deepfake.final_score"},
                "synthetic_count":    {"$sum": {"$cond": [{"$eq": ["$deepfake.prediction", "synthetic"]}, 1, 0]}},
                "suspicious_count":   {"$sum": {"$cond": [{"$eq": ["$deepfake.prediction", "suspicious"]}, 1, 0]}},
                "avg_divergence":     {"$avg": "$deepfake.model_divergence"},
            }},
        ]
        agg = list(db.posts.aggregate(pipeline))
        deepfake_summary = {}
        if agg:
            r = agg[0]
            total = r["total_posts"] or 1
            deepfake_summary = {
                "posts_analyzed":   r["total_posts"],
                "avg_score":        _round(r["avg_score"]),
                "synthetic_count":  r["synthetic_count"],
                "suspicious_count": r["suspicious_count"],
                "synthetic_ratio":  _round(r["synthetic_count"] / total),
                "avg_divergence":   _round(r["avg_divergence"]),
            }

        return {
            "account_id":   _str_id(account_id),
            "platform":     doc.get("platform"),
            "platform_id":  doc.get("platform_id"),
            "username":     doc.get("username"),
            "display_name": doc.get("display_name"),
            "url":          doc.get("url"),
            "profile": {
                "bio":        doc.get("profile", {}).get("bio"),
                "location":   doc.get("profile", {}).get("location"),
                "verified":   doc.get("profile", {}).get("verified"),
                "created_at": str(doc.get("profile", {}).get("created_at") or ""),
                "language":   doc.get("profile", {}).get("language"),
            },
            "stats": doc.get("stats", {}),
            "analysis": {
                "bot_score":         doc.get("analysis", {}).get("bot_score"),
                "bot_signals":       doc.get("analysis", {}).get("bot_signals", {}),
                "language_detected": doc.get("analysis", {}).get("language_detected"),
                "narratives":        [_str_id(n) for n in doc.get("analysis", {}).get("narratives", [])],
            },
            "flags": {
                "is_suspicious":    doc.get("flags", {}).get("is_suspicious"),
                "is_confirmed_bot": doc.get("flags", {}).get("is_confirmed_bot"),
                "campaign_ids":     [_str_id(c) for c in doc.get("flags", {}).get("campaign_ids", [])],
            },
            "deepfake_summary": deepfake_summary,
        }

    return _safe_run("get_account_info", _run)


# ===========================================================================
# Outil 2 — Posts récents d'un compte
# ===========================================================================

def get_account_posts(account_id: str, limit: int = 20) -> list:
    """
    Retourne les derniers posts d'un compte avec scores deepfake et NLP.

    Args:
        account_id : _id MongoDB du compte (str)
        limit      : nombre maximum de posts à retourner (défaut 20, max 50)

    Returns:
        Liste de dicts résumant chaque post (texte tronqué, scores, sentiment).
    """
    def _run():
        db  = _get_db()
        oid = ObjectId(account_id)
        limit_safe = min(int(limit), 50)

        cursor = db.posts.find(
            {"account_id": oid},
            {
                "platform": 1, "platform_id": 1, "url": 1,
                "text.content": 1, "text.hashtags": 1,
                "engagement": 1, "context.published_at": 1,
                "deepfake.final_score": 1, "deepfake.prediction": 1,
                "deepfake.model_divergence": 1, "deepfake.status": 1,
                "nlp.sentiment": 1, "nlp.narrative_id": 1,
                "nlp.is_duplicate_of": 1,
            },
        ).sort("context.published_at", -1).limit(limit_safe)

        posts = []
        for doc in cursor:
            text_raw = (doc.get("text") or {}).get("content", "") or ""
            posts.append({
                "post_id":       _str_id(doc["_id"]),
                "platform_id":   doc.get("platform_id"),
                "url":           doc.get("url"),
                "published_at":  str((doc.get("context") or {}).get("published_at") or ""),
                "text_preview":  text_raw[:200] + ("…" if len(text_raw) > 200 else ""),
                "hashtags":      (doc.get("text") or {}).get("hashtags", []),
                "engagement": {
                    "likes":    (doc.get("engagement") or {}).get("likes", 0),
                    "shares":   (doc.get("engagement") or {}).get("shares", 0),
                    "views":    (doc.get("engagement") or {}).get("views", 0),
                },
                "deepfake": {
                    "status":     (doc.get("deepfake") or {}).get("status"),
                    "score":      _round((doc.get("deepfake") or {}).get("final_score")),
                    "prediction": (doc.get("deepfake") or {}).get("prediction"),
                    "divergence": _round((doc.get("deepfake") or {}).get("model_divergence")),
                },
                "nlp": {
                    "sentiment":      (doc.get("nlp") or {}).get("sentiment", {}),
                    "narrative_id":   _str_id((doc.get("nlp") or {}).get("narrative_id")),
                    "is_duplicate_of": _str_id((doc.get("nlp") or {}).get("is_duplicate_of")),
                },
            })

        return posts

    return _safe_run("get_account_posts", _run)


# ===========================================================================
# Outil 3 — Scores deepfake détaillés (par modèle) des médias d'un compte
# ===========================================================================

def get_media_scores(account_id: str) -> list:
    """
    Retourne les scores deepfake détaillés (par modèle) des médias d'un compte.

    Utile pour détecter la divergence inter-modèles et identifier les médias
    les plus suspects.

    Args:
        account_id : _id MongoDB du compte (str)

    Returns:
        Liste de dicts, un par média analysé, triés par score décroissant.
    """
    def _run():
        db  = _get_db()
        oid = ObjectId(account_id)

        # Récupère le compte pour avoir username et display_name
        acc = db.accounts.find_one({"_id": oid}, {"username": 1, "display_name": 1})
        if not acc:
            return []

        # Stratégie 1 : via reuse.post_ids (si worker déduplication a tourné)
        post_ids = [doc["_id"] for doc in db.posts.find({"account_id": oid}, {"_id": 1})]
        count_by_posts = db.media.count_documents(
            {"reuse.post_ids": {"$in": post_ids}, "deepfake.status": "done"}
        ) if post_ids else 0

        # Stratégie 2 : via source.user (username ou display_name)
        usernames = [v for v in [acc.get("username"), acc.get("display_name")] if v]
        count_by_source = db.media.count_documents(
            {"source.user": {"$in": usernames}, "deepfake.status": "done"}
        ) if usernames else 0

        # Choisit la stratégie qui retourne le plus de résultats
        if count_by_posts >= count_by_source and count_by_posts > 0:
            final_query = {"reuse.post_ids": {"$in": post_ids}, "deepfake.status": "done"}
        elif count_by_source > 0:
            final_query = {"source.user": {"$in": usernames}, "deepfake.status": "done"}
        else:
            return []

        cursor = db.media.find(
            final_query,
            {
                "type": 1, "url_local": 1,
                "deepfake.final_score": 1, "deepfake.prediction": 1,
                "deepfake.scores": 1, "deepfake.raw_scores": 1,
                "deepfake.model_divergence": 1, "deepfake.artifact_score": 1,
                "reuse.seen_count": 1, "reuse.platforms": 1,
            },
        ).sort("deepfake.final_score", -1).limit(100)

        results = []
        for doc in cursor:
            df = doc.get("deepfake", {})
            results.append({
                "media_id":        _str_id(doc["_id"]),
                "type":            doc.get("type"),
                "url_local":       doc.get("url_local"),
                "final_score":     _round(df.get("final_score")),
                "prediction":      df.get("prediction"),
                "artifact_score":  _round(df.get("artifact_score")),
                "model_divergence": _round(df.get("model_divergence")),
                "scores_by_model": {
                    k: _round(v) for k, v in (df.get("scores") or {}).items()
                },
                "reuse": {
                    "seen_count": (doc.get("reuse") or {}).get("seen_count", 1),
                    "platforms":  (doc.get("reuse") or {}).get("platforms", []),
                },
            })

        return results

    return _safe_run("get_media_scores", _run)


# ===========================================================================
# Outil 4 — Voisinage dans le graphe Neo4j
# ===========================================================================

def get_graph_neighbors(platform_id: str, depth: int = 2) -> dict:
    """
    Retourne le voisinage d'un compte dans le graphe Neo4j.

    Collecte : comptes liés, hashtags partagés, médias réutilisés,
    narratives et campagnes associées.

    Args:
        platform_id : identifiant plateforme du compte (ex: "123456789")
        depth       : profondeur du parcours (max 2 pour éviter l'explosion)

    Returns:
        dict avec comptes voisins, hashtags, médias réutilisés, scores de centralité.
    """
    def _run():
        driver = _get_neo4j()
        if driver is None:
            return {"error": "Neo4j non disponible"}

        depth_safe = min(int(depth), 2)

        with driver.session() as session:

            # --- 1. Comptes de la même communauté Louvain ---
            # community_id calculé par le worker réseau via GDS Louvain
            community_result = session.run(
                """
                MATCH (a:Account {platform_id: $pid})
                WHERE a.community_id IS NOT NULL AND a.community_id <> 0
                MATCH (neighbor:Account {community_id: a.community_id})
                WHERE neighbor.platform_id <> $pid
                RETURN DISTINCT
                    neighbor.platform_id    AS platform_id,
                    neighbor.display_name   AS username,
                    neighbor.platform       AS platform,
                    neighbor.pagerank_score AS pagerank,
                    neighbor.is_suspicious  AS is_suspicious,
                    neighbor.community_id   AS community_id
                ORDER BY neighbor.pagerank_score DESC
                LIMIT 20
                """,
                pid=platform_id,
            )
            community_accounts = [dict(r) for r in community_result]

            # --- 2. Hashtags les plus utilisés par ce compte ---
            hashtags_result = session.run(
                """
                MATCH (a:Account {platform_id: $pid})-[:A_PUBLIÉ]->(p:Post)-[:HAS_HASHTAG]->(h:Hashtag)
                RETURN h.name AS tag, count(p) AS usage
                ORDER BY usage DESC
                LIMIT 20
                """,
                pid=platform_id,
            )
            hashtags = [dict(r) for r in hashtags_result]

            # --- 3. Comptes partageant les mêmes hashtags (coordination potentielle) ---
            shared_hashtag_accounts_result = session.run(
                """
                MATCH (a:Account {platform_id: $pid})-[:A_PUBLIÉ]->(:Post)-[:HAS_HASHTAG]->(h:Hashtag)
                MATCH (h)<-[:HAS_HASHTAG]-(:Post)<-[:A_PUBLIÉ]-(other:Account)
                WHERE other.platform_id <> $pid
                WITH other, count(DISTINCT h) AS shared_tags
                WHERE shared_tags >= 2
                RETURN DISTINCT
                    other.platform_id   AS platform_id,
                    other.display_name  AS username,
                    other.platform      AS platform,
                    shared_tags
                ORDER BY shared_tags DESC
                LIMIT 15
                """,
                pid=platform_id,
            )
            shared_hashtag_accounts = [dict(r) for r in shared_hashtag_accounts_result]

            # --- 4. Posts dupliqués — coordination par copie de contenu ---
            duplicates_result = session.run(
                """
                MATCH (a:Account {platform_id: $pid})-[:A_PUBLIÉ]->(p:Post)
                MATCH (p)-[:EST_DOUBLON_DE]->(original:Post)<-[:A_PUBLIÉ]-(other:Account)
                WHERE other.platform_id <> $pid
                RETURN DISTINCT
                    other.platform_id   AS platform_id,
                    other.display_name  AS username,
                    other.platform      AS platform,
                    count(p)            AS duplicate_count,
                    original.text       AS original_text_preview
                ORDER BY duplicate_count DESC
                LIMIT 10
                """,
                pid=platform_id,
            )
            duplicated_from = [dict(r) for r in duplicates_result]

            # Autres comptes qui copient les posts de ce compte
            copied_by_result = session.run(
                """
                MATCH (a:Account {platform_id: $pid})-[:A_PUBLIÉ]->(original:Post)
                MATCH (copy:Post)-[:EST_DOUBLON_DE]->(original)
                MATCH (copy)<-[:A_PUBLIÉ]-(copier:Account)
                WHERE copier.platform_id <> $pid
                RETURN DISTINCT
                    copier.platform_id  AS platform_id,
                    copier.display_name AS username,
                    copier.platform     AS platform,
                    count(copy)         AS copy_count
                ORDER BY copy_count DESC
                LIMIT 10
                """,
                pid=platform_id,
            )
            copied_by = [dict(r) for r in copied_by_result]

            # --- 5. Narratives et campagnes associées ---
            narratives_result = session.run(
                """
                MATCH (a:Account {platform_id: $pid})-[:A_PUBLIÉ]->(p:Post)-[:APPARTIENT_À]->(n:Narrative)
                WITH n, count(p) AS post_count
                OPTIONAL MATCH (camp:Campaign)-[:COUVRE]->(n)
                RETURN DISTINCT
                    n.label      AS label,
                    n.post_count AS total_posts,
                    n.keywords   AS keywords,
                    post_count   AS account_posts,
                    camp.name    AS campaign_name,
                    camp.score   AS campaign_score
                ORDER BY post_count DESC
                LIMIT 10
                """,
                pid=platform_id,
            )
            narratives = [dict(r) for r in narratives_result]

            # --- 6. Campagnes directement liées (via Narrative) ---
            campaigns_result = session.run(
                """
                MATCH (a:Account {platform_id: $pid})-[:A_PUBLIÉ]->(:Post)-[:APPARTIENT_À]->(n:Narrative)
                MATCH (camp:Campaign)-[:COUVRE]->(n)
                RETURN DISTINCT
                    camp.name         AS name,
                    camp.score        AS score,
                    camp.signal_count AS signal_count,
                    camp.signals      AS signals,
                    camp.platforms    AS platforms
                LIMIT 5
                """,
                pid=platform_id,
            )
            campaigns = [dict(r) for r in campaigns_result]

        return {
            "platform_id":            platform_id,
            "community_accounts":     community_accounts,
            "shared_hashtag_accounts": shared_hashtag_accounts,
            "duplicated_from":        duplicated_from,
            "copied_by":              copied_by,
            "shared_hashtags":        hashtags,
            "narratives":             narratives,
            "campaigns":              campaigns,
            "summary": {
                "community_size":          len(community_accounts),
                "shared_hashtag_accounts": len(shared_hashtag_accounts),
                "duplicate_sources":       len(duplicated_from),
                "copiers":                 len(copied_by),
                "hashtag_count":           len(hashtags),
                "narrative_count":         len(narratives),
                "campaign_count":          len(campaigns),
            },
        }

    return _safe_run("get_graph_neighbors", _run)


# ===========================================================================
# Outil 5 — Informations sur une narrative
# ===========================================================================

def get_narrative(narrative_id: str) -> dict:
    """
    Retourne les informations d'une narrative (label, mots-clés,
    ratio synthétique, plateformes, comptes associés).

    Args:
        narrative_id : _id MongoDB de la narrative (str)

    Returns:
        dict avec label, keywords, stats, plateformes, top comptes.
    """
    def _run():
        db  = _get_db()
        oid = ObjectId(narrative_id)
        doc = db.narratives.find_one({"_id": oid})
        if not doc:
            return {"error": f"Narrative introuvable : {narrative_id}"}

        # Comptes qui portent cette narrative (via posts.nlp.narrative_id)
        post_pipeline = [
            {"$match": {"nlp.narrative_id": oid, "nlp.status": "done"}},
            {"$group": {
                "_id": "$account_id",
                "post_count": {"$sum": 1},
                "avg_score":  {"$avg": "$deepfake.final_score"},
            }},
            {"$sort": {"post_count": -1}},
            {"$limit": 10},
        ]
        top_accounts_raw = list(db.posts.aggregate(post_pipeline))

        # Récupère les usernames
        top_accounts = []
        for entry in top_accounts_raw:
            acc = db.accounts.find_one(
                {"_id": entry["_id"]},
                {"username": 1, "platform": 1},
            )
            top_accounts.append({
                "account_id": _str_id(entry["_id"]),
                "username":   acc.get("username") if acc else None,
                "platform":   acc.get("platform") if acc else None,
                "post_count": entry["post_count"],
                "avg_deepfake_score": _round(entry.get("avg_score")),
            })

        return {
            "narrative_id":    _str_id(doc["_id"]),
            "label":           doc.get("label"),
            "keywords":        doc.get("keywords", []),
            "stats": {
                "post_count":      (doc.get("stats") or {}).get("post_count"),
                "account_count":   (doc.get("stats") or {}).get("account_count"),
                "synthetic_ratio": _round((doc.get("stats") or {}).get("synthetic_ratio")),
                "platforms":       (doc.get("stats") or {}).get("platforms", []),
                "first_seen_at":   str((doc.get("stats") or {}).get("first_seen_at") or ""),
                "last_seen_at":    str((doc.get("stats") or {}).get("last_seen_at") or ""),
            },
            "review": {
                "status":   (doc.get("review") or {}).get("status"),
                "label":    (doc.get("review") or {}).get("label"),
                "notes":    (doc.get("review") or {}).get("notes"),
            },
            "top_accounts": top_accounts,
        }

    return _safe_run("get_narrative", _run)


# ===========================================================================
# Outil 6 — Signaux d'une campagne détectée
# ===========================================================================

def get_campaign_signals(campaign_id: str) -> dict:
    """
    Retourne les signaux de coordination d'une campagne détectée.

    Args:
        campaign_id : _id MongoDB de la campagne (str)

    Returns:
        dict avec signaux, comptes membres, narratives associées, confiance.
    """
    def _run():
        db  = _get_db()
        oid = ObjectId(campaign_id)
        doc = db.campaigns.find_one({"_id": oid})
        if not doc:
            return {"error": f"Campagne introuvable : {campaign_id}"}

        # Résumé des comptes membres
        account_ids = [ObjectId(a) if not isinstance(a, ObjectId) else a
                       for a in doc.get("account_ids", [])]
        accounts_summary = []
        if account_ids:
            for acc in db.accounts.find(
                {"_id": {"$in": account_ids[:20]}},  # max 20
                {"username": 1, "platform": 1, "stats.followers_count": 1,
                 "analysis.bot_score": 1, "flags.is_suspicious": 1},
            ):
                accounts_summary.append({
                    "account_id":  _str_id(acc["_id"]),
                    "username":    acc.get("username"),
                    "platform":    acc.get("platform"),
                    "followers":   (acc.get("stats") or {}).get("followers_count"),
                    "bot_score":   _round((acc.get("analysis") or {}).get("bot_score")),
                    "suspicious":  (acc.get("flags") or {}).get("is_suspicious"),
                })

        # Résumé des narratives
        narrative_ids = [ObjectId(n) if not isinstance(n, ObjectId) else n
                         for n in doc.get("narrative_ids", [])]
        narratives_summary = []
        if narrative_ids:
            for narr in db.narratives.find(
                {"_id": {"$in": narrative_ids}},
                {"label": 1, "stats.synthetic_ratio": 1},
            ):
                narratives_summary.append({
                    "narrative_id":   _str_id(narr["_id"]),
                    "label":          narr.get("label"),
                    "synthetic_ratio": _round((narr.get("stats") or {}).get("synthetic_ratio")),
                })

        return {
            "campaign_id": _str_id(doc["_id"]),
            "name":        doc.get("name"),
            "status":      doc.get("status"),
            "signals": {
                "coordinated_posting":    (doc.get("signals") or {}).get("coordinated_posting"),
                "content_reuse":          (doc.get("signals") or {}).get("content_reuse"),
                "bot_accounts_ratio":     _round((doc.get("signals") or {}).get("bot_accounts_ratio")),
                "synthetic_media_ratio":  _round((doc.get("signals") or {}).get("synthetic_media_ratio")),
                "cross_platform":         (doc.get("signals") or {}).get("cross_platform"),
                "telegram_forward_burst": (doc.get("signals") or {}).get("telegram_forward_burst"),
                "narrative_count":        (doc.get("signals") or {}).get("narrative_count"),
            },
            "review": {
                "confidence":  _round((doc.get("review") or {}).get("confidence")),
                "confirmed":   (doc.get("review") or {}).get("confirmed"),
                "reviewed_by": (doc.get("review") or {}).get("reviewed_by"),
                "notes":       (doc.get("review") or {}).get("notes"),
            },
            "account_count":      len(account_ids),
            "accounts_sample":    accounts_summary,
            "narratives_summary": narratives_summary,
            "created_at": str(doc.get("created_at") or ""),
        }

    return _safe_run("get_campaign_signals", _run)


# ===========================================================================
# Outil 7 — Comptes associés à une narrative
# ===========================================================================

def search_accounts_by_narrative(narrative_id: str) -> list:
    """
    Retourne les comptes qui ont posté des contenus associés à une narrative.

    Recherche via posts.nlp.narrative_id → déduplique par account_id.

    Args:
        narrative_id : _id MongoDB de la narrative (str)

    Returns:
        Liste de comptes triés par nombre de posts dans cette narrative (desc).
    """
    def _run():
        db  = _get_db()
        oid = ObjectId(narrative_id)

        pipeline = [
            {"$match": {"nlp.narrative_id": oid, "nlp.status": "done"}},
            {"$group": {
                "_id": "$account_id",
                "post_count":       {"$sum": 1},
                "synthetic_posts":  {"$sum": {"$cond": [
                    {"$eq": ["$deepfake.prediction", "synthetic"]}, 1, 0
                ]}},
                "avg_score":        {"$avg": "$deepfake.final_score"},
                "platforms":        {"$addToSet": "$platform"},
            }},
            {"$sort": {"post_count": -1}},
            {"$limit": 20},
        ]
        grouped = list(db.posts.aggregate(pipeline))

        results = []
        for entry in grouped:
            acc = db.accounts.find_one(
                {"_id": entry["_id"]},
                {
                    "username": 1, "platform": 1, "display_name": 1,
                    "stats.followers_count": 1,
                    "analysis.bot_score": 1,
                    "flags.is_suspicious": 1,
                    "flags.campaign_ids": 1,
                },
            )
            if not acc:
                continue
            results.append({
                "account_id":       _str_id(acc["_id"]),
                "username":         acc.get("username"),
                "display_name":     acc.get("display_name"),
                "platform":         acc.get("platform"),
                "followers":        (acc.get("stats") or {}).get("followers_count"),
                "bot_score":        _round((acc.get("analysis") or {}).get("bot_score")),
                "is_suspicious":    (acc.get("flags") or {}).get("is_suspicious"),
                "campaign_ids":     [_str_id(c) for c in (acc.get("flags") or {}).get("campaign_ids", [])],
                "narrative_stats": {
                    "post_count":      entry["post_count"],
                    "synthetic_posts": entry["synthetic_posts"],
                    "synthetic_ratio": _round(
                        entry["synthetic_posts"] / entry["post_count"]
                        if entry["post_count"] else 0
                    ),
                    "avg_deepfake_score": _round(entry.get("avg_score")),
                    "platforms": entry.get("platforms", []),
                },
            })

        return results

    return _safe_run("search_accounts_by_narrative", _run)



def get_campaign_graph(campaign_mongo_id: str) -> dict:
    """
    Explore le graphe Neo4j d'une campagne détectée via le chemin :
    Campaign -[:COUVRE]-> Narrative <-[:APPARTIENT_À]- Post <-[:A_PUBLIÉ]- Account

    Retourne les comptes impliqués, leurs stats deepfake, les hashtags dominants
    et les relations de duplication de contenu entre comptes.

    Args:
        campaign_mongo_id : _id MongoDB de la campagne (str)

    Returns:
        dict avec comptes, hashtags, doublons, narratives, stats agrégées.
    """
    def _run():
        driver = _get_neo4j()
        if driver is None:
            return {"error": "Neo4j non disponible"}

        with driver.session() as session:

            # --- Narratives couvertes par la campagne ---
            narratives_result = session.run("""
                MATCH (c:Campaign)-[:COUVRE]->(n:Narrative)
                WHERE c.mongo_id = $cid
                RETURN c.name AS campaign_name, c.score AS campaign_score,
                       c.signals AS signals, c.platforms AS platforms,
                       n.label AS label, n.post_count AS post_count,
                       n.keywords AS keywords
            """, cid=campaign_mongo_id)
            narratives = [dict(r) for r in narratives_result]

            # --- Comptes impliqués via Narrative → Post → Account ---
            accounts_result = session.run("""
                MATCH (c:Campaign)-[:COUVRE]->(n:Narrative)<-[:APPARTIENT_À]-(p:Post)<-[:A_PUBLIÉ]-(a:Account)
                WHERE c.mongo_id = $cid
                WITH a,
                     count(p)                                                      AS post_count,
                     avg(p.deepfake_score)                                         AS avg_deepfake,
                     sum(CASE WHEN p.is_synthetic  THEN 1 ELSE 0 END)             AS synthetic_count,
                     sum(CASE WHEN p.is_duplicate  THEN 1 ELSE 0 END)             AS duplicate_count,
                     sum(p.likes + p.shares + p.views)                            AS total_engagement,
                     collect(DISTINCT p.sentiment_label)                           AS sentiments
                RETURN
                    a.display_name   AS username,
                    a.platform       AS platform,
                    a.platform_id    AS platform_id,
                    a.pagerank_score AS pagerank,
                    a.community_id   AS community_id,
                    a.is_suspicious  AS is_suspicious,
                    a.followers      AS followers,
                    post_count,
                    avg_deepfake,
                    synthetic_count,
                    duplicate_count,
                    total_engagement,
                    sentiments
                ORDER BY post_count DESC
                LIMIT 20
            """, cid=campaign_mongo_id)
            accounts = [dict(r) for r in accounts_result]

            # --- Hashtags dominants dans la campagne ---
            hashtags_result = session.run("""
                MATCH (c:Campaign)-[:COUVRE]->(n:Narrative)<-[:APPARTIENT_À]-(p:Post)-[:HAS_HASHTAG]->(h:Hashtag)
                WHERE c.mongo_id = $cid
                RETURN h.name AS hashtag, count(p) AS usage
                ORDER BY usage DESC
                LIMIT 20
            """, cid=campaign_mongo_id)
            hashtags = [dict(r) for r in hashtags_result]

            # --- Doublons de contenu entre comptes ---
            duplicates_result = session.run("""
                MATCH (c:Campaign)-[:COUVRE]->(n:Narrative)<-[:APPARTIENT_À]-(p:Post)
                      -[:EST_DOUBLON_DE]->(orig:Post)
                WHERE c.mongo_id = $cid
                MATCH (orig)<-[:A_PUBLIÉ]-(orig_acc:Account)
                MATCH (p)<-[:A_PUBLIÉ]-(copy_acc:Account)
                RETURN
                    copy_acc.display_name AS copier,
                    copy_acc.platform     AS copier_platform,
                    orig_acc.display_name AS original_author,
                    orig_acc.platform     AS original_platform,
                    count(*)              AS copies
                ORDER BY copies DESC
                LIMIT 15
            """, cid=campaign_mongo_id)
            duplicates = [dict(r) for r in duplicates_result]

            # --- Communautés représentées ---
            communities_result = session.run("""
                MATCH (c:Campaign)-[:COUVRE]->(n:Narrative)<-[:APPARTIENT_À]-(p:Post)<-[:A_PUBLIÉ]-(a:Account)
                WHERE c.mongo_id = $cid AND a.community_id IS NOT NULL
                RETURN a.community_id AS community_id, count(DISTINCT a) AS account_count
                ORDER BY account_count DESC
                LIMIT 10
            """, cid=campaign_mongo_id)
            communities = [dict(r) for r in communities_result]

        # Calculs agrégés
        platforms = list({a.get("platform") for a in accounts if a.get("platform")})
        total_posts = sum(a.get("post_count", 0) for a in accounts)
        total_duplicates = sum(d.get("copies", 0) for d in duplicates)
        avg_deepfake = (
            sum(a.get("avg_deepfake") or 0 for a in accounts if a.get("avg_deepfake")) /
            max(1, sum(1 for a in accounts if a.get("avg_deepfake")))
        )

        return {
            "campaign_mongo_id": campaign_mongo_id,
            "narratives":        narratives,
            "accounts":          accounts,
            "hashtags":          hashtags,
            "duplicates":        duplicates,
            "communities":       communities,
            "summary": {
                "account_count":       len(accounts),
                "platform_count":      len(platforms),
                "platforms":           platforms,
                "total_posts":         total_posts,
                "total_duplicates":    total_duplicates,
                "avg_deepfake_score":  _round(avg_deepfake),
                "narrative_count":     len(narratives),
                "has_cross_platform":  len(platforms) > 1,
                "has_content_reuse":   total_duplicates > 0,
            },
        }

    return _safe_run("get_campaign_graph", _run)


def get_temporal_analysis(entry_id: str, entry_type: str = "campaign") -> dict:
    """
    Analyse complète de la propagation temporelle des publications.

    Détecte :
    - Co-occurrence journalière multi-comptes
    - Régularité robotique (cadence mécanique)
    - Accélérations / pics d'activité coordonnée
    - Chronologie de propagation et compte semence
    - Silences suspects (gaps > 14 jours)
    - Corrélation deepfake × temporel
    - Comptes amplificateurs cross-campagne

    Args:
        entry_id   : mongo_id de la campagne, narrative ou account
        entry_type : "campaign" | "narrative" | "account"
    """
    def _run():
        driver = _get_neo4j()
        if driver is None:
            return {"error": "Neo4j non disponible"}

        with driver.session() as session:

            # Requête de base selon le type d'entrée
            if entry_type == "campaign":
                base_match = "MATCH (c:Campaign)-[:COUVRE]->(n:Narrative)<-[:APPARTIENT_À]-(p:Post)<-[:A_PUBLIÉ]-(a:Account) WHERE c.mongo_id = $eid"
                seed_match  = "MATCH (c:Campaign)-[:COUVRE]->(n:Narrative)<-[:APPARTIENT_À]-(p:Post)<-[:A_PUBLIÉ]-(a:Account) WHERE c.mongo_id = $eid"
                silence_match = None  # silences calculés par compte individuel
            elif entry_type == "narrative":
                base_match = "MATCH (n:Narrative)<-[:APPARTIENT_À]-(p:Post)<-[:A_PUBLIÉ]-(a:Account) WHERE n.mongo_id = $eid"
                seed_match  = base_match
                silence_match = None
            else:  # account — entry_id = platform_id
                base_match = "MATCH (a:Account {platform_id: $eid})-[:A_PUBLIÉ]->(p:Post)"
                seed_match  = None
                silence_match = "MATCH (a:Account {platform_id: $eid})-[:A_PUBLIÉ]->(p:Post)"

            # Pour campaign/narrative : base_match se termine par WHERE ... donc AND est valide
            # Pour account : base_match se termine par ->(p:Post) donc WHERE est requis
            if entry_type == "account":
                date_filter = "WHERE p.published_at IS NOT NULL AND p.published_at <> \'\'"
            else:
                date_filter = "AND p.published_at IS NOT NULL AND p.published_at <> \'\'"

            # --- 1. Posts par jour par compte ---
            timeline_result = session.run(
                f"{base_match} {date_filter} "
                "WITH substring(p.published_at, 0, 10) AS day, a.display_name AS account, "
                "a.platform AS platform, count(p) AS posts, "
                "sum(CASE WHEN p.is_duplicate THEN 1 ELSE 0 END) AS duplicates "
                "RETURN day, account, platform, posts, duplicates ORDER BY day, posts DESC",
                eid=entry_id,
            )
            timeline = [dict(r) for r in timeline_result]

            # --- 2. Co-occurrences multi-comptes ---
            cooccurrence_result = session.run(
                f"{base_match} {date_filter} "
                "WITH substring(p.published_at, 0, 10) AS day, "
                "count(DISTINCT a) AS active_accounts, count(p) AS total_posts, "
                "collect(DISTINCT a.display_name) AS accounts_list "
                "WHERE active_accounts > 1 "
                "RETURN day, active_accounts, total_posts, accounts_list "
                "ORDER BY active_accounts DESC, total_posts DESC LIMIT 20",
                eid=entry_id,
            )
            cooccurrences = [dict(r) for r in cooccurrence_result]

            # --- 3. Cadence par compte ---
            cadence_result = session.run(
                f"{base_match} {date_filter} "
                "WITH a, substring(p.published_at, 0, 10) AS day, count(p) AS daily_posts "
                "WITH a, count(DISTINCT day) AS active_days, avg(daily_posts) AS avg_per_day, "
                "max(daily_posts) AS max_per_day, min(daily_posts) AS min_per_day "
                "RETURN a.display_name AS account, a.platform AS platform, "
                "active_days, avg_per_day, max_per_day, min_per_day "
                "ORDER BY active_days DESC LIMIT 15",
                eid=entry_id,
            )
            cadence = [dict(r) for r in cadence_result]

            # --- 4. Évolution mensuelle ---
            monthly_result = session.run(
                f"{base_match} {date_filter} "
                "WITH substring(p.published_at, 0, 7) AS month, "
                "count(p) AS posts, count(DISTINCT a) AS accounts, "
                "avg(p.deepfake_score) AS avg_deepfake, "
                "sum(CASE WHEN p.is_synthetic THEN 1 ELSE 0 END) AS synthetic_count "
                "RETURN month, posts, accounts, avg_deepfake, synthetic_count "
                "ORDER BY month",
                eid=entry_id,
            )
            monthly = [dict(r) for r in monthly_result]

            # --- 5. Propagation : ordre d'entrée des comptes ---
            propagation_result = session.run(
                f"{base_match} {date_filter} "
                "WITH a, min(p.published_at) AS first_post, max(p.published_at) AS last_post, "
                "count(p) AS total_posts "
                "RETURN a.display_name AS account, a.platform AS platform, "
                "first_post, last_post, total_posts ORDER BY first_post",
                eid=entry_id,
            )
            propagation = [dict(r) for r in propagation_result]

            # --- 6. Compte semence (premier à publier sur la narrative) ---
            seed_account = None
            if seed_match:
                seed_result = session.run(
                    f"{seed_match} {date_filter} "
                    "WITH a, min(p.published_at) AS first_pub, count(p) AS posts "
                    "RETURN a.display_name AS account, a.platform AS platform, "
                    "first_pub, posts ORDER BY first_pub LIMIT 1",
                    eid=entry_id,
                )
                row = seed_result.single()
                if row:
                    seed_account = dict(row)

            # --- 7. Silences suspects par compte (gaps > 14 jours) ---
            silences = {}
            accounts_to_check = list({t.get("account") for t in timeline if t.get("account")})[:5]
            for acc_name in accounts_to_check:
                silence_q = session.run("""
                    MATCH (a:Account {display_name: $acc})-[:A_PUBLIÉ]->(p:Post)
                    WHERE p.published_at IS NOT NULL AND p.published_at <> ''
                    WITH p.published_at AS pub ORDER BY pub
                    WITH collect(pub) AS dates
                    UNWIND range(0, size(dates)-2) AS i
                    WITH dates[i] AS d1, dates[i+1] AS d2,
                         duration.between(
                             date(substring(dates[i],0,10)),
                             date(substring(dates[i+1],0,10))
                         ).days AS gap
                    WHERE gap > 14
                    RETURN substring(d1,0,10) AS gap_start,
                           substring(d2,0,10) AS gap_end,
                           gap AS days
                    ORDER BY gap DESC LIMIT 5
                """, acc=acc_name)
                rows = [dict(r) for r in silence_q]
                if rows:
                    silences[acc_name] = rows

            # --- 8. Corrélation deepfake × temporel ---
            # Mois où deepfake_score est significativement plus élevé
            deepfake_temporal = []
            if monthly:
                avg_df_global = sum(
                    (m.get("avg_deepfake") or 0) for m in monthly
                ) / max(1, len(monthly))
                deepfake_temporal = [
                    m for m in monthly
                    if (m.get("avg_deepfake") or 0) > avg_df_global * 1.5
                    and (m.get("avg_deepfake") or 0) > 0.05
                ]

            # --- 9. Cross-campagne : comptes actifs sur plusieurs campagnes ---
            cross_campaign = []
            if entry_type == "campaign":
                xcamp_result = session.run("""
                    MATCH (c1:Campaign)-[:COUVRE]->(n1:Narrative)<-[:APPARTIENT_À]-(p1:Post)<-[:A_PUBLIÉ]-(a:Account)
                    WHERE c1.mongo_id = $eid
                    MATCH (c2:Campaign)-[:COUVRE]->(n2:Narrative)<-[:APPARTIENT_À]-(p2:Post)<-[:A_PUBLIÉ]-(a)
                    WHERE c2.mongo_id <> $eid
                    WITH a, collect(DISTINCT c2.name) AS other_campaigns,
                         count(DISTINCT c2) AS campaign_count
                    RETURN a.display_name AS account, a.platform AS platform,
                           campaign_count, other_campaigns
                    ORDER BY campaign_count DESC LIMIT 10
                """, eid=entry_id)
                cross_campaign = [dict(r) for r in xcamp_result]
            elif entry_type == "account":
                xcamp_result = session.run("""
                    MATCH (a:Account {platform_id: $eid})-[:A_PUBLIÉ]->(p:Post)
                          -[:APPARTIENT_À]->(n:Narrative)<-[:COUVRE]-(c:Campaign)
                    WITH c, count(p) AS posts
                    RETURN c.name AS campaign, c.score AS score, posts
                    ORDER BY posts DESC LIMIT 10
                """, eid=entry_id)
                cross_campaign = [dict(r) for r in xcamp_result]

        # --- Analyses Python ---
        # Comptes robotiques
        robotic = [
            c for c in cadence
            if c.get("active_days", 0) > 20 and (c.get("avg_per_day") or 0) >= 0.8
        ]

        # Pics d'activité (mois avec 2× la moyenne)
        if monthly:
            avg_m = sum(m.get("posts", 0) for m in monthly) / len(monthly)
            peak_months = [m for m in monthly if m.get("posts", 0) > avg_m * 2]
        else:
            peak_months = []

        max_cooc = max((c.get("active_accounts", 0) for c in cooccurrences), default=0)

        return {
            "entry_id":          entry_id,
            "entry_type":        entry_type,
            "timeline":          timeline,
            "cooccurrences":     cooccurrences,
            "cadence":           cadence,
            "monthly":           monthly,
            "propagation":       propagation,
            "seed_account":      seed_account,
            "silences":          silences,
            "deepfake_temporal": deepfake_temporal,
            "cross_campaign":    cross_campaign,
            "summary": {
                "total_days_active":  len(set(t.get("day") for t in timeline)),
                "max_cooccurrence":   max_cooc,
                "cooccurrence_days":  len(cooccurrences),
                "robotic_accounts":   [r.get("account") for r in robotic],
                "peak_months":        [p.get("month") for p in peak_months],
                "first_post":         propagation[0].get("first_post", "")[:10] if propagation else "",
                "last_post":          propagation[-1].get("last_post", "")[:10] if propagation else "",
                "seed_account":       seed_account.get("account", "") if seed_account else "",
                "seed_date":          seed_account.get("first_pub", "")[:10] if seed_account else "",
                "accounts_with_silences": list(silences.keys()),
                "deepfake_spike_months":  [m.get("month") for m in deepfake_temporal],
                "cross_campaign_amplifiers": len(cross_campaign),
            },
        }

    return _safe_run("get_temporal_analysis", _run)

# ===========================================================================
# Registre des outils — utilisé par investigation_agent.py
# ===========================================================================

TOOLS = {
    "get_account_info":              get_account_info,
    "get_account_posts":             get_account_posts,
    "get_media_scores":              get_media_scores,
    "get_graph_neighbors":           get_graph_neighbors,
    "get_narrative":                 get_narrative,
    "get_campaign_signals":          get_campaign_signals,
    "get_campaign_graph":            get_campaign_graph,
    "get_temporal_analysis":         get_temporal_analysis,
    "search_accounts_by_narrative":  search_accounts_by_narrative,
}

# Descriptions des outils pour le LLM (function calling)
TOOLS_SCHEMA = [
    {
        "name": "get_account_info",
        "description": (
            "Récupère le profil complet d'un compte : stats, score bot, narratives associées, "
            "campagnes détectées, et résumé agrégé des scores deepfake sur ses posts. "
            "À appeler en premier pour tout point d'entrée de type 'compte'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "platform":  {"type": "string", "description": "instagram | tiktok | twitter | telegram"},
                "unique_id": {"type": "string", "description": "@handle ou platform_id numérique"},
            },
            "required": ["platform", "unique_id"],
        },
    },
    {
        "name": "get_account_posts",
        "description": (
            "Retourne les derniers posts d'un compte avec scores deepfake (score, prediction, divergence) "
            "et NLP (sentiment, narrative_id). Utile pour identifier les posts les plus suspects."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string", "description": "_id MongoDB du compte (obtenu via get_account_info)"},
                "limit":      {"type": "integer", "description": "Nombre de posts à retourner (défaut 20, max 50)"},
            },
            "required": ["account_id"],
        },
    },
    {
        "name": "get_media_scores",
        "description": (
            "Retourne les scores deepfake détaillés par modèle pour les médias d'un compte. "
            "Permet d'analyser la divergence inter-modèles et la réutilisation cross-plateforme."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string", "description": "_id MongoDB du compte"},
            },
            "required": ["account_id"],
        },
    },
    {
        "name": "get_graph_neighbors",
        "description": (
            "Explore le graphe Neo4j autour d'un compte : comptes voisins, hashtags partagés, "
            "médias réutilisés cross-comptes, narratives et campagnes associées. "
            "Indispensable pour détecter la coordination."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "platform_id": {"type": "string", "description": "Identifiant plateforme du compte (ex: '123456789')"},
                "depth":       {"type": "integer", "description": "Profondeur du graphe (1 ou 2, défaut 2)"},
            },
            "required": ["platform_id"],
        },
    },
    {
        "name": "get_narrative",
        "description": (
            "Récupère les détails d'une narrative : label, mots-clés, ratio de médias synthétiques, "
            "plateformes concernées, et top comptes qui la portent."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "narrative_id": {"type": "string", "description": "_id MongoDB de la narrative"},
            },
            "required": ["narrative_id"],
        },
    },
    {
        "name": "get_campaign_signals",
        "description": (
            "Retourne les signaux de coordination d'une campagne détectée : "
            "posting coordonné, réutilisation de contenu, ratio synthétique, "
            "cross-plateforme, comptes membres et confiance de la détection."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "campaign_id": {"type": "string", "description": "_id MongoDB de la campagne"},
            },
            "required": ["campaign_id"],
        },
    },
    {
        "name": "search_accounts_by_narrative",
        "description": (
            "Trouve tous les comptes qui ont posté des contenus associés à une narrative donnée. "
            "Retourne les comptes triés par nombre de posts, avec leur ratio de médias synthétiques."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "narrative_id": {"type": "string", "description": "_id MongoDB de la narrative"},
            },
            "required": ["narrative_id"],
        },
    },
]


# ===========================================================================
# CLI de test standalone
# ===========================================================================

if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Test des outils AI-FORENSICS")
    parser.add_argument("--test-account",   nargs=2, metavar=("PLATFORM", "UNIQUE_ID"),
                        help="Ex: --test-account instagram cryptocom")
    parser.add_argument("--test-posts",     metavar="ACCOUNT_ID",
                        help="Ex: --test-posts 6630a1b2c3d4e5f600000001")
    parser.add_argument("--test-media",     metavar="ACCOUNT_ID")
    parser.add_argument("--test-neighbors", metavar="PLATFORM_ID",
                        help="Ex: --test-neighbors 123456789")
    parser.add_argument("--test-narrative", metavar="NARRATIVE_ID")
    parser.add_argument("--test-campaign",  metavar="CAMPAIGN_ID")
    parser.add_argument("--test-narrative-accounts", metavar="NARRATIVE_ID")
    args = parser.parse_args()

    def _print(result):
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    if args.test_account:
        platform, uid = args.test_account
        print(f"\n=== get_account_info({platform}, {uid}) ===")
        _print(get_account_info(platform, uid))

    if args.test_posts:
        print(f"\n=== get_account_posts({args.test_posts}) ===")
        _print(get_account_posts(args.test_posts))

    if args.test_media:
        print(f"\n=== get_media_scores({args.test_media}) ===")
        _print(get_media_scores(args.test_media))

    if args.test_neighbors:
        print(f"\n=== get_graph_neighbors({args.test_neighbors}) ===")
        _print(get_graph_neighbors(args.test_neighbors))

    if args.test_narrative:
        print(f"\n=== get_narrative({args.test_narrative}) ===")
        _print(get_narrative(args.test_narrative))

    if args.test_campaign:
        print(f"\n=== get_campaign_signals({args.test_campaign}) ===")
        _print(get_campaign_signals(args.test_campaign))

    if args.test_narrative_accounts:
        print(f"\n=== search_accounts_by_narrative({args.test_narrative_accounts}) ===")
        _print(search_accounts_by_narrative(args.test_narrative_accounts))

    if not any(vars(args).values()):
        parser.print_help()
