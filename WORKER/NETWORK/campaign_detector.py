"""campaign_detector.py
=======================

Détection automatique de campagnes d'influence.
Analyse les signaux disponibles en base et crée les documents
dans la collection `campaigns`.

Signaux analysés :
    1. Contenu dupliqué massif     → posts quasi-identiques (is_duplicate_of)
    2. Narratif cross-plateforme   → même narratif sur ≥2 plateformes
    3. Burst temporel              → N posts du même narratif en peu de temps
    4. Médias synthétiques         → ratio deepfake élevé dans un narratif
    5. Comptes coordonnés (GDS)    → clusters Louvain de ≥3 comptes suspects
    6. Amplificateurs (GDS)        → comptes PageRank élevé dans un narratif

Score de confiance [0-1] calculé selon le nombre de signaux convergents.

Lancement :
    python campaign_detector.py
    python campaign_detector.py --dry-run        # sans écriture MongoDB
    python campaign_detector.py --min-score 0.3
    python campaign_detector.py --skip-gds       # ignore les signaux GDS (plus rapide)

Dépendances :
    pip install pymongo python-dotenv neo4j
"""

from __future__ import annotations

import argparse
import configparser
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
    load_dotenv(_HERE / ".env")
except ImportError:
    pass

# Résolution schema.py — ordre de priorité :
#   1. ~/AI-FORENSICS/SCHEMA/  (../../SCHEMA/ depuis WORKER/NETWORK/)
#   2. Dossier parent du worker (WORKER/)
#   3. Dossier courant          (WORKER/NETWORK/)
_SCHEMA_CANDIDATES = [
    _HERE.parent.parent / "SCHEMA",
    _HERE.parent,
    _HERE,
]
for _schema_dir in _SCHEMA_CANDIDATES:
    if (_schema_dir / "schema.py").exists():
        sys.path.insert(0, str(_schema_dir))
        break
else:
    for _p in [_HERE, _HERE.parent]:
        sys.path.insert(0, str(_p))

from schema import get_db, new_campaign

logger = logging.getLogger("campaign_detector")


# ---------------------------------------------------------------------------
# Configuration logging
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(ch)
    logging.getLogger("pymongo").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Détecteur de campagnes
# ---------------------------------------------------------------------------

def _cfg_int(section, key: str, fallback: int = 0) -> "int | None":
    """Lit une valeur entière depuis une section configparser.
    Retourne None si la valeur est absente ou vide (champ vidé pour sécurité).
    """
    val = section.get(key, "").strip()
    try:
        return int(val) or None
    except ValueError:
        return fallback or None


class CampaignDetector:

    def __init__(
        self,
        db,
        neo4j_client=None,
        dry_run: bool = False,
        skip_gds: bool = False,
    ) -> None:
        self.db           = db
        self.neo4j        = neo4j_client   # None si GDS non disponible ou --skip-gds
        self.dry_run      = dry_run
        self.skip_gds     = skip_gds or (neo4j_client is None)

        if self.skip_gds:
            logger.info("Signaux GDS désactivés (--skip-gds ou neo4j_client absent)")

    # ------------------------------------------------------------------
    # Orchestrateur principal
    # ------------------------------------------------------------------


    def run(self, min_score: float = 0.30) -> list[dict]:
        """
        Analyse tous les signaux et retourne les campagnes détectées.
        Seules les campagnes avec score >= min_score sont conservées.
        """
        logger.info("=== Détection de campagnes démarrée ===")

        candidates = {}   # narrative_id (str) → dict de signaux

        # --- Signaux MongoDB (1-4) ---
        logger.info("Signal 1 — Contenu dupliqué…")
        self._signal_duplicates(candidates)

        logger.info("Signal 2 — Cross-plateforme…")
        self._signal_cross_platform(candidates)

        logger.info("Signal 3 — Burst temporel…")
        self._signal_burst(candidates)

        logger.info("Signal 4 — Médias synthétiques…")
        self._signal_synthetic(candidates)

        # --- Signaux GDS (5-6) — optionnels ---
        if not self.skip_gds:
            logger.info("Signal 5 — Comptes coordonnés (Louvain)…")
            louvain_results = self._run_louvain_safe()
            self._signal_coordinated_accounts(candidates, louvain_results)

            logger.info("Signal 6 — Amplificateurs (PageRank)…")
            pagerank_results = self._run_pagerank_safe()
            self._signal_amplifiers(candidates, pagerank_results)
        else:
            louvain_results  = []
            pagerank_results = []

        # --- Calcul des scores ---
        logger.info("Calcul des scores de confiance…")
        campaigns = self._score_candidates(candidates, min_score)

        # --- Enrichissement avec les comptes GDS ---
        if louvain_results and campaigns:
            self._enrich_with_community_accounts(campaigns, louvain_results)

        # --- Résumé ---
        logger.info("")
        logger.info("=" * 60)
        logger.info("CAMPAGNES DÉTECTÉES (score >= %.2f)", min_score)
        logger.info("=" * 60)
        for c in campaigns:
            active_signals = [
                k for k, v in c["signals"].items()
                if v and v is not False and v is not None and v != 0
            ]
            logger.info(
                "  [%.2f] %s | signaux: %s",
                c["review"]["confidence"],
                c["name"],
                ", ".join(active_signals),
            )
        logger.info("=" * 60)
        logger.info("%d campagne(s) détectée(s)", len(campaigns))

        if not self.dry_run and campaigns:
            self._save_campaigns(campaigns)
            if self.neo4j:
                self._save_campaigns_to_neo4j(campaigns)

        logger.info("=== Détection terminée ===")
        return campaigns

    # ------------------------------------------------------------------
    # Signal 1 — Contenu dupliqué massif
    # ------------------------------------------------------------------

    def _signal_duplicates(self, candidates: dict) -> None:
        """
        Détecte les narratifs avec beaucoup de posts dupliqués.
        Seuil : > 10% des posts du narratif sont des doublons.
        """
        pipeline = [
            {"$match": {"nlp.is_duplicate_of": {"$ne": None}}},
            {"$group": {
                "_id":   "$nlp.narrative_id",
                "count": {"$sum": 1},
            }},
            {"$match": {"_id": {"$ne": None}, "count": {"$gte": 3}}},
        ]

        for row in self.db.posts.aggregate(pipeline):
            narr_id = row["_id"]
            nb_dups = row["count"]

            total = self.db.posts.count_documents({"nlp.narrative_id": narr_id})
            ratio = nb_dups / total if total > 0 else 0

            if ratio >= 0.10:
                key = str(narr_id)
                if key not in candidates:
                    candidates[key] = self._empty_candidate(narr_id)
                candidates[key]["signals"]["content_reuse"]   = True
                candidates[key]["signals"]["duplicate_count"] = nb_dups
                candidates[key]["signals"]["duplicate_ratio"] = round(ratio, 3)
                candidates[key]["signal_count"] += 1
                logger.debug(
                    "  Narratif %s : %d doublons (%.1f%%)", key, nb_dups, ratio * 100
                )

    # ------------------------------------------------------------------
    # Signal 2 — Cross-plateforme
    # ------------------------------------------------------------------

    def _signal_cross_platform(self, candidates: dict) -> None:
        """Détecte les narratifs présents sur ≥ 2 plateformes."""
        pipeline = [
            {"$match": {"nlp.narrative_id": {"$ne": None}}},
            {"$group": {
                "_id":       "$nlp.narrative_id",
                "platforms": {"$addToSet": "$platform"},
                "count":     {"$sum": 1},
            }},
            {"$match": {"platforms.1": {"$exists": True}}},
        ]

        for row in self.db.posts.aggregate(pipeline):
            narr_id   = row["_id"]
            platforms = row["platforms"]
            key = str(narr_id)

            if key not in candidates:
                candidates[key] = self._empty_candidate(narr_id)

            candidates[key]["signals"]["cross_platform"] = True
            candidates[key]["platforms"]                 = platforms
            candidates[key]["signal_count"]             += 1
            logger.debug("  Narratif %s : plateformes %s", key, platforms)

    # ------------------------------------------------------------------
    # Signal 3 — Burst temporel
    # ------------------------------------------------------------------

    def _signal_burst(self, candidates: dict) -> None:
        """
        Détecte les narratifs avec un pic de publication anormal.
        Seuil : ≥ 20 posts dans une fenêtre de 24h.
        """
        pipeline = [
            {"$match": {
                "nlp.narrative_id":      {"$ne": None},
                "context.published_at":  {"$ne": None},
            }},
            {"$project": {
                "narrative_id": "$nlp.narrative_id",
                "day": {"$dateToString": {
                    "format": "%Y-%m-%d",
                    "date":   "$context.published_at",
                }},
            }},
            {"$group": {
                "_id":   {"narrative": "$narrative_id", "day": "$day"},
                "count": {"$sum": 1},
            }},
            {"$match": {"count": {"$gte": 20}}},
            {"$sort": {"count": -1}},
        ]

        for row in self.db.posts.aggregate(pipeline):
            narr_id = row["_id"]["narrative"]
            day     = row["_id"]["day"]
            count   = row["count"]
            key = str(narr_id)

            if key not in candidates:
                candidates[key] = self._empty_candidate(narr_id)

            candidates[key]["signals"]["coordinated_posting"] = True
            candidates[key]["signals"]["burst_day"]           = day
            candidates[key]["signals"]["burst_count"]         = count
            candidates[key]["signal_count"]                  += 1
            logger.debug(
                "  Narratif %s : %d posts le %s (burst)", key, count, day
            )

    # ------------------------------------------------------------------
    # Signal 4 — Médias synthétiques
    # ------------------------------------------------------------------

    def _signal_synthetic(self, candidates: dict) -> None:
        """
        Détecte les narratifs avec un ratio élevé de médias synthétiques.
        Seuil : > 20% de posts avec prediction=synthetic.
        """
        pipeline = [
            {"$match": {"nlp.narrative_id": {"$ne": None}}},
            {"$group": {
                "_id":       "$nlp.narrative_id",
                "total":     {"$sum": 1},
                "synthetic": {"$sum": {
                    "$cond": [
                        {"$eq": ["$deepfake.prediction", "synthetic"]}, 1, 0
                    ]
                }},
            }},
            {"$match": {"synthetic": {"$gte": 1}}},
        ]

        for row in self.db.posts.aggregate(pipeline):
            narr_id   = row["_id"]
            total     = row["total"]
            synthetic = row["synthetic"]
            ratio     = synthetic / total if total > 0 else 0

            if ratio >= 0.20:
                key = str(narr_id)
                if key not in candidates:
                    candidates[key] = self._empty_candidate(narr_id)

                candidates[key]["signals"]["synthetic_media_ratio"] = round(ratio, 3)
                candidates[key]["signal_count"]                    += 1
                logger.debug(
                    "  Narratif %s : %.1f%% médias synthétiques", key, ratio * 100
                )

    # ------------------------------------------------------------------
    # Signal 5 — Comptes coordonnés (Louvain / GDS)
    # ------------------------------------------------------------------

    def _run_louvain_safe(self) -> list[dict]:
        """Lance Louvain et retourne les résultats sans planter si GDS échoue."""
        if not self.neo4j:
            return []
        try:
            return self.neo4j.run_louvain()
        except Exception as exc:
            logger.warning("Louvain échoué, signal 5 ignoré : %s", exc)
            return []

    def _signal_coordinated_accounts(
        self,
        candidates: dict,
        louvain_results: list[dict],
    ) -> None:
        """
        Détecte les communautés de comptes suspects dans les narratifs.

        Logique :
          1. Récupère les comptes impliqués dans chaque narratif (via les posts)
          2. Croise avec les community_id Louvain
          3. Si une communauté de ≥ 3 comptes est concentrée dans un narratif → signal

        Seuil : ≥ 3 comptes du même cluster Louvain dans un même narratif.
        """
        if not louvain_results:
            logger.info("  Aucun résultat Louvain — signal 5 ignoré")
            return

        # Construire un index mongo_id → community_id depuis les résultats Louvain
        # (les community_id sont aussi écrits sur les nœuds Neo4j par run_louvain)
        account_community: dict[str, int] = {
            r["mongo_id"]: r["communityId"]
            for r in louvain_results
            if r.get("mongo_id") is not None
        }

        if not account_community:
            return

        # Pour chaque narratif, récupérer les comptes qui ont posté dedans
        pipeline = [
            {"$match": {"nlp.narrative_id": {"$ne": None}}},
            {"$group": {
                "_id":        "$nlp.narrative_id",
                "account_ids": {"$addToSet": "$account_id"},
            }},
        ]

        for row in self.db.posts.aggregate(pipeline):
            narr_id     = row["_id"]
            account_ids = [str(a) for a in (row["account_ids"] or []) if a]

            if not account_ids:
                continue

            # Compter combien de comptes de chaque communauté Louvain sont présents
            community_counts: dict[int, int] = defaultdict(int)
            for acc_id in account_ids:
                cid = account_community.get(acc_id)
                if cid is not None:
                    community_counts[cid] += 1

            # Signal si une communauté a ≥ 3 comptes dans ce narratif
            suspicious_communities = {
                cid: count
                for cid, count in community_counts.items()
                if count >= 3
            }

            if suspicious_communities:
                key = str(narr_id)
                if key not in candidates:
                    candidates[key] = self._empty_candidate(narr_id)

                # Stocker la communauté la plus grande
                top_community_id    = max(suspicious_communities, key=suspicious_communities.get)
                top_community_count = suspicious_communities[top_community_id]

                candidates[key]["signals"]["coordinated_accounts"]      = True
                candidates[key]["signals"]["top_community_id"]          = top_community_id
                candidates[key]["signals"]["top_community_size"]        = top_community_count
                candidates[key]["signals"]["suspicious_community_count"] = len(suspicious_communities)
                candidates[key]["signal_count"] += 1

                logger.debug(
                    "  Narratif %s : communauté %d (%d comptes coordonnés)",
                    key, top_community_id, top_community_count,
                )

        total_flagged = sum(
            1 for c in candidates.values()
            if c["signals"].get("coordinated_accounts")
        )
        logger.info(
            "  Signal 5 : %d narratif(s) avec comptes coordonnés détectés", total_flagged
        )

    # ------------------------------------------------------------------
    # Signal 6 — Amplificateurs (PageRank / GDS)
    # ------------------------------------------------------------------

    def _run_pagerank_safe(self) -> list[dict]:
        """Lance PageRank et retourne les résultats sans planter si GDS échoue."""
        if not self.neo4j:
            return []
        try:
            return self.neo4j.run_pagerank(top_n=100)
        except Exception as exc:
            logger.warning("PageRank échoué, signal 6 ignoré : %s", exc)
            return []

    def _signal_amplifiers(
        self,
        candidates: dict,
        pagerank_results: list[dict],
    ) -> None:
        """
        Détecte les narratifs portés par des comptes à fort PageRank.

        Logique :
          1. Identifie les comptes dans le top PageRank
          2. Vérifie si ces comptes ont posté dans chaque narratif
          3. Si ≥ 2 amplificateurs sont dans le même narratif → signal

        Seuil : ≥ 2 comptes dans le top 100 PageRank présents dans un narratif.
        """
        if not pagerank_results:
            logger.info("  Aucun résultat PageRank — signal 6 ignoré")
            return

        # Index mongo_id → pagerank_score pour les comptes amplificateurs
        amplifier_ids = {
            r["mongo_id"]: r["pagerank_score"]
            for r in pagerank_results
            if r.get("mongo_id") and r.get("pagerank_score", 0) > 0
        }

        if not amplifier_ids:
            return

        # Pour chaque narratif, vérifier la présence d'amplificateurs
        pipeline = [
            {"$match": {"nlp.narrative_id": {"$ne": None}}},
            {"$group": {
                "_id":        "$nlp.narrative_id",
                "account_ids": {"$addToSet": "$account_id"},
            }},
        ]

        for row in self.db.posts.aggregate(pipeline):
            narr_id     = row["_id"]
            account_ids = [str(a) for a in (row["account_ids"] or []) if a]

            amplifiers_in_narrative = [
                {"mongo_id": acc_id, "score": amplifier_ids[acc_id]}
                for acc_id in account_ids
                if acc_id in amplifier_ids
            ]

            if len(amplifiers_in_narrative) >= 2:
                key = str(narr_id)
                if key not in candidates:
                    candidates[key] = self._empty_candidate(narr_id)

                # Score max des amplificateurs dans ce narratif
                max_score = max(a["score"] for a in amplifiers_in_narrative)

                candidates[key]["signals"]["key_amplifiers"]       = True
                candidates[key]["signals"]["amplifier_count"]      = len(amplifiers_in_narrative)
                candidates[key]["signals"]["max_amplifier_score"]  = round(max_score, 4)
                candidates[key]["signal_count"] += 1

                logger.debug(
                    "  Narratif %s : %d amplificateur(s) (pagerank max=%.4f)",
                    key, len(amplifiers_in_narrative), max_score,
                )

        total_flagged = sum(
            1 for c in candidates.values()
            if c["signals"].get("key_amplifiers")
        )
        logger.info(
            "  Signal 6 : %d narratif(s) avec amplificateurs clés", total_flagged
        )

    # ------------------------------------------------------------------
    # Calcul des scores
    # ------------------------------------------------------------------

    def _score_candidates(
        self,
        candidates: dict,
        min_score: float,
    ) -> list[dict]:
        """
        Calcule un score de confiance [0-1] pour chaque candidat.

        Pondération (total max = 1.0) :
          content_reuse         : 0.25  (signal fort mais couvert par NLP aussi)
          cross_platform        : 0.20
          coordinated_posting   : 0.20
          synthetic_media       : 0.10
          coordinated_accounts  : 0.15  (GDS Louvain — signal structurel fort)
          key_amplifiers        : 0.10  (GDS PageRank — signal de diffusion)

        Bonus :
          +0.05 si duplicate_ratio > 0.30
          +0.05 si top_community_size > 10
        """
        campaigns = []

        for key, candidate in candidates.items():
            sig = candidate["signals"]

            score = 0.0

            if sig.get("content_reuse"):
                score += 0.25
                if sig.get("duplicate_ratio", 0) > 0.30:
                    score += 0.05   # bonus ratio élevé

            if sig.get("cross_platform"):
                score += 0.20

            if sig.get("coordinated_posting"):
                score += 0.20

            if sig.get("synthetic_media_ratio") and sig["synthetic_media_ratio"] > 0.20:
                score += 0.10

            if sig.get("coordinated_accounts"):
                score += 0.15
                if sig.get("top_community_size", 0) > 10:
                    score += 0.05   # bonus grande communauté

            if sig.get("key_amplifiers"):
                score += 0.10

            score = min(round(score, 2), 1.0)

            if score < min_score:
                continue

            # Récupérer les infos du narratif
            narr = self.db.narratives.find_one({"_id": candidate["narrative_id"]})
            if not narr:
                continue

            name = f"Campagne — {narr.get('label', key)}"
            platforms = (
                candidate.get("platforms")
                or narr.get("stats", {}).get("platforms", [])
            )

            doc = new_campaign(name=name, platforms=platforms)
            doc["narrative_ids"] = [candidate["narrative_id"]]

            # Signaux MongoDB (1-4)
            doc["signals"]["content_reuse"]        = sig.get("content_reuse", False)
            doc["signals"]["cross_platform"]        = sig.get("cross_platform", False)
            doc["signals"]["coordinated_posting"]   = sig.get("coordinated_posting", False)
            doc["signals"]["synthetic_media_ratio"] = sig.get("synthetic_media_ratio")
            doc["signals"]["narrative_count"]       = 1

            # Signaux GDS (5-6) — ajoutés dynamiquement (pas dans le schéma de base)
            doc["signals"]["coordinated_accounts"]       = sig.get("coordinated_accounts", False)
            doc["signals"]["top_community_id"]           = sig.get("top_community_id")
            doc["signals"]["top_community_size"]         = sig.get("top_community_size")
            doc["signals"]["suspicious_community_count"] = sig.get("suspicious_community_count")
            doc["signals"]["key_amplifiers"]             = sig.get("key_amplifiers", False)
            doc["signals"]["amplifier_count"]            = sig.get("amplifier_count")
            doc["signals"]["max_amplifier_score"]        = sig.get("max_amplifier_score")

            doc["review"]["confidence"] = score

            # Date range
            first = self.db.posts.find_one(
                {"nlp.narrative_id": candidate["narrative_id"]},
                sort=[("context.published_at", 1)],
            )
            last = self.db.posts.find_one(
                {"nlp.narrative_id": candidate["narrative_id"]},
                sort=[("context.published_at", -1)],
            )
            if first:
                doc["date_range"]["start"] = (
                    first.get("context", {}).get("published_at") or doc["date_range"]["start"]
                )
            if last:
                doc["date_range"]["end"] = last.get("context", {}).get("published_at")

            campaigns.append(doc)

        campaigns.sort(key=lambda x: x["review"]["confidence"], reverse=True)
        return campaigns

    # ------------------------------------------------------------------
    # Enrichissement avec les comptes des communautés
    # ------------------------------------------------------------------

    def _enrich_with_community_accounts(
        self,
        campaigns: list[dict],
        louvain_results: list[dict],
    ) -> None:
        """
        Ajoute les account_ids des comptes appartenant à la communauté
        suspecte identifiée dans chaque campagne.

        Récupère les mongo_id depuis MongoDB (les Louvain results ont
        les mongo_id des nœuds Account Neo4j).
        """
        # Index community_id → liste de mongo_id
        community_members: dict[int, list[str]] = defaultdict(list)
        for r in louvain_results:
            if r.get("mongo_id") and r.get("communityId") is not None:
                community_members[r["communityId"]].append(r["mongo_id"])

        for campaign in campaigns:
            top_cid = campaign["signals"].get("top_community_id")
            if top_cid is None:
                continue

            member_mongo_ids = community_members.get(top_cid, [])
            if not member_mongo_ids:
                continue

            # Récupérer les ObjectId MongoDB depuis les mongo_id (strings)
            from bson import ObjectId
            account_object_ids = []
            for mid in member_mongo_ids:
                try:
                    account_object_ids.append(ObjectId(mid))
                except Exception:
                    pass

            if account_object_ids:
                campaign["account_ids"] = account_object_ids
                logger.debug(
                    "Campagne '%s' enrichie avec %d comptes (communauté %d)",
                    campaign["name"], len(account_object_ids), top_cid,
                )

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------

    def _save_campaigns(self, campaigns: list[dict]) -> None:
        """Upsert des campagnes dans MongoDB (par narrative_id)."""
        saved = 0
        for doc in campaigns:
            if not doc["narrative_ids"]:
                continue
            self.db.campaigns.update_one(
                {"narrative_ids": doc["narrative_ids"][0]},
                {"$set": {k: v for k, v in doc.items() if k != "_id"}},
                upsert=True,
            )
            saved += 1
        logger.info("%d campagne(s) sauvegardée(s) dans MongoDB", saved)

    def _save_campaigns_to_neo4j(self, campaigns: list[dict]) -> None:
        """
        Crée ou met à jour les nœuds :Campaign dans Neo4j.

        Relations créées :
            (:Campaign)-[:COUVRE]->(:Narrative)
            (:Campaign)-[:IMPLIQUE]->(:Account)
            (:Account)-[:PARTICIPE_À]->(:Campaign)
        """
        if not self.neo4j:
            return

        for doc in campaigns:
            name       = doc.get("name", "")
            score      = doc.get("review", {}).get("confidence", 0.0)
            signals    = doc.get("signals", {})
            platforms  = doc.get("platforms", [])
            narr_ids   = doc.get("narrative_ids", [])
            acc_ids    = doc.get("account_ids", [])

            # Récupérer le mongo_id de la campagne depuis MongoDB
            campaign_doc = self.db.campaigns.find_one(
                {"narrative_ids": narr_ids[0]} if narr_ids else {"name": name},
                {"_id": 1},
            )
            if not campaign_doc:
                continue
            campaign_mongo_id = str(campaign_doc["_id"])

            # Upsert nœud :Campaign
            active_signals = [
                k for k, v in signals.items()
                if v and v is not False and v is not None and v != 0
            ]
            self.neo4j.upsert_campaign({
                "mongo_id":      campaign_mongo_id,
                "name":          name,
                "score":         score,
                "platforms":     platforms,
                "signals":       active_signals,
                "signal_count":  len(active_signals),
            })

            # Relations vers Narrative(s)
            for narr_id in narr_ids:
                try:
                    self.neo4j.link_campaign_narrative(
                        campaign_mongo_id, str(narr_id)
                    )
                except Exception as exc:
                    logger.debug("link_campaign_narrative %s : %s", narr_id, exc)

            # Relations vers Account(s)
            for acc_id in acc_ids:
                try:
                    self.neo4j.link_campaign_account(
                        campaign_mongo_id, str(acc_id)
                    )
                except Exception as exc:
                    logger.debug("link_campaign_account %s : %s", acc_id, exc)

        logger.info(
            "%d campagne(s) synchronisées → Neo4j", len(campaigns)
        )

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _empty_candidate(self, narr_id) -> dict:
        return {
            "narrative_id": narr_id,
            "platforms":    [],
            "signal_count": 0,
            "signals": {
                # Signaux MongoDB
                "content_reuse":          False,
                "cross_platform":         False,
                "coordinated_posting":    False,
                "synthetic_media_ratio":  None,
                "duplicate_count":        0,
                "duplicate_ratio":        0.0,
                "burst_day":              None,
                "burst_count":            0,
                # Signaux GDS
                "coordinated_accounts":        False,
                "top_community_id":            None,
                "top_community_size":          0,
                "suspicious_community_count":  0,
                "key_amplifiers":              False,
                "amplifier_count":             0,
                "max_amplifier_score":         None,
            },
        }


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Détection automatique de campagnes d'influence"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=_HERE / "network_pipeline.cfg",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.30,
        help="Score de confiance minimum [0-1] (défaut : 0.30)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Afficher les campagnes sans écrire dans MongoDB",
    )
    parser.add_argument(
        "--skip-gds",
        action="store_true",
        help="Ignorer les signaux GDS (Louvain + PageRank) — plus rapide",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)

    cfg = configparser.ConfigParser()
    if Path(args.config).exists():
        cfg.read(args.config, encoding="utf-8")

    # MongoDB
    m  = cfg["mongodb"] if "mongodb" in cfg else {}
    db = get_db(
        db_name  = m.get("db")       or None,
        host     = m.get("host")     or None,
        port     = _cfg_int(m, "port"),
        user     = m.get("user")     or None,
        password = m.get("password") or None,
        auth_db  = m.get("auth_db")  or None,
    )
    logger.info("MongoDB connecté : %s", db.name)

    # Neo4j (optionnel — si absent, GDS est désactivé automatiquement)
    neo4j_client = None
    if not args.skip_gds and "neo4j" in cfg:
        try:
            from neo4j_client import Neo4jClient
            n = cfg["neo4j"]
            neo4j_client = Neo4jClient(
                uri      = n.get("uri",      "bolt://localhost:7687"),
                user     = n.get("user",     "neo4j"),
                password = n.get("password", "influence2026!"),
            )
        except Exception as exc:
            logger.warning("Neo4j non disponible — signaux GDS désactivés : %s", exc)

    detector = CampaignDetector(
        db           = db,
        neo4j_client = neo4j_client,
        dry_run      = args.dry_run,
        skip_gds     = args.skip_gds,
    )

    try:
        detector.run(min_score=args.min_score)
    finally:
        if neo4j_client:
            neo4j_client.close()
