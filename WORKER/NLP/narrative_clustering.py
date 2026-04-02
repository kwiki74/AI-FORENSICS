"""narrative_clustering.py
==========================

Job batch de clustering narratif.
Charge tous les embeddings depuis MongoDB, applique HDBSCAN,
et crée/met à jour les documents dans la collection `narratives`.

Ce script est indépendant du worker NLP — il tourne périodiquement
(cron, systemd timer, ou manuellement) une fois que suffisamment
de posts ont été traités par nlp_worker.py.

Flux :
    MongoDB posts (nlp.embedding != null)
        → chargement embeddings en RAM (float16 → float32 pour calcul)
        → UMAP réduction dimensionnelle (768→50 pour HDBSCAN)
        → HDBSCAN clustering
        → extraction keywords par cluster (TF-IDF)
        → upsert collection narratives
        → mise à jour posts.nlp.narrative_id

⚠ Limitation (voir CR) :
    Charge TOUS les embeddings en RAM.
    Limite pratique : ~300 000 documents (float16, 384d, ~220 Mo).
    Au-delà → migration vers Qdrant recommandée.

Lancement :
    python narrative_clustering.py
    python narrative_clustering.py --dry-run    # sans écriture MongoDB
    python narrative_clustering.py --min-cluster-size 5
    python narrative_clustering.py --config nlp_pipeline.cfg

Dépendances (conda env nlp_pipeline) :
    pip install pymongo python-dotenv numpy scikit-learn hdbscan umap-learn
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

import numpy as np

_HERE = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
    load_dotenv(_HERE / ".env")
except ImportError:
    pass

# Résolution schema.py — ordre de priorité :
#   1. ~/AI-FORENSICS/SCHEMA/  (../../SCHEMA/ depuis WORKER/NLP/)
#   2. Dossier parent du worker (WORKER/)
#   3. Dossier courant          (WORKER/NLP/)
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

from schema import get_db, new_narrative

logger = logging.getLogger("narrative_clustering")


# ---------------------------------------------------------------------------
# Configuration logging
# ---------------------------------------------------------------------------


def _cfg_int(section, key: str, fallback: int = 0) -> int | None:
    """Lit une valeur entière depuis une section configparser.
    Retourne None si la valeur est absente ou vide (champ vidé pour sécurité).
    """
    val = section.get(key, "").strip()
    try:
        return int(val) or None
    except ValueError:
        return fallback or None

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
    # Silencer les libs tierces
    for noisy in ("pymongo", "urllib3", "numba", "umap"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Chargement des embeddings depuis MongoDB
# ---------------------------------------------------------------------------

def load_embeddings(db, collection: str = "posts") -> tuple[list, np.ndarray, list[str]]:
    """
    Charge tous les embeddings depuis MongoDB.

    Returns:
        doc_ids   : liste des ObjectId dans le même ordre que les vecteurs
        matrix    : numpy array float32, shape (N, 384)
        texts     : liste des textes (pour TF-IDF)
    """
    logger.info("Chargement des embeddings depuis '%s'…", collection)

    cursor = db[collection].find(
        {"nlp.embedding": {"$ne": None}},
        {"nlp.embedding": 1, "text.content": 1, "nlp.embedding_model": 1},
    )

    doc_ids = []
    vectors = []
    texts   = []

    for doc in cursor:
        emb_bytes = doc.get("nlp", {}).get("embedding")
        if not emb_bytes:
            continue
        try:
            vec = np.frombuffer(emb_bytes, dtype=np.float16).astype(np.float32)
            if vec.shape[0] != 384:
                continue
            doc_ids.append(doc["_id"])
            vectors.append(vec)
            texts.append(doc.get("text", {}).get("content", "") or "")
        except Exception as exc:
            logger.debug("Erreur désérialisation %s : %s", doc["_id"], exc)
            continue

    if not vectors:
        logger.warning("Aucun embedding trouvé dans '%s'", collection)
        return [], np.array([]), []

    matrix = np.vstack(vectors)
    logger.info(
        "%d embeddings chargés (shape=%s, RAM≈%.1f Mo)",
        len(doc_ids), matrix.shape,
        matrix.nbytes / (1024 * 1024),
    )
    return doc_ids, matrix, texts


# ---------------------------------------------------------------------------
# Réduction dimensionnelle UMAP
# ---------------------------------------------------------------------------

def reduce_dimensions(matrix: np.ndarray, n_components: int = 50) -> np.ndarray:
    """
    Réduit les embeddings de 384d à n_components dimensions via UMAP.
    HDBSCAN fonctionne mieux sur des dimensions réduites (malédiction de la dimensionnalité).
    """
    import umap

    logger.info("Réduction UMAP %dd → %dd…", matrix.shape[1], n_components)
    reducer = umap.UMAP(
        n_components   = n_components,
        n_neighbors    = 15,
        min_dist       = 0.1,
        metric         = "cosine",
        random_state   = 42,
        verbose        = False,
    )
    reduced = reducer.fit_transform(matrix)
    logger.info("UMAP terminé → shape=%s", reduced.shape)
    return reduced


# ---------------------------------------------------------------------------
# Clustering HDBSCAN
# ---------------------------------------------------------------------------

def cluster(reduced: np.ndarray, min_cluster_size: int = 5) -> np.ndarray:
    """
    Applique HDBSCAN sur les embeddings réduits.

    Label -1 = outlier (pas de cluster assigné).
    """
    import hdbscan

    logger.info("Clustering HDBSCAN (min_cluster_size=%d)…", min_cluster_size)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size = min_cluster_size,
        min_samples      = 3,
        metric           = "euclidean",
        cluster_selection_method = "eom",
    )
    labels = clusterer.fit_predict(reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = int((labels == -1).sum())
    logger.info(
        "HDBSCAN terminé : %d clusters, %d outliers (%.1f%%)",
        n_clusters, n_outliers, 100 * n_outliers / len(labels),
    )
    return labels


# ---------------------------------------------------------------------------
# Extraction de keywords par cluster (TF-IDF)
# ---------------------------------------------------------------------------

def extract_keywords(
    texts_by_cluster: dict[int, list[str]],
    n_keywords: int = 8,
) -> dict[int, list[str]]:
    """
    Extrait les mots-clés représentatifs de chaque cluster via TF-IDF.
    Fonctionne en français et en anglais.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    keywords = {}

    # Stop words combinés FR + EN
    stop_words = [
        # FR
        "le","la","les","de","du","des","un","une","en","et","est","à",
        "je","tu","il","elle","nous","vous","ils","elles","ce","se","sa",
        "son","ses","mon","ma","mes","ton","ta","tes","sur","dans","par",
        "pas","plus","que","qui","pour","avec","au","aux","ou","si","ne",
        "on","lui","y","tout","mais","donc","or","ni","car","très","bien",
        # EN
        "the","a","an","is","are","was","were","be","been","have","has",
        "do","does","did","will","would","could","should","may","might",
        "i","you","he","she","we","they","it","this","that","these","those",
        "in","on","at","to","for","of","with","by","from","up","about",
        "not","no","but","and","or","so","if","as","than","then","just",
    ]

    for cluster_id, texts in texts_by_cluster.items():
        if not texts:
            keywords[cluster_id] = []
            continue
        try:
            corpus = [t for t in texts if len(t.strip()) > 5]
            if not corpus:
                keywords[cluster_id] = []
                continue

            vec = TfidfVectorizer(
                max_features = 200,
                stop_words   = stop_words,
                ngram_range  = (1, 2),
                min_df       = 2,
            )
            tfidf = vec.fit_transform(corpus)
            scores = np.asarray(tfidf.mean(axis=0)).flatten()
            top_indices = scores.argsort()[-n_keywords:][::-1]
            feature_names = vec.get_feature_names_out()
            keywords[cluster_id] = [feature_names[i] for i in top_indices]

        except Exception as exc:
            logger.debug("Erreur TF-IDF cluster %d : %s", cluster_id, exc)
            keywords[cluster_id] = []

    return keywords


# ---------------------------------------------------------------------------
# Job principal
# ---------------------------------------------------------------------------

def run_clustering(
    db,
    min_cluster_size: int = 5,
    umap_components:  int = 50,
    dry_run:          bool = False,
    embedding_model:  str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> dict:
    """
    Orchestre le clustering complet et retourne un résumé des résultats.
    """
    now = datetime.now(timezone.utc)

    # 1. Chargement
    doc_ids, matrix, texts = load_embeddings(db, collection="posts")
    if len(doc_ids) == 0:
        logger.warning("Aucun document à clustériser.")
        return {}

    # 2. Réduction UMAP (sauf si très peu de docs)
    if len(doc_ids) >= 20:
        n_comp = min(umap_components, len(doc_ids) - 2)
        reduced = reduce_dimensions(matrix, n_components=n_comp)
    else:
        logger.info("Peu de documents (%d) — UMAP ignoré", len(doc_ids))
        reduced = matrix

    # 3. Clustering HDBSCAN
    labels = cluster(reduced, min_cluster_size=min_cluster_size)

    # 4. Groupement par cluster
    cluster_docs:  dict[int, list] = defaultdict(list)
    cluster_texts: dict[int, list] = defaultdict(list)

    for doc_id, label, text in zip(doc_ids, labels, texts):
        if label == -1:
            continue
        cluster_docs[label].append(doc_id)
        cluster_texts[label].append(text)

    n_clusters = len(cluster_docs)
    logger.info("%d clusters non-outlier à traiter", n_clusters)

    if n_clusters == 0:
        logger.warning("Aucun cluster trouvé. Essaie --min-cluster-size 3")
        return {}

    # 5. Extraction keywords
    logger.info("Extraction des keywords (TF-IDF)…")
    kw_by_cluster = extract_keywords(cluster_texts, n_keywords=8)

    # 6. Calcul centroïdes
    centroids = {}
    for label, doc_list in cluster_docs.items():
        indices = [i for i, l in enumerate(labels) if l == label]
        centroid = matrix[indices].mean(axis=0).astype(np.float16)
        centroids[label] = centroid

    # 7. Résumé console
    logger.info("")
    logger.info("=" * 60)
    logger.info("RÉSULTATS DU CLUSTERING")
    logger.info("=" * 60)
    for label in sorted(cluster_docs.keys()):
        kw = ", ".join(kw_by_cluster.get(label, [])[:5])
        logger.info(
            "  Cluster %2d : %4d posts | keywords : %s",
            label, len(cluster_docs[label]), kw or "—",
        )
    logger.info("=" * 60)
    logger.info("")

    if dry_run:
        logger.info("[DRY RUN] Pas d'écriture MongoDB")
        return {"clusters": n_clusters, "dry_run": True}

    # 8. Upsert narratives + mise à jour posts
    results = {}
    for label in sorted(cluster_docs.keys()):
        keywords = kw_by_cluster.get(label, [])
        post_ids = cluster_docs[label]

        # Label lisible : "Narratif #N — keyword1, keyword2"
        kw_label = ", ".join(keywords[:3]) if keywords else f"cluster_{label}"
        narrative_label = f"Narratif #{label} — {kw_label}"

        # Créer ou mettre à jour le narratif
        # On identifie un narratif existant par son label (simpliste pour v1)
        existing = db.narratives.find_one({"label": narrative_label})

        if existing:
            narrative_id = existing["_id"]
            db.narratives.update_one(
                {"_id": narrative_id},
                {"$set": {
                    "keywords":           keywords,
                    "embedding_centroid": centroids[label].tobytes(),
                    "stats.post_count":   len(post_ids),
                    "stats.last_seen_at": now,
                    "updated_at":         now,
                }},
            )
            logger.info("Narratif mis à jour : %s (%d posts)", narrative_label, len(post_ids))
        else:
            doc = new_narrative(
                label             = narrative_label,
                keywords          = keywords,
                embedding_model   = embedding_model,
                similarity_threshold = 0.82,
            )
            doc["embedding_centroid"] = centroids[label].tobytes()
            doc["stats"]["post_count"] = len(post_ids)
            result = db.narratives.insert_one(doc)
            narrative_id = result.inserted_id
            logger.info("Narratif créé : %s (%d posts)", narrative_label, len(post_ids))

        # Mise à jour des posts avec leur narrative_id
        db.posts.update_many(
            {"_id": {"$in": post_ids}},
            {"$set": {
                "nlp.narrative_id": narrative_id,
                "updated_at":       now,
            }},
        )
        results[label] = {
            "narrative_id": narrative_id,
            "post_count":   len(post_ids),
            "keywords":     keywords,
        }

    logger.info(
        "Clustering terminé : %d narratifs créés/mis à jour, %d posts rattachés",
        len(results),
        sum(r["post_count"] for r in results.values()),
    )
    return results


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Job batch de clustering narratif — HDBSCAN sur embeddings MongoDB"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=_HERE / "nlp_pipeline.cfg",
        help="Fichier de configuration (défaut : nlp_pipeline.cfg)",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=15,
        help="Taille minimale d'un cluster HDBSCAN (défaut : 15)",
    )
    parser.add_argument(
        "--umap-components",
        type=int,
        default=50,
        help="Dimensions après réduction UMAP (défaut : 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Afficher les clusters sans écrire dans MongoDB",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Niveau de log console (défaut : INFO)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)

    # Lecture config
    cfg = configparser.ConfigParser()
    if Path(args.config).exists():
        cfg.read(args.config, encoding="utf-8")

    # Connexion MongoDB
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

    embedding_model = (
        cfg["models"].get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")
        if "models" in cfg else "paraphrase-multilingual-MiniLM-L12-v2"
    )

    run_clustering(
        db               = db,
        min_cluster_size = args.min_cluster_size,
        umap_components  = args.umap_components,
        dry_run          = args.dry_run,
        embedding_model  = embedding_model,
    )
