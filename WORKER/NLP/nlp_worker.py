"""nlp_worker.py  v5
====================

Worker NLP — écoute les Change Streams MongoDB et analyse le sentiment
de chaque nouveau post / commentaire avec nlp.status = "pending".

Flux :
    MongoDB (posts + comments, nlp.status=pending)
        → Change Stream
        → SentimentAnalyzer (FR / EN)
        → EmbeddingEngine
        → patch_post_nlp() / patch_comment_nlp()
        → MongoDB (nlp.status=done)

Clustering narratif automatique (v5) :
    Quand la file Change Stream est vide pendant `clustering_idle_seconds`
    (défaut: 60s), le worker déclenche automatiquement une passe de
    narrative_clustering en arrière-plan (thread daemon).
    Si une passe est déjà en cours, un flag "relance" est posé —
    la passe suivante se lance à la fin de la passe courante
    (une seule relance maximum, pas d'accumulation).

Supervision :
    - systemd : voir nlp-worker.service (redémarrage automatique, journald)
    - Heartbeat : log périodique + upsert MongoDB jobs{type:nlp_heartbeat}
      → détection d'un worker silencieux sans crash

Lancement :
    python nlp_worker.py
    python nlp_worker.py --backfill            # traite les pending existants
    python nlp_worker.py --backfill --dry-run  # simulation sans écriture MongoDB
    python nlp_worker.py --skip-clustering     # désactive le clustering auto
    python nlp_worker.py --config /chemin/vers/nlp_pipeline.cfg

Prérequis :
    - MongoDB avec ReplicaSet actif (rs0)
    - nlp_pipeline.cfg dans le même dossier (ou passé via --config)
    - sentiment.py, embeddings.py, narrative_clustering.py dans le même dossier
    - schema.py dans SCHEMA/ (résolution automatique)

Dépendances (conda env nlp_pipeline) :
    pip install pymongo python-dotenv sentencepiece protobuf
    pip install transformers torch accelerate lingua-language-detector
    pip install sentence-transformers numpy
    pip install hdbscan umap-learn scikit-learn
"""

from __future__ import annotations

import argparse
import configparser
import logging
import logging.handlers
import os
import re
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from pymongo.errors import PyMongoError

# ---------------------------------------------------------------------------
# Résolution des chemins
# ---------------------------------------------------------------------------

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

from sentiment import SentimentAnalyzer
from embeddings import EmbeddingEngine
from schema import get_db, patch_post_nlp, patch_comment_nlp

# Import clustering — chargé à la demande pour ne pas ralentir le démarrage
# (hdbscan / umap sont lourds). L'import réel se fait dans _run_clustering().
_narrative_clustering_available = False
try:
    from narrative_clustering import run_clustering as _run_clustering_fn
    _narrative_clustering_available = True
except ImportError:
    _run_clustering_fn = None


# ---------------------------------------------------------------------------
# Codes ANSI
# ---------------------------------------------------------------------------

class C:
    RESET    = "\033[0m"
    BOLD     = "\033[1m"
    DIM      = "\033[2m"
    RED      = "\033[31m"
    GREEN    = "\033[32m"
    YELLOW   = "\033[33m"
    BLUE     = "\033[34m"
    MAGENTA  = "\033[35m"
    CYAN     = "\033[36m"
    WHITE    = "\033[37m"
    BG_RED   = "\033[41m"

    NEGATIVE = RED
    POSITIVE = GREEN
    NEUTRAL  = CYAN
    SKIP     = YELLOW
    DRY      = MAGENTA
    BACKFILL = BLUE
    SUCCESS  = f"{BOLD}{GREEN}"
    SECTION  = f"{BOLD}{BLUE}"
    MONGO    = f"{BOLD}{CYAN}"
    HEART    = f"{BOLD}{MAGENTA}"
    CLUSTER  = f"{BOLD}{YELLOW}"


_LEVEL_COLORS = {
    "DEBUG":    C.DIM + C.WHITE,
    "INFO":     C.WHITE,
    "WARNING":  C.YELLOW,
    "ERROR":    C.RED,
    "CRITICAL": C.BOLD + C.BG_RED + C.WHITE,
}

_MSG_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(===.*?===)"),                                          C.SECTION),
    (re.compile(r"(\[DRY RUN\])"),                                        C.DRY),
    (re.compile(r"\b(BACKFILL)\b"),                                       C.BACKFILL),
    (re.compile(r"(♥ heartbeat)"),                                        C.HEART),
    (re.compile(r"\b(negative)\b"),                                       C.NEGATIVE),
    (re.compile(r"\b(positive)\b"),                                       C.POSITIVE),
    (re.compile(r"\b(neutral)\b"),                                        C.NEUTRAL),
    (re.compile(r"(\[skip\])"),                                           C.SKIP),
    (re.compile(r"(⚠[^\s]*)"),                                            C.YELLOW),
    (re.compile(r"(✓ nlp\.status=done)"),                                 C.SUCCESS),
    (re.compile(r"(\[posts\]|\[comments\])"),                             C.MONGO),
    (re.compile(r"(MongoDB connecté|Change Stream ouvert|Écoute Change)"), C.MONGO),
    (re.compile(r"(erreur\s+\w+)"),                                       C.RED),
    (re.compile(r"(arrêt propre|Worker NLP arrêté)"),                     C.YELLOW),
    (re.compile(r"(\(0\.\d+\))"),                                         C.DIM + C.WHITE),
    (re.compile(r"(🔤 Clustering|clustering auto|narratif)"),              C.CLUSTER),
]


def _colorize(msg: str) -> str:
    for pattern, color in _MSG_PATTERNS:
        msg = pattern.sub(lambda m, c=color: f"{c}{m.group(1)}{C.RESET}", msg)
    return msg


class NLPConsoleFormatter(logging.Formatter):
    _DATE = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        level_color   = _LEVEL_COLORS.get(record.levelname, C.WHITE)
        level_colored = f"{level_color}{record.levelname:<8}{C.RESET}"
        raw_msg       = record.getMessage()
        msg_colored   = _colorize(raw_msg)
        timestamp     = self.formatTime(record, self._DATE)
        line = (
            f"{C.DIM}{timestamp}{C.RESET} "
            f"[{level_colored}] "
            f"{C.DIM}{record.name}{C.RESET} — "
            f"{msg_colored}"
        )
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


# ---------------------------------------------------------------------------
# Lecture de la configuration
# ---------------------------------------------------------------------------

DEFAULT_CFG = _HERE / "nlp_pipeline.cfg"



def _cfg_int(section, key: str, fallback: int = 0) -> int | None:
    """Lit une valeur entière depuis une section configparser.
    Retourne None si la valeur est absente ou vide (champ vidé pour sécurité).
    """
    val = section.get(key, "").strip()
    try:
        return int(val) or None
    except ValueError:
        return fallback or None

def load_config(cfg_path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Fichier de configuration introuvable : {cfg_path}\n"
            f"Créez-le à partir du modèle nlp_pipeline.cfg fourni."
        )
    cfg.read(cfg_path, encoding="utf-8")
    return cfg


# ---------------------------------------------------------------------------
# Initialisation des logs
# ---------------------------------------------------------------------------

def setup_logging(cfg: configparser.ConfigParser) -> logging.Logger:
    """
    Console colorisée + fichier rotatif par taille (texte brut).
    Tous les paramètres sont dans la section [logging] du cfg.
    """
    sec = cfg["logging"]

    level_console = getattr(logging, sec.get("log_level_console", "INFO").upper(),  logging.INFO)
    level_file    = getattr(logging, sec.get("log_level_file",    "DEBUG").upper(), logging.DEBUG)
    to_console    = sec.getboolean("log_to_console", fallback=True)
    to_file       = sec.getboolean("log_to_file",    fallback=True)
    log_dir       = Path(sec.get("log_dir",       "logs"))
    log_filename  = sec.get("log_filename",       "nlp_worker.log")
    max_bytes     = sec.getint("log_max_bytes",    fallback=5 * 1024 * 1024)
    backup_count  = sec.getint("log_backup_count", fallback=5)

    fmt_file = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    # Silencer les loggers tiers verbeux — toujours WARNING minimum
    # même si log_level_file = DEBUG, on ne veut pas les internals pymongo/urllib
    for noisy in ("pymongo", "urllib3", "httpx", "httpcore", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if to_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level_console)
        ch.setFormatter(NLPConsoleFormatter())
        root.addHandler(ch)

    if to_file:
        if not log_dir.is_absolute():
            log_dir = _HERE / log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / log_filename
        fh = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8",
        )
        fh.setLevel(level_file)
        fh.setFormatter(fmt_file)
        root.addHandler(fh)

    logger = logging.getLogger("nlp_worker")
    if to_file:
        logger.info(
            "Logs fichier → %s (max %d Mo, %d backups)",
            log_path, max_bytes // (1024 * 1024), backup_count,
        )
    return logger


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

class Heartbeat:
    """
    Thread daemon qui pulse toutes les `interval` secondes.

    Deux actions à chaque pulse :
      1. Log INFO  "♥ heartbeat — en veille | traités: N | erreurs: E"
      2. Upsert MongoDB dans la collection `jobs` :
            { type: "nlp_heartbeat", worker_id: <pid> }
         avec les champs : status, last_seen_at, stats
         → un outil de supervision peut alerter si last_seen_at est trop vieux.

    Le document heartbeat est créé au 1er pulse et mis à jour ensuite
    (upsert sur worker_id + type). Un seul document par worker.
    """

    HEARTBEAT_TYPE = "nlp_heartbeat"

    def __init__(self, db, worker_id: str, interval: int = 60) -> None:
        self._db        = db
        self._worker_id = worker_id
        self._interval  = interval
        self._logger    = logging.getLogger("nlp_worker.heartbeat")
        self._stop_evt  = threading.Event()

        # Compteurs partagés (mis à jour par le worker principal)
        self.processed = 0
        self.errors    = 0

        self._thread = threading.Thread(
            target=self._loop,
            name="heartbeat",
            daemon=True,   # s'arrête automatiquement quand le process principal quitte
        )

    def start(self) -> None:
        self._thread.start()
        self._logger.info(
            "♥ heartbeat démarré (interval=%ds, worker_id=%s)",
            self._interval, self._worker_id,
        )

    def stop(self) -> None:
        self._stop_evt.set()

    def _loop(self) -> None:
        while not self._stop_evt.wait(timeout=self._interval):
            self._pulse()

    def _pulse(self) -> None:
        now = datetime.now(timezone.utc)

        # --- Log console / fichier ---
        self._logger.info(
            "♥ heartbeat — en veille | traités: %d | erreurs: %d",
            self.processed, self.errors,
        )

        # --- Upsert MongoDB ---
        try:
            self._db.jobs.update_one(
                {
                    "type":      self.HEARTBEAT_TYPE,
                    "worker_id": self._worker_id,
                },
                {"$set": {
                    "type":         self.HEARTBEAT_TYPE,
                    "status":       "running",
                    "worker_id":    self._worker_id,
                    "last_seen_at": now,
                    "stats": {
                        "processed": self.processed,
                        "errors":    self.errors,
                        "updated_at": now,
                    },
                }},
                upsert=True,
            )
        except Exception as exc:
            # On ne crashe pas le worker pour un heartbeat raté
            self._logger.warning("♥ heartbeat MongoDB échoué : %s", exc)


# ---------------------------------------------------------------------------
# Worker principal
# ---------------------------------------------------------------------------

class NLPWorker:
    """
    Écoute les Change Streams sur posts et comments,
    analyse le sentiment, et met à jour MongoDB.
    """

    def __init__(
        self,
        cfg: configparser.ConfigParser,
        dry_run: bool = False,
        skip_clustering: bool = False,
    ) -> None:
        self.cfg      = cfg
        self.dry_run  = dry_run
        self.logger   = logging.getLogger("nlp_worker")
        self._running = True

        self.logger.info("=== Worker NLP démarré (dry_run=%s) ===", dry_run)

        # --- Paramètres worker ---
        w = cfg["worker"]
        self._max_retries           = w.getint("max_retries",         fallback=10)
        self._retry_delay           = w.getint("retry_delay",         fallback=5)
        self._embedding_placeholder = w.get("embedding_model_placeholder", "pending")
        self._heartbeat_interval    = w.getint("heartbeat_interval",  fallback=60)
        self._collections           = [
            c.strip() for c in w.get("watch_collections", "posts, comments").split(",")
        ]

        # --- Paramètres clustering [v5] ---
        self.skip_clustering          = skip_clustering or not _narrative_clustering_available
        self._clustering_idle_seconds = w.getint("clustering_idle_seconds", fallback=60)
        self._clustering_min_size     = w.getint("clustering_min_cluster_size", fallback=5)
        self._clustering_umap_comp    = w.getint("clustering_umap_components",  fallback=50)
        # Verrou + flag "relance demandée"
        self._clustering_lock         = threading.Lock()
        self._clustering_thread       = None
        self._clustering_rerun        = False   # posé si déclenchement pendant une passe

        if self.skip_clustering:
            if not _narrative_clustering_available:
                self.logger.warning(
                    "🔤 Clustering auto désactivé — narrative_clustering.py introuvable"
                )
            else:
                self.logger.info("🔤 Clustering auto désactivé (--skip-clustering)")
        else:
            self.logger.info(
                "🔤 Clustering auto activé (idle=%ds, min_cluster_size=%d)",
                self._clustering_idle_seconds, self._clustering_min_size,
            )

        # Identifiant unique du worker (PID — lisible dans les logs et MongoDB)
        self._worker_id = f"nlp_worker_{os.getpid()}"
        self.logger.info("Worker ID : %s", self._worker_id)

        # --- Connexion MongoDB ---
        m = cfg["mongodb"]
        self.db = get_db(
            db_name  = m.get("db")       or None,
            host     = m.get("host")     or None,
            port     = _cfg_int(m, "port"),
            user     = m.get("user")     or None,
            password = m.get("password") or None,
            auth_db  = m.get("auth_db")  or None,
        )
        self.logger.info("MongoDB connecté : %s", self.db.name)

        # --- Analyseur de sentiment ---
        device = cfg["models"].get("device", "auto")
        self.analyzer = SentimentAnalyzer(device=device)
        self.logger.info("SentimentAnalyzer prêt (device=%s)", self.analyzer._device)

        # --- Moteur d'embedding ---
        emb_model       = cfg["models"].get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")
        emb_thresh      = cfg["embeddings"].getfloat("dedup_threshold",        fallback=0.95)
        self._dedup_limit = cfg["embeddings"].getint("dedup_candidates_limit", fallback=5000)
        self.embedding_engine = EmbeddingEngine(
            model_name      = emb_model,
            dedup_threshold = emb_thresh,
            device          = device,
        )
        self.logger.info(
            "EmbeddingEngine prêt (model=%s, dedup_threshold=%.2f, dedup_limit=%d)",
            emb_model, emb_thresh, self._dedup_limit,
        )

        # --- Heartbeat ---
        self._heartbeat = Heartbeat(
            db        = self.db,
            worker_id = self._worker_id,
            interval  = self._heartbeat_interval,
        )

        # --- Signaux d'arrêt propre ---
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame) -> None:
        self.logger.warning("Signal %s reçu — arrêt propre en cours…", signum)
        self._running = False

    # ------------------------------------------------------------------
    # Clustering narratif automatique  [v5]
    # ------------------------------------------------------------------

    def _trigger_clustering(self) -> None:
        """
        Déclenche une passe de clustering narratif en arrière-plan.

        Règles :
          - Si désactivé (skip_clustering) → ignore
          - Si une passe est déjà en cours → pose le flag relance, skip
          - Sinon → lance dans un thread daemon
        """
        if self.skip_clustering or self.dry_run:
            return

        if not self._clustering_lock.acquire(blocking=False):
            # Passe en cours — on mémorise qu'une relance est souhaitée
            if not self._clustering_rerun:
                self._clustering_rerun = True
                self.logger.debug(
                    "🔤 Clustering auto : passe en cours — relance programmée"
                )
            return

        # Vérification thread précédent
        if self._clustering_thread and self._clustering_thread.is_alive():
            self._clustering_lock.release()
            if not self._clustering_rerun:
                self._clustering_rerun = True
            return

        self.logger.info(
            "🔤 Clustering auto : file vide depuis %ds — lancement",
            self._clustering_idle_seconds,
        )
        self._clustering_thread = threading.Thread(
            target=self._run_clustering,
            name="narrative-clustering",
            daemon=True,
        )
        self._clustering_thread.start()

    def _run_clustering(self) -> None:
        """
        Exécute run_clustering() dans le thread dédié.
        Gère la relance unique si le flag _clustering_rerun est posé.
        """
        try:
            results = _run_clustering_fn(
                db               = self.db,
                min_cluster_size = self._clustering_min_size,
                umap_components  = self._clustering_umap_comp,
                dry_run          = False,
            )
            n = len(results) if results else 0
            self.logger.info(
                "🔤 Clustering auto terminé : %d narratif(s) créé(s)/mis à jour", n
            )
        except Exception as exc:
            self.logger.error(
                "🔤 Clustering auto : erreur inattendue : %s", exc, exc_info=True
            )
        finally:
            self._clustering_lock.release()

        # Relance unique si demandée pendant la passe
        if self._clustering_rerun:
            self._clustering_rerun = False
            self.logger.info("🔤 Clustering auto : relance unique (flag posé pendant la passe)")
            self._trigger_clustering()

    # ------------------------------------------------------------------
    # Backfill
    # ------------------------------------------------------------------

    def backfill(self) -> None:
        """Traite tous les documents pending existants en base."""
        self.logger.info("=== BACKFILL démarré ===")
        for col_name in self._collections:
            count = 0
            for doc in self.db[col_name].find({"nlp.status": "pending"}):
                self._process_document(doc, col_name)
                count += 1
            self.logger.info(
                "Backfill %-10s : %d documents traités", col_name, count
            )
        self.logger.info("=== BACKFILL terminé ===")

    # ------------------------------------------------------------------
    # Boucle principale Change Streams
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Démarre le heartbeat puis l'écoute des Change Streams."""
        self._heartbeat.start()

        self.logger.info(
            "Écoute Change Streams sur : %s", ", ".join(self._collections)
        )
        retries = 0
        while self._running and retries < self._max_retries:
            try:
                self._watch_loop()
                retries = 0
            except PyMongoError as exc:
                retries += 1
                self.logger.error(
                    "Erreur MongoDB (%d/%d) : %s — retry dans %ds",
                    retries, self._max_retries, exc, self._retry_delay,
                )
                time.sleep(self._retry_delay)
            except Exception as exc:
                self.logger.critical("Erreur inattendue : %s", exc, exc_info=True)
                break

        self._heartbeat.stop()

        # Marquer le worker comme arrêté dans MongoDB
        try:
            self.db.jobs.update_one(
                {"type": Heartbeat.HEARTBEAT_TYPE, "worker_id": self._worker_id},
                {"$set": {
                    "status":       "stopped",
                    "last_seen_at": datetime.now(timezone.utc),
                }},
            )
        except Exception:
            pass

        self.logger.info("=== Worker NLP arrêté ===")

    def _watch_loop(self) -> None:
        pipeline = [
            {"$match": {
                "operationType": {"$in": ["insert", "update", "replace"]},
                "$or": [
                    {"fullDocument.nlp.status": "pending"},
                    {"updateDescription.updatedFields.nlp.status": "pending"},
                ],
                "ns.coll": {"$in": self._collections},
            }}
        ]
        self.logger.info("Change Stream ouvert")

        # Compteur idle pour le déclenchement du clustering.
        # max_await_time_ms=1000 → chaque try_next() attend 1s max.
        # Après clustering_idle_seconds None consécutifs → clustering déclenché.
        # Le flag se reset à chaque document entrant.
        idle_count                = 0
        idle_trigger              = self._clustering_idle_seconds
        _clustering_done_this_idle = False

        with self.db.watch(
            pipeline,
            full_document="updateLookup",
            max_await_time_ms=1000,
        ) as stream:
            while self._running:
                change = stream.try_next()

                if change is None:
                    idle_count += 1
                    if (
                        not self.skip_clustering
                        and not _clustering_done_this_idle
                        and idle_count >= idle_trigger
                    ):
                        _clustering_done_this_idle = True
                        self._trigger_clustering()
                    continue

                # Document reçu → reset idle
                idle_count                 = 0
                _clustering_done_this_idle = False

                doc        = change.get("fullDocument")
                collection = change["ns"]["coll"]
                if doc is None:
                    continue
                if doc.get("nlp", {}).get("status") != "pending":
                    continue
                self._process_document(doc, collection)

    # ------------------------------------------------------------------
    # Traitement d'un document
    # ------------------------------------------------------------------

    def _process_document(self, doc: dict, collection_name: str) -> None:
        doc_id = doc["_id"]
        text   = (doc.get("text", {}).get("content", "") or "").strip()

        self.logger.info(
            "[%s] %s | %.60s",
            collection_name, doc_id,
            text.replace("\n", " ") or "<vide>",
        )

        if not self.dry_run:
            self._set_status(collection_name, doc_id, "processing")

        # --- Sentiment ---
        try:
            sentiment = self.analyzer.analyze(text)
        except Exception as exc:
            self.logger.error(
                "[%s] %s | erreur sentiment : %s", collection_name, doc_id, exc
            )
            self._heartbeat.errors += 1
            if not self.dry_run:
                self._set_error(collection_name, doc_id, str(exc))
            return

        score_warn = sentiment.score < 0.30 and not sentiment.skipped
        log_fn = self.logger.warning if score_warn else self.logger.info
        log_fn(
            "[%s] %s | %s → %s (%.4f) %s%s%s",
            collection_name, doc_id,
            sentiment.lang, sentiment.label, sentiment.score, sentiment.model,
            " [skip]" if sentiment.skipped else "",
            " ⚠ score faible" if score_warn else "",
        )

        # --- Embedding ---
        emb_result     = None
        dup_id         = None
        dup_score      = None

        try:
            emb_result = self.embedding_engine.embed(text)

            if not emb_result.skipped:
                # Déduplication : charge les embeddings récents de la même collection
                # On limite à 5000 candidats pour éviter de surcharger la RAM
                candidates = [
                    (d["_id"], d.get("nlp", {}).get("embedding"))
                    for d in self.db[collection_name].find(
                        {
                            "nlp.embedding": {"$ne": None},
                            "_id":           {"$ne": doc_id},
                        },
                        {"nlp.embedding": 1},
                    ).limit(self._dedup_limit)
                ]
                dup_id, dup_score = self.embedding_engine.find_duplicate(
                    emb_result.vector, candidates
                )
                if dup_id is not None:
                    self.logger.warning(
                        "[%s] %s | ⚠ doublon détecté → %s (score=%.4f)",
                        collection_name, doc_id, dup_id, dup_score,
                    )
                else:
                    self.logger.debug(
                        "[%s] %s | embedding OK (dim=%d)", collection_name, doc_id, emb_result.dim
                    )

        except Exception as exc:
            self.logger.warning(
                "[%s] %s | erreur embedding : %s", collection_name, doc_id, exc
            )
            # On ne bloque pas le pipeline pour une erreur d'embedding

        if self.dry_run:
            self.logger.debug("[DRY RUN] Pas d'écriture MongoDB")
            self._heartbeat.processed += 1
            return

        # --- Écriture MongoDB ---
        try:
            embedding_bytes = emb_result.vector_bytes if emb_result and not emb_result.skipped else None
            embedding_model = emb_result.model if emb_result else self._embedding_placeholder

            if collection_name == "posts":
                patch = patch_post_nlp(
                    sentiment_label  = sentiment.label,
                    sentiment_score  = sentiment.score,
                    sentiment_model  = sentiment.model,
                    embedding_model  = embedding_model,
                    embedding        = embedding_bytes,
                    topics           = [],
                    narrative_id     = None,
                    is_duplicate_of  = dup_id,
                    similarity_score = dup_score,
                )
            else:
                patch = patch_comment_nlp(
                    sentiment_label  = sentiment.label,
                    sentiment_score  = sentiment.score,
                    sentiment_model  = sentiment.model,
                    embedding_model  = embedding_model,
                    embedding        = embedding_bytes,
                    topics           = [],
                    is_duplicate_of  = dup_id,
                    similarity_score = dup_score,
                )
            self.db[collection_name].update_one({"_id": doc_id}, patch)
            self.logger.debug("[%s] %s | ✓ nlp.status=done", collection_name, doc_id)
            self._heartbeat.processed += 1

        except PyMongoError as exc:
            self.logger.error(
                "[%s] %s | erreur écriture : %s", collection_name, doc_id, exc
            )
            self._heartbeat.errors += 1
            self._set_error(collection_name, doc_id, str(exc))

    # ------------------------------------------------------------------
    # Helpers MongoDB
    # ------------------------------------------------------------------

    def _set_status(self, collection: str, doc_id, status: str) -> None:
        now = datetime.now(timezone.utc)
        self.db[collection].update_one(
            {"_id": doc_id},
            {"$set": {"nlp.status": status, "updated_at": now}},
        )

    def _set_error(self, collection: str, doc_id, error: str) -> None:
        now = datetime.now(timezone.utc)
        self.db[collection].update_one(
            {"_id": doc_id},
            {"$set": {
                "nlp.status":       "error",
                "nlp.error":        error,
                "nlp.processed_at": now,
                "updated_at":       now,
            }},
        )


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Worker NLP — sentiment + embedding + clustering narratif automatique"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=DEFAULT_CFG,
        help=f"Chemin vers le fichier de configuration (défaut : {DEFAULT_CFG.name})",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Traiter les documents pending existants avant de démarrer le stream",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyser sans écrire dans MongoDB (test)",
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Désactiver le clustering narratif automatique",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    cfg    = load_config(args.config)
    logger = setup_logging(cfg)
    logger.info("Configuration chargée : %s", args.config)

    worker = NLPWorker(
        cfg             = cfg,
        dry_run         = args.dry_run,
        skip_clustering = args.skip_clustering,
    )

    if args.backfill:
        worker.backfill()

    worker.run()
