"""network_worker.py  v4
========================

Worker réseau — ETL MongoDB → Neo4j via Change Streams.

Extrait les relations entre comptes et posts depuis MongoDB
et les pousse dans Neo4j pour l'analyse de réseau et la
détection de campagnes coordonnées.

Relations créées dans Neo4j :
    (:Account)-[:A_PUBLIÉ]->(:Post)
    (:Account)-[:A_COMMENTÉ]->(:Post)
    (:Account)-[:A_FORWARDÉ {count}]->(:Post)      Telegram
    (:Post)-[:EST_DOUBLON_DE]->(:Post)              NLP déduplication
    (:Post)-[:APPARTIENT_À]->(:Narrative)           NLP clustering
    (:Post)-[:HAS_HASHTAG]->(:Hashtag)              Hashtags coordonnés  [v3]
    (:Post)-[:IS_DEEPFAKE {score}]->(:Deepfake)     Médias synthétiques  [v3]
    (:Post)-[:A_MEDIA]->(:Media)                    Fichiers médias      [v4]

Détection automatique (v2) :
    Quand la file Change Stream est vide pendant `detection_idle_seconds`
    (défaut: 30s), le worker déclenche automatiquement une passe de
    campaign_detector en arrière-plan. Si une passe est déjà en cours,
    la nouvelle est ignorée (skip).

Mode --projet (v3) :
    Traite uniquement les documents dont source.project correspond
    aux projets passés en argument. Purge Neo4j + reset sync.neo4j=False
    dans MongoDB avant l'import (sauf si --add est spécifié).

    python network_worker.py --projet ProJet0 --projet TIKTOK_crypto_2026-03-25
    python network_worker.py --projet ProJet0 --add   # ajout sans purge

Supervision :
    - systemd : voir network-worker.service
    - Heartbeat : log périodique + upsert MongoDB jobs{type:network_heartbeat}

Lancement :
    python network_worker.py
    python network_worker.py --backfill            # sync l'existant au démarrage
    python network_worker.py --backfill --dry-run  # simulation sans écriture Neo4j
    python network_worker.py --skip-detection      # désactive la détection auto
    python network_worker.py --config network_pipeline.cfg
    python network_worker.py --projet ProJet0 [-p ProJet1 …]
    python network_worker.py --projet ProJet0 --add   # sans purge Neo4j

Dépendances :
    pip install pymongo python-dotenv neo4j
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
    load_dotenv(_HERE / ".env",               override=False)  # WORKER/NETWORK/.env
    load_dotenv(_HERE.parent.parent / ".env", override=False)  # ~/AI-FORENSICS/.env
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
    # Aucun trouvé — ajout des chemins standards pour que l'ImportError soit lisible
    for _p in [_HERE, _HERE.parent]:
        sys.path.insert(0, str(_p))

from schema import get_db, patch_post_sync, patch_comment_sync, patch_media_sync, patch_account_sync
from neo4j_client import Neo4jClient


# ---------------------------------------------------------------------------
# Codes ANSI (même palette que le worker NLP)
# ---------------------------------------------------------------------------

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"
    BG_RED  = "\033[41m"

    SUCCESS   = f"{BOLD}{GREEN}"
    SECTION   = f"{BOLD}{BLUE}"
    MONGO     = f"{BOLD}{CYAN}"
    NEO4J     = f"{BOLD}{MAGENTA}"
    HEART     = f"{BOLD}{MAGENTA}"
    RELATION  = f"{BOLD}{CYAN}"
    DETECTION = f"{BOLD}{YELLOW}"


_LEVEL_COLORS = {
    "DEBUG":    C.DIM + C.WHITE,
    "INFO":     C.WHITE,
    "WARNING":  C.YELLOW,
    "ERROR":    C.RED,
    "CRITICAL": C.BOLD + C.BG_RED + C.WHITE,
}

_MSG_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(===.*?===)"),                                           C.SECTION),
    (re.compile(r"(\[DRY RUN\])"),                                         C.MAGENTA),
    (re.compile(r"\b(BACKFILL)\b"),                                        C.BLUE),
    (re.compile(r"(♥ heartbeat)"),                                         C.HEART),
    (re.compile(r"(Neo4j connecté|Neo4j déconnecté)"),                     C.NEO4J),
    (re.compile(r"(→ Neo4j)"),                                             C.NEO4J),
    (re.compile(r"(A_PUBLIÉ|A_COMMENTÉ|A_FORWARDÉ|EST_DOUBLON_DE|APPARTIENT_À|HAS_HASHTAG|IS_DEEPFAKE|A_MEDIA)"), C.RELATION),
    (re.compile(r"(✓\s+\w+)"),                                             C.SUCCESS),
    (re.compile(r"(\[posts\]|\[comments\]|\[accounts\])"),                 C.MONGO),
    (re.compile(r"(MongoDB connecté|Change Stream ouvert|Écoute Change)"), C.MONGO),
    (re.compile(r"(🔍 Détection|détection auto|campagne)"),                C.DETECTION),
    (re.compile(r"(🗑 Purge|purge Neo4j|reset sync)"),                      C.RED),
    (re.compile(r"(📂 Projet|--projet|projets filtrés)"),                   C.CYAN),
    (re.compile(r"(erreur\s+\w+)"),                                        C.RED),
    (re.compile(r"(arrêt propre|Worker réseau arrêté)"),                   C.YELLOW),
]


def _colorize(msg: str) -> str:
    for pattern, color in _MSG_PATTERNS:
        msg = pattern.sub(lambda m, c=color: f"{c}{m.group(1)}{C.RESET}", msg)
    return msg


class NetworkConsoleFormatter(logging.Formatter):
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

DEFAULT_CFG = _HERE / "network_pipeline.cfg"


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
        raise FileNotFoundError(f"Configuration introuvable : {cfg_path}")
    cfg.read(cfg_path, encoding="utf-8")
    return cfg


# ---------------------------------------------------------------------------
# Initialisation des logs
# ---------------------------------------------------------------------------

def setup_logging(cfg: configparser.ConfigParser) -> logging.Logger:
    sec = cfg["logging"]

    level_console = getattr(logging, sec.get("log_level_console", "INFO").upper(),  logging.INFO)
    level_file    = getattr(logging, sec.get("log_level_file",    "DEBUG").upper(), logging.DEBUG)
    to_console    = sec.getboolean("log_to_console", fallback=True)
    to_file       = sec.getboolean("log_to_file",    fallback=True)
    log_dir       = Path(sec.get("log_dir",      "logs"))
    log_filename  = sec.get("log_filename",      "network_worker.log")
    max_bytes     = sec.getint("log_max_bytes",    fallback=5 * 1024 * 1024)
    backup_count  = sec.getint("log_backup_count", fallback=5)

    fmt_file = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    for noisy in ("pymongo", "urllib3", "neo4j", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Supprime les warnings de dépréciation GDS (bavards mais non bloquants)
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

    if to_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level_console)
        ch.setFormatter(NetworkConsoleFormatter())
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

    logger = logging.getLogger("network_worker")
    if to_file:
        logger.info("Logs fichier → %s (max %d Mo, %d backups)",
                    log_path, max_bytes // (1024 * 1024), backup_count)
    return logger


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

class Heartbeat:
    HEARTBEAT_TYPE = "network_heartbeat"

    def __init__(self, db, worker_id: str, interval: int = 60) -> None:
        self._db        = db
        self._worker_id = worker_id
        self._interval  = interval
        self._logger    = logging.getLogger("network_worker.heartbeat")
        self._stop_evt  = threading.Event()
        self.synced     = 0
        self.errors     = 0
        self._thread    = threading.Thread(target=self._loop, name="heartbeat", daemon=True)

    def start(self) -> None:
        self._thread.start()
        self._logger.info("♥ heartbeat démarré (interval=%ds, worker_id=%s)",
                          self._interval, self._worker_id)

    def stop(self) -> None:
        self._stop_evt.set()

    def _loop(self) -> None:
        while not self._stop_evt.wait(timeout=self._interval):
            self._pulse()

    def _pulse(self) -> None:
        now = datetime.now(timezone.utc)
        self._logger.info("♥ heartbeat — en veille | syncés: %d | erreurs: %d",
                          self.synced, self.errors)
        try:
            self._db.jobs.update_one(
                {"type": self.HEARTBEAT_TYPE, "worker_id": self._worker_id},
                {"$set": {
                    "type":         self.HEARTBEAT_TYPE,
                    "status":       "running",
                    "worker_id":    self._worker_id,
                    "last_seen_at": now,
                    "stats": {"synced": self.synced, "errors": self.errors, "updated_at": now},
                }},
                upsert=True,
            )
        except Exception as exc:
            self._logger.warning("♥ heartbeat MongoDB échoué : %s", exc)


# ---------------------------------------------------------------------------
# Worker principal
# ---------------------------------------------------------------------------

class NetworkWorker:
    """
    Écoute les Change Streams MongoDB et synchronise vers Neo4j.

    v2 : déclenche automatiquement campaign_detector quand la file
    est vide depuis `detection_idle_seconds` secondes.

    v3 : mode --projet pour filtrer par projet(s) + nœuds Hashtag/Deepfake.
    """

    def __init__(
        self,
        cfg: configparser.ConfigParser,
        dry_run: bool = False,
        skip_detection: bool = False,
        projets: list[str] | None = None,
        add_mode: bool = False,
    ) -> None:
        self.cfg            = cfg
        self.dry_run        = dry_run
        self.skip_detection = skip_detection
        self.projets        = projets or []          # liste des projets filtrés
        self.add_mode       = add_mode               # True = pas de purge Neo4j
        self.logger         = logging.getLogger("network_worker")
        self._running       = True

        mode_label = "projet" if self.projets else "stream"
        self.logger.info("=== Worker réseau démarré (dry_run=%s, mode=%s) ===",
                         dry_run, mode_label)

        # --- Paramètres worker ---
        w = cfg["worker"]
        self._max_retries  = w.getint("max_retries",        fallback=10)
        self._retry_delay  = w.getint("retry_delay",        fallback=5)
        self._batch_size   = w.getint("batch_size",         fallback=100)
        self._hb_interval  = w.getint("heartbeat_interval", fallback=60)
        self._collections  = [
            c.strip() for c in w.get("watch_collections", "posts, comments, accounts, media").split(",")
        ]

        # Délai d'inactivité avant déclenchement de la détection (secondes)
        # max_await_time_ms=1000 → chaque try_next() attend 1s max
        # idle_seconds=30 → après 30 try_next() None consécutifs → détection
        self._detection_idle_seconds = w.getint("detection_idle_seconds", fallback=30)

        e = cfg["etl"]
        self._sync_accounts      = e.getboolean("sync_accounts",      fallback=True)
        self._sync_posts         = e.getboolean("sync_posts",          fallback=True)
        self._sync_duplicates    = e.getboolean("sync_duplicates",     fallback=True)
        self._sync_narratives    = e.getboolean("sync_narratives",     fallback=True)
        self._sync_forwards      = e.getboolean("sync_forwards",       fallback=True)
        self._forward_min        = e.getint("forward_min_count",       fallback=2)
        self._sync_hashtags      = e.getboolean("sync_hashtags",       fallback=True)
        self._sync_deepfake_nodes= e.getboolean("sync_deepfake_nodes", fallback=True)
        self._sync_media         = e.getboolean("sync_media",          fallback=True)
        self._sync_projects      = e.getboolean("sync_projects",       fallback=True)  # [v5]

        self._backfill_log_interval = w.getint("backfill_log_interval", fallback=50)  # [v5]

        self._worker_id = f"network_worker_{os.getpid()}"
        self.logger.info("Worker ID : %s", self._worker_id)

        if self.projets:
            self.logger.info(
                "📂 Projet — projets filtrés : %s | add_mode=%s",
                self.projets, self.add_mode,
            )

        # --- MongoDB ---
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

        # --- Neo4j ---
        if not dry_run:
            import os as _os
            n = cfg["neo4j"] if "neo4j" in cfg else {}
            self.neo4j = Neo4jClient(
                uri      = n.get("uri")      or _os.getenv("NEO4J_URI",      "bolt://localhost:7687"),
                user     = n.get("user")     or _os.getenv("NEO4J_USER",     "neo4j"),
                password = n.get("password") or _os.getenv("NEO4J_PASSWORD", ""),
            )
            self.neo4j.create_constraints()
        else:
            self.neo4j = None
            self.logger.info("[DRY RUN] Neo4j non connecté")

        # --- Détecteur de campagnes ---
        # Verrou pour éviter deux passes simultanées (skip si déjà en cours)
        self._detection_lock    = threading.Lock()
        self._detection_thread  = None   # thread courant de détection

        if skip_detection:
            self.logger.info("Détection automatique désactivée (--skip-detection)")
        else:
            self.logger.info(
                "Détection automatique activée (idle=%ds)",
                self._detection_idle_seconds,
            )

        # --- Heartbeat ---
        self._heartbeat = Heartbeat(self.db, self._worker_id, self._hb_interval)

        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame) -> None:
        self.logger.warning("Signal %s reçu — arrêt propre en cours…", signum)
        self._running = False

    # ------------------------------------------------------------------
    # Purge Neo4j + reset MongoDB  (mode --projet sans --add)
    # ------------------------------------------------------------------

    def purge_and_reset(self) -> None:
        """
        Purge totale de Neo4j puis reset de sync.neo4j=False dans MongoDB
        pour tous les posts/comments/accounts concernés.

        Appelé en mode --projet sauf si --add est actif.
        """
        self.logger.warning("🗑 Purge Neo4j — SUPPRESSION DE TOUS LES NŒUDS ET RELATIONS")
        if not self.dry_run and self.neo4j:
            self.neo4j.purge_all()
            self.logger.info("🗑 Purge Neo4j terminée")
        elif self.dry_run:
            self.logger.info("[DRY RUN] Purge Neo4j simulée (aucune suppression)")

        # Reset sync.neo4j=False dans MongoDB pour les projets concernés
        # Les accounts n'ont pas de source.project → on reset tous les accounts
        # Les posts et comments sont filtrés par source.project si projets définis
        self.logger.info("🗑 reset sync.neo4j=False dans MongoDB…")
        if not self.dry_run:
            self._reset_mongo_sync()
        else:
            self.logger.info("[DRY RUN] Reset MongoDB simulé")

    def _reset_mongo_sync(self) -> None:
        """Remet sync.neo4j=False pour les collections concernées."""
        now = datetime.now(timezone.utc)

        # Posts, comments et media : filtrer par source.project si projets définis
        if self.projets:
            filt_posts    = {"source.project": {"$in": self.projets}}
            filt_comments = {"source.project": {"$in": self.projets}}
            filt_media    = {"source.project": {"$in": self.projets}}
        else:
            filt_posts    = {}
            filt_comments = {}
            filt_media    = {}

        patch = {"$set": {"sync.neo4j": False, "sync.synced_at": None, "updated_at": now}}

        r_posts    = self.db.posts.update_many(filt_posts, patch)
        r_comments = self.db.comments.update_many(filt_comments, patch)
        r_media    = self.db.media.update_many(filt_media, patch)
        # Accounts : reset global (pas de source.project sur les accounts)
        # Inclut les accounts sans champ sync (anciens documents)
        r_accounts = self.db.accounts.update_many({}, patch)

        self.logger.info(
            "🗑 reset sync — posts: %d | comments: %d | media: %d | accounts: %d",
            r_posts.modified_count,
            r_comments.modified_count,
            r_media.modified_count,
            r_accounts.modified_count,
        )

    # ------------------------------------------------------------------
    # Backfill — synchronise tous les documents existants
    # ------------------------------------------------------------------

    def backfill(self) -> None:
        """
        Synchronise l'ensemble des données existantes vers Neo4j.
        En mode --projet, filtre par source.project.
        """
        self.logger.info("=== BACKFILL démarré (projets=%s) ===",
                         self.projets if self.projets else "tous")

        if self._sync_accounts:
            self._backfill_accounts()

        if self._sync_narratives:
            self._backfill_narratives()

        if self._sync_posts:
            self._backfill_posts()

        if self._sync_media:
            self._backfill_media()

        self.logger.info("=== BACKFILL terminé ===")

    def _backfill_accounts(self) -> None:
        self.logger.info("BACKFILL accounts…")
        batch     = []
        batch_ids = []
        count     = 0
        interval  = self._backfill_log_interval

        filt = {"$or": [
            {"sync.neo4j": False},
            {"sync":       {"$exists": False}},
        ]}
        total = self.db.accounts.count_documents(filt)
        self.logger.info("  %d account(s) à synchroniser", total)

        for doc in self.db.accounts.find(filt):
            node = self._account_to_node(doc)
            if node:
                batch.append(node)
                batch_ids.append(doc["_id"])

            if len(batch) >= self._batch_size:
                self._flush_accounts(batch)
                if not self.dry_run:
                    self.db.accounts.update_many(
                        {"_id": {"$in": batch_ids}},
                        patch_account_sync(neo4j=True),
                    )
                count += len(batch)
                if interval > 0 and count % interval == 0:
                    self.logger.info("  ↳ accounts : %d / %d (%.0f%%)",
                                     count, total, 100*count/total if total else 0)
                batch, batch_ids = [], []

        if batch:
            self._flush_accounts(batch)
            if not self.dry_run:
                self.db.accounts.update_many(
                    {"_id": {"$in": batch_ids}},
                    patch_account_sync(neo4j=True),
                )
            count += len(batch)

        self.logger.info("Backfill accounts : %d synchronisés → Neo4j", count)

    def _backfill_narratives(self) -> None:
        self.logger.info("BACKFILL narratives…")
        count = 0
        for doc in self.db.narratives.find({}):
            if not self.dry_run and self.neo4j:
                self.neo4j.upsert_narrative({
                    "mongo_id":   str(doc["_id"]),
                    "label":      doc.get("label", ""),
                    "keywords":   doc.get("keywords", []),
                    "post_count": doc.get("stats", {}).get("post_count", 0),
                    "updated_at": str(doc.get("updated_at", "")),
                })
                count += 1
        self.logger.info("Backfill narratives : %d synchronisés → Neo4j", count)

    def _backfill_posts(self) -> None:
        self.logger.info("BACKFILL posts…")
        batch_posts     = []
        batch_pub       = []
        batch_dup       = []
        batch_narr      = []
        batch_ids       = []
        batch_proj_post = []   # liens Project→Post  [v5]
        batch_proj_acc  = []   # liens Project→Account [v5]
        count    = 0
        interval = self._backfill_log_interval

        filt = {"sync.neo4j": False}
        if self.projets:
            filt["source.project"] = {"$in": self.projets}
        total = self.db.posts.count_documents(filt)
        self.logger.info("  %d post(s) à synchroniser", total)

        for doc in self.db.posts.find(filt):
            node = self._post_to_node(doc)
            if not node:
                continue

            batch_posts.append(node)
            batch_ids.append(doc["_id"])

            account_id = str(doc.get("account_id", "")) if doc.get("account_id") else None
            if account_id:
                batch_pub.append({
                    "from_id": account_id,
                    "to_id":   str(doc["_id"]),
                    "props":   {},
                })

            dup_id = doc.get("nlp", {}).get("is_duplicate_of")
            sim    = doc.get("nlp", {}).get("similarity_score")
            if dup_id and self._sync_duplicates:
                batch_dup.append({
                    "from_id": str(doc["_id"]),
                    "to_id":   str(dup_id),
                    "props":   {"similarity_score": sim or 0.0},
                })

            narr_id = doc.get("nlp", {}).get("narrative_id")
            if narr_id and self._sync_narratives:
                batch_narr.append({
                    "from_id": str(doc["_id"]),
                    "to_id":   str(narr_id),
                    "props":   {},
                })

            # Nœud Project [v5]
            proj = (doc.get("source") or {}).get("project") or ""
            if proj and self._sync_projects:
                batch_proj_post.append({"project": proj, "post_id": str(doc["_id"])})
                if account_id:
                    batch_proj_acc.append({"project": proj, "account_id": account_id})

            if len(batch_posts) >= self._batch_size:
                self._flush_posts_batch(batch_posts, batch_pub, batch_dup, batch_narr, [])
                if self._sync_projects and not self.dry_run and self.neo4j:
                    self.neo4j.upsert_project_batch(batch_proj_post, batch_proj_acc)
                if not self.dry_run:
                    self.db.posts.update_many(
                        {"_id": {"$in": batch_ids}},
                        patch_post_sync(neo4j=True),
                    )
                count += len(batch_posts)
                if interval > 0 and count % interval == 0:
                    self.logger.info("  ↳ posts : %d / %d (%.0f%%)",
                                     count, total, 100*count/total if total else 0)
                batch_posts, batch_pub, batch_dup, batch_narr = [], [], [], []
                batch_ids, batch_proj_post, batch_proj_acc = [], [], []

        if batch_posts:
            self._flush_posts_batch(batch_posts, batch_pub, batch_dup, batch_narr, [])
            if self._sync_projects and not self.dry_run and self.neo4j:
                self.neo4j.upsert_project_batch(batch_proj_post, batch_proj_acc)
            if not self.dry_run:
                self.db.posts.update_many(
                    {"_id": {"$in": batch_ids}},
                    patch_post_sync(neo4j=True),
                )
            count += len(batch_posts)

        self.logger.info("Backfill posts : %d synchronisés → Neo4j", count)

    def _backfill_media(self) -> None:
        """Synchronise les documents media existants (sync.neo4j=False) vers Neo4j."""
        self.logger.info("BACKFILL media…")
        batch     = []
        batch_ids = []
        count     = 0
        interval  = self._backfill_log_interval

        filt = {"sync.neo4j": False}
        if self.projets:
            filt["source.project"] = {"$in": self.projets}
        total = self.db.media.count_documents(filt)
        self.logger.info("  %d média(s) à synchroniser", total)

        for doc in self.db.media.find(filt):
            node = self._media_to_node(doc)
            if not node:
                continue
            batch.append(node)
            batch_ids.append(doc["_id"])

            if len(batch) >= self._batch_size:
                self._flush_media_batch(batch)
                if not self.dry_run:
                    self.db.media.update_many(
                        {"_id": {"$in": batch_ids}},
                        patch_media_sync(neo4j=True),
                    )
                count += len(batch)
                if interval > 0 and count % interval == 0:
                    self.logger.info("  ↳ media : %d / %d (%.0f%%)",
                                     count, total, 100*count/total if total else 0)
                batch, batch_ids = [], []

        if batch:
            self._flush_media_batch(batch)
            if not self.dry_run:
                self.db.media.update_many(
                    {"_id": {"$in": batch_ids}},
                    patch_media_sync(neo4j=True),
                )
            count += len(batch)

        self.logger.info("Backfill media : %d synchronisés → Neo4j", count)

    def _flush_media_batch(self, batch: list[dict]) -> None:
        if self.dry_run or not self.neo4j or not batch:
            return
        self.neo4j.upsert_media_batch(batch)
        # Liens Post→Media
        for node in batch:
            media_mongo_id = node["mongo_id"]
            post_ids = node.get("post_ids") or []

            # Fallback : si reuse.post_ids est vide, chercher les posts
            # qui référencent ce média dans posts.media[].media_id
            if not post_ids:
                from bson import ObjectId
                try:
                    oid = ObjectId(media_mongo_id)
                    post_ids = [
                        str(p["_id"])
                        for p in self.db.posts.find(
                            {"media.media_id": oid}, {"_id": 1}
                        )
                    ]
                except Exception as exc:
                    self.logger.debug(
                        "Fallback post_ids pour media %s : %s", media_mongo_id, exc
                    )

            for post_id in post_ids:
                try:
                    self.neo4j.link_post_media(str(post_id), media_mongo_id)
                except Exception as exc:
                    self.logger.debug(
                        "link_post_media %s → %s : %s", post_id, media_mongo_id, exc
                    )

    # ------------------------------------------------------------------
    # Boucle Change Streams
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._heartbeat.start()
        self.logger.info("Écoute Change Streams sur : %s", ", ".join(self._collections))

        retries = 0
        while self._running and retries < self._max_retries:
            try:
                self._watch_loop()
                retries = 0
            except PyMongoError as exc:
                retries += 1
                self.logger.error("Erreur MongoDB (%d/%d) : %s — retry dans %ds",
                                  retries, self._max_retries, exc, self._retry_delay)
                time.sleep(self._retry_delay)
            except Exception as exc:
                self.logger.critical("Erreur inattendue : %s", exc, exc_info=True)
                break

        self._heartbeat.stop()

        # Attendre la fin d'une éventuelle passe de détection en cours
        if self._detection_thread and self._detection_thread.is_alive():
            self.logger.info("Attente fin de la passe de détection…")
            self._detection_thread.join(timeout=120)

        if self.neo4j:
            self.neo4j.close()

        try:
            self.db.jobs.update_one(
                {"type": Heartbeat.HEARTBEAT_TYPE, "worker_id": self._worker_id},
                {"$set": {"status": "stopped", "last_seen_at": datetime.now(timezone.utc)}},
            )
        except Exception:
            pass

        self.logger.info("=== Worker réseau arrêté ===")

    def _watch_loop(self) -> None:
        match_filter: dict = {
            "operationType": {"$in": ["insert", "update", "replace"]},
            "$or": [
                {"fullDocument.sync.neo4j": False},
                {"updateDescription.updatedFields.sync.neo4j": False},
            ],
            "ns.coll": {"$in": self._collections},
        }

        # En mode --projet : filtrer par source.project dans le Change Stream
        if self.projets:
            match_filter["fullDocument.source.project"] = {"$in": self.projets}

        pipeline = [{"$match": match_filter}]

        self.logger.info("Change Stream ouvert")
        self.logger.info(
            "En attente de documents… (détection auto dans %ds d'inactivité)",
            self._detection_idle_seconds,
        )

        # Compteur de try_next() vides consécutifs.
        # max_await_time_ms=1000 → chaque appel attend 1s max
        # Après idle_seconds None consécutifs → détection déclenchée UNE SEULE FOIS.
        # Le flag se reset uniquement quand un nouveau document arrive,
        # ce qui garantit qu'on ne relance pas la détection à vide en boucle.
        idle_count                = 0
        idle_trigger              = self._detection_idle_seconds
        _detection_done_this_idle = False
        _idle_log_interval        = 10   # log "en attente" toutes les 10s

        with self.db.watch(pipeline, full_document="updateLookup", max_await_time_ms=1000) as stream:
            while self._running:
                change = stream.try_next()

                if change is None:
                    idle_count += 1

                    # Log périodique d'attente (toutes les 10s) avant le déclenchement
                    if (
                        not _detection_done_this_idle
                        and idle_count < idle_trigger
                        and idle_count % _idle_log_interval == 0
                    ):
                        remaining = idle_trigger - idle_count
                        self.logger.debug(
                            "⏳ En attente — détection dans %ds", remaining
                        )

                    if (
                        not self.skip_detection
                        and not _detection_done_this_idle
                        and idle_count >= idle_trigger
                    ):
                        _detection_done_this_idle = True
                        self._trigger_detection()
                    continue

                # Un document est arrivé → reset complet
                idle_count                = 0
                _detection_done_this_idle = False

                doc        = change.get("fullDocument")
                collection = change["ns"]["coll"]
                if doc is None:
                    continue
                if doc.get("sync", {}).get("neo4j") is not False:
                    continue
                self._process_document(doc, collection)

    # ------------------------------------------------------------------
    # Déclenchement de la détection (v2)
    # ------------------------------------------------------------------

    def _trigger_detection(self) -> None:
        """
        Lance une passe de campaign_detector en arrière-plan.

        Règles :
          - Si une passe est déjà en cours → skip (on ne cumule pas)
          - En dry_run → skip (pas d'écriture Neo4j de toute façon)
          - Lance dans un thread daemon pour ne pas bloquer le Change Stream
        """
        if self.dry_run:
            return

        # Tenter d'acquérir le verrou sans bloquer
        if not self._detection_lock.acquire(blocking=False):
            self.logger.debug(
                "🔍 Détection auto : passe déjà en cours — skip"
            )
            return

        # Vérifier que le thread précédent est bien terminé
        if self._detection_thread and self._detection_thread.is_alive():
            self._detection_lock.release()
            self.logger.debug(
                "🔍 Détection auto : thread précédent encore actif — skip"
            )
            return

        self.logger.info("🔍 Détection auto : file vide depuis %ds — lancement en arrière-plan…",
                         self._detection_idle_seconds)

        self._detection_thread = threading.Thread(
            target  = self._run_detection,
            name    = "campaign-detector",
            daemon  = True,
        )
        self._detection_thread.start()

    def _run_detection(self) -> None:
        """
        Exécute campaign_detector dans le thread de détection.
        Libère le verrou à la fin, qu'il y ait une erreur ou non.
        """
        try:
            from campaign_detector import CampaignDetector

            self.logger.info("🔍 Détection — analyse des signaux en cours…")

            # Paramètres GDS depuis le cfg si disponibles
            skip_gds = self.neo4j is None
            min_score = 0.30
            if "detection" in self.cfg:
                min_score = self.cfg["detection"].getfloat("min_score", fallback=0.30)
                skip_gds  = self.cfg["detection"].getboolean("skip_gds", fallback=skip_gds)

            detector = CampaignDetector(
                db           = self.db,
                neo4j_client = self.neo4j if not skip_gds else None,
                dry_run      = False,
                skip_gds     = skip_gds,
            )
            campaigns = detector.run(min_score=min_score)
            self.logger.info(
                "🔍 Détection terminée ✓ — %d campagne(s) détectée(s) | "
                "worker réseau en écoute",
                len(campaigns),
            )

        except ImportError:
            self.logger.warning(
                "🔍 Détection auto : campaign_detector.py introuvable — "
                "vérifier qu'il est dans le même dossier"
            )
        except Exception as exc:
            self.logger.error("🔍 Détection auto : erreur inattendue : %s", exc, exc_info=True)
        finally:
            self._detection_lock.release()

    # ------------------------------------------------------------------
    # Traitement unitaire
    # ------------------------------------------------------------------

    def _process_document(self, doc: dict, collection_name: str) -> None:
        doc_id = doc["_id"]
        self.logger.info("[%s] %s | → Neo4j", collection_name, doc_id)

        if self.dry_run:
            self.logger.debug("[DRY RUN] Pas d'écriture Neo4j")
            self._heartbeat.synced += 1
            return

        try:
            if collection_name == "accounts":
                self._sync_account(doc)
            elif collection_name == "posts":
                self._sync_post(doc)
            elif collection_name == "comments":
                self._sync_comment(doc)
            elif collection_name == "media":
                self._sync_media_doc(doc)

            if collection_name == "posts":
                patch = patch_post_sync(neo4j=True)
            elif collection_name == "comments":
                patch = patch_comment_sync(neo4j=True)
            elif collection_name == "media":
                patch = patch_media_sync(neo4j=True)
            else:
                patch = patch_post_sync(neo4j=True)  # accounts : pas de patch sync

            if collection_name != "accounts":
                self.db[collection_name].update_one({"_id": doc_id}, patch)
            self.logger.debug("[%s] %s | ✓ sync.neo4j=True", collection_name, doc_id)
            self._heartbeat.synced += 1

        except Exception as exc:
            self.logger.error("[%s] %s | erreur sync : %s", collection_name, doc_id, exc)
            self._heartbeat.errors += 1

    def _sync_account(self, doc: dict) -> None:
        node = self._account_to_node(doc)
        if node and self.neo4j:
            self.neo4j.upsert_account(node)
            if not self.dry_run:
                self.db.accounts.update_one(
                    {"_id": doc["_id"]},
                    patch_account_sync(neo4j=True),
                )

    def _sync_post(self, doc: dict) -> None:
        node = self._post_to_node(doc)
        if not node or not self.neo4j:
            return

        self.neo4j.upsert_post(node)

        account_id = str(doc.get("account_id", "")) if doc.get("account_id") else None
        if account_id and self._sync_posts:
            self.neo4j.link_account_post(account_id, str(doc["_id"]), "A_PUBLIÉ")

        dup_id = doc.get("nlp", {}).get("is_duplicate_of")
        sim    = doc.get("nlp", {}).get("similarity_score")
        if dup_id and self._sync_duplicates:
            self.neo4j.link_post_duplicate(str(doc["_id"]), str(dup_id), sim or 0.0)
            self.logger.info("[posts] %s | EST_DOUBLON_DE → %s", doc["_id"], dup_id)

        narr_id = doc.get("nlp", {}).get("narrative_id")
        if narr_id and self._sync_narratives:
            narr = self.db.narratives.find_one({"_id": narr_id})
            if narr:
                self.neo4j.upsert_narrative({
                    "mongo_id":   str(narr_id),
                    "label":      narr.get("label", ""),
                    "keywords":   narr.get("keywords", []),
                    "post_count": narr.get("stats", {}).get("post_count", 0),
                    "updated_at": str(narr.get("updated_at", "")),
                })
                self.neo4j.link_post_narrative(str(doc["_id"]), str(narr_id))

        fwd_count = doc.get("platform_specific", {}).get("forward_count", 0) or 0
        if (fwd_count >= self._forward_min
                and doc.get("platform") == "telegram"
                and self._sync_forwards
                and account_id):
            self.neo4j.link_account_post(
                account_id, str(doc["_id"]), "A_FORWARDÉ",
                props={"count": fwd_count}
            )
            self.logger.info("[posts] %s | A_FORWARDÉ (count=%d)", doc["_id"], fwd_count)

        # --- Nœuds :Hashtag  [v3] ---
        if self._sync_hashtags:
            hashtags = [t for t in (doc.get("text", {}).get("hashtags") or [])
                        if isinstance(t, str) and t.strip()]
            if hashtags:
                self.neo4j.upsert_hashtags_for_post(str(doc["_id"]), hashtags)
                self.logger.debug(
                    "[posts] %s | HAS_HASHTAG → %s", doc["_id"], hashtags
                )

        # --- Nœud :Deepfake  [v3] ---
        if self._sync_deepfake_nodes:
            dfk = doc.get("deepfake", {})
            pred = dfk.get("prediction") or ""
            score = dfk.get("final_score")
            if pred and pred != "likely_real" and score is not None:
                self.neo4j.upsert_deepfake_node(
                    post_id   = str(doc["_id"]),
                    pred_type = pred,
                    score     = score,
                )
                self.logger.debug(
                    "[posts] %s | IS_DEEPFAKE → %s (%.3f)", doc["_id"], pred, score
                )

        # --- Nœud :Project  [v5] ---
        if self._sync_projects:
            proj = (doc.get("source") or {}).get("project") or ""
            if proj:
                self.neo4j.upsert_project(proj)
                self.neo4j.link_project_post(proj, str(doc["_id"]))
                if account_id:
                    self.neo4j.link_project_account(proj, account_id)
                self.logger.debug("[posts] %s | CONTIENT ← Project %s", doc["_id"], proj)

    def _sync_comment(self, doc: dict) -> None:
        if not self.neo4j:
            return
        account_id = str(doc.get("account_id", "")) if doc.get("account_id") else None
        post_id    = str(doc.get("post_id", ""))    if doc.get("post_id")    else None
        if account_id and post_id:
            self.neo4j.link_account_post(account_id, post_id, "A_COMMENTÉ")

    def _sync_media_doc(self, doc: dict) -> None:
        """Sync unitaire d'un document media vers Neo4j.  [v4]"""
        if not self.neo4j or not self._sync_media:
            return
        node = self._media_to_node(doc)
        if not node:
            return

        self.neo4j.upsert_media(node)

        post_ids = node.get("post_ids") or []

        # Fallback : chercher dans posts.media[].media_id si reuse.post_ids est vide
        if not post_ids:
            from bson import ObjectId
            try:
                oid = ObjectId(node["mongo_id"])
                post_ids = [
                    str(p["_id"])
                    for p in self.db.posts.find(
                        {"media.media_id": oid}, {"_id": 1}
                    )
                ]
            except Exception:
                pass

        for post_id in post_ids:
            try:
                self.neo4j.link_post_media(str(post_id), node["mongo_id"])
                self.logger.debug(
                    "[media] %s | A_MEDIA ← post %s", node["mongo_id"], post_id
                )
            except Exception as exc:
                self.logger.debug(
                    "[media] %s | link_post_media échoué : %s", node["mongo_id"], exc
                )

    # ------------------------------------------------------------------
    # Helpers de conversion MongoDB → Neo4j
    # ------------------------------------------------------------------

    def _account_to_node(self, doc: dict) -> dict | None:
        if not doc.get("platform") or not doc.get("platform_id"):
            return None
        profile  = doc.get("profile", {})
        stats    = doc.get("stats", {})
        ps       = doc.get("platform_specific", {})
        analysis = doc.get("analysis", {})
        signals  = analysis.get("bot_signals", {})

        return {
            "mongo_id":     str(doc["_id"]),
            "platform":     doc.get("platform", ""),
            "platform_id":  doc.get("platform_id", ""),
            "username":     doc.get("username") or "",
            "display_name": doc.get("display_name") or "",
            "url":          doc.get("url") or "",
            "bio":          (profile.get("bio") or "")[:300],
            "location":     profile.get("location") or "",
            "website":      profile.get("website") or "",
            "avatar_url":   profile.get("avatar_url") or "",
            "verified":     profile.get("verified", False),
            "created_at":   str(profile.get("created_at") or ""),
            "language":     profile.get("language") or "",
            "followers":    stats.get("followers_count", 0) or 0,
            "following":    stats.get("following_count", 0) or 0,
            "posts_count":  stats.get("posts_count", 0) or 0,
            "likes_count":  stats.get("likes_count", 0) or 0,
            "tg_is_channel":   ps.get("is_channel", False),
            "tg_is_group":     ps.get("is_group", False),
            "tg_is_bot":       ps.get("is_bot", False),
            "tg_member_count": ps.get("member_count", 0) or 0,
            "tg_invite_link":  ps.get("invite_link") or "",
            "tw_verified_type":    ps.get("verified_type") or "",
            "tw_is_blue_verified": ps.get("is_blue_verified", False),
            "tw_protected":        ps.get("protected", False),
            "tw_listed_count":     ps.get("listed_count", 0) or 0,
            "tt_region":       ps.get("region") or "",
            "tt_is_creator":   ps.get("is_creator", False),
            "ig_is_business":  ps.get("is_business", False),
            "ig_is_private":   ps.get("is_private", False),
            "bot_score":           analysis.get("bot_score"),
            "bot_post_regularity": signals.get("post_regularity"),
            "bot_follower_ratio":  signals.get("follower_ratio"),
            "bot_avg_engagement":  signals.get("avg_engagement"),
            "bot_account_age":     signals.get("account_age_days"),
            "bot_burst_score":     signals.get("post_burst_score"),
            "language_detected":   analysis.get("language_detected") or "",
            "is_confirmed_bot":    doc.get("flags", {}).get("is_confirmed_bot", False),
            "is_suspicious":       doc.get("flags", {}).get("is_suspicious", False),
            "scraped_at": str(doc.get("scraped_at") or ""),
            "updated_at": str(doc.get("updated_at") or ""),
        }

    def _post_to_node(self, doc: dict) -> dict | None:
        if not doc.get("platform"):
            return None
        nlp  = doc.get("nlp", {})
        dfk  = doc.get("deepfake", {})
        text = doc.get("text", {})
        eng  = doc.get("engagement", {})
        ctx  = doc.get("context", {})
        ps   = doc.get("platform_specific", {})

        content  = (text.get("content") or "")[:500]
        hashtags = [t for t in (text.get("hashtags") or []) if isinstance(t, str)]
        mentions = [m for m in (text.get("mentions") or []) if isinstance(m, str)]
        urls     = [u for u in (text.get("urls") or [])     if isinstance(u, str)]

        # Signaux scrapper [v4]
        sig = doc.get("scrapper_signals") or {}
        ps  = doc.get("platform_specific") or {}

        # cover_url et music_author : présents dans platform_specific selon la plateforme
        cover_url    = ps.get("cover_url") or ""
        music_author = ps.get("music_author") or ""

        return {
            "mongo_id":     str(doc["_id"]),
            "platform":     doc.get("platform", ""),
            "platform_id":  doc.get("platform_id", ""),
            "url":          doc.get("url") or "",
            "text":         content,
            "language":     text.get("language") or "",
            "hashtags":     hashtags,
            "mentions":     mentions,
            "urls":         urls,
            "is_truncated": text.get("is_truncated", False),
            "published_at":  str(ctx.get("published_at") or ""),
            "is_reply_to":   str(ctx.get("is_reply_to") or ""),
            "is_repost_of":  str(ctx.get("is_repost_of") or ""),
            "reply_count":   ctx.get("reply_count", 0) or 0,
            "thread_id":     str(ctx.get("thread_id") or ""),
            "likes":         eng.get("likes", 0) or 0,
            "shares":        eng.get("shares", 0) or 0,
            "comments":      eng.get("comments", 0) or 0,
            "views":         eng.get("views", 0) or 0,
            "saves":         eng.get("saves", 0) or 0,
            "tg_forward_from":  str(ps.get("forward_from") or ""),
            "tg_forward_count": ps.get("forward_count", 0) or 0,
            "tg_views":         ps.get("views", 0) or 0,
            "tg_is_forwarded":  ps.get("is_forwarded", False),
            "tg_channel_id":    str(ps.get("channel_id") or ""),
            "tg_message_id":    str(ps.get("message_id") or ""),
            "tw_retweet_count":  ps.get("retweet_count", 0) or 0,
            "tw_quote_count":    ps.get("quote_count", 0) or 0,
            "tw_is_retweet":     ps.get("is_retweet", False),
            "tw_verified_type":  str(ps.get("verified_type") or ""),
            "is_video":          ps.get("is_video", False),
            "video_duration":    ps.get("video_duration", 0) or 0,
            "play_count":        ps.get("play_count", 0) or 0,
            "sentiment_label":   nlp.get("sentiment", {}).get("label") or "",
            "sentiment_score":   nlp.get("sentiment", {}).get("score"),
            "embedding_model":   nlp.get("embedding_model") or "",
            "topics":            nlp.get("topics") or [],
            "narrative_id":      str(nlp.get("narrative_id")) if nlp.get("narrative_id") else "",
            "is_duplicate":      nlp.get("is_duplicate_of") is not None,
            "similarity_score":  nlp.get("similarity_score"),
            "deepfake_score":    dfk.get("final_score"),
            "deepfake_pred":     dfk.get("prediction") or "",
            "is_synthetic":      dfk.get("prediction") == "synthetic",
            "has_media":         dfk.get("has_media", False),
            "artifact_score":    dfk.get("artifact_score"),
            # Engagement explicite [v3] — noms lisibles dans Neo4j
            "like_count":        eng.get("likes", 0) or 0,
            "comment_count":     eng.get("comments", 0) or 0,
            "share_count":       eng.get("shares", 0) or 0,
            "view_count":        eng.get("views", 0) or 0,
            # Signaux scrapper [v4]
            "influence_score":   sig.get("influence_score"),
            "is_bot_suspected":  sig.get("is_bot_suspected", False),
            "cover_url":         cover_url,
            "music_author":      music_author,
            # Source projet/scan [v3]
            "source_project":    (doc.get("source") or {}).get("project") or "",
            "source_scan":       (doc.get("source") or {}).get("scan") or "",
            "source_user":       (doc.get("source") or {}).get("user") or "",
            "scraped_at":  str(doc.get("scraped_at") or ""),
            "updated_at":  str(doc.get("updated_at") or ""),
        }

    def _media_to_node(self, doc: dict) -> dict | None:
        """Convertit un document media MongoDB en dict pour Neo4j.  [v4]"""
        if not doc.get("_id"):
            return None
        dfk  = doc.get("deepfake", {})
        meta = doc.get("metadata", {})
        reuse = doc.get("reuse", {})

        # post_ids : liste des ObjectId des posts qui utilisent ce média
        # stockés dans reuse.post_ids — on les convertit en str pour Neo4j
        post_ids = [str(pid) for pid in (reuse.get("post_ids") or [])]

        return {
            "mongo_id":      str(doc["_id"]),
            "type":          doc.get("type") or "",           # video | image | gif | audio
            "url_local":     doc.get("url_local") or "",
            "url_original":  doc.get("url_original") or "",
            "platform":      (doc.get("source") or {}).get("project") or "",
            "deepfake_score": dfk.get("final_score"),
            "deepfake_pred":  dfk.get("prediction") or "",
            "reuse_count":   reuse.get("seen_count", 1) or 1,
            "width":         meta.get("width"),
            "height":        meta.get("height"),
            "duration_sec":  meta.get("duration_sec"),
            "source_project": (doc.get("source") or {}).get("project") or "",
            "source_scan":    (doc.get("source") or {}).get("scan") or "",
            # post_ids transmis au flush pour créer les relations A_MEDIA
            # (non stocké sur le nœud Neo4j, utilisé uniquement en transit)
            "post_ids":      post_ids,
            "updated_at":    str(doc.get("updated_at") or ""),
        }

    # ------------------------------------------------------------------
    # Flush batches
    # ------------------------------------------------------------------

    def _flush_accounts(self, batch: list[dict]) -> None:
        if not self.dry_run and self.neo4j and batch:
            self.neo4j.upsert_accounts_batch(batch)

    def _flush_posts_batch(
        self,
        posts: list[dict],
        pub:   list[dict],
        dup:   list[dict],
        narr:  list[dict],
        ids:   list,
    ) -> None:
        if self.dry_run or not self.neo4j:
            return
        self.neo4j.upsert_posts_batch(posts)
        if pub:
            self.neo4j.create_relations_batch(pub, "A_PUBLIÉ", "Account", "Post")
        if dup and self._sync_duplicates:
            self.neo4j.create_relations_batch(dup, "EST_DOUBLON_DE", "Post", "Post")
        if narr and self._sync_narratives:
            self.neo4j.create_relations_batch(narr, "APPARTIENT_À", "Post", "Narrative")

        # Hashtags et deepfake nodes — traitement unitaire par post [v3]
        if self._sync_hashtags or self._sync_deepfake_nodes:
            for node in posts:
                post_id = node["mongo_id"]
                if self._sync_hashtags:
                    tags = [t for t in (node.get("hashtags") or []) if isinstance(t, str) and t.strip()]
                    if tags:
                        self.neo4j.upsert_hashtags_for_post(post_id, tags)
                if self._sync_deepfake_nodes:
                    pred  = node.get("deepfake_pred") or ""
                    score = node.get("deepfake_score")
                    if pred and pred != "likely_real" and score is not None:
                        self.neo4j.upsert_deepfake_node(post_id, pred, score)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Worker réseau — ETL MongoDB → Neo4j via Change Streams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Mode stream continu (défaut)
  python network_worker.py --backfill

  # Mode projet : purge Neo4j + réinjection filtrée
  python network_worker.py --projet ProJet0 -p TIKTOK_crypto_2026-03-25

  # Mode projet sans purge (ajout d'un nouveau projet)
  python network_worker.py --projet ProJet0 --add

  # Simulation sans écriture
  python network_worker.py --projet ProJet0 --dry-run
        """,
    )
    parser.add_argument("--config", "-c", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--backfill", action="store_true",
                        help="Synchroniser les documents existants avant de démarrer le stream")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simuler sans écrire dans Neo4j")
    parser.add_argument("--skip-detection", action="store_true",
                        help="Désactiver la détection automatique de campagnes")
    parser.add_argument(
        "--projet", "-p",
        dest="projets",
        metavar="NOM_PROJET",
        action="append",
        default=[],
        help=(
            "Nom d'un projet à traiter (répétable : -p ProJet0 -p ProJet1). "
            "Active le mode projet : purge Neo4j + reset sync.neo4j + réinjection filtrée. "
            "Sans --add, effectue une purge complète de Neo4j avant l'import."
        ),
    )
    parser.add_argument(
        "--add",
        action="store_true",
        help=(
            "Utilisé avec --projet : ajoute les projets dans Neo4j SANS purger la base. "
            "Utile pour intégrer un nouveau projet sans repartir de zéro."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    cfg    = load_config(args.config)
    logger = setup_logging(cfg)
    logger.info("Configuration chargée : %s", args.config)

    worker = NetworkWorker(
        cfg            = cfg,
        dry_run        = args.dry_run,
        skip_detection = args.skip_detection,
        projets        = args.projets,
        add_mode       = args.add,
    )

    if args.projets:
        # ── Mode projet ──────────────────────────────────────────────────
        # 1. Purge Neo4j + reset MongoDB (sauf si --add)
        if not args.add:
            worker.purge_and_reset()
        else:
            logger.info("📂 Projet --add : purge ignorée, injection en complément")

        # 2. Backfill filtré par projet (one-shot, pas de Change Stream)
        worker.backfill()

        logger.info("=== Mode --projet terminé. Lancement du stream continu. ===")

    elif args.backfill:
        # ── Mode stream avec backfill initial ────────────────────────────
        worker.backfill()

    # 3. Change Stream continu (dans tous les cas)
    worker.run()
