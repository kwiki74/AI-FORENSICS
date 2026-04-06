"""worker_import.py  v1.4
========================

Worker d'ingestion MongoDB — Projet influence_detection
Lit les JSON produits par le scrapper et les insère dans MongoDB via schema.py.

Deux modes de fonctionnement :
  - Mode inbox   : surveille storage/inbox/, déplace les fichiers (flux scrapper)
  - Mode source  : scanne un dossier DOSSIER_INPUT structuré par projets/scans

Flux mode inbox :
    inbox/  ──(move)──►  processing/  ──(insert)──►  MongoDB
                                       ──(move)──►  done/

Flux mode source (arborescence DOSSIER_INPUT) :
    DOSSIER_INPUT/
      Projet1/
        converted_NomReseau_motclé_2026-03-17/   ← dossiers "converted"
          User1/
            converted_post1.json                  ← JSON importés
            converted_post2.json
          User2/
            converted_post3.json
        NomReseau_motclé_2026-03-17/             ← dossiers "raw" symétriques
          User1/
            post1.jpg                             ← médias référencés
            post1.mp4
          User2/
            post3.mp4

    Pour chaque converted_*.json, le worker :
      1. Importe le post en MongoDB (account + post + comments)
      2. Résout le dossier raw symétrique (retire le préfixe "converted_")
      3. Cherche tous les médias du même stem (jpg, mp4, mp3, gif, webp, png...)
      4. Crée un document `media` par fichier trouvé
      5. Attache chaque référence au post via patch_post_media()
      6. Crée un job `deepfake_analysis` par média dans la collection `jobs`

Usage :
    python worker_import.py                                    # mode inbox, surveillance continue
    python worker_import.py --once                             # mode inbox, traite et quitte
    python worker_import.py --source /chemin/DOSSIER_INPUT     # mode source, import one-shot
    python worker_import.py --source /chemin --watch           # mode source, surveillance continue
    python worker_import.py --source /chemin --dry-run         # validation sans écriture
    python worker_import.py --config /chemin/mon.cfg           # fichier de config personnalisé

Configuration :
    Par défaut, le script cherche worker_import.cfg dans son propre dossier.
    Les arguments CLI ont priorité sur le fichier de config, qui a priorité sur .env.

Dépendances :
    pip install pymongo python-dotenv

Shell (v1.4) :
    - Couleurs ANSI natives dans le terminal (pas de dépendance externe).
      Les fichiers de log restent en texte brut (sans codes ANSI).
    - Heartbeat en mode surveillance continue : ligne périodique (stderr)
      toutes les `heartbeat_interval` secondes confirmant que le worker tourne.
    - Résumés de fin de run colorés (OK en vert, erreurs en rouge, médias en cyan).
    - Détection automatique TTY : couleurs désactivées si stdout est redirigé
      (pipe, fichier, cron, nohup).

Gestion des médias (v1.2+) :
    - Un post peut avoir plusieurs médias (jpg, mp4, mp3, gif, webp, png, etc.)
    - Les médias ne sont PAS copiés — seul le chemin absolu est stocké en base
    - Un document `media` est créé par fichier trouvé (deepfake.status = "pending")
    - Un job `deepfake_analysis` est créé par média pour detect_ai_pipeline-v4.0.2
    - Les hashes (md5, perceptual) seront calculés par le pipeline lors de l'analyse

Notes sur le dataset actuel (2026-03-18) :
    - post_url absent dans tous les JSON → reconstruit automatiquement
    - media_type absent → déduit de video_url/cover
    - X/Twitter : id = nom du compte (bug binôme) → log WARNING, import quand même
    - scrapper_id = "converter_01" (conversion depuis données brutes)
"""

from __future__ import annotations

import argparse
import configparser
import json
import logging
import logging.handlers
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pymongo.errors import PyMongoError

# --- Chargement .env ---
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# --- Résolution schema.py — ordre de priorité :
#     1. ~/AI-FORENSICS/SCHEMA/  (../../SCHEMA/ depuis WORKER/IMPORT/)
#     2. Dossier parent du worker (WORKER/)
#     3. Dossier courant          (WORKER/IMPORT/)
_HERE = Path(__file__).resolve().parent
_SCHEMA_CANDIDATES = [
    _HERE.parent.parent / "SCHEMA",
    _HERE.parent,
    _HERE,
]
for _schema_dir in _SCHEMA_CANDIDATES:
    if (_schema_dir / "schema.py").exists():
        sys.path.insert(0, str(_schema_dir))
        break

# --- Import du schéma projet ---
try:
    from schema import (
        get_db, new_account, new_post, new_comment, new_media, new_job,
        patch_post_media, patch_media_reuse, patch_media_sync, PLATFORMS,
    )
except ImportError as e:
    print(f"[ERREUR] Impossible d'importer schema.py : {e}")
    print(f"         Chemins cherchés : {[str(p) for p in _SCHEMA_CANDIDATES]}")
    sys.exit(1)


# ===========================================================================
# Configuration
# ===========================================================================

DEFAULT_CFG        = Path(__file__).resolve().parent / "worker_import.cfg"
DEFAULT_STORAGE    = Path("storage")
POLL_INTERVAL      = 5
HEARTBEAT_INTERVAL = 30   # secondes entre deux lignes heartbeat en mode watch
BATCH_SIZE         = 50

PLATFORM_ALIASES = {
    "x":         "twitter",
    "twitter":   "twitter",
    "tiktok":    "tiktok",
    "instagram": "instagram",
    "telegram":  "telegram",
}

POST_URL_TEMPLATES = {
    "twitter":   "https://twitter.com/{author_id}/status/{post_id}",
    "tiktok":    "https://www.tiktok.com/@{author_unique}/video/{post_id}",
    "instagram": "https://www.instagram.com/p/{post_id}/",
    "telegram":  "https://t.me/s/{channel_id}",
}

# Extensions média reconnues, par type
MEDIA_EXTENSIONS: dict[str, str] = {
    ".jpg":  "image",
    ".jpeg": "image",
    ".png":  "image",
    ".webp": "image",
    ".bmp":  "image",
    ".tiff": "image",
    ".tif":  "image",
    ".mp4":  "video",
    ".mov":  "video",
    ".avi":  "video",
    ".mkv":  "video",
    ".webm": "video",
    ".flv":  "video",
    ".mp3":  "audio",
    ".m4a":  "audio",
    ".ogg":  "audio",
    ".wav":  "audio",
    ".aac":  "audio",
    ".gif":  "gif",
}


# ===========================================================================
# Couleurs ANSI — terminal uniquement, jamais dans les fichiers de log
# ===========================================================================
#
# _TTY = True  → stdout est un terminal interactif → couleurs activées
# _TTY = False → redirection (fichier, pipe, cron…) → codes ANSI vides
#
# Le _ColorConsoleFormatter applique les couleurs uniquement sur le handler
# console ; les FileHandlers utilisent un Formatter standard sans ANSI.

_TTY = sys.stdout.isatty()


class C:
    """Constantes de couleur ANSI. Chaînes vides automatiquement hors TTY."""
    RESET   = "\033[0m"  if _TTY else ""
    BOLD    = "\033[1m"  if _TTY else ""
    DIM     = "\033[2m"  if _TTY else ""
    RED     = "\033[31m" if _TTY else ""
    GREEN   = "\033[32m" if _TTY else ""
    YELLOW  = "\033[33m" if _TTY else ""
    BLUE    = "\033[34m" if _TTY else ""
    MAGENTA = "\033[35m" if _TTY else ""
    CYAN    = "\033[36m" if _TTY else ""
    WHITE   = "\033[37m" if _TTY else ""
    # Raccourcis sémantiques
    OK      = "\033[32m" if _TTY else ""   # vert    — succès
    WARN    = "\033[33m" if _TTY else ""   # jaune   — avertissement
    ERR     = "\033[31m" if _TTY else ""   # rouge   — erreur
    INFO    = "\033[36m" if _TTY else ""   # cyan    — information neutre
    HEART   = "\033[35m" if _TTY else ""   # magenta — heartbeat


def _c(color: str, text: str) -> str:
    """Applique une couleur ANSI à `text` et réinitialise après."""
    if not color:
        return text
    return f"{color}{text}{C.RESET}"


# Mapping niveau de log → couleur pour le formatter console
_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG:    C.DIM,
    logging.INFO:     C.WHITE,
    logging.WARNING:  C.WARN,
    logging.ERROR:    C.ERR,
    logging.CRITICAL: (C.BOLD + C.ERR) if _TTY else "",
}


class _ColorConsoleFormatter(logging.Formatter):
    """
    Formatter console avec niveaux colorés :
      DEBUG    → grisé
      INFO     → blanc normal
      WARNING  → jaune
      ERROR    → rouge
      CRITICAL → rouge gras
    Le timestamp est atténué (DIM). Les fichiers de log n'utilisent PAS ce formatter.
    """
    _DATEFMT = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        color   = _LEVEL_COLORS.get(record.levelno, "")
        reset   = C.RESET
        ts      = self.formatTime(record, self._DATEFMT)
        level   = f"{record.levelname:<8}"
        message = record.getMessage()
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)
        if _TTY:
            return (
                f"{C.DIM}{ts}{reset} "
                f"{color}[{level}]{reset} "
                f"{color}{message}{reset}"
            )
        return f"{ts} [{level}] {message}"


# ===========================================================================
# Chargement du fichier de config
# ===========================================================================

def load_config(cfg_path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    if cfg_path.exists():
        cfg.read(cfg_path, encoding="utf-8")
    return cfg


def cfg_get(cfg: configparser.ConfigParser, section: str, key: str, fallback=None):
    """Lit une valeur dans le cfg, avec fallback."""
    try:
        val = cfg.get(section, key)
        return val if val.strip() != "" else fallback
    except (configparser.NoSectionError, configparser.NoOptionError):
        return fallback


# ===========================================================================
# Logging — 3 fichiers dédiés + console colorée
# ===========================================================================
#
#  Console           — couleurs ANSI, niveau configurable
#  worker_import.log — supervision : résumé par run + résumé par plateforme
#                      + une ligne par fichier NOK (texte brut, sans ANSI)
#  errors.log        — uniquement les erreurs bloquantes (texte brut)
#  warnings.log      — anomalies non bloquantes dédupliquées (texte brut)
#
# Rotation quotidienne, conservation 30 jours.

class _LevelFilter(logging.Filter):
    """Filtre qui n'accepte qu'une plage de niveaux."""
    def __init__(self, min_level: int, max_level: int):
        super().__init__()
        self.min_level = min_level
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return self.min_level <= record.levelno <= self.max_level


class _StripAnsiFormatter(logging.Formatter):
    """Formatter fichier qui supprime les codes ANSI résiduels."""
    import re as _re
    _ANSI_RE = _re.compile(r"\033\[[0-9;]*m")

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        return self._ANSI_RE.sub("", formatted)


def _make_rotating_handler(path: Path, min_lvl: int, max_lvl: int,
                            fmt: str) -> logging.Handler:
    """Handler fichier rotatif avec formatter sans ANSI."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h = logging.handlers.TimedRotatingFileHandler(
        path, when="midnight", backupCount=30, encoding="utf-8"
    )
    h.setFormatter(_StripAnsiFormatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    h.addFilter(_LevelFilter(min_lvl, max_lvl))
    h.setLevel(min_lvl)
    return h


def setup_logging(
    level_str: str        = "INFO",
    log_dir:   Optional[str] = None,
    verbose:   bool       = False,
) -> tuple[logging.Logger, logging.Logger, logging.Logger]:
    """
    Retourne (log_main, log_errors, log_warnings).

    Console  : _ColorConsoleFormatter (couleurs ANSI si TTY).
    Fichiers : _StripAnsiFormatter    (texte brut, codes ANSI retirés).
    """
    root_level = logging.DEBUG if verbose else getattr(logging, level_str.upper(), logging.INFO)
    FILE_FMT   = "%(asctime)s [%(levelname)-8s] %(message)s"

    console = logging.StreamHandler()
    console.setFormatter(_ColorConsoleFormatter())
    console.setLevel(root_level)

    handlers_main = [console]
    handlers_err  = []
    handlers_warn = []

    if log_dir:
        d = Path(log_dir)
        handlers_main.append(
            _make_rotating_handler(d / "worker_import.log",
                                   logging.INFO, logging.CRITICAL, FILE_FMT))
        handlers_err.append(
            _make_rotating_handler(d / "worker_import_errors.log",
                                   logging.ERROR, logging.CRITICAL, FILE_FMT))
        handlers_warn.append(
            _make_rotating_handler(d / "worker_import_warnings.log",
                                   logging.WARNING, logging.WARNING,
                                   "%(asctime)s %(message)s"))

    def _make_logger(name: str, handlers: list) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        for h in handlers:
            logger.addHandler(h)
        return logger

    return (
        _make_logger("worker.main",     handlers_main),
        _make_logger("worker.errors",   handlers_main + handlers_err),
        _make_logger("worker.warnings", handlers_main + handlers_warn),
    )


# ===========================================================================
# Résumés colorés
# ===========================================================================

def _bar(ok: int, errors: int, width: int = 20) -> str:
    """
    Mini barre de progression colorée.
      vert  = posts OK
      rouge = erreurs
      gris  = reste
    Ex :  ████████████░░░░░░░░
    """
    if not _TTY:
        return ""
    total = ok + errors
    if total == 0:
        return ""
    filled = round(ok     / total * width)
    erred  = round(errors / total * width)
    rest   = width - filled - erred
    return (
        " "
        + C.OK  + "█" * filled
        + C.ERR + "█" * erred
        + C.DIM + "░" * rest
        + C.RESET
    )


def print_run_summary(
    total:       dict,
    by_platform: dict,
    run_label:   str,
    log:         logging.Logger,
) -> None:
    """
    Résumé coloré d'un batch complet.
    Distingue clairement : nouveaux insérés / déjà en base / erreurs.
    Émis vers log.info → affiché en couleur dans le terminal,
    enregistré en texte brut dans worker_import.log.
    """
    taux = (total["ok"] / max(total["ok"] + total["errors"], 1)) * 100
    sep  = "━" * 60

    log.info(_c(C.BOLD, sep))
    log.info(f"  {_c(C.BOLD, 'Résumé par plateforme')} — {_c(C.INFO, run_label)}")

    for plat, bp in sorted(by_platform.items()):
        t_ok      = bp["ok"]
        t_ins     = bp.get("inserted", 0)
        t_skip    = t_ok - t_ins          # parsés OK mais déjà en base
        t_err     = bp["errors"]
        taux_p    = (t_ok / max(t_ok + t_err, 1)) * 100
        media_str = (f"  {_c(C.CYAN, str(bp['media']))} médias"
                     if bp["media"] else "")
        err_str   = (f"  {_c(C.ERR,  str(t_err))} err"
                     if t_err else "")
        ins_str   = _c(C.OK,  f"{t_ins} nouveaux")
        skip_str  = (_c(C.DIM, f"  {t_skip} déjà en base") if t_skip else "")
        log.info(
            f"  {_c(C.INFO, f'{plat:<12}')}"
            f"  {bp['files']:>4} fichiers"
            f"  {ins_str}"
            f"{skip_str}"
            f"{err_str}"
            f"{media_str}"
            f"  {_c(C.DIM, f'{taux_p:.1f}%')}"
            f"{_bar(t_ok, t_err)}"
        )

    log.info(_c(C.DIM, "  " + "─" * 56))

    t_ins_total  = total.get("inserted", 0)
    t_skip_total = total["ok"] - t_ins_total
    t_skip_cur   = total.get("skipped_cursor", 0)

    ok_str   = _c(C.OK   + C.BOLD, f"{t_ins_total} nouveaux")
    skip_str = (_c(C.DIM, f"  {t_skip_total} déjà en base") if t_skip_total else "")
    cur_str  = (_c(C.DIM, f"  {t_skip_cur} ignorés (inchangés)")
                if t_skip_cur else "")
    err_str  = (_c(C.ERR + C.BOLD, str(total["errors"]))
                if total["errors"] else _c(C.DIM, "0"))
    med_str  = (_c(C.CYAN, str(total["media"]))
                if total["media"] else _c(C.DIM, "0"))

    log.info(
        f"  {_c(C.BOLD, 'TOTAL')}"
        f"  {total['files']} traités"
        f"  {ok_str}"
        f"{skip_str}"
        f"{cur_str}"
        f"  {err_str} erreurs"
        f"  {med_str} médias"
        f"  {_c(C.DIM, f'{taux:.1f}%')}"
        f"{_bar(total['ok'], total['errors'])}"
    )
    log.info(_c(C.BOLD, sep))


def print_final_summary(stats: dict, log: logging.Logger) -> None:
    """Résumé final après un run one-shot."""
    sep     = "━" * 60
    ok_str  = _c(C.OK   + C.BOLD, str(stats["ok"]))
    err_str = (_c(C.ERR + C.BOLD, str(stats["errors"]))
               if stats["errors"] else _c(C.DIM, "0"))
    med_str = _c(C.CYAN + C.BOLD, str(stats.get("media", 0)))

    log.info("")
    log.info(_c(C.BOLD, sep))
    log.info(f"  Fichiers traités  : {stats['files']}")
    log.info(f"  Posts OK          : {ok_str}")
    log.info(f"  Médias référencés : {med_str}")
    log.info(f"  Erreurs           : {err_str}")
    log.info(_c(C.BOLD, sep))


# ===========================================================================
# Heartbeat — mode surveillance continue
# ===========================================================================

def _format_idle(seconds: int) -> str:
    """Formate une durée en secondes : '45s', '2m 05s', '1h 03m'."""
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


class Heartbeat:
    """
    Émet une ligne colorée sur stderr toutes les `interval` secondes
    pendant les périodes d'inactivité en mode watch.

    Écrit sur stderr pour ne pas interférer avec stdout ni les fichiers de log.
    Utilise les couleurs ANSI si stderr est un TTY (test indépendant de stdout).

    Usage :
        hb = Heartbeat(interval=30)
        hb.tick()    # à chaque tour de boucle vide
        hb.reset()   # après un batch actif
    """
    _TTY_ERR = sys.stderr.isatty()

    def __init__(self, interval: int = HEARTBEAT_INTERVAL):
        self._interval      = interval
        self._last_hb       = time.monotonic()
        self._last_activity = time.monotonic()

    def reset(self) -> None:
        now = time.monotonic()
        self._last_hb       = now
        self._last_activity = now

    def tick(self) -> None:
        now = time.monotonic()
        if now - self._last_hb < self._interval:
            return

        idle_sec = int(now - self._last_activity)
        ts       = datetime.now().strftime("%H:%M:%S")
        idle_str = _format_idle(idle_sec)

        if self._TTY_ERR:
            line = (
                f"\033[2m{ts}\033[0m "
                f"\033[35m♥ worker actif\033[0m"
                f"\033[2m — en attente depuis {idle_str}\033[0m"
            )
        else:
            line = f"{ts} [HEARTBEAT] worker actif — en attente depuis {idle_str}"

        print(line, file=sys.stderr, flush=True)
        self._last_hb = now


# ===========================================================================
# Agrégateur de warnings — déduplique par type sur un run entier
# ===========================================================================

class WarnAggregator:
    """
    Collecte les warnings non bloquants pendant un run et les écrit
    en une seule fois à la fin, dédupliqués avec compteur.
    """
    EXPECTED = {"post_url absent — sera reconstruit",
                "media_type absent — sera déduit"}

    def __init__(self):
        self._counts: dict[str, int] = {}

    def add(self, msg: str):
        clean = msg.split("] ", 1)[-1] if "] " in msg else msg
        self._counts[clean] = self._counts.get(clean, 0) + 1

    def flush(self, log_warn: logging.Logger, run_label: str = ""):
        if not self._counts:
            return
        prefix = f"[{run_label}] " if run_label else ""
        total  = sum(self._counts.values())
        log_warn.warning(
            _c(C.WARN, f"{prefix}Warnings dédupliqués ({total} total) :")
        )
        for msg, count in sorted(self._counts.items(), key=lambda x: -x[1]):
            tag   = " [ATTENDU]" if msg in self.EXPECTED else " [ANOMALIE]"
            color = C.DIM if msg in self.EXPECTED else C.WARN
            log_warn.warning(
                f"  {_c(C.DIM, f'x{count:>5}')}  {_c(color, msg)}{_c(C.DIM, tag)}"
            )
        self._counts.clear()


# ===========================================================================
# Validation
# ===========================================================================

class ValidationError(Exception):
    pass


def validate_scrapper_json(data: dict) -> list[str]:
    warnings = []
    info = data.get("scrappeurInfo")
    if not info:
        raise ValidationError("Bloc 'scrappeurInfo' absent — import impossible")

    platform_raw = info.get("platform", "")
    if platform_raw.lower() not in PLATFORM_ALIASES:
        raise ValidationError(f"Plateforme inconnue : '{platform_raw}'")

    post_id = data.get("id")
    if not post_id:
        raise ValidationError("Champ 'id' absent ou vide")

    platform_norm = PLATFORM_ALIASES[platform_raw.lower()]
    author_name   = (data.get("author") or {}).get("name", "")
    if platform_norm == "twitter" and post_id == author_name:
        warnings.append(
            f"BUG_ID: 'id' = nom du compte ('{post_id}') sur Twitter — "
            f"déduplication impossible, import quand même"
        )

    if not data.get("post_url"):
        warnings.append("post_url absent — sera reconstruit")
    if not data.get("media_type"):
        warnings.append("media_type absent — sera déduit")

    return warnings


# ===========================================================================
# Normalisation
# ===========================================================================

def normalize_platform(raw: str) -> str:
    return PLATFORM_ALIASES.get(raw.lower(), raw.lower())


def infer_post_url(data: dict, platform: str) -> Optional[str]:
    if data.get("post_url"):
        return data["post_url"]
    post_id    = data.get("id", "")
    author     = data.get("author") or {}
    author_id  = author.get("uniqueId") or author.get("name") or ""
    channel_id = str(post_id).split("/")[0]
    tmpl       = POST_URL_TEMPLATES.get(platform, "")
    if not tmpl:
        return None
    try:
        return tmpl.format(post_id=post_id, author_id=author_id,
                           author_unique=author_id, channel_id=channel_id)
    except KeyError:
        return None


def infer_media_type(data: dict) -> Optional[str]:
    if data.get("media_type"):
        return data["media_type"]
    if data.get("video_url"):
        return "video"
    if data.get("cover"):
        return "image"
    return None


def parse_datetime(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    if isinstance(value, (int, float)):
        return None if value == 0 else datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        if value.startswith("1970-01-01"):
            return None
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
                    "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                    "%Y-%m-%dT%H:%M:%S.000Z"):
            try:
                dt = datetime.strptime(value, fmt)
                return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
            except ValueError:
                continue
    return None


# ===========================================================================
# Construction des documents MongoDB
# ===========================================================================

def build_account_doc(data: dict, platform: str) -> dict:
    author       = data.get("author") or {}
    platform_id  = str(author.get("id") or author.get("uniqueId") or
                       author.get("name") or "unknown")
    username     = author.get("uniqueId") or author.get("name") or ""
    display_name = author.get("name") or username
    doc          = new_account(platform=platform, platform_id=platform_id,
                               username=username, display_name=display_name, raw=author)
    doc["url"] = author.get("url")
    if author.get("avatar"):
        doc["profile"]["avatar_url"] = author["avatar"]
    return doc


def build_post_doc(
    data: dict,
    platform: str,
    account_id,
    account_platform_id: str,
    source_context: Optional[dict] = None,
) -> dict:
    post_url   = infer_post_url(data, platform)
    media_type = infer_media_type(data)
    stats      = data.get("stats") or {}
    doc        = new_post(platform=platform, platform_id=str(data["id"]),
                          account_id=account_id, account_platform_id=account_platform_id,
                          text_content=data.get("desc") or "", raw=data)

    doc["url"]                     = post_url
    doc["text"]["hashtags"]        = data.get("hashtags") or []
    doc["text"]["mentions"]        = data.get("mentions") or []
    doc["context"]["published_at"] = parse_datetime(data.get("createTime"))
    doc["engagement"]["likes"]     = stats.get("likes")     or 0
    doc["engagement"]["shares"]    = stats.get("shares")    or 0
    doc["engagement"]["comments"]  = stats.get("comments")  or 0
    doc["engagement"]["views"]     = stats.get("plays")     or 0
    doc["engagement"]["saves"]     = stats.get("favorites") or 0

    # --- Source projet/scan/user [v4] ---
    if source_context:
        scan_raw = source_context.get("scan", "") or ""
        doc["source"] = {
            "project": source_context.get("project") or None,
            "scan":    scan_raw.removeprefix("converted_") or None,
            "user":    source_context.get("user") or None,
        }

    if data.get("is_fake") is not None:
        doc["deepfake"]["prediction"]  = "synthetic" if data["is_fake"] else "likely_real"
        doc["deepfake"]["final_score"] = float(data.get("fake_confidence") or 0)
        doc["deepfake"]["status"]      = "done" if data.get("manually_reviewed") else "pending"

    ps = doc["platform_specific"]
    if platform == "tiktok":
        music = data.get("music") or {}
        ps["music_title"]  = music.get("title")
        ps["music_author"] = music.get("author")          # [v4] son viral coordonné
        ps["cover_url"]    = data.get("cover") or None    # [v4] miniature vidéo
    elif platform == "instagram":
        ps["shortcode"]  = str(data["id"])
        ps["media_type"] = media_type
        ps["is_reel"]    = (media_type == "reel")
        ps["cover_url"]  = data.get("cover") or None      # [v4] miniature post
    elif platform == "twitter":
        ps["cover_url"]  = data.get("cover") or None      # [v4]
    elif platform == "telegram":
        parts = str(data["id"]).split("/")
        if len(parts) == 2:
            ps["channel_id"] = parts[0]
            ps["message_id"] = parts[1]
        ps["views"] = stats.get("plays") or 0

    # --- Signaux scrapper [v4] ---
    doc["scrapper_signals"]["influence_score"]  = data.get("influence_score")
    doc["scrapper_signals"]["is_bot_suspected"] = bool(data.get("is_bot_suspected", False))

    doc["deepfake"]["has_media"] = False
    doc["deepfake"]["status"]    = "pending"
    return doc


def build_comment_docs(data: dict, post_id, post_platform_id: str,
                       platform: str) -> list[dict]:
    docs = []
    for idx, c in enumerate(data.get("comments") or []):
        if not isinstance(c, dict):
            continue
        comment_platform_id = (
            str(c.get("comment_id")) if c.get("comment_id")
            else f"{post_platform_id}_c{idx}"
        )
        author_id = str(c.get("author_id") or c.get("author") or "unknown")
        doc = new_comment(
            platform            = platform,
            platform_id         = comment_platform_id,
            post_id             = post_id,
            post_platform_id    = post_platform_id,
            account_id          = None,
            account_platform_id = author_id,
            text_content        = c.get("text") or "",
            raw                 = c,
        )
        doc["published_at"]        = parse_datetime(c.get("timestamp"))
        doc["engagement"]["likes"] = c.get("likes") or 0
        if c.get("reply_to_id"):
            doc["parent_comment_platform_id"] = str(c["reply_to_id"])
            doc["depth"]                      = 1
        docs.append(doc)
    return docs


# ===========================================================================
# Gestion des médias — résolution arborescence DOSSIER_INPUT
# ===========================================================================

def resolve_raw_dir(converted_file: Path, source_root: Path) -> Optional[Path]:
    """
    Résout le dossier raw symétrique depuis un fichier converted_*.json.

        source_root / projet / converted_scan / user / converted_post.json
                                    ↓ retire "converted_"
        source_root / projet / scan            / user /   ← retourné

    Retourne None si le dossier parent n'a pas le préfixe "converted_"
    ou si le dossier raw n'existe pas physiquement.
    """
    try:
        user_dir    = converted_file.parent
        scan_dir    = user_dir.parent
        project_dir = scan_dir.parent

        if not scan_dir.name.startswith("converted_"):
            return None

        raw_scan_name = scan_dir.name[len("converted_"):]
        raw_user_dir  = project_dir / raw_scan_name / user_dir.name
        return raw_user_dir if raw_user_dir.is_dir() else None
    except Exception:
        return None


def find_media_files(converted_file: Path, raw_dir: Path) -> list[Path]:
    """
    Cherche tous les fichiers médias dont le stem correspond au stem original
    (sans préfixe "converted_" ni suffixe ".info") dans raw_dir.

    Exemple :
        converted_post.info.json  → cherche post.mp4, post.jpg, etc.
        converted_post.json       → cherche post.mp4, post.jpg, etc.
    """
    stem          = converted_file.stem
    # Retire le préfixe "converted_"
    original_stem = stem[len("converted_"):] if stem.startswith("converted_") else stem
    # Retire le suffixe ".info" si présent (fichiers nommés *.info.json)
    if original_stem.endswith(".info"):
        original_stem = original_stem[:-len(".info")]
    return [
        candidate
        for ext in MEDIA_EXTENSIONS
        if (candidate := raw_dir / f"{original_stem}{ext}").is_file()
    ]


def infer_media_type_from_ext(file_path: Path) -> str:
    return MEDIA_EXTENSIONS.get(file_path.suffix.lower(), "image")


def get_url_original_from_json(data: dict, media_type_guess: str) -> Optional[str]:
    if media_type_guess == "video":
        return data.get("video_url") or None
    if media_type_guess == "image":
        return data.get("cover") or data.get("thumbnail_url") or None
    return None


def build_and_insert_media(
    media_file:     Path,
    post_data:      dict,
    post_id,
    platform:       str,
    db,
    dry_run:        bool,
    source_context: dict,
    log:            logging.Logger,
    log_err:        logging.Logger,
) -> Optional[dict]:
    """
    Crée un document media, l'insère en base, l'attache au post,
    et crée un job deepfake_analysis (idempotent sur post_id + url_local).
    Retourne le media_ref ou None si erreur.
    """
    media_type   = infer_media_type_from_ext(media_file)
    url_original = get_url_original_from_json(post_data, media_type)
    url_local    = str(media_file.resolve())

    ctx = ""
    if source_context and any(source_context.values()):
        p = source_context.get("project", "")
        s = source_context.get("scan",    "")
        u = source_context.get("user",    "")
        ctx = _c(C.DIM, f" (projet={p} scan={s} user={u})")

    log.debug(
        f"    {_c(C.CYAN, '↳')} média "
        f"[{_c(C.INFO, media_type)}] "
        f"{_c(C.DIM, media_file.name)}{ctx}"
    )

    if dry_run:
        return {
            "media_id":     None,
            "type":         media_type,
            "url_original": url_original,
            "url_local":    url_local,
            "downloaded":   True,
        }

    try:
        # Normalise source_context : retire le préfixe "converted_" du scan
        # pour que le champ source.scan corresponde au dossier raw (sans préfixe).
        source_doc = None
        if source_context:
            scan_raw = source_context.get("scan", "") or ""
            source_doc = {
                "project": source_context.get("project") or None,
                "scan":    scan_raw.removeprefix("converted_") or None,
                "user":    source_context.get("user") or None,
            }

        media_doc = new_media(media_type=media_type,
                              url_original=url_original or None,
                              url_local=url_local,
                              source=source_doc)

        # Upsert sur url_local — évite le conflit d'index hash_md5 (null unique)
        # en cas de double import du même fichier physique.
        result   = db.media.update_one(
            {"url_local": url_local},
            {"$setOnInsert": media_doc},
            upsert=True,
        )
        if result.upserted_id:
            media_id = result.upserted_id
        else:
            existing = db.media.find_one({"url_local": url_local}, {"_id": 1})
            media_id = existing["_id"] if existing else None

        if media_id is None:
            log_err.error(f"  {_c(C.ERR, '✗')} Média : impossible de récupérer l'_id [{media_file.name}]")
            return None

        media_ref = {
            "media_id":     media_id,
            "type":         media_type,
            "url_original": url_original,
            "url_local":    url_local,
            "downloaded":   True,
        }
        db.posts.update_one({"_id": post_id}, patch_post_media(media_ref))
        # Lien inverse média → post (reuse.post_ids)
        db.media.update_one({"_id": media_id}, patch_media_reuse(post_id, platform))

        # Job deepfake_analysis — idempotent sur (post_id, url_local)
        db.jobs.update_one(
            {"type":              "deepfake_analysis",
             "payload.post_id":   post_id,
             "payload.url_local": url_local},
            {"$setOnInsert": new_job(
                job_type = "deepfake_analysis",
                payload  = {
                    "post_id":    post_id,
                    "media_id":   media_id,
                    "platform":   platform,
                    "url_local":  url_local,
                    "file_name":  media_file.name,
                    "media_type": media_type,
                },
                priority = 1,
            )},
            upsert=True,
        )
        return media_ref

    except PyMongoError as e:
        log_err.error(
            f"  {_c(C.ERR, '✗')} Média MongoDB "
            f"[{_c(C.WARN, media_file.name)}] : {e}"
        )
        return None


# ===========================================================================
# Import d'un fichier JSON
# ===========================================================================

def import_json_file(
    json_path:   Path,
    db,
    dry_run:     bool,
    log:         logging.Logger,
    log_err:     logging.Logger,
    log_warn:    logging.Logger,
    warn_agg:    Optional[WarnAggregator] = None,
    source_root: Optional[Path]           = None,
) -> dict:
    stats = {"ok": 0, "inserted": 0, "skipped": 0, "errors": 0, "warnings": [], "media": 0}

    try:
        with open(json_path, encoding="utf-8") as f:
            raw_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log_err.error(
            f"{_c(C.ERR, '✗')} Lecture JSON : "
            f"{_c(C.WARN, json_path.name)} — {e}"
        )
        stats["errors"] += 1
        return stats

    items = raw_data if isinstance(raw_data, list) else [raw_data]

    # Résolution dossier raw (une seule fois par fichier JSON)
    raw_dir: Optional[Path] = None
    if source_root is not None:
        raw_dir = resolve_raw_dir(json_path, source_root)

    # Contexte source pour les logs médias
    source_context: dict = {}
    if source_root:
        try:
            parts = json_path.relative_to(source_root).parts
            source_context = {
                "project": parts[0] if len(parts) > 0 else "",
                "scan":    parts[1] if len(parts) > 1 else "",
                "user":    parts[2] if len(parts) > 2 else "",
            }
        except Exception:
            pass

    for item in items:
        try:
            w_list = validate_scrapper_json(item)
            for w in w_list:
                if warn_agg:
                    warn_agg.add(w)
                else:
                    log_warn.warning(f"[{json_path.name}] {w}")
            stats["warnings"].extend(w_list)

            platform_raw = item["scrappeurInfo"]["platform"]
            platform     = normalize_platform(platform_raw)
            author       = item.get("author") or {}
            author_pid   = str(author.get("id") or author.get("uniqueId") or
                               author.get("name") or "unknown")
            post_id_str  = str(item["id"])

            if not dry_run:
                # Upsert account
                account_doc = build_account_doc(item, platform)
                acc_result  = db.accounts.update_one(
                    {"platform": platform, "platform_id": author_pid},
                    {"$setOnInsert": account_doc}, upsert=True,
                )
                account_id = acc_result.upserted_id
                if account_id is None:
                    ex = db.accounts.find_one(
                        {"platform": platform, "platform_id": author_pid}, {"_id": 1})
                    account_id = ex["_id"] if ex else None

                # Upsert post
                post_doc    = build_post_doc(item, platform, account_id, author_pid,
                                             source_context=source_context)
                post_result = db.posts.update_one(
                    {"platform": platform, "platform_id": post_id_str},
                    {"$setOnInsert": post_doc}, upsert=True,
                )
                mongo_post_id = post_result.upserted_id
                if mongo_post_id is not None:
                    stats["inserted"] += 1   # nouveau post inséré
                else:
                    ex = db.posts.find_one(
                        {"platform": platform, "platform_id": post_id_str}, {"_id": 1})
                    mongo_post_id = ex["_id"] if ex else None

                # Comments
                if mongo_post_id:
                    for cdoc in build_comment_docs(item, mongo_post_id,
                                                   post_id_str, platform):
                        db.comments.update_one(
                            {"platform": platform, "post_id": mongo_post_id,
                             "platform_id": cdoc["platform_id"]},
                            {"$setOnInsert": cdoc}, upsert=True,
                        )

                # Médias (mode source)
                if mongo_post_id and raw_dir is not None:
                    for mf in find_media_files(json_path, raw_dir):
                        ref = build_and_insert_media(
                            mf, item, mongo_post_id, platform,
                            db, False, source_context, log, log_err,
                        )
                        if ref is not None:
                            stats["media"] += 1

                # Fallback mode inbox : job si URL média présente dans le JSON
                elif mongo_post_id and raw_dir is None:
                    has_media = bool(
                        item.get("video_url") or item.get("cover") or
                        (item.get("scrappeurInfo") or {}).get("media_telecharge")
                    )
                    if has_media:
                        db.posts.update_one(
                            {"_id": mongo_post_id},
                            {"$set": {"deepfake.has_media": True}},
                        )
                        db.jobs.update_one(
                            {"type":             "deepfake_analysis",
                             "payload.post_id":  mongo_post_id},
                            {"$setOnInsert": new_job(
                                job_type = "deepfake_analysis",
                                payload  = {"post_id":   mongo_post_id,
                                            "platform":  platform,
                                            "url_local": None},
                                priority = 1,
                            )}, upsert=True,
                        )

            else:
                # dry-run : simuler résolution médias sans écriture
                if raw_dir is not None:
                    for mf in find_media_files(json_path, raw_dir):
                        ref = build_and_insert_media(
                            mf, item, None, platform,
                            db, True, source_context, log, log_err,
                        )
                        if ref is not None:
                            stats["media"] += 1

            stats["ok"] += 1
            # skipped = parsés OK mais déjà en base (pas de nouvel insert)
            if not dry_run and stats["inserted"] == 0 and stats["ok"] == 1:
                stats["skipped"] += 1
            _n_media   = stats["media"]
            media_info = (
                f" {_c(C.CYAN, f'+{_n_media} média(s)')}"
                if _n_media else ""
            )
            log.debug(
                f"  {_c(C.OK, '✓')} "
                f"[{_c(C.INFO, platform)}] "
                f"{_c(C.DIM, post_id_str[:40])}"
                f"{media_info}"
            )

        except ValidationError as e:
            log_err.error(
                f"  {_c(C.ERR, '✗')} Validation "
                f"[{_c(C.WARN, json_path.name)}] : {e}"
            )
            stats["errors"] += 1
        except PyMongoError as e:
            log_err.error(
                f"  {_c(C.ERR, '✗')} MongoDB "
                f"[{_c(C.WARN, json_path.name)}] : {e}"
            )
            stats["errors"] += 1
        except Exception as e:
            log_err.error(
                f"  {_c(C.ERR, '✗')} Inattendu "
                f"[{_c(C.WARN, json_path.name)}] : {e}",
                exc_info=True,
            )
            stats["errors"] += 1

    return stats



# ===========================================================================
# Curseur d'import — évite de rescanner les fichiers inchangés
# ===========================================================================

class ImportCursor:
    """
    Maintient un état persistant des fichiers déjà importés.

    Stocké dans <log_dir>/import_cursor.json.
    Chaque entrée : chemin relatif → {size, mtime, imported_at}.

    À chaque run :
      1. Charge le curseur existant.
      2. Filtre les fichiers dont (size, mtime) sont inchangés → skip.
      3. Après le run, met à jour les entrées des fichiers traités.
      4. Purge les entrées dont le fichier n'existe plus sur disque
         → déplace ces entrées dans <log_dir>/archives/import_cursor_<date>.json

    Le curseur est proportionnel au contenu actuel de INPUT/, jamais à
    l'historique total.
    """

    VERSION = 1

    def __init__(self, log_dir: Optional[Path]):
        self._path:    Optional[Path] = (
            (Path(log_dir) / "import_cursor.json") if log_dir else None
        )
        self._archive_dir: Optional[Path] = (
            (Path(log_dir) / "archives") if log_dir else None
        )
        self._data: dict[str, dict] = {}   # chemin_relatif → {size, mtime, imported_at}
        self._load()

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and raw.get("version") == self.VERSION:
                self._data = raw.get("files", {})
        except Exception:
            self._data = {}

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version":    self.VERSION,
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "files":      self._data,
        }
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def is_unchanged(self, json_path: Path, source_root: Path) -> bool:
        """
        Retourne True si le fichier est déjà dans le curseur avec les mêmes
        size et mtime → peut être skippé.
        """
        key = self._key(json_path, source_root)
        if key not in self._data:
            return False
        try:
            st = json_path.stat()
            entry = self._data[key]
            return entry["size"] == st.st_size and abs(entry["mtime"] - st.st_mtime) < 1.0
        except OSError:
            return False

    def mark_done(self, json_path: Path, source_root: Path) -> None:
        """Enregistre le fichier comme traité avec ses métadonnées actuelles."""
        try:
            st  = json_path.stat()
            key = self._key(json_path, source_root)
            self._data[key] = {
                "size":        st.st_size,
                "mtime":       st.st_mtime,
                "imported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        except OSError:
            pass

    def purge_missing(
        self,
        existing_files: list[Path],
        source_root:    Path,
        log:            logging.Logger,
    ) -> int:
        """
        Supprime du curseur les entrées dont le fichier n'existe plus.
        Archive ces entrées dans <log_dir>/archives/import_cursor_<date>.json.
        Retourne le nombre d'entrées purgées.
        """
        existing_keys = {self._key(f, source_root) for f in existing_files}
        obsolete = {k: v for k, v in self._data.items() if k not in existing_keys}
        if not obsolete:
            return 0

        # Archivage
        if self._archive_dir is not None:
            self._archive_dir.mkdir(parents=True, exist_ok=True)
            ts        = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            arch_path = self._archive_dir / f"import_cursor_{ts}.json"
            arch_payload = {
                "version":    self.VERSION,
                "archived_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source_root": str(source_root),
                "files":      obsolete,
            }
            arch_path.write_text(
                json.dumps(arch_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            log.info(
                f"  Curseur : {len(obsolete)} entrée(s) obsolète(s) archivée(s) "
                f"→ {arch_path.name}"
            )

        for k in obsolete:
            del self._data[k]

        return len(obsolete)

    def save(self) -> None:
        """Persiste le curseur sur disque."""
        self._save()

    # ------------------------------------------------------------------
    # Interne
    # ------------------------------------------------------------------

    @staticmethod
    def _key(json_path: Path, source_root: Path) -> str:
        try:
            return str(json_path.relative_to(source_root))
        except ValueError:
            return str(json_path)


# ===========================================================================
# Mode source — arborescence DOSSIER_INPUT
# ===========================================================================

def collect_converted_json(source: Path) -> list[Path]:
    """Collecte tous les converted_*.json (ignore JSON bruts et Zone.Identifier)."""
    return sorted(
        f for f in source.rglob("converted_*.json")
        if not f.name.endswith("Zone.Identifier")
    )


def _run_source_batch(
    source:   Path,
    db,
    dry_run:  bool,
    log:      logging.Logger,
    log_err:  logging.Logger,
    log_warn: logging.Logger,
    cursor:   Optional["ImportCursor"] = None,
) -> dict:
    """Traite un batch complet. Partagé par run_source et run_source_watch."""
    all_files = collect_converted_json(source)
    if not all_files:
        log.info(_c(C.DIM, f"Aucun fichier converted_*.json dans {source}"))
        return {"ok": 0, "inserted": 0, "skipped_cursor": 0, "errors": 0, "files": 0, "media": 0}

    # Purge des entrées obsolètes du curseur (fichiers supprimés du dossier INPUT)
    if cursor is not None:
        cursor.purge_missing(all_files, source, log)

    # Filtrage des fichiers inchangés via le curseur
    if cursor is not None and not dry_run:
        new_files      = [f for f in all_files if not cursor.is_unchanged(f, source)]
        skipped_cursor = len(all_files) - len(new_files)
    else:
        new_files      = all_files
        skipped_cursor = 0

    run_label = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    log.info(_c(C.BOLD, "━" * 60))
    log.info(f"  {_c(C.BOLD, 'Début du run')} — {_c(C.INFO, run_label)}")
    log.info(f"  Source   : {_c(C.DIM, str(source))}")
    log.info(
        f"  Fichiers : {_c(C.BOLD, str(len(all_files)))} trouvés"
        + (f"  {_c(C.DIM, str(skipped_cursor) + ' ignorés (inchangés)')}"
           if skipped_cursor else "")
        + (f"  {_c(C.INFO, str(len(new_files)) + ' à traiter')}"
           if skipped_cursor else "")
    )
    if dry_run:
        log.info(_c(C.WARN + C.BOLD, "  ⚠  DRY-RUN — aucune écriture en base"))
    log.info(_c(C.BOLD, "━" * 60))

    if not new_files:
        log.info(_c(C.DIM, "  Aucun nouveau fichier à importer."))
        return {"ok": 0, "inserted": 0, "skipped_cursor": skipped_cursor, "errors": 0, "files": 0, "media": 0}

    warn_agg    = WarnAggregator()
    total       = {"ok": 0, "inserted": 0, "skipped_cursor": skipped_cursor,
                   "errors": 0, "files": 0, "media": 0}
    by_platform: dict[str, dict] = {}

    for json_path in new_files:
        stats = import_json_file(
            json_path=json_path, db=db, dry_run=dry_run,
            log=log, log_err=log_err, log_warn=log_warn,
            warn_agg=warn_agg, source_root=source,
        )

        # Mise à jour du curseur si import réussi
        if cursor is not None and not dry_run and stats["errors"] == 0:
            cursor.mark_done(json_path, source)

        try:
            parts    = json_path.relative_to(source).parts
            raw_scan = parts[1].removeprefix("converted_") if len(parts) > 1 else ""
            plat_key = PLATFORM_ALIASES.get(
                raw_scan.split("_")[0].lower(),
                raw_scan.split("_")[0].lower()
            ) if raw_scan else "unknown"
        except Exception:
            plat_key = "unknown"

        bp = by_platform.setdefault(
            plat_key, {"ok": 0, "inserted": 0, "errors": 0, "files": 0, "media": 0})
        bp["ok"]       += stats["ok"]
        bp["inserted"] += stats.get("inserted", 0)
        bp["errors"]   += stats["errors"]
        bp["files"]    += 1
        bp["media"]    += stats.get("media", 0)

        total["ok"]       += stats["ok"]
        total["inserted"] += stats.get("inserted", 0)
        total["errors"]   += stats["errors"]
        total["files"]    += 1
        total["media"]    += stats.get("media", 0)

        if stats["errors"] > 0:
            rel = str(json_path.relative_to(source))
            log.info(
                f"{_c(C.ERR, '[NOK]')} {_c(C.WARN, rel)} — "
                f"{_c(C.ERR, str(stats['errors']))} erreur(s)"
            )

    # Persist curseur
    if cursor is not None and not dry_run:
        cursor.save()

    print_run_summary(total, by_platform, run_label, log)
    warn_agg.flush(log_warn, run_label)
    return total


def run_source(
    source:   Path,
    db,
    dry_run:  bool,
    log:      logging.Logger,
    log_err:  logging.Logger,
    log_warn: logging.Logger,
    log_dir:  Optional[str] = None,
) -> dict:
    """Mode source one-shot."""
    cursor = ImportCursor(log_dir) if log_dir else None
    return _run_source_batch(source, db, dry_run, log, log_err, log_warn, cursor)


def run_source_watch(
    source:      Path,
    db,
    dry_run:     bool,
    log:         logging.Logger,
    log_err:     logging.Logger,
    log_warn:    logging.Logger,
    poll:        int = POLL_INTERVAL,
    hb_interval: int = HEARTBEAT_INTERVAL,
    log_dir:     Optional[str] = None,
):
    """
    Mode source continu : surveille le DOSSIER_INPUT en boucle.
    Heartbeat sur stderr toutes les `hb_interval` secondes.
    Idempotent — les posts déjà importés sont ignorés par les upserts.
    """
    log.info(
        f"{_c(C.BOLD, 'Surveillance DOSSIER_INPUT')} : "
        f"{_c(C.INFO, str(source))}"
        f"{_c(C.DIM, f'  poll={poll}s  heartbeat={hb_interval}s')}"
        f"  {_c(C.DIM, 'Ctrl+C pour arrêter')}"
    )
    cursor = ImportCursor(log_dir) if log_dir else None
    if cursor is not None:
        log.info(f"  Curseur  : {_c(C.DIM, str(cursor._path))}")
    hb = Heartbeat(interval=hb_interval)
    try:
        while True:
            files = collect_converted_json(source)
            if files:
                result = _run_source_batch(source, db, dry_run, log, log_err, log_warn, cursor)
                # Si rien n'a été traité (tout ignoré par curseur), on ne reset pas
                # le heartbeat — il continuera à s'afficher pendant l'attente.
                if result.get("files", 0) > 0:
                    hb.reset()
            # Attente découpée en tranches de 1s pour que le heartbeat
            # s'affiche à l'intervalle voulu même pendant un long poll.
            elapsed = 0
            while elapsed < poll:
                time.sleep(1)
                elapsed += 1
                hb.tick()
    except KeyboardInterrupt:
        log.info(_c(C.DIM, "Arrêt demandé (Ctrl+C) — worker stoppé proprement."))


# ===========================================================================
# Mode inbox (flux scrapper continu)
# ===========================================================================

def ensure_dirs(storage: Path) -> tuple[Path, Path, Path]:
    inbox, processing, done = (storage / "inbox",
                                storage / "processing",
                                storage / "done")
    for d in (inbox, processing, done):
        d.mkdir(parents=True, exist_ok=True)
    return inbox, processing, done


def collect_json_files(inbox: Path) -> list[Path]:
    return sorted(
        f for f in inbox.rglob("*.json")
        if not f.name.endswith("Zone.Identifier")
    )


def process_batch(
    files, inbox, processing, done, db, dry_run,
    log, log_err, log_warn,
    warn_agg: Optional[WarnAggregator] = None,
) -> dict:
    total = {"ok": 0, "errors": 0, "files": 0, "media": 0}
    for json_path in files:
        rel       = json_path.relative_to(inbox.parent)
        proc_path = processing / json_path.name
        if not dry_run:
            try:
                shutil.move(str(json_path), proc_path)
            except OSError as e:
                log_err.error(
                    f"{_c(C.ERR, '✗')} Déplacement vers processing échoué : {e}"
                )
                continue
        else:
            proc_path = json_path

        stats = import_json_file(
            proc_path, db, dry_run, log, log_err, log_warn, warn_agg,
            source_root=None,
        )

        if stats["errors"] > 0:
            log.info(
                f"{_c(C.ERR, '[NOK]')} {_c(C.WARN, str(rel))} — "
                f"{_c(C.ERR, str(stats['errors']))} erreur(s)"
            )

        if not dry_run:
            if stats["errors"] > 0 and stats["ok"] == 0:
                shutil.move(str(proc_path),
                            inbox / (json_path.stem + ".error.json"))
                log_err.warning(
                    f"{_c(C.WARN, 'Fichier remis dans inbox')} "
                    f"(.error.json) : {rel}"
                )
            else:
                shutil.move(str(proc_path), done / json_path.name)

        total["ok"]     += stats["ok"]
        total["errors"] += stats["errors"]
        total["files"]  += 1
        total["media"]  += stats.get("media", 0)
    return total


def run_once(
    storage: Path, db, dry_run: bool,
    log: logging.Logger, log_err: logging.Logger, log_warn: logging.Logger,
) -> dict:
    inbox, processing, done = ensure_dirs(storage)
    files = collect_json_files(inbox)
    if not files:
        log.info(_c(C.DIM, "Inbox vide — rien à faire."))
        return {"ok": 0, "errors": 0, "files": 0, "media": 0}
    warn_agg  = WarnAggregator()
    run_label = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    log.info(
        f"{_c(C.BOLD, str(len(files)))} fichier(s) trouvé(s) dans inbox/"
    )
    total = process_batch(files, inbox, processing, done, db, dry_run,
                          log, log_err, log_warn, warn_agg)
    warn_agg.flush(log_warn, run_label)
    return total


def run_watch(
    storage:     Path,
    db,
    dry_run:     bool,
    log:         logging.Logger,
    log_err:     logging.Logger,
    log_warn:    logging.Logger,
    poll:        int = POLL_INTERVAL,
    hb_interval: int = HEARTBEAT_INTERVAL,
):
    inbox, processing, done = ensure_dirs(storage)
    log.info(
        f"{_c(C.BOLD, 'Surveillance inbox')} : "
        f"{_c(C.INFO, str(inbox))}"
        f"{_c(C.DIM, f'  poll={poll}s  heartbeat={hb_interval}s')}"
        f"  {_c(C.DIM, 'Ctrl+C pour arrêter')}"
    )
    hb = Heartbeat(interval=hb_interval)
    try:
        while True:
            files = collect_json_files(inbox)
            if files:
                warn_agg  = WarnAggregator()
                run_label = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                log.info(
                    f"{_c(C.BOLD, str(len(files)))} fichier(s) — "
                    f"batch {_c(C.INFO, run_label)}"
                )
                total = process_batch(files, inbox, processing, done, db, dry_run,
                                      log, log_err, log_warn, warn_agg)
                ok_str  = _c(C.OK,   str(total["ok"]))
                err_str = (_c(C.ERR, str(total["errors"]))
                           if total["errors"] else _c(C.DIM, "0"))
                med_str = (_c(C.CYAN, str(total["media"]))
                           if total["media"]   else _c(C.DIM, "0"))
                log.info(
                    f"Batch terminé — {ok_str} ok  "
                    f"{err_str} erreurs  "
                    f"{med_str} médias  "
                    f"{_c(C.DIM, str(total['files']) + ' fichiers')}"
                    f"{_bar(total['ok'], total['errors'])}"
                )
                warn_agg.flush(log_warn, run_label)
                hb.reset()
            # Attente découpée en tranches de 1s pour que le heartbeat
            # s'affiche à l'intervalle voulu même pendant un long poll.
            elapsed = 0
            while elapsed < poll:
                time.sleep(1)
                elapsed += 1
                hb.tick()
    except KeyboardInterrupt:
        log.info(_c(C.DIM, "Arrêt demandé (Ctrl+C) — worker stoppé proprement."))


# ===========================================================================
# Entrée principale
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Worker import MongoDB — influence_detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Import one-shot depuis le DOSSIER_INPUT
  python worker_import.py --source ~/AI-FORENSICS/DOSSIER_INPUT --dry-run
  python worker_import.py --source ~/AI-FORENSICS/DOSSIER_INPUT

  # Surveillance continue du DOSSIER_INPUT
  python worker_import.py --source ~/AI-FORENSICS/DOSSIER_INPUT --watch
  python worker_import.py --source ~/AI-FORENSICS/DOSSIER_INPUT --watch --poll 30

  # Mode surveillance continue inbox/ (flux scrapper)
  python worker_import.py

  # Mode inbox, traite et quitte
  python worker_import.py --once
        """,
    )
    parser.add_argument(
        "--source", default=None, metavar="DOSSIER",
        help="Import depuis un DOSSIER_INPUT (arborescence projets/scans/users)"
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Mode source continu : surveille le DOSSIER_INPUT en boucle (avec --source)"
    )
    parser.add_argument(
        "--storage", default=None,
        help="Racine du stockage inbox/processing/done (mode scrapper)"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Mode inbox : traite le contenu actuel et quitte"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validation et parsing sans écriture en base"
    )
    parser.add_argument(
        "--poll", type=int, default=None,
        help="Intervalle de polling en secondes (défaut cfg ou 5s)"
    )
    parser.add_argument(
        "--heartbeat", type=int, default=None, metavar="SEC",
        help=f"Intervalle heartbeat en secondes (défaut : {HEARTBEAT_INTERVAL}s)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Mode verbeux (DEBUG)"
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CFG), metavar="FICHIER",
        help=f"Fichier de configuration (défaut : {DEFAULT_CFG})"
    )
    args = parser.parse_args()

    # --- Chargement config ---
    cfg = load_config(Path(args.config))

    # --- Logging ---
    log_level = "DEBUG" if args.verbose else cfg_get(cfg, "worker", "log_level", "INFO")
    log_dir   = cfg_get(cfg, "worker", "log_dir")
    log, log_err, log_warn = setup_logging(log_level, log_dir, args.verbose)

    # --- Bannière de démarrage ---
    log.info(_c(C.BOLD, "━" * 60))
    log.info(
        f"  {_c(C.BOLD + C.CYAN, 'Worker Import MongoDB')}"
        f"  {_c(C.DIM, 'v1.4 — influence_detection')}"
    )
    log.info(_c(C.BOLD, "━" * 60))

    if args.config != str(DEFAULT_CFG):
        log.info(f"Config : {_c(C.DIM, args.config)}")
    elif DEFAULT_CFG.exists():
        log.info(f"Config : {_c(C.DIM, str(DEFAULT_CFG))}")

    if log_dir:
        log.info(f"Logs   : {_c(C.DIM, log_dir)}")

    # --- Résolution des options ---
    source_dir   = args.source or cfg_get(cfg, "storage", "source_dir")
    storage_dir  = (args.storage
                    or cfg_get(cfg, "storage", "storage_dir", str(DEFAULT_STORAGE)))
    poll         = (args.poll
                    or int(cfg_get(cfg, "worker", "poll_interval", str(POLL_INTERVAL))))
    hb_interval  = (args.heartbeat
                    or int(cfg_get(cfg, "worker", "heartbeat_interval",
                                   str(HEARTBEAT_INTERVAL))))
    source_watch = args.watch or (
        cfg_get(cfg, "worker", "source_watch", "false").lower() == "true"
    )

    # --- Connexion MongoDB ---
    if args.dry_run:
        log.info(_c(C.WARN + C.BOLD, "⚠  DRY-RUN — aucune écriture en base"))
        db = None
    else:
        log.info("Connexion à MongoDB...")
        try:
            db = get_db(
                host     = cfg_get(cfg, "mongodb", "host"),
                port     = int(cfg_get(cfg, "mongodb", "port", "27017")),
                user     = cfg_get(cfg, "mongodb", "user"),
                password = cfg_get(cfg, "mongodb", "password"),
                db_name  = cfg_get(cfg, "mongodb", "db"),
                auth_db  = cfg_get(cfg, "mongodb", "auth_db"),
            )
            log.info(_c(C.OK, "✓ Connexion MongoDB OK"))
        except ConnectionError as e:
            log_err.error(f"{_c(C.ERR, '✗')} Connexion MongoDB échouée : {e}")
            sys.exit(1)

    # --- Dispatch selon le mode ---
    if source_dir:
        source = Path(source_dir).expanduser().resolve()
        if not source.exists():
            log_err.error(
                f"{_c(C.ERR, '✗')} Dossier source introuvable : {source}"
            )
            sys.exit(1)
        if source_watch:
            run_source_watch(source, db, args.dry_run,
                             log, log_err, log_warn, poll, hb_interval,
                             log_dir=log_dir)
            return
        else:
            stats = run_source(source, db, args.dry_run, log, log_err, log_warn,
                               log_dir=log_dir)

    elif args.once or args.dry_run:
        storage = Path(storage_dir).expanduser().resolve()
        stats   = run_once(storage, db, args.dry_run, log, log_err, log_warn)

    else:
        storage = Path(storage_dir).expanduser().resolve()
        run_watch(storage, db, args.dry_run, log, log_err, log_warn,
                  poll, hb_interval)
        return

    # Résumé final (modes one-shot uniquement)
    print_final_summary(stats, log)


if __name__ == "__main__":
    main()
