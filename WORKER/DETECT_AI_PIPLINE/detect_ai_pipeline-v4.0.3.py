"""detect_ai_pipeline v4.0.3
==============================

Nouveautés v4.0.3
  1. Mode --mongojob : traitement depuis la file MongoDB jobs
     → À intervalle configurable (--poll), consomme les jobs deepfake_analysis
     → Pour chaque job : analyse le fichier media (image ou vidéo)
     → Écrit les résultats dans media.deepfake (patch_media_deepfake)
     → Met à jour posts.deepfake selon la stratégie "pire cas" :
        final_score = max(score de tous les médias analysés du post)
        status = "done" si tous les médias du post sont traités
     → Gestion des erreurs : skip + log si fichier absent/illisible
     → Heartbeat identique au worker_import (boucle 1s)
     → Log dédié detect_ai_pipeline_error.log pour les erreurs job
  2. Import schema.py depuis ../../SCHEMA/schema.py
     (fallback : dossier du script si absent)
  3. folder est optionnel si et seulement si --mongojob est passé

Nouveautés v4.0.2
  1. Correction bug NaN sur sdxl-detector et swinv2_openfake (local)
     → load_models() reconnaît désormais les chemins locaux SwinV2
       (ex: ./swinv2_openfake) via is_swinv2_model()
     → run_model() route correctement les chemins locaux vers run_swinv2()
     → sdxl-detector : ajout try/except explicite dans _get_ai_score
       pour éviter les scores None silencieux sur frames PIL sans EXIF
  2. decord : num_threads=1 pour éviter le thrashing CPU en mode parallèle
  3. Renommage logs : ai_forensics → detect_ai_pipeline
  4. Correction version affichée dans les logs (v3.4.2 → v4.0.2)
  5. Suppression warnings parasites sauf en --verbose/--debug :
     - CUDA capability sm_120 not compatible
     - pipelines sequentially on GPU

Nouveautés v4.0.1
  1. Extraction de frames via decord (remplacement de ffmpeg)
     → decord fait un seek direct vers les timestamps cibles sans décoder
       toute la vidéo — gain 2-3× sur l'extraction des frames
     → Fallback automatique sur ffmpeg si decord n'est pas installé
       (comportement identique à v4.0, aucune régression)
     → Installation : pip install decord
     → Sur machine sans GPU : decord fonctionne identiquement en CPU
     → get_video_duration() utilise désormais decord en priorité (plus rapide
       que ffprobe), avec fallback ffprobe si decord absent
     → L'argument --use-ffmpeg force le backend ffmpeg même si decord est dispo

v4.0 — Refonte complète de la sélection des modèles. 3 modèles orthogonaux optimisés
pour les générateurs dominants sur les réseaux sociaux en 2025-2026 (FLUX,
Midjourney v7, GPT Image 1, SDXL, Ideogram).

Philosophie : chaque modèle détecte via un mécanisme différent.
              La calibration FP-first pondère selon la performance réelle
              sur votre dataset — minimisation des faux positifs prioritaire.

Modèles v4 :
    1. Organika/sdxl-detector          (CNN diffusion, SD/SDXL/FLUX partiel)
    2. microsoft/swinv2-openfake       (SwinV2 fine-tuné OpenFake, FLUX/MJ/GPT)
       → Si non disponible sur HF, utilise le backbone microsoft/swinv2-small
         à fine-tuner sur votre dataset (voir README)
    3. synthbuster/synthbuster         (Fourier/sklearn, artefacts fréquentiels)

    Slot 4 optionnel — configurable via [models] dans le .cfg
       → laisser vide pour rester à 3 modèles (recommandé si RAM < 24 Go)
       → ajouter ex: Organika/sdxl-detector-v2 ou un modèle custom fine-tuné

Justification du choix :
    - sdxl-detector    → spécialiste SD/SDXL, léger, déjà calibré sur votre dataset
    - SwinV2 OpenFake  → seul modèle public entraîné sur FLUX 1.1-pro, MJ v6,
                         DALL-E 3, Grok-2, Ideogram 3 (dataset OpenFake, McGill/Mila)
                         F1=0.99 in-domain, meilleure généralisation hors-distribution
                         vs tous les modèles HF disponibles (GenImage, Semi-Truths)
    - Synthbuster      → approche physique Fourier, orthogonale aux deux CNN,
                         résilient aux générateurs inconnus qui laissent des
                         artefacts spectraux

Modèles retirés vs v3.x (et pourquoi) :
    - prithivMLmods/Deep-Fake-Detector-v2-Model → poids 0.00 après calibration,
      entraîné face-swap vidéo uniquement, sans valeur sur images T2I
    - umm-maybe/AI-image-detector → spécialisé contenu artistique, inutile sur
      photos RS naturalistes
    - dima806/ai_vs_real_image_detection → trop biaisé (moy.REAL=0.71)
    - Ateeqq/ai-vs-human-image-detector → remplacé par SwinV2 OpenFake (meilleur
      sur générateurs 2025 tout en restant léger)

SwinV2 OpenFake — note d'installation :
    ComplexDataLab ne publie pas les poids fine-tunés sur HuggingFace.
    Deux options :
      A) Utiliser le backbone de base (microsoft/swinv2-small-patch4-window16-256)
         et le fine-tuner sur votre calib_dataset avec le script fourni
         (fine_tune_swinv2.py)
      B) Attendre la publication officielle des poids par ComplexDataLab
         (suivre https://huggingface.co/ComplexDataLab)
    En attendant, le pipeline tourne avec les 2 autres modèles si SwinV2 échoue.

Toutes les fonctionnalités v3.5.x conservées :
    - Extraction adaptative des frames vidéo selon la durée
    - Workers parallèles (--workers N)
    - Calibration bi-dossier FP-first avec backup .cfg
    - Logs enrichis (rotation quotidienne, 7 jours)
    - Score JPEG artefact
    - Support GPU/CPU/nightly CUDA

Installation :
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install transformers accelerate opencv-python pandas tqdm Pillow
    pip install scikit-learn joblib numba imageio timm
    pip install decord                        # optionnel mais recommandé (2-3× plus rapide)
    git clone https://github.com/qbammey/synthbuster synthbuster

Exemples :
    python detect_ai_pipeline-v4.0.2.py ./data --ensemble --workers 8 --verbose
    python detect_ai_pipeline-v4.0.2.py ./calib_dataset --calibrate --workers 8 --verbose
    python detect_ai_pipeline-v4.0.2.py ./data --no-synthbuster --ensemble
    python detect_ai_pipeline-v4.0.2.py ./data --use-ffmpeg --ensemble   # forcer ffmpeg
    CUDA_VISIBLE_DEVICES="" python detect_ai_pipeline-v4.0.2.py ./data --ensemble
"""

from __future__ import annotations

import argparse
import configparser
import json
import logging
import logging.handlers
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Chargement du .env — fait ICI, en tout début de script, avant tout import
# de schema.py et avant la résolution des credentials.
# Chaîne : WORKER/DETECT_AI_PIPLINE/.env → AI-FORENSICS/.env (racine projet)
# override=False : les variables déjà dans os.environ ont priorité (supervisord, etc.)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv as _load_dotenv
    _script_dir = Path(__file__).resolve().parent
    _loaded_local = _load_dotenv(_script_dir / ".env",               override=False)
    _loaded_root  = _load_dotenv(_script_dir.parent.parent / ".env", override=False)
    # Décommentez pour déboguer le chargement .env :
    # import sys
    # print(f"[.env] local={_loaded_local}  racine={_loaded_root}", file=sys.stderr)
except ImportError:
    import sys
    print(
        "[AVERTISSEMENT] python-dotenv non installé — les credentials MongoDB doivent "
        "être définis dans les variables d'environnement système (MONGO_USER, "
        "MONGO_PASSWORD, etc.) ou dans le fichier .cfg.\n"
        "  → Installation : pip install python-dotenv",
        file=sys.stderr,
    )

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import pipeline as hf_pipeline

# ---------------------------------------------------------------------------
# decord — extraction de frames rapide (optionnel, fallback ffmpeg si absent)
# ---------------------------------------------------------------------------
try:
    import decord
    decord.bridge.set_bridge("native")
    _DECORD_AVAILABLE = True
except ImportError:
    _DECORD_AVAILABLE = False

# ---------------------------------------------------------------------------
# Suppression des warnings non-bloquants
# ---------------------------------------------------------------------------
import warnings
import os

# ViTImageProcessor / SiglipImageProcessor : changement de comportement par
# défaut dans transformers >= 4.49 (fast processor). Différence négligeable
# pour la détection deepfake.
warnings.filterwarnings("ignore", message=".*ViTImageProcessor.*fast processor.*")
warnings.filterwarnings("ignore", message=".*SiglipImageProcessor.*fast processor.*")

# scikit-learn : incompatibilité de version sur model.joblib de Synthbuster.
# Résoudre en réentraînant Synthbuster dans l'environnement courant (voir README).
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# HuggingFace Hub : requêtes non authentifiées (rate limit réduit).
# Solution sécurisée : stocker le token dans ~/.huggingface/token
#   mkdir -p ~/.huggingface
#   echo "hf_xxx" > ~/.huggingface/token
#   chmod 600 ~/.huggingface/token
# transformers le détecte automatiquement — aucune modification du code requise.
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

# ---------------------------------------------------------------------------
# Import schema.py — depuis ../../SCHEMA/ en priorité, fallback local
# ---------------------------------------------------------------------------
import sys as _sys
import sys

def _import_schema():
    """
    Cherche schema.py dans cet ordre :
      1. <script_dir>/../../SCHEMA/schema.py  (emplacement standard projet)
      2. <script_dir>/schema.py               (fallback local)
    Retourne le module chargé, ou None si introuvable.
    """
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "SCHEMA" / "schema.py",
        Path(__file__).resolve().parent / "schema.py",
    ]
    for p in candidates:
        if p.exists():
            import importlib.util as _ilu
            spec = _ilu.spec_from_file_location("schema", p)
            mod  = _ilu.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(mod)         # type: ignore[union-attr]
            return mod, p
    return None, None

_schema_mod, _schema_path = _import_schema()
if _schema_mod is not None:
    get_db             = _schema_mod.get_db
    patch_media_deepfake = _schema_mod.patch_media_deepfake
    patch_post_deepfake  = _schema_mod.patch_post_deepfake
    complete_job         = _schema_mod.complete_job
    fail_job             = _schema_mod.fail_job
    claim_job            = _schema_mod.claim_job
    _SCHEMA_AVAILABLE    = True
else:
    _SCHEMA_AVAILABLE    = False
    # Stubs silencieux — le mode --mongojob vérifiera _SCHEMA_AVAILABLE au démarrage
    def get_db(**_kw):             raise ImportError("schema.py introuvable")
    def patch_media_deepfake(*_a): return {}
    def patch_post_deepfake(*_a):  return {}
    def complete_job():            return {}
    def fail_job(*_a, **_kw):      return {}
    def claim_job(*_a):            return {}

# ---------------------------------------------------------------------------
# Logger (configuré plus tard dans setup_logging)
# ---------------------------------------------------------------------------
logger = logging.getLogger("detect_ai_pipeline")

# ---------------------------------------------------------------------------
# Chemins par défaut
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "ai_forensics.cfg"

# ---------------------------------------------------------------------------
# Helper : reconnaît un modèle SwinV2 quel que soit son identifiant
# (chemin HuggingFace "microsoft/swinv2-openfake" OU chemin local "./swinv2_openfake")
# ---------------------------------------------------------------------------
def is_swinv2_model(name: str) -> bool:
    """Retourne True si name désigne le modèle SwinV2 OpenFake (HF ou local)."""
    n = name.lower().replace("\\", "/")
    return (
        n == SWINV2_MODEL_ID.lower()
        or "swinv2_openfake" in n
        or "swinv2-openfake" in n
    )

# ---------------------------------------------------------------------------
# Valeurs de repli — v4.0
# 3 modèles orthogonaux : CNN diffusion + SwinV2 OpenFake + Fourier
# ---------------------------------------------------------------------------

# Identifiant interne du modèle SwinV2 OpenFake
# → Si les poids fine-tunés ComplexDataLab sont publiés sur HF, changer ici
# → Sinon le pipeline utilise le backbone de base (voir fine_tune_swinv2.py)
SWINV2_OPENFAKE_ID  = "microsoft/swinv2-openfake"       # poids fine-tunés (à venir)
SWINV2_BACKBONE_ID  = "microsoft/swinv2-small-patch4-window16-256"  # backbone base
SWINV2_MODEL_ID     = SWINV2_OPENFAKE_ID  # identifiant utilisé dans le pipeline

_FALLBACK_MODELS = [
    "Organika/sdxl-detector",
    SWINV2_MODEL_ID,
    "synthbuster/synthbuster",
]

# Poids initiaux — à recalibrer avec --calibrate sur votre dataset
# Répartition équilibrée en attendant la calibration
_FALLBACK_WEIGHTS: dict[str, float] = {
    "Organika/sdxl-detector": 0.40,
    SWINV2_MODEL_ID:           0.45,
    "synthbuster/synthbuster": 0.15,
}

# Biais initiaux — à mesurer avec --calibrate sur données réelles
# Valeurs conservatives pour minimiser les faux positifs avant calibration
_FALLBACK_BIAS: dict[str, float] = {
    "Organika/sdxl-detector": 0.21,   # valeur mesurée sur votre dataset (v3.5.x)
    SWINV2_MODEL_ID:           0.50,   # à calibrer — valeur neutre par défaut
    "synthbuster/synthbuster": 0.50,   # à recalibrer après réentraînement
}


# ===========================================================================
# Configuration
# ===========================================================================

class ForensicsConfig:
    """Charge la configuration depuis un fichier .cfg INI."""

    def __init__(self, config_path: Path) -> None:
        self._cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
        self._cfg.optionxform = str          # préserve la casse (noms de modèles)
        self._path = config_path

        if config_path.exists():
            self._cfg.read(config_path, encoding="utf-8")
        else:
            # Pas de logger encore — on passe, setup_logging n'a pas été appelé
            pass

    # ------------------------------------------------------------------
    def _get(self, section: str, key: str, fallback: str = "") -> str:
        try:
            return self._cfg.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback

    def _getfloat(self, s: str, k: str, fb: float) -> float:
        try:
            return self._cfg.getfloat(s, k)
        except Exception:
            return fb

    def _getint(self, s: str, k: str, fb: int) -> int:
        try:
            return self._cfg.getint(s, k)
        except Exception:
            return fb

    def _getbool(self, s: str, k: str, fb: bool) -> bool:
        try:
            return self._cfg.getboolean(s, k)
        except Exception:
            return fb

    # ------------------------------------------------------------------
    @property
    def default_models(self) -> list[str]:
        raw = self._get("models", "default_models", "")
        if not raw.strip():
            return list(_FALLBACK_MODELS)
        return [m.strip() for m in raw.replace("\n", ",").split(",") if m.strip()]

    @property
    def weights(self) -> dict[str, float]:
        result = dict(_FALLBACK_WEIGHTS)
        if self._cfg.has_section("weights"):
            for k, v in self._cfg.items("weights"):
                try:
                    result[k] = float(v)
                except ValueError:
                    pass
        return result

    @property
    def bias(self) -> dict[str, float]:
        result = dict(_FALLBACK_BIAS)
        if self._cfg.has_section("bias"):
            for k, v in self._cfg.items("bias"):
                try:
                    result[k] = float(v)
                except ValueError:
                    pass
        return result

    @property
    def threshold_high(self) -> float:
        return self._getfloat("thresholds", "threshold_high", 0.82)

    @property
    def threshold_low(self) -> float:
        return self._getfloat("thresholds", "threshold_low", 0.65)

    @property
    def fps(self) -> int:
        return self._getint("video", "fps", 1)

    @property
    def max_frames(self) -> Optional[int]:
        raw = self._get("video", "max_frames", "").strip()
        return int(raw) if raw else None

    @property
    def adaptive_frames(self) -> bool:
        """Active l'extraction adaptative (ignoré si --fps est passé en CLI)."""
        return self._getbool("video", "adaptive_frames", True)

    @property
    def adaptive_tiers(self) -> list[tuple[float, int]]:
        """
        Paliers (durée_max_secondes, nb_frames) pour l'extraction adaptative.
        Format dans le .cfg :
            adaptive_tiers = 5:3, 15:5, 60:8, 180:12, inf:16
        """
        raw = self._get("video", "adaptive_tiers", "").strip()
        if not raw:
            # Valeurs par défaut calibrées pour les médias RS
            return [(5, 3), (15, 5), (60, 8), (180, 12), (float("inf"), 16)]
        tiers = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                dur_str, n_str = part.split(":")
                dur = float("inf") if dur_str.strip().lower() == "inf" else float(dur_str)
                tiers.append((dur, int(n_str)))
            except ValueError:
                pass
        return tiers if tiers else [(5, 3), (15, 5), (60, 8), (180, 12), (float("inf"), 16)]

    @property
    def output_csv(self) -> str:
        return self._get("output", "output_csv", "results.csv")

    @property
    def json_output(self) -> Optional[str]:
        raw = self._get("output", "json_output", "").strip()
        return raw if raw else None

    @property
    def log_dir(self) -> Path:
        raw = self._get("output", "log_dir", "logs").strip()
        p = Path(raw)
        return p if p.is_absolute() else SCRIPT_DIR / p

    @property
    def mode(self) -> str:
        return self._get("behaviour", "mode", "balanced").strip()

    @property
    def ensemble(self) -> bool:
        return self._getbool("behaviour", "ensemble", True)

    @property
    def require_face(self) -> bool:
        return self._getbool("behaviour", "require_face", False)

    @property
    def bias_correction(self) -> bool:
        return self._getbool("behaviour", "bias_correction", True)

    @property
    def skip_errors(self) -> bool:
        return self._getbool("behaviour", "skip_errors", False)

    @property
    def log_level(self) -> str:
        return self._get("behaviour", "log_level", "WARNING").strip().upper()

    @property
    def divergence_alert_threshold(self) -> float:
        """Écart-type inter-modèles au-dessus duquel une alerte est émise."""
        return self._getfloat("behaviour", "divergence_alert_threshold", 0.20)

    # --- Section [mongodb] — utilisée par le mode --mongojob ---
    @property
    def mongo_host(self) -> Optional[str]:
        v = self._get("mongodb", "host", "").strip()
        return v or None

    @property
    def mongo_port(self) -> Optional[int]:
        v = self._get("mongodb", "port", "").strip()
        return int(v) if v else None

    @property
    def mongo_user(self) -> Optional[str]:
        v = self._get("mongodb", "user", "").strip()
        return v or None

    @property
    def mongo_password(self) -> Optional[str]:
        v = self._get("mongodb", "password", "").strip()
        return v or None

    @property
    def mongo_db(self) -> Optional[str]:
        v = self._get("mongodb", "db", "").strip()
        return v or None

    @property
    def mongo_auth_db(self) -> Optional[str]:
        v = self._get("mongodb", "auth_db", "").strip()
        return v or None

    # --- Section [mongojob] ---
    @property
    def mongojob_poll(self) -> int:
        return self._getint("mongojob", "poll_interval", 10)

    @property
    def mongojob_heartbeat(self) -> int:
        return self._getint("mongojob", "heartbeat_interval", 60)


# ===========================================================================
# Logging enrichi
# ===========================================================================

class SessionStats:
    """Collecte les statistiques de la session courante."""

    def __init__(self) -> None:
        self.start_time = time.monotonic()
        self.files_ok = 0
        self.files_err = 0
        self.label_counts: dict[str, int] = {}
        self.anomalies: list[str] = []
        self.total_score: float = 0.0

    def record(self, row: dict) -> None:
        self.files_ok += 1
        label = row.get("prediction", "unknown")
        self.label_counts[label] = self.label_counts.get(label, 0) + 1
        self.total_score += row.get("final_score", 0.0)

    def record_error(self, filename: str) -> None:
        self.files_err += 1

    def record_anomaly(self, msg: str) -> None:
        self.anomalies.append(msg)
        logger.warning("⚠  ANOMALIE — %s", msg)

    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    def summary(self) -> str:
        elapsed = self.elapsed()
        total = self.files_ok + self.files_err
        fps = self.files_ok / elapsed if elapsed > 0 else 0
        avg_score = self.total_score / self.files_ok if self.files_ok else 0
        lines = [
            "=" * 60,
            "RÉSUMÉ DE SESSION",
            f"  Durée          : {elapsed:.1f}s",
            f"  Fichiers OK    : {self.files_ok}  ({fps:.2f} fichiers/s)",
            f"  Fichiers erreur: {self.files_err}",
            f"  Score moyen    : {avg_score:.4f}",
            "  Distribution   :",
        ]
        for label, count in sorted(self.label_counts.items()):
            pct = 100 * count / self.files_ok if self.files_ok else 0
            lines.append(f"    {label:15s} : {count:4d}  ({pct:.1f}%)")
        if self.anomalies:
            lines.append(f"  Anomalies      : {len(self.anomalies)}")
            for a in self.anomalies[:10]:
                lines.append(f"    • {a}")
            if len(self.anomalies) > 10:
                lines.append(f"    … et {len(self.anomalies) - 10} autres")
        lines.append("=" * 60)
        return "\n".join(lines)


_session: SessionStats = SessionStats()


# ===========================================================================
# Couleurs console — même style que nlp_worker
# ===========================================================================

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
    WARNING  = YELLOW
    INFO_CLR = CYAN
    DEBUG_CLR = DIM
    SUCCESS  = f"{BOLD}{GREEN}"
    HEADER   = f"{BOLD}{CYAN}"


_LEVEL_COLORS = {
    "DEBUG":    C.DIM,
    "INFO":     C.CYAN,
    "WARNING":  C.YELLOW,
    "ERROR":    C.RED,
    "CRITICAL": C.BOLD + C.BG_RED + C.WHITE,
}

import re as _re
_MSG_PATTERNS = [
    (_re.compile(r"(✓[^\n]*)"),                                          C.GREEN),
    (_re.compile(r"(✗[^\n]*)"),                                          C.RED),
    (_re.compile(r"(⚠[^\n]*)"),                                          C.YELLOW),
    (_re.compile(r"(synthetic)"),                                         C.BOLD + C.RED),
    (_re.compile(r"(suspicious)"),                                        C.YELLOW),
    (_re.compile(r"(likely_real)"),                                       C.GREEN),
    (_re.compile(r"(fallback CPU)"),                                      C.YELLOW),
    (_re.compile(r"(CUDA out of memory|Échec|erreur\s+\w+)", _re.I),     C.RED),
    (_re.compile(r"(Modèles prêts[^\n]*)"),                               C.SUCCESS),
    (_re.compile(r"(Analyse terminée[^\n]*)"),                            C.SUCCESS),
    (_re.compile(r"(RÉSUMÉ DE SESSION)"),                                 C.HEADER),
    (_re.compile(r"(CALIBRATION[^\n]*)"),                                 C.HEADER),
]


def _colorize(msg: str) -> str:
    for pattern, color in _MSG_PATTERNS:
        msg = pattern.sub(lambda m, c=color: f"{c}{m.group(1)}{C.RESET}", msg)
    return msg


class ColorFormatter(logging.Formatter):
    """Formatter console avec couleurs ANSI — fichier log reste sans couleurs."""

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        level_color = _LEVEL_COLORS.get(record.levelname, "")
        # Coloriser le niveau
        colored = msg.replace(
            f"[{record.levelname:<8}]",
            f"[{level_color}{record.levelname:<8}{C.RESET}]",
            1,
        )
        # Coloriser le contenu du message
        return _colorize(colored)


def setup_logging(level: int, log_dir: Path) -> None:
    """
    Configure trois handlers :
      - Console       : niveau demandé, avec couleurs ANSI
      - Fichier main  : toujours INFO minimum, sans couleurs,
                        rotation quotidienne, 7 jours de rétention
      - Fichier error : ERROR + CRITICAL uniquement (mode --mongojob)
                        detect_ai_pipeline_error.log
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file  = log_dir / f"detect_ai_pipeline_{datetime.now().strftime('%Y%m%d')}.log"
    err_file  = log_dir / "detect_ai_pipeline_error.log"

    fmt_plain = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fmt_color = ColorFormatter(
        "%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt_color)

    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_file, when="midnight", backupCount=7, encoding="utf-8"
    )
    file_handler.setLevel(min(level, logging.INFO))
    file_handler.setFormatter(fmt_plain)

    # Handler dédié aux erreurs — toujours actif, indépendant du niveau CLI
    class _ErrorFilter(logging.Filter):
        def filter(self, r): return r.levelno >= logging.ERROR

    err_handler = logging.handlers.TimedRotatingFileHandler(
        err_file, when="midnight", backupCount=30, encoding="utf-8"
    )
    err_handler.setLevel(logging.ERROR)
    err_handler.setFormatter(fmt_plain)
    err_handler.addFilter(_ErrorFilter())

    root = logging.getLogger("detect_ai_pipeline")
    root.setLevel(min(level, logging.INFO))
    root.addHandler(console)
    root.addHandler(file_handler)
    root.addHandler(err_handler)

    logger.info("━" * 60)
    logger.info("detect_ai_pipeline v4.0.3 — démarrage")
    logger.info("Log      : %s", log_file)
    logger.info("Log err  : %s", err_file)
    if _SCHEMA_AVAILABLE and _schema_path:
        logger.info("Schema   : %s", _schema_path)


# ===========================================================================
# Heartbeat — mode --mongojob
# ===========================================================================

def _format_idle(seconds: int) -> str:
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
    pendant les périodes d'inactivité en mode --mongojob.
    """
    _TTY_ERR = sys.stderr.isatty() if hasattr(sys.stderr, "isatty") else False

    def __init__(self, interval: int = 60):
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
                f"[2m{ts}[0m "
                f"[35m♥ pipeline actif[0m"
                f"[2m — en attente depuis {idle_str}[0m"
            )
        else:
            line = f"{ts} [HEARTBEAT] pipeline actif — en attente depuis {idle_str}"
        print(line, file=sys.stderr, flush=True)
        self._last_hb = now


# ===========================================================================
# Device
# ===========================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"


# ===========================================================================
# Détection visage
# ===========================================================================
_face_cascade: Optional[cv2.CascadeClassifier] = None


def load_face_detector() -> None:
    global _face_cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    _face_cascade = cv2.CascadeClassifier(cascade_path)
    if _face_cascade.empty():
        logger.warning("Cascade Haar introuvable — has_face retournera toujours True")
        _face_cascade = None
    else:
        logger.info("Détecteur visage Haar Cascade chargé")


def has_face(image: Image.Image) -> bool:
    if _face_cascade is None:
        return True
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
    )
    return len(faces) > 0


# ===========================================================================
# Synthbuster — wrapper (P1)
# ===========================================================================
# Identifiant spécial : "synthbuster/synthbuster"
# Le modèle est un HistGradientBoostingClassifier scikit-learn (pas PyTorch).
# Il opère sur des features FFT extraites de l'image — approche physique bas-niveau,
# orthogonale aux classificateurs CNN HuggingFace.

SYNTHBUSTER_MODEL_ID = "synthbuster/synthbuster"
_synthbuster_model   = None   # HistGradientBoostingClassifier chargé lazily
_synthbuster_cfg     = None   # dict de config (method, rank_sz, max_period…)
_synthbuster_preproc = None   # fonction preprocess_for_fft_features importée du repo


def _find_synthbuster_dir(cfg_synthbuster_dir: Optional[str] = None) -> Optional[Path]:
    """
    Cherche le dossier du repo Synthbuster dans cet ordre :
      1. Valeur de cfg_synthbuster_dir (depuis le .cfg)
      2. SCRIPT_DIR/synthbuster/
      3. Variable d'environnement SYNTHBUSTER_DIR
    """
    candidates = []
    if cfg_synthbuster_dir:
        candidates.append(Path(cfg_synthbuster_dir))
    candidates.append(SCRIPT_DIR / "synthbuster")
    env_dir = os.environ.get("SYNTHBUSTER_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    for p in candidates:
        if p.is_dir() and (p / "inference_common.py").exists():
            return p
    return None


def load_synthbuster(synthbuster_dir: Optional[str] = None) -> bool:
    """
    Charge le modèle Synthbuster pré-entraîné.
    Retourne True si le chargement réussit, False sinon (Synthbuster désactivé).

    Le modèle attendu : <synthbuster_dir>/models/model.joblib
    La config attendue: <synthbuster_dir>/models/config.json

    Si plusieurs .joblib sont présents, prend le premier trouvé.
    """
    global _synthbuster_model, _synthbuster_cfg, _synthbuster_preproc

    sb_dir = _find_synthbuster_dir(synthbuster_dir)
    if sb_dir is None:
        logger.warning(
            "Synthbuster introuvable. Clonez le repo dans ./synthbuster/ :\n"
            "  git clone https://github.com/qbammey/synthbuster synthbuster"
        )
        return False

    # Ajouter le dossier au sys.path pour importer preprocess.py
    import sys
    if str(sb_dir) not in sys.path:
        sys.path.insert(0, str(sb_dir))

    try:
        from preprocess import preprocess_for_fft_features  # type: ignore
        _synthbuster_preproc = preprocess_for_fft_features
    except ImportError as e:
        logger.error("Impossible d'importer preprocess.py de Synthbuster : %s", e)
        logger.error("Installez les dépendances : pip install joblib numba imageio scikit-learn")
        return False

    # Cherche models/model.joblib et models/config.json
    models_dir = sb_dir / "models"
    model_path = models_dir / "model.joblib"
    config_path = models_dir / "config.json"

    if not model_path.exists():
        # Fallback : premier .joblib trouvé
        candidates = list(models_dir.glob("*.joblib")) if models_dir.exists() else []
        if not candidates:
            logger.warning(
                "Aucun modèle Synthbuster trouvé dans %s.\n"
                "Entraînez un modèle d'abord :\n"
                "  cd synthbuster && uv run train_fixed.py --config config.json "
                "--save-model models/model.joblib --save-config models/config.json",
                models_dir,
            )
            return False
        model_path = candidates[0]
        # config associée = même nom mais .json
        config_path = model_path.with_suffix(".json")
        if not config_path.exists():
            # essai du config.json générique
            config_path = models_dir / "config.json"

    try:
        import joblib  # type: ignore
        _synthbuster_model = joblib.load(model_path)
        logger.info("    ✓ Synthbuster modèle chargé : %s", model_path.name)
    except Exception as e:
        logger.error("Erreur chargement modèle Synthbuster : %s", e)
        return False

    # Charger la config (optionnel — valeurs par défaut si absente)
    if config_path.exists():
        try:
            import json as _json
            with open(config_path, encoding="utf-8") as f:
                raw_cfg = _json.load(f)

            # Les paramètres de preprocessing peuvent être à la racine
            # ou imbriqués sous "preprocess" selon la version du script d'entraînement
            preproc = raw_cfg.get("preprocess", raw_cfg)

            _synthbuster_cfg = {
                "method":     preproc.get("method",     raw_cfg.get("method",     "rank")),
                "rank_sz":    int(preproc.get("rank_sz",    raw_cfg.get("rank_sz",    4))),
                "max_period": int(preproc.get("max_period", raw_cfg.get("max_period", 16))),
            }

            # Log de la config réellement utilisée pour faciliter le diagnostic
            logger.info(
                "    ✓ Synthbuster config : method=%s  rank_sz=%d  max_period=%d",
                _synthbuster_cfg["method"],
                _synthbuster_cfg["rank_sz"],
                _synthbuster_cfg["max_period"],
            )
        except Exception as e:
            logger.warning("Config Synthbuster illisible (%s) — valeurs par défaut utilisées", e)
            _synthbuster_cfg = {"method": "rank", "rank_sz": 4, "max_period": 16}
    else:
        _synthbuster_cfg = {"method": "rank", "rank_sz": 4, "max_period": 16}
        logger.info("    ℹ  Synthbuster : config absente — valeurs par défaut (rank, sz=4, P=16)")

    return True


def run_synthbuster(image: Image.Image) -> Optional[float]:
    """
    Lance l'inférence Synthbuster sur une image PIL.
    Retourne P(synthétique) ∈ [0, 1] ou None si indisponible.

    Conversion PIL → numpy uint8 RGB → features FFT → predict_proba[:, 1]
    """
    if _synthbuster_model is None or _synthbuster_preproc is None:
        return None
    try:
        img_np = np.array(image.convert("RGB"), dtype=np.uint8)
        features = _synthbuster_preproc(
            img_np,
            method=_synthbuster_cfg["method"],
            rank_sz=_synthbuster_cfg["rank_sz"],
            max_period=_synthbuster_cfg["max_period"],
        )
        # features : vecteur 1D float64 → reshape (1, N) pour sklearn
        proba = _synthbuster_model.predict_proba(features.reshape(1, -1))
        # proba shape : (1, 2) → P(classe 1 = fake)
        return float(proba[0, 1])
    except Exception as exc:
        logger.debug("Erreur inférence Synthbuster : %s", exc)
        return None


# ===========================================================================
# Score d'artefact — JPEG-aware (P2)
# ===========================================================================

def jpeg_artifact_score(image: Image.Image, path: Optional[Path] = None) -> float:
    """
    Score d'artefact combinant deux signaux complémentaires :

    Signal 1 — Analyse de la table de quantification JPEG (principal)
      Extrait directement du fichier si disponible (évite toute perte).
      Logique :
        • Si table absente (PNG, image IA sans recompression) → score élevé
        • Si table présente mais qualité estimée > 95 → score modéré
          (les images IA stockées en haute qualité ont des tables "trop propres")
        • Si qualité 75–95 (RS typique) → score faible (photo réelle compressée)
        • Si qualité < 75 (très compressé) → score intermédiaire
          (signal ambigu : peut être réel ou IA recompressée)

    Signal 2 — Variance du Laplacien (fallback)
      Utilisé si le fichier n'est pas JPEG ou si l'extraction échoue.
      Mesure la netteté globale / présence d'artefacts de flou.

    Retourne un float ∈ [0, 1].
    """
    JPEG_EXTS = {".jpg", ".jpeg"}

    # ---- Tentative d'extraction de la table JPEG ----
    if path is not None and path.suffix.lower() in JPEG_EXTS:
        try:
            # Ouvre le fichier brut sans décodage pour accéder aux métadonnées
            with Image.open(path) as img_raw:
                qt = img_raw.quantization  # dict {0: [64 coefficients], 1: [...]}

            if qt:
                # Qualité estimée à partir de la table de luminance (composante 0)
                luma_table = qt.get(0, [])
                if luma_table:
                    # La qualité JPEG standard est inversement proportionnelle
                    # aux valeurs de la table de quantification
                    avg_q_val = sum(luma_table) / len(luma_table)

                    # Table de référence qualité 75 : moyenne ≈ 16
                    # Table de référence qualité 95 : moyenne ≈ 4
                    # Table qualité 50 : moyenne ≈ 32
                    # Images IA sans compression : table absente → avg_q_val = 0 ici
                    # Images IA en PNG converties en JPEG : table très basse (haute qualité)

                    if avg_q_val < 3:
                        # Table "presque vide" = qualité quasi-lossless → signal IA
                        return 0.75
                    elif avg_q_val < 6:
                        # Haute qualité (95+) — peu typique d'un vrai RS, signal modéré
                        return 0.55
                    elif avg_q_val <= 20:
                        # Qualité 80–95 : zone normale d'un vrai cliché RS
                        return 0.15
                    elif avg_q_val <= 35:
                        # Qualité 65–80 : très compressé, signal ambigu
                        return 0.35
                    else:
                        # Qualité < 65 : très basse qualité, signal ambigu
                        return 0.40
        except Exception:
            pass  # fallback Laplacien

    # ---- Fallback : variance du Laplacien ----
    try:
        gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalisation : images floues (IA générées) ont un Laplacien bas
        # Images nettes (photos réelles) ont un Laplacien élevé
        # On inverse : score élevé = probable IA (image trop lisse)
        sharpness = float(min(1.0, lap / 500.0))
        return float(max(0.0, 1.0 - sharpness))
    except Exception:
        return 0.5


# ===========================================================================
# SwinV2 OpenFake — wrapper spécialisé
# ===========================================================================
# SwinV2-Small fine-tuné sur OpenFake (McGill/Mila, 2025) :
# Dataset entraîné sur FLUX 1.0/1.1-pro, Midjourney v6, DALL-E 3, Grok-2,
# Ideogram 3, Stable Diffusion XL, et leurs variantes LoRA/FT.
# F1=0.99 in-distribution, meilleure généralisation hors-distribution vs
# tous les modèles HF disponibles (benchmark GenImage + Semi-Truths).
#
# Architecture : SwinV2-Small, entrée 256×256, classification binaire
# (label 0 = real, label 1 = fake dans le modèle OpenFake)
#
# Note : si les poids ComplexDataLab ne sont pas encore publiés sur HF,
# le pipeline bascule automatiquement sur le backbone de base (non fine-tuné)
# avec un avertissement. Utiliser fine_tune_swinv2.py pour fine-tuner sur
# votre calib_dataset.

_swinv2_model     = None   # AutoModelForImageClassification chargé lazily
_swinv2_processor = None   # AutoImageProcessor
_swinv2_available = False


def load_swinv2(model_id: str = SWINV2_MODEL_ID) -> bool:
    """
    Charge le modèle SwinV2 OpenFake.
    Priorité :
      1. model_id si c'est un chemin local (ex: ./swinv2_openfake)
      2. SWINV2_OPENFAKE_ID sur HuggingFace
      3. SWINV2_BACKBONE_ID (backbone de base, non fine-tuné)
    Retourne True si le chargement réussit.
    """
    global _swinv2_model, _swinv2_processor, _swinv2_available

    import warnings
    warnings.filterwarnings("ignore", message=".*SiglipImageProcessor.*")
    warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

    try:
        from transformers import AutoModelForImageClassification, AutoImageProcessor
        import torch as _torch

        # Tentative 1 : chemin local fourni (ex: ./swinv2_openfake)
        local_path = Path(model_id)
        is_local = local_path.exists() and local_path.is_dir()

        if is_local:
            try:
                _swinv2_processor = AutoImageProcessor.from_pretrained(
                    str(local_path), use_fast=True
                )
                _swinv2_model = AutoModelForImageClassification.from_pretrained(
                    str(local_path)
                )
                logger.info("    ✓ SwinV2 local chargé depuis %s", local_path)
            except Exception as exc:
                logger.error("    ✗ Échec chargement local %s : %s", local_path, exc)
                return False
        else:
            # Tentative 2 : poids fine-tunés HuggingFace
            try:
                _swinv2_processor = AutoImageProcessor.from_pretrained(
                    SWINV2_OPENFAKE_ID, use_fast=True
                )
                _swinv2_model = AutoModelForImageClassification.from_pretrained(
                    SWINV2_OPENFAKE_ID
                )
                logger.info("    ✓ SwinV2 OpenFake (poids fine-tunés HF) chargé")
            except Exception:
                # Tentative 3 : backbone de base (non fine-tuné, performances réduites)
                logger.warning(
                    "    ⚠ Poids fine-tunés OpenFake indisponibles sur HF.\n"
                    "      Utilisation du backbone de base (non fine-tuné).\n"
                    "      → Lancez fine_tune_swinv2.py pour améliorer les performances."
                )
                _swinv2_processor = AutoImageProcessor.from_pretrained(
                    SWINV2_BACKBONE_ID, use_fast=True
                )
                _swinv2_model = AutoModelForImageClassification.from_pretrained(
                    SWINV2_BACKBONE_ID
                )
                logger.info("    ✓ SwinV2 backbone de base chargé (à fine-tuner)")

        _swinv2_model.eval()
        if device == "cuda":
            _swinv2_model = _swinv2_model.to("cuda")
            # Test rapide pour détecter CUDA kernel incompatible (sm_120 sur PyTorch stable)
            try:
                import torch as _torch
                _test_input = _swinv2_processor(
                    images=Image.new("RGB", (32, 32)), return_tensors="pt"
                )
                _test_input = {k: v.to("cuda") for k, v in _test_input.items()}
                with _torch.no_grad():
                    _swinv2_model(**_test_input)
            except Exception as _cuda_exc:
                if any(k in str(_cuda_exc).lower() for k in ("cuda", "kernel", "device")):
                    import multiprocessing as _mp
                    _log = logger.debug if _mp.current_process().name != "MainProcess" else logger.warning
                    _log(
                        "    ⚠ SwinV2 GPU incompatible (%s) — fallback CPU",
                        type(_cuda_exc).__name__,
                    )
                    _swinv2_model = _swinv2_model.to("cpu")
                else:
                    raise
        _swinv2_available = True
        return True

    except Exception as exc:
        logger.error("    ✗ SwinV2 — échec chargement : %s", exc)
        return False


def run_swinv2(image: Image.Image) -> Optional[float]:
    """
    Inférence SwinV2 OpenFake sur une image PIL.
    Retourne P(fake) ∈ [0, 1], ou None en cas d'erreur.

    Le modèle OpenFake fine-tuné prédit :
      label 0 → real   (id2label: {0: "LABEL_0"} ou {0: "real"})
      label 1 → fake   (id2label: {1: "LABEL_1"} ou {1: "fake"})

    Pour le backbone de base non fine-tuné, on ne peut pas interpréter
    les sorties comme real/fake — on retourne 0.5 (neutre).
    """
    if not _swinv2_available or _swinv2_model is None:
        return None

    import torch as _torch

    try:
        # Utilise le device réel du modèle (peut avoir basculé en CPU après fallback)
        _model_device = next(_swinv2_model.parameters()).device
        inputs = _swinv2_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(_model_device) for k, v in inputs.items()}

        with _torch.no_grad():
            logits = _swinv2_model(**inputs).logits

        probs = _torch.softmax(logits, dim=-1)[0]
        id2label = _swinv2_model.config.id2label

        # Cherche l'index correspondant à "fake" dans les labels du modèle
        fake_idx = None
        for idx, lbl in id2label.items():
            lbl_clean = str(lbl).lower().replace("-", "").replace("_", "")
            if any(k in lbl_clean for k in ("fake", "synthetic", "ai", "generated")):
                fake_idx = int(idx)
                break

        if fake_idx is not None:
            return float(probs[fake_idx].item())

        # Backbone non fine-tuné : classes ImageNet, pas interprétables
        # Retourne 0.5 (neutre) pour ne pas polluer la calibration
        logger.debug("SwinV2 backbone : labels ImageNet non interprétables → score neutre 0.5")
        return 0.5

    except Exception as exc:
        logger.debug("SwinV2 inférence échouée : %s", exc)
        return None


# ===========================================================================
# Chargement des modèles
# ===========================================================================
_pipelines: dict[str, object] = {}


def load_models(model_names: list[str], synthbuster_dir: Optional[str] = None) -> None:
    global _pipelines
    failed: list[str] = []

    # Séparer les modèles par type
    hf_standard = [
        n for n in model_names
        if n != SYNTHBUSTER_MODEL_ID and not is_swinv2_model(n)
    ]
    logger.info("Chargement des modèles sur %s", device.upper())

    # Modèles HuggingFace standard (pipeline image-classification)
    if hf_standard:
        logger.info("  %d modèle(s) HuggingFace standard", len(hf_standard))
    for name in hf_standard:
        logger.info("  → %s", name)
        try:
            pipe = hf_pipeline(
                "image-classification",
                model=name,
                device=0 if device == "cuda" else -1,
            )
            # Test rapide pour détecter CUDA kernel incompatible (ex: sm_120 sur PyTorch stable)
            if device == "cuda":
                try:
                    from PIL import Image as _PIL
                    pipe(_PIL.new("RGB", (32, 32)))
                except Exception as _cuda_exc:
                    if any(k in str(_cuda_exc).lower() for k in ("cuda", "kernel", "device")):
                        import multiprocessing as _mp
                        _log = logger.debug if _mp.current_process().name != "MainProcess" else logger.warning
                        _log(
                            "    ⚠ GPU incompatible pour %s (%s) — fallback CPU",
                            name, type(_cuda_exc).__name__,
                        )
                        pipe = hf_pipeline(
                            "image-classification",
                            model=name,
                            device=-1,
                        )
                    else:
                        raise
            _pipelines[name] = pipe
            logger.info("    ✓ prêt")
        except Exception as exc:
            logger.error("    ✗ Échec : %s", exc)
            failed.append(name)

    # SwinV2 OpenFake — wrapper dédié (chemin HF ou local)
    swinv2_names = [n for n in model_names if is_swinv2_model(n)]
    for swinv2_name in swinv2_names:
        logger.info("  → %s (SwinV2 OpenFake)", swinv2_name)
        if load_swinv2(model_id=swinv2_name):
            _pipelines[swinv2_name] = "swinv2_loaded"
        else:
            failed.append(swinv2_name)

    # Synthbuster — wrapper Fourier/sklearn
    if SYNTHBUSTER_MODEL_ID in model_names:
        logger.info("  → %s (Fourier/sklearn)", SYNTHBUSTER_MODEL_ID)
        if load_synthbuster(synthbuster_dir):
            _pipelines[SYNTHBUSTER_MODEL_ID] = "synthbuster_loaded"
        else:
            failed.append(SYNTHBUSTER_MODEL_ID)

    load_face_detector()

    if not _pipelines:
        raise RuntimeError("Aucun modèle chargé. Vérifiez votre connexion.")

    logger.info(
        "Modèles prêts : %d/%d  (échecs : %s)",
        len(_pipelines), len(model_names),
        failed if failed else "aucun",
    )


# ===========================================================================
# Inférence
# ===========================================================================

def _get_ai_score(results: list[dict]) -> float:
    ai_kw   = {"artificial", "ai", "fake", "generated", "synthetic", "deepfake", "manipulated"}
    real_kw = {"real", "human", "authentic", "natural", "original", "not_ai", "notai"}
    for r in results:
        lbl = r["label"].lower().replace(" ", "").replace("-", "").replace("_", "")
        if any(k in lbl for k in ai_kw):
            return float(r["score"])
        if any(k in lbl for k in real_kw):
            return float(1.0 - r["score"])
    return float(results[1]["score"] if len(results) >= 2 else results[0]["score"])


def run_model(
    model_name: str,
    image: Image.Image,
    model_bias: dict[str, float],
    apply_bias: bool = True,
) -> Optional[float]:
    """
    Lance l'inférence d'un modèle sur une image PIL.
    Route selon le type de modèle :
      - synthbuster/synthbuster        → run_synthbuster() (Fourier/sklearn)
      - microsoft/swinv2-openfake      → run_swinv2() (SwinV2 wrapper dédié)
      - autres                         → HuggingFace pipeline standard
    Retourne le score corrigé du biais ∈ [0, 1], ou None en cas d'erreur.
    """
    def _apply_bias_correction(raw: float, name: str) -> float:
        if not apply_bias:
            return raw
        bias = model_bias.get(name, 0.0)
        if bias <= 0:
            return raw
        corrected = (raw - bias) / max(1.0 - bias, 0.01)
        return float(max(0.0, min(1.0, corrected)))

    # --- Synthbuster ---
    if model_name == SYNTHBUSTER_MODEL_ID:
        raw = run_synthbuster(image)
        return _apply_bias_correction(raw, model_name) if raw is not None else None

    # --- SwinV2 OpenFake (chemin HF ou local) ---
    if is_swinv2_model(model_name):
        raw = run_swinv2(image)
        return _apply_bias_correction(raw, model_name) if raw is not None else None

    # --- HuggingFace pipeline standard ---
    pipe = _pipelines.get(model_name)
    if pipe is None or pipe in ("synthbuster_loaded", "swinv2_loaded"):
        return None
    try:
        raw = _get_ai_score(pipe(image))   # type: ignore[operator]
        return _apply_bias_correction(raw, model_name)
    except Exception as exc:
        logger.debug("Erreur inférence %s : %s", model_name, exc)
        return None


# ===========================================================================
# Divergence inter-modèles
# ===========================================================================

def model_divergence(scores: dict[str, Optional[float]]) -> float:
    """Écart-type des scores disponibles — mesure la cohérence entre modèles."""
    vals = [v for v in scores.values() if v is not None]
    if len(vals) < 2:
        return 0.0
    arr = np.array(vals, dtype=float)
    return float(np.std(arr))


# ===========================================================================
# Fusion
# ===========================================================================

def combine_scores(
    scores: dict[str, Optional[float]],
    weights: dict[str, float],
    art_score: float,
) -> float:
    total_w = total_s = 0.0
    for name, score in scores.items():
        if score is None:
            continue
        w = weights.get(name, 1.0 / max(len(scores), 1))
        total_s += w * score
        total_w += w
    if total_w == 0:
        return 0.0
    base = (total_s / total_w) * 0.95 + art_score * 0.05
    return float(max(0.0, min(1.0, base)))


# ===========================================================================
# Logging détaillé par fichier
# ===========================================================================

def _log_file_detail(
    filename: str,
    scores: dict[str, Optional[float]],
    raw_scores: dict[str, Optional[float]],
    final_score: float,
    prediction: str,
    divergence: float,
    divergence_threshold: float,
    extra: str = "",
) -> None:
    """
    Écrit dans le log (niveau INFO) les scores détaillés par modèle.
    Émet un WARNING si la divergence dépasse le seuil.
    """
    logger.info("┌─ %s  [%s]  final=%.4f  div=%.4f%s",
                filename, prediction.upper(), final_score, divergence, extra)
    for name, score in scores.items():
        short = name.split("/")[-1][:35]
        raw   = raw_scores.get(name)
        if score is None:
            logger.info("│  %-38s → N/A (non exécuté)", short)
        elif raw is not None and abs(raw - score) > 0.001:
            logger.info("│  %-38s → corr=%.4f  brut=%.4f", short, score, raw)
        else:
            logger.info("│  %-38s → %.4f", short, score)
    logger.info("└─" + "─" * 50)

    if divergence > divergence_threshold:
        logger.warning(
            "⚠  Divergence élevée sur '%s' : écart-type=%.4f > seuil=%.4f "
            "— les modèles sont en désaccord, résultat peu fiable.",
            filename, divergence, divergence_threshold,
        )


# ===========================================================================
# Analyse image
# ===========================================================================

def analyze_image(
    path: Path,
    model_names: list[str],
    weights: dict[str, float],
    model_bias: dict[str, float],
    threshold_high: float,
    threshold_low: float,
    require_face: bool,
    ensemble: bool,
    divergence_threshold: float,
    apply_bias: bool = True,
    stats: Optional[SessionStats] = None,
) -> dict:
    image = Image.open(path).convert("RGB")
    face_found = has_face(image) if require_face else True
    art_val    = jpeg_artifact_score(image, path)   # P2 : analyse JPEG-aware

    if require_face and not face_found:
        logger.info("Pas de visage détecté : %s", path.name)
        if stats:
            stats.record_anomaly(f"Pas de visage : {path.name}")

    scores:     dict[str, Optional[float]] = {}
    raw_scores: dict[str, Optional[float]] = {}

    for name in model_names:
        if require_face and not face_found and "deep" in name.lower():
            scores[name] = raw_scores[name] = None
        else:
            scores[name]     = run_model(name, image, model_bias, apply_bias)
            raw_scores[name] = run_model(name, image, model_bias, False) if apply_bias else scores[name]

    if ensemble:
        final_score = combine_scores(scores, weights, art_val)
    else:
        avail = [s for s in scores.values() if s is not None]
        final_score = float(sum(avail) / len(avail)) if avail else 0.0

    label = (
        "synthetic"   if final_score > threshold_high
        else "suspicious" if final_score > threshold_low
        else "likely_real"
    )

    div = model_divergence(scores)
    _log_file_detail(path.name, scores, raw_scores, final_score, label,
                     div, divergence_threshold)

    row: dict = {
        "source_type":          "image",
        "source":               path.name,
        "face_found":           face_found,
        "jpeg_artifact_score":  round(art_val, 4),   # P2 : nouveau nom de colonne
        "final_score":          round(final_score, 4),
        "model_divergence":     round(div, 4),
        "prediction":           label,
    }
    for name, score in scores.items():
        short = name.split("/")[-1][:35]
        row[f"score_{short}"] = round(score, 4) if score is not None else None
        if apply_bias:
            raw = raw_scores.get(name)
            row[f"raw_{short}"] = round(raw, 4) if raw is not None else None

    if stats:
        stats.record(row)
    return row


# ===========================================================================
# Analyse vidéo
# ===========================================================================

def get_video_duration(video_path: Path) -> Optional[float]:
    """
    Retourne la durée en secondes.
    Priorité : decord (seek direct, plus rapide) → ffprobe (fallback).
    Retourne None si les deux échouent.
    """
    # --- Tentative decord ---
    if _DECORD_AVAILABLE:
        try:
            vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
            fps = vr.get_avg_fps()
            if fps > 0:
                return len(vr) / fps
        except Exception:
            pass

    # --- Fallback ffprobe ---
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        val = result.stdout.strip()
        return float(val) if val else None
    except Exception:
        return None


def _frames_for_duration(
    duration: float,
    tiers: list[tuple[float, int]],
) -> int:
    """
    Retourne le nombre de frames cible selon les paliers configurés.
    Les paliers sont (durée_max, nb_frames), triés par durée croissante.
    """
    for max_dur, n_frames in sorted(tiers, key=lambda x: x[0]):
        if duration <= max_dur:
            return n_frames
    # Fallback : dernier palier
    return sorted(tiers, key=lambda x: x[0])[-1][1]


def extract_frames_decord(
    video_path: Path,
    n_frames_target: int,
    duration: float,
) -> list[Image.Image]:
    """
    Extrait N frames uniformément réparties via decord (seek direct).
    Retourne une liste de PIL.Image.Image directement en mémoire —
    pas de fichiers temporaires sur disque.
    """
    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
    total_frames = len(vr)
    if total_frames == 0:
        return []

    # Indices uniformément répartis sur toute la durée
    indices = [
        int(round(i * (total_frames - 1) / (n_frames_target - 1)))
        for i in range(n_frames_target)
    ] if n_frames_target > 1 else [total_frames // 2]

    # Clamp pour éviter les dépassements
    indices = [min(idx, total_frames - 1) for idx in indices]

    # Lecture batch — decord retourne un NDArray (N, H, W, C) en RGB
    frames_array = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(frame) for frame in frames_array]


def extract_frames(
    video_path: Path,
    tmp_dir: Path,
    fps: float = 1.0,
    max_frames: Optional[int] = None,
    n_frames_target: Optional[int] = None,
    duration: Optional[float] = None,
    use_ffmpeg: bool = False,
) -> list[Path]:
    """
    Extrait des frames d'une vidéo via ffmpeg (écriture sur disque).
    Utilisé comme fallback quand decord n'est pas disponible,
    ou forcé via --use-ffmpeg.

    Modes :
      - n_frames_target fourni : extrait exactement N frames réparties uniformément
        (utilise le filtre ffmpeg select= pour un échantillonnage précis)
      - sinon : utilise fps fixe comme avant

    n_frames_target prend la priorité sur fps si les deux sont fournis.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if n_frames_target and duration and duration > 0:
        interval = duration / n_frames_target
        effective_fps = 1.0 / interval
        vf_filter = f"fps={effective_fps:.6f}"
        logger.info(
            "  Extraction adaptative (ffmpeg) : %.1fs → %d frames (1 frame / %.1fs)",
            duration, n_frames_target, interval,
        )
    else:
        vf_filter = f"fps={fps}"

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vf", vf_filter,
         str(tmp_dir / "%04d.jpg")],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    frames = sorted(tmp_dir.glob("*.jpg"))
    return frames[:max_frames] if max_frames else frames


def analyze_video(
    video_path: Path,
    model_names: list[str],
    weights: dict[str, float],
    model_bias: dict[str, float],
    fps: float,
    max_frames: Optional[int],
    threshold_high: float,
    threshold_low: float,
    require_face: bool,
    ensemble: bool,
    divergence_threshold: float,
    apply_bias: bool = True,
    stats: Optional[SessionStats] = None,
    adaptive: bool = True,
    adaptive_tiers: Optional[list] = None,
    use_ffmpeg: bool = False,
) -> dict:
    # Vérification backend disponible
    if not _DECORD_AVAILABLE or use_ffmpeg:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg absent du PATH et decord non installé")

    # --- Durée et calcul du nombre de frames cible ---
    duration = get_video_duration(video_path)
    n_frames_target: Optional[int] = None

    if adaptive and duration is not None:
        tiers = adaptive_tiers or [(5, 3), (15, 5), (60, 8), (180, 12), (float("inf"), 16)]
        n_frames_target = _frames_for_duration(duration, tiers)
        if max_frames:
            n_frames_target = min(n_frames_target, max_frames)

    # --- Extraction des frames ---
    # decord : frames PIL directement en mémoire, pas de disque
    # ffmpeg : fallback via fichiers temporaires
    frames_pil: list[Image.Image] = []
    backend_used = "ffmpeg"

    if _DECORD_AVAILABLE and not use_ffmpeg and n_frames_target:
        try:
            frames_pil = extract_frames_decord(video_path, n_frames_target, duration or 0.0)
            if max_frames:
                frames_pil = frames_pil[:max_frames]
            backend_used = "decord"
        except Exception as exc:
            logger.warning(
                "decord a échoué sur '%s' (%s) — fallback ffmpeg",
                video_path.name, exc,
            )
            frames_pil = []

    if not frames_pil:
        # Fallback ffmpeg
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg absent du PATH")
        with tempfile.TemporaryDirectory(prefix="ai_forensics_") as tmp:
            frame_paths = extract_frames(
                video_path, Path(tmp),
                fps=fps,
                max_frames=max_frames,
                n_frames_target=n_frames_target,
                duration=duration,
            )
            if not frame_paths:
                raise RuntimeError("Aucune frame extraite")
            frames_pil = [Image.open(fp).convert("RGB") for fp in frame_paths]

    if not frames_pil:
        raise RuntimeError("Aucune frame extraite")

    dur_str = f"{duration:.1f}s" if duration else "durée inconnue"
    mode_str = (
        f"adaptatif ({n_frames_target} frames cibles, {backend_used})"
        if n_frames_target else f"fps={fps} ({backend_used})"
    )
    logger.info("Vidéo '%s' — %s  %d frames extraites  [%s]",
                video_path.name, dur_str, len(frames_pil), mode_str)

    all_scores: dict[str, list[float]] = {n: [] for n in model_names}
    all_raw:    dict[str, list[float]] = {n: [] for n in model_names}
    art_scores: list[float] = []
    face_count = 0
    no_face_frames = 0

    for img in frames_pil:
        img = img.convert("RGB")
        art_scores.append(jpeg_artifact_score(img))
        face_found = has_face(img) if require_face else True
        if face_found:
            face_count += 1
        else:
            no_face_frames += 1

        for name in model_names:
            if require_face and not face_found and "deep" in name.lower():
                continue
            s = run_model(name, img, model_bias, apply_bias)
            if s is not None:
                all_scores[name].append(s)
            if apply_bias:
                r = run_model(name, img, model_bias, False)
                if r is not None:
                    all_raw[name].append(r)

    if require_face and no_face_frames > 0:
        ratio = no_face_frames / len(frames_pil)
        logger.info("  Frames sans visage : %d/%d (%.0f%%)",
                    no_face_frames, len(frames_pil), 100 * ratio)
        if ratio > 0.5 and stats:
            stats.record_anomaly(
                f"{video_path.name} : {100*ratio:.0f}% des frames sans visage détecté"
            )

    avg_scores: dict[str, Optional[float]] = {
        n: float(sum(v) / len(v)) if v else None for n, v in all_scores.items()
    }
    avg_raw: dict[str, Optional[float]] = {
        n: float(sum(v) / len(v)) if v else None for n, v in all_raw.items()
    }
    avg_art = float(sum(art_scores) / len(art_scores)) if art_scores else 0.5

    if ensemble:
        final_score = combine_scores(avg_scores, weights, avg_art)
    else:
        avail = [s for s in avg_scores.values() if s is not None]
        final_score = float(sum(avail) / len(avail)) if avail else 0.0

    label = (
        "synthetic"   if final_score > threshold_high
        else "suspicious" if final_score > threshold_low
        else "likely_real"
    )

    div = model_divergence(avg_scores)
    _log_file_detail(
        video_path.name, avg_scores, avg_raw, final_score, label,
        div, divergence_threshold,
        extra=f"  frames={len(frames_pil)}  faces={face_count}  backend={backend_used}",
    )

    row: dict = {
        "source_type":             "video",
        "source":                  video_path.name,
        "duration_sec":            round(duration, 1) if duration else None,
        "frames_analyzed":         len(frames_pil),
        "faces_detected":          face_count,
        "avg_jpeg_artifact_score": round(avg_art, 4),
        "final_score":             round(final_score, 4),
        "model_divergence":        round(div, 4),
        "prediction":              label,
        "frame_backend":           backend_used,
    }
    for name, score in avg_scores.items():
        short = name.split("/")[-1][:35]
        row[f"score_{short}"] = round(score, 4) if score is not None else None
        if apply_bias:
            raw = avg_raw.get(name)
            row[f"raw_{short}"] = round(raw, 4) if raw is not None else None

    if stats:
        stats.record(row)
    return row


# ===========================================================================
# Worker multiprocessing — traitement parallèle de fichiers
# ===========================================================================
# Chaque worker est un processus indépendant (spawn) qui charge ses propres
# modèles. C'est nécessaire car PyTorch CPU n'est pas thread-safe pour
# l'inférence — le fork de threads partagerait les mêmes modèles et
# produirait des résultats corrompus ou des deadlocks.

def _worker_init(
    model_names: list[str],
    model_bias: dict[str, float],
    synthbuster_dir: Optional[str],
) -> None:
    """Initialise les modèles dans le processus worker (appelé une seule fois)."""
    import warnings as _warnings
    import torch as _torch

    # Supprimer les warnings parasites dans les workers
    # (déjà signalés une fois dans le process principal)
    _warnings.filterwarnings("ignore", message=".*CUDA capability.*")
    _warnings.filterwarnings("ignore", message=".*pipelines sequentially.*")
    _warnings.filterwarnings("ignore", message=".*allow_in_graph is deprecated.*")
    _warnings.filterwarnings("ignore", message=".*ViTImageProcessor.*fast processor.*")
    _warnings.filterwarnings("ignore", message=".*SiglipImageProcessor.*fast processor.*")

    _torch.set_grad_enabled(False)
    load_models(model_names, synthbuster_dir=synthbuster_dir)


def _worker_process_file(args_tuple: tuple) -> Optional[dict]:
    """
    Fonction exécutée par chaque worker pour un fichier.
    Reçoit un tuple de tous les paramètres nécessaires.
    """
    (
        file_path_str, model_names, weights, model_bias,
        threshold_high, threshold_low, require_face, ensemble,
        divergence_threshold, apply_bias, fps, max_frames,
        image_exts, video_exts, adaptive, adaptive_tiers, use_ffmpeg,
    ) = args_tuple

    file_path = Path(file_path_str)
    ext = file_path.suffix.lower()

    try:
        if ext in image_exts:
            return analyze_image(
                file_path, model_names, weights, model_bias,
                threshold_high, threshold_low, require_face, ensemble,
                divergence_threshold, apply_bias,
            )
        elif ext in video_exts:
            return analyze_video(
                file_path, model_names, weights, model_bias,
                fps, max_frames, threshold_high, threshold_low,
                require_face, ensemble, divergence_threshold, apply_bias,
                adaptive=adaptive, adaptive_tiers=adaptive_tiers,
                use_ffmpeg=use_ffmpeg,
            )
    except Exception as exc:
        logger.error("Erreur worker sur %s : %s", file_path.name, exc)
        return {"source_type": "error", "source": file_path.name, "error": str(exc)}
    return None


# ===========================================================================
# Analyse dossier
# ===========================================================================

def analyze_folder(
    folder: Path,
    model_names: list[str],
    weights: dict[str, float],
    model_bias: dict[str, float],
    threshold_high: float,
    threshold_low: float,
    fps: float,
    max_frames: Optional[int],
    skip_errors: bool,
    require_face: bool,
    ensemble: bool,
    divergence_threshold: float,
    apply_bias: bool,
    stats: SessionStats,
    workers: int = 1,
    synthbuster_dir: Optional[str] = None,
    adaptive: bool = True,
    adaptive_tiers: Optional[list] = None,
    use_ffmpeg: bool = False,
) -> pd.DataFrame:
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    video_exts  = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    files = [p for p in sorted(folder.iterdir()) if p.is_file()
             if p.suffix.lower() in image_exts | video_exts]
    if not files:
        logger.warning("Aucun fichier trouvé dans %s", folder)
        return pd.DataFrame()

    # ---- Mode séquentiel (workers=1, comportement original) ----
    if workers <= 1:
        results = []
        for file_path in tqdm(files, desc="Analyse"):
            ext = file_path.suffix.lower()
            try:
                if ext in image_exts:
                    row = analyze_image(
                        file_path, model_names, weights, model_bias,
                        threshold_high, threshold_low, require_face, ensemble,
                        divergence_threshold, apply_bias, stats,
                    )
                else:
                    row = analyze_video(
                        file_path, model_names, weights, model_bias,
                        fps, max_frames, threshold_high, threshold_low,
                        require_face, ensemble, divergence_threshold, apply_bias, stats,
                        adaptive=adaptive, adaptive_tiers=adaptive_tiers,
                        use_ffmpeg=use_ffmpeg,
                    )
                results.append(row)
            except Exception as exc:
                logger.exception("Erreur sur %s : %s", file_path.name, exc)
                stats.record_error(file_path.name)
                if skip_errors:
                    results.append({
                        "source_type": "error", "source": file_path.name, "error": str(exc)
                    })
                else:
                    raise
        return pd.DataFrame(results)

    # ---- Mode parallèle (workers > 1) ----
    import multiprocessing as _mp
    logger.info("Mode parallèle : %d workers", workers)

    # Prépare les tuples d'arguments pour chaque fichier
    task_args = [
        (
            str(f), model_names, weights, model_bias,
            threshold_high, threshold_low, require_face, ensemble,
            divergence_threshold, apply_bias, fps, max_frames,
            image_exts, video_exts, adaptive, adaptive_tiers, use_ffmpeg,
        )
        for f in files
    ]

    results = []
    ctx = _mp.get_context("spawn")

    # Pool avec initialisation des modèles dans chaque worker
    with ctx.Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(model_names, model_bias, synthbuster_dir),
    ) as pool:
        for row in tqdm(
            pool.imap_unordered(_worker_process_file, task_args),
            total=len(files),
            desc=f"Analyse ({workers} workers)",
        ):
            if row is None:
                continue
            if row.get("source_type") == "error":
                stats.record_error(row["source"])
                if not skip_errors:
                    pool.terminate()
                    raise RuntimeError(f"Erreur sur {row['source']} : {row.get('error')}")
            else:
                stats.record(row)
            results.append(row)

    return pd.DataFrame(results)


# ===========================================================================
# Calibration bi-dossier REAL / ALT
# ===========================================================================

def _score_one_file(args_tuple: tuple) -> dict[str, list[float]]:
    """Worker unitaire pour la calibration : score un fichier, retourne ses scores bruts."""
    file_path_str, model_names, model_bias, fps, max_frames = args_tuple
    file_path = Path(file_path_str)
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    video_exts  = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    ext = file_path.suffix.lower()
    images: list = []

    try:
        if ext in image_exts:
            images = [Image.open(file_path).convert("RGB")]
        elif ext in video_exts:
            with tempfile.TemporaryDirectory(prefix="calib_") as tmp:
                frames = extract_frames(file_path, Path(tmp), fps=fps, max_frames=max_frames)
                images = [Image.open(f).convert("RGB") for f in frames]
    except Exception:
        return {n: [] for n in model_names}

    result: dict[str, list[float]] = {n: [] for n in model_names}
    for img in images:
        for name in model_names:
            s = run_model(name, img, model_bias, apply_bias=False)
            if s is not None:
                result[name].append(s)
    return result


def _collect_scores_from_folder(
    folder: Path,
    model_names: list[str],
    model_bias: dict[str, float],
    fps: int,
    max_frames: Optional[int],
    label: str,
    workers: int = 1,
    synthbuster_dir: Optional[str] = None,
) -> dict[str, list[float]]:
    """
    Parcourt un dossier et retourne les scores bruts de chaque modèle.
    Supporte le traitement parallèle via workers > 1.
    """
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    video_exts  = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    files = [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in image_exts | video_exts
    ]
    accum: dict[str, list[float]] = {n: [] for n in model_names}

    if workers <= 1:
        # Mode séquentiel original
        for file_path in tqdm(files, desc=f"  Scoring {label}"):
            ext = file_path.suffix.lower()
            images: list = []
            try:
                if ext in image_exts:
                    images = [Image.open(file_path).convert("RGB")]
                elif ext in video_exts:
                    with tempfile.TemporaryDirectory(prefix="calib_") as tmp:
                        frames = extract_frames(file_path, Path(tmp), fps=fps, max_frames=max_frames)
                        images = [Image.open(f).convert("RGB") for f in frames]
            except Exception:
                continue
            for img in images:
                for name in model_names:
                    s = run_model(name, img, model_bias, apply_bias=False)
                    if s is not None:
                        accum[name].append(s)
        return accum

    # Mode parallèle
    import multiprocessing as _mp
    logger.info("  Calibration %s — %d workers", label, workers)

    task_args = [
        (str(f), model_names, model_bias, fps, max_frames)
        for f in files
    ]
    ctx = _mp.get_context("spawn")
    with ctx.Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(model_names, model_bias, synthbuster_dir),
    ) as pool:
        for partial in tqdm(
            pool.imap_unordered(_score_one_file, task_args),
            total=len(files),
            desc=f"  Scoring {label} ({workers}w)",
        ):
            for name, scores in partial.items():
                accum[name].extend(scores)

    return accum


def _compute_fp_first_score(
    scores_real: list[float], scores_alt: list[float]
) -> dict:
    """
    Calcule le score de pondération selon le critère FP-first :

      Priorité 1 — pénalité faux positifs (éliminatoire)
        fp_penalty = score brut moyen sur REAL
        Un modèle qui score haut sur du vrai est peu fiable quelle que soit
        sa détection : la pénalité est quadratique pour amplifier cet effet.
        facteur_fp = (1 - fp_penalty)²

      Priorité 2 — marge de détection nette
        detect_margin = avg(ALT) − avg(REAL)
        Mesure combien le modèle "voit" vraiment le contenu alteré
        au-delà de son bruit de base sur les vrais contenus.

      Score final = max(0, detect_margin) × facteur_fp
        → 0 si le modèle score autant sur ALT que sur REAL
        → 0 si le modèle est totalement aveugle (fp_penalty ≈ 1)
        → proche de 1 seulement si ALT >> REAL et REAL ≈ 0

    Retourne un dict avec toutes les métriques intermédiaires pour le log.
    """
    if not scores_real:
        return {"fp_penalty": 1.0, "avg_real": 1.0, "avg_alt": 0.0,
                "detect_margin": 0.0, "fp_factor": 0.0, "final_score": 0.0}
    if not scores_alt:
        return {"fp_penalty": 0.0, "avg_real": 0.0, "avg_alt": 0.0,
                "detect_margin": 0.0, "fp_factor": 1.0, "final_score": 0.0}

    avg_real       = sum(scores_real) / len(scores_real)
    avg_alt        = sum(scores_alt)  / len(scores_alt)
    fp_penalty     = avg_real                            # [0, 1]
    fp_factor      = (1.0 - fp_penalty) ** 2            # pénalité quadratique
    detect_margin  = avg_alt - avg_real                  # peut être négatif
    final_score    = max(0.0, detect_margin) * fp_factor

    return {
        "avg_real":      round(avg_real,      4),
        "avg_alt":       round(avg_alt,        4),
        "fp_penalty":    round(fp_penalty,     4),
        "fp_factor":     round(fp_factor,      4),
        "detect_margin": round(detect_margin,  4),
        "final_score":   round(final_score,    4),
    }


def run_calibration(
    calib_dir: Path,
    model_names: list[str],
    model_bias: dict[str, float],
    current_weights: dict[str, float],
    current_bias: dict[str, float],
    fps: int,
    max_frames: Optional[int],
    config_path: Path,
    json_output: Optional[str],
    workers: int = 1,
    synthbuster_dir: Optional[str] = None,
) -> None:
    """
    Calibration bi-dossier :
      <calib_dir>/REAL/  — contenu authentique
      <calib_dir>/ALT/   — contenu IA / altéré

    Critère FP-first :
      Pour chaque modèle :
        biais  = avg(score_brut sur REAL)  → formule de correction du pipeline
        poids  = max(0, avg_ALT − avg_REAL) × (1 − avg_REAL)²
                 normalisé sur l'ensemble des modèles

    Met à jour le .cfg automatiquement (backup horodaté).
    Affiche la comparaison avant/après dans les logs.
    """
    real_dir = calib_dir / "REAL"
    alt_dir  = calib_dir / "ALT"

    for d in (real_dir, alt_dir):
        if not d.is_dir():
            raise ValueError(
                f"Sous-dossier manquant : {d}\n"
                "Le dossier de calibration doit contenir REAL/ et ALT/"
            )

    logger.info("━" * 60)
    logger.info("CALIBRATION (FP-first) — dossier : %s", calib_dir)
    logger.info("  REAL : %s  ALT : %s", real_dir, alt_dir)

    # ------------------------------------------------------------------ scoring
    print("\nÉtape 1/2 — scoring REAL...")
    scores_real = _collect_scores_from_folder(
        real_dir, model_names, model_bias, fps, max_frames, "REAL",
        workers=workers, synthbuster_dir=synthbuster_dir,
    )
    print("Étape 2/2 — scoring ALT...")
    scores_alt = _collect_scores_from_folder(
        alt_dir, model_names, model_bias, fps, max_frames, "ALT",
        workers=workers, synthbuster_dir=synthbuster_dir,
    )

    # --------------------------------------------------------- calcul FP-first
    metrics: dict[str, dict] = {}
    for name in model_names:
        r_vals = scores_real.get(name, [])
        a_vals = scores_alt.get(name,  [])
        m = _compute_fp_first_score(r_vals, a_vals)
        m["n_real"] = len(r_vals)
        m["n_alt"]  = len(a_vals)
        # biais = avg_real (score brut moyen sur contenu réel)
        m["bias"]   = m["avg_real"]
        metrics[name] = m

    # --------------------------------------------------- normalisation des poids
    total = sum(max(m["final_score"], 0.0) for m in metrics.values())
    for name, m in metrics.items():
        if total > 0:
            m["weight"] = round(m["final_score"] / total, 4)
        else:
            # Tous les modèles sont nuls → poids égaux
            m["weight"] = round(1.0 / len(metrics), 4)

    # ----------------------------------------------------------------- affichage
    sep = "=" * 78
    print(f"\n{sep}")
    print("RÉSULTATS DE CALIBRATION — critère FP-first")
    print(
        f"  {'Modèle':40s}  {'moy.REAL':>8}  {'moy.ALT':>7}  "
        f"{'marge':>6}  {'fp²':>5}  {'Poids':>6}  n_r/n_a"
    )
    print("-" * 78)
    for name, m in metrics.items():
        short = name.split("/")[-1][:40]
        print(
            f"  {short:40s}  {m['avg_real']:8.4f}  {m['avg_alt']:7.4f}  "
            f"{m['detect_margin']:+6.4f}  {m['fp_factor']:5.3f}  "
            f"{m['weight']:6.4f}  {m['n_real']}/{m['n_alt']}"
        )
    print(sep)

    # Explication du critère dans le log
    logger.info("━" * 60)
    logger.info("RÉSUMÉ CALIBRATION — critère FP-first")
    logger.info(
        "  %-40s  %8s  %7s  %6s  %5s  %6s",
        "Modèle", "moy.REAL", "moy.ALT", "marge", "fp²", "Poids",
    )
    logger.info("  " + "-" * 60)
    for name, m in metrics.items():
        logger.info(
            "  %-40s  %8.4f  %7.4f  %+6.4f  %5.3f  %6.4f",
            name.split("/")[-1],
            m["avg_real"], m["avg_alt"],
            m["detect_margin"], m["fp_factor"], m["weight"],
        )

    # ----------------------------------------- comparaison avant / après
    logger.info("━" * 60)
    logger.info("COMPARAISON AVANT / APRÈS")
    logger.info(
        "  %-40s  %12s  %12s  %12s  %12s",
        "Modèle", "biais_avant", "biais_après", "poids_avant", "poids_après",
    )
    logger.info("  " + "-" * 70)
    for name, m in metrics.items():
        short = name.split("/")[-1]
        b_old = current_bias.get(name, float("nan"))
        w_old = current_weights.get(name, float("nan"))
        b_new = m["bias"]
        w_new = m["weight"]
        delta_b = f"({b_new - b_old:+.4f})" if not (b_old != b_old) else ""  # NaN guard
        delta_w = f"({w_new - w_old:+.4f})" if not (w_old != w_old) else ""
        logger.info(
            "  %-40s  %6.4f %-7s  %6.4f %-7s  %6.4f %-7s  %6.4f %-7s",
            short,
            b_old, "", b_new, delta_b,
            w_old, "", w_new, delta_w,
        )

    # print comparaison aussi sur console
    print("\nComparaison avant / après :")
    print(
        f"  {'Modèle':40s}  {'biais_avant':>11}  {'biais_après':>11}  "
        f"{'poids_avant':>11}  {'poids_après':>11}"
    )
    print("  " + "-" * 76)
    for name, m in metrics.items():
        short = name.split("/")[-1][:40]
        b_old = current_bias.get(name, float("nan"))
        w_old = current_weights.get(name, float("nan"))
        delta_b = f"{m['bias'] - b_old:+.4f}" if b_old == b_old else "N/A"
        delta_w = f"{m['weight'] - w_old:+.4f}" if w_old == w_old else "N/A"
        print(
            f"  {short:40s}  {b_old:>11.4f}  {m['bias']:>6.4f}({delta_b})  "
            f"{w_old:>11.4f}  {m['weight']:>6.4f}({delta_w})"
        )
    print(sep)

    # ----------------------------------------------- backup + écriture .cfg
    if config_path.exists():
        backup = config_path.with_suffix(
            f".cfg.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        shutil.copy2(config_path, backup)
        print(f"\n  Backup du .cfg → {backup.name}")
        logger.info("Backup .cfg → %s", backup)

    cfg_writer = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    cfg_writer.optionxform = str  # type: ignore[method-assign]
    if config_path.exists():
        cfg_writer.read(config_path, encoding="utf-8")

    if not cfg_writer.has_section("bias"):
        cfg_writer.add_section("bias")
    for name, m in metrics.items():
        cfg_writer.set("bias", name, str(m["bias"]))

    if not cfg_writer.has_section("weights"):
        cfg_writer.add_section("weights")
    for name, m in metrics.items():
        cfg_writer.set("weights", name, str(m["weight"]))

    with open(config_path, "w", encoding="utf-8") as f:
        cfg_writer.write(f)
    print(f"  .cfg mis à jour → {config_path}")
    logger.info(".cfg mis à jour : %s", config_path)
    print(sep)

    # ----------------------------------------------------------- JSON optionnel
    if json_output:
        calib_json = {
            "timestamp":   datetime.now().isoformat(),
            "calib_dir":   str(calib_dir),
            "criterion":   "fp_first",
            "models":      {
                name: {
                    **m,
                    "bias_before":   current_bias.get(name),
                    "weight_before": current_weights.get(name),
                }
                for name, m in metrics.items()
            },
        }
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(calib_json, f, ensure_ascii=False, indent=2)
        print(f"  JSON calibration → {json_output}")
        logger.info("JSON calibration → %s", json_output)


# ===========================================================================
# Mode --mongojob : traitement depuis la file MongoDB jobs
# ===========================================================================

MONGO_JOB_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
MONGO_JOB_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}


def _analyze_file_for_job(
    file_path:    Path,
    model_names:  list[str],
    weights:      dict,
    model_bias:   dict,
    threshold_high: float,
    threshold_low:  float,
    require_face:   bool,
    ensemble:       bool,
    div_threshold:  float,
    apply_bias:     bool,
    fps:            float,
    max_frames:     Optional[int],
    adaptive:       bool,
    adaptive_tiers: Optional[list],
    use_ffmpeg:     bool,
) -> dict:
    """
    Analyse un fichier image ou vidéo et retourne le dict de résultats.
    Factorisation pour le mode --mongojob (fichier unique, pas de dossier).
    """
    ext = file_path.suffix.lower()
    if ext in MONGO_JOB_IMAGE_EXTS:
        return analyze_image(
            file_path, model_names, weights, model_bias,
            threshold_high, threshold_low, require_face, ensemble,
            div_threshold, apply_bias,
        )
    elif ext in MONGO_JOB_VIDEO_EXTS:
        return analyze_video(
            file_path, model_names, weights, model_bias,
            fps, max_frames, threshold_high, threshold_low,
            require_face, ensemble, div_threshold, apply_bias,
            adaptive=adaptive, adaptive_tiers=adaptive_tiers,
            use_ffmpeg=use_ffmpeg,
        )
    else:
        raise ValueError(f"Extension non supportée : {ext}")


def _update_post_worst_case(db, post_id, pipeline_version: str = "4.0.3") -> None:
    """
    Met à jour posts.deepfake selon la stratégie "pire cas" :
      - final_score  = max(deepfake.final_score) parmi tous les médias du post
      - prediction   = label correspondant à ce score max
      - status       = "done" si aucun média n'est encore pending/processing,
                       "processing" sinon

    Appelé après chaque mise à jour d'un document media associé au post.
    """
    try:
        from bson import ObjectId as _OID

        # Récupère tous les media_ids référencés par le post
        post = db.posts.find_one({"_id": post_id}, {"media": 1, "deepfake": 1})
        if not post or not post.get("media"):
            return

        media_ids = [ref.get("media_id") for ref in post["media"] if ref.get("media_id")]
        if not media_ids:
            return

        # Charge les documents media correspondants
        media_docs = list(db.media.find(
            {"_id": {"$in": media_ids}},
            {"deepfake.final_score": 1, "deepfake.prediction": 1, "deepfake.status": 1}
        ))

        if not media_docs:
            return

        # Statuts des médias
        statuses = [m.get("deepfake", {}).get("status", "pending") for m in media_docs]
        all_done = all(s in ("done", "skipped", "error") for s in statuses)

        # Score max (pire cas)
        scores = [
            m.get("deepfake", {}).get("final_score")
            for m in media_docs
            if m.get("deepfake", {}).get("final_score") is not None
        ]
        if not scores:
            return

        max_score = max(scores)
        cfg_th    = db.posts.database.client  # on n'a pas accès au cfg ici → valeurs standard
        th_high   = 0.82
        th_low    = 0.65
        prediction = (
            "synthetic"   if max_score > th_high
            else "suspicious" if max_score > th_low
            else "likely_real"
        )

        db.posts.update_one(
            {"_id": post_id},
            {"$set": {
                "deepfake.final_score":      round(max_score, 4),
                "deepfake.prediction":       prediction,
                "deepfake.status":           "done" if all_done else "processing",
                "deepfake.has_media":        True,
                "deepfake.pipeline_version": pipeline_version,
                "updated_at":                datetime.utcnow(),
            }}
        )
        logger.info(
            "  Post %s — deepfake mis à jour : %s (score=%.4f, %s)",
            post_id, prediction, max_score,
            "done" if all_done else "processing",
        )

    except Exception as exc:
        logger.error("  Erreur mise à jour post %s : %s", post_id, exc)


def _mongojob_worker_process(kwargs: dict) -> None:
    """
    Processus worker indépendant pour le mode --mongojob multiprocessing.

    Appelé via multiprocessing spawn — doit être une fonction top-level.
    Chaque worker :
      1. Charge ses propres modèles (via load_models)
      2. Ouvre sa propre connexion MongoDB
      3. Tourne sa propre boucle run_mongojob() indépendamment
         → claim_job() est atomique : pas de race condition entre workers
    """
    import warnings as _w
    import torch as _torch

    _w.filterwarnings("ignore", message=".*CUDA capability.*")
    _w.filterwarnings("ignore", message=".*pipelines sequentially.*")
    _w.filterwarnings("ignore", message=".*allow_in_graph is deprecated.*")
    _w.filterwarnings("ignore", message=".*ViTImageProcessor.*fast processor.*")
    _w.filterwarnings("ignore", message=".*SiglipImageProcessor.*fast processor.*")

    worker_num      = kwargs.pop("worker_num", 0)
    model_names     = kwargs["model_names"]
    model_bias      = kwargs["model_bias"]
    synthbuster_dir = kwargs.pop("synthbuster_dir", None)
    mongo_kwargs    = kwargs.pop("mongo_kwargs", {})

    _torch.set_grad_enabled(False)
    load_models(model_names, synthbuster_dir=synthbuster_dir)

    try:
        db = get_db(**mongo_kwargs)
    except Exception as exc:
        logger.error("Worker %d — connexion MongoDB échouée : %s", worker_num, exc)
        return

    run_mongojob(db=db, worker_num=worker_num, **kwargs)


def run_mongojob(
    db,
    model_names:      list[str],
    weights:          dict,
    model_bias:       dict,
    threshold_high:   float,
    threshold_low:    float,
    require_face:     bool,
    ensemble:         bool,
    div_threshold:    float,
    apply_bias:       bool,
    fps:              float,
    max_frames:       Optional[int],
    adaptive:         bool,
    adaptive_tiers:   Optional[list],
    use_ffmpeg:       bool,
    poll:             int   = 10,
    hb_interval:      int   = 60,
    pipeline_version: str   = "4.0.3",
    skip_errors:      bool  = False,
    worker_num:       int   = 0,
) -> None:
    """
    Mode --mongojob : boucle de consommation de la file jobs MongoDB.

    Conçu pour tourner en parallèle — chaque instance est un process indépendant.
    Le claim_job() est atomique (findOneAndUpdate) : pas de race condition même
    avec N workers consommant la même file simultanément.

    À chaque cycle :
      1. Claim atomique d'un job deepfake_analysis pending
      2. Vérifie que url_local existe sur le disque → skip si absent
      3. Analyse le fichier (analyze_image ou analyze_video)
      4. Écrit les résultats dans media.deepfake (patch_media_deepfake)
      5. Met à jour posts.deepfake stratégie pire cas (_update_post_worst_case)
      6. Marque le job done ou failed
      Heartbeat sur stderr si aucun job disponible.
    """
    from pymongo.errors import PyMongoError

    wlabel = f"W{worker_num}" if worker_num else "main"
    if worker_num == 0:
        logger.info("━" * 60)
        logger.info("Mode --mongojob — consommation file jobs MongoDB")
        logger.info("  poll=%ds  heartbeat=%ds", poll, hb_interval)
        logger.info("━" * 60)

    hb      = Heartbeat(interval=hb_interval)
    session = SessionStats()
    worker_id = f"pipeline_{wlabel}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        while True:
            # --- Claim atomique d'un job ---
            params = claim_job(worker_id)
            try:
                job = db.jobs.find_one_and_update(
                    params["filter"],
                    params["update"],
                    sort=params["sort"],
                    return_document=True,
                )
            except PyMongoError as exc:
                logger.error("Erreur MongoDB claim_job : %s", exc)
                job = None

            if job is None:
                # Aucun job disponible — heartbeat et attente
                elapsed = 0
                while elapsed < poll:
                    time.sleep(1)
                    elapsed += 1
                    hb.tick()
                continue

            hb.reset()
            job_id   = job["_id"]
            payload  = job.get("payload", {})
            media_id = payload.get("media_id")
            post_id  = payload.get("post_id")
            url_local = payload.get("url_local")
            file_name = payload.get("file_name", "?")

            logger.info("[%s] Job %s — %s", wlabel, job_id, file_name)

            # --- Vérification fichier ---
            if not url_local:
                msg = f"url_local absent dans le payload du job {job_id}"
                logger.error("  ✗ SKIP — %s", msg)
                try:
                    db.jobs.update_one({"_id": job_id},
                                       fail_job(msg, retry=False))
                    if media_id:
                        db.media.update_one(
                            {"_id": media_id},
                            {"$set": {"deepfake.status": "skipped",
                                      "deepfake.error": msg}}
                        )
                except Exception:
                    pass
                continue

            file_path = Path(url_local)
            if not file_path.exists():
                msg = f"Fichier introuvable : {url_local}"
                logger.error("  ✗ SKIP — %s", msg)
                try:
                    db.jobs.update_one({"_id": job_id},
                                       fail_job(msg, retry=False))
                    if media_id:
                        db.media.update_one(
                            {"_id": media_id},
                            {"$set": {"deepfake.status": "skipped",
                                      "deepfake.error": msg}}
                        )
                except Exception:
                    pass
                continue

            # --- Analyse ---
            try:
                result = _analyze_file_for_job(
                    file_path,
                    model_names, weights, model_bias,
                    threshold_high, threshold_low,
                    require_face, ensemble, div_threshold, apply_bias,
                    fps, max_frames, adaptive, adaptive_tiers, use_ffmpeg,
                )
            except Exception as exc:
                msg = f"Analyse échouée : {exc}"
                logger.error("  ✗ Erreur analyse [%s] : %s", file_name, exc)
                try:
                    db.jobs.update_one({"_id": job_id}, fail_job(msg, retry=True))
                    if media_id:
                        db.media.update_one(
                            {"_id": media_id},
                            {"$set": {"deepfake.status": "error",
                                      "deepfake.error": msg}}
                        )
                except Exception:
                    pass
                session.record_error(file_name)
                continue

            # --- Mise à jour media ---
            try:
                if media_id:
                    db.media.update_one(
                        {"_id": media_id},
                        patch_media_deepfake(result, pipeline_version=pipeline_version),
                    )
                    logger.info(
                        "  ✓ media %s — %s (score=%.4f)",
                        media_id, result.get("prediction"), result.get("final_score", 0),
                    )
            except PyMongoError as exc:
                logger.error("  ✗ Erreur écriture media : %s", exc)

            # --- Mise à jour post (pire cas) ---
            try:
                if post_id:
                    _update_post_worst_case(db, post_id, pipeline_version)
            except Exception as exc:
                logger.error("  ✗ Erreur mise à jour post : %s", exc)

            # --- Clore le job ---
            try:
                db.jobs.update_one({"_id": job_id}, complete_job())
            except PyMongoError as exc:
                logger.error("  ✗ Erreur clôture job : %s", exc)

            session.record(result)

    except KeyboardInterrupt:
        logger.info("Arrêt demandé (Ctrl+C) — pipeline stoppé proprement.")

    finally:
        # Résumé de session
        for line in session.summary().splitlines():
            logger.info(line)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    # Pré-parsing pour récupérer --config avant de charger la config
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre.parse_known_args()
    cfg = ForensicsConfig(Path(pre_args.config))

    parser = argparse.ArgumentParser(
        description="detect_ai_pipeline v4.0.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("folder", type=Path, nargs="?", default=None,
                        help="Dossier à analyser (optionnel en mode --mongojob). "
                             "En mode --calibrate : dossier parent contenant REAL/ et ALT/")
    parser.add_argument("--config",          default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output",          default=None,
                        help=f"Fichier CSV (défaut cfg: {cfg.output_csv})")
    parser.add_argument("--json-output",     default=None)
    parser.add_argument("--models",          nargs="+", default=None)
    parser.add_argument("--fps",             type=int,   default=None,
                        help=f"FPS vidéo (défaut cfg: {cfg.fps})")
    parser.add_argument("--max-frames",      type=int,   default=None)
    parser.add_argument("--threshold-high",  type=float, default=None,
                        help=f"Seuil synthetic (défaut cfg: {cfg.threshold_high})")
    parser.add_argument("--threshold-low",   type=float, default=None,
                        help=f"Seuil suspicious (défaut cfg: {cfg.threshold_low})")
    parser.add_argument("--skip-errors",     action="store_true", default=None)
    parser.add_argument("--ensemble",        action="store_true", default=None)
    parser.add_argument("--no-ensemble",     action="store_true")
    parser.add_argument("--require-face",    action="store_true", default=None)
    parser.add_argument("--no-bias-correction", action="store_true")
    parser.add_argument("--calibrate",       action="store_true",
                        help="Mode calibration bi-dossier (REAL/ + ALT/ dans <folder>)")
    parser.add_argument("--mode", choices=["fast", "balanced", "accurate"], default=None,
                        help=f"Preset (défaut cfg: {cfg.mode})")
    # --- P1 : options Synthbuster ---
    parser.add_argument("--no-synthbuster",  action="store_true",
                        help="Désactive Synthbuster même s'il est dans la liste de modèles")
    parser.add_argument("--synthbuster-dir", default=None, metavar="DIR",
                        help="Chemin vers le repo Synthbuster cloné "
                             "(défaut: ./synthbuster/ à côté du script)")
    # --- Parallélisme ---
    parser.add_argument("--workers", type=int, default=None,
                        help="Nombre de processus parallèles (défaut cfg: 1). "
                             "Recommandé : 2 avec 30 Go RAM.")
    # --- Extraction adaptative ---
    parser.add_argument("--no-adaptive", action="store_true",
                        help="Désactive l'extraction adaptative et utilise --fps fixe")
    parser.add_argument("--use-ffmpeg", action="store_true",
                        help="Force le backend ffmpeg pour l'extraction de frames "
                             "(désactive decord même s'il est installé)")
    # --- Mode MongoDB jobs ---
    parser.add_argument("--mongojob", action="store_true",
                        help="Mode file MongoDB : consomme les jobs deepfake_analysis "
                             "en attente au lieu d'analyser un dossier")
    parser.add_argument("--poll", type=int, default=10, metavar="SEC",
                        help="Intervalle de polling en secondes en mode --mongojob (défaut: 10)")
    parser.add_argument("--heartbeat", type=int, default=60, metavar="SEC",
                        help="Intervalle heartbeat en secondes (défaut: 60)")
    parser.add_argument("--mongo-host",     default=None)
    parser.add_argument("--mongo-port",     type=int, default=None)
    parser.add_argument("--mongo-user",     default=None)
    parser.add_argument("--mongo-password", default=None)
    parser.add_argument("--mongo-db",       default=None)
    parser.add_argument("--mongo-auth-db",  default=None)
    parser.add_argument("--verbose",         action="store_true")
    parser.add_argument("--debug",           action="store_true")

    args = parser.parse_args()

    # ---- Warnings parasites — masqués sauf en --verbose / --debug ----
    if not args.verbose and not args.debug:
        warnings.filterwarnings("ignore", message=".*CUDA capability.*")
        warnings.filterwarnings("ignore", message=".*pipelines sequentially.*")
        warnings.filterwarnings("ignore", message=".*allow_in_graph is deprecated.*")

    # ---- Niveau de log ----
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = getattr(logging, cfg.log_level, logging.WARNING)

    setup_logging(log_level, cfg.log_dir)

    # --- Validation --mongojob ---
    if args.mongojob:
        if not _SCHEMA_AVAILABLE:
            logger.error(
                "Mode --mongojob requiert schema.py. "
                "Vérifiez que ../../SCHEMA/schema.py existe."
            )
            raise SystemExit(1)
        if args.folder is not None:
            logger.warning(
                "En mode --mongojob, le dossier '%s' est ignoré.", args.folder
            )
    else:
        # Mode normal : folder obligatoire
        if args.folder is None:
            logger.error("Un dossier est requis sauf en mode --mongojob.")
            raise SystemExit(1)
        if not args.folder.exists() or not args.folder.is_dir():
            logger.error("Dossier introuvable : %s", args.folder)
            raise SystemExit(1)

    # ---- Résolution des paramètres (CLI > cfg > fallback) ----
    mode        = args.mode or cfg.mode
    model_names = args.models or (
        ["Organika/sdxl-detector"] if mode == "fast" else cfg.default_models
    )

    # Retirer Synthbuster si --no-synthbuster
    if args.no_synthbuster and SYNTHBUSTER_MODEL_ID in model_names:
        model_names = [m for m in model_names if m != SYNTHBUSTER_MODEL_ID]
        logger.info("Synthbuster désactivé via --no-synthbuster")

    weights    = cfg.weights
    model_bias = cfg.bias

    threshold_high = args.threshold_high if args.threshold_high is not None else cfg.threshold_high
    threshold_low  = args.threshold_low  if args.threshold_low  is not None else cfg.threshold_low
    fps            = args.fps        if args.fps        is not None else cfg.fps
    max_frames     = args.max_frames if args.max_frames is not None else cfg.max_frames
    # Adaptatif activé par défaut, désactivé si --no-adaptive OU si --fps passé explicitement
    adaptive       = cfg.adaptive_frames and not args.no_adaptive and args.fps is None
    adaptive_tiers = cfg.adaptive_tiers
    apply_bias     = not args.no_bias_correction and cfg.bias_correction
    ensemble       = (not args.no_ensemble) and (args.ensemble or cfg.ensemble)
    require_face   = args.require_face or cfg.require_face
    skip_errors    = args.skip_errors  or cfg.skip_errors
    output_csv     = args.output      or cfg.output_csv
    json_output    = args.json_output or cfg.json_output
    div_threshold  = cfg.divergence_alert_threshold
    synthbuster_dir = args.synthbuster_dir or cfg._get("behaviour", "synthbuster_dir", "")
    workers        = args.workers if args.workers is not None else int(cfg._get("behaviour", "workers", "1"))
    use_ffmpeg     = args.use_ffmpeg

    _backend = "ffmpeg (forcé)" if use_ffmpeg else ("decord" if _DECORD_AVAILABLE else "ffmpeg (decord absent)")
    logger.info(
        "Paramètres — mode=%s  ensemble=%s  bias_correction=%s  "
        "threshold=%.2f/%.2f  fps=%g  adaptive=%s  workers=%d  synthbuster=%s  frames=%s",
        mode, ensemble, apply_bias, threshold_high, threshold_low, fps,
        adaptive, workers, SYNTHBUSTER_MODEL_ID in model_names, _backend,
    )

    load_models(model_names, synthbuster_dir=synthbuster_dir or None)

    if shutil.which("ffmpeg") is None:
        logger.warning("ffmpeg absent du PATH — les vidéos échoueront")

    torch.set_grad_enabled(False)

    # ---- Mode --mongojob ----
    if args.mongojob:
        # Résolution credentials : CLI > cfg > .env / variables d'environnement
        # Priorité : argument CLI > section [mongodb] du .cfg > variables MONGO_* (.env)
        # IMPORTANT : la résolution complète est faite ICI, dans le process principal,
        # avant le spawn des workers. Les workers reçoivent des valeurs explicites dans
        # mongo_kwargs — ils n'appellent pas os.getenv() eux-mêmes (le .env n'est pas
        # rechargé dans les process enfants spawn).
        mongo_host     = (args.mongo_host     or cfg.mongo_host
                          or os.getenv("MONGO_HOST",     "localhost"))
        mongo_port     = (args.mongo_port     or cfg.mongo_port
                          or int(os.getenv("MONGO_PORT", "27017")))
        mongo_user     = (args.mongo_user     or cfg.mongo_user
                          or os.getenv("MONGO_USER",     ""))
        mongo_password = (args.mongo_password or cfg.mongo_password
                          or os.getenv("MONGO_PASSWORD", ""))
        mongo_db       = (args.mongo_db       or cfg.mongo_db
                          or os.getenv("MONGO_DB",       "influence_detection"))
        mongo_auth_db  = (args.mongo_auth_db  or cfg.mongo_auth_db
                          or os.getenv("MONGO_AUTH_DB",  mongo_db))

        # Résolution poll/heartbeat : CLI > cfg > défaut
        mj_poll = args.poll      if args.poll      != 10 else cfg.mongojob_poll
        mj_hb   = args.heartbeat if args.heartbeat != 60 else cfg.mongojob_heartbeat

        logger.info("Connexion MongoDB pour --mongojob...")
        logger.info(
            "  host=%s  port=%s  db=%s  user=%s",
            mongo_host,
            mongo_port,
            mongo_db,
            mongo_user or "(non défini — vérifier .env)",
        )
        try:
            db = get_db(
                host     = mongo_host,
                port     = mongo_port,
                user     = mongo_user,
                password = mongo_password,
                db_name  = mongo_db,
                auth_db  = mongo_auth_db,
            )
            logger.info("✓ Connexion MongoDB OK")
        except Exception as exc:
            logger.error("✗ Connexion MongoDB échouée : %s", exc)
            raise SystemExit(1)

        # Paramètres communs passés à chaque worker
        _mj_kwargs = dict(
            model_names      = model_names,
            weights          = weights,
            model_bias       = model_bias,
            threshold_high   = threshold_high,
            threshold_low    = threshold_low,
            require_face     = require_face,
            ensemble         = ensemble,
            div_threshold    = div_threshold,
            apply_bias       = apply_bias,
            fps              = fps,
            max_frames       = max_frames,
            adaptive         = adaptive,
            adaptive_tiers   = adaptive_tiers,
            use_ffmpeg       = use_ffmpeg,
            poll             = mj_poll,
            hb_interval      = mj_hb,
            pipeline_version = "4.0.3",
            skip_errors      = skip_errors,
            synthbuster_dir  = synthbuster_dir or None,
            mongo_kwargs     = dict(
                host     = mongo_host,
                port     = mongo_port,
                user     = mongo_user,
                password = mongo_password,
                db_name  = mongo_db,
                auth_db  = mongo_auth_db,
            ),
        )

        if workers <= 1:
            # Mode séquentiel — process principal
            run_mongojob(db=db, worker_num=0, **{
                k: v for k, v in _mj_kwargs.items()
                if k not in ("synthbuster_dir", "mongo_kwargs")
            })
        else:
            # Mode parallèle — N workers spawn indépendants
            # Chaque worker charge ses modèles et consomme la file de façon autonome.
            # claim_job() est atomique — pas de race condition.
            import multiprocessing as _mp
            logger.info("Mode --mongojob parallèle : %d workers", workers)
            ctx = _mp.get_context("spawn")
            procs = []
            for i in range(workers):
                kw = dict(_mj_kwargs)
                kw["worker_num"] = i + 1
                p = ctx.Process(
                    target = _mongojob_worker_process,
                    args   = (kw,),
                    name   = f"mongojob_worker_{i+1}",
                    daemon = True,
                )
                p.start()
                procs.append(p)
                logger.info("  Worker %d démarré (pid=%d)", i + 1, p.pid)

            try:
                for p in procs:
                    p.join()
            except KeyboardInterrupt:
                logger.info("Arrêt demandé (Ctrl+C) — arrêt des workers...")
                for p in procs:
                    p.terminate()
                for p in procs:
                    p.join(timeout=5)
                logger.info("Workers arrêtés.")
        return

    # ---- Mode calibration ----
    if args.calibrate:
        run_calibration(
            calib_dir=args.folder,
            model_names=model_names,
            model_bias=model_bias,
            current_weights=weights,
            current_bias=model_bias,
            fps=fps,
            max_frames=max_frames,
            config_path=Path(args.config),
            json_output=json_output,
            workers=workers,
            synthbuster_dir=synthbuster_dir or None,
        )
        return

    # ---- Analyse normale ----
    stats = SessionStats()

    df = analyze_folder(
        folder=args.folder,
        model_names=model_names,
        weights=weights,
        model_bias=model_bias,
        threshold_high=threshold_high,
        threshold_low=threshold_low,
        fps=fps,
        max_frames=max_frames,
        skip_errors=skip_errors,
        require_face=require_face,
        ensemble=ensemble,
        divergence_threshold=div_threshold,
        apply_bias=apply_bias,
        stats=stats,
        workers=workers,
        synthbuster_dir=synthbuster_dir or None,
        adaptive=adaptive,
        adaptive_tiers=adaptive_tiers,
        use_ffmpeg=use_ffmpeg,
    )

    # ---- Sauvegarde CSV ----
    now = datetime.now().strftime("%y%m%d-%H%M%S")
    output_path = Path(f"{now}-{output_csv}")
    df.to_csv(output_path, index=False)

    # ---- Résumé console + log ----
    summary = stats.summary()
    print(f"\n✓ Analyse terminée → {output_path}  ({len(df)} fichiers)")
    if apply_bias:
        print("  (correction de biais activée — utilisez --no-bias-correction pour scores bruts)")
    if "prediction" in df.columns:
        print("\nRésumé des prédictions :")
        for label, count in df["prediction"].value_counts().items():
            print(f"  {label:15s} : {count}")

    # Log du résumé de session complet
    for line in summary.splitlines():
        logger.info(line)

    if json_output:
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
        logger.info("JSON écrit : %s", json_output)


if __name__ == "__main__":
    main()
