"""investigation_agent.py
========================
Agent d'investigation AI-FORENSICS.

Analyse automatique de comptes, posts, narratives ou campagnes suspects
en combinant lecture MongoDB/Neo4j (tools.py) et raisonnement LLM (LangChain).
Produit un rapport Markdown dans AI-FORENSICS/reports/.

Usage :
    python investigation_agent.py --account nolimitmoneyy0 --platform tiktok
    python investigation_agent.py --narrative 69d2774d7dc202593e1b35fa
    python investigation_agent.py --campaign <campaign_id>
    python investigation_agent.py --account nolimitmoneyy0 --platform tiktok \\
        --output ./reports/ --model llama3.1:8b --verbose

Variables d'environnement (.env) :
    AI_PROVIDER=ollama          # ollama | groq | anthropic
    AI_MODEL=llama3.1:8b
    AI_BASE_URL=http://localhost:11434
    AI_API_KEY=                 # pour groq / anthropic
    AI_REPORTS_DIR=./reports
    AI_MAX_TOKENS=4096
    AI_TEMPERATURE=0.2

Dépendances :
    pip install langchain langchain-community langchain-ollama \\
                langchain-groq langchain-anthropic ollama python-dotenv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Chargement .env
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Configuration ai_agent.cfg
# ---------------------------------------------------------------------------
import configparser as _configparser

def _load_agent_cfg() -> _configparser.ConfigParser:
    cfg = _configparser.ConfigParser()
    cfg_path = Path(__file__).resolve().parent / "ai_agent.cfg"
    if cfg_path.exists():
        cfg.read(cfg_path, encoding="utf-8")
    return cfg

_AGENT_CFG = _load_agent_cfg()

def _cfgget(section: str, key: str, default: str = "") -> str:
    try:
        return _AGENT_CFG.get(section, key)
    except Exception:
        return default

def _cfgfloat(section: str, key: str, default: float) -> float:
    try:
        return float(_cfgget(section, key, str(default)))
    except Exception:
        return default

def _cfgint(section: str, key: str, default: int) -> int:
    try:
        return int(_cfgget(section, key, str(default)))
    except Exception:
        return default

# ---------------------------------------------------------------------------
# Import des modules locaux (même dossier AI/)
# ---------------------------------------------------------------------------
_AI_DIR = Path(__file__).resolve().parent
if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))

from tools import (                          # noqa: E402
    get_account_info,
    get_account_posts,
    get_media_scores,
    get_graph_neighbors,
    get_narrative,
    get_campaign_signals,
    get_campaign_graph,
    get_temporal_analysis,
    search_accounts_by_narrative,
    TOOLS_SCHEMA,
)
from prompts import (                        # noqa: E402
    build_system_prompt,
    build_initial_query,
    score_to_label,
    divergence_to_confidence,
    CONFIDENCE_LABELS,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ai_forensics.agent")


# ===========================================================================
# Configuration LLM
# ===========================================================================

def _get_env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


class AgentConfig:
    """Lit la configuration LLM depuis les variables d'environnement."""

    provider:     str
    model:        str
    base_url:     str
    api_key:      str
    max_tokens:   int
    temperature:  float
    reports_dir:  Path

    def __init__(
        self,
        provider:    Optional[str] = None,
        model:       Optional[str] = None,
        output_dir:  Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        self.provider    = provider    or _get_env("AI_PROVIDER",    "ollama")
        self.model       = model       or _get_env("AI_MODEL",       "llama3.1:8b")
        self.base_url    = _get_env("AI_BASE_URL", "http://localhost:11434")
        self.api_key     = _get_env("AI_API_KEY",  "")
        self.max_tokens  = int(_get_env("AI_MAX_TOKENS", "4096"))
        self.temperature = temperature if temperature is not None else float(_get_env("AI_TEMPERATURE", "0.2"))

        _cfg_reports = _cfgget("output", "reports_dir", "").strip()
        _env_reports = _get_env("AI_REPORTS_DIR", "").strip()
        logger.info(f"reports_dir sources — CLI:{output_dir!r} ENV:{_env_reports!r} CFG:{_cfg_reports!r}")
        reports_raw = (
            output_dir                         # 1. --output CLI
            or _env_reports                    # 2. variable d'environnement
            or _cfg_reports                    # 3. ai_agent.cfg
        )
        if reports_raw:
            self.reports_dir = Path(reports_raw).expanduser().resolve()
            logger.info(f"reports_dir retenu : {self.reports_dir}")
        else:
            self.reports_dir = Path(__file__).resolve().parent.parent / "reports"
            logger.info(f"reports_dir par défaut : {self.reports_dir}")

    def __repr__(self) -> str:
        return (
            f"AgentConfig(provider={self.provider}, model={self.model}, "
            f"temperature={self.temperature}, reports_dir={self.reports_dir})"
        )


# ===========================================================================
# Construction du LLM LangChain
# ===========================================================================

def build_llm(cfg: AgentConfig):
    """
    Instancie le LLM LangChain selon le provider configuré.

    Providers supportés : ollama | groq | anthropic
    """
    provider = cfg.provider.lower()

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama  # fallback

        logger.info(f"LLM : Ollama — {cfg.model} @ {cfg.base_url}")
        return ChatOllama(
            model       = cfg.model,
            base_url    = cfg.base_url,
            temperature = cfg.temperature,
            num_predict = cfg.max_tokens,
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq
        if not cfg.api_key:
            raise ValueError("AI_API_KEY requis pour le provider groq")
        logger.info(f"LLM : Groq — {cfg.model}")
        return ChatGroq(
            model_name  = cfg.model,
            api_key     = cfg.api_key,
            temperature = cfg.temperature,
            max_tokens  = cfg.max_tokens,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        if not cfg.api_key:
            raise ValueError("AI_API_KEY requis pour le provider anthropic")
        logger.info(f"LLM : Anthropic — {cfg.model}")
        return ChatAnthropic(
            model_name  = cfg.model,
            api_key     = cfg.api_key,
            temperature = cfg.temperature,
            max_tokens  = cfg.max_tokens,
        )

    else:
        raise ValueError(f"Provider LLM inconnu : {cfg.provider}. Valeurs acceptées : ollama | groq | anthropic")


# ===========================================================================
# Construction des outils LangChain
# ===========================================================================

def build_langchain_tools() -> list:
    """
    Enveloppe les fonctions de tools.py dans des StructuredTool LangChain.
    Utilise TOOLS_SCHEMA pour les descriptions et signatures.
    """
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field, create_model

    lc_tools = []

    # Mapping nom → fonction Python
    _fn_map = {
        "get_account_info":             get_account_info,
        "get_account_posts":            get_account_posts,
        "get_media_scores":             get_media_scores,
        "get_graph_neighbors":          get_graph_neighbors,
        "get_narrative":                get_narrative,
        "get_campaign_signals":         get_campaign_signals,
        "search_accounts_by_narrative": search_accounts_by_narrative,
    }

    # Mapping type JSON Schema → type Python
    _type_map = {"string": str, "integer": int, "number": float, "boolean": bool}

    for schema in TOOLS_SCHEMA:
        name        = schema["name"]
        description = schema["description"]
        props       = schema["parameters"].get("properties", {})
        required    = schema["parameters"].get("required", [])
        fn          = _fn_map[name]

        # Construction dynamique du modèle Pydantic pour les arguments
        fields = {}
        for param_name, param_def in props.items():
            py_type = _type_map.get(param_def.get("type", "string"), str)
            if param_name not in required:
                py_type = Optional[py_type]
                fields[param_name] = (py_type, Field(default=None, description=param_def.get("description", "")))
            else:
                fields[param_name] = (py_type, Field(description=param_def.get("description", "")))

        ArgsModel = create_model(f"{name}_args", **fields)

        # Wrapper qui sérialise le résultat en JSON lisible par le LLM
        def make_wrapper(f):
            def wrapper(**kwargs):
                # Retire les arguments None optionnels
                kwargs = {k: v for k, v in kwargs.items() if v is not None}
                try:
                    result = f(**kwargs)
                    return json.dumps(result, ensure_ascii=False, indent=2, default=str)
                except Exception as exc:
                    logger.error(f"Erreur outil {f.__name__} : {exc}", exc_info=True)
                    return json.dumps({"error": str(exc), "tool": f.__name__})
            wrapper.__name__ = f.__name__
            return wrapper

        lc_tools.append(
            StructuredTool(
                name        = name,
                description = description,
                func        = make_wrapper(fn),
                args_schema = ArgsModel,
            )
        )

    logger.info(f"{len(lc_tools)} outils LangChain construits")
    return lc_tools


# ===========================================================================
# Agent ReAct
# ===========================================================================

def _collect_account(entry_value: str, platform: str, neo4j_available: bool) -> dict:
    """Collecte déterministe pour un point d'entrée compte."""
    data = {}
    logger.info("Collecte : get_account_info")
    info = get_account_info(platform, entry_value)

    # Fallback : si non trouvé par username, cherche par display_name dans MongoDB
    if "error" in info:
        logger.info(f"Recherche par username échouée, tentative par display_name...")
        try:
            from tools import _get_db
            db = _get_db()
            acc = db.accounts.find_one(
                {"platform": platform, "display_name": entry_value},
                {"platform_id": 1, "username": 1}
            )
            if acc:
                alt_id = acc.get("username") or acc.get("platform_id")
                logger.info(f"Compte trouvé via display_name → username={alt_id}")
                info = get_account_info(platform, alt_id)
            else:
                # Dernière tentative : recherche insensible à la casse
                import re
                acc = db.accounts.find_one(
                    {"platform": platform,
                     "display_name": {"$regex": f"^{re.escape(entry_value)}$", "$options": "i"}},
                    {"platform_id": 1, "username": 1}
                )
                if acc:
                    alt_id = acc.get("username") or acc.get("platform_id")
                    logger.info(f"Compte trouvé via display_name (insensible casse) → {alt_id}")
                    info = get_account_info(platform, alt_id)
        except Exception as e:
            logger.warning(f"Fallback display_name échoué : {e}")

    data["account"] = info
    if "error" in info:
        logger.warning(f"Compte introuvable : {info['error']}")
        return data

    account_id  = info.get("account_id")
    platform_id = info.get("platform_id")

    logger.info("Collecte : get_account_posts")
    data["posts"] = get_account_posts(account_id, limit=20)

    logger.info("Collecte : get_media_scores")
    data["media"] = get_media_scores(account_id)

    if neo4j_available and platform_id:
        logger.info("Collecte : get_graph_neighbors")
        data["graph"] = get_graph_neighbors(platform_id, depth=2)

    # Narratives identifiées dans les posts
    narrative_ids = list({
        p["nlp"]["narrative_id"]
        for p in data.get("posts", [])
        if p.get("nlp", {}).get("narrative_id")
    })
    data["narratives"] = []
    for nid in narrative_ids[:3]:  # max 3 narratives
        logger.info(f"Collecte : get_narrative({nid[:8]}...)")
        data["narratives"].append(get_narrative(nid))

    # Campagnes
    campaign_ids = info.get("flags", {}).get("campaign_ids", [])
    data["campaigns"] = []
    for cid in campaign_ids[:2]:  # max 2 campagnes
        logger.info(f"Collecte : get_campaign_signals({cid[:8]}...)")
        data["campaigns"].append(get_campaign_signals(cid))

    # Analyse temporelle du compte (via platform_id pour les requêtes Neo4j)
    if platform_id:
        logger.info("Collecte : get_temporal_analysis (account)")
        data["temporal"] = get_temporal_analysis(platform_id, entry_type="account")
    return data


def _collect_narrative(entry_value: str, neo4j_available: bool) -> dict:
    """Collecte déterministe pour un point d'entrée narrative."""
    data = {}
    logger.info("Collecte : get_narrative")
    data["narrative"] = get_narrative(entry_value)

    logger.info("Collecte : search_accounts_by_narrative")
    accounts_list = search_accounts_by_narrative(entry_value)
    data["accounts_list"] = accounts_list

    # Détaille les 2 comptes les plus actifs
    data["accounts_detail"] = []
    for acc in accounts_list[:2]:
        aid = acc.get("account_id")
        plat = acc.get("platform", "tiktok")
        if aid:
            logger.info(f"Collecte : get_account_info({acc.get('username', aid[:8])})")
            info = get_account_info(plat, acc.get("username") or aid)
            data["accounts_detail"].append({
                "info":  info,
                "media": get_media_scores(aid),
            })
    # Analyse temporelle de la narrative
    if entry_value:
        logger.info("Collecte : get_temporal_analysis (narrative)")
        data["temporal"] = get_temporal_analysis(entry_value, entry_type="narrative")
    return data


def _collect_campaign(entry_value: str) -> dict:
    """Collecte déterministe pour un point d'entrée campagne."""
    data = {}

    logger.info("Collecte : get_campaign_signals")
    data["campaign"] = get_campaign_signals(entry_value)

    logger.info("Collecte : get_campaign_graph (Neo4j)")
    data["campaign_graph"] = get_campaign_graph(entry_value)

    logger.info("Collecte : get_temporal_analysis (campaign)")
    data["temporal"] = get_temporal_analysis(entry_value, entry_type="campaign")

    # Détaille les 2 comptes les plus actifs (depuis le graphe Neo4j si disponible)
    data["accounts_detail"] = []
    graph_accounts = (data["campaign_graph"].get("accounts") or [])[:2]
    if graph_accounts:
        for acc in graph_accounts:
            uid  = acc.get("username") or acc.get("platform_id")
            plat = acc.get("platform", "tiktok")
            if uid:
                logger.info(f"Collecte : get_account_info({uid})")
                info = get_account_info(plat, uid)
                if "error" not in info:
                    data["accounts_detail"].append({
                        "info":  info,
                        "media": get_media_scores(info.get("account_id", "")),
                    })
    else:
        # Fallback MongoDB si Neo4j vide
        for acc in (data["campaign"].get("accounts_sample") or [])[:2]:
            aid  = acc.get("account_id")
            plat = acc.get("platform", "tiktok")
            if aid:
                logger.info(f"Collecte : get_account_info({acc.get('username', aid[:8])})")
                info = get_account_info(plat, acc.get("username") or aid)
                data["accounts_detail"].append({
                    "info":  info,
                    "media": get_media_scores(aid),
                })
    return data


def _collect_post(entry_value: str, neo4j_available: bool) -> dict:
    """Collecte déterministe pour un point d'entrée post."""
    # On cherche le post en MongoDB pour récupérer account_id
    from tools import _get_db
    from bson import ObjectId
    db = _get_db()
    post = db.posts.find_one({"_id": ObjectId(entry_value)})
    if not post:
        return {"error": f"Post introuvable : {entry_value}"}
    account_id = str(post.get("account_id", ""))
    platform   = post.get("platform", "tiktok")
    acc = db.accounts.find_one({"_id": post["account_id"]}, {"username": 1, "platform_id": 1})
    uid = (acc or {}).get("username") or (acc or {}).get("platform_id") or account_id
    return _collect_account(uid, platform, neo4j_available)


# ===========================================================================
# Calcul du score de suspicion global
# ===========================================================================

def compute_suspicion_score(intermediate_steps: list) -> tuple[float, str]:
    """
    Calcule un score de suspicion global [0-1] à partir des résultats
    des outils collectés pendant l'investigation.

    Retourne (score, confidence_label).
    """
    scores       = []
    divergences  = []
    has_campaign = False
    has_reuse    = False

    for _tool_name, observation in intermediate_steps:
        try:
            data = json.loads(observation) if isinstance(observation, str) else observation
        except Exception:
            continue

        # get_account_info → deepfake_summary
        if isinstance(data, dict) and "deepfake_summary" in data:
            ds = data["deepfake_summary"]
            if ds.get("avg_score") is not None:
                scores.append(ds["avg_score"])
            if ds.get("synthetic_ratio") is not None:
                scores.append(ds["synthetic_ratio"])
            if data.get("flags", {}).get("campaign_ids"):
                has_campaign = True

        # get_media_scores → liste de médias
        if isinstance(data, list) and data and "final_score" in (data[0] if data else {}):
            for m in data[:20]:
                if m.get("final_score") is not None:
                    scores.append(m["final_score"])
                if m.get("model_divergence") is not None:
                    divergences.append(m["model_divergence"])
                if (m.get("reuse") or {}).get("seen_count", 1) > 1:
                    has_reuse = True

        # get_campaign_signals → confidence
        if isinstance(data, dict) and "signals" in data:
            conf = (data.get("review") or {}).get("confidence")
            if conf is not None:
                scores.append(float(conf))
            has_campaign = True

        # get_narrative → synthetic_ratio
        if isinstance(data, dict) and "stats" in data and "synthetic_ratio" in (data.get("stats") or {}):
            sr = data["stats"]["synthetic_ratio"]
            if sr is not None:
                scores.append(float(sr))

    # Score de base : moyenne des scores collectés
    base_score = sum(scores) / len(scores) if scores else 0.0

    # Bonus pour signaux de coordination
    if has_campaign:
        base_score = min(1.0, base_score + 0.15)
    if has_reuse:
        base_score = min(1.0, base_score + 0.10)

    # Niveau de confiance selon divergence
    avg_div = sum(divergences) / len(divergences) if divergences else None
    conf_key = divergence_to_confidence(avg_div)
    conf_label = CONFIDENCE_LABELS.get(conf_key, "Moyen")

    return round(base_score, 2), conf_label


# ===========================================================================
# Sauvegarde du rapport
# ===========================================================================

def save_report(
    content:      str,
    reports_dir:  Path,
    entry_type:   str,
    entry_value:  str,
) -> Path:
    """
    Sauvegarde le rapport Markdown dans reports_dir.
    Nom du fichier : YYYY-MM-DD_HHMMSS_<type>_<cible>.md
    """
    reports_dir.mkdir(parents=True, exist_ok=True)

    ts    = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    # Nettoie entry_value pour le nom de fichier
    safe  = re.sub(r"[^\w\-]", "_", entry_value)[:40]
    fname = f"{ts}_{entry_type}_{safe}.md"
    path  = reports_dir / fname

    path.write_text(content, encoding="utf-8")
    return path





# ===========================================================================
# Chargement de la configuration agent
# ===========================================================================

def _load_agent_cfg(cfg_path: Optional[Path] = None) -> dict:
    """Charge ai_agent.cfg et retourne un dict de dicts par section."""
    import configparser
    cfg = configparser.ConfigParser()
    candidates = [
        cfg_path,
        Path(__file__).parent / "ai_agent.cfg",
        Path(__file__).parent.parent / "CONTEXT" / "ai_agent.cfg",
    ]
    for p in candidates:
        if p and p.exists():
            cfg.read(p)
            break
    # Convertir en dict simple
    result = {}
    for section in cfg.sections():
        result[section] = dict(cfg[section])
    return result


# ===========================================================================
# Enrichissement adaptatif
# ===========================================================================

def _enrich_if_needed(raw_data: dict, entry_type: str, agent_cfg: dict) -> dict:
    """
    Inspecte les données collectées et déclenche des appels supplémentaires
    selon les seuils définis dans ai_agent.cfg [enrichment].

    Règles :
      - Score deepfake élevé → search_accounts_by_narrative
      - Duplications > threshold → temporal_analysis des comptes sources
      - Communauté > min_size → lister les comptes suspects de la communauté
      - Cross-campagne détecté → pas d'appel supplémentaire (déjà dans temporal)

    Retourne raw_data enrichi.
    """
    ecfg = agent_cfg.get("enrichment", {})
    bot_threshold   = float(ecfg.get("bot_score_threshold", 0.7))
    dup_threshold   = int(ecfg.get("duplicate_threshold", 3))
    comm_min        = int(ecfg.get("community_min_size", 5))
    df_threshold    = float(ecfg.get("deepfake_alert_threshold", 0.65))
    max_calls       = int(ecfg.get("max_extra_calls", 4))
    extra_calls     = 0

    # --- Règle 1 : score deepfake élevé sur les médias ---
    media = raw_data.get("media", [])
    if media and extra_calls < max_calls:
        avg_df = sum(m.get("final_score") or 0 for m in media) / max(1, len(media))
        if avg_df > df_threshold:
            # Chercher d'autres comptes portant les mêmes narratives
            narratives = raw_data.get("narratives", [])
            for narr in narratives[:2]:
                narr_id = narr.get("mongo_id") or narr.get("_id")
                if narr_id and extra_calls < max_calls:
                    logger.info(f"Enrichissement : deepfake élevé ({avg_df:.3f}) → search_accounts_by_narrative")
                    extra = search_accounts_by_narrative(str(narr_id))
                    if "error" not in extra:
                        raw_data.setdefault("related_accounts", []).extend(
                            extra.get("accounts", [])[:5]
                        )
                    extra_calls += 1

    # --- Règle 2 : duplications importantes → analyser les sources ---
    graph = raw_data.get("graph", {})
    if graph and extra_calls < max_calls:
        dup_from = graph.get("duplicated_from", [])
        for dup in dup_from:
            if int(dup.get("duplicate_count", 0)) > dup_threshold and extra_calls < max_calls:
                source_pid = dup.get("platform_id")
                if source_pid:
                    logger.info(f"Enrichissement : {dup.get('duplicate_count')} duplications → temporal source @{dup.get('username','?')}")
                    extra_temporal = get_temporal_analysis(source_pid, entry_type="account")
                    if "error" not in extra_temporal:
                        raw_data.setdefault("source_temporals", {})[dup.get("username", source_pid)] = extra_temporal
                    extra_calls += 1

    # --- Règle 3 : communauté Louvain grande → chercher les suspects ---
    if graph and extra_calls < max_calls:
        comm_accounts = graph.get("community_accounts", [])
        if len(comm_accounts) >= comm_min:
            suspicious_in_comm = [a for a in comm_accounts if a.get("is_suspicious")]
            if suspicious_in_comm:
                logger.info(f"Enrichissement : {len(comm_accounts)} comptes en communauté, {len(suspicious_in_comm)} suspects")
                raw_data["suspicious_community"] = suspicious_in_comm[:10]
                extra_calls += 1

    # --- Règle 4 (campaign) : comptes cross-campagne → investiguer top amplificateur ---
    if entry_type == "campaign" and extra_calls < max_calls:
        cross = raw_data.get("temporal", {}).get("cross_campaign", [])
        if cross:
            top = cross[0]
            uid = top.get("account")
            plat = top.get("platform", "tiktok")
            if uid and uid not in [a.get("display_name") for a in
                                    [d.get("info",{}) for d in raw_data.get("accounts_detail",[])]]:
                logger.info(f"Enrichissement : amplificateur cross-campagne → get_account_info @{uid}")
                info = get_account_info(plat, uid)
                if "error" not in info:
                    raw_data.setdefault("accounts_detail", []).append({
                        "info": info, "media": [], "note": "amplificateur cross-campagne"
                    })
                extra_calls += 1

    if extra_calls > 0:
        logger.info(f"Enrichissement terminé — {extra_calls} appel(s) supplémentaire(s)")

    return raw_data


# ===========================================================================
# Suggestions de scrapping
# ===========================================================================

def _build_scraping_suggestions(raw_data: dict, entry_type: str, agent_cfg: dict) -> str:
    """
    Génère une section Markdown de suggestions de scrapping actionnables
    à partir des données collectées. Entièrement calculé en Python (pas le LLM).
    """
    scfg = agent_cfg.get("scraping_suggestions", {})
    min_copies_high  = int(scfg.get("min_copies_for_priority_high", 5))
    min_hashtags     = int(scfg.get("min_shared_hashtags", 10))
    cross_auto       = str(scfg.get("cross_platform_auto_suggest", "true")).lower() == "true"
    max_accs         = int(scfg.get("max_account_suggestions", 8))
    max_tags         = int(scfg.get("max_hashtag_suggestions", 5))

    high, medium, watch = [], [], []

    # --- Comptes prioritaires ---
    # Compte semence
    temporal = raw_data.get("temporal", {})
    seed = temporal.get("seed_account")
    if seed:
        high.append({
            "cible": f"@{seed.get('account','?')}",
            "type": "compte",
            "plateformes": [seed.get("platform", "?")],
            "motif": f"Compte semence — premier à publier le {str(seed.get('first_pub',''))[:10]}",
            "mots_clés": [],
        })

    # Comptes robotiques
    for acc in temporal.get("summary", {}).get("robotic_accounts", [])[:3]:
        if not any(s["cible"] == f"@{acc}" for s in high):
            medium.append({
                "cible": f"@{acc}",
                "type": "compte",
                "plateformes": ["?"],
                "motif": "Cadence robotique détectée (≥1 post/jour sur >20 jours)",
            })

    # Comptes copiant ce compte (amplificateurs)
    graph = raw_data.get("graph", {})
    copied_by = graph.get("copied_by", [])
    for d in copied_by[:min(3, max_accs)]:
        copies = int(d.get("copy_count", 0))
        prio = high if copies >= min_copies_high else medium
        if not any(s["cible"] == f"@{d.get('username','?')}" for s in high+medium):
            prio.append({
                "cible": f"@{d.get('username','?')}",
                "type": "compte",
                "plateformes": [d.get("platform", "?")],
                "motif": f"Copie {copies} posts de la cible",
            })

    # Comptes cross-campagne
    cross = temporal.get("cross_campaign", [])
    for x in cross[:3]:
        acc = x.get("account", "?")
        if not any(s["cible"] == f"@{acc}" for s in high+medium):
            medium.append({
                "cible": f"@{acc}",
                "type": "compte",
                "plateformes": [x.get("platform", "?")],
                "motif": f"Actif sur {x.get('campaign_count', x.get('posts','?'))} autres campagnes",
            })

    # Comptes amplificateurs campaign_graph
    cg = raw_data.get("campaign_graph", {})
    for acc in (cg.get("accounts", []) or [])[:5]:
        name = acc.get("username", "?")
        if acc.get("duplicate_count", 0) > 0 and not any(s["cible"] == f"@{name}" for s in high+medium):
            medium.append({
                "cible": f"@{name}",
                "type": "compte",
                "plateformes": [acc.get("platform", "?")],
                "motif": f"{acc.get('duplicate_count',0)} posts dupliqués, {acc.get('post_count',0)} posts dans la campagne",
            })

    # --- Hashtags ---
    hashtags = graph.get("shared_hashtags", []) or cg.get("hashtags", [])
    for h in hashtags[:max_tags]:
        tag   = h.get("tag") or h.get("hashtag", "?")
        usage = h.get("usage", 0)
        if int(usage) >= min_hashtags:
            high.append({
                "cible": f"#{tag}",
                "type": "hashtag",
                "plateformes": [],
                "motif": f"{usage} usages détectés dans la campagne",
            })
        elif int(usage) >= 3:
            watch.append({
                "cible": f"#{tag}",
                "type": "hashtag",
                "plateformes": [],
                "motif": f"{usage} usages",
            })

    # --- Cross-plateforme ---
    camp_summary = cg.get("summary", {})
    if cross_auto and camp_summary.get("has_cross_platform"):
        platforms = camp_summary.get("platforms", [])
        if len(platforms) > 1:
            watch.append({
                "cible": "Campagne cross-plateforme",
                "type": "expansion",
                "plateformes": platforms,
                "motif": f"Actifs sur {', '.join(platforms)} — élargir le scrapping à toutes les plateformes",
            })

    # --- Comptes avec silences suspects ---
    for acc_name, gaps in temporal.get("silences", {}).items():
        max_gap = max((g.get("days", 0) for g in gaps), default=0)
        if max_gap > 20 and not any(s["cible"] == f"@{acc_name}" for s in high+medium+watch):
            watch.append({
                "cible": f"@{acc_name}",
                "type": "surveillance",
                "plateformes": [],
                "motif": f"Silence de {max_gap} jours — possible rotation de gestionnaire",
            })

    # --- Mise en forme Markdown ---
    if not high and not medium and not watch:
        return ""

    lines = ["\n## Suggestions de scrapping\n"]
    lines.append("> Générées automatiquement à partir des signaux détectés. "
                 "À valider par un analyste avant envoi au scrapper.\n")

    def _fmt_table(items: list, show_motif: bool = True) -> list[str]:
        rows = ["| Cible | Type | Plateformes | Motif |",
                "|---|---|---|---|"]
        for item in items[:max_accs]:
            plats = ", ".join(item.get("plateformes", [])) or "?"
            motif = item.get("motif", "")
            rows.append(f"| `{item['cible']}` | {item['type']} | {plats} | {motif} |")
        return rows

    if high:
        lines.append("\n### 🔴 Priorité haute\n")
        lines.extend(_fmt_table(high))
        lines.append("")

    if medium:
        lines.append("\n### 🟡 Priorité moyenne\n")
        lines.extend(_fmt_table(medium))
        lines.append("")

    if watch:
        lines.append("\n### 🟢 Surveillance continue\n")
        lines.extend(_fmt_table(watch))
        lines.append("")

    return "\n".join(lines)


def _build_llm_context(raw_data: dict, suspicion_score: float, confidence: str, neo4j_available: bool) -> str:
    """
    Construit un résumé compact et structuré des données collectées
    pour le LLM — évite la saturation du contexte avec du JSON brut.
    """
    lines = []

    # --- Compte ---
    acc = raw_data.get("account", {})
    if acc and "error" not in acc:
        lines.append("## COMPTE ANALYSÉ")
        lines.append(f"- Plateforme     : {acc.get('platform', 'N/A')}")
        lines.append(f"- Username       : {acc.get('display_name') or acc.get('username', 'N/A')}")
        lines.append(f"- URL            : {acc.get('url', 'N/A')}")
        lines.append(f"- Vérifié        : {acc.get('profile', {}).get('verified', False)}")
        lines.append(f"- Followers      : {acc.get('stats', {}).get('followers_count', 0)}")
        ds = acc.get("deepfake_summary", {})
        if ds:
            lines.append(f"- Posts analysés : {ds.get('posts_analyzed', 0)}")
            lines.append(f"- Score deepfake moyen : {ds.get('avg_score', 'N/A')}")
            lines.append(f"- Posts synthétiques   : {ds.get('synthetic_count', 0)} ({ds.get('synthetic_ratio', 0)*100:.1f}%)")
            lines.append(f"- Posts suspects       : {ds.get('suspicious_count', 0)}")
        flags = acc.get("flags", {})
        lines.append(f"- Suspect flagué : {flags.get('is_suspicious', False)}")
        lines.append(f"- Campagnes liées: {len(flags.get('campaign_ids', []))}")
        lines.append("")

    # --- Posts (top 5) ---
    posts = raw_data.get("posts", [])
    if posts:
        lines.append("## DERNIERS POSTS (top 5 sur {})".format(len(posts)))
        for p in posts[:5]:
            df = p.get("deepfake", {})
            nlp = p.get("nlp", {})
            sent = nlp.get("sentiment", {})
            lines.append(f"- [{p.get('published_at', '')[:10]}] score={df.get('score', 'N/A')} pred={df.get('prediction', 'N/A')} div={df.get('divergence', 'N/A')} sentiment={sent.get('label', 'N/A')} | {p.get('text_preview', '')[:80]}")
        lines.append("")

    # --- Médias (stats agrégées) ---
    media = raw_data.get("media", [])
    if media:
        lines.append("## MÉDIAS ANALYSÉS ({} médias)".format(len(media)))
        scores = [m.get("final_score") for m in media if m.get("final_score") is not None]
        divs   = [m.get("model_divergence") for m in media if m.get("model_divergence") is not None]
        synth  = [m for m in media if m.get("prediction") == "synthetic"]
        susp   = [m for m in media if m.get("prediction") == "suspicious"]
        reused = [m for m in media if (m.get("reuse") or {}).get("seen_count", 1) > 1]
        if scores:
            lines.append(f"- Score moyen          : {sum(scores)/len(scores):.3f}")
            lines.append(f"- Score max            : {max(scores):.3f}")
        if divs:
            lines.append(f"- Divergence moyenne   : {sum(divs)/len(divs):.3f}")
        lines.append(f"- Synthétiques         : {len(synth)} ({len(synth)/len(media)*100:.1f}%)")
        lines.append(f"- Suspects             : {len(susp)} ({len(susp)/len(media)*100:.1f}%)")
        lines.append(f"- Médias réutilisés    : {len(reused)}")
        # Top 3 médias par score
        top3 = sorted(media, key=lambda m: m.get("final_score") or 0, reverse=True)[:3]
        lines.append("- Top 3 scores :")
        for m in top3:
            df = m.get("scores_by_model", {})
            lines.append(f"  * score={m.get('final_score')} div={m.get('model_divergence')} sdxl={df.get('sdxl-detector','?')} swinv2={df.get('swinv2-openfake','?')} synth={df.get('synthbuster','?')}")
        lines.append("")

    # --- Graphe Neo4j ---
    graph = raw_data.get("graph", {})
    if graph and "error" not in graph:
        s = graph.get("summary", {})
        lines.append("## RÉSEAU NEO4J")
        lines.append(f"- Comptes même communauté Louvain : {s.get('community_size', 0)}")
        lines.append(f"- Comptes partageant ≥2 hashtags  : {s.get('shared_hashtag_accounts', 0)}")
        lines.append(f"- Sources de duplication (copie de) : {s.get('duplicate_sources', 0)}")
        lines.append(f"- Comptes copiant ce compte        : {s.get('copiers', 0)}")
        lines.append(f"- Hashtags distincts utilisés      : {s.get('hashtag_count', 0)}")
        lines.append(f"- Narratives liées                 : {s.get('narrative_count', 0)}")
        lines.append(f"- Campagnes détectées              : {s.get('campaign_count', 0)}")

        # Communauté Louvain
        community = graph.get("community_accounts", [])
        if community:
            lines.append(f"- Communauté ({len(community)} comptes, top 5 par PageRank) :")
            for n in community[:5]:
                susp = " ⚠️ suspect" if n.get("is_suspicious") else ""
                lines.append(f"  * {n.get('username','?')} ({n.get('platform','?')}) pagerank={n.get('pagerank','?')}{susp}")

        # Comptes partageant des hashtags
        sha = graph.get("shared_hashtag_accounts", [])
        if sha:
            lines.append(f"- Comptes avec hashtags communs (top 5) :")
            for n in sha[:5]:
                lines.append(f"  * {n.get('username','?')} ({n.get('platform','?')}) — {n.get('shared_tags','?')} hashtags en commun")

        # Duplication de contenu
        dup_from = graph.get("duplicated_from", [])
        if dup_from:
            lines.append(f"- Ce compte copie du contenu de :")
            for d in dup_from[:5]:
                lines.append(f"  * {d.get('username','?')} ({d.get('platform','?')}) — {d.get('duplicate_count','?')} posts copiés")
        copied_by = graph.get("copied_by", [])
        if copied_by:
            lines.append(f"- Ce compte est copié par :")
            for d in copied_by[:5]:
                lines.append(f"  * {d.get('username','?')} ({d.get('platform','?')}) — {d.get('copy_count','?')} copies")

        # Hashtags fréquents
        hashtags = graph.get("shared_hashtags", [])
        if hashtags:
            lines.append("- Hashtags fréquents : " + ", ".join(
                f"#{h.get('tag','?')}({h.get('usage','?')})" for h in hashtags[:10]
            ))

        # Narratives et campagnes
        narratives_g = graph.get("narratives", [])
        if narratives_g:
            lines.append(f"- Narratives ({len(narratives_g)}) :")
            for n in narratives_g[:5]:
                camp_info = f" → Campagne: {n.get('campaign_name','?')} (score={n.get('campaign_score','?')})" if n.get('campaign_name') else ""
                lines.append(f"  * {n.get('label','?')} — {n.get('account_posts','?')} posts{camp_info}")

        campaigns_g = graph.get("campaigns", [])
        if campaigns_g:
            lines.append(f"- CAMPAGNES DÉTECTÉES ({len(campaigns_g)}) :")
            for c in campaigns_g:
                lines.append(f"  * {c.get('name','?')} score={c.get('score','?')} signaux={c.get('signal_count','?')}")
        lines.append("")
    else:
        lines.append("## RÉSEAU NEO4J")
        lines.append("- Relations non encore synchronisées dans Neo4j (ETL en attente)")
        lines.append("")

    # --- Narrative principale (pour --narrative) ---
    narrative = raw_data.get("narrative", {})
    if narrative and "error" not in narrative:
        lines.append("## NARRATIVE ANALYSÉE")
        stats = narrative.get("stats", {})
        lines.append(f"- Label          : {narrative.get('label', 'N/A')}")
        lines.append(f"- Posts          : {stats.get('post_count', 'N/A')}")
        lines.append(f"- Comptes        : {stats.get('account_count', 'N/A')}")
        lines.append(f"- Ratio synth.   : {stats.get('synthetic_ratio', 'N/A')}")
        lines.append(f"- Plateformes    : {', '.join(stats.get('platforms', []))}")
        lines.append(f"- Mots-clés      : {', '.join(narrative.get('keywords', [])[:8])}")
        lines.append(f"- Première vue   : {str(stats.get('first_seen_at', ''))[:10]}")
        lines.append(f"- Dernière vue   : {str(stats.get('last_seen_at', ''))[:10]}")
        top = narrative.get("top_accounts", [])
        if top:
            lines.append(f"- Top comptes ({len(top)}) :")
            for a in top[:5]:
                lines.append(f"  * {a.get('username','?')} ({a.get('platform','?')}) — {a.get('post_count','?')} posts, score deepfake moyen: {a.get('avg_deepfake_score','N/A')}")
        lines.append("")

    # --- Comptes associés à la narrative (pour --narrative) ---
    accounts_list = raw_data.get("accounts_list", [])
    if accounts_list:
        lines.append("## COMPTES PORTANT LA NARRATIVE ({})".format(len(accounts_list)))
        for a in accounts_list[:10]:
            ns = a.get("narrative_stats", {})
            lines.append(f"- @{a.get('display_name') or a.get('username','?')} ({a.get('platform','?')})")
            lines.append(f"  Posts: {ns.get('post_count','?')} | Synthétiques: {ns.get('synthetic_posts','?')} ({ns.get('synthetic_ratio',0)*100:.1f}%) | Score moyen: {ns.get('avg_deepfake_score','N/A')}")
        lines.append("")

    # --- Campagne MongoDB (pour --campaign) ---
    campaign = raw_data.get("campaign", {})
    if campaign and "error" not in campaign:
        sig = campaign.get("signals", {})
        rev = campaign.get("review", {})
        lines.append("## CAMPAGNE — SIGNAUX MONGODB")
        lines.append(f"- Nom            : {campaign.get('name', 'N/A')}")
        lines.append(f"- Statut         : {campaign.get('status', 'N/A')}")
        lines.append(f"- Confiance      : {rev.get('confidence', 'N/A')}")
        lines.append(f"- Confirmée      : {rev.get('confirmed', False)}")
        lines.append(f"- Posting coordonné    : {sig.get('coordinated_posting', 'N/A')}")
        lines.append(f"- Réutilisation contenu: {sig.get('content_reuse', 'N/A')}")
        lines.append(f"- Ratio bots           : {sig.get('bot_accounts_ratio', 'N/A')}")
        lines.append(f"- Ratio médias synth.  : {sig.get('synthetic_media_ratio', 'N/A')}")
        lines.append(f"- Cross-plateforme     : {sig.get('cross_platform', 'N/A')}")
        lines.append(f"- Burst Telegram       : {sig.get('telegram_forward_burst', 'N/A')}")
        lines.append(f"- Notes          : {rev.get('notes', '')}")
        lines.append("")

    # --- Graphe de la campagne Neo4j (pour --campaign) ---
    cg = raw_data.get("campaign_graph", {})
    if cg and "error" not in cg:
        s = cg.get("summary", {})
        lines.append("## CAMPAGNE — GRAPHE NEO4J")
        lines.append(f"- Comptes impliqués   : {s.get('account_count', 0)}")
        lines.append(f"- Plateformes         : {', '.join(s.get('platforms', []))}")
        lines.append(f"- Cross-plateforme    : {s.get('has_cross_platform', False)}")
        lines.append(f"- Total posts         : {s.get('total_posts', 0)}")
        lines.append(f"- Doublons détectés   : {s.get('total_duplicates', 0)}")
        lines.append(f"- Score deepfake moyen: {s.get('avg_deepfake_score', 'N/A')}")
        # Narratives
        for n in cg.get("narratives", []):
            lines.append(f"- Narrative : {n.get('label','?')} ({n.get('post_count','?')} posts)")
            lines.append(f"  Mots-clés : {', '.join((n.get('keywords') or [])[:8])}")
            lines.append(f"  Signaux   : {', '.join(n.get('signals') or [])}")
        # Comptes avec stats
        lines.append(f"- Comptes ({len(cg.get('accounts', []))}) :")
        for a in cg.get("accounts", []):
            susp = " SUSPECT" if a.get("is_suspicious") else ""
            avg_df = round(a.get("avg_deepfake"), 3) if a.get("avg_deepfake") else "N/A"
            lines.append(
                f"  * @{a.get('username','?')} ({a.get('platform','?')}) "
                f"posts={a.get('post_count',0)} deepfake={avg_df} "
                f"doublons={a.get('duplicate_count',0)} "
                f"community={a.get('community_id','?')}{susp}"
            )
        # Hashtags dominants
        htags = cg.get("hashtags", [])
        if htags:
            lines.append("- Hashtags dominants : " + ", ".join(
                f"#{h.get('hashtag','?')}({h.get('usage','?')})" for h in htags[:12]
            ))
        # Duplication de contenu
        dups = cg.get("duplicates", [])
        if dups:
            lines.append(f"- Duplication de contenu ({len(dups)} paires) :")
            for d in dups:
                lines.append(
                    f"  * @{d.get('copier','?')} ({d.get('copier_platform','?')}) "
                    f"copie @{d.get('original_author','?')} ({d.get('original_platform','?')}) "
                    f"— {d.get('copies','?')} fois"
                )
        # Communautés Louvain
        comms = cg.get("communities", [])
        if len(comms) > 1:
            lines.append(
                f"- Communautés Louvain : {len(comms)} communautés distinctes "
                f"({'coordination faible' if len(comms) > 3 else 'coordination possible'})"
            )
        lines.append("")

    # --- Comptes détaillés (pour --narrative et --campaign) ---
    accounts_detail = raw_data.get("accounts_detail", [])
    if accounts_detail:
        lines.append("## COMPTES ANALYSÉS EN DÉTAIL ({})".format(len(accounts_detail)))
        for entry in accounts_detail:
            info = entry.get("info", {})
            media = entry.get("media", [])
            if "error" in info:
                continue
            ds = info.get("deepfake_summary", {})
            lines.append(f"- @{info.get('display_name') or info.get('username','?')} ({info.get('platform','?')})")
            lines.append(f"  Posts analysés: {ds.get('posts_analyzed','?')} | Score moyen: {ds.get('avg_score','N/A')} | Synthétiques: {ds.get('synthetic_count',0)}")
            if media:
                m_scores = [m.get("final_score") for m in media if m.get("final_score") is not None]
                m_synth  = [m for m in media if m.get("prediction") == "synthetic"]
                score_moy = f"{sum(m_scores)/len(m_scores):.3f}" if m_scores else "N/A"
                lines.append(f"  Médias: {len(media)} | Score moy: {score_moy} | Synthétiques: {len(m_synth)}")
        lines.append("")

    # --- Narratives liées à un compte (pour --account) ---
    narratives = raw_data.get("narratives", [])
    if narratives:
        lines.append("## NARRATIVES ({} identifiées)".format(len(narratives)))
        for n in narratives:
            if "error" not in n:
                stats = n.get("stats", {})
                lines.append(f"- {n.get('label', 'N/A')}")
                lines.append(f"  Posts: {stats.get('post_count', 'N/A')} | Ratio synthétique: {stats.get('synthetic_ratio', 'N/A')}")
                lines.append(f"  Mots-clés: {', '.join(n.get('keywords', [])[:6])}")
        lines.append("")

    # --- Analyse temporelle ---
    temporal = raw_data.get("temporal", {})
    if temporal and "error" not in temporal:
        s = temporal.get("summary", {})
        lines.append("## ANALYSE TEMPORELLE")

        # Vue d'ensemble
        lines.append(f"- Période couverte        : {s.get('first_post','?')} → {s.get('last_post','?')}")
        lines.append(f"- Jours d'activité totaux : {s.get('total_days_active', 0)}")
        lines.append(f"- Jours de co-occurrence  : {s.get('cooccurrence_days', 0)}")
        lines.append(f"- Co-occurrence max        : {s.get('max_cooccurrence', 0)} comptes le même jour")

        # Compte semence
        seed = temporal.get("seed_account")
        if seed:
            lines.append(
                f"- COMPTE SEMENCE : @{seed.get('account','?')} ({seed.get('platform','?')}) "
                f"— premier post le {str(seed.get('first_pub',''))[:10]} "
                f"({seed.get('posts',0)} posts au total)"
            )

        # Comptes robotiques
        robotic = s.get("robotic_accounts", [])
        if robotic:
            lines.append(f"- Cadence robotique détectée : {', '.join('@'+r for r in robotic)}")

        # Silences suspects
        silences = temporal.get("silences", {})
        if silences:
            lines.append("- Silences suspects (gaps > 14 jours) :")
            for acc, gaps in silences.items():
                for g in gaps[:2]:
                    lines.append(
                        f"  * @{acc} : {g.get('gap_start','?')} → {g.get('gap_end','?')} "
                        f"({g.get('days','?')} jours) — possible rotation de gestionnaire"
                    )

        # Mois de pic
        peaks = s.get("peak_months", [])
        if peaks:
            lines.append(f"- Mois de forte accélération : {', '.join(peaks)}")

        # Corrélation deepfake × temporel
        df_spikes = temporal.get("deepfake_temporal", [])
        if df_spikes:
            lines.append("- Pics de deepfake corrélés aux pics d'activité :")
            for m in df_spikes:
                lines.append(
                    f"  * {m.get('month','?')} : score deepfake moy={round((m.get('avg_deepfake') or 0), 3)} "
                    f"sur {m.get('posts',0)} posts ({m.get('synthetic_count',0)} synthétiques)"
                )
        else:
            lines.append("- Corrélation deepfake × temporel : aucun pic deepfake corrélé")

        # Cross-campagne
        cross = temporal.get("cross_campaign", [])
        if cross:
            lines.append(f"- Amplificateurs cross-campagne ({len(cross)}) :")
            for x in cross[:5]:
                if "other_campaigns" in x:
                    lines.append(
                        f"  * @{x.get('account','?')} ({x.get('platform','?')}) "
                        f"— actif sur {x.get('campaign_count','?')} autres campagnes"
                    )
                else:
                    lines.append(
                        f"  * Campagne : {str(x.get('campaign','?'))[:50]} "
                        f"(score={x.get('score','?')}, {x.get('posts',0)} posts)"
                    )

        # Propagation chronologique
        prop = temporal.get("propagation", [])
        if prop:
            lines.append("- Chronologie d'entrée des comptes :")
            for p in prop:
                lines.append(
                    f"  * @{p.get('account','?')} ({p.get('platform','?')}) : "
                    f"entrée {str(p.get('first_post',''))[:10]} — "
                    f"sortie {str(p.get('last_post',''))[:10]} — "
                    f"{p.get('total_posts',0)} posts"
                )

        # Top jours de co-occurrence
        coocs = temporal.get("cooccurrences", [])
        if coocs:
            lines.append(f"- Top jours de publication simultanée :")
            for c in coocs[:5]:
                accs = ", ".join(f"@{a}" for a in (c.get("accounts_list") or [])[:4])
                lines.append(
                    f"  * {c.get('day','?')} : {c.get('active_accounts','?')} comptes, "
                    f"{c.get('total_posts','?')} posts → {accs}"
                )

        # Cadence
        cadence = temporal.get("cadence", [])
        if cadence:
            lines.append("- Cadence de publication :")
            for c in cadence:
                avg = round(c.get("avg_per_day") or 0, 2)
                lines.append(
                    f"  * @{c.get('account','?')} ({c.get('platform','?')}) : "
                    f"{c.get('active_days',0)} jours actifs, "
                    f"moy={avg}/jour, max={c.get('max_per_day',0)}/jour"
                )

        # Évolution mensuelle — frise ASCII enrichie
        monthly = temporal.get("monthly", [])
        if monthly:
            max_posts = max(m.get("posts", 0) for m in monthly) or 1
            lines.append("- FRISE TEMPORELLE (posts par mois) :")
            lines.append("  Mois    | Posts | Cptes | Deepfake | Activité")
            lines.append("  --------|-------|-------|----------|" + "-" * 20)
            for m in monthly[-24:]:  # 24 derniers mois max
                posts   = m.get("posts", 0)
                accts   = m.get("accounts", 0)
                df_val  = round(m.get("avg_deepfake") or 0, 3)
                synth   = m.get("synthetic_count", 0)
                bar_len = max(1, round(posts / max_posts * 20))
                bar     = "█" * bar_len
                synth_marker = " ⚠️synth" if synth > 0 else ""
                lines.append(
                    f"  {m.get('month','?')} | {str(posts).rjust(5)} | {str(accts).rjust(5)} "
                    f"| {str(df_val).ljust(8)} | {bar}{synth_marker}"
                )
        lines.append("")

    # --- Score global ---
    lines.append("## MÉTRIQUES CALCULÉES")
    lines.append(f"- Score de suspicion global : {suspicion_score} / 1.0")
    lines.append(f"- Niveau de confiance       : {confidence}")
    lines.append(f"- Neo4j disponible          : {neo4j_available}")
    lines.append("")

    return "\n".join(lines)

# ===========================================================================
# Runner principal — mode déterministe
# ===========================================================================

def run_investigation(
    entry_type:  str,
    entry_value: str,
    platform:    Optional[str],
    cfg:         AgentConfig,
    verbose:     bool = False,
) -> str:
    """
    Lance l'investigation complète en mode déterministe :
      1. Collecte les données via les outils Python (séquence fixe)
      2. Calcule le score de suspicion
      3. Demande au LLM de rédiger le rapport à partir des données réelles

    Args:
        entry_type  : account | post | narrative | campaign
        entry_value : identifiant de la cible
        platform    : plateforme (obligatoire pour account)
        cfg         : configuration LLM
        verbose     : active les logs détaillés

    Returns:
        str — rapport Markdown complet
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info(f"Investigation : {entry_type} — {entry_value}")
    if platform:
        logger.info(f"Plateforme    : {platform}")
    logger.info(f"Modèle        : {cfg.provider}/{cfg.model}")
    logger.info("=" * 60)

    # --- Étape 1 : vérification Neo4j ---
    neo4j_available = True
    try:
        from tools import _get_neo4j
        neo4j_available = _get_neo4j() is not None
    except Exception:
        neo4j_available = False
    if not neo4j_available:
        logger.warning("Neo4j indisponible — investigation MongoDB uniquement")

    # --- Étape 2 : chargement config agent ---
    agent_cfg = _load_agent_cfg()

    # --- Étape 3 : collecte déterministe des données ---
    logger.info("Phase 1 : collecte des données...")
    if entry_type == "account":
        raw_data = _collect_account(entry_value, platform, neo4j_available)
    elif entry_type == "narrative":
        raw_data = _collect_narrative(entry_value, neo4j_available)
    elif entry_type == "campaign":
        raw_data = _collect_campaign(entry_value)
    elif entry_type == "post":
        raw_data = _collect_post(entry_value, neo4j_available)
    else:
        raw_data = {}

    logger.info(f"Collecte initiale terminée — {len(raw_data)} sections récupérées")

    # --- Étape 3b : enrichissement adaptatif ---
    logger.info("Phase 1b : enrichissement adaptatif...")
    raw_data = _enrich_if_needed(raw_data, entry_type, agent_cfg)

    # --- Étape 2b : enrichissement adaptatif ---
    if entry_type in ("account", "campaign", "narrative"):
        raw_data = _enrich_if_needed(raw_data, entry_type, agent_cfg)

    # --- Étape 3 : calcul du score de suspicion ---
    # On reconstruit intermediate_steps depuis raw_data pour réutiliser compute_suspicion_score
    intermediate = []
    if "account" in raw_data:
        intermediate.append(("get_account_info", json.dumps(raw_data["account"], default=str)))
    if "media" in raw_data:
        intermediate.append(("get_media_scores", json.dumps(raw_data["media"], default=str)))
    if "campaign" in raw_data:
        intermediate.append(("get_campaign_signals", json.dumps(raw_data["campaign"], default=str)))
    for narr in raw_data.get("narratives", []):
        intermediate.append(("get_narrative", json.dumps(narr, default=str)))

    suspicion_score, confidence = compute_suspicion_score(intermediate)
    logger.info(f"Score de suspicion : {suspicion_score} (confiance : {confidence})")

    # --- Étape 4 : rédaction du rapport par le LLM ---
    logger.info("Phase 2 : rédaction du rapport par le LLM...")
    llm = build_llm(cfg)

    system = build_system_prompt(neo4j_available=neo4j_available)

    # Construit un résumé compact structuré (pas de JSON brut)
    ctx = _build_llm_context(raw_data, suspicion_score, confidence, neo4j_available)

    # --- Cible lisible selon le type d'entrée ---
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S UTC")

    if entry_type == "account":
        acc = raw_data.get("account", {})
        target_label    = acc.get("display_name") or acc.get("username") or entry_value
        target_platform = acc.get("platform") or platform or ""
        target_header   = f"@{target_label} ({target_platform})"
        subject_desc    = f"le compte @{target_label} sur {target_platform}"
        sections = """## Synthèse
[3-5 phrases résumant l'activité du compte, les scores deepfake, la cadence de publication et l'évaluation globale. Mentionner que l'analyse s'appuie sur des données potentiellement partielles.]

## Analyse des médias
[Statistiques réelles : nombre de médias, scores moyens, ratio synthétique, divergence inter-modèles, interprétation du pattern sdxl vs swinv2]

## Analyse temporelle
[Habitudes de publication. Inclure une frise ASCII des posts par mois (format : `2025-01 ███ 3 posts`) basée sur les données ANALYSE TEMPORELLE > évolution mensuelle. Mentionner cadence, silences suspects, cross-campagne.]

## Position dans le réseau

### Communauté Louvain
[Détail : nombre de comptes dans la communauté, community_id, PageRank. Lister les comptes de la communauté avec leur platform et pagerank.]

### Comptes partageant des hashtags
[Tableau Markdown : | Compte | Plateforme | Hashtags communs | — liste tous les comptes avec ≥2 hashtags en commun]

### Duplication de contenu
[Détail : ce compte copie qui (liste avec nb de copies) ? qui copie ce compte (liste avec nb de copies) ?]

### Campagnes et narratives liées
[Lister les campagnes détectées avec score et les narratives associées]

## Signaux de coordination
[Tableau Markdown récapitulatif :
| Signal | Valeur | Interprétation |
|---|---|---|
| Cadence robotique | oui/non | ... |
| Co-occurrences | N jours | ... |
| Duplication | N posts | ... |
| Cross-campagne | N campagnes | ... |
| Hashtags partagés | N comptes | ... |
Conclusion : coordonné / organique / insuffisant]

## Narratives portées
[Tableau Markdown : | Narrative | Posts | Ratio synthétique | Mots-clés |]

## Conclusion
[Qualification : campagne probable / viralité organique / insuffisant pour conclure. Rappeler les données partielles.]

## Requêtes Cypher suggérées
Propose 3 requêtes Cypher Neo4j Browser exploitables. Utilise UNIQUEMENT ces relations réelles et cette syntaxe exacte (pas de GROUP BY, pas de JOIN — c'est du Cypher pas du SQL) :

Syntaxe correcte pour compter et grouper :
```cypher
// Compter les hashtags d'un compte (CORRECT)
MATCH (a:Account {display_name: "nom_compte"})-[:A_PUBLIÉ]->(p:Post)-[:HAS_HASHTAG]->(h:Hashtag)
RETURN h.name AS hashtag, count(p) AS usage
ORDER BY usage DESC LIMIT 15
```

Relations disponibles : A_PUBLIÉ, EST_DOUBLON_DE, HAS_HASHTAG, APPARTIENT_À, COUVRE
Propriétés Account : display_name, platform, community_id, pagerank_score
Propriétés Post : published_at, deepfake_score, is_duplicate, sentiment_label, text

Propose 3 requêtes utiles pour ce compte en utilisant son vrai display_name. Format : bloc cypher + une ligne d'explication.

## Recommandations
[2-4 points concrets et actionnables]"""

    elif entry_type == "narrative":
        narr = raw_data.get("narrative", {})
        target_label    = narr.get("label") or entry_value
        target_platform = ""
        target_header   = f"Narrative — {target_label}"
        subject_desc    = f'la narrative "{target_label}"'
        sections = """## Synthèse
[3-5 phrases résumant la narrative : contenu, portée, ratio synthétique, comptes impliqués, période couverte. Mentionner les données potentiellement partielles.]

## Tableau des comptes portant la narrative
[Tableau Markdown de TOUS les comptes issus des données :
| Compte | Plateforme | Posts | Deepfake moy | Ratio synth. | Bot score |
|---|---|---|---|---|---|]

## Analyse temporelle
[Frise ASCII par mois : `AAAA-MM | ████ N posts | K comptes`
Analyser : compte semence, co-occurrences, silences, accélérations. Utilise ANALYSE TEMPORELLE.]

## Analyse des médias
[Statistiques deepfake des comptes les plus actifs si disponibles]

## Signaux de coordination
[Tableau Markdown :
| Signal | Valeur | Interprétation |
|---|---|---|
| Doublons | N paires | ... |
| Co-occurrence max | N comptes | ... |
| Hashtags partagés | N | ... |
Conclusion : coordonnée / organique / insuffisant]

## Conclusion
[Qualification : narrative coordonnée / viralité organique / insuffisant pour conclure. Rappeler les données partielles.]

## Requêtes Cypher suggérées
Propose 3 requêtes Cypher Neo4j Browser exploitables. Syntaxe obligatoire (Cypher, pas SQL — pas de GROUP BY) :

```cypher
// Comptes actifs sur une narrative (CORRECT)
MATCH (n:Narrative {label: "label_narrative"})<-[:APPARTIENT_À]-(p:Post)<-[:A_PUBLIÉ]-(a:Account)
RETURN a.display_name AS compte, a.platform AS plateforme, count(p) AS posts
ORDER BY posts DESC

// Doublons dans une narrative (CORRECT)
MATCH (n:Narrative {label: "label_narrative"})<-[:APPARTIENT_À]-(p:Post)-[:EST_DOUBLON_DE]->(orig:Post)
MATCH (p)<-[:A_PUBLIÉ]-(copier:Account), (orig)<-[:A_PUBLIÉ]-(source:Account)
RETURN copier.display_name AS copie, source.display_name AS source, count(*) AS nb
ORDER BY nb DESC
```

Relations disponibles : A_PUBLIÉ, EST_DOUBLON_DE, HAS_HASHTAG, APPARTIENT_À, COUVRE
Propose 3 requêtes utiles pour cette narrative. Format : bloc cypher + une ligne d'explication.

## Recommandations
[2-4 points concrets et actionnables]"""

    elif entry_type == "campaign":
        camp = raw_data.get("campaign", {})
        target_label    = camp.get("name") or entry_value
        target_platform = ""
        target_header   = f"Campagne — {target_label}"
        subject_desc    = f'la campagne détectée "{target_label}"'
        sections = """## Synthèse
[3-5 phrases résumant la campagne : signaux détectés, nombre de comptes, plateformes, période couverte, compte semence, niveau de confiance. Mentionner que l'analyse s'appuie sur des données potentiellement partielles.]

## Signaux de coordination
[Tableau Markdown récapitulatif de tous les signaux :
| Signal | Valeur | Évaluation |
|---|---|---|
| Réutilisation contenu | oui/non | ... |
| Cross-plateforme | oui/non + plateformes | ... |
| Doublons détectés | N paires | ... |
| Cadence robotique | comptes concernés | ... |
| Co-occurrence max | N comptes le même jour | ... |
| Score deepfake moyen | valeur | ... |
Conclusion sur le niveau de coordination.]

## Analyse temporelle
[Inclure une frise ASCII des posts par mois basée sur ANALYSE TEMPORELLE > évolution mensuelle :
`AAAA-MM | ████ N posts | K comptes | deepfake=X.XXX`
Analyser : compte semence, silences suspects, pics d'activité, propagation.]

## Tableau des comptes membres
[Tableau Markdown de TOUS les comptes issus des données GRAPHE NEO4J :
| Compte | Plateforme | Posts | Deepfake moy | Doublons | Communauté | Suspect |
|---|---|---|---|---|---|---|
Calculer les moyennes en pied de tableau.]

## Analyse des médias
[Statistiques deepfake des comptes membres si disponibles]

## Conclusion
[Qualification : campagne coordonnée confirmée / probable / insuffisant pour conclure. Appuie-toi sur les signaux temporels, de duplication et cross-plateforme. Rappeler les données partielles.]

## Requêtes Cypher suggérées
Propose 4 requêtes Cypher Neo4j Browser exploitables. Syntaxe obligatoire (Cypher, pas SQL — pas de GROUP BY, utiliser WITH + count()) :

```cypher
// Réseau complet d'une campagne (CORRECT)
MATCH path = (c:Campaign)-[:COUVRE]->(n:Narrative)<-[:APPARTIENT_À]-(p:Post)<-[:A_PUBLIÉ]-(a:Account)
WHERE c.mongo_id = "id_campagne"
RETURN path LIMIT 100

// Propagation chronologique (CORRECT)
MATCH (c:Campaign)-[:COUVRE]->(n:Narrative)<-[:APPARTIENT_À]-(p:Post)<-[:A_PUBLIÉ]-(a:Account)
WHERE c.mongo_id = "id_campagne" AND p.published_at IS NOT NULL
RETURN a.display_name AS compte, p.published_at AS date, p.text AS texte
ORDER BY p.published_at LIMIT 50

// Hashtags dominants (CORRECT — pas de GROUP BY)
MATCH (c:Campaign)-[:COUVRE]->(n:Narrative)<-[:APPARTIENT_À]-(p:Post)-[:HAS_HASHTAG]->(h:Hashtag)
WHERE c.mongo_id = "id_campagne"
RETURN h.name AS hashtag, count(p) AS usage
ORDER BY usage DESC LIMIT 15

// Doublons entre comptes (CORRECT)
MATCH (c:Campaign)-[:COUVRE]->(n:Narrative)<-[:APPARTIENT_À]-(p:Post)-[:EST_DOUBLON_DE]->(orig:Post)
WHERE c.mongo_id = "id_campagne"
MATCH (p)<-[:A_PUBLIÉ]-(copier:Account), (orig)<-[:A_PUBLIÉ]-(source:Account)
RETURN copier.display_name AS copie, source.display_name AS source, count(*) AS copies
ORDER BY copies DESC
```

Adapte ces 4 requêtes avec le vrai mongo_id de la campagne et les vrais display_name des comptes issus des données. Remplace les placeholders par les vraies valeurs.

## Recommandations
[2-4 points concrets et actionnables]"""

    else:
        acc = raw_data.get("account", {})
        target_label    = acc.get("display_name") or acc.get("username") or entry_value
        target_platform = acc.get("platform") or platform or ""
        target_header   = f"@{target_label} ({target_platform})"
        subject_desc    = f"la cible {target_label}"
        sections = """## Synthèse
[Résumé de l'investigation]

## Conclusion
[Qualification]

## Recommandations
[Points actionnables]"""

    user_prompt = f"""Tu dois rédiger un rapport d'investigation complet en Markdown sur {subject_desc}.

## Données collectées et métriques

{ctx}

## Instructions

Rédige le rapport en français avec exactement ces sections :

# Rapport d'investigation — {target_header}
**Date :** {now_str}
**Score de suspicion global :** {suspicion_score} / 1.0
**Niveau de confiance :** {confidence}

---

{sections}

RÈGLES STRICTES :
- Utilise uniquement les valeurs numériques fournies dans les données ci-dessus
- Ne pas inventer de dates, scores, noms ou identifiants
- Si une donnée est manquante, l'indiquer explicitement sans l'inventer
- La date du rapport est {now_str}

AVERTISSEMENT DONNÉES PARTIELLES :
Le scrapping sur lequel repose cette analyse n'est pas nécessairement exhaustif.
Les données disponibles représentent un échantillon de l'activité réelle — il existe
probablement des comptes, posts, médias et relations non capturés. Mentionne
systématiquement dans la synthèse et la conclusion que l'analyse s'appuie sur
des données potentiellement partielles, et que les signaux détectés peuvent être
sous-estimés ou sur-représentés par rapport à la réalité.
"""

    from langchain_core.messages import HumanMessage, SystemMessage
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_prompt),
    ])

    report_content = response.content if hasattr(response, "content") else str(response)

    # --- Étape 4b : génération des graphes ---
    graph_links = ""
    if _cfgget("graphs", "enabled", "true").lower() == "true":
        try:
            from graphs import GraphGenerator
            _safe_val = re.sub(r'[^\w]', '_', entry_value)[:30]
            _ts_safe  = now_str.replace(':', '').replace(' ', '_')
            report_subdir = cfg.reports_dir / f"{_ts_safe}_{entry_type}_{_safe_val}"
            gen   = GraphGenerator(report_subdir, _AGENT_CFG)
            paths = gen.generate_all(raw_data, entry_type)
            # markdown_graph_links : import ou fallback inline
            try:
                from graphs import markdown_graph_links as _mgl
                graph_links = _mgl(paths, report_subdir)
            except ImportError:
                _lns = ["\n## Graphes annexes\n"]
                _labels = {"temporal": "Évolution temporelle", "propagation": "Propagation par compte",
                           "deepfake": "Distribution deepfake", "network": "Graphe réseau interactif"}
                for _k, _p in paths.items():
                    if _p is None:
                        continue
                    _lbl = _labels.get(_k, _k)
                    if _p.suffix == ".html":
                        _lns.append(f"- [{_lbl}](./{_p.name}) *(interactif)*")
                    else:
                        _lns.append(f"\n### {_lbl}\n\n![{_lbl}](./{_p.name})\n")
                graph_links = "\n".join(_lns)
        except Exception as exc:
            logger.warning(f"Génération graphes échouée : {exc}")

    # --- Étape 4c : suggestions de scrapping ---
    scraping_md = _build_scraping_suggestions(raw_data, entry_type, agent_cfg)

    # --- Étape 5 : ajout du score et disclaimer ---
    if "Score de suspicion" not in report_content:
        header = (
            f"**Score de suspicion global :** {suspicion_score} / 1.0  \n"
            f"**Niveau de confiance :** {confidence}\n\n"
        )
        report_content = header + report_content

    # --- Suggestions de scrapping (Python pur) ---
    scraping_md = _build_scraping_suggestions(raw_data, entry_type, agent_cfg)
    if scraping_md and "Suggestions de scrapping" not in report_content:
        report_content += scraping_md

    disclaimer = (
        "\n\n---\n"
        f"*Rapport généré automatiquement par AI-FORENSICS Investigation Agent*  \n"
        f"*Modèle : {cfg.model} via {cfg.provider} — Ce rapport nécessite une vérification humaine.*  \n"
        f"*Les scores deepfake sont des indicateurs probabilistes, pas des preuves.*  \n"
        f"*⚠️ Données potentiellement partielles : le scrapping peut être incomplet. "
        f"Les signaux détectés s'appuient sur un échantillon de l'activité réelle.*"
    )
    if "vérification humaine" not in report_content:
        report_content += disclaimer

    # --- Génération des graphes ---
    try:
        from graphs import GraphGenerator
        ts        = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        safe_val  = re.sub(r"[^\w\-]", "_", entry_value)[:30]
        graph_dir = cfg.reports_dir / f"{ts}_{entry_type}_{safe_val}"
        gen       = GraphGenerator(report_dir=graph_dir, cfg=agent_cfg)
        graph_paths = gen.generate_all(raw_data=raw_data, entry_type=entry_type)
        if graph_paths:
            graph_md = "\n\n## Annexes graphiques\n\n"
            labels = {
                "temporal":    "Évolution temporelle des publications",
                "propagation": "Propagation chronologique par compte",
                "deepfake":    "Distribution des scores deepfake",
                "network":     "Graphe réseau (interactif)",
            }
            for key, path in graph_paths.items():
                label = labels.get(key, key)
                if path.suffix == ".html":
                    graph_md += f"- [{label}](./{graph_dir.name}/{path.name}) — ouvrir dans un navigateur\n"
                else:
                    graph_md += f"\n### {label}\n\n![{label}](./{graph_dir.name}/{path.name})\n"
            report_content += graph_md
            logger.info(f"Graphes générés dans : {graph_dir.name}/")
    except Exception as exc:
        logger.debug(f"Graphes non générés : {exc}")

    return report_content



# ===========================================================================
# Enrichissement adaptatif
# ===========================================================================


def _generate_synthesis(batch_results: list, cfg: AgentConfig, args) -> None:
    """
    Génère un CR de synthèse après un batch --all-campaigns ou --project.
    Agrège les métadonnées de tous les rapports et produit un résumé global.
    """
    from tools import _get_db
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S UTC")
    title   = f"Synthèse — {'Projet : ' + args.project if args.project else 'Toutes les campagnes'}"
    max_rows = _cfgint("batch", "max_campaigns_in_table", 50)
    top_amp  = _cfgint("batch", "top_amplifiers", 10)

    # Récupérer les métadonnées des campagnes depuis MongoDB
    try:
        db = _get_db()
        camp_ids_str = [r["id"] for r in batch_results]
        from bson import ObjectId
        campaigns_meta = {
            str(c["_id"]): c
            for c in db.campaigns.find(
                {"_id": {"$in": [ObjectId(i) for i in camp_ids_str]}},
                {"name": 1, "signals": 1, "review": 1, "narrative_ids": 1}
            )
        }
    except Exception as exc:
        logger.warning(f"Métadonnées campagnes : {exc}")
        campaigns_meta = {}

    # Comptes amplificateurs transversaux via Neo4j
    amplifiers = []
    try:
        from tools import _get_neo4j
        driver = _get_neo4j()
        if driver:
            with driver.session() as s:
                r = s.run("""
                    MATCH (c:Campaign)-[:COUVRE]->(n:Narrative)<-[:APPARTIENT_À]-(p:Post)<-[:A_PUBLIÉ]-(a:Account)
                    WITH a, count(DISTINCT c) AS nb_campaigns, count(p) AS total_posts
                    WHERE nb_campaigns > 1
                    RETURN a.display_name AS account, a.platform AS platform,
                           nb_campaigns, total_posts
                    ORDER BY nb_campaigns DESC, total_posts DESC
                    LIMIT $limit
                """, limit=top_amp)
                amplifiers = [dict(row) for row in r]
    except Exception as exc:
        logger.warning(f"Amplificateurs Neo4j : {exc}")

    # Construction du rapport de synthèse
    lines = [
        f"# {title}",
        f"**Date :** {now_str}",
        f"**Campagnes analysées :** {len(batch_results)}",
        "",
        "---",
        "",
        "> ⚠️ Ce rapport de synthèse s'appuie sur des données potentiellement partielles.",
        "> Les signaux détectés peuvent être sous-estimés par rapport à la réalité.",
        "",
        "## Tableau des campagnes analysées",
        "",
        "| Campagne | Score | Confiance | Signaux clés |",
        "|---|---|---|---|",
    ]

    scores = []
    for r in batch_results[:max_rows]:
        cid  = r["id"]
        name = r["name"][:50]
        meta = campaigns_meta.get(cid, {})
        rev  = meta.get("review", {})
        sigs = meta.get("signals", [])
        conf = rev.get("confidence")
        score_str = f"{conf:.2f}" if conf is not None else "N/A"
        if conf is not None:
            scores.append(conf)
        sigs_str  = ", ".join(sigs[:4]) if sigs else "—"
        lines.append(f"| {name} | {score_str} | — | {sigs_str} |")

    # Score moyen global
    if scores:
        avg = sum(scores) / len(scores)
        lines.extend([
            "",
            f"**Score de suspicion moyen :** {avg:.2f} / 1.0",
        ])

    # Amplificateurs transversaux
    if amplifiers:
        lines.extend([
            "",
            "## Comptes amplificateurs transversaux",
            "",
            f"Comptes actifs sur plusieurs campagnes simultanément :",
            "",
            "| Compte | Plateforme | Campagnes | Posts totaux |",
            "|---|---|---|---|",
        ])
        for a in amplifiers:
            lines.append(
                f"| @{a.get('account','?')} | {a.get('platform','?')} "
                f"| {a.get('nb_campaigns','?')} | {a.get('total_posts','?')} |"
            )

    # Demander au LLM une synthèse narrative
    try:
        llm = build_llm(cfg)
        from langchain_core.messages import HumanMessage, SystemMessage
        summary_prompt = f"""Tu es un analyste en détection de campagnes d'influence.
Voici les résultats d'une analyse batch de {len(batch_results)} campagnes.

Campagnes analysées : {[r["name"] for r in batch_results[:10]]}
Score moyen : {avg:.2f if scores else "N/A"}
Amplificateurs transversaux : {[a.get("account","?") for a in amplifiers[:5]]}

Rédige en 3-5 phrases :
1. Une synthèse globale du niveau de coordination détecté
2. Les patterns les plus préoccupants
3. Les priorités d'investigation recommandées

Mentionne que les données sont potentiellement partielles."""

        resp = llm.invoke([
            SystemMessage(content="Tu es un expert en analyse de campagnes d'influence sur les réseaux sociaux."),
            HumanMessage(content=summary_prompt),
        ])
        narrative = resp.content if hasattr(resp, "content") else str(resp)
        lines.extend(["", "## Analyse globale", "", narrative])
    except Exception as exc:
        logger.warning(f"Synthèse LLM : {exc}")

    lines.extend([
        "",
        "---",
        f"*CR de synthèse généré automatiquement par AI-FORENSICS Investigation Agent*",
        f"*Modèle : {cfg.model} via {cfg.provider} — Nécessite une vérification humaine.*",
    ])

    synthesis_content = "\n".join(lines)

    # Sauvegarde
    if not args.no_save:
        ts    = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        label = re.sub(r"[^\w]", "_", args.project or "all_campaigns")[:30]
        path  = cfg.reports_dir / f"{ts}_synthese_{label}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(synthesis_content, encoding="utf-8")
        logger.info(f"CR de synthèse sauvegardé : {path}")
        print(f"\n✓ CR de synthèse : {path}")


def _generate_synthesis(campaigns: list, cfg: "AgentConfig", agent_cfg: dict, args) -> None:
    """Génère un rapport de synthèse après un batch de campagnes."""
    from tools import _get_db, _get_neo4j
    from bson import ObjectId

    logger.info("Génération du rapport de synthèse...")
    db = _get_db()
    bcfg = agent_cfg.get("batch", {})
    max_camps  = int(bcfg.get("max_campaigns_in_table", 50))
    top_amps   = int(bcfg.get("top_amplifiers", 10))
    label      = getattr(args, "project", None) or "toutes_campagnes"

    # Récupérer les données de toutes les campagnes
    rows = []
    for camp in campaigns[:max_camps]:
        doc = db.campaigns.find_one({"_id": camp["_id"]})
        if not doc:
            continue
        sig = doc.get("signals", [])
        rows.append({
            "id":        str(camp["_id"]),
            "name":      camp.get("name", str(camp["_id"]))[:60],
            "score":     doc.get("score", 0),
            "signals":   ", ".join(sig[:4]) if sig else "—",
            "platforms": ", ".join(doc.get("platforms", [])),
        })

    # Comptes amplificateurs transversaux via Neo4j
    amplifiers = []
    try:
        driver = _get_neo4j()
        if driver:
            with driver.session() as s:
                r = s.run("""
                    MATCH (c:Campaign)-[:COUVRE]->(n:Narrative)<-[:APPARTIENT_À]-(p:Post)<-[:A_PUBLIÉ]-(a:Account)
                    WITH a, count(DISTINCT c) AS nb_camps, count(p) AS nb_posts
                    WHERE nb_camps > 1
                    RETURN a.display_name AS account, a.platform AS platform,
                           nb_camps, nb_posts
                    ORDER BY nb_camps DESC, nb_posts DESC
                    LIMIT $top
                """, top=top_amps)
                amplifiers = [dict(row) for row in r]
    except Exception as exc:
        logger.debug(f"Amplificateurs Neo4j : {exc}")

    # Rédiger la synthèse avec le LLM
    llm = build_llm(cfg)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S UTC")

    # Contexte pour le LLM
    ctx_lines = [f"## {len(rows)} CAMPAGNES ANALYSÉES\n"]
    ctx_lines.append("| Campagne | Score | Plateformes | Signaux |")
    ctx_lines.append("|---|---|---|---|")
    for r in rows:
        ctx_lines.append(f"| {r['name']} | {r['score']} | {r['platforms']} | {r['signals']} |")

    if amplifiers:
        ctx_lines.append(f"\n## AMPLIFICATEURS TRANSVERSAUX ({len(amplifiers)})\n")
        ctx_lines.append("| Compte | Plateforme | Campagnes | Posts |")
        ctx_lines.append("|---|---|---|---|")
        for a in amplifiers:
            ctx_lines.append(f"| @{a['account']} | {a['platform']} | {a['nb_camps']} | {a['nb_posts']} |")

    ctx = "\n".join(ctx_lines)

    prompt = f"""Rédige un rapport de synthèse en Markdown sur l'ensemble des campagnes d'influence analysées.

## Données agrégées
{ctx}

## Structure du rapport

# Synthèse — {len(rows)} campagnes analysées ({now_str})

## Vue d'ensemble
[Tableau récapitulatif des campagnes avec score, plateformes et signaux clés]

## Comptes amplificateurs transversaux
[Comptes présents dans plusieurs campagnes — tableau avec nombre de campagnes et posts]

## Patterns communs
[3-5 phrases sur les signaux récurrents, plateformes dominantes, thèmes narratifs]

## Score de risque global
[Évaluation globale : campagnes à haut risque vs modéré vs faible]

## Recommandations prioritaires
[Top 5 actions prioritaires cross-campagnes]

RÈGLES : données potentiellement partielles, ne pas inventer de valeurs, date = {now_str}
"""
    from langchain_core.messages import HumanMessage, SystemMessage
    try:
        response = llm.invoke([
            SystemMessage(content="Tu es un analyste CTI spécialisé en opérations d'influence."),
            HumanMessage(content=prompt),
        ])
        synthesis_content = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        logger.error(f"LLM synthèse : {exc}")
        synthesis_content = f"# Synthèse batch\n\n{ctx}\n\n*Rédaction LLM échouée : {exc}*"

    # Ajouter disclaimer
    synthesis_content += (
        "\n\n---\n"
        f"*Synthèse générée automatiquement — {now_str}*  \n"
        f"*Modèle : {cfg.model} via {cfg.provider}*  \n"
        "*⚠️ Données potentiellement partielles.*"
    )

    if not getattr(args, "no_save", False):
        ts    = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        fname = f"{ts}_synthese_{label}.md"
        path  = cfg.reports_dir / fname
        cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(synthesis_content, encoding="utf-8")
        logger.info(f"Synthèse sauvegardée : {fname}")
    else:
        print("\n" + "=" * 60)
        print(synthesis_content)
        print("=" * 60)


def _run_batch(args, cfg: AgentConfig) -> None:
    """
    Lance une investigation sur plusieurs campagnes en séquence.
    Modes :
      --all-campaigns          : toutes les campagnes en base
      --project NOM_PROJET     : campagnes dont les narratives sont liées au projet
    """
    from tools import _get_db
    from bson import ObjectId

    db = _get_db()
    logger.info("=" * 60)

    if args.project:
        # Campagnes liées à un projet via les posts du projet
        logger.info(f"Mode batch : projet '{args.project}'")
        # On cherche les narrative_ids des posts du projet
        pipeline = [
            {"$match": {"source.project": args.project, "nlp.narrative_id": {"$ne": None}}},
            {"$group": {"_id": "$nlp.narrative_id"}},
        ]
        narr_ids = [r["_id"] for r in db.posts.aggregate(pipeline)]
        if not narr_ids:
            logger.warning(f"Aucun post trouvé pour le projet '{args.project}'")
            return
        # Campagnes qui couvrent ces narratives
        campaigns = list(db.campaigns.find(
            {"narrative_ids": {"$in": narr_ids}},
            {"_id": 1, "name": 1}
        ))
        if not campaigns:
            # Fallback : toutes les campagnes (le projet n'est pas encore lié)
            logger.warning("Aucune campagne liée aux narratives du projet — analyse toutes les campagnes")
            campaigns = list(db.campaigns.find({}, {"_id": 1, "name": 1}))
    else:
        logger.info("Mode batch : toutes les campagnes")
        campaigns = list(db.campaigns.find({}, {"_id": 1, "name": 1}))

    if not campaigns:
        logger.warning("Aucune campagne trouvée en base")
        return

    logger.info(f"{len(campaigns)} campagne(s) à analyser")
    logger.info("=" * 60)

    success, errors = 0, 0
    batch_results = []  # collecte les métadonnées pour la synthèse
    for i, camp in enumerate(campaigns, 1):
        camp_id   = str(camp["_id"])
        camp_name = camp.get("name", camp_id)[:60]
        logger.info(f"[{i}/{len(campaigns)}] {camp_name}")

        try:
            report = run_investigation(
                entry_type  = "campaign",
                entry_value = camp_id,
                platform    = None,
                cfg         = cfg,
                verbose     = args.verbose,
            )
            if not args.no_save:
                path = save_report(
                    content     = report,
                    reports_dir = cfg.reports_dir,
                    entry_type  = "campaign",
                    entry_value = camp_id,
                )
                logger.info(f"  ✓ Sauvegardé : {path.name}")
            success += 1
            # Collecter les métadonnées pour la synthèse
            batch_results.append({
                "id":    camp_id,
                "name":  camp_name,
                "report": report,
            })
        except KeyboardInterrupt:
            logger.info("Batch interrompu par l'utilisateur.")
            break
        except Exception as exc:
            logger.error(f"  ✗ Erreur : {exc}")
            errors += 1

    logger.info("=" * 60)
    logger.info(f"Batch terminé — {success} OK, {errors} erreurs")
    if not args.no_save:
        logger.info(f"Rapports dans : {cfg.reports_dir}")

    # --- Synthèse batch ---
    agent_cfg = _load_agent_cfg()
    if str(agent_cfg.get("batch", {}).get("synthesis_report", "true")).lower() == "true":
        _generate_synthesis(campaigns, cfg, agent_cfg, args)

    # Synthèse batch
    if _cfgget("batch", "synthesis_report", "true").lower() == "true" and batch_results:
        logger.info("Génération du CR de synthèse...")
        _generate_synthesis(batch_results, cfg, args)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Agent d'investigation AI-FORENSICS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Analyser un compte TikTok
  python investigation_agent.py --account nolimitmoneyy0 --platform tiktok

  # Analyser une narrative
  python investigation_agent.py --narrative 69d2774d7dc202593e1b35fa

  # Analyser une campagne
  python investigation_agent.py --campaign <campaign_id>

  # Avec options
  python investigation_agent.py --account nolimitmoneyy0 --platform tiktok \\
      --output ~/AI-FORENSICS/reports --model llama3.1:8b --verbose
        """,
    )

    # Points d'entrée (mutuellement exclusifs)
    entry = parser.add_mutually_exclusive_group(required=True)
    entry.add_argument("--account",       metavar="UNIQUE_ID",
                       help="@handle ou platform_id du compte à analyser")
    entry.add_argument("--narrative",     metavar="NARRATIVE_ID",
                       help="_id MongoDB de la narrative")
    entry.add_argument("--campaign",      metavar="CAMPAIGN_ID",
                       help="_id MongoDB de la campagne")
    entry.add_argument("--post",          metavar="POST_ID",
                       help="_id MongoDB du post")
    entry.add_argument("--all-campaigns", action="store_true",
                       help="Analyse toutes les campagnes en base")
    entry.add_argument("--project",       metavar="NOM_PROJET",
                       help="Analyse toutes les campagnes d'un projet (source.project)")

    # Options compte
    parser.add_argument("--platform", default=None,
                        choices=["tiktok", "instagram", "twitter", "telegram"],
                        help="Plateforme (obligatoire avec --account)")

    # Options LLM
    parser.add_argument("--provider", default=None,
                        choices=["ollama", "groq", "anthropic"],
                        help="Provider LLM (défaut : $AI_PROVIDER ou ollama)")
    parser.add_argument("--model",    default=None,
                        help="Modèle LLM (défaut : $AI_MODEL ou llama3.1:8b)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Température LLM (défaut : 0.2)")

    # Options sortie
    parser.add_argument("--output",  default=None, metavar="DIR",
                        help="Dossier de sortie des rapports (défaut : ../reports/)")
    parser.add_argument("--no-save", action="store_true",
                        help="Affiche le rapport sans le sauvegarder")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Logs détaillés")

    args = parser.parse_args()

    # Validation
    if args.account and not args.platform:
        parser.error("--platform est obligatoire avec --account")

    # Configuration (construite avant le dispatch pour être disponible dans tous les modes)
    cfg = AgentConfig(
        provider    = args.provider,
        model       = args.model,
        output_dir  = args.output,
        temperature = args.temperature,
    )
    if args.verbose:
        logger.info(f"Config : {cfg}")

    # Détermine le point d'entrée
    if args.account:
        entry_type  = "account"
        entry_value = args.account
    elif args.narrative:
        entry_type  = "narrative"
        entry_value = args.narrative
    elif args.campaign:
        entry_type  = "campaign"
        entry_value = args.campaign
    elif args.post:
        entry_type  = "post"
        entry_value = args.post
    elif args.all_campaigns or args.project:
        # Mode batch : analyse multiple de campagnes
        _run_batch(args, cfg)
        return

    # Lance l'investigation
    try:
        report = run_investigation(
            entry_type  = entry_type,
            entry_value = entry_value,
            platform    = args.platform,
            cfg         = cfg,
            verbose     = args.verbose,
        )
    except KeyboardInterrupt:
        logger.info("Investigation interrompue par l'utilisateur.")
        sys.exit(0)
    except Exception as exc:
        logger.error(f"Erreur fatale : {exc}", exc_info=True)
        sys.exit(1)

    # Affichage
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    # Sauvegarde
    if not args.no_save:
        path = save_report(
            content     = report,
            reports_dir = cfg.reports_dir,
            entry_type  = entry_type,
            entry_value = entry_value,
        )
        logger.info(f"Rapport sauvegardé : {path}")
        print(f"\n✓ Rapport sauvegardé : {path}")


if __name__ == "__main__":
    main()
