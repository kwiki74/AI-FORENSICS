"""
AI Forensics Explorer  —  Streamlit
=====================================
Visualisation des posts et médias stockés en MongoDB (influence_detection).

Layout 3 colonnes :
  ┌──────────┬──────────────────────────┬───────────────────────┐
  │ FILTRES  │  LISTE posts / médias    │  DÉTAIL               │
  │ (gauche) │  (centre)                │  (droite)             │
  └──────────┴──────────────────────────┴───────────────────────┘

Lancement :
    streamlit run forensics_explorer.py

Dépendances :
    pip install streamlit pymongo python-dotenv
"""

from __future__ import annotations

import base64
import mimetypes
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

import streamlit as st
from bson import ObjectId

# ─── Chargement .env ───────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# ─── CSS  ──────────────────────────────────────────────────────────────────────
_CSS = """
<style>
/* ── Pleine largeur ── */
.block-container { max-width: 100% !important; padding-left: 1rem !important; padding-right: 1rem !important; }

/* ── Fond global (mode clair) ── */
.stApp { background-color: #f6f8fa; color: #1f2328; }
[data-testid="stHeader"] { background-color: #f6f8fa; }

/* ── Colonnes ── */
[data-testid="stHorizontalBlock"] > div { padding: 0 6px; }

/* ── Cartes liste ── */
.doc-card {
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 8px;
    padding: 11px 13px;
    margin-bottom: 0px;
    line-height: 1.45;
}
.doc-card.selected { border-color: #0969da; background: #ddf4ff; }

/* ── Panneau détail ── */
.detail-panel {
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 8px;
    padding: 16px 18px;
}

/* ── Titres de section ── */
.section-title {
    color: #0969da;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    border-bottom: 1px solid #d0d7de;
    padding-bottom: 3px;
    margin: 14px 0 8px 0;
}

/* ── Paires clé / valeur ── */
.kv { display: flex; justify-content: space-between; margin-bottom: 3px; }
.kv-k { color: #656d76; font-size: 12px; }
.kv-v { color: #1f2328; font-size: 12px; font-weight: 500; max-width: 65%; text-align: right; word-break: break-word; }

/* ── Badges plateforme ── */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
    margin-right: 4px;
}
.badge-twitter   { background:#1d9bf0; color:#fff; }
.badge-instagram { background:#e1306c; color:#fff; }
.badge-tiktok    { background:#010101; color:#fff; }
.badge-telegram  { background:#229ed9; color:#fff; }
.badge-unknown   { background:#eaeef2; color:#656d76; border:1px solid #d0d7de; }

/* ── Scores deepfake ── */
.pred-synthetic { color: #cf222e; font-weight: 700; }
.pred-suspicious{ color: #9a6700; font-weight: 700; }
.pred-real      { color: #1a7f37; font-weight: 700; }
.pred-pending   { color: #656d76; font-style: italic; }

/* ── Barre de score ── */
.bar-wrap { background:#eaeef2; border-radius:4px; height:8px; overflow:hidden; margin:2px 0 10px 0; }
.bar-fill { height:100%; border-radius:4px; }

/* ── Tags topics NLP ── */
.topic-tag {
    display: inline-block;
    background: #ddf4ff;
    border: 1px solid #54aeff;
    color: #0969da;
    border-radius: 12px;
    font-size: 11px;
    padding: 1px 8px;
    margin: 2px 3px 2px 0;
}

/* ── Séparateur header ── */
.filter-hdr {
    font-size: 11px;
    font-weight: 700;
    color: #656d76;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 10px 0 6px 0;
}

/* ── Bloc contenu texte ── */
.content-block {
    font-size: 13px; color: #1f2328; line-height: 1.6;
    background: #f6f8fa; border: 1px solid #d0d7de;
    border-radius: 6px; padding: 10px;
}

/* ── Bouton discret (▶) dans la carte ── */
[data-testid="stHorizontalBlock"]:has(.doc-card) { margin-bottom: 7px; }
button[kind="secondary"].btn-select {
    background: transparent !important; border: 1px solid #d0d7de !important;
    border-radius: 6px !important; color: #656d76 !important;
    padding: 0 !important; min-height: unset !important;
    height: 100% !important; font-size: 14px !important;
}
button[kind="secondary"].btn-select:hover {
    border-color: #0969da !important; color: #0969da !important;
    background: #ddf4ff !important;
}

/* ══════════════ DARK MODE (thème OS) ══════════════ */
@media (prefers-color-scheme: dark) {
    .stApp                    { background-color: #0d1117; color: #e6edf3; }
    [data-testid="stHeader"]  { background-color: #0d1117; }
    .doc-card                 { background: #161b22; border-color: #30363d; }
    .doc-card.selected        { border-color: #1f6feb; background: #1d3348; }
    .detail-panel             { background: #161b22; border-color: #30363d; }
    .section-title            { color: #79c0ff; border-bottom-color: #30363d; }
    .kv-k                     { color: #8b949e; }
    .kv-v                     { color: #e6edf3; }
    .bar-wrap                 { background: #21262d; }
    .topic-tag                { background: #1d3348; border-color: #388bfd; color: #79c0ff; }
    .filter-hdr               { color: #8b949e; }
    .content-block            { background: #161b22; border-color: #30363d; color: #e6edf3; }
    .pred-suspicious          { color: #d4a72c; }
}
</style>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# CONNEXION MONGODB
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Connexion à MongoDB…")
def get_db():
    from pymongo import MongoClient

    # ── Lecture des credentials depuis .env ou variables d'environnement ──────
    # Aucune valeur par défaut pour les secrets : si MONGO_USER ou MONGO_PASSWORD
    # ne sont pas définis, l'application affiche une erreur explicite et s'arrête.
    host     = os.getenv("MONGO_HOST", "localhost")
    port     = int(os.getenv("MONGO_PORT", 27017))
    user     = os.getenv("MONGO_USER")
    password = os.getenv("MONGO_PASSWORD")
    db_name  = os.getenv("MONGO_DB", "influence_detection")
    auth_db  = os.getenv("MONGO_AUTH_DB", db_name)

    missing = [name for name, val in [("MONGO_USER", user), ("MONGO_PASSWORD", password)] if not val]
    if missing:
        st.error(
            f"❌ Variable(s) d'environnement manquante(s) : {', '.join(missing)}\n\n"
            "Créez le fichier `~/AI-FORENSICS/.env` à partir de `.env.example` "
            "et relancez Streamlit."
        )
        st.stop()

    if user and password:
        uri = (
            f"mongodb://{quote_plus(user)}:{quote_plus(password)}"
            f"@{host}:{port}/?authSource={auth_db}"
            f"&replicaSet=rs0&directConnection=true"
        )
    else:
        uri = f"mongodb://{host}:{port}/"

    client = MongoClient(uri, serverSelectionTimeoutMS=4000)
    client.admin.command("ping")          # lève une exception si connexion KO
    return client[db_name]


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

PLATFORM_SVGS = {
    "twitter": (
        '#1d9bf0',
        '<svg viewBox="0 0 24 24" width="13" height="13" fill="white">'
        '<path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-4.714-6.231'
        '-5.401 6.231H2.744l7.73-8.835L1.254 2.25H8.08l4.259 5.63zm-1.161 17.52'
        'h1.833L7.084 4.126H5.117z"/></svg>'
    ),
    "instagram": (
        '#e1306c',
        '<svg viewBox="0 0 24 24" width="13" height="13" fill="white">'
        '<path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919'
        ' 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149'
        ' 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584'
        '-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07'
        '-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266'
        '-.057 1.645-.069 4.849-.069zM12 0C8.741 0 8.333.014 7.053.072 2.695.272.273'
        ' 2.69.073 7.052.014 8.333 0 8.741 0 12c0 3.259.014 3.668.072 4.948.2 4.358'
        ' 2.618 6.78 6.98 6.98C8.333 23.986 8.741 24 12 24c3.259 0 3.668-.014 4.948'
        '-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259'
        '-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98C15.668.014 15.259 0'
        ' 12 0zm0 5.838a6.162 6.162 0 100 12.324 6.162 6.162 0 000-12.324zM12 16a4 4'
        ' 0 110-8 4 4 0 010 8zm6.406-11.845a1.44 1.44 0 100 2.881 1.44 1.44 0 000'
        '-2.881z"/></svg>'
    ),
    "tiktok": (
        '#010101',
        '<svg viewBox="0 0 24 24" width="13" height="13" fill="white">'
        '<path d="M19.59 6.69a4.83 4.83 0 01-3.77-4.25V2h-3.45v13.67a2.89 2.89 0'
        ' 01-2.88 2.5 2.89 2.89 0 01-2.89-2.89 2.89 2.89 0 012.89-2.89c.28 0 .54'
        '.04.79.1V9.01a6.33 6.33 0 00-.79-.05 6.34 6.34 0 00-6.34 6.34 6.34 6.34'
        ' 0 006.34 6.34 6.34 6.34 0 006.33-6.34V8.69a8.27 8.27 0 004.84 1.55V6.78'
        'a4.85 4.85 0 01-1.07-.09z"/></svg>'
    ),
    "telegram": (
        '#229ed9',
        '<svg viewBox="0 0 24 24" width="13" height="13" fill="white">'
        '<path d="M11.944 0A12 12 0 000 12a12 12 0 0012 12 12 12 0 0012-12A12 12 0'
        ' 0012 0a12 12 0 00-.056 0zm4.962 7.224c.1-.002.321.023.465.14a.506.506 0'
        ' 01.171.325c.016.093.036.306.02.472-.18 1.898-.962 6.502-1.36 8.627-.168'
        '.9-.499 1.201-.82 1.23-.696.065-1.225-.46-1.9-.902-1.056-.693-1.653-1.124'
        '-2.678-1.8-1.185-.78-.417-1.21.258-1.91.177-.184 3.247-2.977 3.307-3.23'
        '.007-.032.014-.15-.056-.212s-.174-.041-.249-.024c-.106.024-1.793 1.14-5.061'
        ' 3.345-.48.33-.913.49-1.302.48-.428-.008-1.252-.241-1.865-.44-.752-.245'
        '-1.349-.374-1.297-.789.027-.216.325-.437.893-.663 3.498-1.524 5.83-2.529'
        ' 6.998-3.014 3.332-1.386 4.025-1.627 4.476-1.635z"/></svg>'
    ),
    "facebook": (
        '#1877f2',
        '<svg viewBox="0 0 24 24" width="13" height="13" fill="white">'
        '<path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388'
        ' 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669'
        ' 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925'
        '-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062'
        ' 24 12.073z"/></svg>'
    ),
    "youtube": (
        '#ff0000',
        '<svg viewBox="0 0 24 24" width="13" height="13" fill="white">'
        '<path d="M23.498 6.186a3.016 3.016 0 00-2.122-2.136C19.505 3.545 12 3.545'
        ' 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 00.502 6.186C0 8.07 0 12 0 12s0'
        ' 3.93.502 5.814a3.016 3.016 0 002.122 2.136c1.871.505 9.376.505 9.376.505s'
        '7.505 0 9.377-.505a3.015 3.015 0 002.122-2.136C24 15.93 24 12 24 12s0-3.93'
        '-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/></svg>'
    ),
}

PREDICTION_META = {
    # prediction  → (icône, classe CSS, label affiché)
    "synthetic":   ("🔴", "pred-synthetic",  "SYNTHÉTIQUE"),
    "suspicious":  ("🟡", "pred-suspicious", "SUSPECT"),
    "likely_real": ("🟢", "pred-real",       "RÉEL"),
    None:          ("⬜", "pred-pending",    "—"),
}

SENTIMENT_ICONS = {"positive": "😊", "negative": "😠", "neutral": "😐"}

IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
VIDEO_EXTS  = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}


def fmt_dt(val) -> str:
    if not val:
        return "—"
    if isinstance(val, datetime):
        return val.strftime("%Y-%m-%d %H:%M")
    s = str(val)
    return s[:16].replace("T", " ")


def score_color(score) -> str:
    if score is None:
        return "#656d76"
    return "#cf222e" if score >= 0.75 else ("#9a6700" if score >= 0.45 else "#1a7f37")


def bar_html(score, label="") -> str:
    if score is None:
        return "<span style='color:#8b949e;font-size:12px;'>—</span>"
    pct   = int(score * 100)
    color = score_color(score)
    return (
        f"<div style='font-size:11px;color:#8b949e;margin-bottom:1px;'>"
        f"{label} <strong style='color:{color};'>{pct}%</strong></div>"
        f"<div class='bar-wrap'>"
        f"<div class='bar-fill' style='width:{pct}%;background:{color};'></div></div>"
    )


def kv_html(key, val) -> str:
    val_str = str(val) if val not in (None, "", []) else "—"
    return f"<div class='kv'><span class='kv-k'>{key}</span><span class='kv-v'>{val_str}</span></div>"


def badge_html(platform) -> str:
    p = (platform or "").lower()
    if p in PLATFORM_SVGS:
        color, svg = PLATFORM_SVGS[p]
        label = "X" if p == "twitter" else p.upper()
        return (
            f"<span style='display:inline-flex;align-items:center;gap:4px;"
            f"background:{color};color:#fff;border-radius:12px;padding:2px 8px;"
            f"font-size:11px;font-weight:700;'>{svg}{label}</span>"
        )
    return (
        f"<span style='display:inline-flex;align-items:center;background:#eaeef2;"
        f"color:#656d76;border:1px solid #d0d7de;border-radius:12px;padding:2px 8px;"
        f"font-size:11px;font-weight:700;'>🌐 {(platform or '?').upper()}</span>"
    )


def section(title: str) -> str:
    return f"<div class='section-title'>{title}</div>"


def prediction_html(prediction, score) -> str:
    icon, css, label = PREDICTION_META.get(prediction, PREDICTION_META[None])
    sc = f"{score:.3f}" if score is not None else "—"
    return f"<span class='{css}'>{icon} {label} &nbsp;<code style='font-size:13px;'>{sc}</code></span>"


def load_media_b64(path: str):
    """Lit un fichier local → (data_uri, mime_type) ou (None, None) si absent."""
    if not path:
        return None, None
    p = Path(path)
    if not p.exists():
        return None, None
    mime, _ = mimetypes.guess_type(str(p))
    if not mime:
        mime = {
            ".mp4": "video/mp4",   ".webm": "video/webm", ".mov": "video/quicktime",
            ".jpg": "image/jpeg",  ".jpeg": "image/jpeg",  ".png": "image/png",
            ".gif": "image/gif",   ".webp": "image/webp",
        }.get(p.suffix.lower(), "application/octet-stream")
    try:
        data = base64.b64encode(p.read_bytes()).decode()
        return f"data:{mime};base64,{data}", mime
    except Exception:
        return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# REQUÊTES MONGODB
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=30, show_spinner=False)
def get_distinct(_col_name: str, field: str) -> list:
    """Valeurs distinctes pour un champ donné (cachées 30 s)."""
    try:
        db  = get_db()
        col = db[_col_name]
        vals = col.distinct(field)
        return sorted(str(v) for v in vals if v)
    except Exception:
        return []


def build_query(col_name: str, f: dict) -> dict:
    q: dict = {}

    if f.get("project"):
        q["source.project"] = f["project"]
    if f.get("scan"):
        q["source.scan"] = f["scan"]
    if f.get("user"):
        q["source.user"] = f["user"]
    if f.get("platform"):
        q["platform"] = f["platform"]
    if f.get("df_status"):
        q["deepfake.status"] = f["df_status"]
    if f.get("df_prediction"):
        q["deepfake.prediction"] = f["df_prediction"]
    if f.get("df_score_min") is not None and f["df_score_min"] > 0.0:
        q["deepfake.final_score"] = {"$gte": f["df_score_min"]}
    if col_name == "posts" and f.get("search"):
        q["text.content"] = {"$regex": f["search"], "$options": "i"}
    if col_name == "media" and f.get("media_type"):
        q["type"] = f["media_type"]
    if col_name == "posts" and f.get("has_media"):
        q["deepfake.has_media"] = True

    return q


@st.cache_data(ttl=15, show_spinner=False)
def fetch_docs(_col_name: str, query_repr: str, skip: int = 0, limit: int = 20) -> list[dict]:
    """Récupère les documents paginés (caché 15 s)."""
    import json
    q = json.loads(query_repr)
    try:
        db  = get_db()
        col = db[_col_name]
        docs = list(col.find(q).sort("scraped_at", -1).skip(skip).limit(limit))
        return [_serialize(d) for d in docs]
    except Exception as e:
        st.error(f"Erreur MongoDB : {e}")
        return []


def _serialize(doc: dict) -> dict:
    """Convertit ObjectId → str récursivement."""
    out = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            out[k] = str(v)
        elif isinstance(v, dict):
            out[k] = _serialize(v)
        elif isinstance(v, list):
            out[k] = [_serialize(i) if isinstance(i, dict) else (str(i) if isinstance(i, ObjectId) else i) for i in v]
        else:
            out[k] = v
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# RENDU — CARTE LISTE
# ═══════════════════════════════════════════════════════════════════════════════

def card_post(doc: dict, selected: bool) -> str:
    platform   = doc.get("platform", "?")
    df         = doc.get("deepfake", {}) or {}
    prediction = df.get("prediction")
    score      = df.get("final_score")
    icon, css, label = PREDICTION_META.get(prediction, PREDICTION_META[None])
    score_str  = f"{score:.2f}" if score is not None else "—"
    text       = (doc.get("text") or {}).get("content") or ""
    preview    = (text[:85] + "…") if len(text) > 85 else text
    short_text = (text[:30] + "…") if len(text) > 30 else (text if text else "Sans texte")
    pub        = fmt_dt((doc.get("context") or {}).get("published_at"))
    media_icon = "📎" if df.get("has_media") else ""
    sel_cls    = "selected" if selected else ""
    src        = doc.get("source") or {}
    proj_tag   = f"<span style='color:#656d76;font-size:10px;'>📁 {src.get('project','')}</span>" if src.get("project") else ""

    return f"""
    <div class='doc-card {sel_cls}'>
      <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;'>
        {badge_html(platform)}
        <span class='{css}' style='font-size:12px;'>{icon} {score_str}</span>
      </div>
      <div style='font-size:11px;color:#656d76;margin-bottom:3px;'>
        {short_text} · {pub} {media_icon} {proj_tag}
      </div>
      <div style='font-size:12px;color:#1f2328;line-height:1.4;'>
        {preview or "<em style='color:#999'>pas de texte</em>"}
      </div>
    </div>"""


# ═══════════════════════════════════════════════════════════════════════════════
# RENDU — PANNEAU DÉTAIL
# ═══════════════════════════════════════════════════════════════════════════════

def render_post_detail(doc: dict):
    """Affiche le panneau de détail pour un post."""
    if not doc:
        st.markdown("<div style='color:#999;text-align:center;padding:40px;'>← Sélectionne un post</div>",
                    unsafe_allow_html=True)
        return

    platform = doc.get("platform", "?")
    df       = doc.get("deepfake", {}) or {}
    nlp      = doc.get("nlp", {}) or {}
    eng      = doc.get("engagement", {}) or {}
    ctx      = doc.get("context", {}) or {}
    txt      = doc.get("text", {}) or {}
    src      = doc.get("source", {}) or {}

    # ── En-tête ──────────────────────────────────────────────────────────────
    st.markdown(
        f"{badge_html(platform)} "
        f"<strong>@{doc.get('account_platform_id','?')}</strong>",
        unsafe_allow_html=True,
    )
    if doc.get("url"):
        st.markdown(f"[🔗 Lien original]({doc['url']})")

    # ── Contenu texte ─────────────────────────────────────────────────────────
    st.markdown(section("Contenu"), unsafe_allow_html=True)
    text_content = txt.get("content") or ""
    if text_content:
        st.markdown(
            f"<div class='content-block'>{text_content}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<span style='color:#8b949e;font-size:12px;'>Pas de texte</span>",
                    unsafe_allow_html=True)

    tags = txt.get("hashtags") or []
    if tags:
        st.markdown(
            " ".join(f"<span class='topic-tag'>#{t}</span>" for t in tags),
            unsafe_allow_html=True,
        )

    # ── Source projet/scan ────────────────────────────────────────────────────
    if any(src.values()):
        st.markdown(section("Source (projet / scan)"), unsafe_allow_html=True)
        st.markdown(
            kv_html("Projet", src.get("project")) +
            kv_html("Scan",   src.get("scan")) +
            kv_html("User",   src.get("user")),
            unsafe_allow_html=True,
        )

    # ── Deepfake ──────────────────────────────────────────────────────────────
    st.markdown(section("🔍 Analyse Deepfake"), unsafe_allow_html=True)
    st.markdown(prediction_html(df.get("prediction"), df.get("final_score")),
                unsafe_allow_html=True)
    st.markdown(bar_html(df.get("final_score"), "Score final"),
                unsafe_allow_html=True)

    df_status = df.get("status", "—")
    st.markdown(
        kv_html("Statut",          df_status) +
        kv_html("Divergence",      f"{df.get('model_divergence'):.3f}" if df.get("model_divergence") is not None else "—") +
        kv_html("Score artefact",  f"{df.get('artifact_score'):.3f}" if df.get("artifact_score") is not None else "—") +
        kv_html("Frames analysées",df.get("frames_analyzed")) +
        kv_html("Version pipeline",df.get("pipeline_version")) +
        kv_html("Analysé le",      fmt_dt(df.get("processed_at"))),
        unsafe_allow_html=True,
    )

    # Scores par modèle
    scores = df.get("scores") or {}
    if scores:
        st.markdown("<div style='margin-top:8px;'>", unsafe_allow_html=True)
        for model, val in scores.items():
            st.markdown(bar_html(val, f"↳ {model}"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if df.get("error"):
        st.markdown(
            f"<div style='color:#cf222e;font-size:11px;background:#ffebe9;"
            f"border:1px solid #ff8182;border-radius:4px;padding:6px 8px;margin-top:6px;'>"
            f"⚠️ {df['error']}</div>",
            unsafe_allow_html=True,
        )

    # ── NLP ───────────────────────────────────────────────────────────────────
    nlp_status = nlp.get("status", "pending")
    st.markdown(section("💬 NLP"), unsafe_allow_html=True)
    st.markdown(kv_html("Statut NLP", nlp_status), unsafe_allow_html=True)

    if nlp_status == "done":
        sent  = nlp.get("sentiment") or {}
        label = sent.get("label")
        s_icon = SENTIMENT_ICONS.get(label, "❓")
        st.markdown(
            kv_html("Sentiment", f"{s_icon} {label or '—'} ({sent.get('score') or '—'})") +
            kv_html("Modèle sentiment", sent.get("model")),
            unsafe_allow_html=True,
        )
        topics = nlp.get("topics") or []
        if topics:
            st.markdown(
                " ".join(f"<span class='topic-tag'>{t}</span>" for t in topics),
                unsafe_allow_html=True,
            )
        if nlp.get("is_duplicate_of"):
            st.markdown(
                f"<div style='color:#9a6700;font-size:12px;margin-top:4px;"
                f"background:#fff8c5;border:1px solid #d4a72c;border-radius:4px;padding:5px 8px;'>"
                f"⚠️ Doublon de : <code>{nlp['is_duplicate_of']}</code> "
                f"(sim={nlp.get('similarity_score','—')})</div>",
                unsafe_allow_html=True,
            )

    # ── Engagement ────────────────────────────────────────────────────────────
    st.markdown(section("📊 Engagement"), unsafe_allow_html=True)
    st.markdown(
        kv_html("Likes",     eng.get("likes", 0)) +
        kv_html("Partages",  eng.get("shares", 0)) +
        kv_html("Vues",      eng.get("views", 0)) +
        kv_html("Commentaires", eng.get("comments", 0)) +
        kv_html("Publié le", fmt_dt(ctx.get("published_at"))) +
        kv_html("Scrapé le", fmt_dt(doc.get("scraped_at"))),
        unsafe_allow_html=True,
    )

    # ── Médias attachés ───────────────────────────────────────────────────────
    media_refs = doc.get("media") or []
    if media_refs:
        st.markdown(section(f"🖼️ Médias attachés ({len(media_refs)})"),
                    unsafe_allow_html=True)
        for ref in media_refs:
            url_local = ref.get("url_local") or ""
            path      = Path(url_local) if url_local else None
            ext       = path.suffix.lower() if path else ""

            if path and path.exists():
                if ext in IMAGE_EXTS or ext == ".gif":
                    st.image(str(path), use_container_width=True)
                elif ext in VIDEO_EXTS:
                    _, mime = load_media_b64(url_local)
                    st.video(path.read_bytes(), format=mime or "video/mp4")
                else:
                    st.info(f"Type non prévisualisable : `{ext}`")
            else:
                st.caption(f"Fichier introuvable : `{url_local or '—'}`")


# ═══════════════════════════════════════════════════════════════════════════════
# POP-UP MÉDIA  (@st.dialog)
# ═══════════════════════════════════════════════════════════════════════════════

@st.dialog("📽️ Visualisation du média", width="large")
def show_media_popup(url_local: str, media_type: str, doc: dict):
    path = Path(url_local) if url_local else None
    ext  = path.suffix.lower() if path else ""

    if not path or not path.exists():
        st.warning(f"Fichier introuvable : `{url_local}`")
        st.markdown(f"URL originale : `{doc.get('url_original','—')}`")
        return

    # Affichage selon le type
    if ext in IMAGE_EXTS:
        st.image(str(path), use_container_width=True)

    elif ext in VIDEO_EXTS:
        data_uri, mime = load_media_b64(url_local)
        if data_uri:
            st.markdown(
                f"<video controls style='width:100%;max-height:500px;border-radius:8px;'>"
                f"<source src='{data_uri}' type='{mime}'>"
                f"Ton navigateur ne supporte pas la lecture vidéo.</video>",
                unsafe_allow_html=True,
            )
        else:
            st.error("Impossible de charger la vidéo.")

    elif ext == ".gif":
        st.image(str(path), use_container_width=True)

    else:
        st.info(f"Type de fichier non prévisualisable : `{ext}`")

    # Mini résumé deepfake en dessous
    df = doc.get("deepfake", {}) or {}
    if df.get("status") == "done":
        st.markdown("---")
        st.markdown(prediction_html(df.get("prediction"), df.get("final_score")),
                    unsafe_allow_html=True)
        scores = df.get("scores") or {}
        for model, val in scores.items():
            st.markdown(bar_html(val, f"↳ {model}"), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def render():
    st.markdown(_CSS, unsafe_allow_html=True)
    # ── Connexion ─────────────────────────────────────────────────────────────
    try:
        db = get_db()
    except Exception as e:
        st.error(f"❌ Connexion MongoDB impossible : {e}")
        st.stop()

    # ── Session state ─────────────────────────────────────────────────────────
    if "selected_id"  not in st.session_state:
        st.session_state.selected_id  = None
    if "selected_doc" not in st.session_state:
        st.session_state.selected_doc = None
    if "page"         not in st.session_state:
        st.session_state.page         = 0

    # ── Titre ─────────────────────────────────────────────────────────────────
    st.markdown(
        "<h2 style='color:#e6edf3;margin:0 0 12px 0;'>"
        "🔬 AI Forensics Explorer</h2>",
        unsafe_allow_html=True,
    )

    col_name = "posts"

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # 3 COLONNES : filtres | liste | détail
    # ══════════════════════════════════════════════════════════════════════════
    col_left, col_center, col_right = st.columns([1.8, 3.5, 3.2], gap="small")

    # ─────────────────────────────────────────────────────────────────────────
    # COLONNE GAUCHE : filtres
    # ─────────────────────────────────────────────────────────────────────────
    with col_left:
        st.markdown("<div class='filter-hdr'>Filtres</div>", unsafe_allow_html=True)

        # Projet / scan / user  (champ source — inséré par worker_import)
        projects = get_distinct(col_name, "source.project")
        project  = st.selectbox("📁 Projet", ["(tous)"] + projects, key="f_project")
        project  = None if project == "(tous)" else project

        scans = []
        if project:
            # Filtrer les scans liés au projet sélectionné
            try:
                scans = sorted(str(v) for v in db[col_name].distinct(
                    "source.scan", {"source.project": project}
                ) if v)
            except Exception:
                scans = get_distinct(col_name, "source.scan")
        else:
            scans = get_distinct(col_name, "source.scan")
        scan = st.selectbox("🗂️ Scan", ["(tous)"] + scans, key="f_scan")
        scan = None if scan == "(tous)" else scan

        users = []
        if project or scan:
            q_u: dict = {}
            if project: q_u["source.project"] = project
            if scan:    q_u["source.scan"]    = scan
            try:
                users = sorted(str(v) for v in db[col_name].distinct("source.user", q_u) if v)
            except Exception:
                pass
        else:
            users = get_distinct(col_name, "source.user")
        user = st.selectbox("👤 User", ["(tous)"] + users, key="f_user")
        user = None if user == "(tous)" else user

        st.markdown("---")

        # Plateforme
        platforms = get_distinct(col_name, "platform")
        platform  = st.selectbox("🌐 Plateforme", ["(toutes)"] + platforms, key="f_plat")
        platform  = None if platform == "(toutes)" else platform

        # Deepfake status
        df_statuses = ["(tous)", "done", "pending", "error", "skipped"]
        df_status   = st.selectbox("⚙️ Statut deepfake", df_statuses, key="f_dfs")
        df_status   = None if df_status == "(tous)" else df_status

        # Prédiction
        predictions = ["(toutes)", "synthetic", "suspicious", "likely_real"]
        df_pred     = st.selectbox("🎯 Prédiction", predictions, key="f_pred")
        df_pred     = None if df_pred == "(toutes)" else df_pred

        # Score minimum
        df_score_min = st.slider("Score deepfake min", 0.0, 1.0, 0.0, 0.05, key="f_score")

        st.markdown("---")

        # Options spécifiques
        search    = st.text_input("🔎 Recherche texte", key="f_search", placeholder="mot-clé…")
        has_media = st.checkbox("Avec média uniquement", key="f_hmedia")

        st.markdown("---")
        if st.button("🔄 Rafraîchir", use_container_width=True):
            fetch_docs.clear()
            get_distinct.clear()
            st.session_state.page = 0
            st.rerun()

        st.markdown("---")
        page_size = st.selectbox(
            "Posts par page",
            [10, 20, 50, 100],
            index=1,
            key="f_page_size",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Construction du filtre et requête
    # ─────────────────────────────────────────────────────────────────────────
    filters: dict = {
        "project":       project,
        "scan":          scan,
        "user":          user,
        "platform":      platform,
        "df_status":     df_status,
        "df_prediction": df_pred,
        "df_score_min":  df_score_min,
        "search":        search,
        "has_media":     has_media,
        "media_type":    None,
    }

    import json
    query     = build_query(col_name, filters)
    query_str = json.dumps(query, default=str)

    # Nombre total (pour la pagination) — limité à 5000 pour les perfs
    try:
        total_count = min(db[col_name].count_documents(query), 5000)
    except Exception:
        total_count = 0

    page_size  = int(st.session_state.get("f_page_size", 20))
    total_pages = max(1, -(-total_count // page_size))  # ceil division
    # Clamp la page courante si les filtres ont réduit les résultats
    if st.session_state.page >= total_pages:
        st.session_state.page = max(0, total_pages - 1)

    # Fetch uniquement la page courante
    docs = fetch_docs(col_name, query_str, skip=st.session_state.page * page_size, limit=page_size)

    # ─────────────────────────────────────────────────────────────────────────
    # COLONNE CENTRE : liste
    # ─────────────────────────────────────────────────────────────────────────
    with col_center:
        # ── Header : compteur + pagination ───────────────────────────────────
        h_left, h_mid, h_right = st.columns([2, 3, 2])
        with h_left:
            st.markdown(
                f"<div style='font-size:12px;color:#656d76;padding-top:6px;'>"
                f"<strong style='color:#1f2328;'>{total_count}</strong> résultat(s)</div>",
                unsafe_allow_html=True,
            )
        with h_mid:
            # Prev / Next
            p_left, p_label, p_right = st.columns([1, 2, 1])
            with p_left:
                if st.button("←", disabled=st.session_state.page == 0,
                             use_container_width=True, key="btn_prev"):
                    st.session_state.page -= 1
                    st.rerun()
            with p_label:
                st.markdown(
                    f"<div style='text-align:center;font-size:12px;padding-top:6px;color:#656d76;'>"
                    f"Page <strong>{st.session_state.page + 1}</strong> / {total_pages}</div>",
                    unsafe_allow_html=True,
                )
            with p_right:
                if st.button("→", disabled=st.session_state.page >= total_pages - 1,
                             use_container_width=True, key="btn_next"):
                    st.session_state.page += 1
                    st.rerun()
        with h_right:
            # Aller à une page précise
            target = st.number_input(
                "Aller à", min_value=1, max_value=total_pages,
                value=st.session_state.page + 1,
                step=1, key="goto_page", label_visibility="collapsed",
            )
            if int(target) - 1 != st.session_state.page:
                st.session_state.page = int(target) - 1
                st.rerun()

        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

        # ── Liste des cartes ──────────────────────────────────────────────────
        if not docs:
            st.markdown(
                "<div style='color:#999;text-align:center;padding:40px;'>Aucun document</div>",
                unsafe_allow_html=True,
            )
        else:
            card_fn = card_post
            for doc in docs:
                doc_id   = str(doc.get("_id", ""))
                selected = doc_id == st.session_state.selected_id

                # Carte à gauche + bouton ▶ discret à droite, alignés
                c_card, c_btn = st.columns([11, 1])
                with c_card:
                    st.markdown(card_fn(doc, selected), unsafe_allow_html=True)
                with c_btn:
                    # Petit espacement vertical pour centrer le bouton sur la carte
                    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
                    if st.button(
                        "▶",
                        key=f"sel_{doc_id}",
                        help="Voir le détail",
                        use_container_width=True,
                    ):
                        st.session_state.selected_id  = doc_id
                        st.session_state.selected_doc = doc
                        st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # COLONNE DROITE : détail
    # ─────────────────────────────────────────────────────────────────────────
    with col_right:
        with st.container(border=True):
            selected_doc = st.session_state.selected_doc
            render_post_detail(selected_doc)


if __name__ == "__main__":
    st.set_page_config(
        page_title="AI Forensics Explorer",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    render()
