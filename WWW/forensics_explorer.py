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
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Forensics Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS  ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
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

/* ── Bouton discret (▶) dans la carte ── */
/* Supprime le margin-bottom entre la carte et le bouton adjacent */
[data-testid="stHorizontalBlock"]:has(.doc-card) { margin-bottom: 7px; }

/* Bouton ▶ : minimaliste, juste une icône */
button[kind="secondary"].btn-select {
    background: transparent !important;
    border: 1px solid #d0d7de !important;
    border-radius: 6px !important;
    color: #656d76 !important;
    padding: 0 !important;
    min-height: unset !important;
    height: 100% !important;
    font-size: 14px !important;
}
button[kind="secondary"].btn-select:hover {
    border-color: #0969da !important;
    color: #0969da !important;
    background: #ddf4ff !important;
}
</style>
""", unsafe_allow_html=True)


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

PLATFORM_ICONS = {
    "twitter": "🐦", "instagram": "📸",
    "tiktok": "🎵",  "telegram": "✈️",
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
    cls  = f"badge-{platform}" if platform in PLATFORM_ICONS else "badge-unknown"
    icon = PLATFORM_ICONS.get(platform, "🌐")
    return f"<span class='badge {cls}'>{icon} {(platform or '?').upper()}</span>"


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
    username   = doc.get("account_platform_id", "?")
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
        @{username} · {pub} {media_icon} {proj_tag}
      </div>
      <div style='font-size:12px;color:#1f2328;line-height:1.4;'>
        {preview or "<em style='color:#999'>pas de texte</em>"}
      </div>
    </div>"""


def card_media(doc: dict, selected: bool) -> str:
    df         = doc.get("deepfake", {}) or {}
    prediction = df.get("prediction")
    score      = df.get("final_score")
    icon, css, label = PREDICTION_META.get(prediction, PREDICTION_META[None])
    score_str  = f"{score:.2f}" if score is not None else "—"
    media_type = doc.get("type", "?")
    url_local  = doc.get("url_local") or ""
    fname      = Path(url_local).name if url_local else str(doc.get("_id", ""))[:14]
    meta       = doc.get("metadata") or {}
    dims       = f"{meta.get('width','?')}×{meta.get('height','?')}" if meta.get("width") else "—"
    size_kb    = f"{(meta.get('size_bytes') or 0) // 1024} KB" if meta.get("size_bytes") else "—"
    reuse      = doc.get("reuse", {}) or {}
    seen       = reuse.get("seen_count", 1)
    reuse_warn = f"<span style='color:#9a6700;'>⚠️ ×{seen}</span>" if seen > 1 else ""
    type_icon  = {"image": "🖼️", "video": "🎬", "audio": "🎵", "gif": "🎞️"}.get(media_type, "📄")
    sel_cls    = "selected" if selected else ""
    src        = doc.get("source") or {}
    proj_tag   = f"<span style='color:#656d76;font-size:10px;'>📁 {src.get('project','')}</span>" if src.get("project") else ""

    return f"""
    <div class='doc-card {sel_cls}'>
      <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;'>
        <span style='font-size:13px;color:#1f2328;'>{type_icon} <strong>{fname}</strong></span>
        <span class='{css}' style='font-size:12px;'>{icon} {score_str}</span>
      </div>
      <div style='font-size:11px;color:#656d76;'>
        {dims} · {size_kb} {reuse_warn} {proj_tag}
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
            f"<div style='font-size:13px;color:#1f2328;line-height:1.6;"
            f"background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;padding:10px;'>"
            f"{text_content}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<span style='color:#555;font-size:12px;'>Pas de texte</span>",
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
        for i, ref in enumerate(media_refs):
            url_local = ref.get("url_local") or ""
            mtype     = ref.get("type", "?")
            btn_label = f"{'🎬' if mtype == 'video' else '🖼️'} Ouvrir média {i+1}"
            if st.button(btn_label, key=f"open_media_ref_{doc.get('_id')}_{i}"):
                show_media_popup(url_local, mtype, ref)


def render_media_detail(doc: dict):
    """Affiche le panneau de détail pour un média."""
    if not doc:
        st.markdown("<div style='color:#999;text-align:center;padding:40px;'>← Sélectionne un média</div>",
                    unsafe_allow_html=True)
        return

    df        = doc.get("deepfake", {}) or {}
    meta      = doc.get("metadata", {}) or {}
    reuse     = doc.get("reuse", {}) or {}
    src       = doc.get("source", {}) or {}
    media_type = doc.get("type", "?")
    url_local  = doc.get("url_local") or ""
    fname      = Path(url_local).name if url_local else str(doc.get("_id", ""))

    type_icon = {"image": "🖼️", "video": "🎬", "audio": "🎵", "gif": "🎞️"}.get(media_type, "📄")
    st.markdown(f"**{type_icon} {fname}**")

    # Bouton d'ouverture média
    if url_local:
        if st.button("▶ Ouvrir le média", key=f"open_media_detail_{doc.get('_id')}"):
            show_media_popup(url_local, media_type, doc)

    # ── Source ───────────────────────────────────────────────────────────────
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

    st.markdown(
        kv_html("Statut",          df.get("status", "—")) +
        kv_html("Divergence",      f"{df.get('model_divergence'):.3f}" if df.get("model_divergence") is not None else "—") +
        kv_html("Score artefact",  f"{df.get('artifact_score'):.3f}" if df.get("artifact_score") is not None else "—") +
        kv_html("Frames analysées",df.get("frames_analyzed")) +
        kv_html("Faces détectées", df.get("faces_detected")) +
        kv_html("Version pipeline",df.get("pipeline_version")) +
        kv_html("Analysé le",      fmt_dt(df.get("processed_at"))),
        unsafe_allow_html=True,
    )

    # Scores par modèle
    scores = df.get("scores") or {}
    if scores:
        for model, val in scores.items():
            st.markdown(bar_html(val, f"↳ {model}"), unsafe_allow_html=True)

    raw_scores = df.get("raw_scores") or {}
    if raw_scores:
        st.markdown("<div style='font-size:11px;color:#555;margin-top:4px;'>Scores bruts (avant calibration)</div>",
                    unsafe_allow_html=True)
        for model, val in raw_scores.items():
            st.markdown(bar_html(val, f"↳ {model} (raw)"), unsafe_allow_html=True)

    if df.get("error"):
        st.markdown(
            f"<div style='color:#cf222e;font-size:11px;background:#ffebe9;"
            f"border:1px solid #ff8182;border-radius:4px;padding:6px 8px;margin-top:6px;'>"
            f"⚠️ {df['error']}</div>",
            unsafe_allow_html=True,
        )

    # ── Métadonnées fichier ───────────────────────────────────────────────────
    st.markdown(section("📁 Fichier"), unsafe_allow_html=True)
    size_str = f"{(meta.get('size_bytes') or 0) // 1024} KB" if meta.get("size_bytes") else "—"
    st.markdown(
        kv_html("Type",       media_type) +
        kv_html("Format",     meta.get("format")) +
        kv_html("Codec",      meta.get("codec")) +
        kv_html("Dimensions", f"{meta.get('width','?')}×{meta.get('height','?')}" if meta.get("width") else "—") +
        kv_html("Durée",      f"{meta.get('duration_sec')} s" if meta.get("duration_sec") else "—") +
        kv_html("FPS",        meta.get("fps")) +
        kv_html("Taille",     size_str) +
        kv_html("Hash MD5",   (doc.get("hash_md5") or "—")[:20] + "…" if doc.get("hash_md5") else "—") +
        kv_html("Chemin local", url_local[:55] + "…" if len(url_local) > 55 else url_local),
        unsafe_allow_html=True,
    )

    # ── Réutilisation ─────────────────────────────────────────────────────────
    seen = reuse.get("seen_count", 1)
    if seen > 1:
        st.markdown(section("⚠️ Réutilisation (signal campagne)"), unsafe_allow_html=True)
        platforms = reuse.get("platforms") or []
        st.markdown(
            f"<div style='color:#9a6700;font-weight:700;background:#fff8c5;"
            f"border:1px solid #d4a72c;border-radius:4px;padding:6px 8px;'>"
            f"Vu {seen}× sur {', '.join(platforms) or '—'}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            kv_html("Première apparition", fmt_dt(reuse.get("first_seen_at"))),
            unsafe_allow_html=True,
        )


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

def main():
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
    if "col_switch"   not in st.session_state:
        st.session_state.col_switch   = "posts"
    if "page"         not in st.session_state:
        st.session_state.page         = 0

    # ── Titre ─────────────────────────────────────────────────────────────────
    st.markdown(
        "<h2 style='color:#e6edf3;margin:0 0 12px 0;'>"
        "🔬 AI Forensics Explorer</h2>",
        unsafe_allow_html=True,
    )

    # ── Switch Posts / Médias ─────────────────────────────────────────────────
    col_tab1, col_tab2, col_tab3 = st.columns([1, 1, 6])
    with col_tab1:
        if st.button(
            "📝 Posts",
            type="primary" if st.session_state.col_switch == "posts" else "secondary",
            use_container_width=True,
        ):
            st.session_state.col_switch   = "posts"
            st.session_state.selected_id  = None
            st.session_state.selected_doc = None
            st.session_state.page         = 0
    with col_tab2:
        if st.button(
            "🖼️ Médias",
            type="primary" if st.session_state.col_switch == "media" else "secondary",
            use_container_width=True,
        ):
            st.session_state.col_switch   = "media"
            st.session_state.selected_id  = None
            st.session_state.selected_doc = None
            st.session_state.page         = 0

    col_name = st.session_state.col_switch

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
        if col_name == "posts":
            search   = st.text_input("🔎 Recherche texte", key="f_search", placeholder="mot-clé…")
            has_media = st.checkbox("Avec média uniquement", key="f_hmedia")
        else:
            search   = None
            has_media = False
            media_types = ["(tous)", "image", "video", "audio", "gif"]
            mt = st.selectbox("Type de média", media_types, key="f_mtype")
            media_type = None if mt == "(tous)" else mt

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
        "search":        search if col_name == "posts" else None,
        "has_media":     has_media if col_name == "posts" else False,
        "media_type":    media_type if col_name == "media" else None,
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
            card_fn = card_post if col_name == "posts" else card_media
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
            if col_name == "posts":
                render_post_detail(selected_doc)
            else:
                render_media_detail(selected_doc)


if __name__ == "__main__":
    main()
