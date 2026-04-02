"""
supervision.py  —  Supervision du pipeline AI-FORENSICS

Gère les 4 workers + état MongoDB depuis une interface Streamlit.
Les processus sont stockés dans _REGISTRY (module-level) :
ils survivent aux changements de page et aux reruns Streamlit.

Intégration dans l'app principale :
    from supervision import render
    render()
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# ══════════════════════════════════════════════════════════════════════════════
#  CHEMINS & ENV
# ══════════════════════════════════════════════════════════════════════════════

_ROOT       = Path("/home/kwiki/AI-FORENSICS")
_CONDA      = Path("/home/kwiki/anaconda3/envs")
_CURSOR     = _ROOT / "logs" / "import_cursor.json"
_IMPORT_CFG = _ROOT / "WORKER" / "IMPORT" / "worker_import.cfg"

# Chargement .env pour les credentials MongoDB
load_dotenv(_ROOT / ".env")

# ══════════════════════════════════════════════════════════════════════════════
#  REGISTRE GLOBAL DES PROCESSUS  (survit aux changements de page)
# ══════════════════════════════════════════════════════════════════════════════

_REGISTRY: dict[str, "ManagedProcess"] = {}

# ══════════════════════════════════════════════════════════════════════════════
#  ANSI → HTML (couleurs préservées)
# ══════════════════════════════════════════════════════════════════════════════

_ANSI_FG = {
    "30": "#555",     "31": "#e05252",  "32": "#4caf50",  "33": "#f0a500",
    "34": "#4fc3f7",  "35": "#ba68c8",  "36": "#4dd0e1",  "37": "#e0e0e0",
    "90": "#777",     "91": "#ff6b6b",  "92": "#69f0ae",  "93": "#ffd54f",
    "94": "#82b1ff",  "95": "#ea80fc",  "96": "#84ffff",  "97": "#ffffff",
}

def _ansi_to_html(text: str) -> str:
    """Convertit les codes ANSI en <span> HTML. Échappe le HTML au passage."""
    out: list[str] = []
    span_open = False
    i = 0
    while i < len(text):
        if text[i] == "\x1b" and i + 1 < len(text) and text[i + 1] == "[":
            j = i + 2
            while j < len(text) and text[j] not in "mGKHFABCDJsu":
                j += 1
            seq   = text[i + 2: j]
            final = text[j] if j < len(text) else ""
            i     = j + 1
            if final != "m":        # on ignore les séquences de déplacement
                continue
            if span_open:
                out.append("</span>")
                span_open = False
            codes  = seq.split(";") if seq else ["0"]
            styles: list[str] = []
            for c in codes:
                c = c.lstrip("0") or "0"
                if c == "0":
                    styles.clear()
                elif c == "1":
                    styles.append("font-weight:bold")
                elif c == "2":
                    styles.append("opacity:0.55")
                elif c in _ANSI_FG:
                    styles = [s for s in styles if not s.startswith("color:")]
                    styles.append(f"color:{_ANSI_FG[c]}")
            if styles:
                out.append(f'<span style="{";".join(styles)}">')
                span_open = True
        else:
            c = text[i]
            if   c == "&": out.append("&amp;")
            elif c == "<": out.append("&lt;")
            elif c == ">": out.append("&gt;")
            elif c == "\r": pass
            else: out.append(c)
            i += 1
    if span_open:
        out.append("</span>")
    return "".join(out)

# ══════════════════════════════════════════════════════════════════════════════
#  GESTIONNAIRE DE PROCESSUS
# ══════════════════════════════════════════════════════════════════════════════

class ManagedProcess:
    """Subprocess avec capture stdout/stderr dans un buffer circulaire."""

    def __init__(self, key: str, label: str, env: str, script: Path, cwd: Path):
        self.key    = key
        self.label  = label
        self.env    = env
        self.script = script
        self.cwd    = cwd

        self.proc:        Optional[subprocess.Popen] = None
        self.buffer:      deque[str]                 = deque(maxlen=500)
        self.started_at:  Optional[float]            = None
        self._thread:     Optional[threading.Thread] = None
        self._extra_args: list[str]                  = []

    # ── Cycle de vie ──────────────────────────────────────────────────────────

    def start(self, extra_args: list[str] | None = None) -> None:
        if self.is_running:
            return
        self._extra_args = extra_args or []
        python = str(_CONDA / self.env / "bin" / "python")
        cmd    = [python, str(self.script)] + self._extra_args
        self._log(f"▶ {' '.join(cmd)}")
        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(self.cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,   # crée un nouveau groupe de processus
            )
            self.started_at = time.time()
            self._thread = threading.Thread(target=self._capture, daemon=True)
            self._thread.start()
        except Exception as e:
            self._log(f"✗ Erreur démarrage : {e}")

    def stop(self) -> None:
        if self.proc and self.is_running:
            self._log("■ Arrêt (SIGTERM → groupe)…")
            try:
                pgid = os.getpgid(self.proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                self.proc.wait(timeout=15)
            except ProcessLookupError:
                pass   # déjà mort
            except subprocess.TimeoutExpired:
                self._log("↯ Timeout SIGTERM → SIGKILL")
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            except Exception as e:
                self._log(f"✗ Erreur arrêt : {e}")
        self.started_at = None

    def restart(self, extra_args: list[str] | None = None) -> None:
        self.stop()
        time.sleep(1)
        self.start(extra_args if extra_args is not None else self._extra_args)

    # ── Propriétés ────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def status(self) -> tuple[str, str]:
        if self.is_running:
            return "🟢", f"running  {self.uptime}"
        if self.proc is not None:
            rc = self.proc.returncode
            # rc < 0 → tué par un signal (ex. SIGTERM rc=-15) = arrêt normal
            return ("🔴", f"erreur (rc={rc})") if rc is not None and rc > 0 else ("⚫", "arrêté")
        return "⚫", "inactif"

    @property
    def uptime(self) -> str:
        if not self.is_running or self.started_at is None:
            return ""
        s = int(time.time() - self.started_at)
        h, r = divmod(s, 3600)
        m, s = divmod(r, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def log_text(self) -> str:
        return "\n".join(self.buffer) if self.buffer else "(aucune sortie)"

    def _log(self, msg: str) -> None:
        self.buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def _capture(self) -> None:
        try:
            for line in self.proc.stdout:
                self.buffer.append(line.rstrip())
        except Exception:
            pass
        rc = self.proc.returncode if self.proc else "?"
        self._log(f"— processus terminé (rc={rc})")


def _get(key: str, label: str, env: str, script: Path, cwd: Path) -> ManagedProcess:
    if key not in _REGISTRY:
        _REGISTRY[key] = ManagedProcess(key, label, env, script, cwd)
    return _REGISTRY[key]


# ══════════════════════════════════════════════════════════════════════════════
#  MONGODB — STATS (cache TTL 5s)
# ══════════════════════════════════════════════════════════════════════════════

_MONGO_CACHE:    dict = {"ts": 0.0, "data": None}
_PROJECTS_CACHE: dict = {"ts": 0.0, "data": []}
_NEO4J_CACHE:    dict = {"ts": 0.0, "connected": None, "error": "", "uri": ""}
_MONGO_CLIENT = None


def _mongo_db():
    global _MONGO_CLIENT
    try:
        from pymongo import MongoClient
        host    = os.getenv("MONGO_HOST", "localhost")
        port    = int(os.getenv("MONGO_PORT", 27017))
        user    = os.getenv("MONGO_USER")
        pwd     = os.getenv("MONGO_PASSWORD")
        db_name = os.getenv("MONGO_DB", "influence_detection")
        auth_db = os.getenv("MONGO_AUTH_DB", db_name)

        if _MONGO_CLIENT is None:
            kwargs = {"serverSelectionTimeoutMS": 2000}
            if user and pwd:
                kwargs.update(username=user, password=pwd, authSource=auth_db)
            _MONGO_CLIENT = MongoClient(host, port, **kwargs)

        return _MONGO_CLIENT[db_name]
    except Exception:
        return None


def _get_mongo_stats() -> dict:
    now = time.time()
    if now - _MONGO_CACHE["ts"] < 5 and _MONGO_CACHE["data"]:
        return _MONGO_CACHE["data"]

    db = _mongo_db()
    if db is None:
        return {"connected": False, "error": "pymongo non disponible"}

    try:
        db.client.admin.command("ping")
    except Exception as e:
        return {"connected": False, "error": str(e)}

    COLS = ["accounts", "posts", "comments", "media", "narratives", "campaigns", "jobs"]
    counts = {c: db[c].count_documents({}) for c in COLS}

    def job_stats(jtype: str) -> dict:
        return {
            s: db.jobs.count_documents({"type": jtype, "status": s})
            for s in ("pending", "done", "error")
        }

    def pipe_stats(col: str, field: str) -> dict:
        return {
            s: db[col].count_documents({f"{field}.status": s})
            for s in ("pending", "done", "error", "skipped")
        }

    def neo4j_stats(col: str) -> dict:
        return {
            "synced":  db[col].count_documents({"sync.neo4j": True}),
            "pending": db[col].count_documents({"sync.neo4j": False}),
        }

    data = {
        "connected": True,
        "host":   f"{os.getenv('MONGO_HOST','localhost')}:{os.getenv('MONGO_PORT',27017)}",
        "db":     os.getenv("MONGO_DB", "influence_detection"),
        "counts": counts,
        "jobs": {
            "deepfake_analysis": job_stats("deepfake_analysis"),
            "nlp_analysis":      job_stats("nlp_analysis"),
            "etl_sync":          job_stats("etl_sync"),
        },
        "pipeline": {
            "posts_deepfake": pipe_stats("posts", "deepfake") if counts["posts"] else None,
            "posts_nlp":      pipe_stats("posts", "nlp")      if counts["posts"] else None,
            "media_deepfake": pipe_stats("media", "deepfake") if counts["media"] else None,
        },
        "neo4j": {
            col: neo4j_stats(col)
            for col in ("posts", "comments", "media")
            if counts.get(col, 0) > 0
        },
    }
    _MONGO_CACHE["ts"]   = now
    _MONGO_CACHE["data"] = data
    return data


def _get_projects() -> list[str]:
    """Retourne les noms de projets distincts (source.project) depuis MongoDB. Cache 30 s."""
    now = time.time()
    if now - _PROJECTS_CACHE["ts"] < 30 and _PROJECTS_CACHE["data"]:
        return _PROJECTS_CACHE["data"]
    db = _mongo_db()
    if db is None:
        return _PROJECTS_CACHE["data"]
    try:
        projects = sorted(p for p in db.posts.distinct("source.project") if p)
    except Exception:
        return _PROJECTS_CACHE["data"]
    _PROJECTS_CACHE["ts"]   = now
    _PROJECTS_CACHE["data"] = projects
    return projects


def _get_neo4j_status() -> dict:
    """Teste la connectivité Neo4j. Cache 10 s."""
    now = time.time()
    if now - _NEO4J_CACHE["ts"] < 10 and _NEO4J_CACHE["connected"] is not None:
        return _NEO4J_CACHE
    uri  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER",     "neo4j")
    pwd  = os.getenv("NEO4J_PASSWORD", "")
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        driver.verify_connectivity()
        driver.close()
        _NEO4J_CACHE.update({"ts": now, "connected": True,  "error": "",       "uri": uri})
    except Exception as e:
        _NEO4J_CACHE.update({"ts": now, "connected": False, "error": str(e),   "uri": uri})
    return _NEO4J_CACHE


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def _read_source_dir() -> str:
    m = re.search(r"^source_dir\s*=\s*(.*)$", _IMPORT_CFG.read_text(), re.MULTILINE)
    return m.group(1).strip() if m else ""


def _write_source_dir(path: str) -> None:
    text = _IMPORT_CFG.read_text()
    text = re.sub(
        r"^(source_dir\s*=\s*).*$",
        lambda m: m.group(1) + path,
        text,
        flags=re.MULTILINE,
    )
    _IMPORT_CFG.write_text(text)


# ══════════════════════════════════════════════════════════════════════════════
#  LOG BOX (iframe — auto-scroll fiable)
# ══════════════════════════════════════════════════════════════════════════════

def _log_box(text: str, height: int = 430) -> None:
    html_lines = _ansi_to_html(text)
    components.html(
        f"""<!DOCTYPE html>
<html>
<head><style>
  body {{
    margin:0; padding:10px 14px;
    background:#0e1117; color:#e0e0e0;
    font-family:'Courier New',Courier,monospace;
    font-size:0.77rem; line-height:1.5;
    white-space:pre-wrap; word-break:break-all;
    overflow-y:auto;
  }}
</style></head>
<body>{html_lines}</body>
<script>
  function scrollBottom() {{ window.scrollTo(0, document.body.scrollHeight); }}
  scrollBottom();
  setTimeout(scrollBottom, 80);
</script>
</html>""",
        height=height,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MINI-WIDGETS STATS
# ══════════════════════════════════════════════════════════════════════════════


def _pipe_row(label: str, stats: dict | None) -> None:
    """Affiche une ligne done/pending/err pour un champ de pipeline."""
    if stats is None:
        return
    d  = stats.get("done",    0)
    p  = stats.get("pending", 0)
    e  = stats.get("error",   0)
    sk = stats.get("skipped", 0)
    color_p = "#f0a500" if p else "#555"
    color_e = "#e05252" if e else "#555"
    skip_str = f"  <span style='color:#555'>⏭{sk}</span>" if sk else ""
    st.markdown(
        f"<div style='font-size:0.72rem;margin:1px 0;'>"
        f"<span style='color:#888'>{label}</span>  "
        f"<span style='color:#4caf50'>✓{d}</span>  "
        f"<span style='color:{color_p}'>⏳{p}</span>  "
        f"<span style='color:{color_e}'>✗{e}</span>{skip_str}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CARTES WORKERS
# ══════════════════════════════════════════════════════════════════════════════

def _card_header(w: ManagedProcess) -> None:
    emoji, txt = w.status
    st.markdown(f"### {w.label}")
    st.caption(f"env : `{w.env}`")
    st.markdown(f"{emoji} **{txt}**")


def _buttons(w: ManagedProcess, extra_args: list[str] | None = None) -> None:
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("▶", key=f"start_{w.key}",
                     disabled=w.is_running, use_container_width=True):
            w.start(extra_args)
            st.session_state.log_target = w.key
            st.rerun()
    with b2:
        if st.button("■", key=f"stop_{w.key}",
                     disabled=not w.is_running, use_container_width=True):
            w.stop()
            st.session_state.log_target = w.key
            st.rerun()
    with b3:
        if st.button("↺", key=f"restart_{w.key}",
                     use_container_width=True):
            w.restart(extra_args)
            st.session_state.log_target = w.key
            st.rerun()


def _card_deepfake(w: ManagedProcess, stats: dict) -> None:
    with st.container(border=True):
        _card_header(w)
        if stats.get("connected"):
            _pipe_row("deepfake_analysis", stats["jobs"].get("deepfake_analysis"))
        st.markdown("---")
        n_workers = st.number_input(
            "Workers", min_value=1, max_value=16, value=2, step=1,
            key="df_workers",
        )
        _buttons(w, ["--mongojob", "--workers", str(int(n_workers))])


def _card_nlp(w: ManagedProcess, stats: dict) -> None:
    with st.container(border=True):
        _card_header(w)
        if stats.get("connected"):
            _pipe_row("posts.nlp", stats["pipeline"].get("posts_nlp"))
        st.markdown("---")
        _buttons(w)


def _card_import(w: ManagedProcess, stats: dict) -> None:
    with st.container(border=True):
        _card_header(w)
        if stats.get("connected"):
            counts = stats.get("counts", {})
            st.markdown("<div style='margin:4px 0 2px;font-size:0.7rem;color:#666;'>COLLECTIONS</div>",
                        unsafe_allow_html=True)
            for col in ("accounts", "posts", "comments", "media"):
                n = counts.get(col, 0)
                color = "#4caf50" if n > 0 else "#555"
                st.markdown(
                    f"<div style='font-size:0.72rem;'>"
                    f"<span style='color:#888'>{col}</span>  "
                    f"<span style='color:{color};font-weight:600'>{n:,}</span></div>",
                    unsafe_allow_html=True,
                )
        st.markdown("---")
        source = st.text_input(
            "source_dir",
            value=_read_source_dir(),
            key="import_source_dir",
            placeholder="/chemin/vers/DATA_IN",
        )
        if st.button("💾 Sauvegarder source_dir", key="import_save",
                     use_container_width=True):
            _write_source_dir(source)
            st.success("Sauvegardé ✓")
        _buttons(w)
        cursor_exists = _CURSOR.exists()
        if st.button(
            f"🗑 Supprimer cursor {'✓' if cursor_exists else '(absent)'}",
            key="del_cursor",
            disabled=not cursor_exists,
            use_container_width=True,
            help="Supprime import_cursor.json → réimporte tout depuis zéro",
        ):
            _CURSOR.unlink(missing_ok=True)
            st.success("import_cursor.json supprimé ✓")


def _card_network(w: ManagedProcess, stats: dict) -> None:
    with st.container(border=True):
        _card_header(w)
        if stats.get("connected"):
            st.markdown("<div style='margin:4px 0 2px;font-size:0.7rem;color:#666;'>SYNC NEO4J</div>",
                        unsafe_allow_html=True)
            s = stats.get("neo4j", {}).get("posts", {})
            ok  = s.get("synced", 0)
            pnd = s.get("pending", 0)
            color_p = "#f0a500" if pnd else "#555"
            st.markdown(
                f"<div style='font-size:0.72rem;'>"
                f"<span style='color:#888'>posts</span>  "
                f"<span style='color:#4caf50'>✓{ok}</span>  "
                f"<span style='color:{color_p}'>⏳{pnd}</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown("---")
        projects = _get_projects()
        selected = st.multiselect(
            "Projets",
            options=projects,
            key="net_projects_sel",
            placeholder="tous les projets" if not projects else "sélectionner…",
            label_visibility="collapsed",
        )
        add_mode = st.checkbox("--add  (ne pas vider Neo4j)", key="net_add_mode")

        def _net_args() -> list[str]:
            args: list[str] = []
            for p in selected:
                args += ["--projet", p]
            if add_mode:
                args.append("--add")
            return args

        _buttons(w, _net_args())


def _card_mongo(stats: dict) -> None:
    with st.container(border=True):
        if not stats.get("connected"):
            st.markdown("### Bases de données")
            st.markdown(f"🔴 **MongoDB déconnecté**")
            st.caption(stats.get("error", ""))
            return

        st.markdown("### Bases de données")
        st.caption(f"`{stats['db']}`  ·  `{stats['host']}`")

        # ── MongoDB ───────────────────────────────────────────────────────────
        st.markdown("🟢 **MongoDB connecté**")
        counts = stats.get("counts", {})
        total  = sum(counts.values())

        st.markdown("<div style='margin:6px 0 2px;font-size:0.7rem;color:#666;'>COLLECTIONS</div>",
                    unsafe_allow_html=True)
        for col in ("accounts", "posts", "comments", "media",
                    "narratives", "campaigns", "jobs"):
            n     = counts.get(col, 0)
            ratio = n / total if total else 0
            filled = round(ratio * 10)
            bar    = "█" * filled + "░" * (10 - filled)
            color  = "#4caf50" if n > 0 else "#444"
            st.markdown(
                f"<div style='font-family:monospace;font-size:0.72rem;'>"
                f"<span style='color:#888;display:inline-block;width:90px'>{col}</span>"
                f"<span style='color:{color};font-weight:600;display:inline-block;width:60px'>{n:,}</span>"
                f"<span style='color:#334;font-size:0.65rem'>{bar}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<div style='font-size:0.72rem;color:#777;margin-top:4px;'>Total : "
            f"<strong style='color:#ccc'>{total:,}</strong></div>",
            unsafe_allow_html=True,
        )

        # ── Neo4j ─────────────────────────────────────────────────────────────
        neo4j_st = _get_neo4j_status()
        if neo4j_st["connected"] is True:
            st.markdown("🟢 **Neo4j connecté**")
        elif neo4j_st["connected"] is False:
            st.markdown("🔴 **Neo4j déconnecté**")
            if neo4j_st.get("error"):
                st.caption(neo4j_st["error"])
        else:
            st.markdown("⚫ **Neo4j inconnu**")

        st.markdown("<div style='margin:4px 0 2px;font-size:0.7rem;color:#666;'>SYNC NEO4J</div>",
                    unsafe_allow_html=True)
        neo4j = stats.get("neo4j", {})
        if neo4j:
            for col, s in neo4j.items():
                ok  = s.get("synced", 0)
                pnd = s.get("pending", 0)
                color_p = "#f0a500" if pnd else "#555"
                st.markdown(
                    f"<div style='font-family:monospace;font-size:0.72rem;'>"
                    f"<span style='color:#888;display:inline-block;width:90px'>{col}</span>"
                    f"<span style='color:#4caf50'>✓{ok}</span>  "
                    f"<span style='color:{color_p}'>⏳{pnd}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("<div style='font-size:0.72rem;color:#555;'>aucune donnée</div>",
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def render() -> None:

    # ── Workers ───────────────────────────────────────────────────────────────
    w_deepfake = _get(
        "deepfake", "Deepfake", "forensics",
        _ROOT / "WORKER/DETECT_AI_PIPLINE/detect_ai_pipeline-v4.0.3.py",
        _ROOT / "WORKER/DETECT_AI_PIPLINE",
    )
    w_nlp = _get(
        "nlp", "NLP", "nlp_pipeline",
        _ROOT / "WORKER/NLP/nlp_worker.py",
        _ROOT / "WORKER/NLP",
    )
    w_import = _get(
        "import", "Import", "nlp_pipeline",
        _ROOT / "WORKER/IMPORT/worker_import.py",
        _ROOT / "WORKER/IMPORT",
    )
    w_network = _get(
        "network", "Network", "nlp_pipeline",
        _ROOT / "WORKER/NETWORK/network_worker.py",
        _ROOT / "WORKER/NETWORK",
    )

    workers_map = {
        "deepfake": w_deepfake,
        "nlp":      w_nlp,
        "import":   w_import,
        "network":  w_network,
    }

    # ── Session state ─────────────────────────────────────────────────────────
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = True
    if "log_target" not in st.session_state:
        st.session_state.log_target = "deepfake"

    # ── Stats MongoDB ─────────────────────────────────────────────────────────
    stats = _get_mongo_stats()

    # ── En-tête ───────────────────────────────────────────────────────────────
    h1, h2 = st.columns([5, 2])
    with h1:
        st.markdown("## Supervision Pipeline")
    with h2:
        st.toggle("🔄 Auto-refresh (5 s)", key="auto_refresh")

    st.divider()

    # ── Cartes : 4 workers + MongoDB ──────────────────────────────────────────
    col_df, col_nlp, col_imp, col_net, col_mg = st.columns(5)
    with col_df:   _card_deepfake(w_deepfake, stats)
    with col_nlp:  _card_nlp(w_nlp, stats)
    with col_imp:  _card_import(w_import, stats)
    with col_net:  _card_network(w_network, stats)
    with col_mg:   _card_mongo(stats)

    st.divider()

    # ── Logs ──────────────────────────────────────────────────────────────────
    log_col, _ = st.columns([3, 1])
    with log_col:
        st.selectbox(
            "Logs du worker",
            options=list(workers_map.keys()),
            format_func=lambda k: workers_map[k].label,
            key="log_target",
            label_visibility="collapsed",
        )

    _log_box(workers_map[st.session_state.log_target].log_text())

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    if st.session_state.auto_refresh:
        time.sleep(5)
        st.rerun()
