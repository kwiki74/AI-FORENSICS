"""
AI-FORENSICS · Application principale
======================================
Point d'entrée unique — navigue entre les pages via leurs render().

Lancement :
    streamlit run WWW/app.py
"""

from __future__ import annotations
import sys
from pathlib import Path

# Rend les modules WWW importables
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

st.set_page_config(
    page_title="AI-FORENSICS",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS GLOBAL
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* Pleine largeur, suppression header Streamlit */
.block-container  { max-width:100% !important; padding:0 1rem 1rem 1rem !important; }
[data-testid="stHeader"] { display:none; }

/* ── Barre de navigation ── */
.nav-bar {
    display: flex;
    gap: 8px;
    align-items: center;
    padding: 8px 0 6px 0;
}

/* ── Titre principal ── */
.app-title {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: 0.04em;
    background: linear-gradient(90deg, #58a6ff 0%, #a371f7 50%, #f78166 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
}
.app-subtitle {
    font-size: 12px;
    color: #656d76;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0 0 10px 0;
}

/* ── CPU/RAM bar ── */
.sysbar {
    display: flex;
    align-items: center;
    gap: 16px;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 7px 14px;
    margin-bottom: 10px;
    flex-wrap: wrap;
}
.sysbar-label {
    font-size: 11px;
    color: #8b949e;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    white-space: nowrap;
}
.core-grid {
    display: flex;
    gap: 3px;
    align-items: flex-end;
    height: 28px;
}
.core-bar {
    width: 12px;
    border-radius: 2px 2px 0 0;
    transition: height 0.3s;
}
.ram-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
}
.ram-track {
    width: 120px;
    height: 8px;
    background: #30363d;
    border-radius: 4px;
    overflow: hidden;
}
.ram-fill {
    height: 100%;
    border-radius: 4px;
}
.ram-text {
    font-size: 12px;
    color: #e6edf3;
    font-weight: 600;
    white-space: nowrap;
}

/* ── Séparateur ── */
.page-divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 6px 0 12px 0;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════════════
PAGES: dict[str, dict] = {
    "supervision": {"label": "⚙️  Supervision",        "module": "supervision"},
    "forensics":   {"label": "🔬  Forensics Explorer",  "module": "forensics_explorer"},
    "graph":       {"label": "🕸  Graph",               "module": "graph"},
    "neo4j":       {"label": "🗃  Neo4j Explorer",      "module": "neo4j_explorer"},
}


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
if "active_page" not in st.session_state:
    st.session_state.active_page = "supervision"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER : CPU / RAM
# ═══════════════════════════════════════════════════════════════════════════════
def _core_color(pct: float) -> str:
    if pct >= 80:
        return "#cf222e"
    if pct >= 50:
        return "#d4a72c"
    return "#3fb950"


def sysbar_html() -> str:
    try:
        import psutil

        # CPU par cœur (intervalle court pour ne pas bloquer)
        per_cpu: list[float] = psutil.cpu_percent(percpu=True, interval=0.1)
        avg_cpu = sum(per_cpu) / len(per_cpu)

        # RAM
        ram = psutil.virtual_memory()
        ram_pct  = ram.percent
        ram_used = ram.used  / (1024 ** 3)
        ram_tot  = ram.total / (1024 ** 3)
        ram_color = _core_color(ram_pct)

        # Barres par cœur
        bars = ""
        for p in per_cpu:
            h   = max(4, int(p * 0.28))   # max height ≈ 28px à 100 %
            col = _core_color(p)
            bars += (
                f"<div class='core-bar' title='Core: {p:.0f}%' "
                f"style='height:{h}px;background:{col};'></div>"
            )

        avg_color = _core_color(avg_cpu)

        return f"""
        <div class='sysbar'>
          <span class='sysbar-label'>CPU</span>
          <div class='core-grid'>{bars}</div>
          <span style='font-size:12px;color:{avg_color};font-weight:700;white-space:nowrap;'>
            {avg_cpu:.0f}% moy · {len(per_cpu)} cœurs
          </span>

          <span class='sysbar-label' style='margin-left:8px;'>RAM</span>
          <div class='ram-wrap'>
            <div class='ram-track'>
              <div class='ram-fill' style='width:{ram_pct:.0f}%;background:{ram_color};'></div>
            </div>
            <span class='ram-text' style='color:{ram_color};'>
              {ram_used:.1f} / {ram_tot:.1f} Go &nbsp;({ram_pct:.0f}%)
            </span>
          </div>
        </div>
        """
    except ImportError:
        return "<div style='color:#656d76;font-size:11px;padding:4px 0;'>psutil non installé — <code>pip install psutil</code></div>"


# ═══════════════════════════════════════════════════════════════════════════════
# EN-TÊTE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<p class='app-title'>AI-FORENSICS</p>"
    "<p class='app-subtitle'>Plateforme de détection d'influence & analyse de médias</p>",
    unsafe_allow_html=True,
)

# ── Barre CPU / RAM ──────────────────────────────────────────────────────────
st.markdown(sysbar_html(), unsafe_allow_html=True)

# ── Navigation ───────────────────────────────────────────────────────────────
nav_cols = st.columns([2, 2, 2, 2, 6])
for i, (key, info) in enumerate(PAGES.items()):
    with nav_cols[i]:
        active = st.session_state.active_page == key
        if st.button(
            info["label"],
            key=f"nav_{key}",
            type="primary" if active else "secondary",
            use_container_width=True,
        ):
            st.session_state.active_page = key
            st.rerun()

st.markdown("<hr class='page-divider'>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# RENDU DE LA PAGE SÉLECTIONNÉE
# ═══════════════════════════════════════════════════════════════════════════════
_page = st.session_state.active_page

if _page == "supervision":
    from supervision import render
    render()

elif _page == "forensics":
    from forensics_explorer import render
    render()

elif _page == "graph":
    from graph import render
    render()

elif _page == "neo4j":
    from neo4j_explorer import render
    render()
