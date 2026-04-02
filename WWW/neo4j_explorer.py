#!/usr/bin/env python3
"""
neo4j_explorer.py — Explorateur de graphe Neo4j pour AI-FORENSICS
Streamlit + neovis.js (interactions identiques à Neo4j Browser)

Interactions :
  • Double-clic sur un nœud  → expand ses voisins directs (LIMIT 50)
  • Simple clic              → panneau de propriétés à droite
  • Drag / zoom / pan        → natif neovis.js
  • Clic sur une relation    → label + propriétés dans le panneau

Lancement :
  conda activate forensics
  streamlit run neo4j_explorer.py

Dépendances supplémentaires (pip) :
  pip install streamlit neo4j python-dotenv
"""

import json
import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as _stc
from dotenv import load_dotenv
from neo4j import GraphDatabase, exceptions as neo4j_exc

# ── Chargement .env ────────────────────────────────────────────────────────────
_root = Path(__file__).resolve().parent
for _candidate in [
    _root / ".env",               # WWW/.env         (local, prioritaire)
    _root.parent / ".env",        # AI-FORENSICS/.env (principal ✅)
    _root.parent.parent / ".env", # fallback
]:
    if _candidate.exists():
        load_dotenv(_candidate, override=False)
        break

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# ── Constantes visuelles ───────────────────────────────────────────────────────
NODE_STYLES = {
    "Account":   {"color": "#4A90D9", "size": 28, "icon": "👤"},
    "Post":      {"color": "#7B68EE", "size": 22, "icon": "📝"},
    "Hashtag":   {"color": "#50C878", "size": 18, "icon": "#"},
    "Campaign":  {"color": "#FF6B6B", "size": 34, "icon": "🚨"},
    "Media":     {"color": "#FFB347", "size": 20, "icon": "🖼"},
    "Deepfake":  {"color": "#FF4500", "size": 26, "icon": "⚠️"},
    "Narrative": {"color": "#9370DB", "size": 24, "icon": "💬"},
    "Project":   {"color": "#20B2AA", "size": 30, "icon": "📁"},
}
DEFAULT_STYLE = {"color": "#AAAAAA", "size": 18, "icon": "●"}

# ── Connexion Neo4j ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def neo4j_ok() -> bool:
    try:
        get_driver().verify_connectivity()
        return True
    except Exception:
        return False


def run_query(cypher: str, params: dict = None) -> list[dict]:
    params = params or {}
    with get_driver().session() as session:
        result = session.run(cypher, **params)
        return [dict(record) for record in result]


# ── Helpers données ────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def get_projects() -> list[str]:
    try:
        rows = run_query("MATCH (p:Project) RETURN p.name AS name ORDER BY name")
        return [r["name"] for r in rows if r["name"]]
    except Exception:
        return []


@st.cache_data(ttl=30)
def get_campaigns(project: str | None) -> list[str]:
    try:
        if project:
            rows = run_query(
                "MATCH (c:Campaign)-[:BELONGS_TO]->(p:Project {name:$proj}) "
                "RETURN c.name AS name ORDER BY name",
                {"proj": project},
            )
        else:
            rows = run_query("MATCH (c:Campaign) RETURN c.name AS name ORDER BY name")
        return [r["name"] for r in rows if r["name"]]
    except Exception:
        return []


@st.cache_data(ttl=15)
def get_stats() -> dict:
    """Compteurs globaux pour la sidebar."""
    try:
        labels = run_query("CALL db.labels() YIELD label RETURN label")
        counts = {}
        for row in labels:
            lbl = row["label"]
            r = run_query(f"MATCH (n:`{lbl}`) RETURN count(n) AS c")
            counts[lbl] = r[0]["c"] if r else 0
        rel_count = run_query("MATCH ()-[r]->() RETURN count(r) AS c")
        counts["_relations"] = rel_count[0]["c"] if rel_count else 0
        return counts
    except Exception:
        return {}


def build_initial_cypher(
    project: str | None,
    campaign: str | None,
    node_types: list[str],
    limit: int,
) -> str:
    """
    Cypher de chargement initial selon les filtres choisis.
    Retourne uniquement des nœuds + relations directes.
    """
    type_filter = ""
    if node_types:
        labels_or = "|".join(node_types)
        type_filter = f" WHERE any(lbl IN labels(n) WHERE lbl IN {json.dumps(node_types)})"

    if campaign:
        return (
            f"MATCH (c:Campaign {{name: '{campaign}'}})-[r*1..2]-(n) "
            f"WITH n, r LIMIT {limit} "
            f"MATCH (n)-[rel]-(m) RETURN n, rel, m LIMIT {limit}"
        )
    elif project:
        return (
            f"MATCH (p:Project {{name: '{project}'}})-[r*1..3]-(n) "
            f"WITH n LIMIT {limit} "
            f"MATCH (n)-[rel]-(m) RETURN n, rel, m LIMIT {limit}"
        )
    else:
        return (
            f"MATCH (n)-[rel]-(m)"
            f"{' WHERE any(lbl IN labels(n) WHERE lbl IN ' + json.dumps(node_types) + ')' if node_types else ''} "
            f"RETURN n, rel, m LIMIT {limit}"
        )


# ── Génération HTML neovis.js ──────────────────────────────────────────────────
def build_neovis_html(
    bolt_uri: str,
    user: str,
    password: str,
    initial_cypher: str,
    node_styles: dict,
    height: int = 680,
) -> str:
    """
    Génère un composant HTML autonome avec neovis.js.
    Communique avec Streamlit via window.parent.postMessage().
    """

    # Sérialise la config des labels pour neovis
    # Échapper le mot de passe pour injection JS (gère les caractères spéciaux)
    import json as _json
    password_js = _json.dumps(password)  # ajoute les guillemets + échappe correctement

    labels_config = {}
    for label, style in node_styles.items():
        labels_config[label] = {
            "caption": "name",
            "size": style["size"],
            "community": label,
            "title_properties": ["name", "username", "platform", "score", "prediction"],
        }

    # Palette de couleurs pour neovis (via communauté)
    color_map = {label: style["color"] for label, style in node_styles.items()}

    labels_config_json = json.dumps(labels_config)
    color_map_json     = json.dumps(color_map)
    initial_cypher_js  = json.dumps(initial_cypher)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  :root {{
    --bg:           #0e1117;
    --bg-overlay:   rgba(20,24,36,0.97);
    --border:       #2a2f3d;
    --border-row:   #1f2535;
    --text:         #e0e0e0;
    --text-muted:   #888;
    --text-key:     #7eb6ff;
    --hint:         #444;
    --canvas-bg:    #0e1117;
  }}
  body.theme-light {{
    --bg:           #ffffff;
    --bg-overlay:   rgba(245,246,250,0.98);
    --border:       #d0d4e0;
    --border-row:   #e8eaf0;
    --text:         #1a1a2e;
    --text-muted:   #555;
    --text-key:     #1a6fc4;
    --hint:         #aaa;
    --canvas-bg:    #f8f9fc;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); font-family: 'Inter', sans-serif; overflow: hidden; transition: background 0.2s; }}
  #graph {{ width: 100%; height: {height}px; background: var(--canvas-bg); }}
  #overlay {{
    position: absolute; top: 10px; right: 10px; width: 270px;
    background: var(--bg-overlay); border: 1px solid var(--border);
    border-radius: 8px; padding: 14px; display: none;
    font-size: 12px; color: var(--text); z-index: 100;
    max-height: {height - 40}px; overflow-y: auto;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    transition: background 0.2s, border-color 0.2s;
  }}
  #overlay h3 {{
    margin-bottom: 10px; font-size: 13px; color: var(--text-muted);
    border-bottom: 1px solid var(--border); padding-bottom: 6px;
  }}
  .prop-row {{
    display: flex; gap: 8px; margin-bottom: 5px;
    border-bottom: 1px solid var(--border-row); padding-bottom: 4px;
  }}
  .prop-key  {{ color: var(--text-key); min-width: 90px; word-break: break-all; }}
  .prop-val  {{ color: var(--text); word-break: break-all; flex: 1; }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: 11px; margin-bottom: 8px; font-weight: 600;
  }}
  #hint {{ position: absolute; bottom: 10px; left: 10px; color: var(--hint); font-size: 11px; pointer-events: none; }}
  #loading {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); color: var(--text-muted); font-size: 14px; }}
</style>
</head>
<body>
<div id="graph"></div>
<div id="overlay"><h3 id="overlay-title">Nœud</h3><div id="overlay-body"></div></div>
<div id="hint">🖱 Double-clic : expand · Clic : détails · Scroll : zoom · Drag : déplacer</div>
<div id="loading">⏳ Chargement du graphe…</div>

<script src="https://unpkg.com/neovis.js@2.0.2/dist/neovis.js"></script>
<script>
// ── Palette couleurs ──────────────────────────────────────────────
const COLOR_MAP   = {color_map_json};
const DEFAULT_CLR = "#888888";

function nodeColor(labels) {{
  if (!labels) return DEFAULT_CLR;
  for (const lbl of labels) {{
    if (COLOR_MAP[lbl]) return COLOR_MAP[lbl];
  }}
  return DEFAULT_CLR;
}}

// ── Config neovis v2 ──────────────────────────────────────────────
const config = {{
  containerId: "graph",          // v2 : containerId (pas container_id)

  neo4j: {{                       // v2 : credentials dans un sous-objet neo4j
    serverUrl:      "{bolt_uri}",
    serverUser:     "{user}",
    serverPassword: {password_js},
  }},

  labels: {labels_config_json},

  relationships: {{
    "POSTED":        {{ caption: false, thickness: "weight" }},
    "USES_HASHTAG":  {{ caption: false }},
    "REPOSTED":      {{ caption: true  }},
    "BELONGS_TO":    {{ caption: false }},
    "AMPLIFIED_BY":  {{ caption: false }},
    "FORWARDED":     {{ caption: false }},
    "PART_OF":       {{ caption: false }},
  }},

  initialCypher: {initial_cypher_js},   // v2 : initialCypher (camelCase)

  arrows: true,
  hierarchical: false,
  visConfig: {{                  // v2 : options vis.js dans visConfig
    configure: {{ enabled: false }},
    background: {{ color: "transparent" }},
    physics: {{
      enabled: true,
      stabilization: {{ iterations: 150 }},
      barnesHut: {{
        gravitationalConstant: -8000,
        centralGravity: 0.3,
        springLength: 120,
        springConstant: 0.04,
        damping: 0.09,
      }},
    }},
  }},
}};

const viz = new NeoVis.default(config);
window.viz = viz; // debug

// ── État du graphe (nœuds connus) ────────────────────────────────
const knownNodeIds = new Set();
const expandedIds  = new Set();

viz.registerOnEvent("completed", () => {{
  document.getElementById("loading").style.display = "none";

  // neovis v2 : viz.network (sans underscore) est l'instance vis.js
  // viz._network n'est disponible qu'APRÈS le rendu complet
  // On utilise setTimeout pour laisser vis.js terminer l'init
  setTimeout(() => {{
    const network = viz.network;  // API publique neovis v2
    window._vizNetwork = network;
    if (!network) {{
      console.error("viz.network toujours null après timeout");
      return;
    }}

    // ── Coloration par label ────────────────────────────────────
    // neovis v2 : viz.nodes est un DataSet vis.js
    const nodeDataset = viz.nodes;
    if (nodeDataset) {{
      const updates = [];
      nodeDataset.get().forEach(node => {{
        knownNodeIds.add(node.id);
        // neovis v2 : les labels Neo4j sont dans node.raw.labels
        const labels = node.raw?.labels || node.neo4j_labels || [];
        const clr = nodeColor(labels);
        updates.push({{
          id: node.id,
          color: {{ background: clr, border: clr, highlight: {{ background: clr, border: "#FFD700" }} }},
          font:  {{ color: "#fff", size: 13 }},
        }});
      }});
      if (updates.length) nodeDataset.update(updates);
    }}

    // ── Clic simple → panneau propriétés ───────────────────────
    network.on("click", (params) => {{
      if (params.nodes && params.nodes.length > 0) {{
        const nodeId = params.nodes[0];
        const node   = viz.nodes ? viz.nodes.get(nodeId) : null;
        if (node) showPanel(node);
        document.getElementById("overlay").style.display = "block";
        return;
      }}
      if (params.edges && params.edges.length > 0) {{
        const edgeId = params.edges[0];
        const edge   = viz.edges ? viz.edges.get(edgeId) : null;
        if (edge) showRelPanel(edge);
        document.getElementById("overlay").style.display = "block";
        return;
      }}
      document.getElementById("overlay").style.display = "none";
    }});

    // ── Double-clic → expand voisins ───────────────────────────
    network.on("doubleClick", (params) => {{
      if (!params.nodes || params.nodes.length === 0) return;
      const nodeId = params.nodes[0];
      if (expandedIds.has(nodeId)) return;
      expandedIds.add(nodeId);

      const cypher = `MATCH (n)-[r]-(m) WHERE id(n) = ${{nodeId}} RETURN n, r, m LIMIT 50`;
      if (typeof viz.updateWithCypher === "function") {{
        viz.updateWithCypher(cypher);
      }} else {{
        viz.renderWithCypher(cypher);
      }}

      // Bordure dorée = nœud déjà expansé
      if (viz.nodes) {{
        viz.nodes.update({{
          id: nodeId,
          borderWidth: 4,
          color: {{ border: "#FFD700" }},
        }});
      }}
    }});

  }}, 100); // 100ms suffisent pour laisser vis.js terminer
}});

// ── Panneau propriétés ───────────────────────────────────────────
function showPanel(node) {{
  const overlay = document.getElementById("overlay");
  const body    = document.getElementById("overlay-body");
  const title   = document.getElementById("overlay-title");

  // neovis v2 : les données Neo4j sont dans node.raw_node
  // Fallbacks pour couvrir les différentes versions
  const labels = node.raw?.labels
               || node.neo4j_labels
               || [];
  const props  = node.raw?.properties
               || node.raw_node?.properties
               || {{}};

  // Filtrer les clés internes vis.js
  const skipKeys = new Set(["id","label","x","y","fixed","color","font","size",
                             "shape","borderWidth","neo4j_labels","raw_node","raw"]);
  const color = nodeColor(labels);

  title.innerHTML = labels.length
    ? labels.map(l => `<span class="badge" style="background:${{color}}22;color:${{color}};border:1px solid ${{color}}">${{l}}</span>`).join(" ")
    : `<span class="badge" style="background:#33333399;color:#aaa">Nœud</span>`;

  // Propriétés Neo4j en priorité
  let html = "";
  const allProps = Object.assign({{}}, props);
  // Ajouter aussi les props vis.js non-internes si pas dans raw_node
  for (const [k, v] of Object.entries(node)) {{
    if (!skipKeys.has(k) && !(k in allProps)) allProps[k] = v;
  }}

  for (const [k, v] of Object.entries(allProps)) {{
    if (v == null || v === "") continue;
    let display = typeof v === "object" ? JSON.stringify(v) : String(v);
    if (display.length > 200) display = display.slice(0, 200) + "…";
    html += `<div class="prop-row"><span class="prop-key">${{k}}</span><span class="prop-val">${{display}}</span></div>`;
  }}
  body.innerHTML = html || "<em style='color:#666'>Aucune propriété</em>";
  overlay.style.display = "block";
}}

function showRelPanel(rel) {{
  const overlay = document.getElementById("overlay");
  const body    = document.getElementById("overlay-body");
  const title   = document.getElementById("overlay-title");

  // neovis v2 : type dans rel.raw_edge?.type ou rel.label
  const type  = rel.raw_edge?.type || rel.raw?.type || rel.label || "relation";
  const props = rel.raw_edge?.properties || rel.raw?.properties || {{}};

  title.innerHTML = `<span class="badge" style="background:#33330099;color:#FFD700;border:1px solid #FFD700">${{type}}</span>`;

  let html = "";
  for (const [k, v] of Object.entries(props)) {{
    if (v == null || v === "") continue;
    html += `<div class="prop-row"><span class="prop-key">${{k}}</span><span class="prop-val">${{String(v)}}</span></div>`;
  }}
  body.innerHTML = html || "<em style='color:#666'>Pas de propriétés sur cette relation</em>";
  overlay.style.display = "block";
}}

// (clic/désélection gérés via viz._network.on dans "completed")

// ── Détection thème Streamlit ────────────────────────────────────
function applyTheme(isDark) {{
  const bgColor    = isDark ? "#0e1117" : "#f8f9fc";
  const fontColor  = isDark ? "#ffffff" : "#1a1a2e";
  const edgeColor  = isDark ? "#555555" : "#aaaaaa";
  const edgeHL     = isDark ? "#FFD700" : "#1a6fc4";

  // Basculer classe CSS pour overlay/panneau
  document.body.classList.toggle("theme-light", !isDark);

  // ── Fond du canvas vis.js ─────────────────────────────────────
  // vis.js peint le canvas lui-même — il faut patcher via 2D context
  const canvas = document.querySelector("#graph canvas");
  if (canvas) {{
    // Surcharge du fillStyle par défaut de vis.js :
    // on intercepte clearRect pour repeindre le fond à chaque frame
    if (!canvas._themePatched) {{
      const ctx = canvas.getContext("2d");
      const origClearRect = ctx.clearRect.bind(ctx);
      ctx.clearRect = function(x, y, w, h) {{
        origClearRect(x, y, w, h);
        ctx.fillStyle = canvas._themeBg || "#0e1117";
        ctx.fillRect(x, y, w, h);
      }};
      canvas._themePatched = true;
    }}
    canvas._themeBg = bgColor;
    // Forcer un redraw immédiat
    if (window._vizNetwork) window._vizNetwork.redraw();
  }}

  // ── Options vis.js ────────────────────────────────────────────
  if (window._vizNetwork) {{
    window._vizNetwork.setOptions({{
      nodes: {{ font: {{ color: fontColor }} }},
      edges: {{ font: {{ color: edgeColor }}, color: {{ color: edgeColor, highlight: edgeHL }} }},
    }});
  }}
}}
// Détection initiale OS
applyTheme(window.matchMedia("(prefers-color-scheme: dark)").matches);
// Changement OS en temps réel
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", e => applyTheme(e.matches));
// Message explicite depuis Streamlit
window.addEventListener("message", (e) => {{
  if (e.data?.type === "streamlit:theme") applyTheme(e.data?.theme?.base !== "light");
}});

// ── Lancement ────────────────────────────────────────────────────
viz.render();
</script>
</body>
</html>"""
    return html


# ── Interface Streamlit ────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="AI-FORENSICS · Graph Explorer",
        page_icon="🕸",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Détection thème Streamlit → injection dans l'iframe ──────────
    # Streamlit stocke le thème dans localStorage["stActiveTheme"]
    # On injecte un script qui le lit et l'envoie à l'iframe via postMessage
    st.markdown("""
    <script>
    (function() {
      function sendThemeToIframe() {
        // Streamlit met à jour data-theme sur <html>
        const htmlEl = document.documentElement;
        const base = htmlEl.getAttribute("data-theme") || 
                     (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
        const iframe = document.querySelector('iframe[title="neo4j_graph"]') ||
                       document.querySelector('iframe');
        if (iframe && iframe.contentWindow) {
          iframe.contentWindow.postMessage({ type: "streamlit:theme", theme: { base } }, "*");
        }
      }
      // Au chargement
      setTimeout(sendThemeToIframe, 800);
      setTimeout(sendThemeToIframe, 2000);
      // Observer les changements de thème Streamlit
      const obs = new MutationObserver(sendThemeToIframe);
      obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    })();
    </script>
    <style>
    /* Thème adaptatif — surcharge uniquement les couleurs structurelles */
    html[data-theme="light"] body,
    html[data-theme="light"] [data-testid="stAppViewContainer"] { background: #f8f9fc !important; }
    html[data-theme="light"] [data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #e0e4ef !important; }
    html[data-theme="light"] h1, html[data-theme="light"] h2,
    html[data-theme="light"] h3, html[data-theme="light"] p { color: #1a1a2e !important; }
    html[data-theme="light"] .metric-box { background: #ffffff; border: 1px solid #d0d4e0; }
    html[data-theme="light"] .metric-box .lbl { color: #555; }

    html[data-theme="dark"] body,
    html[data-theme="dark"] [data-testid="stAppViewContainer"] { background: #0e1117 !important; }
    html[data-theme="dark"] [data-testid="stSidebar"] { background: #14181f !important; border-right: 1px solid #1f2535 !important; }
    html[data-theme="dark"] h1, html[data-theme="dark"] h2,
    html[data-theme="dark"] h3, html[data-theme="dark"] p { color: #e0e0e0 !important; }
    html[data-theme="dark"] .metric-box { background: #14181f; border: 1px solid #1f2535; }
    html[data-theme="dark"] .metric-box .lbl { color: #777; }

    .metric-box { border-radius: 8px; padding: 10px 14px; text-align: center; }
    .metric-box .val { font-size: 22px; font-weight: 700; color: #4A90D9; }
    .stSelectbox label, .stMultiSelect label, .stSlider label { color: #aaa !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🕸 Graph Explorer")
        st.markdown("---")

        # Statut connexion
        ok = neo4j_ok()
        if ok:
            st.success(f"✅ Connecté · {NEO4J_URI}", icon=None)
        else:
            st.error(f"❌ Hors ligne · {NEO4J_URI}")
            st.info("Vérifiez NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD dans .env")
            st.stop()

        st.markdown("---")
        st.markdown("#### Filtres")

        # Projet
        projects = get_projects()
        project_options = ["— Tous les projets —"] + projects
        project_sel = st.selectbox("Projet", project_options, index=0)
        project = None if project_sel.startswith("—") else project_sel

        # Campagne
        campaigns = get_campaigns(project)
        campaign_options = ["— Toutes les campagnes —"] + campaigns
        campaign_sel = st.selectbox("Campagne", campaign_options, index=0)
        campaign = None if campaign_sel.startswith("—") else campaign_sel

        # Types de nœuds
        all_labels = list(NODE_STYLES.keys())
        node_types = st.multiselect(
            "Types de nœuds",
            all_labels,
            default=["Account", "Post", "Campaign", "Hashtag"],
            help="Laisser vide = tous les types",
        )

        # Limite
        limit = st.slider("Nœuds max (chargement initial)", 50, 500, 150, step=50)

        st.markdown("---")
        st.markdown("#### Requête Cypher libre")
        custom_cypher = st.text_area(
            "Cypher",
            placeholder="MATCH (n:Account)-[r]->(m) RETURN n, r, m LIMIT 100",
            height=90,
        )
        run_custom = st.button("▶ Exécuter", use_container_width=True)

        st.markdown("---")
        st.markdown("#### Statistiques")
        stats = get_stats()
        if stats:
            for lbl, style in NODE_STYLES.items():
                c = stats.get(lbl, 0)
                if c:
                    st.markdown(
                        f"<div class='metric-box'>"
                        f"<div class='val' style='color:{style['color']}'>{c:,}</div>"
                        f"<div class='lbl'>{style['icon']} {lbl}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            rels = stats.get("_relations", 0)
            if rels:
                st.markdown(
                    f"<div class='metric-box'>"
                    f"<div class='val' style='color:#FFD700'>{rels:,}</div>"
                    f"<div class='lbl'>↔ Relations</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Zone principale ────────────────────────────────────────────
    st.markdown(
        "<h2 style='margin-bottom:4px'>🕸 Explorateur de graphe Neo4j</h2>"
        "<p style='color:#666;font-size:13px;margin-bottom:16px'>"
        "Double-clic sur un nœud pour déployer ses voisins · Clic pour les propriétés"
        "</p>",
        unsafe_allow_html=True,
    )

    # Légende inline
    legend_html = "<div style='display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px'>"
    for lbl, style in NODE_STYLES.items():
        legend_html += (
            f"<span style='display:flex;align-items:center;gap:5px;font-size:12px;color:#ccc'>"
            f"<span style='width:12px;height:12px;border-radius:50%;background:{style['color']};display:inline-block'></span>"
            f"{style['icon']} {lbl}</span>"
        )
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)

    # Déterminer le Cypher à rendre
    if run_custom and custom_cypher.strip():
        cypher = custom_cypher.strip()
    else:
        cypher = build_initial_cypher(project, campaign, node_types, limit)

    # Afficher le Cypher actif
    with st.expander("🔍 Cypher actif", expanded=False):
        st.code(cypher, language="cypher")

    # ── Rendu neovis ───────────────────────────────────────────────
    html = build_neovis_html(
        bolt_uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        initial_cypher=cypher,
        node_styles=NODE_STYLES,
        height=660,
    )

    st.components.v1.html(html, height=680, scrolling=False)

    # ── Requête Cypher libre depuis la sidebar (résultats tabulaires) ──
    if run_custom and custom_cypher.strip():
        st.markdown("---")
        st.markdown("#### Résultats bruts")
        try:
            rows = run_query(custom_cypher.strip())
            if rows:
                import pandas as pd
                # Aplatir les valeurs Neo4j
                flat = []
                for row in rows[:200]:
                    flat_row = {}
                    for k, v in row.items():
                        if hasattr(v, "_properties"):   # Node / Relationship
                            flat_row[k] = dict(v._properties)
                        else:
                            flat_row[k] = v
                    flat.append(flat_row)
                st.dataframe(pd.DataFrame(flat), use_container_width=True)
                if len(rows) > 200:
                    st.caption(f"⚠ Affichage limité à 200 lignes sur {len(rows)}")
            else:
                st.info("Aucun résultat.")
        except neo4j_exc.CypherSyntaxError as e:
            st.error(f"Erreur Cypher : {e}")
        except Exception as e:
            st.error(f"Erreur : {e}")


if __name__ == "__main__":
    main()
