"""
pages/graph.py  —  Visualisation Neo4j interactive (neovis.js + Analyse IA Claude)

Interactions graphe :
  • Double-clic  → expand voisins directs (LIMIT 50)
  • Clic simple  → panneau propriétés flottant
  • Drag/zoom    → natif vis.js

Connexion :
  1. Auto depuis .env  (NEO4J_URI bolt://, NEO4J_USER, NEO4J_PASSWORD)
  2. Fallback formulaire si .env absent ou connexion échouée

Dépendances :
  pip install streamlit neo4j python-dotenv requests altair pandas

INTÉGRATION — NOTE POUR L'APP PRINCIPALE :
  Ce module s'utilise via render() et ne contient pas de st.set_page_config().
  Pour un affichage pleine largeur, l'app principale doit inclure layout="wide" :

    import streamlit as st
    from graph import render

    st.set_page_config(
        layout="wide",          # ← obligatoire pour la mise en page pleine largeur
        page_title="AI-FORENSICS",
        page_icon="🕸️",
    )
    render()

  Si l'app principale gère déjà st.set_page_config() (multi-pages),
  ajouter simplement layout="wide" à l'appel existant.
  IMPORTANT : st.set_page_config() doit être le PREMIER appel Streamlit de l'app.
"""

import json
import os
import re
import requests
from collections import Counter
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as _stc
from dotenv import load_dotenv

def _is_dark() -> bool:
    """
    Détecte le thème Streamlit.
    st.get_option("theme.base") retourne :
      - "dark"  si thème sombre configuré
      - "light" si thème clair configuré
      - None    si non configuré (défaut Streamlit = clair)
    """
    try:
        base = st.get_option("theme.base")
        return base == "dark"   # None ou "light" → clair
    except Exception:
        return False  # défaut : clair

def _tc() -> dict:
    dark = _is_dark()
    return {
        "bg":       "#1e2330" if dark else "#f0f2f8",
        "border":   "#2a2f3d" if dark else "#d0d4e0",
        "text":     "#e0e0e0" if dark else "#1a1a2e",
        "muted":    "#888"    if dark else "#555",
        "card_bg":  "#14181f" if dark else "#ffffff",
        "card_bd":  "#1f2535" if dark else "#d8dce8",
        "card_sub": "#777"    if dark else "#666",
    }
from neo4j import GraphDatabase, exceptions as neo4j_exc

# ══════════════════════════════════════════════════════════════════════════════
#  CHARGEMENT .env
# ══════════════════════════════════════════════════════════════════════════════

_root = Path(__file__).resolve().parent
for _candidate in [
    _root / ".env",
    _root.parent / ".env",
    _root.parent.parent / ".env",
]:
    if _candidate.exists():
        load_dotenv(_candidate, override=False)
        break

_ENV_URI  = os.getenv("NEO4J_URI",      "")
_ENV_USER = os.getenv("NEO4J_USER",     "neo4j")
_ENV_PWD  = os.getenv("NEO4J_PASSWORD", "")

# ══════════════════════════════════════════════════════════════════════════════
#  STYLES DES NOEUDS
# ══════════════════════════════════════════════════════════════════════════════

NODE_STYLES = {
    "Account":   {"color": "#4A90D9", "size": 34, "icon": "👤"},
    "Campaign":  {"color": "#FF6B6B", "size": 28, "icon": "🚨"},
    "Narrative": {"color": "#9370DB", "size": 22, "icon": "💬"},
    "Project":   {"color": "#888888", "size": 22, "icon": "📁"},
    "Post":      {"color": "#FFD700", "size": 18, "icon": "📝"},
    "Media":     {"color": "#CC8A30", "size": 18, "icon": "🖼"},
    "Hashtag":   {"color": "#50C878", "size": 14, "icon": "#"},
    "Deepfake":  {"color": "#CC3600", "size": 14, "icon": "⚠️"},
}
DEFAULT_STYLE = {"color": "#AAAAAA", "size": 16, "icon": "●"}

PRESET_QUERIES = {
    "— Choisir une requête —": "",
    "Tous les noeuds et relations (50)":
        "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50",
    "Comptes suspects liés à des deepfakes":
        "MATCH (a:Account)-[r:POSTED]->(m:Media)-[r2:IS_DEEPFAKE]->(d:Deepfake) RETURN a, r, m, r2, d LIMIT 40",
    "Personnes les plus citées":
        "MATCH (p:Person)<-[r:MENTIONS]-(m) RETURN p, r, m LIMIT 40",
    "Cluster de diffusion":
        "MATCH (src:Source)-[r:SPREAD]->(a:Account) RETURN src, r, a LIMIT 30",
    "Comptes coordonnés (même narratif)":
        "MATCH (a:Account)-[r1:POSTED]->(p:Post)-[r2:BELONGS_TO]->(n:Narrative) RETURN a, r1, p, r2, n LIMIT 40",
    "Médias réutilisés cross-comptes":
        "MATCH (a:Account)-[r1:POSTED]->(po:Post)-[r2:USES]->(m:Media) RETURN a, r1, po, r2, m LIMIT 40",
}

# ══════════════════════════════════════════════════════════════════════════════
#  CONNEXION NEO4J
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _get_driver(uri: str, user: str, pwd: str):
    return GraphDatabase.driver(uri, auth=(user, pwd))


def _test_driver(uri: str, user: str, pwd: str) -> bool:
    try:
        _get_driver(uri, user, pwd).verify_connectivity()
        return True
    except Exception:
        return False


def _run_bolt(cypher: str, params: dict = None) -> list[dict]:
    uri  = st.session_state.get("neo4j_uri",  _ENV_URI)
    user = st.session_state.get("neo4j_user", _ENV_USER)
    pwd  = st.session_state.get("neo4j_pwd",  _ENV_PWD)
    with _get_driver(uri, user, pwd).session() as session:
        return [dict(r) for r in session.run(cypher, **(params or {}))]


def _neo4j_http(cypher: str):
    uri  = st.session_state.get("neo4j_uri",  _ENV_URI)
    user = st.session_state.get("neo4j_user", _ENV_USER)
    pwd  = st.session_state.get("neo4j_pwd",  _ENV_PWD)
    base = uri.replace("bolt://", "http://").replace(":7687", ":7474")
    url  = f"{base.rstrip('/')}/db/neo4j/tx/commit"
    try:
        resp = requests.post(
            url,
            json={"statements": [{"statement": cypher, "resultDataContents": ["graph", "row"]}]},
            auth=(user, pwd),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("errors"):
            return [], data["errors"][0].get("message", "Erreur inconnue")
        return data.get("results", [{}])[0].get("data", []), None
    except Exception as e:
        return [], str(e)

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS DONNEES
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=30)
def _get_projects() -> list[str]:
    try:
        return [r["name"] for r in _run_bolt(
            "MATCH (p:Project) RETURN p.name AS name ORDER BY name") if r["name"]]
    except Exception:
        return []


@st.cache_data(ttl=30)
def _get_campaigns(project: str | None) -> list[str]:
    try:
        if project:
            try:
                rows = _run_bolt(
                    "MATCH (c:Campaign)-[:BELONGS_TO]->(p:Project {name:$proj}) "
                    "RETURN c.name AS name ORDER BY name", {"proj": project})
                if not rows:  # relation absente → toutes les campagnes
                    rows = _run_bolt("MATCH (c:Campaign) RETURN c.name AS name ORDER BY name")
            except Exception:
                rows = _run_bolt("MATCH (c:Campaign) RETURN c.name AS name ORDER BY name")
        else:
            rows = _run_bolt("MATCH (c:Campaign) RETURN c.name AS name ORDER BY name")
        return [r["name"] for r in rows if r["name"]]
    except Exception:
        return []


@st.cache_data(ttl=15)
def _get_db_stats() -> dict:
    try:
        labels = _run_bolt("CALL db.labels() YIELD label RETURN label")
        counts = {}
        for row in labels:
            lbl = row["label"]
            r = _run_bolt(f"MATCH (n:`{lbl}`) RETURN count(n) AS c")
            counts[lbl] = r[0]["c"] if r else 0
        r = _run_bolt("MATCH ()-[r]->() RETURN count(r) AS c")
        counts["_relations"] = r[0]["c"] if r else 0
        return counts
    except Exception:
        return {}


def _build_filter_cypher(project, campaign, node_types, limit) -> str:
    # Filtre type sur n ET m — sinon les voisins hors-sélection s'affichent quand même
    if node_types:
        tj = json.dumps(node_types)
        tw = f" WHERE any(l IN labels(n) WHERE l IN {tj}) AND any(l IN labels(m) WHERE l IN {tj})"
    else:
        tw = ""

    if campaign:
        return (
            f"MATCH (c:Campaign {{name: '{campaign}'}})-[r*1..2]-(n) "
            f"WITH n LIMIT {limit} "
            f"MATCH (n)-[rel]-(m){tw} RETURN n, rel, m LIMIT {limit}"
        )
    elif project:
        return (
            f"MATCH (p:Project {{name: '{project}'}})-[r*1..3]-(n) "
            f"WITH n LIMIT {limit} "
            f"MATCH (n)-[rel]-(m){tw} RETURN n, rel, m LIMIT {limit}"
        )
    else:
        return f"MATCH (n)-[rel]-(m){tw} RETURN n, rel, m LIMIT {limit}"


def _ensure_relations(cypher: str) -> str:
    """
    Si la requête ne contient pas de variable de relation ([r...]),
    on tente de la réécrire pour inclure les arêtes.
    Sinon neovis.js affiche uniquement les noeuds sans arêtes.
    """
    # Déjà une relation dans le MATCH ou un PATH → on laisse
    if re.search(r'-\[', cypher) or re.search(r'\bPATH\b', cypher, re.IGNORECASE):
        return cypher
    # Au moins 2 variables dans RETURN → probablement OK
    ret = re.search(r'\bRETURN\b(.+?)(?:\bLIMIT\b|$)', cypher, re.IGNORECASE | re.DOTALL)
    if ret and len([v.strip() for v in ret.group(1).split(',')]) >= 2:
        return cypher
    # Cas : MATCH (n:X) RETURN n LIMIT k → on wrappe
    limit_m = re.search(r'\bLIMIT\s+(\d+)', cypher, re.IGNORECASE)
    limit   = limit_m.group(1) if limit_m else "50"
    var_m   = re.search(r'MATCH\s*\((\w+)', cypher, re.IGNORECASE)
    var     = var_m.group(1) if var_m else "n"
    match_p = re.sub(r'\bRETURN\b.*', '', cypher, flags=re.IGNORECASE | re.DOTALL).strip()
    return (
        f"CALL {{ {match_p} RETURN {var} LIMIT {limit} }} "
        f"MATCH ({var})-[rel]->(m) RETURN {var}, rel, m LIMIT {limit}"
    )


def _compute_graph_stats(rows_http: list) -> dict:
    seen_nodes, seen_edges = set(), set()
    label_counts, out_degree, in_degree = Counter(), Counter(), Counter()
    node_labels = {}
    for row in rows_http:
        g = row.get("graph", {})
        for node in g.get("nodes", []):
            nid = node["id"]
            if nid in seen_nodes:
                continue
            seen_nodes.add(nid)
            labels = node.get("labels", [])
            label  = labels[0] if labels else "Unknown"
            label_counts[label] += 1
            node_labels[nid] = label
            out_degree.setdefault(nid, 0)
            in_degree.setdefault(nid, 0)
        for rel in g.get("relationships", []):
            rid = rel["id"]
            if rid in seen_edges:
                continue
            s, e = rel["startNode"], rel["endNode"]
            if s not in seen_nodes or e not in seen_nodes:
                continue
            seen_edges.add(rid)
            out_degree[s] = out_degree.get(s, 0) + 1
            in_degree[e]  = in_degree.get(e, 0) + 1
    centrality = {n: out_degree.get(n, 0) + in_degree.get(n, 0) for n in seen_nodes}
    return {
        "nb_nodes":     len(seen_nodes),
        "nb_edges":     len(seen_edges),
        "label_counts": dict(label_counts),
        "top_nodes":    sorted(centrality.items(), key=lambda x: -x[1])[:5],
        "node_labels":  node_labels,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSE IA
# ══════════════════════════════════════════════════════════════════════════════

def _build_prompt(stats: dict, cypher: str) -> str:
    density = round(stats["nb_edges"] / max(stats["nb_nodes"], 1), 2)
    top_str = "\n".join(
        f"  - ID {nid} ({stats['node_labels'].get(nid,'?')}) → centralité {score}"
        for nid, score in stats["top_nodes"]
    ) or "  Aucun"
    lc_str = ", ".join(f"{l}: {c}" for l, c in stats["label_counts"].items()) or "Aucun"
    return f"""Tu es un expert OSINT et analyste de graphes de désinformation.
Analyse ce sous-graphe Neo4j extrait d'un projet de détection de deepfakes et campagnes d'influence.

Requête : {cypher}

Statistiques :
- Noeuds : {stats['nb_nodes']} | Relations : {stats['nb_edges']} | Densité : {density} rel/noeud
- Types : {lc_str}

Top 5 par centralité :
{top_str}

Rédige une analyse structurée en français :
1. **Résumé** (2-3 phrases)
2. **Patterns détectés** (hubs, clusters, chaînes de diffusion)
3. **Alertes OSINT** (anomalies, coordination, deepfakes, bots potentiels)
4. **Recommandations** (requêtes Cypher complémentaires à creuser)

Sois concis, factuel, orienté investigation."""


def _claude_analysis(stats: dict, cypher: str) -> str:
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": _build_prompt(stats, cypher)}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]
    except Exception as e:
        return f"⚠️ Analyse IA indisponible : {e}"

# ══════════════════════════════════════════════════════════════════════════════
#  HTML NEOVIS.JS
# ══════════════════════════════════════════════════════════════════════════════

def _build_neovis_html(bolt_uri: str, user: str, password: str,
                       initial_cypher: str, node_styles: dict,
                       height: int = 560,
                       visible_types: list = None) -> str:
    import json as _j
    password_js       = _j.dumps(password)
    initial_cypher_js = _j.dumps(initial_cypher)
    labels_config     = {
        lbl: {
            "caption": "name", "size": s["size"], "community": lbl,
            "title_properties": ["name","username","platform","score","prediction"],
        }
        for lbl, s in node_styles.items()
    }
    color_map          = {lbl: s["color"] for lbl, s in node_styles.items()}
    size_map           = {lbl: s["size"]  for lbl, s in node_styles.items()}
    labels_config_json = json.dumps(labels_config)
    color_map_json     = json.dumps(color_map)
    size_map_json      = json.dumps(size_map)
    # Labels à masquer si filtre actif
    import json as _jj
    hidden_labels = [l for l in node_styles if visible_types and l not in visible_types]
    hidden_labels_json = _jj.dumps(hidden_labels)

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<style>
  :root {{
    --bg:#141720; --bg-ov:rgba(18,22,34,.97);
    --bd:#2a2f3d; --bd-row:#1f2535;
    --tx:#e0e0e0; --tx-m:#888; --tx-k:#7eb6ff; --hint:#444;
  }}
  body.light {{
    --bg:#eef0f5; --bg-ov:rgba(238,240,245,.98);
    --bd:#c8ccd8; --bd-row:#dde0ea;
    --tx:#1a1a2e; --tx-m:#555; --tx-k:#1a6fc4; --hint:#aaa;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);font-family:Inter,sans-serif;overflow:hidden;transition:background .2s}}
  #g{{width:100%;height:{height}px;background:var(--bg)}}
  #ov{{position:absolute;top:10px;right:10px;width:270px;background:var(--bg-ov);
       border:1px solid var(--bd);border-radius:8px;padding:14px;display:none;
       font-size:12px;color:var(--tx);z-index:100;max-height:{height-40}px;
       overflow-y:auto;box-shadow:0 4px 20px rgba(0,0,0,.2);transition:background .2s}}
  #ov h3{{margin-bottom:10px;font-size:13px;color:var(--tx-m);
          border-bottom:1px solid var(--bd);padding-bottom:6px}}
  .pr{{display:flex;gap:8px;margin-bottom:5px;border-bottom:1px solid var(--bd-row);padding-bottom:4px}}
  .pk{{color:var(--tx-k);min-width:90px;word-break:break-all}}
  .pv{{color:var(--tx);word-break:break-all;flex:1}}
  .bg{{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;margin-bottom:8px;font-weight:600}}
  #hint{{position:absolute;bottom:10px;left:10px;color:var(--hint);font-size:11px;pointer-events:none}}
  #ld{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:var(--tx-m);font-size:14px}}
</style>
</head>
<body>
<div id="g"></div>
<div id="ov"><h3 id="ov-t">Noeud</h3><div id="ov-b"></div></div>
<div id="hint">🖱 Double-clic : expand · Clic : détails · Scroll : zoom</div>
<div id="ld">⏳ Chargement…</div>
<script src="https://unpkg.com/neovis.js@2.0.2/dist/neovis.js"></script>
<script>
const CM={color_map_json},SM={size_map_json},DC="#888888",DS=16;
function nc(lb){{if(!lb)return DC;for(const l of lb)if(CM[l])return CM[l];return DC;}}

const viz=new NeoVis.default({{
  containerId:"g",
  neo4j:{{serverUrl:"{bolt_uri}",serverUser:"{user}",serverPassword:{password_js}}},
  labels:{labels_config_json},
  relationships:{{
    "POSTED":{{caption:false}},"USES_HASHTAG":{{caption:false}},"REPOSTED":{{caption:true}},
    "BELONGS_TO":{{caption:false}},"AMPLIFIED_BY":{{caption:false}},"FORWARDED":{{caption:false}},
    "PART_OF":{{caption:false}},"IS_DEEPFAKE":{{caption:false}},"SHARED":{{caption:false}},
    "MENTIONS":{{caption:false}},"USES":{{caption:false}},"SPREAD":{{caption:false}},
  }},
  initialCypher:{initial_cypher_js},
  arrows:true,hierarchical:false,
  visConfig:{{
    configure:{{enabled:false}},
    nodes:{{
      shape:"dot",
      scaling:{{min:10,max:40,label:{{enabled:false}}}},
    }},
    physics:{{enabled:true,stabilization:{{iterations:150}},
      barnesHut:{{gravitationalConstant:-8000,centralGravity:.3,
        springLength:120,springConstant:.04,damping:.09}}}},
  }},
}});
window.viz=viz;
const EX=new Set();

function applyTheme(dark){{
  const bg=dark?"#141720":"#eef0f5",fc=dark?"#fff":"#1a1a2e",
        ec=dark?"#555":"#aaa",eh=dark?"#FFD700":"#1a6fc4";
  document.body.classList.toggle("light",!dark);
  const cv=document.querySelector("#g canvas");
  if(cv){{
    if(!cv._p){{
      const ctx=cv.getContext("2d"),orig=ctx.clearRect.bind(ctx);
      ctx.clearRect=(x,y,w,h)=>{{orig(x,y,w,h);ctx.fillStyle=cv._bg||"#141720";ctx.fillRect(x,y,w,h);}};
      cv._p=true;
    }}
    cv._bg=bg;
    if(window._net)window._net.redraw();
  }}
  if(window._net)window._net.setOptions({{
    nodes:{{font:{{color:fc}}}},
    edges:{{font:{{color:ec}},color:{{color:ec,highlight:eh}}}},
  }});
}}
applyTheme(window.matchMedia("(prefers-color-scheme:dark)").matches);
window.matchMedia("(prefers-color-scheme:dark)").addEventListener("change",e=>applyTheme(e.matches));
window.addEventListener("message",e=>{{if(e.data?.type==="streamlit:theme")applyTheme(e.data?.theme?.base!=="light");}});

viz.registerOnEvent("completed",()=>{{
  document.getElementById("ld").style.display="none";
  setTimeout(()=>{{
    const net=viz.network;window._net=net;if(!net)return;
    const ds=viz.nodes;
    if(ds){{
      const u=[];
      const HIDDEN={hidden_labels_json};
      ds.get().forEach(n=>{{
        const lb=n.raw?.labels||n.neo4j_labels||[],c=nc(lb);
        const sz=lb.length?Math.max(...lb.map(l=>SM[l]||DS)):DS;
        const hide=HIDDEN.length>0&&lb.length>0&&lb.every(l=>HIDDEN.includes(l));
        u.push({{id:n.id,
          size:sz,
          hidden:hide,
          color:{{background:c,border:c,highlight:{{background:c,border:"#FFD700"}}}},
          font:{{color:"#fff",size:Math.max(10,Math.round(sz*0.55))}}}});
      }});
      if(u.length)ds.update(u);
    }}
    net.on("click",p=>{{
      if(p.nodes?.length){{const n=viz.nodes?.get(p.nodes[0]);if(n)showP(n);return;}}
      if(p.edges?.length){{const e=viz.edges?.get(p.edges[0]);if(e)showR(e);return;}}
      document.getElementById("ov").style.display="none";
    }});
    net.on("doubleClick",p=>{{
      if(!p.nodes?.length)return;
      const id=p.nodes[0];if(EX.has(id))return;EX.add(id);
      const q=`MATCH (n)-[r]-(m) WHERE id(n)=${{id}} RETURN n,r,m LIMIT 50`;
      if(typeof viz.updateWithCypher==="function")viz.updateWithCypher(q);else viz.renderWithCypher(q);
      if(viz.nodes)viz.nodes.update({{id,borderWidth:4,color:{{border:"#FFD700"}}}});
    }});
  }},100);
}});

function showP(n){{
  const lb=n.raw?.labels||n.neo4j_labels||[],
        pr=n.raw?.properties||n.raw_node?.properties||{{}},
        c=nc(lb),
        sk=new Set(["id","label","x","y","fixed","color","font","size","shape","borderWidth","neo4j_labels","raw_node","raw"]);
  document.getElementById("ov-t").innerHTML=lb.length
    ?lb.map(l=>`<span class="bg" style="background:${{c}}22;color:${{c}};border:1px solid ${{c}}">${{l}}</span>`).join(" ")
    :`<span class="bg" style="background:#33333399;color:#aaa">Noeud</span>`;
  const all=Object.assign({{}},pr);
  for(const[k,v]of Object.entries(n))if(!sk.has(k)&&!(k in all))all[k]=v;
  let h="";
  for(const[k,v]of Object.entries(all)){{
    if(v==null||v==="")continue;
    let d=typeof v==="object"?JSON.stringify(v):String(v);
    if(d.length>200)d=d.slice(0,200)+"…";
    h+=`<div class="pr"><span class="pk">${{k}}</span><span class="pv">${{d}}</span></div>`;
  }}
  document.getElementById("ov-b").innerHTML=h||"<em style='color:#666'>Aucune propriété</em>";
  document.getElementById("ov").style.display="block";
}}

function showR(r){{
  const t=r.raw_edge?.type||r.raw?.type||r.label||"relation",
        pr=r.raw_edge?.properties||r.raw?.properties||{{}};
  document.getElementById("ov-t").innerHTML=
    `<span class="bg" style="background:#33330099;color:#FFD700;border:1px solid #FFD700">${{t}}</span>`;
  let h="";
  for(const[k,v]of Object.entries(pr)){{
    if(v==null||v==="")continue;
    h+=`<div class="pr"><span class="pk">${{k}}</span><span class="pv">${{String(v)}}</span></div>`;
  }}
  document.getElementById("ov-b").innerHTML=h||"<em style='color:#666'>Pas de propriétés</em>";
  document.getElementById("ov").style.display="block";
}}

viz.render();
</script></body></html>"""

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

def render():

    # Injection thème Streamlit → iframe
    st.markdown("""<script>
    (function(){
      function s(){
        const b=document.documentElement.getAttribute("data-theme")||
                (window.matchMedia("(prefers-color-scheme:dark)").matches?"dark":"light");
        const f=document.querySelector("iframe");
        if(f?.contentWindow)f.contentWindow.postMessage({type:"streamlit:theme",theme:{base:b}},"*");
      }
      setTimeout(s,800);setTimeout(s,2500);
      new MutationObserver(s).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
    })();
    </script>""", unsafe_allow_html=True)

    st.title("🕸️ Graphe Neo4j")
    st.caption("Visualisation interactive · Requêtes Cypher · Analyse IA OSINT")

    # ── Connexion ─────────────────────────────────────────────────────────────
    if "neo4j_connected" not in st.session_state and _ENV_URI:
        if _test_driver(_ENV_URI, _ENV_USER, _ENV_PWD):
            st.session_state.update({
                "neo4j_connected": True, "neo4j_uri": _ENV_URI,
                "neo4j_user": _ENV_USER, "neo4j_pwd": _ENV_PWD, "neo4j_auto": True,
            })

    auto_ok = st.session_state.get("neo4j_auto", False)
    with st.expander("⚙️ Connexion Neo4j",
                     expanded="neo4j_connected" not in st.session_state):
        if auto_ok:
            st.success(f"✅ Connexion automatique via .env → `{_ENV_URI}`")
            if st.button("🔄 Changer de connexion", key="btn_chg"):
                for k in ("neo4j_connected","neo4j_auto","neo4j_uri","neo4j_user","neo4j_pwd"):
                    st.session_state.pop(k, None)
                st.rerun()
        else:
            c1, c2, c3 = st.columns([3, 2, 2])
            uri  = c1.text_input("URL Bolt", value=_ENV_URI or "bolt://localhost:7687", key="f_uri")
            user = c2.text_input("Utilisateur", value=_ENV_USER or "neo4j", key="f_user")
            pwd  = c3.text_input("Mot de passe", type="password", key="f_pwd")
            if st.button("🔌 Connecter", use_container_width=True, key="btn_conn"):
                with st.spinner("Test…"):
                    ok = _test_driver(uri, user, pwd)
                if ok:
                    st.session_state.update({
                        "neo4j_connected": True, "neo4j_uri": uri,
                        "neo4j_user": user, "neo4j_pwd": pwd, "neo4j_auto": False,
                    })
                    st.rerun()
                else:
                    st.error("Connexion échouée — vérifiez les paramètres.")

    if "neo4j_connected" not in st.session_state:
        st.info("🔌 Connectez-vous à Neo4j pour visualiser les graphes.")
        return

    uri  = st.session_state["neo4j_uri"]
    user = st.session_state["neo4j_user"]
    pwd  = st.session_state["neo4j_pwd"]

    # ── Requête Cypher + Filtres ──────────────────────────────────────────────
    st.markdown("---")
    with st.expander("🔎 Requête Cypher", expanded=True):

        # ── Requêtes prédéfinies + zone libre ─────────────────────────────────
        def _on_preset():
            val = PRESET_QUERIES.get(st.session_state["psel"], "")
            if val:
                st.session_state["ctxt"] = val
                st.session_state.pop("active_mode", None)

        st.selectbox("Requêtes prédéfinies", list(PRESET_QUERIES.keys()),
                     key="psel", on_change=_on_preset)

        if "ctxt" not in st.session_state:
            st.session_state["ctxt"] = ""

        cypher_typed = st.text_area(
            "Requête Cypher libre :",
            height=90,
            placeholder="MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 30",
            key="ctxt",
            help="💡 Incluez une variable de relation [r] pour afficher les arêtes.",
        )

        col_run, col_clr = st.columns([6, 1])
        run_btn = col_run.button("▶️ Exécuter la requête", use_container_width=True,
                                 type="primary", key="btn_run")
        clr_btn = col_clr.button("🗑️", use_container_width=True, key="btn_clr",
                                 help="Réinitialiser")
        if clr_btn:
            for k in ("active_cypher","active_mode","graph_stats","ia_result","ctxt","psel"):
                st.session_state.pop(k, None)
            st.rerun()

        st.markdown("---")
        # ── Filtres rapides — appliqués en direct ─────────────────────────────
        st.markdown("**Filtres rapides** — appliqués immédiatement")
        fc1, fc2 = st.columns(2)
        projects    = _get_projects()
        project_sel = fc1.selectbox("Projet", ["— Tous —"] + projects, key="f_proj")
        project     = None if project_sel.startswith("—") else project_sel

        campaigns    = _get_campaigns(project)
        campaign_sel = fc2.selectbox("Campagne", ["— Toutes —"] + campaigns, key="f_camp")
        campaign     = None if campaign_sel.startswith("—") else campaign_sel

        ft1, ft2 = st.columns([3, 1])
        node_types = ft1.multiselect(
            "Types de noeuds", list(NODE_STYLES.keys()),
            default=["Account", "Post", "Campaign", "Hashtag"],
            key="f_types", help="Vide = tous",
        )
        limit = ft2.slider("Limite", 20, 500, 100, step=20, key="f_limit")

    # ── Cypher actif ──────────────────────────────────────────────────────────
    filter_cypher = _build_filter_cypher(project, campaign, node_types, limit)
    rewrote = False

    if run_btn and cypher_typed.strip():
        raw    = cypher_typed.strip()
        cypher = _ensure_relations(raw)
        rewrote = (cypher != raw)
        st.session_state["active_cypher"] = cypher
        st.session_state["active_mode"]   = "manual"
        st.session_state.pop("graph_stats", None)
        st.session_state.pop("ia_result",   None)
    elif st.session_state.get("active_mode") != "manual":
        # Filtres en direct : le graphe suit immédiatement
        st.session_state["active_cypher"] = filter_cypher

    active_cypher = st.session_state.get("active_cypher", filter_cypher)

    if rewrote:
        st.info("💡 Requête adaptée pour inclure les relations (arêtes visibles).")

    with st.expander("🔍 Cypher actif", expanded=False):
        st.code(active_cypher, language="cypher")

    # ── Visualisation ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Visualisation")

    _stc.html(
        _build_neovis_html(uri, user, pwd, active_cypher, NODE_STYLES, height=560,
                           visible_types=node_types if node_types else None),
        height=580, scrolling=False,
    )

    # Légende — pas de fond inline, hérite du thème Streamlit
    leg_items = "".join(
        f"<span style='display:inline-flex;align-items:center;gap:5px;font-size:12px'>"
        f"<span style='width:11px;height:11px;border-radius:50%;background:{s['color']}"
        f";flex-shrink:0;display:inline-block'></span>{s['icon']} {lbl}</span>"
        for lbl, s in NODE_STYLES.items()
    )
    st.markdown(
        f"<div style='display:flex;gap:14px;flex-wrap:wrap;padding:6px 0;margin-top:4px;"
        f"border-top:1px solid rgba(128,128,128,0.2)'>{leg_items}</div>",
        unsafe_allow_html=True,
    )

    # Cards compteurs DB sous la légende — st.metric natif (suit le thème automatiquement)
    db_stats = _get_db_stats()
    if db_stats:
        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
        visible = [(lbl, s, db_stats[lbl]) for lbl, s in NODE_STYLES.items() if db_stats.get(lbl, 0) > 0]
        if visible:
            card_cols = st.columns(min(len(visible), 4))
            for ci, (lbl, style, count) in enumerate(visible):
                card_cols[ci % len(card_cols)].markdown(
                    f"<div style='text-align:center;padding:6px 0'>"
                    f"<div style='font-size:22px;font-weight:700;color:{style['color']};line-height:1.2'>{count:,}</div>"
                    f"<div style='font-size:11px;opacity:0.7'>{style['icon']} {lbl}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        rels = db_stats.get("_relations", 0)
        if rels:
            st.caption(f"↔ {rels:,} relations totales dans la base")

    # ── Calcul stats après exécution ──────────────────────────────────────────
    if run_btn:
        rows_http, err = _neo4j_http(active_cypher)
        if err:
            st.warning(f"Stats OSINT indisponibles (HTTP REST) : {err}")
        elif rows_http:
            s = _compute_graph_stats(rows_http)
            s["_cypher"] = active_cypher
            st.session_state["graph_stats"] = s

    stats = st.session_state.get("graph_stats")

    # ── Analyse OSINT ─────────────────────────────────────────────────────────
    t = _tc()
    st.markdown("---")
    st.subheader("🧠 Analyse OSINT")

    if not stats:
        # Compteurs globaux DB en attendant
        db = _get_db_stats()
        st.info("Exécutez une requête pour voir les statistiques du sous-graphe.")
    else:
        st.markdown("##### Sous-graphe affiché")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Noeuds",           stats["nb_nodes"])
        m2.metric("Relations",         stats["nb_edges"])
        m3.metric("Densité rel/noeud", f"{stats['nb_edges'] / max(stats['nb_nodes'], 1):.2f}")
        m4.metric("Types distincts",   len(stats["label_counts"]))

        if stats["label_counts"]:
            cols = st.columns(min(len(stats["label_counts"]), 4))
            for i, (lbl, count) in enumerate(stats["label_counts"].items()):
                s = NODE_STYLES.get(lbl, DEFAULT_STYLE)
                cols[i % 4].markdown(
                    f"<div style='text-align:center;padding:6px 0'>"
                    f"<div style='font-size:22px;font-weight:700;color:{s['color']};line-height:1.2'>{count:,}</div>"
                    f"<div style='font-size:11px;opacity:0.7'>{s['icon']} {lbl}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        if stats["top_nodes"]:
            with st.container(border=True):
                st.markdown("#### 🔗 Noeuds les plus connectés")
                for rank, (nid, score) in enumerate(stats["top_nodes"], 1):
                    label = stats["node_labels"].get(nid, "?")
                    color = NODE_STYLES.get(label, DEFAULT_STYLE)["color"]
                    st.markdown(
                        f"**#{rank}** &nbsp;"
                        f"<span style='background:{color};color:#fff;padding:1px 8px;"
                        f"border-radius:10px;font-size:12px'>{label}</span>"
                        f" &nbsp;ID `{nid}` → **{score}** connexions",
                        unsafe_allow_html=True,
                    )

        if stats["label_counts"]:
            with st.expander("📈 Distribution des types"):
                import altair as alt
                import pandas as pd
                df = pd.DataFrame([
                    {
                        "Type": f"{NODE_STYLES.get(lbl, DEFAULT_STYLE)['icon']} {lbl}",
                        "Noeuds": count,
                        "color": NODE_STYLES.get(lbl, DEFAULT_STYLE)["color"],
                    }
                    for lbl, count in stats["label_counts"].items()
                ])
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X("Type:N", sort="-y", axis=alt.Axis(labelAngle=-30)),
                    y=alt.Y("Noeuds:Q"),
                    color=alt.Color("Type:N", scale=alt.Scale(
                        domain=df["Type"].tolist(),
                        range=df["color"].tolist(),
                    ), legend=None),
                    tooltip=["Type", "Noeuds"],
                ).properties(height=260)
                st.altair_chart(chart, use_container_width=True)

    # ── Analyse IA ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🤖 Analyse IA (Claude)")

    if not stats:
        st.info("Exécutez une requête pour activer l'analyse IA du sous-graphe.")
    elif "ia_result" not in st.session_state:
        if st.button("✨ Lancer l'analyse IA", use_container_width=True,
                     type="primary", key="btn_ia"):
            with st.spinner("Analyse en cours via Claude…"):
                st.session_state["ia_result"] = _claude_analysis(
                    stats, stats.get("_cypher", active_cypher)
                )
            st.rerun()
    else:
        with st.container(border=True):
            st.markdown(st.session_state["ia_result"])
        if st.button("🔄 Relancer l'analyse IA", use_container_width=True, key="btn_ia_re"):
            with st.spinner("Nouvelle analyse…"):
                st.session_state["ia_result"] = _claude_analysis(
                    stats, stats.get("_cypher", active_cypher)
                )
            st.rerun()

    # ── Lien Neo4j Browser ────────────────────────────────────────────────────
    http_base   = uri.replace("bolt://", "http://").replace(":7687", ":7474")
    browser_url = http_base.rstrip("/") + "/browser/"
    st.markdown("---")
    st.markdown(
        f"""<div style='border-radius:8px;padding:14px 18px;
            display:flex;align-items:center;gap:12px;
            border:1px solid rgba(128,128,128,0.2)'>
          <span style='font-size:22px'>🗄️</span>
          <div>
            <div style='font-weight:600;margin-bottom:2px'>Neo4j Browser</div>
            <div style='font-size:13px;opacity:0.6'>Explorez et interrogez directement votre base</div>
          </div>
          <a href='{browser_url}' target='_blank'
             style='margin-left:auto;background:#018BFF;color:#fff;padding:8px 18px;
                    border-radius:6px;text-decoration:none;font-size:14px;font-weight:500'>
            Ouvrir ↗</a>
        </div>""",
        unsafe_allow_html=True,
    )
