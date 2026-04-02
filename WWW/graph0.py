"""
pages/graph.py  —  Visualisation Neo4j interactive (HTTP REST + IA Claude)
Dépendances : pip install pyvis requests
"""

import streamlit as st
import requests, tempfile, os, textwrap, threading, time
from pyvis.network import Network
from collections import Counter
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver

# ── Palettes par label ────────────────────────────────────────────────────────
LABEL_COLORS = {
    "Person":   "#1565C0",  # bleu foncé
    "Account":  "#2E7D32",  # vert foncé
    "Media":    "#E65100",  # orange foncé
    "Source":   "#AD1457",  # rose foncé
    "Deepfake": "#B71C1C",  # rouge foncé
    "Unknown":  "#546E7A",  # gris bleu
}
DEFAULT_COLOR = "#1976D2"

# ── Limite de nœuds affichés (anti-crash Chromium) ───────────────────────────
MAX_DISPLAY_NODES = 200

# ── Requêtes prédéfinies ──────────────────────────────────────────────────────
PRESET_QUERIES = {
    "— Choisir une requête —": "",
    "Tous les nœuds (limite 50)":
        "MATCH (n) RETURN n LIMIT 50",
    "Toutes les relations (limite 50)":
        "MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 50",
    "Comptes suspects liés à des deepfakes":
        "MATCH (a:Account)-[:POSTED|SHARED]->(m:Media)-[:IS_DEEPFAKE]->(d:Deepfake) RETURN a, m, d LIMIT 40",
    "Personnes les plus citées":
        "MATCH (p:Person)<-[:MENTIONS]-(m) RETURN p, m LIMIT 40",
    "Cluster de diffusion":
        "MATCH path=(src:Source)-[:SPREAD*1..3]->(a:Account) RETURN path LIMIT 30",
}

# ══════════════════════════════════════════════════════════════════════════════
#  MINI-SERVEUR HTTP LOCAL (isolation iframe)
# ══════════════════════════════════════════════════════════════════════════════

_server_instance: HTTPServer | None = None
_server_dir: str = ""
_server_port: int = 0


class _SilentHandler(SimpleHTTPRequestHandler):
    """Handler silencieux (pas de logs dans la console Streamlit)."""
    def log_message(self, format, *args):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=_server_dir, **kwargs)


def _find_free_port(start: int = 8700, end: int = 8800) -> int:
    import socket
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("Aucun port libre trouvé entre 8700 et 8800.")


def ensure_graph_server(serve_dir: str) -> int:
    """
    Lance (ou réutilise) un HTTPServer local servant `serve_dir`.
    Retourne le port utilisé.
    Stocke l'instance dans st.session_state pour survie entre reruns.
    """
    global _server_instance, _server_dir, _server_port

    # Réutiliser le serveur déjà lancé dans cette session Streamlit
    if "graph_server_port" in st.session_state:
        _server_dir = serve_dir   # mettre à jour le répertoire servi
        return st.session_state["graph_server_port"]

    port = _find_free_port()
    _server_dir = serve_dir
    _server_port = port

    server = HTTPServer(("127.0.0.1", port), _SilentHandler)
    _server_instance = server

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    # Petite attente pour s'assurer que le serveur est prêt
    time.sleep(0.2)

    st.session_state["graph_server_port"] = port
    return port

# ══════════════════════════════════════════════════════════════════════════════
#  COUCHE HTTP NEO4J
# ══════════════════════════════════════════════════════════════════════════════

def neo4j_http_query(base_url, user, pwd, cypher):
    url = f"{base_url.rstrip('/')}/db/neo4j/tx/commit"
    payload = {"statements": [{"statement": cypher, "resultDataContents": ["graph", "row"]}]}
    try:
        resp = requests.post(
            url, json=payload, auth=(user, pwd),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("errors"):
            return [], data["errors"][0].get("message", "Erreur inconnue")
        return data.get("results", [{}])[0].get("data", []), None
    except requests.exceptions.ConnectionError:
        return [], f"Impossible de joindre Neo4j sur {base_url}."
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 401:
            return [], "Authentification refusée (401)."
        return [], f"Erreur HTTP {resp.status_code} : {e}"
    except Exception as e:
        return [], str(e)


def test_neo4j_connection(base_url, user, pwd):
    _, err = neo4j_http_query(base_url, user, pwd, "RETURN 1")
    return err

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCTION DU GRAPHE PYVIS  (thème clair)
# ══════════════════════════════════════════════════════════════════════════════

def build_pyvis(rows):
    net = Network(
        height="540px", width="100%",
        bgcolor="#FFFFFF",       # ← fond blanc
        font_color="#212121",    # ← texte sombre
        directed=True,
    )
    net.set_options("""{
      "nodes": {
        "borderWidth": 2,
        "shadow": {"enabled": true, "color": "rgba(0,0,0,0.12)", "size": 6},
        "font": {"size": 13, "face": "monospace", "color": "#212121"}
      },
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.6}},
        "color": {"color": "#90A4AE", "highlight": "#00796B"},
        "font": {"size": 10, "color": "#546E7A"},
        "smooth": {"type": "continuous"}
      },
      "physics": {
        "enabled": true,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -40,
          "springLength": 110,
          "damping": 0.9
        },
        "stabilization": {
          "enabled": true,
          "iterations": 120,
          "fit": true
        },
        "maxVelocity": 30,
        "minVelocity": 1
      },
      "interaction": {
        "hover": true, "tooltipDelay": 200,
        "navigationButtons": true, "keyboard": false,
        "multiselect": false,
        "hideEdgesOnDrag": true
      },
      "background": {
        "color": "#FFFFFF"
      }
    }""")

    seen_nodes, seen_edges = set(), set()
    label_counts, out_degree, in_degree = Counter(), Counter(), Counter()
    node_labels = {}
    truncated = False

    for row in rows:
        graph = row.get("graph", {})

        for node in graph.get("nodes", []):
            if len(seen_nodes) >= MAX_DISPLAY_NODES:
                truncated = True
                break
            nid = node["id"]
            if nid in seen_nodes:
                continue
            seen_nodes.add(nid)
            labels = node.get("labels", [])
            label  = labels[0] if labels else "Unknown"
            color  = LABEL_COLORS.get(label, DEFAULT_COLOR)
            props  = node.get("properties", {})
            display = textwrap.shorten(
                str(props.get("name") or props.get("username") or props.get("url") or f"#{nid}"),
                width=24, placeholder="…"
            )
            tooltip = f"[{label}] id={nid}"
            if props:
                first_key = next(iter(props))
                tooltip += f" | {first_key}: {str(props[first_key])[:40]}"
            net.add_node(nid, label=display, title=tooltip, color=color, size=20, group=label)
            label_counts[label] += 1
            node_labels[nid] = label
            out_degree.setdefault(nid, 0)
            in_degree.setdefault(nid, 0)

        for rel in graph.get("relationships", []):
            rid = rel["id"]
            if rid in seen_edges:
                continue
            sid, eid_n = rel["startNode"], rel["endNode"]
            if sid not in seen_nodes or eid_n not in seen_nodes:
                continue
            seen_edges.add(rid)
            rtype = rel.get("type", "")
            net.add_edge(sid, eid_n, label=rtype, title=rtype)
            out_degree[sid] = out_degree.get(sid, 0) + 1
            in_degree[eid_n] = in_degree.get(eid_n, 0) + 1

    centrality = {nid: out_degree.get(nid, 0) + in_degree.get(nid, 0) for nid in seen_nodes}
    top_nodes  = sorted(centrality.items(), key=lambda x: -x[1])[:5]

    return net, {
        "nb_nodes": len(seen_nodes), "nb_edges": len(seen_edges),
        "label_counts": dict(label_counts), "top_nodes": top_nodes,
        "node_labels": node_labels, "out_degree": dict(out_degree),
        "in_degree": dict(in_degree), "truncated": truncated,
    }


def save_graph_html(net) -> tuple[str, str]:
    """
    Sauvegarde le HTML PyVis dans un répertoire temporaire persistant
    et injecte le JS d'arrêt de physique.
    Retourne (serve_dir, filename).
    """
    # Répertoire stable pour la session (recréé si besoin)
    serve_dir = st.session_state.get("graph_serve_dir")
    if not serve_dir or not os.path.isdir(serve_dir):
        serve_dir = tempfile.mkdtemp(prefix="pyvis_serve_")
        st.session_state["graph_serve_dir"] = serve_dir

    filename = "graph.html"
    path = os.path.join(serve_dir, filename)
    net.save_graph(path)

    with open(path, "r", encoding="utf-8") as f:
        html = f.read()

    # Forcer le fond blanc dans le body (PyVis peut injecter son propre style)
    html = html.replace(
        "<body>",
        "<body style='background:#FFFFFF;margin:0;padding:0;'>"
    )
    # Remplacer un éventuel fond sombre résiduel dans les options vis.js injectées
    html = html.replace('"background": "#0E1117"', '"background": "#FFFFFF"')
    html = html.replace("background: #0E1117", "background: #FFFFFF")

    # JS : couper la physique après stabilisation ou après 6 s
    stop_physics_js = """
    <script>
      document.addEventListener("DOMContentLoaded", function() {
        var tryStop = setInterval(function() {
          if (typeof network !== 'undefined') {
            clearInterval(tryStop);
            network.on("stabilizationIterationsDone", function() {
              network.setOptions({ physics: { enabled: false } });
            });
            setTimeout(function() {
              network.setOptions({ physics: { enabled: false } });
            }, 6000);
          }
        }, 100);
      });
    </script>
    """
    html = html.replace("</body>", stop_physics_js + "</body>")

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    return serve_dir, filename


def render_graph_iframe(net) -> str:
    """
    Sauvegarde le graphe, lance le serveur local si besoin,
    et retourne une balise <iframe> sandboxée pointant vers localhost.
    """
    serve_dir, filename = save_graph_html(net)
    port = ensure_graph_server(serve_dir)
    # Timestamp pour forcer le rechargement à chaque nouvelle requête
    ts = int(time.time())
    src = f"http://127.0.0.1:{port}/{filename}?t={ts}"

    return f"""
    <iframe
      src="{src}"
      width="100%"
      height="560px"
      style="border:1px solid #E0E0E0; border-radius:8px; background:#FFFFFF;"
      sandbox="allow-scripts allow-same-origin"
      loading="lazy"
      title="Graphe Neo4j PyVis"
    ></iframe>
    """

# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSE IA
# ══════════════════════════════════════════════════════════════════════════════

def build_analysis_prompt(stats, cypher):
    density = round(stats["nb_edges"] / max(stats["nb_nodes"], 1), 2)
    top_str = "\n".join(
        f"  - ID {nid} (type: {stats['node_labels'].get(nid, '?')}) → centralité {score}"
        for nid, score in stats["top_nodes"]
    ) or "  Aucun"
    lc_str = ", ".join(f"{l}: {c}" for l, c in stats["label_counts"].items()) or "Aucun"

    return f"""Tu es un expert OSINT et analyste de graphes de désinformation.
Voici les résultats d'une requête Cypher sur une base Neo4j utilisée dans un projet de détection de deepfakes.

Requête exécutée :
{cypher}

Statistiques :
- Nœuds : {stats['nb_nodes']} | Relations : {stats['nb_edges']} | Densité : {density} rel/nœud
- Types de nœuds : {lc_str}

Top 5 nœuds par centralité :
{top_str}

Rédige une analyse structurée en français :
1. **Résumé** : ce que ce sous-graphe représente (2-3 phrases).
2. **Patterns détectés** : hubs, clusters, chaînes de diffusion.
3. **Alertes OSINT** : anomalies, comportements coordonnés, deepfakes, bots.
4. **Recommandations** : requêtes Cypher complémentaires à creuser.

Sois concis, factuel et orienté investigation."""


def claude_analysis(stats, cypher):
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": build_analysis_prompt(stats, cypher)}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]
    except Exception as e:
        return f"⚠️ Analyse IA indisponible : {e}"

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

def render():
    st.title("🕸️ Graphe Neo4j")
    st.caption("Visualisation interactive • Requêtes Cypher • Analyse IA OSINT")

    # ── Connexion ─────────────────────────────────────────────────────────────
    with st.expander("⚙️ Connexion Neo4j (HTTP)", expanded="neo4j_connected" not in st.session_state):
        c1, c2, c3 = st.columns([3, 2, 2])
        base_url = c1.text_input("URL Neo4j", value="http://127.0.0.1:7474", key="neo4j_url")
        user     = c2.text_input("Utilisateur", value="neo4j", key="neo4j_user")
        pwd      = c3.text_input("Mot de passe", type="password", key="neo4j_pwd")

        if st.button("🔌 Connecter", use_container_width=True, key="btn_connect"):
            with st.spinner("Test de connexion…"):
                err = test_neo4j_connection(base_url, user, pwd)
            if err:
                st.error(f"Connexion échouée : {err}")
                st.session_state.pop("neo4j_connected", None)
            else:
                st.session_state["neo4j_connected"] = True
                st.session_state["neo4j_creds"] = (base_url, user, pwd)
                st.success("Connexion Neo4j établie ✅")

    if "neo4j_connected" not in st.session_state:
        st.info("🔌 Connectez-vous à Neo4j pour visualiser les graphes.")
        return

    base_url, user, pwd = st.session_state["neo4j_creds"]

    # ── Zone de requête ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔎 Requête Cypher")

    preset_keys = list(PRESET_QUERIES.keys())

    def _on_preset_change():
        selected = st.session_state["preset_select"]
        val = PRESET_QUERIES.get(selected, "")
        if val:
            st.session_state["cypher_input"] = val

    st.selectbox(
        "Requêtes prédéfinies",
        preset_keys,
        key="preset_select",
        on_change=_on_preset_change,
    )

    if "cypher_input" not in st.session_state:
        st.session_state["cypher_input"] = ""

    cypher = st.text_area(
        "Écrire ou modifier une requête Cypher :",
        height=110,
        placeholder="MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 30",
        key="cypher_input",
    )

    col_run, col_clear = st.columns([4, 1])
    run_btn   = col_run.button("▶️ Exécuter la requête", use_container_width=True, type="primary", key="btn_run")
    clear_btn = col_clear.button("🗑️ Reset", use_container_width=True, key="btn_clear")

    if clear_btn:
        for k in ("graph_result", "ia_result", "cypher_input", "preset_select"):
            st.session_state.pop(k, None)
        st.rerun()

    if run_btn and cypher.strip():
        with st.spinner("Exécution en cours…"):
            rows, err = neo4j_http_query(base_url, user, pwd, cypher.strip())
        if err:
            st.error(f"Erreur : {err}")
        elif not rows:
            st.warning("La requête n'a retourné aucun résultat.")
        else:
            st.session_state["graph_result"] = (rows, cypher.strip())
            st.session_state.pop("ia_result", None)

    # ── Graphe ────────────────────────────────────────────────────────────────
    if "graph_result" not in st.session_state:
        return

    rows, last_cypher = st.session_state["graph_result"]

    with st.spinner("Rendu du graphe…"):
        net, stats = build_pyvis(rows)

    if stats["nb_nodes"] == 0:
        st.warning("Aucun nœud à afficher.")
        return

    if stats.get("truncated"):
        st.warning(
            f"⚠️ Le graphe dépasse {MAX_DISPLAY_NODES} nœuds. Seuls les {MAX_DISPLAY_NODES} premiers "
            f"sont affichés pour éviter un plantage du navigateur. "
            f"Affinez votre requête avec un `LIMIT` plus strict."
        )

    st.markdown("---")
    st.subheader("📊 Visualisation")

    # Légende
    leg_cols = st.columns(len(LABEL_COLORS))
    for col, (label, color) in zip(leg_cols, LABEL_COLORS.items()):
        col.markdown(
            f"<span style='display:inline-block;width:11px;height:11px;border-radius:50%;"
            f"background:{color};margin-right:4px;vertical-align:middle'></span><small>{label}</small>",
            unsafe_allow_html=True,
        )

    # ── Iframe sandboxée ──────────────────────────────────────────────────────
    iframe_html = render_graph_iframe(net)
    st.markdown(iframe_html, unsafe_allow_html=True)

    port = st.session_state.get("graph_server_port", "?")
    st.caption(f"🔒 Graphe isolé via serveur HTTP local · port {port} · sandbox: allow-scripts allow-same-origin")

    # ── Métriques ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🧠 Analyse OSINT")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Nœuds",           stats["nb_nodes"])
    m2.metric("Relations",        stats["nb_edges"])
    m3.metric("Densité rel/nœud", f"{stats['nb_edges'] / max(stats['nb_nodes'], 1):.2f}")
    m4.metric("Types distincts",  len(stats["label_counts"]))

    if stats["top_nodes"]:
        with st.container(border=True):
            st.markdown("#### 🔗 Nœuds les plus connectés (centralité)")
            for rank, (nid, score) in enumerate(stats["top_nodes"], 1):
                label = stats["node_labels"].get(nid, "?")
                color = LABEL_COLORS.get(label, DEFAULT_COLOR)
                st.markdown(
                    f"**#{rank}** &nbsp;"
                    f"<span style='background:{color};color:#fff;padding:1px 8px;"
                    f"border-radius:10px;font-size:12px'>{label}</span> &nbsp;"
                    f"ID `{nid}` &nbsp;→&nbsp; **{score}** connexions",
                    unsafe_allow_html=True,
                )

    # ── Analyse IA ────────────────────────────────────────────────────────────
    st.markdown("#### 🤖 Analyse IA des patterns (Claude)")

    if "ia_result" not in st.session_state:
        if st.button("✨ Lancer l'analyse IA", use_container_width=True, type="primary", key="btn_ia_launch"):
            with st.spinner("Analyse en cours via Claude…"):
                st.session_state["ia_result"] = claude_analysis(stats, last_cypher)
            st.rerun()
    else:
        with st.container(border=True):
            st.markdown(st.session_state["ia_result"])
        if st.button("🔄 Relancer l'analyse IA", use_container_width=True, key="btn_ia_rerun"):
            with st.spinner("Nouvelle analyse…"):
                st.session_state["ia_result"] = claude_analysis(stats, last_cypher)
            st.rerun()

    if stats["label_counts"]:
        with st.expander("📈 Distribution des types de nœuds"):
            st.bar_chart(stats["label_counts"])

    with st.expander("🔍 Requête exécutée"):
        st.code(last_cypher, language="cypher")

    # ── Lien Neo4j Browser ────────────────────────────────────────────────────
    st.markdown("---")
    browser_url = base_url.rstrip("/") + "/browser/"
    st.markdown(
        f"""
        <div style='
            background:#F5F7FA; border:1px solid #CFD8DC;
            border-radius:8px; padding:14px 18px;
            display:flex; align-items:center; gap:12px;
        '>
            <span style='font-size:22px'>🗄️</span>
            <div>
                <div style='font-weight:600; color:#212121; margin-bottom:2px'>
                    Neo4j Browser
                </div>
                <div style='font-size:13px; color:#546E7A'>
                    Explorez et interrogez directement votre base de données
                </div>
            </div>
            <a href='{browser_url}' target='_blank' style='
                margin-left:auto;
                background:#018BFF; color:#fff;
                padding:8px 18px; border-radius:6px;
                text-decoration:none; font-size:14px; font-weight:500;
                white-space:nowrap;
            '>Ouvrir Neo4j Browser ↗</a>
        </div>
        """,
        unsafe_allow_html=True,
    )