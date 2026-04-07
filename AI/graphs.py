"""graphs.py — Génération de graphes visuels pour AI-FORENSICS Investigation Agent."""

from __future__ import annotations
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ai_forensics.graphs")

PLATFORM_COLORS = {
    "tiktok":    "#ff2d55",
    "instagram": "#c13584",
    "twitter":   "#1da1f2",
    "telegram":  "#2ca5e0",
    "unknown":   "#888888",
}
COMMUNITY_PALETTE = [
    "#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
    "#1abc9c","#e67e22","#34495e","#e91e63","#00bcd4",
]


class GraphGenerator:
    def __init__(self, report_dir: Path, cfg: dict):
        self.report_dir = report_dir
        self.cfg        = cfg
        gcfg            = cfg.get("graphs", {})
        self.enabled    = str(gcfg.get("enabled", "true")).lower() == "true"
        self.fmt        = gcfg.get("static_format", "png")
        self.dpi        = int(gcfg.get("dpi", 120))
        self.max_nodes  = int(gcfg.get("network_max_nodes", 50))
        self.do_network = str(gcfg.get("network_graph_html", "true")).lower() == "true"
        self.do_temporal = str(gcfg.get("temporal_graph", "true")).lower() == "true"
        self.do_deepfake = str(gcfg.get("deepfake_histogram", "true")).lower() == "true"
        report_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, raw_data: dict, entry_type: str) -> dict[str, Optional[Path]]:
        if not self.enabled:
            return {}
        paths = {}
        temporal = raw_data.get("temporal", {})
        if temporal and "error" not in temporal and self.do_temporal:
            paths["temporal"]    = self._temporal_evolution(temporal)
            paths["propagation"] = self._propagation_chart(temporal)
        media = raw_data.get("media", [])
        if media and self.do_deepfake:
            paths["deepfake"] = self._deepfake_histogram(media)
        graph = raw_data.get("campaign_graph") or raw_data.get("graph") or {}
        if graph and "error" not in graph and self.do_network:
            paths["network"] = self._network_graph(graph)
        return {k: v for k, v in paths.items() if v is not None}

    def _temporal_evolution(self, temporal: dict) -> Optional[Path]:
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import datetime
        except ImportError:
            logger.warning("matplotlib non disponible — graphe temporel ignoré")
            return None

        monthly = temporal.get("monthly", [])
        if not monthly:
            return None

        months, posts_list, accts_list, df_list = [], [], [], []
        for m in monthly:
            try:
                months.append(datetime.strptime(m["month"] + "-01", "%Y-%m-%d"))
                posts_list.append(m.get("posts", 0))
                accts_list.append(m.get("accounts", 0))
                df_list.append(round(m.get("avg_deepfake") or 0, 3))
            except Exception:
                continue
        if not months:
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        fig.suptitle("Évolution temporelle des publications", fontsize=13, fontweight="bold")

        ax1.bar(months, posts_list, width=20, color="#3498db", alpha=0.7, label="Posts")
        ax1b = ax1.twinx()
        ax1b.plot(months, accts_list, "o-", color="#e74c3c", lw=1.5, ms=4, label="Comptes actifs")
        ax1.set_ylabel("Posts", color="#3498db")
        ax1b.set_ylabel("Comptes actifs", color="#e74c3c")
        ax1.set_ylim(bottom=0); ax1b.set_ylim(bottom=0)
        l1, n1 = ax1.get_legend_handles_labels()
        l2, n2 = ax1b.get_legend_handles_labels()
        ax1.legend(l1+l2, n1+n2, loc="upper left", fontsize=8)
        ax1.grid(axis="y", alpha=0.3)

        colors_df = ["#e74c3c" if d > 0.45 else "#f39c12" if d > 0.20 else "#2ecc71" for d in df_list]
        ax2.bar(months, df_list, width=20, color=colors_df, alpha=0.8)
        ax2.axhline(y=0.45, color="#e74c3c", ls="--", lw=1, alpha=0.6, label="Seuil suspicious")
        ax2.set_ylabel("Deepfake moyen")
        ax2.set_ylim(0, max(0.5, max(df_list)*1.2) if df_list else 0.5)
        interval = max(1, len(months)//12)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
        ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out = self.report_dir / f"temporal.{self.fmt}"
        plt.savefig(out, dpi=self.dpi, bbox_inches="tight"); plt.close()
        logger.info(f"Graphe temporel : {out.name}")
        return out

    def _propagation_chart(self, temporal: dict) -> Optional[Path]:
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.patches import Patch
            from datetime import datetime
        except ImportError:
            return None

        timeline = temporal.get("timeline", [])
        if not timeline:
            return None

        by_account: dict = defaultdict(lambda: {"days": [], "posts": [], "platform": "unknown"})
        for entry in timeline:
            acc = entry.get("account", "?")
            try:
                day = datetime.strptime(entry["day"], "%Y-%m-%d")
                by_account[acc]["days"].append(day)
                by_account[acc]["posts"].append(entry.get("posts", 1))
                by_account[acc]["platform"] = entry.get("platform", "unknown")
            except Exception:
                continue

        accounts = list(by_account.keys())[:15]
        if not accounts:
            return None

        fig, ax = plt.subplots(figsize=(14, max(4, len(accounts)*0.6+2)))
        fig.suptitle("Propagation chronologique par compte", fontsize=13, fontweight="bold")

        for i, acc in enumerate(accounts):
            data  = by_account[acc]
            color = PLATFORM_COLORS.get(data["platform"], "#888888")
            sizes = [min(p*30, 300) for p in data["posts"]]
            ax.scatter(data["days"], [i]*len(data["days"]), s=sizes, c=color, alpha=0.7, zorder=3)
            if len(data["days"]) > 1:
                ax.plot([min(data["days"]), max(data["days"])], [i, i],
                        color=color, alpha=0.3, lw=1, zorder=2)

        ax.set_yticks(range(len(accounts)))
        ax.set_yticklabels([f"@{a}" for a in accounts], fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlabel("Date de publication")
        legend_elements = [Patch(facecolor=c, label=p.capitalize())
                           for p, c in PLATFORM_COLORS.items() if p != "unknown"]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
        plt.tight_layout()
        out = self.report_dir / f"propagation.{self.fmt}"
        plt.savefig(out, dpi=self.dpi, bbox_inches="tight"); plt.close()
        logger.info(f"Graphe propagation : {out.name}")
        return out

    def _deepfake_histogram(self, media: list) -> Optional[Path]:
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch
        except ImportError:
            return None

        scores = [m.get("final_score") for m in media if m.get("final_score") is not None]
        if len(scores) < 3:
            return None

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.suptitle("Distribution des scores deepfake", fontsize=13, fontweight="bold")

        bins = [i/20 for i in range(21)]
        n, bins_out, patches = ax.hist(scores, bins=bins, edgecolor="white", lw=0.5)
        for patch, left in zip(patches, bins_out[:-1]):
            patch.set_facecolor("#e74c3c" if left >= 0.65 else "#f39c12" if left >= 0.45 else "#2ecc71")

        ax.axvline(x=0.45, color="#f39c12", ls="--", lw=1.5)
        ax.axvline(x=0.65, color="#e74c3c", ls="--", lw=1.5)
        avg = sum(scores)/len(scores)
        ax.axvline(x=avg, color="#3498db", ls="-", lw=2, label=f"Moyenne ({avg:.3f})")
        ax.set_xlabel("Score deepfake"); ax.set_ylabel("Médias"); ax.set_xlim(0, 1)

        real_n  = sum(1 for s in scores if s < 0.45)
        susp_n  = sum(1 for s in scores if 0.45 <= s < 0.65)
        synth_n = sum(1 for s in scores if s >= 0.65)
        legend_elements = [
            Patch(facecolor="#2ecc71", label=f"Réel probable ({real_n})"),
            Patch(facecolor="#f39c12", label=f"Suspect ({susp_n})"),
            Patch(facecolor="#e74c3c", label=f"Synthétique ({synth_n})"),
        ]
        ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0][-1:],
                  fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out = self.report_dir / f"deepfake_distribution.{self.fmt}"
        plt.savefig(out, dpi=self.dpi, bbox_inches="tight"); plt.close()
        logger.info(f"Histogramme deepfake : {out.name}")
        return out

    def _network_graph(self, graph: dict) -> Optional[Path]:
        try:
            from pyvis.network import Network
        except ImportError:
            logger.warning("pyvis non disponible — graphe réseau ignoré")
            return None

        net = Network(height="600px", width="100%", bgcolor="#1a1a2e",
                      font_color="white", directed=False)
        net.set_options('{"physics":{"stabilization":{"iterations":100}},'
                        '"nodes":{"borderWidth":2,"size":20},'
                        '"edges":{"smooth":{"type":"continuous"}}}')

        added = set()
        count = 0

        accounts = graph.get("accounts") or graph.get("community_accounts", [])
        for acc in accounts[:self.max_nodes]:
            name  = acc.get("username") or acc.get("account") or "?"
            plat  = acc.get("platform", "unknown")
            comm  = acc.get("community_id") or acc.get("community") or 0
            color = COMMUNITY_PALETTE[int(comm) % len(COMMUNITY_PALETTE)]
            size  = 28 if acc.get("is_suspicious") else 18
            tip   = (f"@{name} ({plat})<br>"
                     f"Posts: {acc.get('post_count', acc.get('posts','?'))}<br>"
                     f"Communauté: {comm}")
            if name not in added:
                net.add_node(name, label=f"@{name}", color=color, size=size, title=tip)
                added.add(name); count += 1

        for dup in graph.get("duplicates", []):
            src = dup.get("original_author") or dup.get("username", "?")
            dst = dup.get("copier", "?")
            if src in added and dst in added:
                net.add_edge(src, dst, title=f"{dup.get('copies','?')} copies",
                             color="#e74c3c", width=2)

        for acc in graph.get("shared_hashtag_accounts", [])[:20]:
            name = acc.get("username", "?")
            if name not in added and count < self.max_nodes:
                net.add_node(name, label=f"@{name}", color="#888888", size=12,
                             title=f"@{name} — {acc.get('shared_tags','?')} hashtags communs")
                added.add(name); count += 1

        if count < 2:
            return None

        out = self.report_dir / "network.html"
        net.save_graph(str(out))

        # Injection de la légende dans le HTML généré par pyvis
        legend_html = """
<style>
  #ai-legend {
    position: fixed; top: 16px; right: 16px; z-index: 9999;
    background: rgba(20,20,40,0.93); border: 1px solid #444;
    border-radius: 8px; padding: 14px 18px; color: #eee;
    font-family: Arial, sans-serif; font-size: 13px;
    min-width: 210px; box-shadow: 0 2px 12px rgba(0,0,0,0.5);
  }
  #ai-legend h4 { margin: 0 0 10px 0; font-size: 14px; color: #fff;
                  border-bottom: 1px solid #555; padding-bottom: 6px; }
  #ai-legend .section { margin-top: 8px; font-size: 11px;
                        color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
  .ai-dot  { display:inline-block; width:12px; height:12px; border-radius:50%;
             margin-right:7px; vertical-align:middle; }
  .ai-line { display:inline-block; width:22px; height:3px;
             margin-right:7px; vertical-align:middle; }
  #ai-legend ul { margin:4px 0; padding:0; }
  #ai-legend li { list-style:none; margin:5px 0; }
</style>
<div id="ai-legend">
  <h4>&#128269; Légende</h4>
  <div class="section">Noeuds — Communauté Louvain</div>
  <ul>
    <li><span class="ai-dot" style="background:#4C72B0"></span>Communauté 0</li>
    <li><span class="ai-dot" style="background:#DD8452"></span>Communauté 1</li>
    <li><span class="ai-dot" style="background:#55A868"></span>Communauté 2</li>
    <li><span class="ai-dot" style="background:#C44E52"></span>Communauté 3</li>
    <li><span class="ai-dot" style="background:#8172B3"></span>Communauté 4+</li>
    <li><span class="ai-dot" style="background:#888888"></span>Hashtags communs</li>
  </ul>
  <div class="section">Taille</div>
  <ul>
    <li><span class="ai-dot" style="background:#aaa;width:18px;height:18px"></span>Compte suspect</li>
    <li><span class="ai-dot" style="background:#aaa;width:10px;height:10px"></span>Compte normal</li>
  </ul>
  <div class="section">Arêtes</div>
  <ul>
    <li><span class="ai-line" style="background:#e74c3c"></span>Duplication de contenu</li>
    <li><span class="ai-line" style="background:#888888"></span>Hashtags partagés</li>
  </ul>
  <div class="section">Interaction</div>
  <ul>
    <li style="color:#ccc;font-size:11px">&#128432; Survoler = détails du compte</li>
    <li style="color:#ccc;font-size:11px">&#128432; Glisser = déplacer les noeuds</li>
    <li style="color:#ccc;font-size:11px">&#128269; Molette = zoom</li>
  </ul>
</div>"""
        html = out.read_text(encoding="utf-8")
        html = html.replace("</body>", legend_html + "\n</body>") if "</body>" in html else html + legend_html
        out.write_text(html, encoding="utf-8")

        logger.info(f"Graphe réseau : {out.name}")
        return out
