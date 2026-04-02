#!/usr/bin/env python3
"""mongo_status.py
==================

Affiche l'état de la base MongoDB influence_detection.
Lit les credentials depuis worker_import.cfg ou les variables d'environnement.

Lancement :
    python mongo_status.py
    python mongo_status.py --cfg /chemin/vers/worker_import.cfg
    python mongo_status.py --verbose    # détails par projet + stats deepfake/nlp
"""

import argparse
import configparser
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Résolution schema.py — même logique que tous les autres workers
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_SCHEMA_CANDIDATES = [
    _HERE.parent.parent / "SCHEMA",
    _HERE.parent,
    _HERE,
]
for _schema_dir in _SCHEMA_CANDIDATES:
    if (_schema_dir / "schema.py").exists():
        sys.path.insert(0, str(_schema_dir))
        break

try:
    from schema import get_db
except ImportError as e:
    print(f"✗ Impossible d'importer schema.py : {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Codes ANSI
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RED    = "\033[31m"
MAGENTA= "\033[35m"

def ok(s):   return f"{BOLD}{GREEN}{s}{RESET}"
def err(s):  return f"{BOLD}{RED}{s}{RESET}"
def hi(s):   return f"{BOLD}{CYAN}{s}{RESET}"
def dim(s):  return f"{DIM}{s}{RESET}"
def warn(s): return f"{YELLOW}{s}{RESET}"
def mag(s):  return f"{MAGENTA}{s}{RESET}"


# ---------------------------------------------------------------------------
# Lecture config
# ---------------------------------------------------------------------------

DEFAULT_CFG = _HERE / "worker_import.cfg"



def _cfg_int(section, key: str, fallback: int = 0) -> int | None:
    """Lit une valeur entière depuis une section configparser.
    Retourne None si la valeur est absente ou vide (champ vidé pour sécurité).
    """
    val = section.get(key, "").strip()
    try:
        return int(val) or None
    except ValueError:
        return fallback or None

def load_credentials(cfg_path: Path):
    """Lit host/port/user/password depuis le cfg. Retourne un dict ou None."""
    if not cfg_path.exists():
        return None
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path, encoding="utf-8")
    if "mongodb" not in cfg:
        return None
    m = cfg["mongodb"]
    return {
        "host":     m.get("host")     or None,
        "port":     _cfg_int(m, "port"),
        "user":     m.get("user")     or None,
        "password": m.get("password") or None,
        "auth_db":  m.get("auth_db")  or None,
        "db":       m.get("db")       or None,
    }


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

COLLECTIONS = ["accounts", "posts", "comments", "media", "narratives", "campaigns", "jobs"]

COL_WIDTH = max(len(c) for c in COLLECTIONS) + 2


def bar(ratio: float, width: int = 20) -> str:
    filled = round(ratio * width)
    return f"{'█' * filled}{'░' * (width - filled)}"


def print_status(db, verbose: bool) -> None:
    host = db.client.address
    print()
    print(hi(f"  Base : {db.name}  —  {host[0]}:{host[1]}"))
    print(dim("  " + "─" * 58))

    counts = {}
    for col in COLLECTIONS:
        counts[col] = db[col].count_documents({})

    total = sum(counts.values())
    col_w = COL_WIDTH

    print()
    for col, n in counts.items():
        ratio  = n / total if total else 0
        b      = bar(ratio, 16)
        marker = ok("✓") if n > 0 else dim("·")
        print(f"  {marker}  {col:<{col_w}} {n:>7,}  {dim(b)}")

    print(dim("  " + "─" * 58))
    print(f"  {'TOTAL':<{col_w + 4}} {BOLD}{total:>7,}{RESET}")
    print()

    # --- Statuts pipeline ---
    print(hi("  Statuts pipeline"))
    print(dim("  " + "─" * 58))

    # Deepfake sur posts
    if counts["posts"]:
        df_done    = db.posts.count_documents({"deepfake.status": "done"})
        df_pending = db.posts.count_documents({"deepfake.status": "pending"})
        df_skip    = db.posts.count_documents({"deepfake.status": "skipped"})
        df_err     = db.posts.count_documents({"deepfake.status": "error"})
        print(f"  {'posts.deepfake':<{col_w + 4}}"
              f"  {ok(f'done:{df_done}')}"
              f"  {warn(f'pending:{df_pending}')}"
              f"  {dim(f'skip:{df_skip}')}"
              f"  {err(f'err:{df_err}') if df_err else dim('err:0')}")

    # NLP sur posts
    if counts["posts"]:
        nlp_done    = db.posts.count_documents({"nlp.status": "done"})
        nlp_pending = db.posts.count_documents({"nlp.status": "pending"})
        nlp_err     = db.posts.count_documents({"nlp.status": "error"})
        print(f"  {'posts.nlp':<{col_w + 4}}"
              f"  {ok(f'done:{nlp_done}')}"
              f"  {warn(f'pending:{nlp_pending}')}"
              f"  {err(f'err:{nlp_err}') if nlp_err else dim('err:0')}")

    # Deepfake sur media
    if counts["media"]:
        mdf_done    = db.media.count_documents({"deepfake.status": "done"})
        mdf_pending = db.media.count_documents({"deepfake.status": "pending"})
        mdf_err     = db.media.count_documents({"deepfake.status": "error"})
        print(f"  {'media.deepfake':<{col_w + 4}}"
              f"  {ok(f'done:{mdf_done}')}"
              f"  {warn(f'pending:{mdf_pending}')}"
              f"  {err(f'err:{mdf_err}') if mdf_err else dim('err:0')}")

    # Sync Neo4j
    for col in ("posts", "comments", "media"):
        if counts[col]:
            synced     = db[col].count_documents({"sync.neo4j": True})
            not_synced = db[col].count_documents({"sync.neo4j": False})
            no_field   = counts[col] - synced - not_synced
            label = f"{col}.sync.neo4j"
            detail = f"{ok(f'ok:{synced}')}  {warn(f'pending:{not_synced}')}"
            if no_field:
                detail += f"  {err(f'⚠ sans champ:{no_field}')}"
            print(f"  {label:<{col_w + 4}}  {detail}")

    # Jobs en attente
    if counts["jobs"]:
        print()
        print(hi("  Jobs"))
        print(dim("  " + "─" * 58))
        for jtype in ("deepfake_analysis", "nlp_analysis", "etl_sync"):
            j_pending = db.jobs.count_documents({"type": jtype, "status": "pending"})
            j_done    = db.jobs.count_documents({"type": jtype, "status": "done"})
            j_err     = db.jobs.count_documents({"type": jtype, "status": "error"})
            if j_pending + j_done + j_err:
                print(f"  {jtype:<{col_w + 4}}"
                      f"  {warn(f'pending:{j_pending}')}"
                      f"  {ok(f'done:{j_done}')}"
                      f"  {err(f'err:{j_err}') if j_err else dim('err:0')}")

    # --- Mode verbose : répartition par projet ---
    if verbose and counts["posts"]:
        print()
        print(hi("  Répartition par projet (posts)"))
        print(dim("  " + "─" * 58))
        pipeline = [
            {"$group": {"_id": "$source.project", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        results = list(db.posts.aggregate(pipeline))
        if results:
            for r in results:
                proj  = r["_id"] or dim("(sans projet)")
                n     = r["count"]
                ratio = n / counts["posts"]
                print(f"  {mag(str(proj)):<40}  {n:>6,}  {dim(bar(ratio, 14))}")
        else:
            print(f"  {dim('Aucun champ source.project trouvé')}")

        # Plateformes
        print()
        print(hi("  Répartition par plateforme (posts)"))
        print(dim("  " + "─" * 58))
        pipeline = [
            {"$group": {"_id": "$platform", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        for r in db.posts.aggregate(pipeline):
            plat  = r["_id"] or dim("(inconnu)")
            n     = r["count"]
            ratio = n / counts["posts"]
            print(f"  {hi(str(plat)):<30}  {n:>6,}  {dim(bar(ratio, 14))}")

    print()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="État de la base MongoDB influence_detection")
    p.add_argument("--cfg", "-c", type=Path, default=DEFAULT_CFG,
                   help=f"Fichier de config credentials (défaut : {DEFAULT_CFG.name})")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Afficher la répartition par projet et plateforme")
    p.add_argument("--watch", "-w", nargs="?", const=5, type=int, metavar="N",
                   help="Mode live : rafraîchit toutes les N secondes (défaut : 5)")
    return p.parse_args()


def _render(db, verbose: bool) -> str:
    """Capture toute la sortie de print_status dans une chaîne."""
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_status(db, verbose=verbose)
    return buf.getvalue()


def _move_to_top(n_lines: int) -> None:
    """Remonte le curseur de n_lines lignes et efface jusqu'en bas."""
    if n_lines > 0:
        sys.stdout.write(f"\033[{n_lines}A\033[J")
        sys.stdout.flush()


def _count_lines(text: str) -> int:
    return text.count("\n")


if __name__ == "__main__":
    import time

    args  = parse_args()
    creds = load_credentials(args.cfg) or {}

    host    = creds.get("host") or "localhost"
    port    = creds.get("port") or 27017
    db_name = creds.get("db")   or "influence_detection"

    print(f"\n  Connexion à MongoDB {host}:{port}/{db_name}…")

    try:
        db = get_db(
            db_name  = db_name,
            host     = host,
            port     = port,
            user     = creds.get("user"),
            password = creds.get("password"),
            auth_db  = creds.get("auth_db"),
        )
        print(f"  {ok('✓ Connecté')}")
    except ConnectionError as e:
        print(f"  {err('✗ Connexion échouée')} : {e}")
        sys.exit(1)

    interval   = args.watch          # None = one-shot, int = secondes
    prev_lines = 0                   # nb de lignes affichées au tour précédent
    # +2 pour les lignes "Connexion…" et "✓ Connecté" affichées avant la boucle
    header_lines = 2

    while True:
        now    = time.strftime("%H:%M:%S")
        output = _render(db, verbose=args.verbose)

        if interval is not None:
            # Effacer l'affichage précédent (sauf le header de connexion)
            _move_to_top(prev_lines)
            footer = (
                f"\n  {dim(f'Rafraîchissement toutes les {interval}s — ')}"
                f"{dim(f'dernière MAJ : {now}')}"
                f"  {dim('[Ctrl+C pour quitter]')}"
                f"\n"
            )
            sys.stdout.write(output + footer)
            sys.stdout.flush()
            prev_lines = _count_lines(output + footer)
        else:
            sys.stdout.write(output)

        if interval is None:
            break

        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n  {dim('Arrêt.')}")
            break
