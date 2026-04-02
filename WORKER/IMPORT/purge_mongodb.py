"""purge_mongodb.py
==================
Purge sélective ou complète de la base MongoDB influence_detection.

Usage :
    python purge_mongodb.py                    # purge complète (demande confirmation)
    python purge_mongodb.py --collections posts jobs media   # collections ciblées
    python purge_mongodb.py --all --yes        # purge complète sans confirmation
    python purge_mongodb.py --reset-deepfake   # remet deepfake.status=pending sur tous les posts
    python purge_mongodb.py --reset-jobs       # supprime uniquement les jobs
    python purge_mongodb.py --dry-run          # affiche ce qui serait supprimé sans toucher la base

Credentials : lus depuis ../../SCHEMA/../worker_import.cfg ou variables d'environnement.
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Import schema.py
# ---------------------------------------------------------------------------
def _load_schema():
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "SCHEMA" / "schema.py",
        Path(__file__).resolve().parent / "schema.py",
    ]
    for p in candidates:
        if p.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("schema", p)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None

schema = _load_schema()
if schema is None:
    print("ERREUR : schema.py introuvable (cherché dans ../../SCHEMA/ et dossier local)")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Credentials — depuis worker_import.cfg si disponible
# ---------------------------------------------------------------------------
import configparser

def _read_cfg():
    candidates = [
        Path(__file__).resolve().parent / "worker_import.cfg",
        Path(__file__).resolve().parent.parent / "IMPORT" / "worker_import.cfg",
    ]
    cfg = configparser.ConfigParser()
    for p in candidates:
        if p.exists():
            cfg.read(p, encoding="utf-8")
            return cfg
    return cfg

cfg = _read_cfg()

def _cfg(section, key, fallback=""):
    try:
        v = cfg.get(section, key)
        return v.strip() if v.strip() else fallback
    except Exception:
        return fallback

# ---------------------------------------------------------------------------
# Couleurs ANSI
# ---------------------------------------------------------------------------
import os
_TTY = sys.stdout.isatty()
def _c(code, s): return f"{code}{s}\033[0m" if _TTY else s
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

# ---------------------------------------------------------------------------
# Collections du projet
# ---------------------------------------------------------------------------
ALL_COLLECTIONS = ["accounts", "posts", "comments", "media", "narratives", "campaigns", "jobs"]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Purge MongoDB — influence_detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python purge_mongodb.py                              # purge complète (confirmation requise)
  python purge_mongodb.py --all --yes                  # purge complète sans confirmation
  python purge_mongodb.py --collections posts media jobs
  python purge_mongodb.py --reset-deepfake             # remet deepfake en pending
  python purge_mongodb.py --reset-jobs                 # supprime seulement les jobs
  python purge_mongodb.py --dry-run                    # simulation
        """
    )
    parser.add_argument("--all",             action="store_true",
                        help="Purge toutes les collections")
    parser.add_argument("--collections",     nargs="+", metavar="COL",
                        choices=ALL_COLLECTIONS,
                        help="Collections à purger (ex: posts media jobs)")
    parser.add_argument("--reset-deepfake",  action="store_true",
                        help="Remet deepfake.status=pending et efface les scores sur tous les posts et médias")
    parser.add_argument("--reset-jobs",      action="store_true",
                        help="Supprime tous les jobs (équivalent --collections jobs)")
    parser.add_argument("--yes", "-y",       action="store_true",
                        help="Ne pas demander de confirmation")
    parser.add_argument("--dry-run",         action="store_true",
                        help="Simulation — affiche les comptages sans supprimer")
    parser.add_argument("--host",            default=_cfg("mongodb", "host", "localhost"))
    parser.add_argument("--port",            type=int, default=int(_cfg("mongodb", "port", "27017")))
    parser.add_argument("--user",            default=_cfg("mongodb", "user", ""))
    parser.add_argument("--password",        default=_cfg("mongodb", "password", ""))
    parser.add_argument("--db",              default=_cfg("mongodb", "db", "influence_detection"))
    parser.add_argument("--auth-db",         default=_cfg("mongodb", "auth_db", "influence_detection"))
    args = parser.parse_args()

    # Connexion
    print(f"Connexion à MongoDB {args.host}:{args.port}/{args.db}...")
    try:
        db = schema.get_db(
            host=args.host, port=args.port,
            user=args.user or None, password=args.password or None,
            db_name=args.db, auth_db=args.auth_db or None,
        )
        print(_c(GREEN, "✓ Connecté"))
    except Exception as e:
        print(_c(RED, f"✗ Connexion échouée : {e}"))
        sys.exit(1)

    # Mode --reset-deepfake
    if args.reset_deepfake:
        _reset_deepfake(db, args.dry_run)
        return

    # Déterminer les collections à purger
    if args.reset_jobs:
        targets = ["jobs"]
    elif args.collections:
        targets = args.collections
    elif args.all:
        targets = ALL_COLLECTIONS
    else:
        # Par défaut : purge complète mais demande confirmation
        targets = ALL_COLLECTIONS

    # Afficher les comptages
    print()
    print(_c(BOLD, "Collections ciblées :"))
    total_docs = 0
    counts = {}
    for col in targets:
        n = db[col].count_documents({})
        counts[col] = n
        total_docs += n
        color = RED if n > 0 else DIM
        print(f"  {_c(CYAN, col):<20} {_c(color, str(n))} document(s)")

    print(f"\n  {'TOTAL':<20} {_c(BOLD, str(total_docs))} document(s)")

    if args.dry_run:
        print(_c(YELLOW, "\n[DRY-RUN] Aucune suppression effectuée."))
        return

    if total_docs == 0:
        print(_c(DIM, "\nBase déjà vide — rien à faire."))
        return

    # Confirmation
    if not args.yes:
        print()
        print(_c(YELLOW + BOLD, f"⚠  ATTENTION : {total_docs} document(s) vont être supprimés définitivement."))
        rep = input("Confirmer ? [oui/NON] : ").strip().lower()
        if rep not in ("oui", "o", "yes", "y"):
            print("Annulé.")
            return

    # Suppression
    print()
    for col in targets:
        if counts[col] == 0:
            print(f"  {_c(DIM, col):<20} déjà vide")
            continue
        result = db[col].delete_many({})
        print(f"  {_c(CYAN, col):<20} {_c(GREEN, str(result.deleted_count))} supprimé(s)")

    print()
    print(_c(GREEN + BOLD, "✓ Purge terminée."))


def _reset_deepfake(db, dry_run: bool):
    """
    Remet deepfake.status=pending et efface les scores sur tous les posts et médias.
    Utile pour relancer une analyse complète sans re-importer les données.
    """
    reset_post = {
        "$set": {
            "deepfake.status":           "pending",
            "deepfake.final_score":      None,
            "deepfake.prediction":       None,
            "deepfake.model_divergence": None,
            "deepfake.scores":           {},
            "deepfake.raw_scores":       {},
            "deepfake.artifact_score":   None,
            "deepfake.frames_analyzed":  None,
            "deepfake.pipeline_version": None,
            "deepfake.processed_at":     None,
            "deepfake.error":            None,
        }
    }
    reset_media = {
        "$set": {
            "deepfake.status":           "pending",
            "deepfake.final_score":      None,
            "deepfake.prediction":       None,
            "deepfake.model_divergence": None,
            "deepfake.scores":           {},
            "deepfake.raw_scores":       {},
            "deepfake.artifact_score":   None,
            "deepfake.frames_analyzed":  None,
            "deepfake.faces_detected":   None,
            "deepfake.pipeline_version": None,
            "deepfake.processed_at":     None,
            "deepfake.error":            None,
        }
    }

    n_posts  = db.posts.count_documents({"deepfake.status": {"$ne": "pending"}})
    n_media  = db.media.count_documents({"deepfake.status": {"$ne": "pending"}})
    n_jobs   = db.jobs.count_documents({"type": "deepfake_analysis"})

    print(_c(BOLD, "\nReset deepfake :"))
    print(f"  posts  à remettre en pending : {_c(CYAN, str(n_posts))}")
    print(f"  médias à remettre en pending : {_c(CYAN, str(n_media))}")
    print(f"  jobs deepfake_analysis       : {_c(CYAN, str(n_jobs))} (seront supprimés)")

    if dry_run:
        print(_c(YELLOW, "\n[DRY-RUN] Aucune modification effectuée."))
        return

    r1 = db.posts.update_many({}, reset_post)
    r2 = db.media.update_many({}, reset_media)
    r3 = db.jobs.delete_many({"type": "deepfake_analysis"})

    print()
    print(f"  {_c(GREEN, str(r1.modified_count))} posts remis en pending")
    print(f"  {_c(GREEN, str(r2.modified_count))} médias remis en pending")
    print(f"  {_c(GREEN, str(r3.deleted_count))} jobs deepfake supprimés")
    print(_c(GREEN + BOLD, "\n✓ Reset deepfake terminé."))


if __name__ == "__main__":
    main()
