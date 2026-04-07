"""prompts.py
============
Prompts système et templates pour l'agent d'investigation AI-FORENSICS.

Contenu :
  - SYSTEM_PROMPT       : instructions principales de l'agent
  - ENTRY_POINTS        : séquences d'outils selon le point d'entrée
  - REPORT_TEMPLATE     : template Markdown du rapport final
  - build_system_prompt : construit le prompt selon le contexte
  - build_initial_query : formule la première requête selon le point d'entrée

Modifier ce fichier pour itérer sur le comportement de l'agent
sans toucher à investigation_agent.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

# ===========================================================================
# Prompt système principal
# ===========================================================================

SYSTEM_PROMPT = """Tu es un agent d'investigation spécialisé en détection de campagnes d'influence sur les réseaux sociaux.
Tu travailles pour le système AI-FORENSICS qui analyse des comptes, médias et narratives suspects sur TikTok, Instagram, Twitter et Telegram.

## Ton rôle

Tu reçois un point d'entrée (compte, post, narrative ou campagne) et tu mènes une investigation structurée en appelant des outils de lecture sur les bases de données MongoDB et Neo4j du système.
Tu produis ensuite un rapport d'analyse en Markdown.

## Outils disponibles

Tu disposes des outils suivants (lecture seule — tu n'écris jamais en base) :

- **get_account_info(platform, unique_id)** : profil complet d'un compte + résumé deepfake agrégé. À appeler EN PREMIER pour tout point d'entrée "compte".
- **get_account_posts(account_id, limit)** : derniers posts avec scores deepfake et NLP.
- **get_media_scores(account_id)** : scores deepfake détaillés par modèle pour les médias du compte.
- **get_graph_neighbors(platform_id, depth)** : voisinage Neo4j — comptes liés, hashtags partagés, médias réutilisés.
- **get_narrative(narrative_id)** : détails d'une narrative (label, mots-clés, ratio synthétique).
- **get_campaign_signals(campaign_id)** : signaux de coordination d'une campagne détectée.
- **search_accounts_by_narrative(narrative_id)** : comptes associés à une narrative.

## Séquences d'investigation recommandées

### Point d'entrée : compte
1. get_account_info → récupère account_id, platform_id, narratives, campaigns
2. get_account_posts → analyse les posts (scores, sentiments, narratives)
3. get_media_scores → scores détaillés par modèle, divergence, réutilisation
4. get_graph_neighbors → voisinage dans le graphe (coordination)
5. get_narrative → pour chaque narrative_id identifié
6. get_campaign_signals → si campaign_ids présents

### Point d'entrée : narrative
1. get_narrative → profil de la narrative
2. search_accounts_by_narrative → comptes qui la portent
3. get_account_info + get_media_scores → pour les 2-3 comptes les plus actifs
4. get_graph_neighbors → pour détecter la coordination

### Point d'entrée : campagne
1. get_campaign_signals → signaux et comptes membres
2. get_account_info → pour les comptes membres les plus suspects
3. get_media_scores → pour les comptes à fort ratio synthétique

## Interprétation des scores deepfake

Les scores sont produits par 3 modèles orthogonaux :
- **sdxl-detector** : détecte les images générées par diffusion (SD/SDXL, FLUX)
- **swinv2-openfake** : détecte FLUX, MidJourney v6, DALL-E 3, Grok-2 (F1=0.943)
- **synthbuster** : détecte les artefacts spectraux Fourier

Interprétation du score final (score_final = moyenne pondérée corrigée des biais) :
- 0.00 – 0.45 : likely_real — contenu probablement authentique
- 0.45 – 0.65 : suspicious — ambigu, à examiner manuellement
- 0.65 – 1.00 : synthetic — contenu probablement généré par IA

**Divergence inter-modèles (model_divergence)** :
- < 0.15 : modèles concordants → résultat fiable
- 0.15 – 0.30 : légère divergence → résultat à nuancer
- > 0.30 : forte divergence → résultat peu fiable, vérification manuelle requise

**Pattern particulier à connaître** : sdxl-detector très élevé (>0.9) + swinv2-openfake bas (<0.1) = divergence élevée = vidéo réelle avec compression JPEG agressive, PAS forcément synthétique. Ne pas sur-interpréter.

## Signaux de coordination à rechercher

- Même média partagé par plusieurs comptes (reuse.seen_count > 1)
- Hashtags identiques sur des comptes sans lien apparent
- Posts quasi-simultanés (nlp.is_duplicate_of renseigné)
- Comptes dans la même campagne (flags.campaign_ids)
- bot_score élevé (> 0.7) sur plusieurs comptes d'une même narrative
- synthetic_ratio élevé dans une narrative (> 0.5)

## Limites à respecter

- Ne jamais conclure à une campagne coordonnée sur la base d'un seul signal
- Si model_divergence > 0.30, mentionner l'incertitude dans l'analyse
- Si Neo4j est indisponible, le mentionner et continuer avec MongoDB seul
- Si les données sont insuffisantes, conclure "données insuffisantes pour conclure"
- Toujours mentionner que le rapport nécessite une vérification humaine

## Format de raisonnement

Raisonne étape par étape avant chaque appel d'outil :
- Qu'est-ce que je cherche à savoir ?
- Quel outil est le plus adapté ?
- Que font les résultats ?
- Quelle est la prochaine étape logique ?

Arrête d'appeler des outils quand tu as suffisamment de données pour rédiger le rapport.
Maximum 10 appels d'outils par investigation.
"""

# ===========================================================================
# Descriptions des points d'entrée
# ===========================================================================

ENTRY_POINTS = {
    "account": {
        "description": "Investigation d'un compte suspect",
        "first_tool":  "get_account_info",
        "sequence": [
            "get_account_info",
            "get_account_posts",
            "get_media_scores",
            "get_graph_neighbors",
            "get_narrative",
            "get_campaign_signals",
        ],
    },
    "narrative": {
        "description": "Investigation d'une narrative suspecte",
        "first_tool":  "get_narrative",
        "sequence": [
            "get_narrative",
            "search_accounts_by_narrative",
            "get_account_info",
            "get_media_scores",
            "get_graph_neighbors",
        ],
    },
    "campaign": {
        "description": "Investigation d'une campagne détectée",
        "first_tool":  "get_campaign_signals",
        "sequence": [
            "get_campaign_signals",
            "get_account_info",
            "get_media_scores",
            "get_narrative",
        ],
    },
    "post": {
        "description": "Investigation à partir d'un post suspect",
        "first_tool":  "get_account_posts",
        "sequence": [
            "get_account_posts",
            "get_account_info",
            "get_media_scores",
            "get_graph_neighbors",
            "get_narrative",
        ],
    },
}

# ===========================================================================
# Template du rapport Markdown
# ===========================================================================

REPORT_TEMPLATE = """# Rapport d'investigation — {target_label}
**Date :** {date}
**Point d'entrée :** {entry_type} — `{entry_value}`
**Score de suspicion global :** {suspicion_score} / 1.0
**Niveau de confiance :** {confidence_level}

---

## Synthèse
{synthesis}

---

## Analyse des médias
{media_analysis}

---

## Position dans le réseau
{network_analysis}

---

## Narratives portées
{narratives_analysis}

---

## Signaux de coordination
{coordination_signals}

---

## Conclusion
{conclusion}

---

## Recommandations
{recommendations}

---

*Rapport généré automatiquement par AI-FORENSICS Investigation Agent*
*Modèle : {model_name} — Ce rapport nécessite une vérification humaine avant toute décision.*
*Les scores deepfake sont des indicateurs probabilistes, pas des preuves.*
"""

# Sections par défaut quand les données sont manquantes
_DEFAULT_SECTIONS = {
    "media_analysis":       "Aucun média analysé ou données insuffisantes.",
    "network_analysis":     "Neo4j indisponible ou aucune relation détectée.",
    "narratives_analysis":  "Aucune narrative associée.",
    "coordination_signals": "Aucun signal de coordination détecté.",
    "recommendations":      "- Approfondir l'investigation manuellement.\n- Vérifier les médias les plus suspects.",
}

# ===========================================================================
# Builders
# ===========================================================================

def build_system_prompt(
    neo4j_available: bool = True,
    verbose: bool = False,
) -> str:
    """
    Construit le system prompt final.

    Args:
        neo4j_available : si False, ajoute une note sur l'indisponibilité Neo4j
        verbose         : si True, ajoute des instructions de logging détaillé

    Returns:
        str — system prompt complet
    """
    prompt = SYSTEM_PROMPT

    if not neo4j_available:
        prompt += """
## Note : Neo4j indisponible

L'outil get_graph_neighbors n'est pas disponible pour cette session.
Mène l'investigation avec MongoDB seul et mentionne cette limitation dans le rapport.
"""

    if verbose:
        prompt += """
## Mode verbeux activé

Détaille chaque étape de ton raisonnement avant d'appeler un outil.
Indique explicitement pourquoi tu choisis cet outil et ce que tu espères trouver.
"""

    return prompt


def build_initial_query(
    entry_type: str,
    entry_value: str,
    platform: Optional[str] = None,
    extra: Optional[str] = None,
) -> str:
    """
    Formule la première requête utilisateur envoyée à l'agent.

    Args:
        entry_type  : account | post | narrative | campaign
        entry_value : identifiant (username, id, etc.)
        platform    : plateforme (pour les comptes)
        extra       : instructions supplémentaires optionnelles

    Returns:
        str — message initial pour l'agent
    """
    ep = ENTRY_POINTS.get(entry_type, ENTRY_POINTS["account"])

    if entry_type == "account":
        if not platform:
            raise ValueError("platform est requis pour un point d'entrée 'account'")
        query = (
            f"Mène une investigation complète sur le compte `{entry_value}` "
            f"sur la plateforme `{platform}`.\n\n"
            f"Commence par récupérer son profil avec get_account_info, "
            f"puis analyse ses posts, médias, position dans le réseau et narratives associées.\n"
            f"Conclus avec un rapport structuré en Markdown."
        )

    elif entry_type == "narrative":
        query = (
            f"Mène une investigation complète sur la narrative `{entry_value}`.\n\n"
            f"Commence par get_narrative pour comprendre le contenu, "
            f"puis identifie les comptes qui la portent avec search_accounts_by_narrative.\n"
            f"Analyse les comptes les plus actifs et conclus avec un rapport structuré en Markdown."
        )

    elif entry_type == "campaign":
        query = (
            f"Mène une investigation complète sur la campagne `{entry_value}`.\n\n"
            f"Commence par get_campaign_signals pour récupérer les signaux de coordination, "
            f"puis analyse les comptes membres les plus suspects.\n"
            f"Conclus avec un rapport structuré en Markdown."
        )

    elif entry_type == "post":
        query = (
            f"Mène une investigation complète à partir du post `{entry_value}`.\n\n"
            f"Récupère les informations du compte auteur, analyse ses médias "
            f"et sa position dans le réseau.\n"
            f"Conclus avec un rapport structuré en Markdown."
        )

    else:
        query = (
            f"Mène une investigation sur : `{entry_value}` (type : {entry_type}).\n"
            f"Utilise les outils disponibles pour collecter les données pertinentes "
            f"et produis un rapport structuré en Markdown."
        )

    if extra:
        query += f"\n\nInstructions supplémentaires : {extra}"

    return query


def build_report(
    target_label:        str,
    entry_type:          str,
    entry_value:         str,
    suspicion_score:     float,
    confidence_level:    str,
    synthesis:           str,
    conclusion:          str,
    model_name:          str,
    media_analysis:      Optional[str] = None,
    network_analysis:    Optional[str] = None,
    narratives_analysis: Optional[str] = None,
    coordination_signals: Optional[str] = None,
    recommendations:     Optional[str] = None,
) -> str:
    """
    Remplit le template de rapport avec les données de l'investigation.

    Args:
        target_label      : label humain de la cible (ex: "@nolimitmoneyy0 (TikTok)")
        entry_type        : account | post | narrative | campaign
        entry_value       : identifiant brut
        suspicion_score   : score [0-1]
        confidence_level  : Faible | Moyen | Élevé
        synthesis         : paragraphe de synthèse
        conclusion        : qualification finale
        model_name        : nom du modèle LLM utilisé
        *_analysis        : sections optionnelles (défaut si None)

    Returns:
        str — rapport Markdown complet
    """
    score_str = f"{suspicion_score:.2f}" if isinstance(suspicion_score, float) else str(suspicion_score)

    return REPORT_TEMPLATE.format(
        target_label        = target_label,
        date                = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S UTC"),
        entry_type          = entry_type,
        entry_value         = entry_value,
        suspicion_score     = score_str,
        confidence_level    = confidence_level,
        synthesis           = synthesis,
        media_analysis      = media_analysis      or _DEFAULT_SECTIONS["media_analysis"],
        network_analysis    = network_analysis    or _DEFAULT_SECTIONS["network_analysis"],
        narratives_analysis = narratives_analysis or _DEFAULT_SECTIONS["narratives_analysis"],
        coordination_signals= coordination_signals or _DEFAULT_SECTIONS["coordination_signals"],
        conclusion          = conclusion,
        recommendations     = recommendations     or _DEFAULT_SECTIONS["recommendations"],
        model_name          = model_name,
    )


# ===========================================================================
# Constantes utiles pour l'agent
# ===========================================================================

# Seuils de décision
SCORE_THRESHOLDS = {
    "likely_real": 0.45,
    "suspicious":  0.65,
    "synthetic":   1.00,
}

# Niveaux de confiance selon la divergence inter-modèles
CONFIDENCE_LEVELS = {
    "high":   (0.00, 0.15),   # divergence < 0.15
    "medium": (0.15, 0.30),   # divergence 0.15–0.30
    "low":    (0.30, 1.00),   # divergence > 0.30
}

CONFIDENCE_LABELS = {
    "high":   "Élevé",
    "medium": "Moyen",
    "low":    "Faible",
}


def score_to_label(score: float) -> str:
    """Convertit un score deepfake en label lisible."""
    if score is None:
        return "non analysé"
    if score < SCORE_THRESHOLDS["likely_real"]:
        return "likely_real"
    if score < SCORE_THRESHOLDS["suspicious"]:
        return "suspicious"
    return "synthetic"


def divergence_to_confidence(divergence: float) -> str:
    """Convertit une divergence inter-modèles en niveau de confiance."""
    if divergence is None:
        return "low"
    for level, (low, high) in CONFIDENCE_LEVELS.items():
        if low <= divergence < high:
            return level
    return "low"


# ===========================================================================
# Test standalone
# ===========================================================================

if __name__ == "__main__":
    print("=== System prompt ===")
    print(build_system_prompt())
    print("\n=== Requête initiale (compte) ===")
    print(build_initial_query("account", "nolimitmoneyy0", platform="tiktok"))
    print("\n=== Requête initiale (narrative) ===")
    print(build_initial_query("narrative", "69d2774d7dc202593e1b35fa"))
    print("\n=== Exemple rapport ===")
    print(build_report(
        target_label     = "@nolimitmoneyy0 (TikTok)",
        entry_type       = "account",
        entry_value      = "nolimitmoneyy0",
        suspicion_score  = 0.32,
        confidence_level = "Faible (forte divergence inter-modèles)",
        synthesis        = "Le compte nolimitmoneyy0 publie du contenu crypto/news sur TikTok. "
                           "Les scores deepfake sont modérés mais la divergence inter-modèles "
                           "est élevée, ce qui limite la fiabilité des résultats.",
        conclusion       = "Activité cohérente avec un compte de veille crypto. "
                           "Aucun signal de campagne coordonnée détecté sur ce dataset.",
        model_name       = "llama3.1:8b via Ollama",
    ))
