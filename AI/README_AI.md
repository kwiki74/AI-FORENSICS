# Module AI/ — Agent d'investigation autonome

> Analyse automatique de comptes, médias, narratives et campagnes suspects.  
> Produit un rapport Markdown structuré à partir des données du pipeline AI-FORENSICS.

Ce module est une **solution parallèle** — il ne modifie aucun worker existant. Il se branche en **lecture seule** sur MongoDB et Neo4j pour exploiter les données déjà produites par le pipeline.

> ⚠️ **Données potentiellement partielles.** Le scrapping sur lequel repose toute analyse n'est pas nécessairement exhaustif. Les données disponibles représentent un échantillon de l'activité réelle. Tous les rapports mentionnent cette limitation systématiquement.

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture](#2-architecture)
3. [Installation](#3-installation)
4. [Configuration](#4-configuration)
5. [Utilisation](#5-utilisation)
6. [Fichiers du module](#6-fichiers-du-module)
7. [Fonctionnement interne](#7-fonctionnement-interne)
8. [Interprétation des rapports](#8-interprétation-des-rapports)
9. [Glossaire](#9-glossaire)
10. [Limites et précautions](#10-limites-et-précautions)

---

## 1. Vue d'ensemble

L'agent fonctionne en trois phases :

**Phase 1 — Collecte déterministe** : données collectées en Python selon une séquence fixe. Aucune décision déléguée au LLM.

**Phase 1b — Enrichissement adaptatif** : si certains seuils sont atteints (bot_score élevé, duplications importantes, deepfake alert...), l'agent déclenche automatiquement des appels supplémentaires. Les seuils sont configurables dans `ai_agent.cfg`.

**Phase 2 — Rédaction par le LLM** : les données collectées sont transmises au modèle de langage, qui rédige le rapport en français.

En sortie, chaque rapport inclut :
- Le rapport Markdown avec tableaux, frises ASCII et requêtes Cypher
- Des graphes visuels annexés (PNG + HTML interactif)
- Des suggestions de scrapping structurées et actionnables

---

## 2. Architecture

```
CLI
  --account / --narrative / --campaign / --post
  --all-campaigns / --project <nom>
    │
    ▼
investigation_agent.py
    │
    ├── Phase 1 : collecte déterministe (tools.py)
    │     ├── get_account_info, get_account_posts, get_media_scores
    │     ├── get_graph_neighbors       → Neo4j (communauté, hashtags, doublons)
    │     ├── get_campaign_signals      → MongoDB campaigns
    │     ├── get_campaign_graph        → Neo4j (comptes, doublons, hashtags)
    │     └── get_temporal_analysis     → Neo4j (propagation, cadence, silences,
    │                                     compte semence, deepfake×temporel, cross-campagne)
    │
    ├── Phase 1b : enrichissement adaptatif (_enrich_if_needed)
    │     └── Appels supplémentaires selon seuils ai_agent.cfg
    │
    ├── Calcul du score de suspicion (Python pur)
    │
    ├── Phase 2 : rédaction LLM
    │
    ├── Génération graphes (graphs.py)
    │     ├── temporal.png
    │     ├── propagation.png
    │     ├── deepfake_distribution.png
    │     └── network.html
    │
    ├── Suggestions de scrapping (Python pur → Markdown)
    │
    └── Rapport final → reports/<timestamp>_<type>_<cible>/

Mode batch :
    --all-campaigns / --project
        ├── Investigation individuelle × N campagnes
        └── CR de synthèse → reports/<timestamp>_synthese_<label>.md
```

---

## 3. Installation

```bash
conda create -n AI_agent python=3.11
conda activate AI_agent
pip install -r AI/requirements_ai.txt
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b

# Graphes (optionnels mais recommandés)
pip install matplotlib pyvis
```

---

## 4. Configuration

### .env

```env
AI_PROVIDER=ollama
AI_MODEL=llama3.1:8b
AI_BASE_URL=http://localhost:11434
AI_API_KEY=
AI_REPORTS_DIR=./reports
AI_TEMPERATURE=0.2
```

Providers alternatifs :

```env
# Groq (gratuit, 70B)
AI_PROVIDER=groq
AI_MODEL=llama-3.1-70b-versatile
AI_API_KEY=gsk_...

# Anthropic
AI_PROVIDER=anthropic
AI_MODEL=claude-sonnet-4-20250514
AI_API_KEY=sk-ant-...
```

### ai_agent.cfg

```ini
[enrichment]
bot_score_threshold = 0.7
duplicate_threshold = 3
deepfake_alert_threshold = 0.65
community_min_size = 5
max_extra_calls = 4

[scraping]
copies_received_high = 5
shared_hashtags_suggest = 10
cross_platform_auto_suggest = true
cross_campaign_high = 3

[graphs]
enabled = true
static_format = png
dpi = 150
network_graph_html = true
network_max_nodes = 50

[batch]
synthesis_report = true
max_campaigns_in_table = 50
top_amplifiers = 10
```

---

## 5. Utilisation

```bash
conda activate AI_agent
cd ~/AI-FORENSICS/AI

# Compte
python investigation_agent.py --account nolimitmoneyy0 --platform tiktok

# Narrative
python investigation_agent.py --narrative 69d2774d7dc202593e1b35fa

# Campagne
python investigation_agent.py --campaign 69d504d00b27a5d21ee8f65b

# Post
python investigation_agent.py --post <mongo_id>

# Toutes les campagnes + CR synthèse
python investigation_agent.py --all-campaigns

# Campagnes d'un projet + CR synthèse
python investigation_agent.py --project projet1

# Options
python investigation_agent.py --account nolimitmoneyy0 --platform tiktok \
    --provider groq --verbose --no-save
```

---

## 6. Fichiers du module

```
AI-FORENSICS/AI/
├── investigation_agent.py   ← orchestration, CLI, enrichissement, rédaction
├── tools.py                 ← outils MongoDB + Neo4j
├── graphs.py                ← graphes PNG (matplotlib) + HTML (pyvis)
├── prompts.py               ← system prompt, templates
├── ai_agent.cfg             ← seuils et configuration
├── requirements_ai.txt
└── README_AI.md

AI-FORENSICS/reports/
├── <ts>_campaign_<id>/
│   ├── rapport.md
│   ├── temporal.png
│   ├── propagation.png
│   ├── deepfake_distribution.png
│   └── network.html
└── <ts>_synthese_<label>.md
```

---

## 7. Fonctionnement interne

### Enrichissement adaptatif

| Condition | Action déclenchée |
|---|---|
| `bot_score ≥ bot_score_threshold` | `get_graph_neighbors` approfondi |
| `duplicate_count ≥ duplicate_threshold` | `get_temporal_analysis` sur les sources |
| `deepfake_score ≥ deepfake_alert_threshold` | `search_accounts_by_narrative` |
| Cross-campagne détecté | `get_campaign_signals` sur les campagnes liées |

Maximum `max_extra_calls` appels supplémentaires par investigation.

### get_temporal_analysis — 9 analyses

1. Timeline posts/jour/compte
2. Co-occurrences multi-comptes
3. Cadence par compte
4. Évolution mensuelle avec deepfake
5. Propagation chronologique
6. Compte semence
7. Silences suspects (gaps > 14 jours)
8. Corrélation deepfake × temporel
9. Cross-campagne

### Suggestions de scrapping

Calculées en Python depuis `raw_data`. Trois niveaux :

- 🔴 **Priorité haute** : compte semence, copieurs massifs (≥ `copies_received_high`), amplificateurs sur ≥ `cross_campaign_high` campagnes
- 🟡 **Priorité moyenne** : hashtags à fort usage, cross-plateforme
- 🔵 **Surveillance continue** : comptes à cadence robotique, comptes avec doublons

### Graphes générés

| Graphe | Format | Contenu |
|---|---|---|
| `temporal.png` | PNG | Évolution mensuelle posts/comptes/deepfake |
| `propagation.png` | PNG | Scatter plot posts par compte dans le temps |
| `deepfake_distribution.png` | PNG | Histogramme des scores deepfake |
| `network.html` | HTML | Graphe réseau interactif (nœuds colorés par communauté) |

### Relations Neo4j utilisées

| Relation | Direction |
|---|---|
| `A_PUBLIÉ` | Account → Post |
| `APPARTIENT_À` | Post → Narrative |
| `HAS_HASHTAG` | Post → Hashtag |
| `EST_DOUBLON_DE` | Post → Post |
| `COUVRE` | Campaign → Narrative |

### Propriétés Neo4j clés

| Nœud | Propriétés |
|---|---|
| `Account` | `display_name`, `platform_id`, `community_id`, `pagerank_score`, `is_suspicious` |
| `Post` | `published_at`, `deepfake_score`, `is_synthetic`, `is_duplicate`, `sentiment_label` |
| `Narrative` | `label`, `keywords`, `post_count` |
| `Campaign` | `name`, `score`, `signals`, `platforms` |
| `Hashtag` | `name` |

---

## 8. Interprétation des rapports

### Scores deepfake

| Score | Prédiction | Interprétation |
|---|---|---|
| 0.00 – 0.45 | `likely_real` | Probablement authentique |
| 0.45 – 0.65 | `suspicious` | Ambigu — vérification recommandée |
| 0.65 – 1.00 | `synthetic` | Probablement généré par IA |

Pattern courant sur vidéos compressées : `sdxl-detector` élevé + `swinv2-openfake` bas = forte divergence = vidéo réelle compressée. Ne pas interpréter comme une alerte.

### Score de suspicion global

Valeur 0–1 calculée en Python depuis : score deepfake moyen, ratio synthétique, signaux des narratives, bonus si campagne ou médias réutilisés détectés.

### Niveau de confiance

- **Élevé** : divergence inter-modèles < 0.15
- **Moyen** : 0.15 – 0.30
- **Faible** : > 0.30 → vérification manuelle requise

### Requêtes Cypher dans les rapports

Chaque rapport inclut des requêtes adaptées aux vrais identifiants. Syntaxe Cypher correcte (pas de `GROUP BY` — utiliser `WITH + count() + RETURN`).

---

## 9. Glossaire

**Cadence robotique** : ≥ 1 post/jour sur > 20 jours consécutifs. Chez un humain la publication est irrégulière. Suggère une automatisation ou une gestion professionnelle du compte.

**Compte semence** : premier compte à avoir publié sur une narrative. Souvent la source qui déclenche la propagation. Priorité haute pour le scrapping.

**Co-occurrence temporelle** : plusieurs comptes publient le même jour sur la même narrative. Des co-occurrences répétées = signal de coordination fort.

**Silence suspect** : interruption > 14 jours pour un compte à cadence régulière. Peut indiquer une rotation de gestionnaire ou une suspension.

**Divergence inter-modèles** : écart entre les scores des trois modèles deepfake. > 0.30 = résultat peu fiable.

**Amplificateur cross-campagne** : compte actif sur plusieurs campagnes détectées. Peut indiquer une infrastructure partagée.

**PageRank** : score de centralité Neo4j GDS. PageRank élevé = compte central dans le réseau.

**Communauté Louvain** : groupe de comptes densément interconnectés (Neo4j GDS). Comptes d'une même communauté publiant sur la même narrative = signal de coordination structurel.

**Enrichissement adaptatif** : mécanisme par lequel l'agent déclenche des appels supplémentaires si les premières données révèlent des signaux forts. Limité par `max_extra_calls`.

**Données partielles** : le scrapping n'est pas exhaustif. Une absence de signal n'est pas une preuve d'absence de coordination.

---

## 10. Limites et précautions

**Disclaimer obligatoire.** Tout rapport nécessite une vérification humaine.

**Données partielles.** Les signaux peuvent être sous-estimés. Une campagne non détectée n'est pas nécessairement absente.

**Lecture seule stricte.** L'agent ne modifie jamais MongoDB ni Neo4j.

**Graphes optionnels.** Si `matplotlib` ou `pyvis` ne sont pas installés, les graphes sont ignorés — le rapport textuel est toujours généré.

**Qualité du LLM.** `llama3.1:8b` peut parfois reformuler maladroitement des valeurs. Pour de meilleurs rapports : Groq (`llama-3.1-70b-versatile`, gratuit) ou Anthropic.

**Calibration v4 en attente.** Les poids de SwinV2 dans `ai_forensics.cfg` sont à 0.50 (neutre) tant que la calibration n'a pas tourné. Performance réelle : F1=0.943.

**Relation campagne Neo4j.** Le chemin Account → Campaign passe par `Account → Post → Narrative → Campaign`. Pas de relation directe dans le schéma actuel.

---

*Module développé avec l'aide de Claude — Anthropic*  
*Telecom Paris — Mastère spécialisé Cyberdéfense / Cybersécurité*  
*Cours CYBER721 — Réseaux et Organisation de données : intelligence et sécurité*
