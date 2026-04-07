# Contexte — Agent d'investigation IA pour AI-FORENSICS
# Point de départ pour le développement du module AI/

---

## Objectif

Développer un agent d'investigation autonome (`investigation_agent.py`) qui s'interface avec l'infrastructure existante AI-FORENSICS (MongoDB + Neo4j) pour analyser automatiquement des comptes, médias, narratives ou campagnes suspects et produire un rapport d'analyse en Markdown.

C'est une **solution parallèle** — elle ne modifie aucun worker existant, elle vient se brancher en lecture sur les données déjà produites par le pipeline.

---

## Emplacement dans le dépôt

```
AI-FORENSICS/
    AI/
        investigation_agent.py    ← script principal (à créer)
        tools.py                  ← outils appelables par l'agent (à créer)
        prompts.py                ← prompts système et templates (à créer)
        requirements_ai.txt       ← dépendances spécifiques (à créer)
        README_AI.md              ← documentation (à créer)
    WORKER/                       ← workers existants (ne pas modifier)
    SCHEMA/                       ← schema.py (ne pas modifier)
    ...
```

---

## Infrastructure existante (à connaître)

### MongoDB

- Base : `influence_detection`
- ReplicaSet `rs0` initialisé (obligatoire pour Change Streams)
- Auth activée — credentials dans `.env`
- Connexion via `schema.py` : `from schema import get_db; db = get_db()`

**Collections disponibles en lecture :**

| Collection | Contenu utile pour l'agent |
|---|---|
| `accounts` | profil, stats, scores NLP, narratives associées, campaign_ids |
| `posts` | texte, deepfake.final_score, deepfake.scores (par modèle), nlp.sentiment, nlp.narrative_id |
| `media` | deepfake.final_score, deepfake.scores, deepfake.model_divergence, deepfake.prediction, reuse (seen_count, platforms) |
| `narratives` | label, keywords, stats.synthetic_ratio, stats.platforms, embedding_centroid |
| `campaigns` | signals (coordinated_posting, content_reuse, synthetic_media_ratio, cross_platform), review.confidence |
| `jobs` | file de traitement — pas utile pour l'agent |

**Structure deepfake dans un document media (exemple) :**
```json
"deepfake": {
    "final_score": 0.73,
    "prediction": "synthetic",
    "model_divergence": 0.12,
    "artifact_score": 0.41,
    "scores": {
        "Organika/sdxl-detector": 0.81,
        "swinv2_openfake": 0.68,
        "synthbuster/synthbuster": 0.61
    },
    "raw_scores": {
        "Organika/sdxl-detector": 0.94,
        "swinv2_openfake": 0.68,
        "synthbuster/synthbuster": 0.61
    }
}
```

### Neo4j

- Bolt : `bolt://localhost:7687`
- Credentials dans `.env`
- Plugin GDS installé (Louvain, PageRank déjà exécutés par network_worker.py)

**Nœuds disponibles :**
```
:Account   — platform, platform_id, uniqueId, name
:Post      — platform_id, platform, createTime, final_score
:Media     — url_local, type, final_score, prediction
:Hashtag   — tag
:Narrative — label, synthetic_ratio
:Campaign  — name, status, confidence
:Project   — name
```

**Relations disponibles :**
```
(:Account)-[:POSTED]->(:Post)
(:Post)-[:CONTAINS]->(:Media)
(:Post)-[:TAGGED]->(:Hashtag)
(:Post)-[:BELONGS_TO]->(:Narrative)
(:Account)-[:REPOSTED]->(:Post)
(:Media)-[:REUSED_BY]->(:Account)
(:Account)-[:MEMBER_OF]->(:Campaign)
```

---

## Architecture de l'agent

### Principe général

L'agent reçoit un point d'entrée (compte, post, narrative ou campagne), appelle une séquence d'outils pour collecter les données pertinentes, raisonne sur ces données, et produit un rapport Markdown structuré.

```
CLI (--account / --post / --narrative / --campaign)
    │
    ▼
investigation_agent.py
    │
    ├── Collecte via tools.py (lecture MongoDB + Neo4j)
    │     ├── get_account_info(platform, unique_id)
    │     ├── get_account_posts(account_id, limit)
    │     ├── get_media_scores(account_id)
    │     ├── get_graph_neighbors(platform_id)
    │     ├── get_narrative(narrative_id)
    │     └── get_campaign_signals(campaign_id)
    │
    ├── Agent LLM (Ollama Llama 3.1 8B par défaut)
    │     └── Raisonnement en boucle + appels d'outils
    │
    └── Rapport Markdown → reports/<timestamp>_<cible>.md
```

### Modèle LLM — configurable dans .env

```env
# Modèle par défaut : Ollama local
AI_PROVIDER=ollama
AI_MODEL=llama3.1:8b
AI_BASE_URL=http://localhost:11434

# Alternatives (commenter/décommenter)
# AI_PROVIDER=groq
# AI_MODEL=llama-3.1-70b-versatile
# AI_API_KEY=gsk_...

# AI_PROVIDER=anthropic
# AI_MODEL=claude-sonnet-4-20250514
# AI_API_KEY=sk-ant-...
```

### Framework agent

Utiliser **LlamaIndex** (orienté interrogation de bases de données, bien adapté à MongoDB + Neo4j) ou **LangChain** (plus généraliste, meilleure documentation pour les débutants). À décider en début de session selon les préférences.

---

## Outils à implémenter (tools.py)

Chaque outil est une fonction Python simple que l'agent peut appeler. Elles lisent uniquement en base — aucune écriture.

```python
def get_account_info(platform: str, unique_id: str) -> dict
    """Retourne le profil, les stats et les scores agrégés d'un compte."""

def get_account_posts(account_id: str, limit: int = 20) -> list
    """Retourne les derniers posts d'un compte avec leurs scores deepfake et NLP."""

def get_media_scores(account_id: str) -> list
    """Retourne les scores deepfake détaillés (par modèle) des médias d'un compte."""

def get_graph_neighbors(platform_id: str, depth: int = 2) -> dict
    """Retourne les nœuds et relations du voisinage dans Neo4j (comptes, hashtags, médias partagés)."""

def get_narrative(narrative_id: str) -> dict
    """Retourne les informations d'une narrative (label, mots-clés, ratio synthétique, plateformes)."""

def get_campaign_signals(campaign_id: str) -> dict
    """Retourne les signaux de campagne (coordination, réutilisation, ratio synthétique)."""

def search_accounts_by_narrative(narrative_id: str) -> list
    """Retourne les comptes associés à une narrative donnée."""
```

---

## Format du rapport de sortie

```markdown
# Rapport d'investigation — @<compte> (<plateforme>)
**Date :** 2026-04-06T14:32:00
**Score de suspicion global :** 0.78 / 1.0
**Niveau de confiance :** Élevé

---

## Synthèse
<paragraphe de synthèse en langage naturel>

## Analyse des médias
- Médias analysés : 42
- Médias synthétiques détectés : 31 (73.8%)
- Score moyen : 0.81
- Divergence inter-modèles moyenne : 0.09 (résultats fiables)

## Position dans le réseau
<analyse du graphe Neo4j — voisinage, centralité, communauté>

## Narratives portées
<narratives identifiées et leur ratio synthétique>

## Signaux de coordination
<comportements suspects identifiés>

## Conclusion
<qualification : campagne probable / viralité organique / insuffisant pour conclure>

## Recommandations
- <requêtes Cypher suggérées pour approfondir>
- <comptes à investiguer en priorité>

---
*Rapport généré automatiquement par AI-FORENSICS Investigation Agent*
*Modèle : llama3.1:8b via Ollama — Requires human review*
```

---

## Commandes CLI cibles

```bash
# Analyser un compte
python AI/investigation_agent.py --account cryptocom --platform instagram

# Analyser à partir d'un post
python AI/investigation_agent.py --post <post_id>

# Analyser une narrative
python AI/investigation_agent.py --narrative <narrative_id>

# Analyser une campagne détectée
python AI/investigation_agent.py --campaign <campaign_id>

# Options communes
python AI/investigation_agent.py --account cryptocom --platform instagram \
    --output ./reports/ \
    --model llama3.1:8b \
    --verbose
```

---

## Variables d'environnement à ajouter au .env existant

```env
# --- Agent IA ---
AI_PROVIDER=ollama          # ollama | groq | anthropic
AI_MODEL=llama3.1:8b
AI_BASE_URL=http://localhost:11434   # pour ollama uniquement
AI_API_KEY=                          # pour groq / anthropic
AI_REPORTS_DIR=./reports             # dossier de sortie des rapports
AI_MAX_TOKENS=4096
AI_TEMPERATURE=0.2                   # bas = raisonnement plus déterministe
```

---

## Dépendances spécifiques (requirements_ai.txt)

```
# Framework agent (choisir l'un ou l'autre)
llama-index>=0.10.0
# langchain>=0.2.0

# Clients LLM
ollama>=0.2.0
groq>=0.9.0                 # optionnel
anthropic>=0.25.0           # optionnel

# Déjà présentes dans requirements.txt principal (ne pas redéclarer)
# pymongo, python-dotenv, neo4j
```

---

## Points d'attention pour le développement

**Taille des résultats Neo4j.** Les requêtes de voisinage peuvent retourner des graphes très larges. Toujours limiter la profondeur (`depth=2` max) et le nombre de nœuds retournés. Résumer les résultats avant de les passer au LLM pour ne pas saturer la fenêtre de contexte.

**Qualité du function calling avec Ollama.** Llama 3.1 8B supporte le function calling mais est moins fiable que les modèles commerciaux sur les appels imbriqués. Prévoir une gestion des cas où l'agent appelle un outil avec des paramètres incorrects (try/except + message d'erreur retourné à l'agent).

**Pas d'écriture en base.** Les outils sont en lecture seule. Aucune écriture dans MongoDB ni Neo4j depuis l'agent — c'est une contrainte de conception à respecter strictement.

**Prompt système.** La qualité du raisonnement dépend fortement du prompt. Prévoir un fichier `prompts.py` séparé pour faciliter les itérations sans toucher au code de l'agent.

**Disclaimer obligatoire dans le rapport.** Tout rapport généré doit mentionner qu'il nécessite une vérification humaine. L'agent est un outil d'aide à l'analyse, pas un système de décision autonome.

---

## Lien avec l'infrastructure existante

- `schema.py` : importer `get_db` pour toutes les connexions MongoDB
- `.env` : ajouter les variables AI_ au fichier existant (pas de nouveau fichier .env)
- `storage/` : les rapports vont dans `AI-FORENSICS/reports/` (nouveau dossier, à ajouter au .gitignore)
- Aucun worker existant n'est modifié
- L'agent peut être lancé en parallèle des workers sans conflit (lecture seule)

---

## Références utiles

- LlamaIndex : https://docs.llamaindex.ai
- LangChain agents : https://python.langchain.com/docs/modules/agents
- Ollama Python : https://github.com/ollama/ollama-python
- Llama 3.1 function calling : https://ollama.com/library/llama3.1
- Groq API (gratuit) : https://console.groq.com
