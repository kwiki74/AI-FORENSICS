# Neo4j — Fiche d'analyse
## Détection de campagnes d'influence — AI Forensics Pipeline

---

## Modèle de graphe en production

| Nœud / Relation | Description | Propriétés clés |
|---|---|---|
| `(:Account)` | Compte sur une plateforme | `username`, `platform`, `pagerank_score`, `community_id`, `bot_score` |
| `(:Post)` | Publication | `platform`, `deepfake_score`, `is_synthetic`, `sentiment_label`, `published_at` |
| `(:Narrative)` | Cluster narratif (NLP) | `label`, `keywords`, `post_count` |
| `A_PUBLIÉ →` | Compte a publié un post | — |
| `A_COMMENTÉ →` | Compte a commenté un post | — |
| `A_FORWARDÉ →` | Forward Telegram | `count` (nb de forwards) |
| `EST_DOUBLON_DE →` | Post quasi-identique | `similarity_score` [0-1] |
| `APPARTIENT_À →` | Post rattaché à un narratif | — |

---

## 1. Connexion et premiers pas

**Accès au Browser Neo4j** : `http://localhost:7474`
Identifiants : `neo4j / influence2026!`

### Vérifier l'état du graphe

```cypher
-- Compter tous les nœuds par type
MATCH (n)
RETURN labels(n)[0] AS type, count(n) AS total
ORDER BY total DESC
```

```cypher
-- Compter toutes les relations
MATCH ()-[r]->()
RETURN type(r) AS relation, count(r) AS total
ORDER BY total DESC
```

```cypher
-- Vérifier les scores GDS disponibles
MATCH (a:Account)
WHERE a.pagerank_score IS NOT NULL
RETURN count(a) AS comptes_avec_pagerank
```

> 💡 Si `pagerank_score = 0` → la détection automatique n'a pas encore tourné. Lancer `network_worker.py --backfill`.

---

## 2. Vue d'ensemble — Requêtes de diagnostic

### Vue synthétique des campagnes par narratif

```cypher
MATCH (a:Account)-[:A_PUBLIÉ]->(p:Post)-[:APPARTIENT_À]->(n:Narrative)
RETURN
  n.label                       AS narratif,
  count(DISTINCT a)             AS nb_comptes,
  count(DISTINCT p)             AS nb_posts,
  collect(DISTINCT a.platform)  AS plateformes,
  avg(p.deepfake_score)         AS score_deepfake_moyen,
  sum(CASE WHEN p.is_synthetic THEN 1 ELSE 0 END) AS posts_synthetiques
ORDER BY nb_posts DESC
LIMIT 20
```

> 💡 C'est la requête de départ. Elle donne une vision macro de l'activité par narratif.

### Comptes les plus actifs cross-plateforme

```cypher
MATCH (a:Account)-[:A_PUBLIÉ]->(p:Post)
RETURN
  a.username                   AS compte,
  a.platform                   AS plateforme,
  count(p)                     AS nb_posts,
  count(DISTINCT p.platform)   AS nb_plateformes,
  a.pagerank_score             AS pagerank
ORDER BY nb_posts DESC
LIMIT 20
```

### Répartition des posts par plateforme

```cypher
MATCH (p:Post)
RETURN p.platform AS plateforme, count(p) AS total
ORDER BY total DESC
```

---

## 3. Analyse des comptes — GDS (PageRank & Louvain)

### Top comptes amplificateurs (PageRank)

Le **PageRank** mesure l'influence d'un compte dans le graphe. Un score élevé = compte très relayé par d'autres comptes influents.

```cypher
MATCH (a:Account)
WHERE a.pagerank_score IS NOT NULL
RETURN
  a.username       AS compte,
  a.platform       AS plateforme,
  a.followers      AS abonnés,
  a.pagerank_score AS pagerank,
  a.community_id   AS communauté,
  a.bot_score      AS score_bot
ORDER BY a.pagerank_score DESC
LIMIT 20
```

> ⚠️ Un PageRank élevé avec `bot_score > 0.7` est un signal fort de compte amplificateur artificiel.

### Comptes par communauté Louvain

Louvain regroupe les comptes qui publient sur les mêmes contenus. Chaque `community_id` identifie un cluster de comptes potentiellement coordonnés.

```cypher
MATCH (a:Account)
WHERE a.community_id IS NOT NULL
RETURN
  a.community_id               AS communauté,
  count(a)                     AS nb_comptes,
  collect(a.username)          AS comptes,
  collect(DISTINCT a.platform) AS plateformes
ORDER BY nb_comptes DESC
LIMIT 10
```

> 💡 Une communauté avec des comptes sur plusieurs plateformes (Instagram + Telegram + TikTok) est suspecte.

### Explorer une communauté spécifique

```cypher
-- Remplacer <ID> par le community_id trouvé ci-dessus
MATCH (a:Account {community_id: <ID>})-[:A_PUBLIÉ]->(p:Post)
      -[:APPARTIENT_À]->(n:Narrative)
RETURN
  a.username  AS compte,
  a.platform  AS plateforme,
  n.label     AS narratif,
  count(p)    AS nb_posts
ORDER BY nb_posts DESC
```

---

## 4. Analyse du contenu dupliqué

La relation `EST_DOUBLON_DE` est créée par le worker NLP quand le score de similarité cosinus dépasse **0.95**. C'est le signal le plus fort de coordination.

### Posts quasi-identiques cross-plateforme

```cypher
MATCH (a1:Account)-[:A_PUBLIÉ]->(p1:Post)
      -[:EST_DOUBLON_DE]->(p2:Post)
      <-[:A_PUBLIÉ]-(a2:Account)
WHERE a1.platform <> a2.platform
RETURN
  a1.username      AS compte_source,
  a1.platform      AS plateforme_source,
  a2.username      AS compte_cible,
  a2.platform      AS plateforme_cible,
  p1.published_at  AS date_publication,
  count(*)         AS nb_doublons
ORDER BY nb_doublons DESC
LIMIT 20
```

### Narratifs avec le plus de contenu recyclé

```cypher
MATCH (p1:Post)-[r:EST_DOUBLON_DE]->(p2:Post)
MATCH (p1)-[:APPARTIENT_À]->(n:Narrative)
RETURN
  n.label                AS narratif,
  count(r)               AS nb_doublons,
  avg(r.similarity_score) AS similarité_moyenne
ORDER BY nb_doublons DESC
LIMIT 10
```

### Chaîne de propagation d'un contenu

```cypher
-- Trouver le post original et tous ses doublons
MATCH path = (original:Post)<-[:EST_DOUBLON_DE*1..5]-(doublon:Post)
WHERE NOT (:Post)-[:EST_DOUBLON_DE]->(original)
MATCH (a)-[:A_PUBLIÉ]->(doublon)
RETURN
  original.platform    AS source,
  a.username           AS relayeur,
  a.platform           AS plateforme,
  doublon.published_at AS date,
  length(path)         AS distance
ORDER BY date
LIMIT 30
```

---

## 5. Détection des médias synthétiques (Deepfake)

### Posts avec score deepfake élevé

```cypher
MATCH (a:Account)-[:A_PUBLIÉ]->(p:Post)
WHERE p.deepfake_score >= 0.7
RETURN
  a.username        AS compte,
  a.platform        AS plateforme,
  p.deepfake_score  AS score_deepfake,
  p.deepfake_pred   AS prédiction,
  p.artifact_score  AS artefact_jpeg,
  p.published_at    AS date
ORDER BY p.deepfake_score DESC
LIMIT 30
```

### Ratio de médias synthétiques par narratif

```cypher
MATCH (p:Post)-[:APPARTIENT_À]->(n:Narrative)
WHERE p.deepfake_score IS NOT NULL
WITH n.label AS narratif,
     count(p) AS total,
     sum(CASE WHEN p.is_synthetic THEN 1 ELSE 0 END) AS synthetiques
RETURN
  narratif,
  total,
  synthetiques,
  round(100.0 * synthetiques / total, 1) AS ratio_pct
ORDER BY ratio_pct DESC
LIMIT 15
```

> ⚠️ Un ratio > 30% dans un narratif est un signal fort de campagne de désinformation.

### Comptes qui publient majoritairement du contenu synthétique

```cypher
MATCH (a:Account)-[:A_PUBLIÉ]->(p:Post)
WHERE p.deepfake_score IS NOT NULL
WITH a,
     count(p) AS total_posts,
     sum(CASE WHEN p.is_synthetic THEN 1 ELSE 0 END) AS posts_synthetiques
WHERE total_posts >= 3
RETURN
  a.username                                         AS compte,
  a.platform                                         AS plateforme,
  total_posts,
  posts_synthetiques,
  round(100.0 * posts_synthetiques / total_posts, 1) AS ratio_pct,
  a.pagerank_score                                   AS pagerank
ORDER BY ratio_pct DESC
LIMIT 15
```

---

## 6. Analyse temporelle — Bursts et coordination

### Volume de posts par jour et par narratif

```cypher
MATCH (p:Post)-[:APPARTIENT_À]->(n:Narrative)
WHERE p.published_at IS NOT NULL AND p.published_at <> ''
WITH n.label AS narratif,
     substring(p.published_at, 0, 10) AS jour,
     count(p) AS nb_posts
WHERE nb_posts >= 5
RETURN narratif, jour, nb_posts
ORDER BY nb_posts DESC
LIMIT 20
```

> 💡 Un pic de 20+ posts le même jour sur le même narratif = burst coordonné suspect.

### Comptes créés récemment avec forte activité

```cypher
MATCH (a:Account)-[:A_PUBLIÉ]->(p:Post)
WHERE a.created_at > '2025-01-01'
WITH a, count(p) AS nb_posts
WHERE nb_posts >= 5
RETURN
  a.username    AS compte,
  a.platform    AS plateforme,
  a.created_at  AS créé_le,
  a.followers   AS abonnés,
  nb_posts,
  a.pagerank_score AS pagerank
ORDER BY nb_posts DESC
LIMIT 15
```

> ⚠️ Compte créé après janvier 2025 avec fort PageRank et nombreux posts = profil de bot probable.

---

## 7. Visualisations graphiques dans le Browser

Pour obtenir un graphe visuel, la requête doit **retourner des nœuds et des relations** (pas juste des valeurs). Le Browser affiche automatiquement le graphe quand les résultats contiennent des nœuds.

### Graphe d'un compte suspect et ses connexions

```cypher
-- Remplacer 'nom_du_compte' par le username trouvé
MATCH (a:Account {username: 'nom_du_compte'})
      -[r]->(p:Post)
      -[:APPARTIENT_À]->(n:Narrative)
RETURN a, r, p, n
LIMIT 50
```

> 💡 Dans le Browser, double-cliquer sur un nœud `:Post` pour expand et voir d'autres comptes qui ont publié le même contenu.

### Graphe d'un narratif complet

```cypher
MATCH (a:Account)-[:A_PUBLIÉ]->(p:Post)-[:APPARTIENT_À]->(n:Narrative)
WHERE n.label CONTAINS 'bitcoin'
RETURN a, p, n
LIMIT 100
```

### Graphe des doublons — réseau de copié-collé

```cypher
MATCH (p1:Post)-[r:EST_DOUBLON_DE]->(p2:Post)
MATCH (a1)-[:A_PUBLIÉ]->(p1)
MATCH (a2)-[:A_PUBLIÉ]->(p2)
RETURN a1, p1, r, p2, a2
LIMIT 60
```

> 💡 Ce graphe est particulièrement révélateur : on voit visuellement les grappes de comptes qui copient le même contenu.

### Graphe cross-plateforme d'une campagne

```cypher
MATCH (a:Account)-[:A_PUBLIÉ]->(p:Post)-[:APPARTIENT_À]->(n:Narrative)
WITH n, collect(DISTINCT a.platform) AS plateformes
WHERE size(plateformes) >= 2
MATCH (a2:Account)-[:A_PUBLIÉ]->(p2:Post)-[:APPARTIENT_À]->(n)
RETURN a2, p2, n
LIMIT 80
```

### Astuces de visualisation dans le Browser

| Action | Comment faire |
|---|---|
| Changer la couleur des nœuds | Cliquer sur le label (`:Account`, `:Post`...) en bas → choisir une couleur |
| Dimensionner les nœuds par valeur | Cliquer sur le label → **Size** → choisir `pagerank_score` ou `deepfake_score` |
| Afficher une propriété sur les nœuds | Cliquer sur le label → **Caption** → choisir `username` ou `label` |
| Épingler un nœud | Double-cliquer puis glisser |
| Expand un nœud | Double-cliquer pour voir ses relations cachées |
| Exporter en PNG | Bouton téléchargement en bas à droite du graphe |

---

## 8. Investigation manuelle — Betweenness & BFS

Ces analyses se lancent depuis **Python** (`neo4j_client.py`), pas directement dans le Browser. Elles sont coûteuses en calcul et réservées à une investigation ciblée.

### Betweenness Centrality — trouver les ponts inter-communautés

Le Betweenness identifie les comptes qui servent de **ponts entre deux clusters**. Ce sont souvent les coordinateurs d'une campagne cross-plateforme.

```python
from neo4j_client import Neo4jClient

with Neo4jClient() as client:
    results = client.run_betweenness(top_n=20)
    for r in results:
        print(f"{r['username']:<25} {r['platform']:<12} "
              f"betweenness={r['betweenness_score']:.4f} "
              f"communauté={r['community_id']}")
```

> ⚠️ Peut prendre plusieurs minutes sur un grand graphe. Ne pas lancer pendant que le worker tourne.

### BFS — Tracer la propagation d'un contenu

```python
# Récupérer d'abord le mongo_id du post dans MongoDB
# db.posts.find_one({'platform': 'telegram', 'deepfake.prediction': 'synthetic'})

from neo4j_client import Neo4jClient

with Neo4jClient() as client:
    results = client.run_bfs('67e1234abc...')  # mongo_id du post source
    for r in results:
        print(f"Profondeur {r['depth']} — {r['username']} ({r['platform']})")
```

---

## 9. Requêtes d'investigation avancée

### Comptes qui agissent en réseau (même contenu, même jour)

```cypher
MATCH (a1:Account)-[:A_PUBLIÉ]->(p1:Post)-[:EST_DOUBLON_DE]->(p2:Post)
      <-[:A_PUBLIÉ]-(a2:Account)
WHERE a1 <> a2
  AND a1.community_id = a2.community_id
  AND substring(p1.published_at, 0, 10) = substring(p2.published_at, 0, 10)
RETURN
  a1.username                        AS compte1,
  a2.username                        AS compte2,
  a1.community_id                    AS communauté,
  substring(p1.published_at, 0, 10)  AS jour,
  count(*)                           AS actions_communes
ORDER BY actions_communes DESC
LIMIT 20
```

### Score de suspicion combiné par compte

```cypher
MATCH (a:Account)-[:A_PUBLIÉ]->(p:Post)
WITH a,
     count(p)                                          AS total_posts,
     avg(p.deepfake_score)                             AS avg_deepfake,
     sum(CASE WHEN p.is_synthetic THEN 1 ELSE 0 END)  AS posts_synthetiques,
     a.pagerank_score                                  AS pagerank,
     a.bot_score                                       AS bot_score
WHERE total_posts >= 3
RETURN
  a.username        AS compte,
  a.platform        AS plateforme,
  total_posts,
  round(100.0 * posts_synthetiques / total_posts, 1)  AS pct_synthetique,
  round(coalesce(avg_deepfake, 0), 3)                 AS deepfake_moyen,
  round(coalesce(pagerank, 0), 4)                     AS pagerank,
  round(coalesce(bot_score, 0), 2)                    AS bot_score
ORDER BY pct_synthetique DESC, pagerank DESC
LIMIT 20
```

> 💡 Cette requête combine tous les signaux pour identifier les comptes les plus suspects. À exporter pour revue manuelle.

### Vérifier les campagnes sauvegardées dans MongoDB

```python
from schema import get_db
db = get_db()

for c in db.campaigns.find({}, {'name': 1, 'review.confidence': 1, 'signals': 1, '_id': 0}) \
           .sort('review.confidence', -1):
    print(f"[{c['review']['confidence']:.2f}] {c['name']}")
    actifs = [k for k, v in c['signals'].items()
              if v and v is not False and v != 0]
    print(f"  Signaux: {', '.join(actifs)}")
```

---

## 10. Référence rapide — Seuils et interprétation

| Indicateur | Seuil d'alerte | Signification |
|---|---|---|
| `deepfake_score` | ≥ 0.70 | Média très probablement synthétique |
| `deepfake_score` | 0.50 – 0.69 | Suspect — vérification manuelle conseillée |
| `similarity_score` (doublon) | ≥ 0.95 | Contenu quasi-identique (copié-collé) |
| `similarity_score` (doublon) | 0.85 – 0.94 | Paraphrase proche — possible coordination |
| `pagerank_score` | Top 10% | Compte amplificateur clé du réseau |
| `bot_score` | ≥ 0.70 | Probable compte automatisé |
| `community_id` identique sur 3+ comptes | — | Cluster coordonné potentiel |
| Ratio doublons / total posts | > 10% | Signal `content_reuse` (campagne) |
| Ratio synthétiques / total posts | > 20% | Signal `synthetic_media` (campagne) |
| Même narratif sur ≥ 2 plateformes | — | Signal `cross_platform` (campagne) |
| ≥ 20 posts sur un narratif en 24h | — | Signal burst coordonné (campagne) |
| Confiance campagne | ≥ 0.55 | Campagne confirmée — 3+ signaux convergents |
| Confiance campagne | 0.30 – 0.54 | Suspicion — investigation recommandée |

---

*AI Forensics Pipeline — Usage interne — Mars 2026*
