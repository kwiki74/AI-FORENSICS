"""neo4j_client.py
==================

Module Neo4j pour le worker réseau.
Gère la connexion, la création des nœuds/relations,
et les requêtes d'analyse GDS.

Modèle de graphe :
    Nœuds :
        (:Account)   — compte sur une plateforme
        (:Post)      — publication
        (:Narrative) — cluster narratif (issu du NLP)
        (:Hashtag)   — hashtag coordonné              [v3]
        (:Deepfake)  — type de média synthétique      [v3]
        (:Media)     — fichier média (vidéo/image)    [v4]
        (:Campaign)  — campagne d'influence détectée  [v5]

    Relations :
        (:Account)-[:A_PUBLIÉ]->(:Post)
        (:Account)-[:A_COMMENTÉ]->(:Post)
        (:Account)-[:A_FORWARDÉ {count}]->(:Post)      Telegram
        (:Post)-[:EST_DOUBLON_DE]->(:Post)              NLP déduplication
        (:Post)-[:APPARTIENT_À]->(:Narrative)           NLP clustering
        (:Post)-[:HAS_HASHTAG]->(:Hashtag)              Hashtags coordonnés  [v3]
        (:Post)-[:IS_DEEPFAKE {score}]->(:Deepfake)     Médias synthétiques  [v3]
        (:Post)-[:A_MEDIA]->(:Media)                    Fichiers médias      [v4]
        (:Campaign)-[:COUVRE]->(:Narrative)             Campagne → narratif  [v5]
        (:Campaign)-[:IMPLIQUE]->(:Account)             Campagne → compte    [v5]
        (:Account)-[:PARTICIPE_À]->(:Campaign)          Compte → campagne    [v5]
        (:Project)-[:CONTIENT]->(:Post)                 Projet → post        [v5]
        (:Project)-[:CONTIENT]->(:Account)              Projet → compte      [v5]

Algorithmes GDS disponibles :
    Automatiques (appelés par campaign_detector) :
        run_louvain()    — détection de communautés de comptes coordonnés
        run_pagerank()   — comptes amplificateurs clés

    Ponctuels / investigation manuelle :
        run_betweenness()           — ponts inter-communautés
        run_bfs(source_mongo_id)    — propagation d'un contenu spécifique

Utilisation standalone (test) :
    python neo4j_client.py

Dépendances :
    pip install neo4j
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Nom du graphe projeté en mémoire GDS
# Un seul graphe à la fois — on drop avant de recréer
_GDS_GRAPH_NAME = "coordination_graph"


# ---------------------------------------------------------------------------
# Client Neo4j
# ---------------------------------------------------------------------------

class Neo4jClient:
    """
    Wrapper autour du driver Neo4j officiel.
    Gère la connexion, les contraintes, et toutes les requêtes Cypher/GDS.
    """

    def __init__(
        self,
        uri:      str = "bolt://localhost:7687",
        user:     str = "neo4j",
        password: str = "influence2026!",
    ) -> None:
        from neo4j import GraphDatabase
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver.verify_connectivity()
        logger.info("Neo4j connecté : %s", uri)

    def close(self) -> None:
        self._driver.close()
        logger.info("Neo4j déconnecté")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Initialisation du schéma (contraintes + index)
    # ------------------------------------------------------------------

    def create_constraints(self) -> None:
        """
        Crée les contraintes d'unicité et les index.
        Idempotent — peut être appelé à chaque démarrage.
        """
        constraints = [
            """CREATE CONSTRAINT account_unique IF NOT EXISTS
               FOR (a:Account) REQUIRE (a.platform, a.platform_id) IS UNIQUE""",
            """CREATE CONSTRAINT post_unique IF NOT EXISTS
               FOR (p:Post) REQUIRE p.mongo_id IS UNIQUE""",
            """CREATE CONSTRAINT narrative_unique IF NOT EXISTS
               FOR (n:Narrative) REQUIRE n.mongo_id IS UNIQUE""",
            # Nœuds v3
            """CREATE CONSTRAINT hashtag_unique IF NOT EXISTS
               FOR (h:Hashtag) REQUIRE h.name IS UNIQUE""",
            """CREATE CONSTRAINT deepfake_unique IF NOT EXISTS
               FOR (d:Deepfake) REQUIRE d.type IS UNIQUE""",
            # Nœuds v4
            """CREATE CONSTRAINT media_unique IF NOT EXISTS
               FOR (m:Media) REQUIRE m.mongo_id IS UNIQUE""",
            # Nœuds v5
            """CREATE CONSTRAINT campaign_unique IF NOT EXISTS
               FOR (c:Campaign) REQUIRE c.mongo_id IS UNIQUE""",
            """CREATE CONSTRAINT project_unique IF NOT EXISTS
               FOR (p:Project) REQUIRE p.name IS UNIQUE""",
        ]

        indexes = [
            "CREATE INDEX account_platform  IF NOT EXISTS FOR (a:Account)  ON (a.platform)",
            "CREATE INDEX post_platform     IF NOT EXISTS FOR (p:Post)     ON (p.platform)",
            "CREATE INDEX post_published    IF NOT EXISTS FOR (p:Post)     ON (p.published_at)",
            "CREATE INDEX post_sentiment    IF NOT EXISTS FOR (p:Post)     ON (p.sentiment_label)",
            "CREATE INDEX account_community IF NOT EXISTS FOR (a:Account)  ON (a.community_id)",
            "CREATE INDEX account_pagerank  IF NOT EXISTS FOR (a:Account)  ON (a.pagerank_score)",
            "CREATE INDEX hashtag_name      IF NOT EXISTS FOR (h:Hashtag)  ON (h.name)",
            "CREATE INDEX deepfake_type     IF NOT EXISTS FOR (d:Deepfake) ON (d.type)",
            "CREATE INDEX media_type        IF NOT EXISTS FOR (m:Media)    ON (m.type)",
            "CREATE INDEX media_deepfake    IF NOT EXISTS FOR (m:Media)    ON (m.deepfake_pred)",
            # Index v5
            "CREATE INDEX campaign_score    IF NOT EXISTS FOR (c:Campaign) ON (c.score)",
            "CREATE INDEX project_name      IF NOT EXISTS FOR (p:Project)  ON (p.name)",
        ]

        with self._driver.session() as session:
            for query in constraints + indexes:
                try:
                    session.run(query)
                except Exception as exc:
                    logger.debug("Contrainte/index déjà existant : %s", exc)

        logger.info("Contraintes et index Neo4j initialisés")

    # ------------------------------------------------------------------
    # Nœuds
    # ------------------------------------------------------------------

    def upsert_account(self, account: dict) -> None:
        query = """
        MERGE (a:Account {platform: $platform, platform_id: $platform_id})
        SET a.mongo_id      = $mongo_id,
            a.username      = $username,
            a.display_name  = $display_name,
            a.verified      = $verified,
            a.followers     = $followers,
            a.bot_score     = $bot_score,
            a.updated_at    = $updated_at
        """
        with self._driver.session() as session:
            session.run(query, **account)

    def upsert_post(self, post: dict) -> None:
        query = """
        MERGE (p:Post {mongo_id: $mongo_id})
        SET p.platform          = $platform,
            p.platform_id       = $platform_id,
            p.published_at      = $published_at,
            p.sentiment_label   = $sentiment_label,
            p.sentiment_score   = $sentiment_score,
            p.deepfake_score    = $deepfake_score,
            p.is_synthetic      = $is_synthetic,
            p.narrative_id      = $narrative_id,
            p.like_count        = $like_count,
            p.comment_count     = $comment_count,
            p.share_count       = $share_count,
            p.view_count        = $view_count,
            p.influence_score   = $influence_score,
            p.is_bot_suspected  = $is_bot_suspected,
            p.music_author      = $music_author,
            p.cover_url         = $cover_url,
            p.source_project    = $source_project,
            p.source_scan       = $source_scan,
            p.updated_at        = $updated_at
        """
        with self._driver.session() as session:
            session.run(query, **post)

    def upsert_narrative(self, narrative: dict) -> None:
        query = """
        MERGE (n:Narrative {mongo_id: $mongo_id})
        SET n.label       = $label,
            n.keywords    = $keywords,
            n.post_count  = $post_count,
            n.updated_at  = $updated_at
        """
        with self._driver.session() as session:
            session.run(query, **narrative)

    # ------------------------------------------------------------------
    # Relations
    # ------------------------------------------------------------------

    def link_account_post(
        self,
        account_mongo_id: str,
        post_mongo_id:    str,
        rel_type:         str = "A_PUBLIÉ",
        props:            Optional[dict] = None,
    ) -> None:
        query = f"""
        MATCH (a:Account {{mongo_id: $account_id}})
        MATCH (p:Post    {{mongo_id: $post_id}})
        MERGE (a)-[r:{rel_type}]->(p)
        """
        if props:
            set_clause = ", ".join(f"r.{k} = ${k}" for k in props)
            query += f"\nSET {set_clause}"

        params = {"account_id": account_mongo_id, "post_id": post_mongo_id}
        if props:
            params.update(props)

        with self._driver.session() as session:
            session.run(query, **params)

    def link_post_duplicate(
        self,
        post_mongo_id:     str,
        original_mongo_id: str,
        similarity_score:  float,
    ) -> None:
        query = """
        MATCH (p:Post {mongo_id: $post_id})
        MATCH (o:Post {mongo_id: $original_id})
        MERGE (p)-[r:EST_DOUBLON_DE]->(o)
        SET r.similarity_score = $score
        """
        with self._driver.session() as session:
            session.run(query,
                post_id     = post_mongo_id,
                original_id = original_mongo_id,
                score       = similarity_score,
            )

    def link_post_narrative(
        self,
        post_mongo_id:      str,
        narrative_mongo_id: str,
    ) -> None:
        query = """
        MATCH (p:Post      {mongo_id: $post_id})
        MATCH (n:Narrative {mongo_id: $narrative_id})
        MERGE (p)-[:APPARTIENT_À]->(n)
        """
        with self._driver.session() as session:
            session.run(query,
                post_id      = post_mongo_id,
                narrative_id = narrative_mongo_id,
            )

    # ------------------------------------------------------------------
    # Nœuds Media  [v4]
    # ------------------------------------------------------------------

    def upsert_media(self, media: dict) -> None:
        """
        Crée ou met à jour un nœud :Media.

        Propriétés clés :
            mongo_id, type (video/image/…), url_local,
            deepfake_score, deepfake_pred, reuse_count
        """
        query = """
        MERGE (m:Media {mongo_id: $mongo_id})
        SET m.type           = $type,
            m.url_local      = $url_local,
            m.url_original   = $url_original,
            m.deepfake_score = $deepfake_score,
            m.deepfake_pred  = $deepfake_pred,
            m.reuse_count    = $reuse_count,
            m.platform       = $platform,
            m.updated_at     = $updated_at
        """
        with self._driver.session() as session:
            session.run(query, **media)

    def upsert_media_batch(self, medias: list[dict]) -> int:
        """Version batch de upsert_media — utilisée par le backfill."""
        if not medias:
            return 0
        query = """
        UNWIND $rows AS row
        MERGE (m:Media {mongo_id: row.mongo_id})
        SET m += row
        """
        with self._driver.session() as session:
            session.run(query, rows=medias)
        return len(medias)

    def link_post_media(self, post_id: str, media_id: str) -> None:
        """
        Crée la relation (:Post)-[:A_MEDIA]->(:Media).
        Les deux nœuds doivent exister (upsert préalable).
        """
        query = """
        MATCH (p:Post  {mongo_id: $post_id})
        MATCH (m:Media {mongo_id: $media_id})
        MERGE (p)-[:A_MEDIA]->(m)
        """
        with self._driver.session() as session:
            session.run(query, post_id=post_id, media_id=media_id)

    # ------------------------------------------------------------------
    # Nœuds Project  [v5]
    # ------------------------------------------------------------------

    def upsert_project(self, name: str) -> None:
        """Crée ou met à jour un nœud :Project {name}."""
        with self._driver.session() as session:
            session.run(
                "MERGE (p:Project {name: $name})",
                name=name,
            )

    def link_project_post(self, project_name: str, post_mongo_id: str) -> None:
        """(:Project)-[:CONTIENT]->(:Post)"""
        with self._driver.session() as session:
            session.run(
                """
                MATCH (proj:Project {name: $name})
                MATCH (p:Post       {mongo_id: $post_id})
                MERGE (proj)-[:CONTIENT]->(p)
                """,
                name    = project_name,
                post_id = post_mongo_id,
            )

    def link_project_account(self, project_name: str, account_mongo_id: str) -> None:
        """(:Project)-[:CONTIENT]->(:Account)"""
        with self._driver.session() as session:
            session.run(
                """
                MATCH (proj:Project {name: $name})
                MATCH (a:Account    {mongo_id: $account_id})
                MERGE (proj)-[:CONTIENT]->(a)
                """,
                name       = project_name,
                account_id = account_mongo_id,
            )

    def upsert_project_batch(
        self,
        project_posts: list[dict],
        project_accounts: list[dict],
    ) -> None:
        """
        Version batch — upsert Project + liens vers Posts et Accounts.

        project_posts    : [{"project": str, "post_id": str}, …]
        project_accounts : [{"project": str, "account_id": str}, …]
        """
        with self._driver.session() as session:
            # Upsert tous les projets distincts
            projects = list({d["project"] for d in project_posts + project_accounts})
            if projects:
                session.run(
                    """
                    UNWIND $names AS name
                    MERGE (:Project {name: name})
                    """,
                    names=projects,
                )

            # Liens Project → Post
            if project_posts:
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (proj:Project {name: row.project})
                    MATCH (p:Post       {mongo_id: row.post_id})
                    MERGE (proj)-[:CONTIENT]->(p)
                    """,
                    rows=project_posts,
                )

            # Liens Project → Account
            if project_accounts:
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (proj:Project {name: row.project})
                    MATCH (a:Account    {mongo_id: row.account_id})
                    MERGE (proj)-[:CONTIENT]->(a)
                    """,
                    rows=project_accounts,
                )

    # ------------------------------------------------------------------
    # Nœuds Campaign  [v5]
    # ------------------------------------------------------------------

    def upsert_campaign(self, campaign: dict) -> None:
        """
        Crée ou met à jour un nœud :Campaign.

        Propriétés :
            mongo_id, name, score, platforms, signals (liste), signal_count
        """
        query = """
        MERGE (c:Campaign {mongo_id: $mongo_id})
        SET c.name         = $name,
            c.score        = $score,
            c.platforms    = $platforms,
            c.signals      = $signals,
            c.signal_count = $signal_count
        """
        with self._driver.session() as session:
            session.run(
                query,
                mongo_id     = campaign["mongo_id"],
                name         = campaign.get("name", ""),
                score        = float(campaign.get("score", 0.0)),
                platforms    = campaign.get("platforms", []),
                signals      = campaign.get("signals", []),
                signal_count = int(campaign.get("signal_count", 0)),
            )

    def link_campaign_narrative(
        self,
        campaign_mongo_id: str,
        narrative_mongo_id: str,
    ) -> None:
        """(:Campaign)-[:COUVRE]->(:Narrative)"""
        query = """
        MATCH (c:Campaign  {mongo_id: $campaign_id})
        MATCH (n:Narrative {mongo_id: $narrative_id})
        MERGE (c)-[:COUVRE]->(n)
        """
        with self._driver.session() as session:
            session.run(
                query,
                campaign_id  = campaign_mongo_id,
                narrative_id = narrative_mongo_id,
            )

    def link_campaign_account(
        self,
        campaign_mongo_id: str,
        account_mongo_id: str,
    ) -> None:
        """
        (:Campaign)-[:IMPLIQUE]->(:Account)
        (:Account)-[:PARTICIPE_À]->(:Campaign)
        Les deux relations sont créées ensemble.
        """
        query = """
        MATCH (c:Campaign {mongo_id: $campaign_id})
        MATCH (a:Account  {mongo_id: $account_id})
        MERGE (c)-[:IMPLIQUE]->(a)
        MERGE (a)-[:PARTICIPE_À]->(c)
        """
        with self._driver.session() as session:
            session.run(
                query,
                campaign_id = campaign_mongo_id,
                account_id  = account_mongo_id,
            )

    # ------------------------------------------------------------------
    # Purge complète  [v3]
    # ------------------------------------------------------------------

    def purge_all(self) -> None:
        """
        Supprime TOUS les nœuds et relations de la base Neo4j.

        Utilisé par network_worker --projet (sans --add) avant réinjection.
        Traitement par lots de 10 000 nœuds pour éviter un OutOfMemory
        sur les grosses bases.
        """
        with self._driver.session() as session:
            session.run("""
                CALL {
                    MATCH (n)
                    DETACH DELETE n
                } IN TRANSACTIONS OF 10000 ROWS
            """)
        logger.info("Purge Neo4j terminée — base vide")

    # ------------------------------------------------------------------
    # Nœuds Hashtag  [v3]
    # ------------------------------------------------------------------

    def upsert_hashtags_for_post(self, post_id: str, hashtags: list[str]) -> None:
        """
        Crée ou met à jour les nœuds :Hashtag et les relie au Post.

        Relation créée :
            (:Post {mongo_id})-[:HAS_HASHTAG]->(:Hashtag {name})

        Les hashtags sont normalisés (minuscules, sans '#' initial) pour
        éviter les doublons dus à la casse ou au caractère '#'.
        """
        clean = [t.lstrip("#").lower().strip() for t in hashtags if t.strip()]
        clean = list(dict.fromkeys(clean))   # dédoublonnage ordre-préservant
        if not clean:
            return

        with self._driver.session() as session:
            session.run(
                """
                MATCH (p:Post {mongo_id: $post_id})
                UNWIND $tags AS tag
                MERGE (h:Hashtag {name: tag})
                MERGE (p)-[:HAS_HASHTAG]->(h)
                """,
                post_id=post_id,
                tags=clean,
            )

    # ------------------------------------------------------------------
    # Nœuds Deepfake  [v3]
    # ------------------------------------------------------------------

    def upsert_deepfake_node(
        self,
        post_id:   str,
        pred_type: str,
        score:     float,
    ) -> None:
        """
        Crée ou met à jour un nœud :Deepfake et le relie au Post.

        Relation créée :
            (:Post {mongo_id})-[:IS_DEEPFAKE {score}]->(:Deepfake {type})

        Un seul nœud :Deepfake par type (MERGE sur type) — plusieurs posts
        peuvent pointer vers le même nœud. La relation porte le score individuel.

        pred_type : deepfake.prediction  (ex: "synthetic", "suspicious")
        score     : deepfake.final_score [0.0 – 1.0]
        """
        with self._driver.session() as session:
            session.run(
                """
                MATCH (p:Post {mongo_id: $post_id})
                MERGE (d:Deepfake {type: $pred_type})
                MERGE (p)-[r:IS_DEEPFAKE]->(d)
                SET r.score = $score
                """,
                post_id=post_id,
                pred_type=pred_type,
                score=float(score),
            )

    # ------------------------------------------------------------------
    # Batch upsert
    # ------------------------------------------------------------------

    def upsert_accounts_batch(self, accounts: list[dict]) -> int:
        if not accounts:
            return 0
        query = """
        UNWIND $rows AS row
        MERGE (a:Account {platform: row.platform, platform_id: row.platform_id})
        SET a += row
        """
        with self._driver.session() as session:
            session.run(query, rows=accounts)
        return len(accounts)

    def upsert_posts_batch(self, posts: list[dict]) -> int:
        if not posts:
            return 0
        query = """
        UNWIND $rows AS row
        MERGE (p:Post {mongo_id: row.mongo_id})
        SET p += row
        """
        with self._driver.session() as session:
            session.run(query, rows=posts)
        return len(posts)

    def create_relations_batch(
        self,
        relations:  list[dict],
        rel_type:   str,
        from_label: str = "Account",
        to_label:   str = "Post",
        from_key:   str = "mongo_id",
        to_key:     str = "mongo_id",
    ) -> int:
        if not relations:
            return 0
        query = f"""
        UNWIND $rows AS row
        MATCH (a:{from_label} {{{from_key}: row.from_id}})
        MATCH (b:{to_label}   {{{to_key}:   row.to_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += row.props
        """
        with self._driver.session() as session:
            session.run(query, rows=relations)
        return len(relations)

    # ------------------------------------------------------------------
    # Vérification GDS
    # ------------------------------------------------------------------

    def _check_gds(self) -> bool:
        """Retourne True si le plugin GDS est disponible."""
        try:
            with self._driver.session() as session:
                session.run("RETURN gds.version() AS v")
            return True
        except Exception:
            logger.warning(
                "Plugin GDS non disponible. "
                "Installer neo4j-graph-data-science pour activer les algos réseau."
            )
            return False

    def _drop_graph(self, session, graph_name: str = _GDS_GRAPH_NAME) -> None:
        """Supprime un graphe GDS projeté s'il existe."""
        try:
            exists = session.run(
                "CALL gds.graph.exists($name) YIELD exists",
                name=graph_name
            ).single()["exists"]
            if exists:
                session.run("CALL gds.graph.drop($name)", name=graph_name)
                logger.debug("Graphe GDS '%s' supprimé", graph_name)
        except Exception as exc:
            logger.debug("Impossible de supprimer le graphe GDS '%s' : %s", graph_name, exc)

    def _project_account_graph(self, session, graph_name: str = _GDS_GRAPH_NAME) -> bool:
        """
        Projette le graphe Account↔Account en mémoire GDS.

        Stratégie : projection Cypher qui matérialise les liens implicites
        entre comptes via les posts partagés. Deux comptes sont reliés s'ils
        ont publié, commenté ou forwardé le même post.

        Retourne True si la projection a réussi.
        """
        # Nœuds : tous les comptes
        node_query = "MATCH (a:Account) RETURN id(a) AS id"

        # Relations : paires de comptes liés par un post commun
        # id(a1) < id(a2) évite les doublons (A↔B et B↔A)
        rel_query = """
        MATCH (a1:Account)-[:A_PUBLIÉ|A_COMMENTÉ|A_FORWARDÉ]->(p:Post)
              <-[:A_PUBLIÉ|A_COMMENTÉ|A_FORWARDÉ]-(a2:Account)
        WHERE id(a1) < id(a2)
        RETURN id(a1) AS source, id(a2) AS target
        """

        try:
            result = session.run(
                """
                CALL gds.graph.project.cypher(
                    $name,
                    $node_query,
                    $rel_query
                )
                YIELD graphName, nodeCount, relationshipCount
                RETURN nodeCount, relationshipCount
                """,
                name=graph_name,
                node_query=node_query,
                rel_query=rel_query,
            ).single()

            if result:
                logger.info(
                    "Graphe GDS '%s' projeté : %d comptes, %d liens",
                    graph_name, result["nodeCount"], result["relationshipCount"]
                )
                return result["nodeCount"] > 0
            return False

        except Exception as exc:
            logger.error("Erreur projection GDS : %s", exc)
            return False

    # ------------------------------------------------------------------
    # GDS — Louvain (détection de communautés)  [AUTOMATIQUE]
    # ------------------------------------------------------------------

    def run_louvain(self) -> list[dict]:
        """
        Détecte les communautés de comptes coordonnés via l'algorithme de Louvain.

        Retourne une liste de dicts :
            [{"mongo_id": str, "platform": str, "username": str, "community_id": int}, ...]

        Écrit aussi le community_id directement sur les nœuds Account dans Neo4j
        via write_community_ids() — les comptes du même cluster ont le même ID.

        Nécessite le plugin GDS.
        """
        if not self._check_gds():
            return []

        results = []
        with self._driver.session() as session:
            # Nettoyage préventif si un graphe traîne d'un run précédent
            self._drop_graph(session)

            if not self._project_account_graph(session):
                logger.warning("Louvain ignoré : graphe vide ou projection échouée")
                return []

            try:
                # stream = ne modifie pas le graphe, retourne les résultats
                rows = session.run(
                    """
                    CALL gds.louvain.stream($name)
                    YIELD nodeId, communityId
                    WITH gds.util.asNode(nodeId) AS node, communityId
                    WHERE node:Account
                    RETURN node.mongo_id  AS mongo_id,
                           node.platform  AS platform,
                           node.username  AS username,
                           communityId
                    ORDER BY communityId, node.username
                    """,
                    name=_GDS_GRAPH_NAME,
                )
                results = [dict(r) for r in rows]

                # Compter les communautés distinctes
                community_ids = {r["communityId"] for r in results}
                logger.info(
                    "Louvain terminé : %d comptes dans %d communautés",
                    len(results), len(community_ids)
                )

            except Exception as exc:
                logger.error("Erreur Louvain : %s", exc)
            finally:
                self._drop_graph(session)

        # Écriture des community_id sur les nœuds Account
        if results:
            self.write_community_ids(results)

        return results

    def write_community_ids(self, louvain_results: list[dict]) -> None:
        """
        Écrit le community_id sur chaque nœud Account dans Neo4j.
        Appelé automatiquement par run_louvain().

        Args:
            louvain_results : liste retournée par run_louvain()
        """
        if not louvain_results:
            return

        # Batch UNWIND pour performance
        rows = [
            {"mongo_id": r["mongo_id"], "community_id": r["communityId"]}
            for r in louvain_results
            if r.get("mongo_id")
        ]

        if not rows:
            return

        with self._driver.session() as session:
            session.run(
                """
                UNWIND $rows AS row
                MATCH (a:Account {mongo_id: row.mongo_id})
                SET a.community_id = row.community_id
                """,
                rows=rows,
            )
        logger.info("community_id écrit sur %d nœuds Account", len(rows))

    # ------------------------------------------------------------------
    # GDS — PageRank (comptes amplificateurs)  [AUTOMATIQUE]
    # ------------------------------------------------------------------

    def run_pagerank(self, top_n: int = 50) -> list[dict]:
        """
        Calcule le PageRank de chaque compte pour identifier les amplificateurs clés.

        Un score élevé = compte très cité/relayé dans le graphe.
        Utile pour prioriser l'investigation humaine.

        Args:
            top_n : nombre de comptes à retourner (les plus influents)

        Retourne :
            [{"mongo_id": str, "platform": str, "username": str,
              "pagerank_score": float, "community_id": int|None}, ...]

        Écrit aussi pagerank_score sur les nœuds Account dans Neo4j.
        Nécessite le plugin GDS.
        """
        if not self._check_gds():
            return []

        results = []
        with self._driver.session() as session:
            self._drop_graph(session)

            if not self._project_account_graph(session):
                logger.warning("PageRank ignoré : graphe vide ou projection échouée")
                return []

            try:
                rows = session.run(
                    """
                    CALL gds.pageRank.stream($name, {
                        dampingFactor: 0.85,
                        maxIterations: 20,
                        tolerance: 0.0000001
                    })
                    YIELD nodeId, score
                    WITH gds.util.asNode(nodeId) AS node, score
                    WHERE node:Account
                    RETURN node.mongo_id    AS mongo_id,
                           node.platform    AS platform,
                           node.username    AS username,
                           node.community_id AS community_id,
                           score            AS pagerank_score
                    ORDER BY score DESC
                    LIMIT $top_n
                    """,
                    name=_GDS_GRAPH_NAME,
                    top_n=top_n,
                )
                results = [dict(r) for r in rows]
                logger.info(
                    "PageRank terminé : top %d comptes amplificateurs calculés",
                    len(results)
                )

            except Exception as exc:
                logger.error("Erreur PageRank : %s", exc)
            finally:
                self._drop_graph(session)

        # Écriture des scores sur les nœuds
        if results:
            self._write_pagerank_scores(results)

        return results

    def _write_pagerank_scores(self, pagerank_results: list[dict]) -> None:
        """Écrit le pagerank_score sur les nœuds Account."""
        rows = [
            {"mongo_id": r["mongo_id"], "score": r["pagerank_score"]}
            for r in pagerank_results
            if r.get("mongo_id")
        ]
        if not rows:
            return
        with self._driver.session() as session:
            session.run(
                """
                UNWIND $rows AS row
                MATCH (a:Account {mongo_id: row.mongo_id})
                SET a.pagerank_score = row.score
                """,
                rows=rows,
            )
        logger.info("pagerank_score écrit sur %d nœuds Account", len(rows))

    # ------------------------------------------------------------------
    # GDS — Betweenness Centrality (ponts inter-communautés) [PONCTUEL]
    # ------------------------------------------------------------------

    def run_betweenness(self, top_n: int = 20) -> list[dict]:
        """
        Identifie les comptes "ponts" entre communautés — souvent les coordinateurs
        d'une campagne cross-plateforme.

        ⚠ COÛTEUX en calcul — réserver à des runs ponctuels sur demande,
          pas dans la boucle automatique.

        Args:
            top_n : nombre de comptes à retourner (les plus centraux)

        Retourne :
            [{"mongo_id": str, "platform": str, "username": str,
              "betweenness_score": float, "community_id": int|None}, ...]

        Nécessite le plugin GDS.
        """
        if not self._check_gds():
            return []

        logger.info(
            "Betweenness Centrality — calcul en cours (top %d)… "
            "[peut être lent sur un grand graphe]",
            top_n
        )

        results = []
        with self._driver.session() as session:
            self._drop_graph(session)

            if not self._project_account_graph(session):
                logger.warning("Betweenness ignoré : graphe vide")
                return []

            try:
                rows = session.run(
                    """
                    CALL gds.betweenness.stream($name)
                    YIELD nodeId, score
                    WITH gds.util.asNode(nodeId) AS node, score
                    WHERE node:Account
                    RETURN node.mongo_id     AS mongo_id,
                           node.platform     AS platform,
                           node.username     AS username,
                           node.community_id AS community_id,
                           score             AS betweenness_score
                    ORDER BY score DESC
                    LIMIT $top_n
                    """,
                    name=_GDS_GRAPH_NAME,
                    top_n=top_n,
                )
                results = [dict(r) for r in rows]
                logger.info(
                    "Betweenness terminé : %d comptes ponts identifiés",
                    len(results)
                )

            except Exception as exc:
                logger.error("Erreur Betweenness : %s", exc)
            finally:
                self._drop_graph(session)

        return results

    # ------------------------------------------------------------------
    # GDS — BFS (propagation d'un contenu)  [PONCTUEL / INVESTIGATION]
    # ------------------------------------------------------------------

    def run_bfs(self, source_mongo_id: str) -> list[dict]:
        """
        Trace le chemin de propagation d'un post depuis sa source.

        Utile pour un analyste qui veut visualiser comment un contenu
        spécifique s'est diffusé dans le réseau.

        ⚠ Usage ponctuel uniquement — pas dans la boucle automatique.

        Args:
            source_mongo_id : mongo_id du Post source

        Retourne :
            [{"mongo_id": str, "username": str, "platform": str,
              "depth": int}, ...]
          Les comptes qui ont relayé le contenu, triés par profondeur.

        Nécessite le plugin GDS.
        """
        if not self._check_gds():
            return []

        logger.info("BFS propagation depuis post %s…", source_mongo_id)

        # Récupérer l'id interne Neo4j du post source
        with self._driver.session() as session:
            source_row = session.run(
                "MATCH (p:Post {mongo_id: $mid}) RETURN id(p) AS node_id",
                mid=source_mongo_id,
            ).single()

            if not source_row:
                logger.warning("Post source introuvable en Neo4j : %s", source_mongo_id)
                return []

            source_node_id = source_row["node_id"]

        # Projection orientée pour suivre la direction de propagation
        # On inclut Account ET Post pour tracer les chemins complets
        bfs_graph = f"bfs_{source_mongo_id[:8]}"
        results   = []

        with self._driver.session() as session:
            # Drop si existe déjà
            self._drop_graph(session, bfs_graph)

            try:
                # Projection avec toutes les relations de diffusion
                session.run(
                    """
                    CALL gds.graph.project(
                        $name,
                        ['Account', 'Post'],
                        {
                            A_PUBLIÉ:   {orientation: 'NATURAL'},
                            A_FORWARDÉ: {orientation: 'NATURAL'},
                            A_COMMENTÉ: {orientation: 'NATURAL'}
                        }
                    )
                    """,
                    name=bfs_graph,
                )

                rows = session.run(
                    """
                    CALL gds.bfs.stream($name, {
                        sourceNode: $source
                    })
                    YIELD nodeId, path
                    WITH gds.util.asNode(nodeId) AS node, size(nodes(path)) - 1 AS depth
                    WHERE node:Account
                    RETURN node.mongo_id  AS mongo_id,
                           node.username  AS username,
                           node.platform  AS platform,
                           depth
                    ORDER BY depth, node.platform
                    """,
                    name=bfs_graph,
                    source=source_node_id,
                )
                results = [dict(r) for r in rows]
                logger.info(
                    "BFS terminé : %d comptes atteignables depuis %s",
                    len(results), source_mongo_id
                )

            except Exception as exc:
                logger.error("Erreur BFS : %s", exc)
            finally:
                self._drop_graph(session, bfs_graph)

        return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Retourne les compteurs de nœuds et relations."""
        with self._driver.session() as session:
            counts = {}
            for label in ("Account", "Post", "Narrative", "Hashtag", "Deepfake", "Media", "Campaign", "Project"):
                r = session.run(f"MATCH (n:{label}) RETURN count(n) AS c")
                counts[label] = r.single()["c"]
            for rel in ("A_PUBLIÉ", "A_COMMENTÉ", "A_FORWARDÉ",
                        "EST_DOUBLON_DE", "APPARTIENT_À",
                        "HAS_HASHTAG", "IS_DEEPFAKE", "A_MEDIA",
                        "COUVRE", "IMPLIQUE", "PARTICIPE_À",
                        "CONTIENT"):
                r = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c")
                counts[rel] = r.single()["c"]

            # Stats GDS si disponible
            try:
                r = session.run(
                    "MATCH (a:Account) WHERE a.community_id IS NOT NULL "
                    "RETURN count(DISTINCT a.community_id) AS communities, "
                    "count(a) AS accounts_with_community"
                )
                row = r.single()
                counts["gds_communities"]            = row["communities"]
                counts["accounts_with_community_id"] = row["accounts_with_community"]
            except Exception:
                pass

        return counts


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("\nTest de connexion Neo4j…")
    try:
        with Neo4jClient() as client:
            client.create_constraints()

            # Données de test
            for i in range(1, 6):
                client.upsert_account({
                    "mongo_id":     f"test_acc_{i:03d}",
                    "platform":     "twitter",
                    "platform_id":  f"user_{i:03d}",
                    "username":     f"test_user_{i}",
                    "display_name": f"Test User {i}",
                    "verified":     False,
                    "followers":    100 * i,
                    "bot_score":    None,
                    "updated_at":   "2026-03-23T00:00:00Z",
                })

            for i in range(1, 4):
                client.upsert_post({
                    "mongo_id":        f"test_post_{i:03d}",
                    "platform":        "twitter",
                    "platform_id":     f"tweet_{i:03d}",
                    "published_at":    "2026-03-23T10:00:00Z",
                    "sentiment_label": "negative",
                    "sentiment_score": 0.85,
                    "deepfake_score":  None,
                    "is_synthetic":    False,
                    "narrative_id":    None,
                    "updated_at":      "2026-03-23T10:00:00Z",
                })

            # Relations : comptes 1-4 ont tous posté le même post → cluster coordonné
            for i in range(1, 5):
                client.link_account_post(f"test_acc_{i:03d}", "test_post_001", "A_PUBLIÉ")
            # Comptes 2-3 partagent aussi post_002
            client.link_account_post("test_acc_002", "test_post_002", "A_PUBLIÉ")
            client.link_account_post("test_acc_003", "test_post_002", "A_PUBLIÉ")
            # Compte 5 isolé
            client.link_account_post("test_acc_005", "test_post_003", "A_PUBLIÉ")

            # Stats avant GDS
            stats = client.get_stats()
            print("\n=== Stats Neo4j ===")
            for k, v in stats.items():
                print(f"  {k:<35} : {v}")

            # Test Louvain
            print("\n=== Test Louvain ===")
            louvain = client.run_louvain()
            if louvain:
                for r in louvain:
                    print(f"  {r['username']:<20} community_id={r['communityId']}")
            else:
                print("  GDS non disponible ou graphe vide")

            # Test PageRank
            print("\n=== Test PageRank (top 5) ===")
            pr = client.run_pagerank(top_n=5)
            if pr:
                for r in pr:
                    print(f"  {r['username']:<20} pagerank={r['pagerank_score']:.4f}")
            else:
                print("  GDS non disponible ou graphe vide")

            # Nettoyage
            with client._driver.session() as s:
                s.run("MATCH (n) WHERE n.mongo_id STARTS WITH 'test_' DETACH DELETE n")
            print("\nTest réussi ✓ — données de test supprimées")

    except Exception as e:
        print(f"\nErreur : {e}")
        print("Vérifier que Neo4j tourne : sudo systemctl status neo4j")
