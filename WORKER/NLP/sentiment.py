"""sentiment.py
==============

Module d'analyse de sentiment pour le pipeline NLP.
Détecte automatiquement la langue (FR / EN / autre) et applique
le modèle adapté. Normalise la sortie en positive / negative / neutral.

Modèles utilisés :
  FR : cmarkea/distilcamembert-base-sentiment  (5 labels → normalisés en 3)
  EN : cardiffnlp/twitter-roberta-base-sentiment-latest  (3 labels)

Détection de langue : lingua (plus fiable que langdetect sur textes courts)

Utilisation standalone (test sans MongoDB) :
    python sentiment.py

Utilisation dans le worker NLP :
    from sentiment import SentimentAnalyzer
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze("Le vote a été manipulé")
    # → {"label": "negative", "score": 0.94, "model": "distilcamembert", "lang": "fr"}

Dépendances (conda env nlp_pipeline) :
    pip install transformers torch accelerate lingua-language-detector
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mapping des labels vers le format normalisé du schéma MongoDB
# ---------------------------------------------------------------------------

# distilcamembert : 1 étoile, 2 étoiles, 3 étoiles, 4 étoiles, 5 étoiles
_CAMEMBERT_MAP: dict[str, str] = {
    "1 star":  "negative",
    "2 stars": "negative",
    "3 stars": "neutral",
    "4 stars": "positive",
    "5 stars": "positive",
}

# twitter-roberta : LABEL_0=negative, LABEL_1=neutral, LABEL_2=positive
_ROBERTA_MAP: dict[str, str] = {
    "LABEL_0":  "negative",
    "LABEL_1":  "neutral",
    "LABEL_2":  "positive",
    # versions récentes exposent parfois les noms directement
    "negative": "negative",
    "neutral":  "neutral",
    "positive": "positive",
}

# Noms courts pour le champ nlp.sentiment.model dans MongoDB
_MODEL_FR   = "distilcamembert"
_MODEL_EN   = "twitter-roberta"
_MODEL_NONE = "unknown"

# Identifiants HuggingFace
_HF_FR = "cmarkea/distilcamembert-base-sentiment"
_HF_EN = "cardiffnlp/twitter-roberta-base-sentiment-latest"


# ---------------------------------------------------------------------------
# Dataclass de résultat
# ---------------------------------------------------------------------------

@dataclass
class SentimentResult:
    label: str              # positive | negative | neutral
    score: float            # [0.0 – 1.0]
    model: str              # distilcamembert | twitter-roberta | unknown
    lang:  str              # fr | en | other
    skipped: bool = False   # True si texte vide ou langue non supportée


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """
    Analyse de sentiment multilingue FR / EN.

    Le premier appel à analyze() charge les modèles en mémoire (lazy loading).
    Les appels suivants réutilisent les pipelines déjà chargés.
    """

    def __init__(self, device: str = "auto") -> None:
        """
        Args:
            device: "auto" → GPU si disponible, sinon CPU.
                    "cpu"  → force le CPU.
                    "cuda" → force le GPU (lève une erreur si absent).
        """
        self._device  = self._resolve_device(device)
        self._pipe_fr = None   # pipeline HuggingFace FR (lazy)
        self._pipe_en = None   # pipeline HuggingFace EN (lazy)
        self._lingua  = None   # détecteur de langue (lazy)

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Traduit "auto" en "cuda" ou "cpu" selon disponibilité."""
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    # ------------------------------------------------------------------
    # Chargement lazy des ressources
    # ------------------------------------------------------------------

    def _load_lingua(self) -> None:
        if self._lingua is not None:
            return
        from lingua import Language, LanguageDetectorBuilder
        self._lingua = (
            LanguageDetectorBuilder
            .from_languages(Language.FRENCH, Language.ENGLISH)
            .with_minimum_relative_distance(0.1)
            .build()
        )
        logger.info("Lingua chargé (FR + EN)")

    def _load_fr(self) -> None:
        if self._pipe_fr is not None:
            return
        from transformers import pipeline
        logger.info("Chargement modèle FR : %s", _HF_FR)
        self._pipe_fr = pipeline(
            "text-classification",
            model=_HF_FR,
            device=self._device,
            truncation=True,
            max_length=512,
        )
        logger.info("Modèle FR prêt")

    def _load_en(self) -> None:
        if self._pipe_en is not None:
            return
        from transformers import pipeline
        logger.info("Chargement modèle EN : %s", _HF_EN)
        self._pipe_en = pipeline(
            "text-classification",
            model=_HF_EN,
            device=self._device,
            truncation=True,
            max_length=512,
        )
        logger.info("Modèle EN prêt")

    # ------------------------------------------------------------------
    # Détection de langue
    # ------------------------------------------------------------------

    def detect_lang(self, text: str) -> str:
        """
        Retourne "fr", "en", ou "other".
        Charge lingua au premier appel.
        """
        self._load_lingua()
        from lingua import Language
        lang = self._lingua.detect_language_of(text)
        if lang == Language.FRENCH:
            return "fr"
        if lang == Language.ENGLISH:
            return "en"
        return "other"

    # ------------------------------------------------------------------
    # Analyse principale
    # ------------------------------------------------------------------

    def analyze(self, text: str, lang: Optional[str] = None) -> SentimentResult:
        """
        Analyse le sentiment d'un texte.

        Args:
            text : texte brut (post ou commentaire).
            lang : langue forcée ("fr" / "en"). Si None, détectée automatiquement.

        Returns:
            SentimentResult avec label normalisé, score, modèle et langue.
        """
        # --- Texte vide ou trop court ---
        text = (text or "").strip()
        if len(text) < 3:
            logger.debug("Texte trop court, skipped")
            return SentimentResult(
                label="neutral", score=0.0,
                model=_MODEL_NONE, lang="other", skipped=True
            )

        # --- Détection de langue ---
        if lang is None:
            lang = self.detect_lang(text)

        # --- Inférence selon la langue ---
        try:
            if lang == "fr":
                return self._analyze_fr(text)
            elif lang == "en":
                return self._analyze_en(text)
            else:
                # Langue non supportée : on tente EN par défaut (meilleure couverture)
                logger.debug("Langue '%s' non supportée, tentative EN", lang)
                result = self._analyze_en(text)
                result.lang = lang   # on garde la langue réelle détectée
                return result

        except Exception as exc:
            logger.error("Erreur inférence sentiment (%s) : %s", lang, exc)
            return SentimentResult(
                label="neutral", score=0.0,
                model=_MODEL_NONE, lang=lang, skipped=True
            )

    def _analyze_fr(self, text: str) -> SentimentResult:
        self._load_fr()
        raw = self._pipe_fr(text)[0]
        label = _CAMEMBERT_MAP.get(raw["label"], "neutral")
        return SentimentResult(
            label=label,
            score=round(float(raw["score"]), 4),
            model=_MODEL_FR,
            lang="fr",
        )

    def _analyze_en(self, text: str) -> SentimentResult:
        self._load_en()
        raw = self._pipe_en(text)[0]
        label = _ROBERTA_MAP.get(raw["label"], "neutral")
        return SentimentResult(
            label=label,
            score=round(float(raw["score"]), 4),
            model=_MODEL_EN,
            lang="en",
        )

    # ------------------------------------------------------------------
    # Batch (liste de textes)
    # ------------------------------------------------------------------

    def analyze_batch(
        self,
        texts: list[str],
        langs: Optional[list[str]] = None,
    ) -> list[SentimentResult]:
        """
        Analyse une liste de textes.
        Plus efficace que d'appeler analyze() en boucle pour de grands volumes.

        Args:
            texts : liste de textes.
            langs : langues forcées (même longueur que texts). None = auto-détection.
        """
        if langs is None:
            langs = [None] * len(texts)

        results = []
        for text, lang in zip(texts, langs):
            results.append(self.analyze(text, lang=lang))
        return results

    # ------------------------------------------------------------------
    # Conversion vers le format patch_post_nlp / patch_comment_nlp
    # ------------------------------------------------------------------

    def to_mongo_fields(self, result: SentimentResult) -> dict:
        """
        Retourne uniquement les champs sentiment compatibles avec schema.py.
        À fusionner dans l'appel patch_post_nlp() ou patch_comment_nlp().

        Exemple :
            fields = analyzer.to_mongo_fields(result)
            patch = patch_post_nlp(
                sentiment_label = fields["label"],
                sentiment_score = fields["score"],
                sentiment_model = fields["model"],
                embedding_model = "...",   # rempli par le worker embeddings
                topics          = [],
            )
        """
        return {
            "label": result.label,
            "score": result.score,
            "model": result.model,
        }


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    analyzer = SentimentAnalyzer(device="auto")

    tests = [
        # FR — négatif
        "Le vote a été manipulé par le gouvernement",
        "Le scrutin est frauduleux, c'est un scandale",
        "L'immigration est hors de contrôle, c'est catastrophique",
        # FR — positif
        "Excellente journée, les résultats sont très encourageants",
        "Je suis très satisfait de cette décision",
        # FR — neutre
        "J'ai fait du sport aujourd'hui",
        "Le soleil brille, je prépare le dîner",
        # EN — négatif
        "The election was rigged, this is outrageous",
        "Security experts warn about escalating threats",
        # EN — positif
        "Great news, the economy is improving significantly",
        # EN — neutre
        "Watching a movie tonight",
        # Court / vide
        "",
        "ok",
    ]

    print("\n" + "="*70)
    print(f"{'Texte':<45} {'Lang':<6} {'Label':<10} {'Score':<7} {'Modèle'}")
    print("="*70)

    for text in tests:
        r = analyzer.analyze(text)
        display = (text[:42] + "...") if len(text) > 45 else text
        skipped = " [skip]" if r.skipped else ""
        print(f"{display:<45} {r.lang:<6} {r.label:<10} {r.score:<7.4f} {r.model}{skipped}")

    print("="*70)
