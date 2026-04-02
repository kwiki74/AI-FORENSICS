"""embeddings.py
================

Module de calcul d'embeddings pour le pipeline NLP.
Utilise paraphrase-multilingual-MiniLM-L12-v2 (384d, multilingue).

Stockage :
    Les vecteurs sont sérialisés en float16 (bytes) avant stockage MongoDB.
    float16 → moitié moins de place que float32, perte négligeable sur cosine similarity.
    384 dims × 2 octets = 768 octets par document (≈ 0.75 Ko)

Déduplication :
    Cosine similarity entre le nouveau vecteur et les vecteurs existants en base.
    Si similarité > seuil (défaut 0.95) → is_duplicate_of = ObjectId du post original.

Limitation connue (à documenter dans le CR) :
    Le clustering HDBSCAN charge TOUS les embeddings en RAM.
    Limite pratique estimée à ~300 000 documents (16 Go RAM).
    Au-delà → migration vers Qdrant (base vectorielle dédiée).

Utilisation standalone (test sans MongoDB) :
    python embeddings.py

Utilisation dans le worker NLP :
    from embeddings import EmbeddingEngine
    engine = EmbeddingEngine()
    result = engine.embed("Le vote a été manipulé")
    # → EmbeddingResult(vector=..., vector_bytes=..., model=..., dim=384)

Dépendances (conda env nlp_pipeline) :
    pip install sentence-transformers numpy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Modèle retenu : multilingue, 384d, ~120 Mo
# Justification dans le CR : bon équilibre qualité/poids pour textes courts RS
_HF_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Seuil de déduplication par défaut
_DEFAULT_DEDUP_THRESHOLD = 0.95

# Dtype de stockage — float16 = moitié moins de RAM que float32
_DTYPE = np.float16


# ---------------------------------------------------------------------------
# Dataclass de résultat
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingResult:
    vector:       np.ndarray   # vecteur float16, shape (384,)
    vector_bytes: bytes        # sérialisé pour stockage MongoDB
    model:        str          # nom court du modèle
    dim:          int          # dimension (384)
    skipped:      bool = False # True si texte vide


# ---------------------------------------------------------------------------
# Moteur d'embedding
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """
    Calcule les embeddings et détecte les doublons sémantiques.

    Lazy loading : le modèle se charge au premier appel à embed().
    """

    def __init__(
        self,
        model_name: str = _HF_MODEL,
        dedup_threshold: float = _DEFAULT_DEDUP_THRESHOLD,
        device: str = "auto",
    ) -> None:
        self._model_name       = model_name
        self._dedup_threshold  = dedup_threshold
        self._device           = self._resolve_device(device)
        self._model            = None   # lazy

        logger.info(
            "EmbeddingEngine initialisé (model=%s, dedup_threshold=%.2f, device=%s)",
            model_name, dedup_threshold, self._device,
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    # ------------------------------------------------------------------
    # Chargement lazy du modèle
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info("Chargement modèle embedding : %s", self._model_name)
        self._model = SentenceTransformer(self._model_name, device=self._device)
        logger.info(
            "Modèle embedding prêt (dim=%d)", self._model.get_sentence_embedding_dimension()
        )

    # ------------------------------------------------------------------
    # Sérialisation / désérialisation
    # ------------------------------------------------------------------

    @staticmethod
    def to_bytes(vector: np.ndarray) -> bytes:
        """Sérialise un vecteur numpy en bytes pour stockage MongoDB."""
        return vector.astype(_DTYPE).tobytes()

    @staticmethod
    def from_bytes(data: bytes, dim: int = 384) -> np.ndarray:
        """Désérialise des bytes MongoDB en vecteur numpy float16."""
        return np.frombuffer(data, dtype=_DTYPE).reshape(dim)

    # ------------------------------------------------------------------
    # Calcul d'embedding
    # ------------------------------------------------------------------

    def embed(self, text: str) -> EmbeddingResult:
        """
        Calcule l'embedding d'un texte.

        Args:
            text : texte brut (post ou commentaire).

        Returns:
            EmbeddingResult avec vecteur float16 et bytes sérialisés.
        """
        text = (text or "").strip()

        # Texte vide ou trop court
        if len(text) < 3:
            dim = 384
            empty = np.zeros(dim, dtype=_DTYPE)
            return EmbeddingResult(
                vector=empty,
                vector_bytes=empty.tobytes(),
                model=self._model_name,
                dim=dim,
                skipped=True,
            )

        self._load()

        vector = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,   # normalisation L2 → cosine = dot product
            show_progress_bar=False,
        ).astype(_DTYPE)

        return EmbeddingResult(
            vector=vector,
            vector_bytes=self.to_bytes(vector),
            model=self._model_name,
            dim=len(vector),
        )

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """
        Calcule les embeddings d'une liste de textes en batch.
        Plus efficace que d'appeler embed() en boucle.
        """
        self._load()
        results = []
        # Sépare les textes valides des vides
        valid_indices = [i for i, t in enumerate(texts) if len((t or "").strip()) >= 3]
        valid_texts   = [texts[i].strip() for i in valid_indices]

        if valid_texts:
            vectors = self._model.encode(
                valid_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=64,
                show_progress_bar=len(valid_texts) > 100,
            ).astype(_DTYPE)
        else:
            vectors = np.array([])

        dim = self._model.get_sentence_embedding_dimension()
        vec_iter = iter(vectors)

        for i, text in enumerate(texts):
            if i in valid_indices:
                v = next(vec_iter)
                results.append(EmbeddingResult(
                    vector=v,
                    vector_bytes=self.to_bytes(v),
                    model=self._model_name,
                    dim=dim,
                ))
            else:
                empty = np.zeros(dim, dtype=_DTYPE)
                results.append(EmbeddingResult(
                    vector=empty,
                    vector_bytes=empty.tobytes(),
                    model=self._model_name,
                    dim=dim,
                    skipped=True,
                ))
        return results

    # ------------------------------------------------------------------
    # Déduplication
    # ------------------------------------------------------------------

    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Similarité cosinus entre deux vecteurs normalisés.
        Comme on utilise normalize_embeddings=True, c'est juste un dot product.
        """
        return float(np.dot(v1.astype(np.float32), v2.astype(np.float32)))

    def find_duplicate(
        self,
        vector: np.ndarray,
        candidates: list[tuple],  # liste de (doc_id, vector_bytes)
        threshold: Optional[float] = None,
    ) -> tuple[Optional[object], Optional[float]]:
        """
        Cherche un doublon parmi une liste de candidats.

        Args:
            vector     : vecteur du document à tester (float16, normalisé)
            candidates : liste de (doc_id, vector_bytes) issus de MongoDB
            threshold  : seuil de similarité (défaut : self._dedup_threshold)

        Returns:
            (doc_id, score) du doublon le plus proche si trouvé, sinon (None, None)
        """
        if not candidates:
            return None, None

        thresh = threshold if threshold is not None else self._dedup_threshold
        best_id    = None
        best_score = 0.0

        for doc_id, vec_bytes in candidates:
            if not vec_bytes:
                continue
            try:
                candidate_vec = self.from_bytes(vec_bytes, dim=len(vector))
                score = self.cosine_similarity(vector, candidate_vec)
                if score > best_score:
                    best_score = score
                    best_id    = doc_id
            except Exception as exc:
                logger.debug("Erreur déduplication doc %s : %s", doc_id, exc)
                continue

        if best_score >= thresh:
            return best_id, round(best_score, 4)
        return None, None


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    engine = EmbeddingEngine(device="auto")

    # --- Test embed simple ---
    tests = [
        "Le vote a été manipulé par le gouvernement",
        "Le scrutin est frauduleux",
        "L'élection a été truquée",
        "J'aime les chats",
        "Le soleil brille aujourd'hui",
        "The election was rigged",
        "",
    ]

    print("\n" + "="*65)
    print(f"{'Texte':<45} {'Dim':<6} {'Bytes':<8} {'Skip'}")
    print("="*65)

    results = []
    for text in tests:
        r = engine.embed(text)
        display = (text[:42] + "...") if len(text) > 45 else text
        print(f"{display:<45} {r.dim:<6} {len(r.vector_bytes):<8} {'oui' if r.skipped else 'non'}")
        results.append((text, r))

    # --- Test déduplication ---
    print("\n" + "="*65)
    print("TEST DÉDUPLICATION (seuil=0.80)")
    print("="*65)

    # Corpus de référence : les 3 premiers textes
    corpus = [(f"id_{i}", r.vector_bytes) for i, (_, r) in enumerate(results[:3])]

    for text, r in results:
        if r.skipped:
            continue
        dup_id, dup_score = engine.find_duplicate(r.vector, corpus, threshold=0.80)
        display = (text[:42] + "...") if len(text) > 45 else text
        if dup_id is not None:
            print(f"  {display:<45} → doublon de {dup_id} (score={dup_score:.4f})")
        else:
            print(f"  {display:<45} → original")

    print("="*65)

    # --- Taille mémoire ---
    n = 300_000
    size_mo = (n * 384 * 2) / (1024 * 1024)
    print(f"\nEstimation mémoire pour {n:,} documents : {size_mo:.0f} Mo")
    print(f"  → float16 (actuel) : {size_mo:.0f} Mo")
    print(f"  → float32          : {size_mo*2:.0f} Mo")
