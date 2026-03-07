"""
Vector Memory — embedding-based storage for similarity retrieval.

Stores:
  * Face embeddings (recognised people)
  * Object embeddings (learned objects)
  * Concept embeddings (abstract ideas, topics)
  * Text embeddings (past utterances, facts)

Characteristics:
  * Persisted to disk (numpy .npz archive)
  * Cosine-similarity search
  * Bounded capacity per category with LRU eviction
  * Thread-safe
  * Model-agnostic — embeddings are opaque float vectors
  * Supports future swap to FAISS, Annoy, or any ANN backend
"""

import os
import time
import threading
from typing import Any

import numpy as np

# Capacity per category
DEFAULT_CAPACITY = 500


class VectorMemory:
    """Embedding store with cosine-similarity retrieval.

    Each stored item is a dict:
        {
          "id":        str,        # unique key
          "category":  str,        # "face" | "object" | "concept" | "text"
          "label":     str,        # human-readable label
          "embedding": np.ndarray, # 1-D float vector
          "metadata":  dict,       # arbitrary extra data
          "timestamp": float,
        }
    """

    def __init__(self, store_dir: str, capacity: int = DEFAULT_CAPACITY):
        self._file = os.path.join(store_dir, "vectors.npz")
        self._meta_file = os.path.join(store_dir, "vectors_meta.npy")
        self._capacity = capacity
        self._lock = threading.Lock()

        # In-memory store: {id → item_dict}
        self._items: dict[str, dict] = {}
        self._load()

    # ── Store ────────────────────────────────────────────────────

    def store(
        self,
        item_id: str,
        category: str,
        label: str,
        embedding: np.ndarray,
        metadata: dict | None = None,
    ) -> None:
        """Add or overwrite a vector in the store."""
        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        item = {
            "id": item_id,
            "category": category,
            "label": label,
            "embedding": embedding,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        with self._lock:
            self._items[item_id] = item
            self._enforce_capacity()
        self._save()

    # ── Retrieve ─────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        category: str | None = None,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[dict]:
        """Find the *top_k* most similar items by cosine similarity.

        Returns list of dicts with an extra ``score`` field, sorted
        descending by score.
        """
        query = np.asarray(query_embedding, dtype=np.float32).flatten()
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        with self._lock:
            candidates = list(self._items.values())

        if category:
            candidates = [c for c in candidates if c["category"] == category]

        scored = []
        for item in candidates:
            emb = item["embedding"]
            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0:
                continue
            score = float(np.dot(query, emb) / (query_norm * emb_norm))
            if score >= threshold:
                scored.append({
                    "id": item["id"],
                    "category": item["category"],
                    "label": item["label"],
                    "metadata": item["metadata"],
                    "score": round(score, 4),
                    "timestamp": item["timestamp"],
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get_by_id(self, item_id: str) -> dict | None:
        with self._lock:
            item = self._items.get(item_id)
        if item is None:
            return None
        return {
            "id": item["id"],
            "category": item["category"],
            "label": item["label"],
            "metadata": item["metadata"],
            "timestamp": item["timestamp"],
        }

    def get_by_category(self, category: str) -> list[dict]:
        """List all items in a category (without raw embeddings)."""
        with self._lock:
            items = [
                {
                    "id": v["id"],
                    "category": v["category"],
                    "label": v["label"],
                    "metadata": v["metadata"],
                    "timestamp": v["timestamp"],
                }
                for v in self._items.values()
                if v["category"] == category
            ]
        return items

    def delete(self, item_id: str) -> bool:
        with self._lock:
            removed = self._items.pop(item_id, None) is not None
        if removed:
            self._save()
        return removed

    # ── Stats ────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._items)

    def category_counts(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {}
            for v in self._items.values():
                cat = v["category"]
                counts[cat] = counts.get(cat, 0) + 1
        return counts

    # ── Capacity management ──────────────────────────────────────

    def _enforce_capacity(self) -> None:
        """Remove oldest items if over capacity. Must hold lock."""
        if len(self._items) <= self._capacity:
            return
        # Sort by timestamp ascending, remove oldest
        sorted_ids = sorted(
            self._items, key=lambda k: self._items[k]["timestamp"]
        )
        to_remove = len(self._items) - self._capacity
        for item_id in sorted_ids[:to_remove]:
            del self._items[item_id]

    # ── Persistence ──────────────────────────────────────────────

    def _save(self) -> None:
        """Save embeddings + metadata to disk."""
        with self._lock:
            if not self._items:
                return
            ids = list(self._items.keys())
            embeddings = np.array(
                [self._items[k]["embedding"] for k in ids], dtype=np.float32
            )
            meta = [
                {
                    "id": self._items[k]["id"],
                    "category": self._items[k]["category"],
                    "label": self._items[k]["label"],
                    "metadata": self._items[k]["metadata"],
                    "timestamp": self._items[k]["timestamp"],
                }
                for k in ids
            ]
        try:
            np.savez_compressed(self._file, embeddings=embeddings)
            np.save(self._meta_file, meta, allow_pickle=True)
        except Exception as e:
            print(f"[VectorMemory] Save failed: {e}")

    def _load(self) -> None:
        """Load embeddings + metadata from disk."""
        if not os.path.exists(self._meta_file):
            return
        try:
            meta = np.load(self._meta_file, allow_pickle=True).tolist()
            npz_path = self._file
            if not npz_path.endswith(".npz"):
                npz_path += ".npz"
            if not os.path.exists(npz_path):
                return
            data = np.load(npz_path)
            embeddings = data["embeddings"]

            with self._lock:
                for i, m in enumerate(meta):
                    self._items[m["id"]] = {
                        "id": m["id"],
                        "category": m["category"],
                        "label": m["label"],
                        "embedding": embeddings[i].astype(np.float32),
                        "metadata": m.get("metadata", {}),
                        "timestamp": m.get("timestamp", 0),
                    }
            print(f"[VectorMemory] Loaded {len(meta)} vectors from disk.")
        except Exception as e:
            print(f"[VectorMemory] Load failed: {e}")
