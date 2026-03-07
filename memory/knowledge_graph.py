"""
Knowledge Graph — structured entity-relationship memory.

Stores knowledge as a directed graph of entities and relationships:

    Entity("Python", type="language", properties={"paradigm": "multi"})
      ──[is_used_for]──▶  Entity("web development", type="domain")
      ──[created_by]──▶   Entity("Guido van Rossum", type="person")

This complements the flat fact list in SemanticMemory by giving
the AI structured, queryable knowledge it can reason over.

Features:
  * Named entities with types and properties
  * Directed, typed relationships between entities
  * Confidence scores on relationships (0..1)
  * Entity merging (deduplication by normalized name)
  * Subgraph extraction (neighbors, paths)
  * LLM-friendly context export
  * Persistent storage via compressed_storage
"""

import os
import time
from threading import Lock
from collections import defaultdict

from .compressed_storage import save_json, load_json, migrate_if_needed


MAX_ENTITIES = 500
MAX_RELATIONS = 2000


class KnowledgeGraph:
    """Directed graph of entities and typed relationships."""

    def __init__(self, store_dir: str):
        self._file = os.path.join(store_dir, "knowledge_graph.json")
        self._lock = Lock()

        migrate_if_needed(self._file)
        data = load_json(self._file, default={
            "entities": {},
            "relations": [],
        })

        # entities: {normalized_name: {name, type, properties, created, updated}}
        self._entities: dict[str, dict] = data.get("entities", {})
        # relations: [{source, target, relation, confidence, created}]
        self._relations: list[dict] = data.get("relations", [])

        # Index: source_key -> list of relation indices
        self._outgoing: dict[str, list[int]] = defaultdict(list)
        self._incoming: dict[str, list[int]] = defaultdict(list)
        self._rebuild_indices()

    # ── Entity operations ────────────────────────────────────────

    @staticmethod
    def _normalize(name: str) -> str:
        return name.strip().lower()

    def add_entity(self, name: str, entity_type: str = "thing",
                   properties: dict | None = None) -> str:
        """Add or update an entity. Returns the normalized key."""
        key = self._normalize(name)
        if not key:
            return ""
        now = time.time()
        with self._lock:
            if key in self._entities:
                ent = self._entities[key]
                ent["updated"] = now
                if entity_type and entity_type != "thing":
                    ent["type"] = entity_type
                if properties:
                    ent["properties"].update(properties)
            else:
                self._entities[key] = {
                    "name": name.strip(),
                    "type": entity_type,
                    "properties": properties or {},
                    "created": now,
                    "updated": now,
                }
                self._enforce_entity_limit()
        self._save()
        return key

    def get_entity(self, name: str) -> dict | None:
        key = self._normalize(name)
        with self._lock:
            ent = self._entities.get(key)
            return dict(ent) if ent else None

    def remove_entity(self, name: str) -> bool:
        key = self._normalize(name)
        with self._lock:
            if key not in self._entities:
                return False
            del self._entities[key]
            # Remove all relations involving this entity
            self._relations = [
                r for r in self._relations
                if r["source"] != key and r["target"] != key
            ]
            self._rebuild_indices()
        self._save()
        return True

    def search_entities(self, keyword: str, limit: int = 10) -> list[dict]:
        """Find entities whose name or type contains the keyword."""
        kw = keyword.lower()
        with self._lock:
            results = []
            for key, ent in self._entities.items():
                if kw in key or kw in ent.get("type", ""):
                    results.append(dict(ent))
                    if len(results) >= limit:
                        break
            return results

    # ── Relationship operations ──────────────────────────────────

    def add_relation(self, source: str, target: str, relation: str,
                     confidence: float = 0.8) -> bool:
        """Add a directed relationship between two entities.

        Creates the entities if they don't exist yet.
        Deduplicates: same (source, target, relation) updates confidence.
        """
        src_key = self._normalize(source)
        tgt_key = self._normalize(target)
        rel_norm = relation.strip().lower()
        if not src_key or not tgt_key or not rel_norm:
            return False

        # Ensure both entities exist
        self.add_entity(source)
        self.add_entity(target)

        with self._lock:
            # Check for duplicate
            for r in self._relations:
                if (r["source"] == src_key and r["target"] == tgt_key
                        and r["relation"] == rel_norm):
                    r["confidence"] = min(1.0, max(r["confidence"], confidence))
                    r["updated"] = time.time()
                    self._save_locked()
                    return True

            self._relations.append({
                "source": src_key,
                "target": tgt_key,
                "relation": rel_norm,
                "confidence": round(max(0.0, min(1.0, confidence)), 2),
                "created": time.time(),
                "updated": time.time(),
            })
            self._enforce_relation_limit()
            self._rebuild_indices()
        self._save()
        return True

    def get_relations(self, entity_name: str,
                      direction: str = "both") -> list[dict]:
        """Get all relations involving an entity.

        Args:
            direction: "outgoing", "incoming", or "both"
        """
        key = self._normalize(entity_name)
        with self._lock:
            results = []
            if direction in ("outgoing", "both"):
                for idx in self._outgoing.get(key, []):
                    if idx < len(self._relations):
                        results.append(dict(self._relations[idx]))
            if direction in ("incoming", "both"):
                for idx in self._incoming.get(key, []):
                    if idx < len(self._relations):
                        results.append(dict(self._relations[idx]))
            return results

    def find_relations(self, relation_type: str,
                       limit: int = 20) -> list[dict]:
        """Find all relations of a given type."""
        rel_norm = relation_type.strip().lower()
        with self._lock:
            return [
                dict(r) for r in self._relations
                if r["relation"] == rel_norm
            ][:limit]

    # ── Graph queries ────────────────────────────────────────────

    def get_neighbors(self, entity_name: str, depth: int = 1) -> dict:
        """Get a subgraph of entities connected within N hops.

        Returns:
            {"entities": [...], "relations": [...]}
        """
        key = self._normalize(entity_name)
        visited = set()
        frontier = {key}
        collected_entities = []
        collected_relations = []

        for _ in range(depth):
            next_frontier = set()
            with self._lock:
                for node in frontier:
                    if node in visited:
                        continue
                    visited.add(node)
                    ent = self._entities.get(node)
                    if ent:
                        collected_entities.append(dict(ent))
                    for idx in self._outgoing.get(node, []):
                        if idx < len(self._relations):
                            r = self._relations[idx]
                            collected_relations.append(dict(r))
                            next_frontier.add(r["target"])
                    for idx in self._incoming.get(node, []):
                        if idx < len(self._relations):
                            r = self._relations[idx]
                            collected_relations.append(dict(r))
                            next_frontier.add(r["source"])
            frontier = next_frontier - visited

        # Collect remaining frontier entities
        with self._lock:
            for node in frontier:
                ent = self._entities.get(node)
                if ent:
                    collected_entities.append(dict(ent))

        return {
            "entities": collected_entities,
            "relations": collected_relations,
        }

    def query(self, subject: str = "", relation: str = "",
              obj: str = "") -> list[dict]:
        """Triple-pattern query: any field can be empty (wildcard).

        Examples:
            query(subject="python")               → all facts about Python
            query(relation="created_by")           → all created_by relations
            query(subject="python", obj="guido")   → Python→?→Guido
        """
        s = self._normalize(subject) if subject else ""
        r = relation.strip().lower() if relation else ""
        o = self._normalize(obj) if obj else ""

        with self._lock:
            results = []
            for rel in self._relations:
                if s and rel["source"] != s:
                    continue
                if r and rel["relation"] != r:
                    continue
                if o and rel["target"] != o:
                    continue
                # Enrich with entity names
                src_ent = self._entities.get(rel["source"], {})
                tgt_ent = self._entities.get(rel["target"], {})
                results.append({
                    "source": src_ent.get("name", rel["source"]),
                    "relation": rel["relation"],
                    "target": tgt_ent.get("name", rel["target"]),
                    "confidence": rel["confidence"],
                })
            return results

    # ── LLM context ──────────────────────────────────────────────

    def get_summary(self, max_triples: int = 15) -> str:
        """Text summary of the knowledge graph for LLM context."""
        with self._lock:
            n_ent = len(self._entities)
            n_rel = len(self._relations)
            if n_ent == 0:
                return "(knowledge graph is empty)"

            # Pick the most recent/confident relations
            sorted_rels = sorted(
                self._relations,
                key=lambda r: (r.get("confidence", 0), r.get("updated", 0)),
                reverse=True,
            )[:max_triples]

            lines = [f"Knowledge graph: {n_ent} entities, {n_rel} relations"]
            for r in sorted_rels:
                src = self._entities.get(r["source"], {}).get("name", r["source"])
                tgt = self._entities.get(r["target"], {}).get("name", r["target"])
                lines.append(f"  {src} --[{r['relation']}]--> {tgt}")

        return "\n".join(lines)

    def get_context_for_llm(self, max_triples: int = 15) -> str:
        return self.get_summary(max_triples)

    # ── Stats ────────────────────────────────────────────────────

    @property
    def entity_count(self) -> int:
        with self._lock:
            return len(self._entities)

    @property
    def relation_count(self) -> int:
        with self._lock:
            return len(self._relations)

    def stats(self) -> dict:
        with self._lock:
            type_counts: dict[str, int] = defaultdict(int)
            for ent in self._entities.values():
                type_counts[ent.get("type", "unknown")] += 1
            rel_counts: dict[str, int] = defaultdict(int)
            for r in self._relations:
                rel_counts[r["relation"]] += 1
            return {
                "entities": len(self._entities),
                "relations": len(self._relations),
                "entity_types": dict(type_counts),
                "relation_types": dict(rel_counts),
            }

    # ── Persistence ──────────────────────────────────────────────

    def _save(self) -> None:
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        save_json(self._file, {
            "entities": self._entities,
            "relations": self._relations,
        })

    def _rebuild_indices(self) -> None:
        self._outgoing = defaultdict(list)
        self._incoming = defaultdict(list)
        for i, r in enumerate(self._relations):
            self._outgoing[r["source"]].append(i)
            self._incoming[r["target"]].append(i)

    def _enforce_entity_limit(self) -> None:
        if len(self._entities) <= MAX_ENTITIES:
            return
        # Drop oldest entities
        sorted_keys = sorted(
            self._entities.keys(),
            key=lambda k: self._entities[k].get("updated", 0),
        )
        to_remove = sorted_keys[:len(self._entities) - MAX_ENTITIES]
        for k in to_remove:
            del self._entities[k]
        # Clean relations referencing removed entities
        removed_set = set(to_remove)
        self._relations = [
            r for r in self._relations
            if r["source"] not in removed_set and r["target"] not in removed_set
        ]
        self._rebuild_indices()

    def _enforce_relation_limit(self) -> None:
        if len(self._relations) <= MAX_RELATIONS:
            return
        # Drop oldest, lowest-confidence relations
        self._relations.sort(
            key=lambda r: (r.get("confidence", 0), r.get("updated", 0)),
        )
        self._relations = self._relations[-(MAX_RELATIONS):]
        self._rebuild_indices()
