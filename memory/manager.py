"""
Memory Manager — coordinates all memory sub-systems and provides
the unified public API (``MemorySystem``).

Architecture:
  ┌─────────────────────────────────────────────────┐
  │              MemoryManager                       │
  │  ┌───────────┬───────────┬──────────┬─────────┐ │
  │  │ ShortTerm │ Episodic  │ Semantic │ Vector  │ │
  │  │  (RAM)    │ (disk)    │ (disk)   │ (disk)  │ │
  │  └───────────┴───────────┴──────────┴─────────┘ │
  └─────────────────────────────────────────────────┘

``MemorySystem`` is a thin subclass that preserves the exact same
method names used by brain.py so all 26 call-sites keep working
with zero changes.
"""

import json
import os
import time
from threading import Lock

from .short_term import ShortTermMemory
from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .vector import VectorMemory
from .knowledge_graph import KnowledgeGraph

MEMORY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "memory_store",
)


class MemoryManager:
    """Low-level coordinator — exposes each sub-system directly."""

    _PROJECTS_FILE = "active_projects.json"

    def __init__(self, store_dir: str | None = None):
        self._store_dir = store_dir or MEMORY_DIR
        os.makedirs(self._store_dir, exist_ok=True)

        self.short_term = ShortTermMemory()
        self.episodic = EpisodicMemory(self._store_dir)
        self.semantic = SemanticMemory(self._store_dir)
        self.vector = VectorMemory(self._store_dir)
        self.knowledge_graph = KnowledgeGraph(self._store_dir)

        # Active projects — persistent multi-session goals
        self._projects_path = os.path.join(self._store_dir, self._PROJECTS_FILE)
        self._projects_lock = Lock()
        self._projects: list[dict] = self._load_projects()

    # ── Active Projects API ──────────────────────────────────────

    def create_project(self, goal: str, *,
                       notes: str = "",
                       priority: int = 0) -> dict:
        """Start a new multi-session project."""
        project = {
            "id": f"proj_{int(time.time() * 1000)}",
            "goal": goal,
            "status": "active",        # active | paused | completed | abandoned
            "progress": 0.0,           # 0.0 – 1.0
            "notes": notes,
            "priority": priority,
            "created_at": time.time(),
            "updated_at": time.time(),
            "steps_completed": [],      # list of step descriptions
            "steps_remaining": [],      # list of step descriptions
        }
        with self._projects_lock:
            self._projects.append(project)
            self._save_projects()
        return project

    def update_project(self, project_id: str, **fields) -> bool:
        """Update fields on an existing project."""
        with self._projects_lock:
            for p in self._projects:
                if p["id"] == project_id:
                    for k, v in fields.items():
                        if k in p:
                            p[k] = v
                    p["updated_at"] = time.time()
                    self._save_projects()
                    return True
        return False

    def complete_project(self, project_id: str) -> bool:
        """Mark a project as completed."""
        return self.update_project(project_id, status="completed", progress=1.0)

    def abandon_project(self, project_id: str) -> bool:
        """Abandon a project (won't be resumed)."""
        return self.update_project(project_id, status="abandoned")

    def get_active_projects(self) -> list[dict]:
        """Return all projects with status 'active'."""
        with self._projects_lock:
            return [p for p in self._projects if p["status"] == "active"]

    def get_all_projects(self) -> list[dict]:
        """Return all projects regardless of status."""
        with self._projects_lock:
            return list(self._projects)

    def get_project(self, project_id: str) -> dict | None:
        with self._projects_lock:
            for p in self._projects:
                if p["id"] == project_id:
                    return dict(p)
        return None

    def get_projects_summary(self, max_projects: int = 5) -> str:
        """Compact text summary of active projects for LLM context."""
        active = self.get_active_projects()
        if not active:
            return ""
        active.sort(key=lambda p: (-p["priority"], p["created_at"]))
        lines = ["Active projects:"]
        for p in active[:max_projects]:
            pct = int(p["progress"] * 100)
            lines.append(f"  - [{pct}%] {p['goal']}")
            if p["notes"]:
                lines.append(f"    Note: {p['notes']}")
            done = p.get("steps_completed", [])
            remain = p.get("steps_remaining", [])
            if done:
                lines.append(f"    Done: {', '.join(done[-3:])}")
            if remain:
                lines.append(f"    Next: {', '.join(remain[:3])}")
        return "\n".join(lines)

    def _load_projects(self) -> list[dict]:
        if not os.path.exists(self._projects_path):
            return []
        try:
            with open(self._projects_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    def _save_projects(self) -> None:
        tmp = self._projects_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._projects, f, indent=2)
        os.replace(tmp, self._projects_path)

    def cross_search(self, keyword: str, limit: int = 10) -> dict:
        """Search across all memory types for *keyword*.

        Returns:
            {
              "facts": [str, ...],
              "episodes": [dict, ...],
              "events": [dict, ...],
            }
        """
        return {
            "facts": self.semantic.search_facts(keyword),
            "episodes": self.episodic.search(keyword, limit=limit),
            "events": [
                e for e in self.short_term.get_recent(50)
                if keyword.lower() in e.get("content", "").lower()
            ][-limit:],
            "knowledge": self.knowledge_graph.search_entities(keyword, limit=limit),
            "vector": self.vector.search_text(keyword, top_k=limit),
        }

    def stats(self) -> dict:
        """Return a summary of all memory sub-system sizes."""
        return {
            "short_term_events": self.short_term.size,
            "episodic_total": self.episodic.total_episodes,
            "semantic_facts": len(self.semantic.get_facts()),
            "knowledge_entities": self.knowledge_graph.entity_count,
            "knowledge_relations": self.knowledge_graph.relation_count,
            "vector_items": self.vector.size,
            "vector_categories": self.vector.category_counts(),
            "vector_embedder": self.vector.embedder_info,
        }


class MemorySystem(MemoryManager):
    """Backwards-compatible public interface.

    Every method that brain.py (and world_state.py) already calls is
    preserved with the exact same signature.  Internally each call
    delegates to the appropriate sub-system.
    """

    # ── Short-term (delegates to self.short_term) ────────────────

    def add_event(self, event_type: str, content: str) -> None:
        self.short_term.add_event(event_type, content)
        self._auto_embed_text(
            category=f"event:{event_type}",
            text=content,
            label=f"{event_type}: {content[:60]}",
            metadata={"event_type": event_type},
        )

    def get_recent_events(self, n: int = 15, event_type: str = None) -> list:
        return self.short_term.get_recent(n, event_type)

    def get_context_summary(self, max_events: int = 20) -> str:
        return self.short_term.get_context_summary(max_events)

    # ── Episodic (delegates to self.episodic) ────────────────────

    def log_episode(self, goal: str, outcome: str, steps: int,
                    success: bool) -> None:
        self.episodic.log_episode(goal, outcome, steps, success)
        self._auto_embed_text(
            category="episode",
            text=f"Goal: {goal}. Outcome: {outcome}. Steps: {steps}. Success: {success}",
            label=f"episode: {goal[:60]}",
            metadata={"goal": goal, "success": success, "steps": steps},
        )

    def get_recent_episodes(self, n: int = 5) -> list:
        return self.episodic.get_recent(n)

    def get_episodes_summary(self, n: int = 5) -> str:
        return self.episodic.get_summary(n)

    # ── Semantic / long-term (delegates to self.semantic) ────────

    def learn_fact(self, fact: str) -> None:
        self.semantic.learn_fact(fact)
        self._auto_embed_text("fact", fact, label=f"fact: {fact[:60]}")

    def learn_preference(self, preference: str) -> None:
        self.semantic.learn_preference(preference)
        self._auto_embed_text("preference", preference, label=f"pref: {preference[:60]}")

    def add_self_note(self, note: str) -> None:
        self.semantic.add_self_note(note)
        self._auto_embed_text("self_note", note, label=f"self: {note[:60]}")

    def set_user_name(self, name: str) -> None:
        self.semantic.set_user_name(name)
        self._auto_embed_text("profile", f"User name is {name}", label=f"user_name:{name}")

    def get_long_term_summary(self) -> str:
        return self.semantic.get_summary()

    # ── Knowledge graph (delegates to self.knowledge_graph) ──────

    def add_knowledge(self, source: str, relation: str, target: str,
                      confidence: float = 0.8) -> bool:
        ok = self.knowledge_graph.add_relation(source, target, relation, confidence)
        if ok:
            self._auto_embed_text(
                "knowledge",
                f"{source} {relation} {target}",
                label=f"kg:{source[:20]}-{relation[:20]}-{target[:20]}",
                metadata={"source": source, "relation": relation, "target": target, "confidence": confidence},
            )
        return ok

    def add_entity(self, name: str, entity_type: str = "thing",
                   properties: dict | None = None) -> str:
        return self.knowledge_graph.add_entity(name, entity_type, properties)

    def query_knowledge(self, subject: str = "", relation: str = "",
                        obj: str = "") -> list[dict]:
        return self.knowledge_graph.query(subject, relation, obj)

    def get_knowledge_summary(self) -> str:
        return self.knowledge_graph.get_summary()

    # ── Vector auto-embedding helpers ───────────────────────────

    def _auto_embed_text(self, category: str, text: str,
                         *, label: str = "", metadata: dict | None = None) -> None:
        """Best-effort local embedding insertion for textual memories."""
        if not text:
            return
        try:
            item_id = f"{category}:{int(time.time()*1000)}:{abs(hash(text)) % 1000000}"
            self.vector.store_text(
                item_id=item_id,
                category=category,
                text=text,
                label=label or text[:80],
                metadata=metadata,
            )
        except Exception:
            # Never let vector indexing break core memory writes.
            pass

    # ── Active Projects (delegates to base class) ────────────────

    # create_project, update_project, complete_project,
    # abandon_project, get_active_projects, get_all_projects,
    # get_project, get_projects_summary — all inherited from
    # MemoryManager and available directly.
