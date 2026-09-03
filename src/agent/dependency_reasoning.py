"""Provider-free dependency coverage primitives for multi-hop reasoning."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterable

@dataclass(frozen=True)
class SubQuestionResult:
    id: str
    status: str = "pending"
    answer: str = ""
    evidence_ids: tuple[str, ...] = ()
    required_fields: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    facts: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        # A status emitted by a retriever is only meaningful when this
        # sub-question brought back an evidence row or structured fact.  Do
        # not let an unrelated global evidence pool satisfy a missing node.
        if not self.has_evidence or self.missing_fields:
            return False
        if self.status.lower() in {"completed", "complete", "supported", "entailed", "success", "succeeded", "resolved", "retrieved", "done"}:
            return True
        return bool(self.metadata.get("completed"))

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence_ids or self.facts or self.metadata.get("fact_count") or self.metadata.get("row_count"))

    @classmethod
    def from_mapping(cls, value: Any, *, fallback_id: str = "") -> "SubQuestionResult":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            return cls(id=fallback_id or "unknown")
        evidence = value.get("evidence_ids") or value.get("new_chunk_ids") or value.get("used_evidence_ids") or []
        facts = value.get("facts") or []
        missing = value.get("missing_fields") or value.get("missing") or []
        required = value.get("required_fields") or []
        source = value.get("source")
        status = str(value.get("status") or value.get("verification_status") or ("completed" if source and source != "skipped_budget" else "pending"))
        return cls(id=str(value.get("id") or value.get("sub_question_id") or fallback_id or "unknown"), status=status, answer=str(value.get("answer") or ""), evidence_ids=tuple(str(x) for x in evidence if x), required_fields=tuple(str(x) for x in required if x), missing_fields=tuple(str(x) for x in missing if x), facts=tuple(x for x in facts if isinstance(x, dict)), metadata=dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "status": self.status, "answer": self.answer, "evidence_ids": list(self.evidence_ids), "required_fields": list(self.required_fields), "missing_fields": list(self.missing_fields), "facts": list(self.facts), "completed": self.completed}

@dataclass(frozen=True)
class DependencyEdge:
    source: str
    target: str
    required: bool = True
    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target, "required": self.required}

def normalize_dependency_edges(edges: Iterable[Any] | None, sub_questions: Iterable[Any] | None = None) -> list[DependencyEdge]:
    out: list[DependencyEdge] = []
    seen: set[tuple[str, str]] = set()
    def add(edge: DependencyEdge) -> None:
        key = (edge.source, edge.target)
        if edge.source and edge.target and key not in seen:
            out.append(edge); seen.add(key)
    for item in edges or []:
        if isinstance(item, DependencyEdge): add(item)
        elif isinstance(item, dict):
            source = item.get("source") or item.get("from") or item.get("depends_on")
            target = item.get("target") or item.get("to") or item.get("id")
            if source and target: add(DependencyEdge(str(source), str(target), bool(item.get("required", True))))
        elif isinstance(item, (tuple, list)) and len(item) >= 2: add(DependencyEdge(str(item[0]), str(item[1])))
    for raw in sub_questions or []:
        if not isinstance(raw, dict): continue
        target = str(raw.get("id") or "")
        deps = raw.get("depends_on") or raw.get("dependencies") or raw.get("parent_ids") or []
        if target and isinstance(deps, str):
            deps = [deps]
        if target and isinstance(deps, (list, tuple, set)):
            for source in deps: add(DependencyEdge(str(source), target))
    return out

def assess_coverage(required: Iterable[Any] | None, results: Iterable[Any] | None, *, edges: Iterable[Any] | None = None, evidence_pool: dict[str, Any] | None = None) -> dict[str, Any]:
    raw_required = list(required or [])
    required_ids: list[str] = []
    required_seen: set[str] = set()
    required_fields: dict[str, list[str]] = {}
    for item in raw_required:
        if isinstance(item, dict):
            ident = item.get("id") or item.get("sub_question_id")
            if ident:
                ident = str(ident)
                if ident not in required_seen:
                    required_ids.append(ident); required_seen.add(ident)
                required_fields[ident] = [str(x) for x in (item.get("required_fields") or []) if x]
        elif item:
            ident = str(item)
            if ident not in required_seen:
                required_ids.append(ident); required_seen.add(ident)
    normalized = [SubQuestionResult.from_mapping(item) for item in (results or [])]
    by_id = {item.id: item for item in normalized}
    completed = {item.id for item in normalized if item.completed}
    edge_list = normalize_dependency_edges(edges, raw_required)
    blocked: dict[str, list[str]] = {}
    # Propagate prerequisite failures to a fixed point so transitive chains are
    # handled regardless of edge ordering (q1 -> q2 -> q3).
    changed = True
    while changed:
        changed = False
        for edge in edge_list:
            if edge.required and edge.target in completed and edge.source not in completed:
                completed.discard(edge.target)
                blocked.setdefault(edge.target, []).append(edge.source)
                changed = True
    for ident, fields in required_fields.items():
        item = by_id.get(ident)
        if item and fields and any(field in item.missing_fields for field in fields): completed.discard(ident)
    missing = [ident for ident in required_ids if ident not in completed]
    count = len(required_ids)
    ratio = (count - len(missing)) / count if count else 0.0
    has_evidence = bool(evidence_pool) or any(item.has_evidence for item in normalized)
    # An empty dependency set means decomposition produced no actionable
    # requirements; do not allow a deterministic conclusion on that basis.
    if not required_ids or not has_evidence: decision = "abstain"
    elif missing: decision = "partial" if completed else "abstain"
    else: decision = "complete"
    return {"required": required_ids, "completed": sorted(completed), "missing": missing, "coverage_ratio": ratio, "required_count": count, "completed_count": len(completed), "dependency_edges": [edge.to_dict() for edge in edge_list], "blocked_by": blocked, "decision": decision, "partial": decision == "partial", "deterministic_conclusion_allowed": decision == "complete", "abstain": decision == "abstain", "has_evidence": has_evidence}

compute_coverage = assess_coverage
build_dependency_edges = normalize_dependency_edges
