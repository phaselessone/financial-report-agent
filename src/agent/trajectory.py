"""Uniform node-level trajectory instrumentation for the agent graph."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable, Mapping
from functools import wraps
from types import MappingProxyType
from time import perf_counter
from typing import Any


_COUNTERS = {
    "steps": "step_count",
    "llm_calls": "llm_call_count",
    "tool_calls": "tool_call_count",
    "tokens": "total_tokens",
    "retrievals": "retrieval_count",
}

_TRACE_ALIASES = (
    "trajectory_events",
    "trace_events",
    "trajectory",
    "agent_steps",
    "events",
)
_DEPENDENCY_ALIASES = ("dependencies", "depends_on", "parent_event_ids", "caused_by")
_RECOVERY_ALIASES = ("recovery_of", "recovers", "recovered_event_ids")


def _ids(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set)) else [value]
    result: list[str] = []
    for item in values:
        event_id = str(item or "").strip()
        if event_id and event_id not in result:
            result.append(event_id)
    return result


def _first_present(event: Mapping[str, Any], aliases: tuple[str, ...]) -> tuple[Any, bool]:
    for alias in aliases:
        if alias in event:
            return event.get(alias), True
    return None, False


def canonicalize_trajectory_events(source: Any) -> list[dict[str, Any]]:
    """Normalize every supported trace alias into one event/lineage contract."""
    if isinstance(source, Mapping):
        raw_events: Any = []
        for alias in _TRACE_ALIASES:
            value = source.get(alias)
            if isinstance(value, (list, tuple)):
                raw_events = value
                break
    elif isinstance(source, (list, tuple)):
        raw_events = source
    else:
        raw_events = []

    reserved_ids = {
        str(raw.get("event_id") or raw.get("id") or raw.get("trace_id") or "").strip()
        for raw in raw_events
        if isinstance(raw, Mapping)
        and str(raw.get("event_id") or raw.get("id") or raw.get("trace_id") or "").strip()
    }
    events: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    next_generated_index = 1
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping):
            continue
        event = dict(raw)
        explicit_event_id = str(
            event.get("event_id") or event.get("id") or event.get("trace_id") or ""
        ).strip()
        if explicit_event_id:
            event_id = explicit_event_id
        else:
            event_id = f"event-{next_generated_index:04d}"
            while event_id in reserved_ids or event_id in seen_ids:
                next_generated_index += 1
                event_id = f"event-{next_generated_index:04d}"
            next_generated_index += 1
        if event_id in seen_ids:
            raise ValueError(f"duplicate trajectory event_id: {event_id}")
        seen_ids.add(event_id)
        dependencies, has_dependencies = _first_present(event, _DEPENDENCY_ALIASES)
        recovery_of, has_recovery = _first_present(event, _RECOVERY_ALIASES)
        declared_lineage = str(event.get("lineage_status") or "").strip().lower()
        structurally_resolved = bool(
            explicit_event_id and (has_dependencies or has_recovery)
        )
        event.update(
            {
                "event_id": event_id,
                "dependencies": _ids(dependencies),
                "recovery_of": _ids(recovery_of),
                "lineage_status": (
                    "unresolved"
                    if declared_lineage == "unresolved"
                    else "resolved"
                    if structurally_resolved
                    else "unresolved"
                ),
            }
        )
        events.append(event)
    return events


def _snapshot(state: Mapping[str, Any]) -> dict[str, int]:
    return {
        label: int(state.get(state_key, 0) or 0)
        for label, state_key in _COUNTERS.items()
    }


def _budget_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    return {
        label: int(after.get(label, 0)) - int(before.get(label, 0))
        for label in _COUNTERS
    }


def _next_event(
    events: list[dict[str, Any]],
    *,
    node_name: str,
    status: str,
    latency_ms: float,
    error_type: str | None,
    termination_reason: Any,
    budget_usage: Mapping[str, int],
) -> dict[str, Any]:
    previous_node = next(
        (event for event in reversed(events) if event.get("event_type") == "node"),
        None,
    )
    dependencies = [previous_node["event_id"]] if previous_node else []
    recovered = {
        event_id
        for event in events
        for event_id in _ids(event.get("recovery_of"))
    }
    recovery_of: list[str] = []
    if status == "SUCCESS":
        for event in reversed(events):
            if (
                event.get("node") == node_name
                and str(event.get("status") or "").upper() in {"FAILED", "ERROR", "EXCEPTION"}
                and event["event_id"] not in recovered
            ):
                recovery_of = [event["event_id"]]
                break
    existing_ids = {str(event.get("event_id") or "") for event in events}
    next_index = len(events) + 1
    event_id = f"event-{next_index:04d}"
    while event_id in existing_ids:
        next_index += 1
        event_id = f"event-{next_index:04d}"
    return {
        "event_id": event_id,
        "event_type": "node",
        "node": node_name,
        "action": "node_execution",
        "status": status,
        "latency_ms": latency_ms,
        "error_type": error_type,
        "termination_reason": termination_reason,
        "budget_usage": dict(budget_usage),
        "dependencies": dependencies,
        "recovery_of": recovery_of,
        "lineage_status": "resolved",
    }


def _partial_state_snapshot(state: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        value = deepcopy(dict(state))
    except Exception:
        value = dict(state)
        value["trajectory_events"] = [
            dict(item)
            for item in state.get("trajectory_events", [])
            if isinstance(item, Mapping)
        ]
    return MappingProxyType(value)


def trace_graph_node(
    node_name: str,
    node: Callable[[dict[str, Any]], dict[str, Any]],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Wrap a graph node with one stable completion/failure event.

    Tool, reasoning-step and recovery events remain available at their finer
    granularity.  This event is the common envelope that makes node coverage,
    latency and budget deltas comparable across every path through the graph.
    """

    @wraps(node)
    def traced(state: dict[str, Any]) -> dict[str, Any]:
        started = perf_counter()
        before = _snapshot(state)
        try:
            result = node(state)
        except Exception as exc:
            after = _snapshot(state)
            events = canonicalize_trajectory_events(state)
            events.append(
                _next_event(
                    events,
                    node_name=node_name,
                    status="FAILED",
                    latency_ms=(perf_counter() - started) * 1000,
                    error_type=type(exc).__name__,
                    termination_reason=state.get("termination_reason"),
                    budget_usage=_budget_delta(before, after),
                )
            )
            state["trajectory_events"] = events
            try:
                setattr(exc, "agent_partial_state", _partial_state_snapshot(state))
            except (AttributeError, TypeError):
                pass
            raise
        target = result if isinstance(result, dict) else state
        after = _snapshot(target)
        events = canonicalize_trajectory_events(target)
        events.append(
            _next_event(
                events,
                node_name=node_name,
                status="SUCCESS",
                latency_ms=(perf_counter() - started) * 1000,
                error_type=None,
                termination_reason=target.get("termination_reason"),
                budget_usage=_budget_delta(before, after),
            )
        )
        target["trajectory_events"] = events
        return target

    return traced


__all__ = ["canonicalize_trajectory_events", "trace_graph_node"]
