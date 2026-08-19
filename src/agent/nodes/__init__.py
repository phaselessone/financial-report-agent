"""Agent nodes (checklist v3.0 §P2 Node boundary).

Nodes are factory functions taking their dependencies (runtime, answerer, LLM
provider, config) and returning a LangGraph-compatible ``(state) -> state``
callable. The agent never allocates devices or opens HTTP itself — retrieval
stays in RetrievalRuntime and LLM traffic stays in the provider layer.
"""

from src.agent.nodes.analyze_query import make_analyze_query
from src.agent.nodes.decompose_query import make_decompose_query
from src.agent.nodes.extract_claims import make_extract_claims
from src.agent.nodes.finalize import make_finalize
from src.agent.nodes.grade_evidence import make_grade_evidence
from src.agent.nodes.retrieve import make_retrieve
from src.agent.nodes.retrieve_subtasks import make_retrieve_subtasks
from src.agent.nodes.rewrite_query import make_rewrite_query
from src.agent.nodes.synthesize import make_synthesize
from src.agent.nodes.verify_answer import make_verify_answer

__all__ = [
    "make_analyze_query",
    "make_decompose_query",
    "make_extract_claims",
    "make_retrieve",
    "make_retrieve_subtasks",
    "make_grade_evidence",
    "make_rewrite_query",
    "make_synthesize",
    "make_verify_answer",
    "make_finalize",
]
