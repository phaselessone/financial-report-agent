"""Agentic RAG package (checklist v3.0 §P2): Retrieve→Grade→Rewrite→Retrieve→Generate→Verify→Final.

Single serial graph, local CPU orchestration, remote LLM API only. No Planner,
no query decomposition, no parallel research, no memory, no Multi-Agent.
"""

from src.agent.config import AgentConfig
from src.agent.state import AgentState, new_agent_state

__all__ = ["AgentConfig", "AgentState", "new_agent_state"]
