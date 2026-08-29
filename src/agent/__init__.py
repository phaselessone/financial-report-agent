"""Financial-report agent package.

The graph keeps the deterministic retrieve/grade/rewrite fallback while also
supporting bounded query decomposition, structured fact lookup, calculation
provenance, claim verification, and a whitelist-controlled tool loop.  The
package does not permit arbitrary code or dynamic network/tool execution.
"""

from src.agent.config import AgentConfig
from src.agent.reasoning_plan import (
    ReasoningPlan,
    ReasoningStep,
    ReasoningStepKind,
    StepResult,
    StepStatus,
)
from src.agent.state import AgentState, new_agent_state

__all__ = [
    "AgentConfig",
    "AgentState",
    "ReasoningPlan",
    "ReasoningStep",
    "ReasoningStepKind",
    "StepResult",
    "StepStatus",
    "new_agent_state",
]
