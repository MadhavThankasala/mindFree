"""
Critic/editor agent.
Reviews a creative concept and returns a structured verdict so the
orchestration graph can decide whether to keep iterating or converge.
"""

from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from graph.state import CreativeState
from agents.llm_client import get_structured_llm

_BASE_PROMPT = """\
You are a trusted creative collaborator giving honest feedback. Talk like a \
real person — direct, warm, and specific. Don't say "the concept effectively \
explores" or "there's a strong foundation here". Just say what works, what \
feels vague, and what you'd change.

Keep feedback to 2-3 sentences. Be the friend who tells you the truth, not \
the workshop facilitator who softens everything.

Verdict:
- "accept" if the idea is specific and interesting enough to move forward
- "revise" if something important is missing or unclear\
"""

_MODE_APPENDIX = {
    "short film":    "\n\nJudge it as a short film — does it have a clear visual situation and an emotional payoff?",
    "short story":   "\n\nJudge it as a short story — is there a distinct voice and a moment that earns its ending?",
    "startup pitch": "\n\nJudge it as a startup pitch — is the problem real, is the solution concrete, is there a reason this team/person should build it?",
    "song":          "\n\nJudge it as a song concept — is there a clear emotional core and something that could become a hook?",
    "general":       "",
}


class CritiqueResult(BaseModel):
    verdict: Literal["accept", "revise"] = Field(
        description="Whether the concept is ready to move forward or needs revision"
    )
    feedback: str = Field(description="Specific, actionable feedback for the author")


def critic_node(state: CreativeState) -> CreativeState:
    mode = state.get("mode", "general")
    system_prompt = _BASE_PROMPT + _MODE_APPENDIX.get(mode, "")
    model = get_structured_llm(CritiqueResult, max_tokens=512)
    result: CritiqueResult = model.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=state["concept"])]
    )
    return {**state, "feedback": result.feedback, "verdict": result.verdict}
