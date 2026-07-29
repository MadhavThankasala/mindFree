from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from graph.state import CreativeState
from agents.llm_client import get_llm, get_structured_llm

# --- Brief validation ---

BRIEF_VALIDATION_PROMPT = """\
You are reading a creative brief someone just typed. Be honest and direct — \
like a smart friend, not a customer service bot.

Check for:
- Internal contradictions (e.g. "tomorrow I have work. tomorrow I don't have work")
- Statements that don't make logical sense together
- Briefs that are too vague or empty to work with (e.g. just "thing" or "idk")

If the brief is fine — even if it's simple or weird — say it's valid. \
Creative briefs don't need to be polished. Only flag real problems.

Be blunt. If something contradicts itself, say exactly what contradicts what.
"""


class BriefValidationResult(BaseModel):
    valid: bool = Field(
        description="True if the brief is usable, False if it has real problems"
    )
    reason: str = Field(
        description="If invalid, explain exactly what's wrong in plain language. If valid, leave empty."
    )


def validate_brief(brief: str) -> BriefValidationResult:
    """Call this before running the pipeline to catch bad briefs early."""
    return get_structured_llm(BriefValidationResult, max_tokens=256).invoke([
        SystemMessage(content=BRIEF_VALIDATION_PROMPT),
        HumanMessage(content=brief),
    ])


# --- Continuity system prompt ---

CONTINUITY_SYSTEM_PROMPT = """\
You are checking whether a developed concept still matches the original brief. \
Be honest — if it's drifted, say so plainly. If it's on track, say that. \
Don't hedge. One or two sentences is enough.
"""


# --- Structured output schema ---
class ContinuityResult(BaseModel):
    status: Literal["consistent", "drifted"] = Field(
        description="Whether the concept still aligns with the original brief"
    )
    explanation: str = Field(
        description="Plain, honest explanation of alignment or drift"
    )


# --- Node function ---
def continuity_node(state: CreativeState) -> CreativeState:
    # Skip on the first pass — the concept IS the brief at iteration 0
    if state["iteration"] == 0:
        return {**state, "continuity_status": "consistent", "continuity_feedback": ""}

    model = get_structured_llm(ContinuityResult, max_tokens=256)
    prompt = f"Original brief: {state['original_brief']}\n\nCurrent concept: {state['concept']}"
    result: ContinuityResult = model.invoke([
        SystemMessage(content=CONTINUITY_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    return {**state, "continuity_status": result.status, "continuity_feedback": result.explanation}
