"""
Critic/editor agent.
Reviews a creative concept and returns a structured verdict so the
orchestration graph can decide whether to keep iterating or converge.
"""

import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

load_dotenv()

CRITIC_SYSTEM_PROMPT = """\
You are a sharp, constructive creative critic reviewing a concept from a \
collaborator. Give specific, actionable feedback in 2-4 sentences — not \
generic praise. Decide a verdict:
- "accept" if the concept is solid enough to move forward as-is
- "revise" if it needs another pass before it's ready
"""


class CritiqueResult(BaseModel):
    verdict: Literal["accept", "revise"] = Field(
        description="Whether the concept is ready to move forward or needs revision"
    )
    feedback: str = Field(description="Specific, actionable feedback for the author")


# --- State schema ---
class CreativeState(TypedDict):
    concept: str
    feedback: str
    verdict: str


def get_critic_model():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_anthropic_api_key_here":
        raise EnvironmentError("ANTHROPIC_API_KEY is not set in your .env file")
    model = ChatAnthropic(model="claude-haiku-4-5", max_tokens=512)
    return model.with_structured_output(CritiqueResult)


def critic_node(state: CreativeState) -> CreativeState:
    model = get_critic_model()
    result: CritiqueResult = model.invoke(
        [SystemMessage(content=CRITIC_SYSTEM_PROMPT), HumanMessage(content=state["concept"])]
    )
    return {**state, "feedback": result.feedback, "verdict": result.verdict}
