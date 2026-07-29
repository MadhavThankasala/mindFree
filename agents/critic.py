"""
Critic/editor agent.
Reviews a creative concept and returns a structured verdict so the
orchestration graph can decide whether to keep iterating or converge.
"""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from graph.state import CreativeState

load_dotenv()

CRITIC_SYSTEM_PROMPT = """\
You are a trusted creative collaborator giving honest feedback. Talk like a \
real person — direct, warm, and specific. Don't say "the concept effectively \
explores" or "there's a strong foundation here". Just say what works, what \
feels vague, and what you'd change.

Keep feedback to 2-3 sentences. Be the friend who tells you the truth, not \
the workshop facilitator who softens everything.

Verdict:
- "accept" if the idea is specific and interesting enough to move forward
- "revise" if something important is missing or unclear
"""


class CritiqueResult(BaseModel):
    verdict: Literal["accept", "revise"] = Field(
        description="Whether the concept is ready to move forward or needs revision"
    )
    feedback: str = Field(description="Specific, actionable feedback for the author")


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
