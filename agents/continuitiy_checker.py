from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

CONTINUITY_SYSTEM_PROMPT = """\
You are a continuity checker in a creative collaboration pipeline. \
Your job is to verify that the current concept still faithfully addresses \
the original brief the user provided. Check for topic drift, missing core \
requirements, or contradictions with the original intent. \
Respond with either "consistent" if the concept aligns with the brief, \
or "drifted" if it has strayed too far, along with a brief explanation.
"""


# --- Structured output schema ---
class ContinuityResult(BaseModel):
    status: Literal["consistent", "drifted"] = Field(
        description="Whether the concept still aligns with the original brief"
    )
    explanation: str = Field(
        description="Brief explanation of why the concept is consistent or has drifted"
    )


# --- Shared state schema ---
class CreativeState(TypedDict):
    concept: str
    feedback: str
    verdict: str
    original_brief: str
    continuity_status: str


# --- Model ---
def get_continuity_model():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_anthropic_api_key_here":
        raise EnvironmentError("ANTHROPIC_API_KEY is not set in your .env file")
    model = ChatAnthropic(model="claude-haiku-4-5", max_tokens=512)
    return model.with_structured_output(ContinuityResult)


# --- Node function ---
def continuity_node(state: CreativeState) -> CreativeState:
    model = get_continuity_model()
    prompt = f"Original brief: {state['original_brief']}\n\nCurrent concept: {state['concept']}"
    result: ContinuityResult = model.invoke([
        SystemMessage(content=CONTINUITY_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])
    return {**state, "continuity_status": result.status, "feedback": result.explanation}
