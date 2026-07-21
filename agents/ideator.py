from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv

load_dotenv()

IDEATOR_SYSTEM_PROMPT = """\
You are an imaginative creative collaborator. Given a topic or brief, generate \
a rich, compelling creative concept in 3-5 sentences. Be specific and original.
"""

class CreativeState(TypedDict):
    concept: str
    feedback: str
    verdict: str
    original_brief: str
    continuity_status: str

llm = ChatAnthropic(model="claude-haiku-4-5", max_tokens=512)

def return_ideas(state: CreativeState) -> CreativeState:
    response = llm.invoke([
        SystemMessage(content=IDEATOR_SYSTEM_PROMPT),
        HumanMessage(content=state["concept"])
    ])
    return {**state, "concept": response.content}

