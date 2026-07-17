from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
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

class ContinuityVerdict(TypedDict):
    str : 