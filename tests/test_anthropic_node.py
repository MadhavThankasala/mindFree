"""
One-node LangGraph test — makes a real Anthropic API call.
Confirms ANTHROPIC_API_KEY and provider setup are working.
"""

import os
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

load_dotenv()


# --- State schema ---
class MessageState(TypedDict):
    user_input: str
    response: str


# --- Build model ---
def get_model():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_anthropic_api_key_here":
        raise EnvironmentError("ANTHROPIC_API_KEY is not set in your .env file")
    return ChatAnthropic(model="claude-haiku-4-5", max_tokens=256)


# --- Node function ---
def call_anthropic(state: MessageState) -> MessageState:
    model = get_model()
    response = model.invoke([HumanMessage(content=state["user_input"])])
    return {"response": response.content}


# --- Build the graph ---
def build_graph():
    builder = StateGraph(MessageState)
    builder.add_node("call_anthropic", call_anthropic)
    builder.set_entry_point("call_anthropic")
    builder.add_edge("call_anthropic", END)
    return builder.compile()


# --- Test ---
def test_anthropic_node():
    graph = build_graph()
    user_input = "Reply in one sentence: what is LangGraph used for?"

    print(f"Input:  '{user_input}'")
    print("Calling Anthropic API...")

    result = graph.invoke({"user_input": user_input, "response": ""})

    assert result["response"], "Response was empty!"
    print(f"Output: '{result['response']}'")
    print("✓ Anthropic API call succeeded!")


if __name__ == "__main__":
    test_anthropic_node()
