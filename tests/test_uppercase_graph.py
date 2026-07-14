"""
Simple LangGraph test — uppercases a string through a graph node.
No LLM calls. Just verifies LangGraph is wired up correctly.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END


# --- State schema ---
class TextState(TypedDict):
    text: str


# --- Node function ---
def uppercase_node(state: TextState) -> TextState:
    return {"text": state["text"].upper()}


# --- Build the graph ---
def build_graph():
    builder = StateGraph(TextState)
    builder.add_node("uppercase", uppercase_node)
    builder.set_entry_point("uppercase")
    builder.add_edge("uppercase", END)
    return builder.compile()


# --- Test ---
def test_uppercase_graph():
    graph = build_graph()
    result = graph.invoke({"text": "hello from langgraph"})
    assert result["text"] == "HELLO FROM LANGGRAPH", f"Unexpected result: {result['text']}"
    print(f"Input:  'hello from langgraph'")
    print(f"Output: '{result['text']}'")
    print("✓ Test passed!")


if __name__ == "__main__":
    test_uppercase_graph()
