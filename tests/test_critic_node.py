"""
One-node LangGraph test for the critic agent.
Feeds a sample concept in and confirms a structured verdict comes back.
"""

from langgraph.graph import StateGraph, END

from agents.critic import CreativeState, critic_node


def build_graph():
    builder = StateGraph(CreativeState)
    builder.add_node("critic", critic_node)
    builder.set_entry_point("critic")
    builder.add_edge("critic", END)
    return builder.compile()


def test_critic_node():
    graph = build_graph()
    concept = (
        "A choose-your-own-adventure story where the reader plays as a "
        "sentient thermostat trying to keep a family comfortable, but the "
        "family never explains what they actually want."
    )

    print(f"Concept: '{concept}'")
    print("Calling critic node...")

    result = graph.invoke({"concept": concept, "feedback": "", "verdict": ""})

    assert result["verdict"] in ("accept", "revise"), f"Unexpected verdict: {result['verdict']}"
    assert result["feedback"], "Feedback was empty!"
    print(f"Verdict:  {result['verdict']}")
    print(f"Feedback: {result['feedback']}")
    print("✓ Critic node test passed!")


if __name__ == "__main__":
    test_critic_node()
