from agents.ideator import return_ideas, CreativeState
from agents.critic import critic_node
from agents.continuitiy_checker import continuity_node
from langgraph.graph import StateGraph, START, END

# --- Build the graph ---
graph = StateGraph(CreativeState)

# --- Add the three agent nodes ---
graph.add_node("ideator", return_ideas)
graph.add_node("continuity", continuity_node)
graph.add_node("critic", critic_node)

# --- Linear edges: START → ideator → continuity → critic ---
graph.add_edge(START, "ideator")
graph.add_edge("ideator", "continuity")
graph.add_edge("continuity", "critic")

# --- Conditional routing after critic ---
# If verdict is "accept" → done
# If verdict is "revise" → loop back to ideator for another pass
def route_after_critic(state: CreativeState) -> str:
    if state["verdict"] == "accept":
        return "end"
    return "ideator"

graph.add_conditional_edges("critic", route_after_critic, {
    "end": END,
    "ideator": "ideator"
})

# --- Compile ---
pipeline = graph.compile()

# --- Run ---
if __name__ == "__main__":
    user_input = input("Enter a creative brief: ")
    result = pipeline.invoke({
        "concept": user_input,
        "original_brief": user_input,
        "feedback": "",
        "verdict": "",
        "continuity_status": ""
    })
    print("\n--- Final Concept ---")
    print(result["concept"])
    print("\n--- Critic Feedback ---")
    print(result["feedback"])
    print("\n--- Continuity Status ---")
    print(result["continuity_status"])
