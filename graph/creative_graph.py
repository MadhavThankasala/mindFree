from agents.ideator import return_ideas
from agents.critic import critic_node
from agents.continuitiy_checker import continuity_node
from agents.image_prompter import image_prompter_node
from graph.state import CreativeState
from langgraph.graph import StateGraph, START, END

# --- Build the graph ---
graph = StateGraph(CreativeState)

# --- Add nodes ---
graph.add_node("ideator", return_ideas)
graph.add_node("continuity", continuity_node)
graph.add_node("critic", critic_node)
graph.add_node("image_prompter", image_prompter_node)

# --- Linear edges: START → ideator → continuity → critic ---
graph.add_edge(START, "ideator")
graph.add_edge("ideator", "continuity")
graph.add_edge("continuity", "critic")

# --- Conditional routing after critic ---
# "accept" or max 3 iterations → image_prompter → END
# "revise" → loop back to ideator
def route_after_critic(state: CreativeState) -> str:
    if state["verdict"] == "accept" or state["iteration"] >= 3:
        return "image_prompter"
    return "ideator"

graph.add_conditional_edges("critic", route_after_critic, {
    "image_prompter": "image_prompter",
    "ideator": "ideator"
})

graph.add_edge("image_prompter", END)

# --- Compile ---
pipeline = graph.compile()

# --- Run ---
if __name__ == "__main__":
    import base64

    user_input = input("Enter a creative brief: ")

    # Optionally accept an image path
    image_path = input("Enter path to a reference image (or press Enter to skip): ").strip()
    image_data = ""
    if image_path:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

    result = pipeline.invoke({
        "concept": user_input,
        "original_brief": user_input,
        "feedback": "",
        "verdict": "",
        "continuity_status": "",
        "iteration": 0,
        "image_input": image_data,
        "image_prompt": ""
    })

    print("\n--- Final Concept ---")
    print(result["concept"])
    print("\n--- Critic Feedback ---")
    print(result["feedback"])
    print("\n--- Continuity Status ---")
    print(result["continuity_status"])
    print("\n--- Image Generation Prompt ---")
    print(result["image_prompt"])
    print(f"\n(Completed in {result['iteration']} iteration(s))")
