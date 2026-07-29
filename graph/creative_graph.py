from agents.ideator import return_ideas
from agents.critic import critic_node
from agents.continuitiy_checker import continuity_node
from agents.image_prompter import image_prompter_node
from agents.formatter import formatter_node
from graph.state import CreativeState
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# --- Build the graph ---
graph = StateGraph(CreativeState)

# --- Add nodes ---
graph.add_node("ideator", return_ideas)
graph.add_node("continuity", continuity_node)
graph.add_node("critic", critic_node)
graph.add_node("formatter", formatter_node)
graph.add_node("image_prompter", image_prompter_node)

# --- Linear edges: START → ideator → continuity → critic ---
graph.add_edge(START, "ideator")
graph.add_edge("ideator", "continuity")
graph.add_edge("continuity", "critic")

# --- Conditional routing after critic ---
# "accept" or max 3 iterations → formatter → image_prompter → END
# drift + revise = stronger signal to loop back
def route_after_critic(state: CreativeState) -> str:
    if state["verdict"] == "accept":
        return "formatter"
    if state["iteration"] >= 3:
        return "formatter"
    if state["continuity_status"] == "drifted" and state["verdict"] == "revise":
        return "ideator"
    if state["verdict"] == "revise":
        return "ideator"
    return "formatter"

graph.add_conditional_edges("critic", route_after_critic, {
    "formatter": "formatter",
    "ideator": "ideator",
})

graph.add_edge("formatter", "image_prompter")
graph.add_edge("image_prompter", END)

# --- Compile with memory checkpointer for human-in-the-loop ---
memory = MemorySaver()
pipeline = graph.compile(checkpointer=memory, interrupt_after=["critic"])

# --- Compile without interrupt for terminal / test use ---
pipeline_auto = graph.compile()

# --- Run (terminal) ---
if __name__ == "__main__":
    import base64
    from graph.state import make_initial_state

    user_input = input("Enter a creative brief: ")

    image_path = input("Enter path to a reference image (or press Enter to skip): ").strip()
    image_data = ""
    if image_path:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

    result = pipeline_auto.invoke(make_initial_state(user_input, image_data))

    print("\n--- Final Concept ---")
    print(result["concept"])
    print("\n--- Critic Feedback ---")
    print(result["feedback"])
    print("\n--- Continuity Status ---")
    print(result["continuity_status"])
    print("\n--- Image Generation Prompt ---")
    print(result["image_prompt"])
    print(f"\n(Completed in {result['iteration']} iteration(s))")
