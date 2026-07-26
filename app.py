import base64
import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from graph.creative_graph import pipeline

load_dotenv()

st.set_page_config(page_title="mindFree", page_icon="🧠", layout="wide")

# ─── Session state defaults ────────────────────────────────────────────────────
for key, default in {
    "brief": "",
    "history": [],
    "thread_id": str(uuid.uuid4()),
    "awaiting_human": False,
    "current_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ───────────────────────────────────────────────────────────────────
def generate_brief() -> str:
    llm = ChatAnthropic(model="claude-haiku-4-5", max_tokens=100)
    response = llm.invoke([HumanMessage(content=
        "Generate a single short creative brief in one sentence. "
        "Be imaginative and specific. Output only the brief, no preamble."
    )])
    return response.content


def generate_image(prompt: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.images.generate(
            model="dall-e-3", prompt=prompt, size="1024x1024", n=1
        )
        return response.data[0].url
    except Exception as e:
        st.warning(f"Image generation failed: {e}")
        return None


def config():
    """LangGraph thread config for human-in-the-loop checkpointing."""
    return {"configurable": {"thread_id": st.session_state["thread_id"]}}


AGENT_LABELS = {
    "ideator":       ("✍️", "Ideator",            "Generating creative concept..."),
    "continuity":    ("🔗", "Continuity Checker",  "Checking brief alignment..."),
    "critic":        ("🔍", "Critic",              "Reviewing concept..."),
    "image_prompter":("🎨", "Image Prompter",      "Building image prompt..."),
}


# ─── Sidebar — history ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🕘 Session History")
    if not st.session_state["history"]:
        st.caption("No runs yet.")
    for i, item in enumerate(reversed(st.session_state["history"])):
        with st.expander(f"Run {len(st.session_state['history']) - i}: {item['brief'][:40]}..."):
            st.markdown(f"**Concept:** {item['concept'][:200]}...")
            st.markdown(f"**Iterations:** {item['iterations']}")
            st.markdown(f"**Continuity:** {item['continuity_status']}")


# ─── Main UI ───────────────────────────────────────────────────────────────────
st.title("🧠 mindFree")
st.caption("Multi-agent creative collaboration powered by LangGraph + Claude")
st.divider()

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("**Creative Brief**")
with col2:
    if st.button("✨ Generate for me"):
        with st.spinner("Generating brief..."):
            st.session_state["brief"] = generate_brief()
        st.rerun()

brief = st.text_area(
    "Creative Brief",
    value=st.session_state["brief"],
    placeholder="e.g. a short film about loneliness in a crowded city",
    height=100,
    label_visibility="collapsed"
)

image_file = st.file_uploader("Reference Image (optional)", type=["jpg", "jpeg", "png"])
run = st.button("Generate", type="primary", disabled=not brief.strip() or st.session_state["awaiting_human"])

# ─── Initial pipeline run ──────────────────────────────────────────────────────
if run:
    # Reset thread for a fresh run
    st.session_state["thread_id"] = str(uuid.uuid4())
    st.session_state["awaiting_human"] = False
    st.session_state["current_result"] = None

    image_data = ""
    if image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

    initial_state = {
        "concept": brief,
        "original_brief": brief,
        "feedback": "",
        "verdict": "",
        "continuity_status": "",
        "iteration": 0,
        "image_input": image_data,
        "image_prompt": ""
    }

    # --- Live agent status via streaming ---
    status_box = st.empty()
    for event in pipeline.stream(initial_state, config(), stream_mode="updates"):
        for node_name in event:
            if node_name in AGENT_LABELS:
                icon, label, message = AGENT_LABELS[node_name]
                status_box.info(f"{icon} **{label}** — {message}")

    status_box.empty()

    # Check if graph paused at critic (human-in-the-loop interrupt)
    state_snapshot = pipeline.get_state(config())
    if state_snapshot.next:
        st.session_state["awaiting_human"] = True
        st.session_state["current_result"] = state_snapshot.values
        st.rerun()
    else:
        result = state_snapshot.values
        st.session_state["history"].append({
            "brief": brief,
            "concept": result["concept"],
            "iterations": result["iteration"],
            "continuity_status": result["continuity_status"]
        })
        st.session_state["current_result"] = result
        st.rerun()

# ─── Human-in-the-loop review ─────────────────────────────────────────────────
if st.session_state["awaiting_human"] and st.session_state["current_result"]:
    result = st.session_state["current_result"]
    st.divider()
    st.subheader("🔍 Critic paused — your review needed")
    st.write(f"**Current concept:** {result['concept']}")
    st.write(f"**Critic verdict:** `{result['verdict']}`")
    st.write(f"**Critic feedback:** {result['feedback']}")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("✅ Accept & continue", type="primary"):
            # Override verdict to accept and resume
            pipeline.update_state(config(), {"verdict": "accept"})
            status_box = st.empty()
            for event in pipeline.stream(None, config(), stream_mode="updates"):
                for node_name in event:
                    if node_name in AGENT_LABELS:
                        icon, label, message = AGENT_LABELS[node_name]
                        status_box.info(f"{icon} **{label}** — {message}")
            status_box.empty()
            final = pipeline.get_state(config()).values
            st.session_state["history"].append({
                "brief": final["original_brief"],
                "concept": final["concept"],
                "iterations": final["iteration"],
                "continuity_status": final["continuity_status"]
            })
            st.session_state["awaiting_human"] = False
            st.session_state["current_result"] = final
            st.rerun()

    with col_b:
        human_feedback = st.text_input("Or give your own feedback and revise:")
        if st.button("🔄 Revise with my feedback") and human_feedback:
            pipeline.update_state(config(), {"feedback": human_feedback, "verdict": "revise"})
            status_box = st.empty()
            for event in pipeline.stream(None, config(), stream_mode="updates"):
                for node_name in event:
                    if node_name in AGENT_LABELS:
                        icon, label, message = AGENT_LABELS[node_name]
                        status_box.info(f"{icon} **{label}** — {message}")
            status_box.empty()
            state_snapshot = pipeline.get_state(config())
            if state_snapshot.next:
                st.session_state["current_result"] = state_snapshot.values
            else:
                final = state_snapshot.values
                st.session_state["history"].append({
                    "brief": final["original_brief"],
                    "concept": final["concept"],
                    "iterations": final["iteration"],
                    "continuity_status": final["continuity_status"]
                })
                st.session_state["awaiting_human"] = False
                st.session_state["current_result"] = final
            st.rerun()

# ─── Final results ─────────────────────────────────────────────────────────────
if st.session_state["current_result"] and not st.session_state["awaiting_human"]:
    result = st.session_state["current_result"]
    st.divider()
    st.caption(f"✅ Completed in {result['iteration']} iteration(s)")

    st.subheader("💡 Final Concept")
    st.write(result["concept"])

    st.subheader("🎨 Generated Image")
    with st.spinner("Generating image..."):
        image_url = generate_image(result["image_prompt"])
    if image_url:
        st.image(image_url, use_container_width=True)
    else:
        st.info("Add OPENAI_API_KEY to your .env to generate images.")
        st.markdown("**Image prompt to paste into Midjourney or DALL-E:**")
        st.code(result["image_prompt"], language=None)

    with st.expander("🔍 Critic Feedback"):
        st.write(result["feedback"])

    with st.expander("🔗 Continuity Status"):
        status = result["continuity_status"]
        colour = "green" if status == "consistent" else "red"
        st.markdown(f":{colour}[**{status.upper()}**]")

    st.divider()
    st.caption("Made with mindFree · LangGraph + Claude")
