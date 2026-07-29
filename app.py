import base64
import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from graph.creative_graph import pipeline
from agents.continuitiy_checker import validate_brief

load_dotenv()

st.set_page_config(page_title="mindFree", page_icon="🧠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 15% 0%, rgba(167,139,250,0.12), transparent 45%),
                radial-gradient(circle at 85% 15%, rgba(236,72,153,0.08), transparent 40%),
                #0d0d15;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: #12121b;
    border-right: 1px solid rgba(255,255,255,0.06);
}

.mf-hero { padding: 0.5rem 0 1.25rem 0; }
.mf-hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(90deg, #a78bfa 0%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.mf-hero p {
    color: #9a9ab0;
    font-size: 0.95rem;
    margin: 0.3rem 0 0 0;
}
.mf-tags { margin-top: 0.75rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
.mf-tag {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    padding: 0.22rem 0.65rem;
    border-radius: 999px;
    background: rgba(167,139,250,0.12);
    border: 1px solid rgba(167,139,250,0.35);
    color: #c9bcf9;
}

.mf-badge {
    display: inline-block;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.03em;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
}
.mf-badge-consistent { background: rgba(52,211,153,0.12); border: 1px solid rgba(52,211,153,0.4); color: #6ee7b7; }
.mf-badge-drifted { background: rgba(248,113,113,0.12); border: 1px solid rgba(248,113,113,0.4); color: #fca5a5; }

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px !important;
    border-color: rgba(255,255,255,0.08) !important;
    background: rgba(255,255,255,0.02);
}

.stButton > button {
    border-radius: 999px;
    font-weight: 600;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    border: 1px solid rgba(255,255,255,0.1);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(167,139,250,0.25);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #a78bfa, #ec4899);
    border: none;
}

.stTextArea textarea, .stTextInput input {
    border-radius: 12px !important;
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

[data-testid="stFileUploaderDropzone"] {
    border-radius: 12px;
    border: 1px dashed rgba(167,139,250,0.35) !important;
    background: rgba(167,139,250,0.03);
}

[data-testid="stExpander"] {
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08) !important;
    overflow: hidden;
}

[data-testid="stAlert"] {
    border-radius: 12px;
    background: rgba(167,139,250,0.08);
    border: 1px solid rgba(167,139,250,0.25);
}

hr { border-color: rgba(255,255,255,0.08) !important; }

.mf-stepper { display: flex; align-items: flex-start; margin: 0.75rem 0; }
.mf-step {
    display: flex; flex-direction: column; align-items: center; gap: 0.35rem;
    flex: 1; position: relative; opacity: 0.35; transition: opacity 0.2s ease;
}
.mf-step:not(:last-child)::after {
    content: ""; position: absolute; top: 18px; left: 55%; width: 90%; height: 2px;
    background: rgba(255,255,255,0.1); z-index: 0;
}
.mf-step-icon {
    width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center;
    justify-content: center; background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1); font-size: 1.1rem; z-index: 1;
}
.mf-step-label { font-size: 0.7rem; font-weight: 600; color: #9a9ab0; text-align: center; }
.mf-step-active { opacity: 1; }
.mf-step-active .mf-step-icon {
    background: linear-gradient(135deg, #a78bfa, #ec4899); border-color: transparent;
    box-shadow: 0 0 0 4px rgba(167,139,250,0.15);
}
.mf-step-active .mf-step-label { color: #eef0f5; }
.mf-step-done { opacity: 0.7; }
.mf-step-done .mf-step-icon { background: rgba(52,211,153,0.15); border-color: rgba(52,211,153,0.4); }
</style>
""", unsafe_allow_html=True)

# ─── Session state defaults ────────────────────────────────────────────────────
for key, default in {
    "brief": "",
    "history": [],
    "thread_id": str(uuid.uuid4()),
    "awaiting_human": False,
    "current_result": None,
    "image_url": None,
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

PIPELINE_STEPS = ["ideator", "continuity", "critic", "image_prompter"]


def render_stepper(container, current_node: str) -> None:
    """Render a horizontal step indicator, highlighting the active/completed stages."""
    idx = PIPELINE_STEPS.index(current_node) if current_node in PIPELINE_STEPS else -1
    parts = []
    for i, node in enumerate(PIPELINE_STEPS):
        icon, label, _ = AGENT_LABELS[node]
        if i < idx:
            css_class = "mf-step mf-step-done"
        elif i == idx:
            css_class = "mf-step mf-step-active"
        else:
            css_class = "mf-step"
        parts.append(
            f'<div class="{css_class}">'
            f'<span class="mf-step-icon">{icon}</span>'
            f'<span class="mf-step-label">{label}</span>'
            f'</div>'
        )
    container.markdown(f'<div class="mf-stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


def build_session_export(result: dict) -> str:
    return f"""# mindFree Session Export

## Original Brief
{result['original_brief']}

## Final Concept
{result['concept']}

## Critic Feedback
{result['feedback']}

## Continuity Status
{result['continuity_status'].upper()}

## Image Prompt
{result['image_prompt']}

## Iterations
{result['iteration']}
"""


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
st.markdown("""
<div class="mf-hero">
    <h1>🧠 mindFree</h1>
    <p>Multi-agent creative collaboration powered by LangGraph + Claude</p>
    <div class="mf-tags">
        <span class="mf-tag">IDEATOR</span>
        <span class="mf-tag">CONTINUITY CHECKER</span>
        <span class="mf-tag">CRITIC</span>
        <span class="mf-tag">IMAGE PROMPTER</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.container(border=True):
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
    # --- Validate brief before running pipeline ---
    with st.spinner("Checking your brief..."):
        validation = validate_brief(brief)
    if not validation.valid:
        st.error(f"**Your brief has an issue:** {validation.reason}")
        st.stop()

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
                render_stepper(status_box, node_name)

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
        st.session_state["image_url"] = None
        st.rerun()

# ─── Human-in-the-loop review ─────────────────────────────────────────────────
if st.session_state["awaiting_human"] and st.session_state["current_result"]:
    result = st.session_state["current_result"]
    st.divider()
    st.subheader("🔍 Critic paused — your review needed")
    review = st.container(border=True)
    with review:
        if result["continuity_status"] == "drifted":
            st.warning(
                "⚠️ **Continuity check flagged this concept as drifted** from the "
                "original brief. Review carefully before accepting — the system "
                "will not block you, this is just a heads-up."
            )
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
                        render_stepper(status_box, node_name)
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
            st.session_state["image_url"] = None
            st.rerun()

    with col_b:
        human_feedback = st.text_input("Or give your own feedback and revise:")
        if st.button("🔄 Revise with my feedback") and human_feedback:
            pipeline.update_state(config(), {"feedback": human_feedback, "verdict": "revise"})
            status_box = st.empty()
            for event in pipeline.stream(None, config(), stream_mode="updates"):
                for node_name in event:
                    if node_name in AGENT_LABELS:
                        render_stepper(status_box, node_name)
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
                st.session_state["image_url"] = None
            st.rerun()

# ─── Final results ─────────────────────────────────────────────────────────────
if st.session_state["current_result"] and not st.session_state["awaiting_human"]:
    result = st.session_state["current_result"]
    st.divider()
    st.caption(f"✅ Completed in {result['iteration']} iteration(s)")

    with st.container(border=True):
        col_export, _ = st.columns([1, 3])
        with col_export:
            st.download_button(
                "⬇️ Export session",
                data=build_session_export(result),
                file_name="mindfree_session.md",
                mime="text/markdown",
            )

        st.subheader("💡 Final Concept")
        st.write(result["concept"])

        st.subheader("🎨 Generated Image")
        # Only auto-generate once per result; regenerate only on explicit request
        # (avoids re-billing DALL-E on every unrelated rerun, e.g. expanding a section).
        if st.session_state["image_url"] is None:
            with st.spinner("Generating image..."):
                st.session_state["image_url"] = generate_image(result["image_prompt"]) or ""
        image_url = st.session_state["image_url"]

        if image_url:
            st.image(image_url, use_container_width=True)
            if st.button("🔁 Regenerate image"):
                with st.spinner("Generating a new image..."):
                    st.session_state["image_url"] = generate_image(result["image_prompt"]) or ""
                st.rerun()
        else:
            st.info("Add OPENAI_API_KEY to your .env to generate images.")
            st.markdown("**Image prompt to paste into Midjourney or DALL-E:**")
            st.code(result["image_prompt"], language=None)

        with st.expander("🔍 Critic Feedback"):
            st.write(result["feedback"])

        with st.expander("🔗 Continuity Status"):
            status = result["continuity_status"]
            badge_class = "mf-badge-consistent" if status == "consistent" else "mf-badge-drifted"
            st.markdown(
                f'<span class="mf-badge {badge_class}">{status.upper()}</span>',
                unsafe_allow_html=True
            )

    st.divider()
    st.caption("Made with mindFree · LangGraph + Claude")
