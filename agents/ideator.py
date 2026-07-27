import base64
from typing import Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from graph.state import CreativeState
import os
from dotenv import load_dotenv

load_dotenv()

IDEATOR_SYSTEM_PROMPT = """\
You are an imaginative creative collaborator. Given a topic or brief, generate \
a rich, compelling creative concept in 3-5 sentences. Be specific and original. \
If feedback from a previous review is provided, incorporate it into your revision.
"""


def _detect_image_media_type(b64_data: str) -> str:
    """Sniff the image format from its magic bytes; default to jpeg if unrecognized."""
    try:
        header = base64.b64decode(b64_data[:16])
    except Exception:
        return "image/jpeg"
    if header.startswith(b"\x89PNG"):
        return "image/png"
    if header.startswith(b"GIF8"):
        return "image/gif"
    if header[:4] == b"RIFF":
        return "image/webp"
    return "image/jpeg"


llm = ChatAnthropic(model="claude-haiku-4-5", max_tokens=512)


def return_ideas(state: CreativeState) -> CreativeState:
    # Build the human message content
    content: Union[str, list] = []

    # If there's feedback from a previous revision pass, include it
    user_text = state["concept"]
    has_continuity_issue = state["continuity_status"] == "drifted" and state["continuity_feedback"]
    if state["iteration"] > 0 and (state["feedback"] or has_continuity_issue):
        feedback_parts = []
        if has_continuity_issue:
            feedback_parts.append(f"Continuity issue: {state['continuity_feedback']}")
        if state["feedback"]:
            feedback_parts.append(f"Critic feedback: {state['feedback']}")
        user_text = (
            f"Original brief: {state['original_brief']}\n\n"
            + "\n\n".join(feedback_parts)
            + "\n\nPlease revise the concept accordingly."
        )

    # If the user supplied an image, send it as a vision message
    if state["image_input"]:
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _detect_image_media_type(state["image_input"]),
                    "data": state["image_input"],
                },
            },
            {"type": "text", "text": user_text},
        ]
    else:
        content = user_text

    response = llm.invoke([
        SystemMessage(content=IDEATOR_SYSTEM_PROMPT),
        HumanMessage(content=content)
    ])
    return {**state, "concept": response.content, "iteration": state["iteration"] + 1}
