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


llm = ChatAnthropic(model="claude-haiku-4-5", max_tokens=512)


def return_ideas(state: CreativeState) -> CreativeState:
    # Build the human message content
    content: Union[str, list] = []

    # If there's feedback from a previous revision pass, include it
    user_text = state["concept"]
    if state["feedback"] and state["iteration"] > 0:
        user_text = (
            f"Original brief: {state['original_brief']}\n\n"
            f"Previous feedback: {state['feedback']}\n\n"
            f"Please revise the concept accordingly."
        )

    # If the user supplied an image, send it as a vision message
    if state["image_input"]:
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
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
