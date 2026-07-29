"""
Image Prompter agent.
Takes the final accepted concept and generates a detailed image generation
prompt suitable for Midjourney, DALL-E, or Stable Diffusion.

If the user supplied a reference image, it is included so the prompt inherits
the visual style and mood from it.
"""

from typing import Union
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import CreativeState
from agents.llm_client import get_llm

IMAGE_PROMPTER_SYSTEM_PROMPT = """\
Write a visual description for an AI image generator (Midjourney, DALL-E, \
Stable Diffusion). Based on the concept, describe what you actually see in \
the image — not the theme, not the metaphor, the literal visual scene. \
Include: what's in the frame, the light, the mood, the colour palette, \
the style (photography, illustration, film still, etc.). \
Be specific and concrete. No abstract nouns. Output only the prompt — \
no labels, no explanation.

If a reference image is provided, carry forward its visual style, colour \
palette, and mood into the prompt.
"""

llm = get_llm(max_tokens=300)


def image_prompter_node(state: CreativeState) -> CreativeState:
    content: Union[str, list]
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
            {"type": "text", "text": state["concept"]},
        ]
    else:
        content = state["concept"]

    response = llm.invoke([
        SystemMessage(content=IMAGE_PROMPTER_SYSTEM_PROMPT),
        HumanMessage(content=content),
    ])
    return {**state, "image_prompt": response.content}
