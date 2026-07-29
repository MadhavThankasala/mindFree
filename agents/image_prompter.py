"""
Image Prompter agent.
Takes the final accepted concept and generates a detailed image generation
prompt suitable for Midjourney, DALL-E, or Stable Diffusion.
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import CreativeState

load_dotenv()

IMAGE_PROMPTER_SYSTEM_PROMPT = """\
Write a visual description for an AI image generator (Midjourney, DALL-E, \
Stable Diffusion). Based on the concept, describe what you actually see in \
the image — not the theme, not the metaphor, the literal visual scene. \
Include: what's in the frame, the light, the mood, the colour palette, \
the style (photography, illustration, film still, etc.). \
Be specific and concrete. No abstract nouns. Output only the prompt — \
no labels, no explanation.
"""

llm = ChatAnthropic(model="claude-haiku-4-5", max_tokens=300)


def image_prompter_node(state: CreativeState) -> CreativeState:
    response = llm.invoke([
        SystemMessage(content=IMAGE_PROMPTER_SYSTEM_PROMPT),
        HumanMessage(content=state["concept"])
    ])
    return {**state, "image_prompt": response.content}
