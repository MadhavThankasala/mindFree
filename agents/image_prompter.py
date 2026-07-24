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
You are an expert at writing image generation prompts for tools like Midjourney, \
DALL-E, and Stable Diffusion. Given a creative concept, write a single detailed \
image generation prompt that visually captures its essence. \
Include style, mood, lighting, composition, and any relevant visual details. \
Output only the prompt text — no explanation, no preamble.
"""

llm = ChatAnthropic(model="claude-haiku-4-5", max_tokens=300)


def image_prompter_node(state: CreativeState) -> CreativeState:
    response = llm.invoke([
        SystemMessage(content=IMAGE_PROMPTER_SYSTEM_PROMPT),
        HumanMessage(content=state["concept"])
    ])
    return {**state, "image_prompt": response.content}
