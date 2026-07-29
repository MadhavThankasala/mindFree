"""
Formatter agent.
Takes the accepted concept and shapes it into a titled, structured artifact
whose form matches the creative mode (film treatment, story sketch, pitch, etc.).
"""

from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import CreativeState
from agents.llm_client import get_llm

_BASE_PROMPT = """\
Turn the concept below into a clean, finished artifact. \
Give it a title and shape it so it reads like the real thing — \
not a description of the real thing. \
No preamble, no "here is your…", no labels. Just the artifact itself.\
"""

_MODE_FORMAT = {
    "short film": """\
Format:

TITLE
[A one-line logline — the whole film in one sentence.]

SETUP
[2-3 sentences: who, where, what situation.]

TURN
[1-2 sentences: the thing that changes everything.]

RESOLUTION / FINAL IMAGE
[1-2 sentences: how it ends or the last thing we see.]""",

    "short story": """\
Format:

TITLE
[First paragraph — drop straight into the scene, no throat-clearing.]

[Continue the story, 3-4 short paragraphs total. End on the image or \
moment that the story was always pointing toward.]""",

    "startup pitch": """\
Format:

COMPANY / PRODUCT NAME

THE PROBLEM
[One sentence, punchy. The pain that exists right now.]

THE INSIGHT
[One sentence. What most people miss that makes this possible.]

THE PRODUCT
[2-3 sentences. What it does, who uses it, and why they switch.]

WHY NOW
[One sentence. What changed that makes this the right moment.]""",

    "song": """\
Format:

TITLE

MOOD / FEEL
[One line: the sonic and emotional register.]

CORE THEME
[One sentence: what this song is actually about underneath.]

VERSE DIRECTION
[2-3 sentences: what the verses explore, what world they build.]

CHORUS / HOOK IDEA
[1-2 sentences: the emotional peak, the line that sticks.]""",

    "general": """\
Format:

TITLE
[A short, memorable title for this concept.]

THE IDEA
[3-4 sentences: the concept written as vividly and concretely as possible.]""",
}


def _system_prompt(mode: str) -> str:
    fmt = _MODE_FORMAT.get(mode, _MODE_FORMAT["general"])
    return f"{_BASE_PROMPT}\n\n{fmt}"


llm = get_llm(max_tokens=700)


def formatter_node(state: CreativeState) -> CreativeState:
    mode = state.get("mode", "general")
    response = llm.invoke([
        SystemMessage(content=_system_prompt(mode)),
        HumanMessage(content=state["concept"]),
    ])
    # Store the formatted output back in concept so downstream (image prompter)
    # and the UI all see the final polished artifact.
    return {**state, "concept": response.content}
