from typing import TypedDict, List


# Creative modes — each shapes the agent prompts and output format
CREATIVE_MODES = ["short film", "short story", "startup pitch", "song", "general"]


class CreativeState(TypedDict):
    concept: str               # the evolving creative concept
    feedback: str              # critic feedback
    verdict: str               # "accept" or "revise"
    original_brief: str        # the user's original input, never overwritten
    continuity_status: str     # "consistent" or "drifted"
    continuity_feedback: str   # continuity checker's explanation
    iteration: int             # tracks how many revision loops have run
    image_input: str           # base64-encoded image from user (optional, "" if none)
    image_prompt: str          # generated image prompt for Midjourney/DALL-E etc.
    mode: str                  # creative mode — one of CREATIVE_MODES
    concept_history: List[str] # snapshot of concept after each ideator pass


def make_initial_state(brief: str, image_data: str = "", mode: str = "general") -> CreativeState:
    """Single source of truth for constructing a fresh pipeline state."""
    return {
        "concept": brief,
        "original_brief": brief,
        "feedback": "",
        "continuity_feedback": "",
        "verdict": "",
        "continuity_status": "",
        "iteration": 0,
        "image_input": image_data,
        "image_prompt": "",
        "mode": mode,
        "concept_history": [],
    }
