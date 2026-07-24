from typing import TypedDict


class CreativeState(TypedDict):
    concept: str            # the evolving creative concept
    feedback: str           # critic or continuity feedback
    verdict: str            # "accept" or "revise"
    original_brief: str     # the user's original input, never overwritten
    continuity_status: str  # "consistent" or "drifted"
    iteration: int          # tracks how many revision loops have run
    image_input: str        # base64-encoded image from user (optional, "" if none)
    image_prompt: str       # generated image prompt for Midjourney/DALL-E etc.
