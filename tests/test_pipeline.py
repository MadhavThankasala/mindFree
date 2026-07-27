"""
Tests for the multi-agent creative collaboration pipeline.
Run with: pytest tests/
"""

import pytest
from unittest.mock import MagicMock, patch
from graph.state import CreativeState


# ─── Helpers ───────────────────────────────────────────────────────────────────

def make_state(**overrides) -> CreativeState:
    """Return a fully populated CreativeState with sensible defaults."""
    base: CreativeState = {
        "concept": "A story about a robot learning to paint.",
        "original_brief": "a story about a robot",
        "feedback": "",
        "continuity_feedback": "",
        "verdict": "",
        "continuity_status": "",
        "iteration": 0,
        "image_input": "",
        "image_prompt": "",
    }
    base.update(overrides)
    return base


# ─── State schema tests ────────────────────────────────────────────────────────

def test_creative_state_has_required_fields():
    """CreativeState must contain all expected keys."""
    state = make_state()
    required = {
        "concept", "original_brief", "feedback", "continuity_feedback",
        "verdict", "continuity_status", "iteration", "image_input", "image_prompt"
    }
    assert required == set(state.keys())


def test_creative_state_iteration_is_int():
    state = make_state(iteration=2)
    assert isinstance(state["iteration"], int)


# ─── Ideator node tests ────────────────────────────────────────────────────────

def test_ideator_increments_iteration():
    """return_ideas must increment iteration by 1."""
    from agents.ideator import return_ideas

    mock_response = MagicMock()
    mock_response.content = "A robot discovers colour through broken pixels."

    with patch("agents.ideator.llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        state = make_state(iteration=0)
        result = return_ideas(state)

    assert result["iteration"] == 1


def test_ideator_updates_concept():
    """return_ideas must write the LLM response into state['concept']."""
    from agents.ideator import return_ideas

    mock_response = MagicMock()
    mock_response.content = "Revised concept text."

    with patch("agents.ideator.llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        state = make_state()
        result = return_ideas(state)

    assert result["concept"] == "Revised concept text."


def test_ideator_preserves_original_brief():
    """return_ideas must never overwrite original_brief."""
    from agents.ideator import return_ideas

    mock_response = MagicMock()
    mock_response.content = "Some new concept."

    with patch("agents.ideator.llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        state = make_state(original_brief="a story about a robot")
        result = return_ideas(state)

    assert result["original_brief"] == "a story about a robot"


# ─── Critic node tests ─────────────────────────────────────────────────────────

def test_critic_node_returns_verdict_and_feedback():
    """critic_node must write verdict and feedback to state."""
    from agents.critic import critic_node

    mock_result = MagicMock()
    mock_result.verdict = "accept"
    mock_result.feedback = "Strong concept with clear emotional arc."

    with patch("agents.critic.get_critic_model") as mock_model_fn:
        mock_model_fn.return_value.invoke.return_value = mock_result
        state = make_state()
        result = critic_node(state)

    assert result["verdict"] == "accept"
    assert result["feedback"] == "Strong concept with clear emotional arc."


def test_critic_preserves_original_brief():
    from agents.critic import critic_node

    mock_result = MagicMock()
    mock_result.verdict = "revise"
    mock_result.feedback = "Needs more specificity."

    with patch("agents.critic.get_critic_model") as mock_model_fn:
        mock_model_fn.return_value.invoke.return_value = mock_result
        state = make_state(original_brief="a story about a robot")
        result = critic_node(state)

    assert result["original_brief"] == "a story about a robot"


# ─── Continuity checker tests ──────────────────────────────────────────────────

def test_continuity_node_writes_to_continuity_feedback_not_feedback():
    """continuity_node must NOT overwrite state['feedback']."""
    from agents.continuitiy_checker import continuity_node

    mock_result = MagicMock()
    mock_result.status = "consistent"
    mock_result.explanation = "Concept aligns with brief."

    with patch("agents.continuitiy_checker.get_continuity_model") as mock_model_fn:
        mock_model_fn.return_value.invoke.return_value = mock_result
        state = make_state(feedback="Critic said: add more detail.")
        result = continuity_node(state)

    assert result["continuity_feedback"] == "Concept aligns with brief."
    assert result["feedback"] == "Critic said: add more detail."


def test_continuity_node_sets_status():
    from agents.continuitiy_checker import continuity_node

    mock_result = MagicMock()
    mock_result.status = "drifted"
    mock_result.explanation = "Concept has moved away from original subject."

    with patch("agents.continuitiy_checker.get_continuity_model") as mock_model_fn:
        mock_model_fn.return_value.invoke.return_value = mock_result
        state = make_state()
        result = continuity_node(state)

    assert result["continuity_status"] == "drifted"


# ─── Graph routing tests ───────────────────────────────────────────────────────

def test_route_accepts_when_verdict_accept():
    from graph.creative_graph import route_after_critic
    state = make_state(verdict="accept", iteration=1)
    assert route_after_critic(state) == "image_prompter"


def test_route_loops_when_verdict_revise():
    from graph.creative_graph import route_after_critic
    state = make_state(verdict="revise", iteration=1)
    assert route_after_critic(state) == "ideator"


def test_route_stops_at_max_iterations():
    from graph.creative_graph import route_after_critic
    state = make_state(verdict="revise", iteration=3)
    assert route_after_critic(state) == "image_prompter"
