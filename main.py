"""
mindFree — CLI entry point.
Run the full multi-agent pipeline from the terminal.
For the web UI, use: streamlit run app.py
"""

import base64
import sys
from dotenv import load_dotenv

load_dotenv()


def main():
    from agents.continuitiy_checker import validate_brief
    from graph.creative_graph import pipeline_auto
    from graph.state import make_initial_state, CREATIVE_MODES

    print("mindFree — Multi-Agent Creative Collaboration")
    print("=" * 50)

    brief = input("Enter a creative brief: ").strip()
    if not brief:
        print("Brief cannot be empty.")
        sys.exit(1)

    print(f"Modes: {', '.join(CREATIVE_MODES)}")
    mode = input("Mode (press Enter for 'general'): ").strip() or "general"
    if mode not in CREATIVE_MODES:
        print(f"Unknown mode '{mode}', defaulting to 'general'.")
        mode = "general"

    print("Checking brief...")
    validation = validate_brief(brief)
    if not validation.valid:
        print(f"\nBrief issue: {validation.reason}")
        sys.exit(1)

    image_path = input("Reference image path (or press Enter to skip): ").strip()
    image_data = ""
    if image_path:
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            print(f"Image loaded: {image_path}")
        except FileNotFoundError:
            print(f"Image not found: {image_path} — continuing without it.")

    print("\nCollaborating...\n")

    result = pipeline_auto.invoke(make_initial_state(brief, image_data, mode))

    print("─" * 50)
    print("FINAL OUTPUT")
    print("─" * 50)
    print(result["concept"])

    history = result.get("concept_history", [])
    if len(history) > 1:
        print(f"\n({len(history)} drafts produced — run the web UI to browse them)")

    print("\n─" * 50)
    print("CRITIC FEEDBACK")
    print("─" * 50)
    print(result["feedback"])

    print("\n─" * 50)
    print("CONTINUITY")
    print("─" * 50)
    status = result["continuity_status"].upper() if result["continuity_status"] else "N/A"
    print(f"Status: {status}")
    if result["continuity_feedback"]:
        print(result["continuity_feedback"])

    print("\n─" * 50)
    print("IMAGE PROMPT")
    print("─" * 50)
    print(result["image_prompt"])

    print(f"\nCompleted in {result['iteration']} iteration(s).")


if __name__ == "__main__":
    main()
