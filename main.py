"""
mindFree — CLI entry point.
Run the full multi-agent pipeline from the terminal.
For the web UI, use: ./venv/bin/streamlit run app.py
"""

import base64
import os
import sys
from dotenv import load_dotenv

load_dotenv()


def check_env():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_anthropic_api_key_here":
        print("Error: ANTHROPIC_API_KEY is not set in your .env file.")
        print("Copy .env.example to .env and add your key.")
        sys.exit(1)


def main():
    check_env()

    from agents.continuitiy_checker import validate_brief
    from graph.creative_graph import pipeline_auto

    print("mindFree — Multi-Agent Creative Collaboration")
    print("=" * 50)

    brief = input("Enter a creative brief: ").strip()
    if not brief:
        print("Brief cannot be empty.")
        sys.exit(1)

    # Validate brief before running
    print("Checking brief...")
    validation = validate_brief(brief)
    if not validation.valid:
        print(f"\nBrief issue: {validation.reason}")
        sys.exit(1)

    # Optional reference image
    image_path = input("Reference image path (or press Enter to skip): ").strip()
    image_data = ""
    if image_path:
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            print(f"Image loaded: {image_path}")
        except FileNotFoundError:
            print(f"Image not found: {image_path} — continuing without it.")

    print("\nAgents are collaborating...\n")

    result = pipeline_auto.invoke({
        "concept": brief,
        "original_brief": brief,
        "feedback": "",
        "continuity_feedback": "",
        "verdict": "",
        "continuity_status": "",
        "iteration": 0,
        "image_input": image_data,
        "image_prompt": ""
    })

    print("─" * 50)
    print("FINAL CONCEPT")
    print("─" * 50)
    print(result["concept"])

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
