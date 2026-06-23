#!/usr/bin/env python3
"""
Cover Letter Generator - CrewAI Implementation

This tool uses AI agents to analyze job descriptions and generate
tailored cover letters based on your professional background.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    from instrumentation import setup_tracing
except ImportError:
    # instrumentation.py is gitignored (local-only Arize tracing); the tool runs
    # fine without it, so degrade to a no-op on a fresh clone.
    def setup_tracing():
        return None

ROOT_DIR = Path(__file__).parent.parent


def check_setup():
    """Verify that required files and environment variables are set up."""
    # Load environment variables
    load_dotenv(ROOT_DIR / ".env")

    # CrewAI requires OPENAI_API_KEY to be set at import time even when not using OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "NA"

    # Check for required API keys
    missing = []
    if not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY (needed by all Claude agents and judges)")
    if missing:
        print("Error: missing API keys in .env:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    # Check for Tavily API key (optional but recommended)
    if not os.getenv("TAVILY_API_KEY"):
        print("Note: TAVILY_API_KEY not found.")
        print("Company research will be limited without it.")
        print("Get a free key at https://tavily.com\n")

    # Check for experience file
    experience_file = ROOT_DIR / "references" / "my_experience.md"
    if not experience_file.exists():
        print("Error: my_experience.md not found.")
        print("Please create this file with your professional background.")
        sys.exit(1)

    # Check if experience file has been filled in
    content = experience_file.read_text()
    if "[Your Full Name]" in content or "[your.email@example.com]" in content:
        print("Warning: my_experience.md appears to still have placeholder content.")
        print("Please fill in your actual professional background for best results.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            sys.exit(0)


def get_job_info():
    """Get job title, company name, and description from user input."""
    print("\n" + "=" * 60)
    print("COVER LETTER GENERATOR")
    print("=" * 60)

    # Get job title
    print("\nJob Title:")
    job_title = input("> ").strip()
    if not job_title:
        print("Error: Job title is required.")
        sys.exit(1)

    # Get company name
    print("\nCompany Name:")
    company_name = input("> ").strip()
    if not company_name:
        print("Error: Company name is required.")
        sys.exit(1)

    # Get job description
    print("\nPaste the job description below.")
    print("When finished, enter a blank line followed by 'END' on its own line.\n")

    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        except EOFError:
            break

    job_description = "\n".join(lines).strip()

    if not job_description:
        print("Error: No job description provided.")
        sys.exit(1)

    return job_title, company_name, job_description


def main():
    """Main entry point."""
    check_setup()

    tracing_provider = setup_tracing()

    from crew import generate_cover_letter, revise_cover_letter

    job_title, company_name, job_description = get_job_info()

    print("\n" + "=" * 60)
    print(f"Generating cover letter for: {job_title} at {company_name}")
    print("This may take a minute as our AI agents analyze and write.")
    print("=" * 60 + "\n")

    try:
        cover_letter = generate_cover_letter(job_title, company_name, job_description)

        # Revision loop
        while True:
            print("\n" + "=" * 60)
            print("YOUR COVER LETTER")
            print("=" * 60 + "\n")
            print(cover_letter)
            print("\n" + "=" * 60)

            # Ask if they want to revise
            print("\nOptions:")
            print("  [r] Request changes")
            print("  [s] Save and exit")
            print("  [q] Quit without saving")
            choice = input("\nChoice (r/s/q): ").strip().lower()

            if choice == "r":
                print("\nWhat would you like to change?")
                print("(Describe your requested changes, then press Enter twice)\n")

                feedback_lines = []
                empty_count = 0
                while empty_count < 1:
                    line = input()
                    if line == "":
                        empty_count += 1
                    else:
                        empty_count = 0
                        feedback_lines.append(line)

                feedback = "\n".join(feedback_lines).strip()
                if feedback:
                    print("\n" + "=" * 60)
                    print("Revising cover letter...")
                    print("=" * 60 + "\n")
                    cover_letter = revise_cover_letter(
                        cover_letter, feedback, job_title, company_name
                    )
                else:
                    print("No changes requested.")

            elif choice == "s":
                filename = input("Filename (default: cover_letter.txt): ").strip()
                if not filename:
                    filename = "cover_letter.txt"
                save_dir = ROOT_DIR / "saved_letters"
                save_dir.mkdir(exist_ok=True)
                save_path = save_dir / filename
                save_path.write_text(cover_letter)
                print(f"Cover letter saved to {save_path}")
                break

            elif choice == "q":
                print("Exiting without saving.")
                break

            else:
                print("Invalid choice. Please enter 'r', 's', or 'q'.")

    except Exception as e:
        print(f"\nError generating cover letter: {e}")
        sys.exit(1)
    finally:
        if tracing_provider:
            tracing_provider.force_flush()
            tracing_provider.shutdown()


if __name__ == "__main__":
    main()
