"""Support task: fix a finished letter that's outside the length target.

Not a numbered pipeline step — this runs only when the deterministic
``check_length`` finds the final draft outside 1600-1900 characters. Re-runs the
Authenticity Editor to trim or expand the letter while preserving its content,
structure, and tone.
"""

from crewai import Task


def build_length_fix_task(agent, draft, length_feedback):
    """Build a one-off editor task that fixes the letter's length."""
    return Task(
        description=f"""The cover letter below is outside the required length and must be
        fixed WITHOUT changing its meaning, structure, or any concrete facts.

        LENGTH PROBLEM:
        {length_feedback}

        RULES:
        - Keep the paragraph structure (intro, two body paragraphs, closing).
        - Keep every real fact, experience, metric, and the greeting/sign-off.
        - Do NOT add any new facts about the candidate. Do NOT fabricate.
        - Only adjust wording and density to land within 1600-1900 characters.
        - Keep the authentic, human tone. No AI clichés. No em dashes.

        CURRENT LETTER:
        {draft}

        Output the corrected cover letter only - no commentary.""",
        expected_output="""The cover letter, rewritten to fall within 1600-1900
        characters, with all facts, structure, and tone preserved. Ready to send.""",
        agent=agent,
    )
