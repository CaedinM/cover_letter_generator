"""Pipeline step 4 task: proofread the draft for authenticity and tone.

Assigned to the Authenticity Editor agent. Depends on the step 1 analysis and
step 3 draft, passed in via ``context``.

NOTE: the description is intentionally a plain (non-f) string to preserve the
original behavior exactly — the ``{examples_section}`` token below is therefore
a literal in the prompt, not interpolated. See the crew module for context.
"""

from crewai import Task


def build_proofread_task(agent, context):
    """Build the proofreading task (pipeline step 4)."""
    return Task(
        description="""Review the cover letter and edit it to ensure it sounds
        authentically human, stays CONCISE, and matches the optimal tone.

        VERIFY THIS STRUCTURE:
        - Paragraph 1: Candidate introduces themselves + why this role/company (3-4 sentences)
        - Paragraph 2 & 3: 2 specific experiences with contribution and impact (5-6 sentences)
        - Paragraph 3: Skills connection + specific excitement + closing (3-4 sentences)
        - Total: 1600-1900 characters

        MUST CHECK:
        - If paragraph 1 is missing personal info about the candidate, ADD IT
        - If paragraph 3 lacks genuine excitement about something SPECIFIC to the
          role or company mission, ADD IT (not generic enthusiasm)

        CRITICAL - REMOVE ANY FABRICATED INFORMATION:
        - Remove any specific "X years of experience" claims unless verified
        - Remove any achievements or metrics that seem invented
        - Remove any job titles or company names not in the experience brief
        - If something sounds made up or too specific to be true, DELETE IT
        - When in doubt, use vaguer but honest language

        REMOVE or REWRITE these AI red flags:
        - Robotic sentence patterns that all start the same way
        - Excessive formality or stiffness
        - Filler sentences that don't add value

        ENSURE the letter has:
        - Natural sentence variety (different lengths, structures)
        - Specific details that only a human would include
        - Appropriate tone for the industry (casual for startups, formal for finance/law)
        - Contractions where natural (I'm, I've, didn't)

        Reference the company research to calibrate the right tone.
        {examples_section}
        Output the final, edited cover letter only - no commentary.""",
        expected_output="""The final cover letter, edited for authenticity:
        - Paragraph 1 introduces the candidate AND shows company interest
        - Paragraph 2 has 2 specific experiences with impact
        - Paragraph 3 ties skills to role and closes
        - 1600-1900 characters total
        - Sounds like a real person wrote it
        - Free of AI clichés and buzzwords
        - Ready to send without further editing""",
        agent=agent,
        context=context,
    )
