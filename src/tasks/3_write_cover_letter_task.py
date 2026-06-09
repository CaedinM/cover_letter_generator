"""Pipeline step 3 task: write the cover letter draft.

Assigned to the Cover Letter Writer agent. Depends on the step 1 analysis and
step 2 brief, passed in via ``context``. ``examples_section`` is the optional
good-examples block injected into the prompt.
"""

from crewai import Task


def build_write_cover_letter_task(agent, job_title, company_name, examples_section, context):
    """Build the cover-letter-writing task (pipeline step 3)."""
    return Task(
        description=f"""Write a CONCISE cover letter for the "{job_title}" position
        at {company_name}. Use the job analysis and experience matching brief provided.

        The experience matching brief is a structured MatchingBrief object with these
        attributes — reference them as directed below:
        - motivations: the candidate's genuine motivations for this role/company
        - company_culture: how to frame the candidate's background to align with the culture
        - strongest_experiences: a mapping of each strong experience to the specific job
          requirements it satisfies
        - strongest_talking_points: the top talking points for this role

        3-PARAGRAPH STRUCTURE:

        PARAGRAPH 1 (3-4 sentences):
        - Introduce yourself: your background, education, and relevant expertise
        - State you're applying for {job_title} at {company_name}
        - Draw on the brief's `motivations` and `company_culture` attributes here. Show
          genuine interest by referencing something about {company_name}; if a motivation
          connects to the company's mission or values, reference it.
        - Connect your background to why this role appeals to you, using `motivations` where
          there is an obvious connection to this specific role.
        Example: "I am excited to apply for the X role at Y. As a data science graduate
        from UC Berkeley with experience at early-stage startups, I am excited by your focus
         on Z. My background in A and B makes me a strong fit for this role."

        PARAGRAPH 2 & 3 (5-6 sentences):
        - Base these paragraphs on the brief's `strongest_experiences` attribute. Each
          paragraph should cover a DIFFERENT entry from `strongest_experiences`.
        - Naturally transition from previous paragraph. Mention a specific skill that is highly relevant to the role.
        - Briefly describe the experience or project that the relevant skill was used in.
        - For each: what you did, your contribution, and the impact (2-3 sentences each)
        - Use specific details and metrics where possible
        - These should directly relate to the job requirements that experience satisfies (per the brief).

        PARAGRAPH 4 (3-4 sentences):
        - Briefly tie your skills to what the role needs
        - Express genuine excitement about something SPECIFIC: either an aspect of
          the role itself, the company's mission/product or a recent development at the company (use research findings)
        - Express confidence in your ability to succeed in the role due to specific skills you have.

        GUIDELINES:
        - Total length: 1600-1900 characters
        - Be specific and personal, not generic or dull.
        - Use "{company_name}" and "{job_title}" naturally
        - No placeholder brackets
        - NEVER directly state that a skill matches a job requirement (e.g., avoid
          "My skills in X align with what this role demands"). Instead, describe a
          project or experience that used the skill — let the reader draw the connection.
        {examples_section}
        CRITICAL - DO NOT FABRICATE:
        - Only include facts from the experience matching brief
        - Do NOT invent years of experience (e.g., don't say "3 years" unless stated)
        - Do NOT make up achievements, metrics, or job details
        - Do NOT assume the candidate has experience they don't have
        - If unsure about a detail, leave it out rather than guess""",
        expected_output=f"""A concise, personal cover letter for {job_title} at {company_name}:
        - Paragraph 1: Who you are + why this role/company (3-4 sentences)
        - Paragraph 2: 2 relevant projects or experiences with specific impact (5-6 sentences)
        - Paragraph 3: Skills tie-in + specific excitement + closing (3-4 sentences)
        - Total: 1600-1900 characters
        - Ends with genuine excitement about the role or company mission
        - Properly formatted with greeting and sign-off""",
        agent=agent,
        context=context,
    )
