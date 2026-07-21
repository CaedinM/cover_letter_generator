"""Pipeline step 3 task: write the cover letter draft.

Assigned to the Cover Letter Writer agent. Depends on the company research (step
1A) and the experience match (step 2), passed in via ``context``.
``examples_section`` is the optional good-examples block injected into the prompt.
"""

from crewai import Task


def build_write_cover_letter_task(agent, job_title, company_name, examples_section, context):
    """Build the cover-letter-writing task (pipeline step 3)."""
    return Task(
        description=f"""Write a CONCISE cover letter for the "{job_title}" position
        at {company_name}. Use the company research and the experience match provided.

        You are given two structured inputs from upstream steps:

        COMPANY RESEARCH (a CompanyResearch object) — use for the opening and closing:
        - company_summary: 1-2 sentences on what {company_name} is and does
        - candidate_alignment: how the candidate genuinely aligns with the company's
          mission/culture (may be empty — if so, do not force it)

        EXPERIENCE MATCH (an ExperienceMatch object) — use for the two body paragraphs:
        - experiences: a list of exactly two matched experiences. Each has:
            - experience_description: the project/experience to describe
            - maps_to_responsibility: the job responsibility it satisfies
            - technical_tools: the relevant tools/methodologies to weave in

        3-PARAGRAPH STRUCTURE:

        PARAGRAPH 1 (3-4 sentences):
        - Introduce yourself: your background, education, and relevant expertise
        - State you're applying for {job_title} at {company_name}
        - Draw on the company research here. Show genuine interest by referencing
          something about {company_name} from `company_summary`, and if
          `candidate_alignment` is present, work that authentic connection in.
        - Connect your background to why this role appeals to you.
        Example: "I am excited to apply for the X role at Y. As a data science graduate
        from UC Berkeley with experience at early-stage startups, I am excited by your focus
         on Z. My background in A and B makes me a strong fit for this role."

        PARAGRAPH 2 & 3 (5-6 sentences):
        - Base these paragraphs on the `experiences` list. Each paragraph should cover a
          DIFFERENT experience from the list (paragraph 2 = first experience, paragraph 3
          = second).
        - Naturally transition from previous paragraph. Mention a specific skill (from that
          experience's `technical_tools`) that is highly relevant to the role.
        - Briefly describe the experience or project that the relevant skill was used in.
        - For each: what you did, your contribution, and the impact (2-3 sentences each)
        - These should directly relate to the responsibility that experience maps to.
        - Use specific details and metrics where possible

        PARAGRAPH 4 (3-4 sentences):
        - Briefly tie your skills to what the role needs
        - Express genuine excitement about something SPECIFIC: an aspect of the role
          itself, or the company's mission/product (use the company research findings)
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
        - Only include facts from the company research and experience match provided
        - Do NOT invent years of experience (e.g., don't say "3 years" unless stated)
        - Do NOT make up achievements, metrics, or job details
        - Do NOT assume the candidate has experience they don't have
        - If unsure about a detail, leave it out rather than guess""",
        expected_output=f"""A concise, personal cover letter for {job_title} at {company_name}:
        - Paragraph 1: Who you are + why this role/company (3-4 sentences)
        - Paragraphs 2 & 3: the two matched experiences with specific impact (5-6 sentences)
        - Paragraph 4: Skills tie-in + specific excitement + closing (3-4 sentences)
        - Total: 1600-1900 characters
        - Ends with genuine excitement about the role or company mission
        - Properly formatted with greeting and sign-off""",
        agent=agent,
        context=context,
    )
