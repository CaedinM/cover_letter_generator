"""Pipeline step 1B: Job Description Analyst.

Extracts the must-haves, nice-to-haves, key responsibilities, and technical
requirements from the job listing. Has no tools — it works purely from the job
description text. Runs in parallel with the Company Researcher (step 1A).
"""

from crewai import Agent


def build_job_analyst(llm):
    """Build the Job Description Analyst agent (pipeline step 1B)."""
    return Agent(
        role="Job Description Analyst",
        goal="Extract the essential requirements, responsibilities, and technologies "
        "from a job description into a clean, structured brief",
        backstory="""You dissect job postings with precision. You separate the must-have
        requirements a hiring manager truly cares about from the nice-to-haves, surface
        the core responsibilities of the role, and pull out the specific technologies and
        tools it calls for. Your structured read of the job gives the downstream steps a
        clear target to match the candidate's experience against.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
