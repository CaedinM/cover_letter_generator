"""Pipeline step 2: Career Strategy Consultant (the "Experience Strategist").

Reads the candidate's experience file and maps their background to the job
requirements surfaced in step 1. Runs on Sonnet.
"""

from crewai import Agent


def build_experience_strategist(llm, tools):
    """Build the Experience Strategist agent (pipeline step 2)."""
    return Agent(
        role="Career Strategy Consultant",
        goal="Match the candidate's background to job requirements and identify the strongest talking points",
        backstory="""You are a career coach who has helped thousands of professionals
        land their dream jobs. You excel at identifying which experiences and skills
        will resonate most with hiring managers. You know how to frame past experiences
        to align perfectly with job requirements, finding connections that candidates
        often miss themselves.""",
        verbose=True,
        allow_delegation=False,
        tools=tools,
        llm=llm,
    )
