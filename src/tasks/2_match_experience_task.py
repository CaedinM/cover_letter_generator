"""Pipeline step 2 task: match the candidate's experience to the requirements.

Assigned to the Experience Strategist agent. Depends on the step 1 analysis,
passed in via ``context``.
"""

from crewai import Task
from pydantic import BaseModel, Field


class MatchingBrief(BaseModel):
    """Structured strategic brief produced by the Experience Strategist."""

    strongest_experiences: dict[str, list[str]] = Field(
        description="Maps each of the candidate's strongest experiences/projects to "
        "the list of specific key requirements (from the job analysis) it satisfies."
    )
    motivations: list[str] = Field(
        description="Genuine motivations for wanting this role or company. Empty if none."
    )
    company_culture: str = Field(
        description="How to frame the candidate's background to align with the company culture."
    )
    strongest_talking_points: list[str] = Field(
        description="The top 2 strongest talking points for this specific role."
    )


def build_match_experience_task(agent, context):
    """Build the experience-matching task (pipeline step 2)."""
    return Task(
        description="""Using the job analysis provided and the candidate's background
        (read from the experience file), identify:

        1. Which of the candidate's projects or experiences directly match the must-have requirements.
        For each strong experience, list the SPECIFIC key requirements (from the job analysis) that it satisfies.
        2. Any motivations the candidate has that might explain why they are eager to land this role or work for this specific company.
        If there is not a strong connection, leave empty, do not force a connection that is not there.
        3. How to frame the candidate's background to align with the company culture.
        4. The top 2 strongest talking points for this specific role

        Read the candidate's experience file to get their full background.
        Create a strategic brief that will guide the cover letter writing.

        CRITICAL: Only use FACTS explicitly stated in the experience file.
        - Do NOT invent years of experience
        - Do NOT make up job titles, companies, or achievements
        - Do NOT assume or infer details not written in the file
        - If something is unclear, leave it out rather than guessing""",
        expected_output="""A structured matching brief with these fields:
        - strongest_experiences: a mapping where each key is one of the candidate's strongest
          experiences/projects (ONLY from the file) and each value is the list of specific key
          requirements from the job analysis that experience satisfies.
        - motivations: list of genuine motivations for this role/company (empty if none).
        - company_culture: how to frame the candidate's background to align with the company culture.
        - strongest_talking_points: the top 2 talking points for this role (ONLY from the file).""",
        agent=agent,
        context=context,
        output_pydantic=MatchingBrief,
    )
