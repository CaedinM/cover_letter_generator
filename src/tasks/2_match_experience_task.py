"""Pipeline step 2 task: match the candidate's experience to the role.

Assigned to the Experience Matcher agent. Depends on the job analysis (step 1B),
passed in via ``context``. Returns exactly the two strongest matching
experiences, each scoped for one body paragraph of the letter.
"""

from crewai import Task
from pydantic import BaseModel, Field


class MatchedExperience(BaseModel):
    """One of the candidate's experiences matched to a job responsibility."""

    experience_description: str = Field(
        description="Description of the candidate's project or experience, drawn only "
        "from the experience file."
    )
    maps_to_responsibility: str = Field(
        description="The key responsibility (or requirement) from the job description "
        "that this experience maps to."
    )
    technical_tools: list[str] = Field(
        description="Short list of the technical tools or methodologies used in this "
        "experience, prioritizing ones that also appear in the job's technical "
        "requirements."
    )


class ExperienceMatch(BaseModel):
    """The two strongest matching experiences to feed the writer."""

    experiences: list[MatchedExperience] = Field(
        description="Exactly two matching experiences, one per body paragraph of the "
        "cover letter."
    )


def build_match_experience_task(agent, experience_content, context):
    """Build the experience-matching task (pipeline step 2).

    ``experience_content`` is the full text of the candidate's experience file,
    injected directly into the prompt rather than read via a tool (see the note
    in ``create_cover_letter_crew``).
    """
    return Task(
        description=f"""Using the job analysis provided (its key responsibilities and
        technical requirements) and the candidate's background below, select the TWO
        strongest experiences to feature in the cover letter's body paragraphs.

        For EACH of the two experiences:
        1. Describe the project or experience (from the experience file only).
        2. State which key responsibility (or requirement) from the job analysis it maps to.
        3. List the technical tools or methodologies it used, PRIORITIZING ones that also
           appear in the job's technical requirements.

        Pick two DIFFERENT experiences that together cover the role's most important
        responsibilities. Favor experiences whose tools overlap with the job's technical
        requirements.

        CRITICAL: Only use FACTS explicitly stated in the experience file.
        - Do NOT invent years of experience, job titles, companies, achievements, or metrics.
        - Do NOT assume or infer details not written in the file.
        - If something is unclear, leave it out rather than guessing.

        CANDIDATE EXPERIENCE FILE (the candidate's full background):
        --- BEGIN EXPERIENCE FILE ---
        {experience_content}
        --- END EXPERIENCE FILE ---""",
        expected_output="""A structured ExperienceMatch with an `experiences` list of
        EXACTLY TWO items. Each item has:
        - experience_description: the project/experience (ONLY from the file)
        - maps_to_responsibility: the job responsibility/requirement it satisfies
        - technical_tools: tools/methodologies used, prioritizing those in the job's
          technical requirements""",
        agent=agent,
        context=context,
        output_pydantic=ExperienceMatch,
    )
