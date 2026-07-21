"""Pipeline step 1A task: research the company and the candidate's alignment.

Assigned to the Company Researcher agent. Runs in parallel with the job-analysis
task (step 1B). The candidate's background is injected directly into the prompt
so the agent can ground its alignment finding in real evidence.
"""

from crewai import Task
from pydantic import BaseModel, Field


class CompanyResearch(BaseModel):
    """What the company is, plus how the candidate aligns with it."""

    company_summary: str = Field(
        description="1-2 sentences on what the company is and what it does."
    )
    candidate_alignment: str = Field(
        description="1-2 sentences on how the candidate aligns with the company's "
        "mission or culture, grounded in specific evidence from the candidate's "
        "background. Leave empty if there is no genuine alignment — do not invent one."
    )


def build_company_research_task(
    agent, job_title, company_name, job_description, experience_content
):
    """Build the company-research task (pipeline step 1A)."""
    return Task(
        description=f"""Research {company_name} for a candidate applying to the
        "{job_title}" role, and assess how the candidate aligns with the company.

        STEP 1 — What is the company?
        First check the job description below: sometimes it already explains what the
        company is and what it stands for. If the description does not give you enough,
        USE YOUR SEARCH TOOL to look up {company_name}'s mission, culture, products, and
        recent news. Summarize what the company is in 1-2 sentences.

        STEP 2 — How does the candidate align?
        Using ONLY facts from the candidate's background below, write 1-2 sentences on
        how the candidate genuinely aligns with the company's mission or culture. Point
        to specific evidence (a motivation, a project, a value). If there is no real
        connection, leave the alignment empty — do NOT manufacture one.

        Keep it tight. This feeds the opening and closing of a cover letter, so it should
        be concise and specific, not a long research dump.

        Job Title: {job_title}
        Company: {company_name}

        Job Description:
        {job_description}

        CANDIDATE BACKGROUND (the candidate's full experience file):
        --- BEGIN EXPERIENCE FILE ---
        {experience_content}
        --- END EXPERIENCE FILE ---""",
        expected_output=f"""A structured CompanyResearch for {company_name}:
        - company_summary: 1-2 sentences on what {company_name} is and does
        - candidate_alignment: 1-2 sentences, grounded in the candidate's real
          background, on how they align with the company's mission/culture (empty if
          there is no genuine alignment)""",
        agent=agent,
        output_pydantic=CompanyResearch,
    )
