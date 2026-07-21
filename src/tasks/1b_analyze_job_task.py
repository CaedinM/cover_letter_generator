"""Pipeline step 1B task: analyze the job description.

Assigned to the Job Description Analyst agent. Runs in parallel with the
company-research task (step 1A). Pure extraction from the job description — no
web research, no company culture (that lives in step 1A).
"""

from crewai import Task
from pydantic import BaseModel, Field


class JobAnalysis(BaseModel):
    """Structured analysis of a job posting produced by the Job Analyst."""

    must_have_requirements: list[str] = Field(
        description="Essential skills, experience, and qualifications, ordered from "
        "highest to lowest priority. These are the key requirements the candidate's "
        "experience will be matched against downstream."
    )
    nice_to_have_requirements: list[str] = Field(
        description="Preferred but non-essential requirements."
    )
    key_responsibilities: list[str] = Field(
        description="The main responsibilities and duties of the role."
    )
    technical_requirements: list[str] = Field(
        description="Specific technologies, tools, or methodologies mentioned."
    )


def build_analyze_job_task(agent, job_title, company_name, job_description):
    """Build the job-analysis task (pipeline step 1B)."""
    return Task(
        description=f"""Analyze the following job description for the position of
        "{job_title}" at "{company_name}" and extract a structured brief.

        Extract:
        1. MUST-HAVE requirements — the skills, experience, and qualifications that are
           essential, ordered from highest to lowest priority.
        2. NICE-TO-HAVE requirements — preferred but non-essential.
        3. KEY RESPONSIBILITIES — the main duties of the role.
        4. TECHNICAL REQUIREMENTS — the specific technologies, tools, or methodologies
           the role mentions.

        Work only from the job description text below. Do not research the company or
        speculate beyond what the posting states.

        Job Title: {job_title}
        Company: {company_name}

        Job Description:
        {job_description}""",
        expected_output=f"""A structured JobAnalysis for {job_title} at {company_name}:
        - must_have_requirements: essential requirements, ordered by priority (highest first)
        - nice_to_have_requirements: preferred but non-essential requirements
        - key_responsibilities: main responsibilities of the role
        - technical_requirements: specific technologies, tools, or methodologies mentioned""",
        agent=agent,
        output_pydantic=JobAnalysis,
    )
