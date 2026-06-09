"""Pipeline step 1 task: analyze the job description and research the company.

Assigned to the Job Analyst agent. First task in the sequential crew.
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
    company_culture: str = Field(
        description="Summary of the company's culture and values, drawn from both the "
        "job description and company research."
    )
    company_research: str = Field(
        description="Company research findings: mission, recent news/achievements, "
        "products or notable projects, and any unique aspects a candidate could reference."
    )


def build_analyze_job_task(agent, job_title, company_name, job_description):
    """Build the job-analysis task (pipeline step 1)."""
    return Task(
        description=f"""Analyze the following job description for the position of
        "{job_title}" at "{company_name}".

        USE YOUR SEARCH TOOL to research {company_name} online to gather context about
        their mission, values, culture, recent news, and what makes them unique.

        Extract and research:
        1. MUST-HAVE requirements (skills, experience, qualifications that are essential)
        2. NICE-TO-HAVE requirements (preferred but not essential)
        3. Company culture signals (values, work style, team dynamics)
        4. Key responsibilities of the role
        5. Any specific technologies, tools, or methodologies mentioned
        6. COMPANY RESEARCH for {company_name}:
           - Company mission and values
           - Recent news or achievements
           - Company culture and work environment
           - Products, services, or notable projects
           - Any unique aspects that a candidate could reference

        Job Title: {job_title}
        Company: {company_name}

        Job Description:
        {job_description}

        Provide a structured analysis that will help match a candidate's experience to this role.""",
        expected_output=f"""A structured JobAnalysis for {job_title} at {company_name} with these fields:
        - must_have_requirements: essential requirements, ordered by priority (highest first)
        - nice_to_have_requirements: preferred but non-essential requirements
        - key_responsibilities: main responsibilities of the role
        - technical_requirements: specific technologies, tools, or methodologies mentioned
        - company_culture: summary of culture and values (from job description AND research)
        - company_research: mission, recent news, products, and unique talking points""",
        agent=agent,
        output_pydantic=JobAnalysis,
    )
