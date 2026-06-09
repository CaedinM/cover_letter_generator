"""Pipeline step 1: Senior Job Requirements Analyst (the "Job Analyst").

Parses the job description for must-have and nice-to-have requirements and
researches the company through the search tool. Runs on Haiku.
"""

from crewai import Agent


def build_job_analyst(llm, tools):
    """Build the Job Analyst agent (pipeline step 1)."""
    return Agent(
        role="Senior Job Requirements Analyst",
        goal="Extract and prioritize key requirements, skills, and qualifications from job descriptions, and research companies to understand their culture and values",
        backstory="""You are an expert at analyzing job postings with over 15 years of
        experience in HR and recruitment. You can quickly identify the critical requirements
        that hiring managers truly care about versus nice-to-haves. You understand how to
        read between the lines to identify company culture and team dynamics from job
        descriptions. When company details are limited, you research online to gather
        insights about the company's mission, values, recent news, and culture.""",
        verbose=True,
        allow_delegation=False,
        tools=tools,
        llm=llm,
    )
