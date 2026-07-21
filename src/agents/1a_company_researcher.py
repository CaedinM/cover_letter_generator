"""Pipeline step 1A: Company Researcher.

Distills what the company is and how the candidate aligns with its mission and
culture. Uses the ``search_company`` tool when the job listing is thin on
company context. Runs in parallel with the Job Description Analyst (step 1B).
"""

from crewai import Agent


def build_company_researcher(llm, tools):
    """Build the Company Researcher agent (pipeline step 1A)."""
    return Agent(
        role="Company Researcher",
        goal="Distill what the company does and find genuine alignment between the "
        "candidate's background and the company's mission and culture",
        backstory="""You research companies to capture, in a sentence or two, what they
        do and what they stand for. You read between the lines of a job posting and,
        when the details are thin, you search the web for the company's mission, culture,
        and recent news. You have a sharp eye for authentic alignment between a
        candidate's background and a company's values, and you never manufacture a
        connection that isn't there.""",
        verbose=True,
        allow_delegation=False,
        tools=tools,
        llm=llm,
    )
