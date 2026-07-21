"""Pipeline step 3: Professional Cover Letter Writer.

Drafts the cover letter from the job analysis and experience-matching brief.
Runs on Sonnet.
"""

from crewai import Agent


def build_cover_letter_writer(llm):
    """Build the Cover Letter Writer agent (pipeline step 3)."""
    return Agent(
        role="Professional Cover Letter Writer",
        goal="Craft compelling, personalized cover letters that get interviews",
        backstory="""You are an award-winning professional writer who specializes in
        career documents. Your cover letters have helped candidates land positions at
        top companies. You know how to balance professionalism with personality, and
        you understand that a great cover letter tells a story that makes the reader
        want to meet the candidate.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
