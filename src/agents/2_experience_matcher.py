"""Pipeline step 2: Experience Matcher.

Maps the candidate's background to the role's key responsibilities and picks the
two strongest matching experiences to feed the writer. The experience file
contents are injected into this agent's task prompt (in
``build_match_experience_task``), so it is built with no tools.
"""

from crewai import Agent


def build_experience_matcher(llm):
    """Build the Experience Matcher agent (pipeline step 2)."""
    return Agent(
        role="Experience Matcher",
        goal="Match the candidate's strongest, most relevant experiences to the role's "
        "key responsibilities, highlighting the overlapping technical skills",
        backstory="""You are an expert at connecting a candidate's real experience to
        what a role actually requires. You scan a candidate's background and pick the few
        experiences that map most directly to the job's responsibilities, always
        surfacing the specific technical skills and tools that overlap with the role. You
        work only from what the candidate has genuinely done, never inventing details to
        force a match.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
