import os
from pathlib import Path

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from crewai_tools import FileReadTool
from tavily import TavilyClient

from agents import (
    JudgeSummary,
    build_authenticity_editor,
    build_cover_letter_writer,
    build_experience_strategist,
    build_job_analyst,
    run_judge_panel,
)
from tasks import (
    MatchingBrief,
    build_analyze_job_task,
    build_match_experience_task,
    build_proofread_task,
    build_write_cover_letter_task,
)

_ROOT_DIR = Path(__file__).parent.parent

MAX_JUDGE_ITERATIONS = 3


@tool("Company Research Tool")
def search_company(query: str) -> str:
    """Search the web for information about a company, including their mission, values, culture, recent news, and products. Use this to research companies mentioned in job descriptions."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Web search unavailable - TAVILY_API_KEY not set."

    client = TavilyClient(api_key=api_key)
    response = client.search(query=query, search_depth="advanced", max_results=5)

    results = []
    if response.get("answer"):
        results.append(f"Summary: {response['answer']}\n")

    for r in response.get("results", []):
        results.append(f"- {r.get('title', 'No title')}: {r.get('content', 'No content')[:500]}")

    return "\n".join(results) if results else "No results found."


def _run_editor_revision(
    draft: str,
    job_title: str,
    company_name: str,
    job_description: str,
    experience_file: str,
    failures: list[JudgeSummary],
) -> str:
    """Re-run the Authenticity Editor with targeted judge feedback. Returns revised letter."""
    examples_file = str(_ROOT_DIR / "references" / "good_examples.md")
    examples_section = ""
    try:
        with open(examples_file, "r") as f:
            content = f.read().strip()
        if content:
            examples_section = f"""

STYLE AND TONE REFERENCE — GOOD COVER LETTER EXAMPLES:
The following are examples of strong cover letters. Study them for tone, sentence
variety, specificity, and structure. Do NOT copy their content — use them only as
a model for how the final letter should feel.

{content}

END OF EXAMPLES"""
    except FileNotFoundError:
        pass

    failure_lines = "\n".join(
        f"- {f.dimension} (score {f.score}/10): {f.feedback}" for f in failures
    )

    authenticity_editor = build_authenticity_editor(LLM(model="claude-sonnet-4-5"))

    revision_task = Task(
        description=f"""Review the cover letter and edit it to ensure it sounds
        authentically human, stays CONCISE, and matches the optimal tone.

        VERIFY THIS STRUCTURE:
        - Paragraph 1: Candidate introduces themselves + why this role/company (3-4 sentences)
        - Paragraph 2 & 3: 2 specific experiences with contribution and impact (5-6 sentences)
        - Paragraph 4: Skills connection + specific excitement + closing (3-4 sentences)
        - Total: 1600-1900 characters

        MUST CHECK:
        - If paragraph 1 is missing personal info about the candidate, ADD IT
        - If the closing lacks genuine excitement about something SPECIFIC to the
          role or company mission, ADD IT (not generic enthusiasm)

        CRITICAL - REMOVE ANY FABRICATED INFORMATION:
        - Remove any specific "X years of experience" claims unless verified
        - Remove any achievements or metrics that seem invented
        - Remove any job titles or company names not in the experience brief
        - If something sounds made up or too specific to be true, DELETE IT
        - When in doubt, use vaguer but honest language

        REMOVE or REWRITE these AI red flags:
        - Robotic sentence patterns that all start the same way
        - Excessive formality or stiffness
        - Filler sentences that don't add value
        - Em dashes "--"

        ENSURE the letter has:
        - Natural sentence variety (different lengths, structures)
        - Specific details that only a human would include
        - Appropriate tone for the industry (casual for startups, formal for finance/law)
        - Contractions where natural (I'm, I've, didn't)

        This is the position of "{job_title}" at {company_name}. Job description for tone reference:
        {job_description}
        {examples_section}

        CURRENT DRAFT TO REVISE:
        {draft}

        QUALITY REVIEW FAILURES — MUST FIX ALL OF THESE:
        {failure_lines}

        Address every failure above while keeping the rest of the letter intact.
        Output the revised cover letter only - no commentary.""",
        expected_output="""The final cover letter, edited for authenticity:
        - Paragraph 1 introduces the candidate AND shows company interest
        - Paragraphs 2 & 3 have 2 specific experiences with impact
        - Paragraph 4 ties skills to role and closes
        - 1600-1900 characters total
        - Sounds like a real person wrote it
        - Free of AI clichés and buzzwords
        - Addresses all quality review failures
        - Ready to send without further editing""",
        agent=authenticity_editor,
    )

    crew = Crew(
        agents=[authenticity_editor],
        tasks=[revision_task],
        process=Process.sequential,
        verbose=True,
    )

    return str(crew.kickoff())


def create_cover_letter_crew(
    job_title: str,
    company_name: str,
    job_description: str,
    experience_file: str = str(_ROOT_DIR / "references" / "my_experience.md"),
    examples_file: str = str(_ROOT_DIR / "references" / "good_examples.md"),
):
    """Create and return the cover letter generation crew."""

    # Tool for reading the user's experience file
    file_read_tool = FileReadTool(file_path=experience_file)

    haiku = LLM(model="claude-haiku-4-5-20251001")
    sonnet = LLM(model="claude-sonnet-4-5")

    # Load example cover letters for style/tone reference
    examples_section = ""
    try:
        with open(examples_file, "r") as f:
            content = f.read().strip()
        if content:
            examples_section = f"""

STYLE AND TONE REFERENCE — GOOD COVER LETTER EXAMPLES:
The following are examples of strong cover letters. Study them for tone, sentence
variety, specificity, and structure. Do NOT copy their content — use them only as
a model for how the final letter should feel.

{content}

END OF EXAMPLES"""
    except FileNotFoundError:
        pass

    # Pipeline agents (defined in src/agents/, one file per pipeline step)
    job_analyst = build_job_analyst(haiku, tools=[search_company])
    experience_strategist = build_experience_strategist(sonnet, tools=[file_read_tool])
    cover_letter_writer = build_cover_letter_writer(sonnet)
    authenticity_editor = build_authenticity_editor(sonnet)

    # Pipeline tasks (defined in src/tasks/, one file per execution step)
    analyze_job_task = build_analyze_job_task(
        job_analyst, job_title, company_name, job_description
    )
    match_experience_task = build_match_experience_task(
        experience_strategist, context=[analyze_job_task]
    )
    write_cover_letter_task = build_write_cover_letter_task(
        cover_letter_writer,
        job_title,
        company_name,
        examples_section,
        context=[analyze_job_task, match_experience_task],
    )
    proofread_task = build_proofread_task(
        authenticity_editor, context=[analyze_job_task, write_cover_letter_task]
    )

    # Create the crew
    crew = Crew(
        agents=[job_analyst, experience_strategist, cover_letter_writer, authenticity_editor],
        tasks=[analyze_job_task, match_experience_task, write_cover_letter_task, proofread_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew


def _clean_letter(text: str) -> str:
    return text.replace("—", ", ").replace("--", ", ")


def generate_cover_letter(
    job_title: str,
    company_name: str,
    job_description: str,
    experience_file: str = str(_ROOT_DIR / "references" / "my_experience.md"),
) -> str:
    """Generate a cover letter for the given job, then run the judge quality loop.

    Deliberately NOT wrapped in a single outer span: a root span held open across
    the whole run (crew + every judge iteration) outlives Arize's trace-assembly
    window, so its children get finalized as a rootless trace before the root
    arrives. Instead the crew run is its own trace and each judge round groups
    under its own short-lived ``judge-panel`` span (see run_judge_panel).
    """
    crew = create_cover_letter_crew(job_title, company_name, job_description, experience_file)
    current_draft = str(crew.kickoff())

    # Pull the Experience Strategist's brief out of the completed crew so the
    # Relevancy judge can check each experience's requirement matches.
    strongest_experiences: dict[str, list[str]] = {}
    for task in crew.tasks:
        brief = getattr(getattr(task, "output", None), "pydantic", None)
        if isinstance(brief, MatchingBrief):
            strongest_experiences = brief.strongest_experiences
            break

    best_letter = current_draft
    best_avg = -1.0

    for iteration in range(1, MAX_JUDGE_ITERATIONS + 1):
        print(f"\n{'=' * 60}")
        print(f"QUALITY REVIEW — Iteration {iteration}/{MAX_JUDGE_ITERATIONS}")
        print(f"{'=' * 60}")
        print("Running 3 judges in parallel + deterministic length check...")

        summaries = run_judge_panel(
            current_draft,
            job_title,
            company_name,
            job_description,
            experience_file,
            strongest_experiences,
        )

        for s in summaries:
            status = "PASS" if s.passed else "FAIL"
            print(f"  [{status}] {s.dimension}: {s.score}/10 — {s.feedback}")

        avg = sum(s.score for s in summaries) / len(summaries)
        failures = [s for s in summaries if not s.passed]

        if avg > best_avg:
            best_avg = avg
            best_letter = current_draft

        if not failures:
            print(f"\nAll judges passed. Letter finalized after {iteration} iteration(s).")
            return _clean_letter(current_draft)

        if iteration == MAX_JUDGE_ITERATIONS:
            print(f"\nMax iterations reached. Returning best letter (avg {best_avg:.1f}/10).")
            return _clean_letter(best_letter)

        print(f"\n{len(failures)} dimension(s) failed — re-running editor with judge feedback...")
        current_draft = _run_editor_revision(
            current_draft, job_title, company_name, job_description, experience_file, failures
        )

    return _clean_letter(best_letter)


def revise_cover_letter(
    current_letter: str,
    feedback: str,
    job_title: str,
    company_name: str,
) -> str:
    """Revise a cover letter based on user feedback."""

    revision_agent = Agent(
        role="Cover Letter Editor",
        goal="Revise cover letters based on specific user feedback while maintaining quality",
        backstory="""You are a skilled editor who takes feedback and implements
        changes precisely. You understand how to incorporate requested changes
        while keeping the letter professional and authentic. You make exactly
        the changes requested - no more, no less.""",
        verbose=True,
        allow_delegation=False,
        llm=LLM(model="claude-haiku-4-5-20251001"),
    )

    revision_task = Task(
        description=f"""Revise this cover letter based on the user's feedback.

        CURRENT COVER LETTER:
        {current_letter}

        USER'S REQUESTED CHANGES:
        {feedback}

        INSTRUCTIONS:
        - Make ONLY the changes the user requested
        - Keep everything else the same
        - Maintain the same overall structure and length
        - Keep the tone authentic and natural
        - Do NOT add AI-sounding phrases
        - Do NOT fabricate any new information about the candidate

        Output the revised cover letter only - no commentary.""",
        expected_output=f"""The revised cover letter for {job_title} at {company_name}:
        - Incorporates the user's requested changes
        - Maintains professional quality
        - Ready to send""",
        agent=revision_agent,
    )

    crew = Crew(
        agents=[revision_agent],
        tasks=[revision_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return _clean_letter(str(result))
