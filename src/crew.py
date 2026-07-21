from pathlib import Path

from crewai import Agent, Task, Crew, Process
from tools import search_company
from models import models

from agents import (
    build_authenticity_editor,
    build_company_researcher,
    build_cover_letter_writer,
    build_experience_matcher,
    build_job_analyst,
)
from tasks import (
    build_analyze_job_task,
    build_company_research_task,
    build_length_fix_task,
    build_match_experience_task,
    build_proofread_task,
    build_write_cover_letter_task,
)
from length_check import check_length

_ROOT_DIR = Path(__file__).parent.parent

# How many times to re-run the editor to pull an out-of-range letter into the
# 1600-1900 character target before giving up and returning the closest draft.
MAX_LENGTH_FIX_ATTEMPTS = 2


def _load_examples_section(examples_file: str) -> str:
    """Load the optional good-examples block injected into writer/editor prompts."""
    try:
        with open(examples_file, "r") as f:
            content = f.read().strip()
    except FileNotFoundError:
        return ""

    if not content:
        return ""

    return f"""

STYLE AND TONE REFERENCE — GOOD COVER LETTER EXAMPLES:
The following are examples of strong cover letters. Study them for tone, sentence
variety, specificity, and structure. Do NOT copy their content — use them only as
a model for how the final letter should feel.

{content}

END OF EXAMPLES"""


def create_cover_letter_crew(
    job_title: str,
    company_name: str,
    job_description: str,
    experience_file: str = str(_ROOT_DIR / "references" / "my_experience.md"),
    examples_file: str = str(_ROOT_DIR / "references" / "good_examples.md"),
):
    """Create and return the cover letter generation crew.

    Pipeline: Company Researcher (1A) and Job Description Analyst (1B) run in
    parallel; the Experience Matcher (2) consumes the job analysis; the Cover
    Letter Writer (3) consumes the company research and the experience match; the
    Authenticity Editor (4) proofreads the draft.
    """

    # Read the candidate's experience file up front and inject it directly into
    # the researcher's and matcher's tasks (below). We don't hand them a
    # FileReadTool: on CrewAI's native Anthropic provider, Sonnet 5 skips tool
    # calls when the task also has a structured (output_pydantic) output, so it
    # would never read the file. Injecting the content is deterministic and
    # model-agnostic.
    experience_content = Path(experience_file).read_text()

    examples_section = _load_examples_section(examples_file)

    # Pipeline agents (defined in src/agents/, one file per pipeline step)
    company_researcher = build_company_researcher(
        models["company_researcher"], tools=[search_company]
    )
    job_analyst = build_job_analyst(models["job_analyst"])
    experience_matcher = build_experience_matcher(models["experience_matcher"])
    cover_letter_writer = build_cover_letter_writer(models["writer"])
    authenticity_editor = build_authenticity_editor(models["editor"])

    # Pipeline tasks (defined in src/tasks/, one file per execution step).
    # 1A and 1B have no dependencies, so they run in parallel (async_execution).
    company_research_task = build_company_research_task(
        company_researcher,
        job_title,
        company_name,
        job_description,
        experience_content,
    )
    company_research_task.async_execution = True

    analyze_job_task = build_analyze_job_task(
        job_analyst,
        job_title,
        company_name,
        job_description,
    )
    analyze_job_task.async_execution = True

    match_experience_task = build_match_experience_task(
        experience_matcher,
        experience_content,
        context=[analyze_job_task],
    )

    write_cover_letter_task = build_write_cover_letter_task(
        cover_letter_writer,
        job_title,
        company_name,
        examples_section,
        context=[company_research_task, match_experience_task],
    )

    proofread_task = build_proofread_task(
        authenticity_editor,
        context=[company_research_task, write_cover_letter_task],
    )

    crew = Crew(
        agents=[
            company_researcher,
            job_analyst,
            experience_matcher,
            cover_letter_writer,
            authenticity_editor,
        ],
        tasks=[
            company_research_task,
            analyze_job_task,
            match_experience_task,
            write_cover_letter_task,
            proofread_task,
        ],
        process=Process.sequential,
        verbose=True,
    )

    return crew


def _clean_letter(text: str) -> str:
    return text.replace("—", ", ").replace("--", ", ")


def _run_length_fix(draft: str, length_feedback: str) -> str:
    """Re-run the Authenticity Editor to pull the letter into the length target."""
    editor = build_authenticity_editor(models["editor"])
    task = build_length_fix_task(editor, draft, length_feedback)
    crew = Crew(
        agents=[editor],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )
    return str(crew.kickoff())


def _enforce_length(letter: str) -> str:
    """Return ``letter`` unchanged if within the length target, otherwise re-run
    the editor up to ``MAX_LENGTH_FIX_ATTEMPTS`` times to fix it (deterministic
    character-count check, no LLM judges)."""
    for _ in range(MAX_LENGTH_FIX_ATTEMPTS):
        passed, feedback = check_length(letter)
        if passed:
            return letter
        print(f"\nLength check: {feedback}\nRe-running editor to fix length...")
        letter = _clean_letter(_run_length_fix(letter, feedback))

    passed, feedback = check_length(letter)
    if not passed:
        print(f"\nLength check: {feedback}\nReturning closest draft after "
              f"{MAX_LENGTH_FIX_ATTEMPTS} fix attempt(s).")
    return letter


def generate_cover_letter(
    job_title: str,
    company_name: str,
    job_description: str,
    experience_file: str = str(_ROOT_DIR / "references" / "my_experience.md"),
) -> str:
    """Generate a cover letter for the given job.

    Runs the crew (company research + job analysis in parallel -> experience
    matching -> writing -> authenticity editing), then enforces the 1600-1900
    character target with a deterministic length check and a bounded editor
    re-run (``_enforce_length``), and returns the final draft.

    Deliberately NOT wrapped in a single outer span: a root span held open across
    the whole run outlives Arize's trace-assembly window, so its children get
    finalized as a rootless trace before the root arrives. The crew run is its
    own self-rooted trace.
    """
    crew = create_cover_letter_crew(job_title, company_name, job_description, experience_file)
    result = _clean_letter(str(crew.kickoff()))
    return _enforce_length(result)


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
        llm=models["revision_agent"],
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
