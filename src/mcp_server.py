#!/usr/bin/env python3
"""
MCP server wrapping the cover letter generator.

Exposes the CrewAI pipeline as two MCP tools with clear, structured I/O so other
agents can call it programmatically:

  - generate_cover_letter(job_title, company_name, job_description, experience_file?)
  - revise_cover_letter(current_letter, feedback, job_title, company_name)

Transport is stdio (the MCP default). The pipeline runs CrewAI with verbose=True
and prints progress with print(), all of which goes to STDOUT — the same stream
stdio MCP uses for JSON-RPC. We redirect that chatter to STDERR at the file
descriptor level for the duration of each crew run so it cannot corrupt the
protocol stream. Diagnostic logs are therefore safe to read on stderr.

Run it directly (so src/ is on sys.path, matching main.py):
    python src/mcp_server.py
"""

import contextlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

ROOT_DIR = Path(__file__).parent.parent


def _setup_env() -> None:
    """Load .env and validate required keys. Raises RuntimeError with a clear
    message (surfaced to the calling agent) instead of exiting the process."""
    load_dotenv(ROOT_DIR / ".env")

    # CrewAI requires OPENAI_API_KEY to be set at import time even when unused.
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "NA"

    missing = []
    if not os.getenv("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY")
    if not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if missing:
        raise RuntimeError(
            "Missing required API keys in .env: " + ", ".join(missing)
        )


@contextlib.contextmanager
def _stdout_to_stderr():
    """Redirect fd 1 (stdout) to fd 2 (stderr) for the duration of the block.

    Done at the OS file-descriptor level so it also captures output from
    CrewAI's rich console and any child processes — not just Python print().
    """
    sys.stdout.flush()
    saved_fd = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, 1)
        os.close(saved_fd)


_setup_env()

# Import after env setup: importing crew triggers the crewai import chain, which
# expects OPENAI_API_KEY to already be present.
from crew import generate_cover_letter as _generate, revise_cover_letter as _revise

# CRITICAL for stdio MCP: at import time, CrewAI (crewai/llm.py) globally replaces
# sys.stdout/sys.stderr with its own crewai.llm.FilteredStream wrapper (to filter
# litellm banners). For a stdio MCP server, sys.stdout IS the JSON-RPC channel, so
# that wrapper ends up sitting on the protocol stream — its extra locking/buffering
# corrupts or stalls the initialize handshake (symptom: the client hangs ~30s then
# reports "failed to connect"). Unwrap back to the real streams here. Runtime crew
# chatter is kept off the protocol stream by the per-call _stdout_to_stderr()
# redirect below, not by FilteredStream.
for _name in ("stdout", "stderr"):
    _stream = getattr(sys, _name)
    if type(_stream).__name__ == "FilteredStream":
        setattr(sys, _name, _stream._original_stream)

mcp = FastMCP("cover-letter-generator")


@mcp.tool()
def generate_cover_letter(
    job_title: str,
    company_name: str,
    job_description: str,
    experience_file: str | None = None,
) -> dict:
    """Generate a tailored cover letter for a job using the multi-agent pipeline.

    Runs the full CrewAI pipeline (job analysis -> experience matching -> writing
    -> authenticity editing) followed by an LLM-judge quality loop, and returns
    the best-scoring draft. Facts are drawn only from the candidate's experience
    file (references/my_experience.md by default); nothing is fabricated.

    Args:
        job_title: The job title being applied for, e.g. "Data Scientist".
        company_name: The hiring company's name.
        job_description: The full job description text.
        experience_file: Optional absolute path to an alternate experience
            markdown file. Defaults to references/my_experience.md.

    Returns:
        dict with: cover_letter (str), char_count (int), job_title,
        company_name.
    """
    kwargs = {}
    if experience_file:
        kwargs["experience_file"] = experience_file
    with _stdout_to_stderr():
        letter = _generate(job_title, company_name, job_description, **kwargs)
    return {
        "cover_letter": letter,
        "char_count": len(letter),
        "job_title": job_title,
        "company_name": company_name,
    }


@mcp.tool()
def revise_cover_letter(
    current_letter: str,
    feedback: str,
    job_title: str,
    company_name: str,
) -> dict:
    """Apply targeted, user-requested edits to an existing cover letter.

    Makes only the changes described in `feedback`, keeping the rest of the
    letter intact. Does not fabricate new facts about the candidate.

    Args:
        current_letter: The existing cover letter text to revise.
        feedback: Description of the changes to make.
        job_title: The job title (for context).
        company_name: The hiring company's name (for context).

    Returns:
        dict with: cover_letter (str), char_count (int).
    """
    with _stdout_to_stderr():
        letter = _revise(current_letter, feedback, job_title, company_name)
    return {
        "cover_letter": letter,
        "char_count": len(letter),
    }


if __name__ == "__main__":
    mcp.run()
