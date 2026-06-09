"""Pipeline agent factories.

Each agent in the cover-letter pipeline lives in its own module, named for the
step it performs (``1_...`` through ``4_...``). Those numeric-prefixed module
names aren't valid Python identifiers, so they're loaded via importlib and their
factory functions re-exported here.
"""

import importlib

build_job_analyst = importlib.import_module(
    "agents.1_senior_job_requirements_agent"
).build_job_analyst
build_experience_strategist = importlib.import_module(
    "agents.2_career_strategy_consultant_agent"
).build_experience_strategist
build_cover_letter_writer = importlib.import_module(
    "agents.3_professional_cover_letter_writer_agent"
).build_cover_letter_writer
build_authenticity_editor = importlib.import_module(
    "agents.4_senior_editorial_proofreader_agent"
).build_authenticity_editor

_llm_judges = importlib.import_module("agents.5_llm_judges")
JudgeSummary = _llm_judges.JudgeSummary
JUDGE_THRESHOLD = _llm_judges.JUDGE_THRESHOLD
run_single_judge = _llm_judges.run_single_judge
run_judge_panel = _llm_judges.run_judge_panel

__all__ = [
    "build_job_analyst",
    "build_experience_strategist",
    "build_cover_letter_writer",
    "build_authenticity_editor",
    "JudgeSummary",
    "JUDGE_THRESHOLD",
    "run_single_judge",
    "run_judge_panel",
]
