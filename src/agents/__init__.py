"""Pipeline agent factories.

Each agent in the cover-letter pipeline lives in its own module, named for the
step it performs (``1a`` / ``1b`` run in parallel, then ``2`` through ``4``).
Those numeric-prefixed module names aren't valid Python identifiers, so they're
loaded via importlib and their factory functions re-exported here.
"""

import importlib

build_company_researcher = importlib.import_module(
    "agents.1a_company_researcher"
).build_company_researcher
build_job_analyst = importlib.import_module(
    "agents.1b_job_analyst"
).build_job_analyst
build_experience_matcher = importlib.import_module(
    "agents.2_experience_matcher"
).build_experience_matcher
build_cover_letter_writer = importlib.import_module(
    "agents.3_writer"
).build_cover_letter_writer
build_authenticity_editor = importlib.import_module(
    "agents.4_editor"
).build_authenticity_editor

__all__ = [
    "build_company_researcher",
    "build_job_analyst",
    "build_experience_matcher",
    "build_cover_letter_writer",
    "build_authenticity_editor",
]
