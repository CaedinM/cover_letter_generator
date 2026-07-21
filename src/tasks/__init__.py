"""Pipeline task factories.

Each task in the cover-letter crew lives in its own module, named for its step in
the pipeline (``1a`` / ``1b`` run in parallel, then ``2`` through ``4``). Those
numeric-prefixed module names aren't valid Python identifiers, so they're loaded
via importlib and their factory functions re-exported here.
"""

import importlib

_company_research = importlib.import_module("tasks.1a_company_research_task")
build_company_research_task = _company_research.build_company_research_task
CompanyResearch = _company_research.CompanyResearch

_analyze_job = importlib.import_module("tasks.1b_analyze_job_task")
build_analyze_job_task = _analyze_job.build_analyze_job_task
JobAnalysis = _analyze_job.JobAnalysis

_match_experience = importlib.import_module("tasks.2_match_experience_task")
build_match_experience_task = _match_experience.build_match_experience_task
ExperienceMatch = _match_experience.ExperienceMatch
MatchedExperience = _match_experience.MatchedExperience

build_write_cover_letter_task = importlib.import_module(
    "tasks.3_write_cover_letter_task"
).build_write_cover_letter_task
build_proofread_task = importlib.import_module(
    "tasks.4_proofread_task"
).build_proofread_task

# Support task (not a numbered pipeline step): fixes a letter outside the length target.
from tasks.length_fix_task import build_length_fix_task

__all__ = [
    "build_company_research_task",
    "CompanyResearch",
    "build_analyze_job_task",
    "JobAnalysis",
    "build_match_experience_task",
    "ExperienceMatch",
    "MatchedExperience",
    "build_write_cover_letter_task",
    "build_proofread_task",
    "build_length_fix_task",
]
