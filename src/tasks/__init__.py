"""Pipeline task factories.

Each task in the cover-letter crew lives in its own module, named for its order
in the sequential execution pipeline (``1_...`` through ``4_...``). Those
numeric-prefixed module names aren't valid Python identifiers, so they're loaded
via importlib and their factory functions re-exported here.
"""

import importlib

_analyze_job = importlib.import_module("tasks.1_analyze_job_task")
build_analyze_job_task = _analyze_job.build_analyze_job_task
JobAnalysis = _analyze_job.JobAnalysis
_match_experience = importlib.import_module("tasks.2_match_experience_task")
build_match_experience_task = _match_experience.build_match_experience_task
MatchingBrief = _match_experience.MatchingBrief
build_write_cover_letter_task = importlib.import_module(
    "tasks.3_write_cover_letter_task"
).build_write_cover_letter_task
build_proofread_task = importlib.import_module(
    "tasks.4_proofread_task"
).build_proofread_task

_judge_prompts = importlib.import_module("tasks.5_judge_prompts")
JUDGE_DIMENSIONS = _judge_prompts.JUDGE_DIMENSIONS
PASS_FAIL_DIMENSIONS = _judge_prompts.PASS_FAIL_DIMENSIONS
JUDGE_SYSTEM_PROMPTS = _judge_prompts.JUDGE_SYSTEM_PROMPTS
build_judge_messages = _judge_prompts.build_judge_messages

__all__ = [
    "build_analyze_job_task",
    "JobAnalysis",
    "build_match_experience_task",
    "MatchingBrief",
    "build_write_cover_letter_task",
    "build_proofread_task",
    "JUDGE_DIMENSIONS",
    "PASS_FAIL_DIMENSIONS",
    "JUDGE_SYSTEM_PROMPTS",
    "build_judge_messages",
]
