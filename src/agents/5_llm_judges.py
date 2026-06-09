"""LLM judge agents for the post-generation quality loop (pipeline "step 5").

Each LLM "judge" is a single Haiku LLM call that scores a finished draft on one
dimension and returns a structured verdict. ``run_judge_panel`` fans them out in
parallel and appends a deterministic ``check_length`` result (length is measured
in Python rather than judged by an LLM). The prompts they use live in
``src/tasks/5_judge_prompts.py``.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from crewai import LLM
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import context as otel_context, trace

from tasks import JUDGE_DIMENSIONS, PASS_FAIL_DIMENSIONS, build_judge_messages

JUDGE_THRESHOLD = 7

# Judges are bare LLM.call()s (not CrewAI agents), so CrewAIInstrumentor doesn't
# fold them into the agent/task span tree. We open our own span here to group
# them, and propagate context into the worker threads (OTel's current span is a
# contextvar that does NOT cross thread boundaries on its own). If tracing is
# disabled, this resolves to a no-op tracer and adds no overhead.
_tracer = trace.get_tracer(__name__)

# Target letter length (characters). Kept in sync with the writer/editor tasks.
LENGTH_MIN = 1600
LENGTH_MAX = 1900


@dataclass
class JudgeSummary:
    dimension: str
    score: int
    feedback: str
    passed: bool


def check_length(draft: str) -> JudgeSummary:
    """Deterministic length check that replaces the old LLM length judge.

    LLMs can't reliably count characters, so length is measured in Python and
    turned into a pass/fail (score 10/0) ``JudgeSummary`` so it slots into the
    existing judge loop and editor-revision feedback unchanged.
    """
    count = len(draft.strip())
    passed = LENGTH_MIN <= count <= LENGTH_MAX
    if passed:
        feedback = f"Letter is {count} characters (within the {LENGTH_MIN}-{LENGTH_MAX} target)."
    elif count < LENGTH_MIN:
        feedback = (
            f"Letter is {count} characters — {LENGTH_MIN - count} short of the "
            f"{LENGTH_MIN}-{LENGTH_MAX} target. Expand it."
        )
    else:
        feedback = (
            f"Letter is {count} characters — {count - LENGTH_MAX} over the "
            f"{LENGTH_MIN}-{LENGTH_MAX} target. Trim it."
        )
    return JudgeSummary(dimension="Length", score=10 if passed else 0, feedback=feedback, passed=passed)


def run_single_judge(
    dimension: str, messages: list[dict], llm: LLM, pass_fail: bool = False
) -> JudgeSummary:
    """Run one judge LLM call and parse its JSON verdict. Never raises.

    Scored judges return a 0-10 ``score``; ``pass_fail`` judges return a boolean
    ``passed`` (mapped to score 10/0 so it slots into the averaging loop).
    """
    try:
        raw = llm.call(messages)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in judge output: {raw!r}")
        data = json.loads(match.group(0))
        if pass_fail:
            passed = bool(data["passed"])
            score = 10 if passed else 0
        else:
            score = int(data["score"])
            passed = score >= JUDGE_THRESHOLD
        return JudgeSummary(
            dimension=dimension,
            score=score,
            feedback=str(data.get("feedback", "")),
            passed=passed,
        )
    except Exception as e:
        return JudgeSummary(
            dimension=dimension,
            score=0,
            feedback=f"Judge error: {e}",
            passed=False,
        )


def run_judge_panel(
    draft: str,
    job_title: str,
    company_name: str,
    job_description: str,
    experience_file: str,
    strongest_experiences: dict[str, list[str]] | None = None,
) -> list[JudgeSummary]:
    """Run the LLM judges in parallel, then append the deterministic length check.

    Returns summaries in fixed order (LLM judges in ``JUDGE_DIMENSIONS`` order,
    followed by the ``Length`` check). ``strongest_experiences`` is the Experience
    Strategist's brief (each experience mapped to the job requirements it
    satisfies); the Relevancy judge uses it.
    """
    try:
        with open(experience_file, "r") as f:
            experience_content = f.read()
    except FileNotFoundError:
        experience_content = ""

    judge_llm = LLM(model="claude-haiku-4-5-20251001")

    results_by_dimension: dict[str, JudgeSummary] = {}
    with _tracer.start_as_current_span("judge-panel") as panel_span:
        panel_span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value
        )
        # Capture the active context (with panel-span current) so each worker can
        # re-attach it — otherwise the judge LLM spans wouldn't nest under us.
        parent_ctx = otel_context.get_current()

        def _judge_in_context(dimension, messages, pass_fail):
            token = otel_context.attach(parent_ctx)
            try:
                return run_single_judge(dimension, messages, judge_llm, pass_fail)
            finally:
                otel_context.detach(token)

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_dim = {}
            for dimension in JUDGE_DIMENSIONS:
                messages = build_judge_messages(
                    dimension,
                    draft,
                    job_title,
                    company_name,
                    job_description,
                    experience_content,
                    strongest_experiences,
                )
                future = executor.submit(
                    _judge_in_context,
                    dimension,
                    messages,
                    dimension in PASS_FAIL_DIMENSIONS,
                )
                future_to_dim[future] = dimension

            for future, dimension in future_to_dim.items():
                results_by_dimension[dimension] = future.result()

    summaries = [results_by_dimension[d] for d in JUDGE_DIMENSIONS]
    summaries.append(check_length(draft))
    return summaries
