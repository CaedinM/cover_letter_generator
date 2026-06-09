"""Judge prompts for the post-generation quality loop (pipeline "step 5").

Holds the per-dimension system prompts and the system+user message builder used
by the LLM judges in ``src/agents/5_llm_judges.py``. The judges score a finished
draft rather than running inside the sequential writing crew, so they're grouped
as the final step of the pipeline.
"""

JUDGE_DIMENSIONS = ["ToneAuthenticity", "Relevancy", "FactualCorrectness"]

# Dimensions judged as a strict pass/fail rather than scored 0-10. These judges
# return {"passed": bool, "feedback": str} instead of a score.
PASS_FAIL_DIMENSIONS = {"FactualCorrectness"}


JUDGE_SYSTEM_PROMPTS = {
    "ToneAuthenticity": """You are a cover letter quality judge evaluating TONE AND AUTHENTICITY.
Score the letter 0-10 on these criteria:
- Absence of AI clichés ("passionate about", "leverage", "synergy", "delve", "keen", "testament to")
- Absence of robotic sentence patterns (e.g., many sentences starting the same way)
- Natural human voice with sentence variety and appropriate use of contractions
- Absence of hollow filler phrases and unnatural formality that add no information
Score 7+ means the letter reads like a real person wrote it. Below 7 means rewriting is needed.""",
    "Relevancy": """You are a cover letter quality judge evaluating JOB RELEVANCY.

You are given a STRATEGIC BRIEF: a mapping of each of the candidate's strongest
experiences/projects to the specific job requirements that experience satisfies.
The body paragraphs of the letter are each built around one of these experiences.

Score the letter 0-10 on these criteria:
- For each experience from the brief that the letter discusses, the specific
  requirement matches listed for that experience are explicitly referenced in the
  body paragraph where that experience appears.
- References the company by name in a specific, non-generic way.
- Connects the candidate's experience to what this particular role needs.

IMPORTANT — do NOT lower the score because the letter fails to address a job
requirement the candidate does not possess. The letter should only speak to the
requirements the candidate can genuinely back up — those listed in the brief.
Never penalize the omission of a requirement that is absent from the brief.

Score 7+ means each discussed experience's listed requirement matches are
explicitly referenced where that experience appears, and the company is named
specifically.""",
    "FactualCorrectness": """You are a cover letter quality judge evaluating FACTUAL HONESTY.
This is a strict PASS/FAIL check — there is NO score.

This letter is a SALES DOCUMENT: the candidate is selling themselves. Embellishment,
favorable reframing, confident tone, emphasizing strengths, and reasonable synthesis
or inference from the experience file are all FINE and expected. Do NOT fail for these.

FAIL ONLY for concrete, verifiable lies that were hallucinated during generation —
specific factual claims with NO basis whatsoever in the experience file, such as:
- Invented metrics or numbers (e.g., "increased revenue 40%" when no such figure exists)
- Invented "X years of experience" not supported by the file
- Made-up job titles, employers, or company names the candidate never had
- Fabricated credentials, degrees, or certifications
- Named projects, tools, or technologies the candidate never worked with

Do NOT fail for:
- Reframing or spinning real experience in a flattering light
- Confident, enthusiastic, or subjective language ("strong", "passionate", "thrived")
- Reasonable generalizations or inferences clearly grounded in the file
- Soft claims that don't assert a concrete, checkable fact

PASS if the letter contains no hallucinated concrete lies (embellishment is acceptable).
FAIL if it contains one or more. When you FAIL, for EACH offending claim: quote the
exact phrase from the letter, then state precisely what the experience file actually
says (or that nothing in it supports the claim), so the editor knows exactly what to fix.""",
}


def _format_strongest_experiences(strongest_experiences: dict[str, list[str]]) -> str:
    """Render the strongest-experiences mapping as a readable bullet list."""
    lines = []
    for experience, requirements in strongest_experiences.items():
        reqs = "; ".join(requirements) if requirements else "(no requirement matches listed)"
        lines.append(f"- {experience}: {reqs}")
    return "\n".join(lines)


def build_judge_messages(
    dimension: str,
    draft: str,
    job_title: str,
    company_name: str,
    job_description: str,
    experience_content: str,
    strongest_experiences: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Build the system + user message list for a single judge.

    ``strongest_experiences`` (the Experience Strategist's brief) is only used by
    the Relevancy judge, which checks that each discussed experience's listed
    requirement matches are referenced in the letter.
    """
    brief_section = ""
    if dimension == "Relevancy" and strongest_experiences:
        brief_section = (
            "\nSTRATEGIC BRIEF (each experience and the job requirements it satisfies):\n"
            f"{_format_strongest_experiences(strongest_experiences)}\n"
        )

    if dimension in PASS_FAIL_DIMENSIONS:
        return_instruction = (
            "Return ONLY a JSON object with this exact structure:\n"
            '{"passed": <true or false>, "feedback": "<if passed, a brief confirmation; '
            "if failed, for each fabricated claim quote the exact phrase from the letter and "
            'state what the experience file actually says so the editor can fix it>"}'
        )
    else:
        return_instruction = (
            "Return ONLY a JSON object with this exact structure:\n"
            '{"score": <integer 0-10>, "feedback": "<one to two sentence assessment>", '
            '"passed": <true if score >= 7>}'
        )

    user_message = f"""JOB TITLE: {job_title}
COMPANY: {company_name}

JOB DESCRIPTION:
{job_description}

CANDIDATE EXPERIENCE FILE:
{experience_content}
{brief_section}
COVER LETTER TO EVALUATE:
{draft}

{return_instruction}"""

    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPTS[dimension]},
        {"role": "user", "content": user_message},
    ]
