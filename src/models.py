from crewai import LLM

models = {
    # Sonnet 4.5 (not 5) so its search_company tool fires alongside its
    # output_pydantic output — Sonnet 5 skips tools on structured-output tasks.
    "company_researcher": LLM(
        model="anthropic/claude-sonnet-4-5",
        temperature=0.2,
    ),
    "job_analyst": LLM(
        model="anthropic/claude-sonnet-4-5",
        temperature=0.2,
    ),
    # Sonnet 5 deprecates temperature (API 400s if passed).
    "experience_matcher": LLM(
        model="anthropic/claude-sonnet-5",
    ),
    "writer": LLM(
        model="anthropic/claude-haiku-4-5-20251001",
        temperature=0.7,
    ),
    "editor": LLM(
        model="anthropic/claude-haiku-4-5-20251001",
        temperature=0.2,
    ),
    "revision_agent": LLM(
        model="anthropic/claude-haiku-4-5-20251001",
    ),
}
