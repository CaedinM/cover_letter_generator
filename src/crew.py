import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from crewai_tools import FileReadTool
from tavily import TavilyClient

_ROOT_DIR = Path(__file__).parent.parent

JUDGE_THRESHOLD = 7
MAX_JUDGE_ITERATIONS = 3
_JUDGE_DIMENSIONS = ["ToneAuthenticity", "Relevancy", "FactualCorrectness", "LengthStructure"]


@dataclass
class JudgeSummary:
    dimension: str
    score: int
    feedback: str
    passed: bool


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


_JUDGE_SYSTEM_PROMPTS = {
    "ToneAuthenticity": """You are a cover letter quality judge evaluating TONE AND AUTHENTICITY.
Score the letter 0-10 on these criteria:
- Absence of AI clichés ("passionate about", "leverage", "synergy", "delve", "keen", "testament to")
- Absence of robotic sentence patterns (e.g., many sentences starting the same way)
- Natural human voice with sentence variety and appropriate use of contractions
- Absence of hollow filler phrases and unnatural formality that add no information
Score 7+ means the letter reads like a real person wrote it. Below 7 means rewriting is needed.""",
    "Relevancy": """You are a cover letter quality judge evaluating JOB RELEVANCY.
Score the letter 0-10 on these criteria:
- Directly addresses the specific requirements in the job description
- References the company by name in a specific, non-generic way
- Connects candidate experience to what this particular role needs
- Does not read like a generic letter that could apply to any job
Score 7+ means the letter is clearly targeted to this specific role and company.""",
    "FactualCorrectness": """You are a cover letter quality judge evaluating FACTUAL CORRECTNESS.
Score the letter 0-10 on these criteria:
- Every specific claim (project names, outcomes, metrics, job titles) appears in the experience file
- No invented years of experience (e.g., "3 years of X" unless stated)
- No fabricated metrics or achievements not present in the experience file
- No made-up company names or job titles
Score 7+ means every claim in the letter can be traced back to the experience file.""",
    "LengthStructure": """You are a cover letter quality judge evaluating LENGTH AND STRUCTURE.
Score the letter 0-10 on these criteria:
- Word count between 200 and 280 words (count carefully)
- Exactly 4 paragraphs: (1) introduction/company interest, (2) first experience,
  (3) second experience, (4) closing/excitement
- Proper greeting and sign-off present
- No bracket placeholders like [Company] or [Your Name]
Score 7+ means the letter conforms to the required structure and length.""",
}


def _build_judge_messages(
    dimension: str,
    draft: str,
    job_title: str,
    company_name: str,
    job_description: str,
    experience_content: str,
) -> list[dict]:
    """Build the system + user message list for a single judge."""
    user_message = f"""JOB TITLE: {job_title}
COMPANY: {company_name}

JOB DESCRIPTION:
{job_description}

CANDIDATE EXPERIENCE FILE:
{experience_content}

COVER LETTER TO EVALUATE:
{draft}

Return ONLY a JSON object with this exact structure:
{{"score": <integer 0-10>, "feedback": "<one to two sentence assessment>", "passed": <true if score >= 7>}}"""

    return [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPTS[dimension]},
        {"role": "user", "content": user_message},
    ]


def _run_single_judge(dimension: str, messages: list[dict], llm: LLM) -> JudgeSummary:
    """Run one judge LLM call and parse its JSON verdict. Never raises."""
    try:
        raw = llm.call(messages)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in judge output: {raw!r}")
        data = json.loads(match.group(0))
        score = int(data["score"])
        return JudgeSummary(
            dimension=dimension,
            score=score,
            feedback=str(data.get("feedback", "")),
            passed=score >= JUDGE_THRESHOLD,
        )
    except Exception as e:
        return JudgeSummary(
            dimension=dimension,
            score=0,
            feedback=f"Judge error: {e}",
            passed=False,
        )


def _run_judge_panel(
    draft: str,
    job_title: str,
    company_name: str,
    job_description: str,
    experience_file: str,
) -> list[JudgeSummary]:
    """Run all four judges in parallel and return their summaries in fixed order."""
    try:
        with open(experience_file, "r") as f:
            experience_content = f.read()
    except FileNotFoundError:
        experience_content = ""

    judge_llm = LLM(model="claude-haiku-4-5-20251001")

    results_by_dimension: dict[str, JudgeSummary] = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_dim = {}
        for dimension in _JUDGE_DIMENSIONS:
            messages = _build_judge_messages(
                dimension, draft, job_title, company_name, job_description, experience_content
            )
            future = executor.submit(_run_single_judge, dimension, messages, judge_llm)
            future_to_dim[future] = dimension

        for future, dimension in future_to_dim.items():
            results_by_dimension[dimension] = future.result()

    return [results_by_dimension[d] for d in _JUDGE_DIMENSIONS]


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

    authenticity_editor = Agent(
        role="Senior Editorial Proofreader",
        goal="Ensure cover letters sound authentically human and match the optimal tone for each role",
        backstory="""You are a veteran editor with 20 years of experience who has
        developed an expert eye for detecting artificial or templated writing. You
        know exactly what makes writing sound robotic versus genuine. You understand
        that different industries and roles require different tones - a startup wants
        energy and personality, while a law firm expects measured professionalism.
        You ruthlessly eliminate clichés, buzzwords, and hollow phrases that scream
        'AI-generated' and replace them with authentic, conversational language that
        still maintains professionalism.""",
        verbose=True,
        allow_delegation=False,
        llm=LLM(model="claude-sonnet-4-5"),
    )

    revision_task = Task(
        description=f"""Review the cover letter and edit it to ensure it sounds
        authentically human, stays CONCISE, and matches the optimal tone.

        VERIFY THIS STRUCTURE:
        - Paragraph 1: Candidate introduces themselves + why this role/company (3-4 sentences)
        - Paragraph 2 & 3: 2 specific experiences with contribution and impact (5-6 sentences)
        - Paragraph 4: Skills connection + specific excitement + closing (3-4 sentences)
        - Total: 200-280 words

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
        - 200-280 words total
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

    # Agent 1: Job Analyst
    job_analyst = Agent(
        role="Senior Job Requirements Analyst",
        goal="Extract and prioritize key requirements, skills, and qualifications from job descriptions, and research companies to understand their culture and values",
        backstory="""You are an expert at analyzing job postings with over 15 years of
        experience in HR and recruitment. You can quickly identify the critical requirements
        that hiring managers truly care about versus nice-to-haves. You understand how to
        read between the lines to identify company culture and team dynamics from job
        descriptions. When company details are limited, you research online to gather
        insights about the company's mission, values, recent news, and culture.""",
        verbose=True,
        allow_delegation=False,
        tools=[search_company],
        llm=haiku,
    )

    # Agent 2: Company Research Agent
    company_research_agent = Agent(
        role="Company Research Agent",
        goal="Research the company and gather information about their mission, values, culture, recent news, and what makes them unique",
        backstory="""You are an expert at researching companies and gathering information about their mission, values, culture, recent news, and what makes them unique.""",
        verbose=True,
        allow_delegation=False,
        tools=[search_company],
        llm=haiku,
    )
    # Agent 2: Experience Strategist
    experience_strategist = Agent(
        role="Career Strategy Consultant",
        goal="Match the candidate's background to job requirements and identify the strongest talking points",
        backstory="""You are a career coach who has helped thousands of professionals
        land their dream jobs. You excel at identifying which experiences and skills
        will resonate most with hiring managers. You know how to frame past experiences
        to align perfectly with job requirements, finding connections that candidates
        often miss themselves.""",
        verbose=True,
        allow_delegation=False,
        tools=[file_read_tool],
        llm=sonnet,
    )

    # Agent 3: Cover Letter Writer
    cover_letter_writer = Agent(
        role="Professional Cover Letter Writer",
        goal="Craft compelling, personalized cover letters that get interviews",
        backstory="""You are an award-winning professional writer who specializes in
        career documents. Your cover letters have helped candidates land positions at
        top companies. You know how to balance professionalism with personality, and
        you understand that a great cover letter tells a story that makes the reader
        want to meet the candidate.""",
        verbose=True,
        allow_delegation=False,
        llm=sonnet,
    )

    # Agent 4: Authenticity Editor
    authenticity_editor = Agent(
        role="Senior Editorial Proofreader",
        goal="Ensure cover letters sound authentically human and match the optimal tone for each role",
        backstory="""You are a veteran editor with 20 years of experience who has
        developed an expert eye for detecting artificial or templated writing. You
        know exactly what makes writing sound robotic versus genuine. You understand
        that different industries and roles require different tones - a startup wants
        energy and personality, while a law firm expects measured professionalism.
        You ruthlessly eliminate clichés, buzzwords, and hollow phrases that scream
        'AI-generated' and replace them with authentic, conversational language that
        still maintains professionalism.""",
        verbose=True,
        allow_delegation=False,
        llm=sonnet,
    )

    # Task 1: Analyze the job description and research the company
    analyze_job_task = Task(
        description=f"""Analyze the following job description for the position of
        "{job_title}" at "{company_name}".

        USE YOUR SEARCH TOOL to research {company_name} online to gather context about
        their mission, values, culture, recent news, and what makes them unique.

        Extract and research:
        1. MUST-HAVE requirements (skills, experience, qualifications that are essential)
        2. NICE-TO-HAVE requirements (preferred but not essential)
        3. Company culture signals (values, work style, team dynamics)
        4. Key responsibilities of the role
        5. Any specific technologies, tools, or methodologies mentioned
        6. COMPANY RESEARCH for {company_name}:
           - Company mission and values
           - Recent news or achievements
           - Company culture and work environment
           - Products, services, or notable projects
           - Any unique aspects that a candidate could reference

        Job Title: {job_title}
        Company: {company_name}

        Job Description:
        {job_description}

        Provide a structured analysis that will help match a candidate's experience to this role.""",
        expected_output=f"""A structured analysis for {job_title} at {company_name} containing:
        - List of must-have requirements with priority ranking
        - List of nice-to-have requirements
        - Summary of company culture and values (from job description AND research)
        - Key responsibilities summary
        - Technical requirements list
        - Company research findings (mission, recent news, unique talking points)""",
        agent=job_analyst,
    )

    # Task 2: Match experience to requirements
    match_experience_task = Task(
        description="""Using the job analysis provided and the candidate's background
        (read from the experience file), identify:

        1. Which of the candidate's projects or experiences directly match the must-have requirements
        2. Any motivations the candidate has that might explain why they are eager to land this role or work for this specific company.
        If there is not a strong connection, leave as "None", do not force a connection that is not there.
        3. How to frame the candidate's background to align with the company culture.
        4. Any gaps and how to address or minimize them.
        5. The top 3-4 strongest talking points for this specific role

        Read the candidate's experience file to get their full background.
        Create a strategic brief that will guide the cover letter writing.

        CRITICAL: Only use FACTS explicitly stated in the experience file.
        - Do NOT invent years of experience
        - Do NOT make up job titles, companies, or achievements
        - Do NOT assume or infer details not written in the file
        - If something is unclear, note it as a gap rather than guessing""",
        expected_output="""A strategic matching brief containing:
        - Matched experiences for each key requirement (ONLY from the file)
        - Top talking points with specific examples (ONLY from the file)
        - Recommended tone and approach based on company culture
        - Strategy for addressing any gaps
        - Key achievements to highlight (ONLY those explicitly stated)""",
        agent=experience_strategist,
        context=[analyze_job_task],
    )

    # Task 3: Write the cover letter
    write_cover_letter_task = Task(
        description=f"""Write a CONCISE cover letter for the "{job_title}" position
        at {company_name}. Use the job analysis and experience matching brief provided.

        3-PARAGRAPH STRUCTURE:

        PARAGRAPH 1 (3-4 sentences):
        - Introduce yourself: your background, education, and relevant expertise
        - State you're applying for {job_title} at {company_name}
        - Show genuine interest by referencing something about {company_name}, If there is a specific connection between 
        the candidate's motivations and the company's mission or values, reference it here.
        - Connect your background to why this role appeals to you If there is an obvious connection between 
        the candidate's motivations and this specific role, reference it here.
        Example: "I am excited to apply for the X role at Y. As a data science graduate
        from UC Berkeley with experience at early-stage startups, I am excited by your focus
         on Z. My background in A and B makes me a strong fit for this role."

        PARAGRAPH 2 & 3 (5-6 sentences):
        - Naturally transition from previous paragraph. Mention a specific skill that is highly relevant to the role.
        - Briefly describe the experience or project that the relevant skill was used in.
        - For each: what you did, your contribution, and the impact (2-3 sentences each)
        - Use specific details and metrics where possible
        - These should directly relate to what the job requires
        - Each paragraphs should dicuss a DIFFERENT skill and project/experience.

        PARAGRAPH 4 (3-4 sentences):
        - Briefly tie your skills to what the role needs
        - Express genuine excitement about something SPECIFIC: either an aspect of
          the role itself, the company's mission/product or a recent development at the company (use research findings)
        - Express confidence in your ability to succeed in the role due to specific skills you have.

        GUIDELINES:
        - Total length: 200-280 words
        - Be specific and personal, not generic or dull.
        - Use "{company_name}" and "{job_title}" naturally
        - No placeholder brackets
        - NEVER directly state that a skill matches a job requirement (e.g., avoid
          "My skills in X align with what this role demands"). Instead, describe a
          project or experience that used the skill — let the reader draw the connection.
        {examples_section}
        CRITICAL - DO NOT FABRICATE:
        - Only include facts from the experience matching brief
        - Do NOT invent years of experience (e.g., don't say "3 years" unless stated)
        - Do NOT make up achievements, metrics, or job details
        - Do NOT assume the candidate has experience they don't have
        - If unsure about a detail, leave it out rather than guess""",
        expected_output=f"""A concise, personal cover letter for {job_title} at {company_name}:
        - Paragraph 1: Who you are + why this role/company (3-4 sentences)
        - Paragraph 2: 2 relevant projects or experiences with specific impact (5-6 sentences)
        - Paragraph 3: Skills tie-in + specific excitement + closing (3-4 sentences)
        - Total: 200-280 words
        - Ends with genuine excitement about the role or company mission
        - Properly formatted with greeting and sign-off""",
        agent=cover_letter_writer,
        context=[analyze_job_task, match_experience_task],
    )

    # Task 4: Proofread for authenticity and tone
    proofread_task = Task(
        description="""Review the cover letter and edit it to ensure it sounds
        authentically human, stays CONCISE, and matches the optimal tone.

        VERIFY THIS STRUCTURE:
        - Paragraph 1: Candidate introduces themselves + why this role/company (3-4 sentences)
        - Paragraph 2 & 3: 2 specific experiences with contribution and impact (5-6 sentences)
        - Paragraph 3: Skills connection + specific excitement + closing (3-4 sentences)
        - Total: 200-280 words

        MUST CHECK:
        - If paragraph 1 is missing personal info about the candidate, ADD IT
        - If paragraph 3 lacks genuine excitement about something SPECIFIC to the
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

        ENSURE the letter has:
        - Natural sentence variety (different lengths, structures)
        - Specific details that only a human would include
        - Appropriate tone for the industry (casual for startups, formal for finance/law)
        - Contractions where natural (I'm, I've, didn't)

        Reference the job analysis to calibrate the right tone.
        {examples_section}
        Output the final, edited cover letter only - no commentary.""",
        expected_output="""The final cover letter, edited for authenticity:
        - Paragraph 1 introduces the candidate AND shows company interest
        - Paragraph 2 has 2 specific experiences with impact
        - Paragraph 3 ties skills to role and closes
        - 200-280 words total
        - Sounds like a real person wrote it
        - Free of AI clichés and buzzwords
        - Ready to send without further editing""",
        agent=authenticity_editor,
        context=[analyze_job_task, write_cover_letter_task],
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
    """Generate a cover letter for the given job, then run the judge quality loop."""
    crew = create_cover_letter_crew(job_title, company_name, job_description, experience_file)
    current_draft = str(crew.kickoff())

    best_letter = current_draft
    best_avg = -1.0

    for iteration in range(1, MAX_JUDGE_ITERATIONS + 1):
        print(f"\n{'=' * 60}")
        print(f"QUALITY REVIEW — Iteration {iteration}/{MAX_JUDGE_ITERATIONS}")
        print(f"{'=' * 60}")
        print("Running 4 judges in parallel...")

        summaries = _run_judge_panel(
            current_draft, job_title, company_name, job_description, experience_file
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
