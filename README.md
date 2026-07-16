# Cover Letter Generator

An AI-powered tool that writes tailored cover letters using a multi-agent pipeline — usable as an interactive CLI or as an [MCP](https://modelcontextprotocol.io) server that other agents can call directly. You provide a job description and your background — a team of specialized agents researches the company, maps your experience to the role, drafts a letter, and edits out any AI-sounding language. A panel of LLM judges then scores the draft and sends it back for revision until it passes, before handing it to you for review.

---

## How It Works

The pipeline is built with **CrewAI** and runs four agents sequentially, each with a distinct role. The first two agents emit **structured (Pydantic) output** that downstream steps and the judges consume directly:

| Step | Agent | Model | Role |
|---|---|---|---|
| 1 | Job Analyst | Claude Haiku | Parses the job description and researches the company (via Tavily web search) into a structured `JobAnalysis` |
| 2 | Experience Strategist | Claude Sonnet | Reads your experience file and produces a `MatchingBrief` mapping each of your strongest experiences to the specific requirements it satisfies |
| 3 | Cover Letter Writer | Claude Sonnet | Drafts a 4-paragraph, 1600–1900 character letter (intro → two experience paragraphs → close) using only verified facts from the brief |
| 4 | Authenticity Editor | Claude Sonnet | Removes AI clichés, enforces structure, and deletes any fabricated claims |

### Quality loop

After the crew produces a draft, it goes through up to **three rounds** of automated review. Each round runs a panel in parallel:

- **Tone & Authenticity** *(LLM judge, scored 0–10)* — flags AI clichés, robotic patterns, and hollow filler
- **Relevancy** *(LLM judge, scored 0–10)* — uses the `MatchingBrief` to verify each experience the letter discusses actually references the requirements it's meant to satisfy, and that the company is named specifically
- **Factual Correctness** *(LLM judge, strict pass/fail)* — fails only on hallucinated concrete claims (invented metrics, titles, credentials); embellishment and favorable framing are allowed
- **Length** *(deterministic Python check)* — measures the character count against the 1600–1900 target (LLMs can't count reliably, so this is done in code)

If any dimension fails, the **Authenticity Editor** is re-run with the judges' targeted feedback. The best-scoring draft across all iterations is returned.

A separate **Revision Agent** (Haiku) handles targeted edits if you request changes after the draft is finalized.

**Infrastructure & integrations:**
- **Anthropic API** — powers all Claude agents and judges
- **Tavily** — optional web search for real-time company research

---

## Project Structure

```
src/
├── main.py              CLI entry point: input, generation, revision loop, saving
├── crew.py              Orchestration: builds the crew + runs the judge quality loop
├── agents/              One file per pipeline step (build_* agent factories)
│   ├── 1_senior_job_requirements_agent.py
│   ├── 2_career_strategy_consultant_agent.py
│   ├── 3_professional_cover_letter_writer_agent.py
│   ├── 4_senior_editorial_proofreader_agent.py
│   └── 5_llm_judges.py          The LLM judges + deterministic length check
└── tasks/               One file per pipeline step (build_* task factories + prompts)
    ├── 1_analyze_job_task.py    (JobAnalysis schema)
    ├── 2_match_experience_task.py  (MatchingBrief schema)
    ├── 3_write_cover_letter_task.py
    ├── 4_proofread_task.py
    └── 5_judge_prompts.py        Judge dimensions + system prompts
```

Agents and tasks are split out of `crew.py`, one module per pipeline step. The module names are numeric-prefixed (`1_…` through `5_…`), so each package's `__init__.py` loads them via `importlib` and re-exports the factories — import them from the package, e.g. `from agents import build_job_analyst`.

---

## Setup

**1. Clone the repo**

```bash
git clone <repo-url>
cd cover_letter_generator
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure API keys**

```bash
cp .env.example .env
```

Then open `.env` and fill in your keys:
- `ANTHROPIC_API_KEY` **(required)** — from [console.anthropic.com](https://console.anthropic.com)
- `TAVILY_API_KEY` *(optional)* — from [tavily.com](https://tavily.com) (free tier available; enables company research)

**4. Add your reference files**

Copy the template references file at the project root:

```bash
cp -r references.example references
```


Inside it, fill in the template markdown files with your own information:
- `references/my_experience.md` **(required)** — your full professional background: work history, projects, skills, and any personal motivations or connections you want the agents to draw on
- `references/good_examples.md` *(optional)* — examples of cover letters you've written that you're happy with; used to calibrate tone and style

---

## Running the Pipeline

**Start the script:**

```bash
python src/main.py
```

You'll be prompted for three inputs:

```
Job Title:
> [title]
Company Name:
> [company]
Paste the job description below.
When finished, enter a blank line followed by 'END' on its own line.
> [paste JD here]
> END
```

The agent team and the judge loop run for roughly 1–3 minutes, printing each judge's verdict per iteration, then the finished letter.

**Review options:**

```
[r] Request changes
[s] Save and exit (prompts for a filename; saved to saved_letters/)
[q] Quit without saving
```

If you choose `r`, describe the changes you want and the Revision Agent will apply them. You can iterate as many times as needed before saving.

---

## Using It From Another Agent (MCP Server)

The same pipeline is exposed as an [MCP](https://modelcontextprotocol.io) server (`src/mcp_server.py`), so other agents — Claude Code, the Claude Agent SDK, or any MCP-compatible client — can call it programmatically with structured input/output instead of going through the interactive CLI.

It uses the same setup: `.env` with `ANTHROPIC_API_KEY` and `references/my_experience.md` must exist (see [Setup](#setup)).

### Tools exposed

| Tool | Inputs | Returns |
|---|---|---|
| `generate_cover_letter` | `job_title`, `company_name`, `job_description`, `experience_file` *(optional — path to an alternate experience file)* | `{ cover_letter, char_count, job_title, company_name }` |
| `revise_cover_letter` | `current_letter`, `feedback`, `job_title`, `company_name` | `{ cover_letter, char_count }` |

`generate_cover_letter` runs the full crew + judge quality loop (~1–3 minutes per call). `revise_cover_letter` applies a single targeted edit and is much faster.

### Connecting Claude Code

```bash
claude mcp add-json cover-letter-generator '{"command":"/abs/path/to/python","args":["/abs/path/to/cover_letter_generator/src/mcp_server.py"]}'
```

Verify it registered with the script path in `args` (not empty), then restart the session:

```bash
claude mcp get cover-letter-generator
```

### Connecting any MCP client

Point your client at the server as a stdio process:

```json
{
  "mcpServers": {
    "cover-letter-generator": {
      "command": "/abs/path/to/python",
      "args": ["/abs/path/to/cover_letter_generator/src/mcp_server.py"]
    }
  }
}
```

**Use the absolute path to the Python interpreter** that has the dependencies installed (e.g. your venv's `bin/python` or `which python`), not a bare `python`. MCP clients often launch servers with a minimal `PATH`, and a bare `python` can resolve to a system interpreter that lacks the dependencies — causing the server to fail on launch.

### Notes

- **Transport:** stdio (the MCP default). The server redirects the pipeline's verbose progress output to stderr so it can't corrupt the JSON-RPC stream on stdout.
- **Tool naming:** most clients namespace the tools as `mcp__cover-letter-generator__generate_cover_letter`, etc. — reference that form in allowlists and prompts.
- **Timeouts:** a `generate_cover_letter` call can take 1–3 minutes. If your client enforces a short tool-call timeout, raise it.
