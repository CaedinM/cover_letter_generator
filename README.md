# Cover Letter Generator

An AI-powered tool that writes tailored cover letters using a multi-agent pipeline — usable as an interactive CLI or as an [MCP](https://modelcontextprotocol.io) server that other agents can call directly. You provide a job description and your background — a team of specialized agents researches the company, analyzes the job, maps your experience to the role, drafts a letter, and edits out any AI-sounding language or fabricated claims.

---

## How It Works

The pipeline is built with **CrewAI**. The first two agents run **in parallel** (both depend only on the job description), then the rest run sequentially:

| Step | Agent | Model | Role |
|---|---|---|---|
| 1A | Company Researcher | Claude Sonnet 4.5 | Researches the company (via Tavily web search) and summarizes how your background aligns with it |
| 1B | Job Description Analyst | Claude Sonnet 4.5 | Extracts must-haves, nice-to-haves, key responsibilities, and technical requirements from the job description |
| 2 | Experience Matcher | Claude Sonnet 5 | Reads your experience file and picks exactly two experiences that map to the job's requirements |
| 3 | Cover Letter Writer | Claude Haiku 4.5 | Drafts a 3-paragraph, 1600–1900 character letter (opening/closing from the company research, body from the two matched experiences) |
| 4 | Authenticity Editor | Claude Haiku 4.5 | Removes AI clichés and deletes any claim not grounded in your experience file |

### Revisions

A separate **Revision Agent** (Haiku) handles targeted edits if you request changes after the draft is finalized.

LLMs can't reliably count characters, so the 1600–1900 character target is checked in plain Python after the crew finishes. If the draft is out of range, the **Authenticity Editor** is re-run (up to 2 attempts) with specific feedback on how much to trim or expand, and the closest draft is returned.


**Infrastructure & integrations:**
- **Anthropic API** — powers all Claude agents
- **Tavily** — web search tool for real-time company research

---

## Project Structure

```
src/
├── main.py              CLI entry point: input, generation, revision loop, saving
├── mcp_server.py         MCP server exposing the pipeline as tools
├── crew.py               Orchestration: builds the crew + enforces the length target
├── models.py              Model/LLM assignment per agent
├── tools.py                Tavily search_company tool (used by the Company Researcher)
├── length_check.py         Deterministic character-count check (LENGTH_MIN/MAX)
├── agents/               One file per pipeline step (build_* agent factories)
│   ├── 1a_company_researcher.py
│   ├── 1b_job_analyst.py
│   ├── 2_experience_matcher.py
│   ├── 3_writer.py
│   └── 4_editor.py
└── tasks/                One file per pipeline step (build_* task factories + prompts)
    ├── 1a_company_research_task.py   (CompanyResearch schema)
    ├── 1b_analyze_job_task.py        (JobAnalysis schema)
    ├── 2_match_experience_task.py    (ExperienceMatch/MatchedExperience schema)
    ├── 3_write_cover_letter_task.py
    ├── 4_proofread_task.py
    └── length_fix_task.py            Re-runs the editor to fix an out-of-range draft
```

---

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/CaedinM/cover_letter_generator
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
- `references/good_examples.md` *(optional)* — examples of cover letters you've written that you're happy with; used to calibrate tone and style for the Writer and Editor

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

The agent team runs for roughly 1–3 minutes, printing progress, then the finished letter.

**Review options:**

```
[r] Request changes
[s] Save and exit (prompts for a filename; saved to saved_letters/)
[q] Quit without saving
```

If you choose `r`, describe the changes you want and the Revision Agent will apply them. You can iterate as many times as needed before saving.

---

## Connecting Your Agent (MCP Server)

The same pipeline is exposed as an MCP server (`src/mcp_server.py`), so other agents — Claude Code, the Claude Agent SDK, or any MCP-compatible client — can call it programmatically with structured input/output instead of going through the interactive CLI.

It uses the same setup: `.env` with `ANTHROPIC_API_KEY` and `references/my_experience.md` must exist (see [Setup](#setup)).

### Tools exposed

| Tool | Inputs | Returns |
|---|---|---|
| `generate_cover_letter` | `job_title`, `company_name`, `job_description`, `experience_file` *(optional — path to an alternate experience file)* | `{ cover_letter, char_count, job_title, company_name }` |
| `revise_cover_letter` | `current_letter`, `feedback`, `job_title`, `company_name` | `{ cover_letter, char_count }` |

`generate_cover_letter` runs the full crew plus the length-enforcement step (~1–3 minutes per call). `revise_cover_letter` applies a single targeted edit and is much faster.

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
