# Cover Letter Generator

An AI-powered CLI tool that writes tailored cover letters using a multi-agent pipeline. You provide a job description and your background — a team of specialized agents researches the company, maps your experience to the role, drafts a letter, and edits out any AI-sounding language before handing it back to you for review.

---

## How It Works

The pipeline is built with **CrewAI** and runs four agents sequentially, each with a distinct role:

| Agent | Model | Role |
|---|---|---|
| Job Analyst | Claude Haiku | Parses the job description and researches the company (via Tavily web search) |
| Experience Strategist | Claude Sonnet | Reads your experience file and maps your background to the job requirements |
| Cover Letter Writer | Claude Sonnet | Drafts a focused 3-paragraph, 200–280 word letter using only verified facts |
| Authenticity Editor | Claude Sonnet | Removes AI clichés, enforces structure, and deletes any fabricated claims |

A fifth **Revision Agent** (Haiku) handles targeted edits if you request changes after the initial draft.

**Infrastructure & integrations:**
- **Anthropic API** — powers all Claude agents
- **Tavily** — optional web search for real-time company research

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
- `ANTHROPIC_API_KEY` — from [console.anthropic.com](https://console.anthropic.com)
- `TAVILY_API_KEY` — from [tavily.com](https://tavily.com) (free tier available; enables company research)

**4. Add your reference files**

Create a `references/` folder at the project root:

```bash
mkdir references
```

Inside it, add:
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
Job title:    > Software Engineer
Company name: > Acme Corp
Job description (type END on a new line when done):
> [paste JD here]
> END
```

The agent team runs for roughly 1–2 minutes, then prints the finished letter.

**Review options:**

```
[s] Save the letter to saved_letters/
[r] Request revisions
[q] Quit without saving
```

If you choose `r`, describe the changes you want and the Revision Agent will apply them. You can iterate as many times as needed before saving.
