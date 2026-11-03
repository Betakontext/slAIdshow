# SKILL.md – slAIdshow Chat-Only Development Workflow

## Workflow per Feature (routinized)

┌─────────────────────────────────────────────────────────────┐
│ 1. PLAN (Chat, high effort) │
│ Skill: /arch │
│ → You get: architecture, file plan, API contracts │
│ → You save the plan as a local Markdown file │
├─────────────────────────────────────────────────────────────┤
│ 2. IMPLEMENT (Chat, medium effort) │
│ Skill: /impl (per file/module) │
│ → Chat returns complete code │
│ → You copy locally, run, test │
│ → IMPORTANT: After each file: "Works" / "Error: X" │
├─────────────────────────────────────────────────────────────┤
│ 3. TEST (Chat, low effort) │
│ Skill: /test │
│ → Chat returns test code │
│ → You run pytest locally, paste result back │
├─────────────────────────────────────────────────────────────┤
│ 4. REVIEW (Chat, medium effort) │
│ Skill: /review (paste code block) │
│ → Chat returns fixes │
│ → You apply them │
├─────────────────────────────────────────────────────────────┤
│ 5. DOC (Chat, low effort) │
│ Skill: /doc │
│ → Chat returns README/CHANGELOG text │
│ → You paste in │
└─────────────────────────────────────────────────────────────┘


---

## Critical Rules for Chat-Only Workflow

| Rule | Why |
|------|-----|
| One file per request | "Give me the complete code for `vram_scheduler.py`" – not "implement the whole feature". Otherwise the chat loses focus. |
| Always complete code, never snippets | "Give me the entire file, not just the changed function." Avoids copy-paste errors. |
| Paste results back | After local execution: pytest output, `docker compose up` log, traceback – always back into the chat. This is your "terminal". |
| Context management | Every 5–10 features: start a new session. First message: "Here is the current file structure [tree] and current state [git log --oneline -10]." |
| Use screenshots | UI bugs, ComfyUI workflows, GPU monitoring → screenshot into the chat. GPT 6.1 Sol understands images. |
| Never let it guess | If you're unsure: "I have X locally, it gives error Y. What is the cause?" – not "What could go wrong?" |

---

## Skill Prompts

### /arch (high effort)

Design the architecture for [FEATURE].

Deliver:

ASCII state/flow diagram
File structure (new + modified)
API contracts (request/response shapes)
Step-by-step implementation list (one item = one file)
Constraints:

Must work on 6 GB VRAM (sequential scheduler)
No dependencies outside the defined stack
Respect existing conventions (type hints, ruff, pytest)

### /impl (medium effort)

Implement [STEP] from the current plan.
Give me the complete code for [FILE].

Rules:

Complete file, no snippets, no "# ... unchanged"
Python 3.12+, type hints, ruff-compliant
Must work on 6 GB VRAM tier (sequential mode)
Module-level docstring
if __name__ == "__main__": for testability
End with:

File path
How to test (one command)

### /test (low effort)

Write pytest tests for [FILE/MODULE].

Rules:

Mock GPU (pynvml), Ollama, ComfyUI – no real GPU in CI
Cover edge cases: no GPU, 6 GB, 16 GB, Apple Silicon, VRAM timeout
pytest fixtures for shared mocks
Target: 100% branch coverage on the tested module
End with: how to run (pytest tests/test_X.py -v)


### /review (medium effort)

Review the code below.

Check for:

Bugs / logic errors
Race conditions (async code)
VRAM leaks (missing empty_cache, missing unload)
Missing error handling (GPU OOM, network timeout, empty input)
Performance bottlenecks (blocking calls in async context)
Output:

Issues list (severity: critical / warning / nit)
Concrete fix for each (diff or full file)
[PASTE CODE HERE]


### /doc (low effort)

Write/update documentation for [FEATURE]:

README.md section (user-facing, 3-minute read)
CHANGELOG.md entry (## [Unreleased] – Added/Changed/Fixed)
One "How to" paragraph: "You have a [HARDWARE]. Here's what happens when you run make install."
Keep it concise. No marketing language.


---

## Session Start Template

Paste this at the beginning of every new session:

Here is the current state:

File structure: [tree -L 3]
Last commits: [git log --oneline -10]
Current goal: Phase X, Feature Y
Known open bugs: [list or "none"]
Last change: [one-line description]
Background knowledge is in Custom Instructions.


---

## Where This File Lives

slaidshow/
├── Makefile
├── docker-compose.yml
├── .env.example
├── SKILL.md ← here
├── slaidshow/
├── tests/
├── config/
├── models/
└── recordings/


Versioned with the code. `git diff SKILL.md` shows how the workflow evolves.
