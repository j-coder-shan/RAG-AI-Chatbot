# How to use this bundle with Claude Code

This folder contains everything needed to brief Claude Code on building
Project 1 (RAG AI Chatbot) from an empty folder through to a deployed,
tested application — running fully on local/free tools (Ollama +
HuggingFace ecosystem, ChromaDB), no paid API keys.

## Files in this bundle

| File | Purpose |
|---|---|
| `CLAUDE.md` | Persistent project memory. Claude Code reads this automatically at the start of every session. Contains the tech stack, architecture, commands, and non-negotiable rules. |
| `PLAN.md` | The phased task checklist (Phase 0 → Phase 8). This is the actual work order — Claude Code should work through it top to bottom. |
| `.env.example` | Every environment variable the project needs, with safe local defaults. No real secrets — copy to `.env` in Phase 1. |
| `docker-compose.yml` | Starter container setup for Phase 7 (deployment), wiring Ollama + backend + frontend together. |

## Setup steps

1. Create a new empty folder for the project (or let Claude Code create
   it as the first step of `PLAN.md` Phase 1 — both work).
2. Copy `CLAUDE.md`, `PLAN.md`, `.env.example`, and `docker-compose.yml`
   into that folder root.
3. Open the folder in Claude Code.
4. Start the session with:
   > "Read CLAUDE.md and PLAN.md. Confirm Phase 0 prerequisites are met
   > on this machine, then start Phase 1."
5. Let Claude Code work phase by phase. After each phase it should run
   the relevant tests and report the checkpoint result before moving on.
6. To resume a session later, use the "Quick status check" prompt at the
   bottom of `PLAN.md`.

## Why it's structured this way

- **`CLAUDE.md` stays short** (commands + architecture + hard rules) so
  it doesn't eat context budget every session — the detailed task-by-task
  work lives in `PLAN.md` instead, which Claude Code reads on demand.
- **Checkpoints, not just checkboxes** — each phase ends with a concrete,
  verifiable condition (tests passing, a manual smoke test) so Claude
  Code has an objective signal for "done," rather than self-assessing.
- **Local-first by design** — every tool (Ollama, ChromaDB, HuggingFace
  loaders) runs without a paid API key, which matters both for cost
  control while iterating and for letting a reviewer run the repo
  without needing your credentials.
