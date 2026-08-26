# CLAUDE.md — Rory

## What this is

Rory is a voice-first personal AI assistant that runs as a small desktop app on Linux.
Click to talk, speak, get a spoken answer. It answers from a personal markdown knowledge
base and can perform a few whitelisted desktop actions.

**This is a learning project.** The owner is building it to understand modern AI agent
architecture in depth. A small codebase that is fully understood beats a large one that
works. Optimise for clarity over cleverness, always.

## Architecture

Three bands. Dependencies point downward only.

```
Adapters   ui/widget.py, cli.py, voice/*      audio + pixels
              ↓  str in, Reply out
Core       core.py, agent/*, llm.py,          pure text, no I/O devices
           tools/*, rag/*
              ↓  ═══ trust boundary ═══
External   subprocess, psutil, cloud APIs,    side effects + network
           knowledge/*.md, data/*
```

Load-bearing rules:

1. **The core is audio-agnostic.** `RoryCore.handle_text(str) -> Reply` is the whole
   system. Voice and GUI are adapters over that one method. Never let audio types,
   Qt types, or device state leak into `core/`, `agent/`, `tools/`, or `rag/`.
2. **The LLM is a router, not an executor.** It emits tool *names and arguments*. It
   never produces a command, a path, or a shell string.
3. **Every external provider sits behind a small Protocol** (LLM, STT, TTS, Embedder).
   Normalise only what we use — not every provider feature.
4. **One process.** No HTTP layer, no services, no message queue.

## Tech stack

Python 3.11+ · pydantic-settings · httpx · google-genai · sounddevice · soundfile ·
numpy · fastembed · psutil · PySide6 · pytest

That is the complete dependency list. **Adding a dependency requires asking the owner
first.** Explicitly rejected: LangChain, LlamaIndex, Chroma, FAISS, FastAPI, Docker,
SQLAlchemy, Redis, any agent framework.

## Repository structure

```
rory/
  config.py         Settings (pydantic-settings), validated at import
  core.py           RoryCore facade — session, turn_id, trace, error envelope
  cli.py            text REPL — the primary development interface
  trace.py          structured JSONL turn logging
  llm.py            LLM Protocol + one provider implementation
  agent/loop.py     the tool-calling loop
  agent/prompts.py  persona, profile card, grounding rules
  rag/ingest.py     chunk + embed + write index
  rag/embed.py      Embedder Protocol + local model
  rag/retrieve.py   load index, cosine search, score threshold
  tools/registry.py @tool decorator, schema generation, validation, dispatch
  tools/desktop.py  THE WHITELIST — subprocess + psutil
  tools/clock.py    get_datetime
  voice/audio.py    record to WAV bytes, play WAV bytes
  voice/stt.py      STT Protocol + Sarvam
  voice/tts.py      TTS Protocol + Sarvam + local fallback + cache
  ui/widget.py      Qt widget, 5 states, worker thread
```

No `utils.py`, no `helpers.py`, no `base/`, no `interfaces/`, no `services/`.
A Protocol lives beside its implementation until there are two implementations.
Split a file when it has two reasons to change — not when it gets long.

## Security rules — non-negotiable

- `subprocess` is called with an **argv list** and `shell=False`. Never `shell=True`,
  never `os.system`, never string interpolation into a command.
- Model output reaches `subprocess` only as a **dict key** into a hardcoded whitelist,
  never as a value. The tool schema enum *is* the whitelist.
- No V1 tool accepts a filesystem path. No V1 tool writes, deletes, or sends anything.
- Truncate tool results (~2KB) before returning them to the model.
- Secrets come from environment only. Never logged, never in prompts, never committed.
- `knowledge/`, `data/`, `logs/`, `.env` are gitignored.

## Behaviour rules

- **Never fabricate a successful action.** Tools return `{ok, data, error}`; process
  checks additionally return `verified: bool`. If `ok` is false or `verified` is false,
  Rory says plainly that it could not do or check the thing.
- **Never invent personal facts.** If retrieval returns nothing above threshold, Rory
  says it isn't in the notes. It does not guess project names, dates, or details.
- Retrieval returning fewer than `k` results is correct behaviour, not a bug.
- The agent loop has a hard iteration cap. A confused model must not loop forever.

## Code style

- Type hints on public functions. Dataclasses for structured returns.
- Functions under ~40 lines. Files under ~200 lines. If either is exceeded, the design
  is probably wrong — say so rather than splitting mechanically.
- Comments explain *why*, never *what*. No docstrings that restate the signature.
- No `try/except` unless a specific failure is being handled in a specific way.
- Logging goes through `trace.py`. No stray `print` outside `cli.py`.

## What not to do

Do not add: abstract base classes with one implementation, factories, dependency
injection containers, plugin systems, config for things that never vary, retry
decorators, generic `Result`/`Maybe` wrappers, speculative "we might need this" code,
`# TODO` comments, or defensive error handling for conditions that cannot occur.

Do not restructure code outside the current feature's scope.
Do not "improve" working code that wasn't part of the request.

## Working method

1. **Read before writing.** Inspect existing files and match their conventions.
2. **State the plan** — files to create, files to modify, key decisions — before coding.
3. Implement.
4. Run the tests. Fix what fails.
5. **Summarise**: what changed, why the important decisions were made, what tradeoffs
   were accepted, and what the owner should inspect personally.

Explain the *interesting* parts: the agent loop, the security boundary, the retrieval
scoring, the threading model. Do not narrate trivial code.

## Testing

Minimum meaningful tests, not exhaustive suites. For each feature cover: the happy path,
the important edge case, and the failure case. Use `tests/fakes.py::FakeLLM` for anything
touching the agent loop — agent tests must be deterministic, offline, and free.
Tests that hit real APIs are marked `@pytest.mark.live` and are not part of the default run.

## Documentation

Only four documents are maintained:

- `README.md` — what it is, how to run it, what it does, known limitations
- `docs/ARCHITECTURE.md` — components, data flow, where state lives
- `docs/DECISIONS.md` — append-only ADRs: Decision / Context / Alternatives / Tradeoff / Date
- `knowledge/README.md` — knowledge base format and data-sensitivity note

Update them only when a feature changes something they assert. Add an ADR whenever a
non-obvious engineering decision is made. Do not create new documentation files.

## When to STOP and ask

Stop immediately, explain clearly, and wait — do not work around it, do not stub it,
do not assume it's done:

- **A manual action is needed** (API key, account signup, system package, audio device
  config, permissions, desktop setting). Give exact step-by-step instructions, state
  what you need back, and stop.
- **An architectural decision in this file appears wrong.** Explain the problem, why
  the current approach is insufficient, the proposed alternative, and the tradeoff.
  Get approval before changing it.
- **A new dependency seems necessary.** Justify it and wait.
- **The feature is ambiguous or larger than described.** Ask rather than guess.s