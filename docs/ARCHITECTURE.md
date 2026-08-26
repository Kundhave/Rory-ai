# Architecture

Rory is a single Python process with three bands. Dependencies point downward only.

```
Adapters   ui/widget.py, cli.py, voice/*      audio + pixels
              ↓  str in, Reply out
Core       core.py, agent/*, llm.py,          pure text, no I/O devices
           tools/*, rag/*
              ↓  ═══ trust boundary ═══
External   subprocess, psutil, cloud APIs,    side effects + network
           knowledge/*.md, data/*
```

The load-bearing rule: `RoryCore.handle_text(str) -> Reply` is the entire
boundary between "how the user talks to Rory" and "how Rory thinks." Adapters
translate their medium (audio, Qt widgets) down to text going in and a `Reply`
coming out. Nothing in `core/`, `agent/`, `tools/`, or `rag/` knows an
audio device or a Qt event loop exists.

## Feature 1 components (this feature)

- **`rory/config.py`** — `Settings`, loaded via `pydantic-settings` and validated
  at import time. A missing `GEMINI_API_KEY` raises `SystemExit` with a clear
  message immediately, rather than surfacing as a confusing failure on the
  first turn.

- **`rory/llm.py`** — the `LLM` Protocol (`generate(messages, system) -> LLMResponse`)
  and `GeminiLLM`, its one implementation. `LLMResponse` carries text, tool
  calls, and token usage — the three things every provider gives us. Tool
  calls are unused this feature but the shape exists now so Feature 2 (tool
  calling) is an addition, not an interface change.

- **`rory/core.py`** — `RoryCore`, the facade. Owns the session's message
  history (a simple in-memory list), generates a `turn_id` per call, opens and
  closes a `Trace` for that turn, and catches any exception so a turn always
  returns a `Reply` — never an unhandled exception. `Reply` is a plain
  dataclass: `turn_id`, `text`, `error`. No audio or UI concepts.

- **`rory/trace.py`** — one JSON object per line, correlated by `turn_id`,
  with `elapsed_ms` per stage. Logs metadata (lengths, token counts) by
  default; `RORY_TRACE_VERBOSE=true` opts into logging full text for local
  debugging. See [DECISIONS.md](DECISIONS.md) for why this default matters.

- **`rory/agent/prompts.py`** — the system prompt: a fixed persona plus a
  `PROFILE_CARD` placeholder. Feature 3 (RAG) will populate the profile card
  from the personal knowledge base without changing how the prompt is built.

- **`rory/cli.py`** — a text REPL over `RoryCore.handle_text`. The primary
  development interface; see DECISIONS.md for why.

## The tool pipeline (Feature 2)

```
model emits ToolCall(name, arguments)   <- untrusted
        ↓
registry.dispatch(name, arguments)
        ↓  tool exists?            no -> {ok: false, error}
        ↓  arguments validated?    no -> {ok: false, error}   <- enum checked HERE
        ↓  ═══════════ trust boundary ═══════════
tool function runs           APPS[app] -> hardcoded argv list
        ↓  raises?                 yes -> {ok: false, error}
        ↓
truncate_result(...)  ~2KB
        ↓
appended to history as a TOOL RESULTS message -> next iteration
```

**`rory/tools/registry.py`** — `@tool` registers a function, deriving its JSON
schema from type hints. A `Literal[...]` hint becomes an enum. `dispatch`
validates *completely* before the tool body runs and never raises: an unknown
tool, a bad argument, or an exception inside a tool all become an ordinary
`{ok: false, error}` result the model can read and explain.

**`rory/tools/desktop.py`** — the whitelist. `APPS` maps an enum key to an
`App` holding a hardcoded argv list and the process name to match. Model
output is only ever a dict *key*; nothing it produces is interpolated into a
command. `AppName = Literal[*APPS]` derives the schema enum from the whitelist
itself, so the two cannot drift.

**`rory/tools/clock.py`** — `get_datetime`, no arguments, no I/O.

**`rory/agent/loop.py`** — generate → dispatch → append results → repeat, with
a hard cap of `MAX_ITERATIONS = 4`. Tool exchanges are written into the normal
message history as text (`TOOL CALLS` / `TOOL RESULTS`), which keeps `Message`
a plain `{role, content}` pair rather than adding a provider-specific tool
message type.

### Verified vs. running

`check_app_running` returns `verified` separately from `running` because
"it is not running" and "I could not tell whether it is running" are different
answers, and collapsing them is how an assistant ends up fabricating. Four
outcomes:

| Situation | ok | running | verified |
|---|---|---|---|
| Process matched | true | true | true |
| Full scan, no match | true | false | true |
| Some processes unreadable, no match | true | false | **false** |
| Scan failed, or app has no distinct process name | false | null | **false** |

`running: null` rather than `false` in the last row is deliberate — there is no
value the model could mistake for an answer. The grounding rules in
`agent/prompts.py` tell it to report uncertainty when `verified` is false, but
the *mechanism* is this envelope, not the prompt.

## Where state lives

- Conversation history: in-memory, inside a `RoryCore` instance, bounded to
  `MAX_HISTORY_MESSAGES` (a sliding window — see `core.py`). Lost when the
  process exits; there is no persistence in this feature.
- Trace events: appended to `logs/trace.jsonl` (gitignored).
- Configuration: `.env`, loaded once at import via `Settings`.

## What's deliberately not here yet

Voice (`voice/*`), tools (`tools/*`), and retrieval (`rag/*`) are named in the
target repository structure but not built in this feature. They will attach
to `RoryCore` — voice as an adapter calling `handle_text`, tools and RAG as
things the agent loop and prompt-builder use internally — without changing
the adapter-facing contract.
