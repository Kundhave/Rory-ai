# Architecture Decision Records

Append-only. Each entry: Decision / Context / Alternatives / Tradeoff / Date.

---

## ADR-001: No HTTP/FastAPI layer

**Decision**: Rory is one process with no HTTP server, no service boundary,
and no message queue. `RoryCore.handle_text` is called in-process by
whichever adapter (CLI, voice, GUI) is running.

**Context**: Rory is a single-user desktop app that runs on one machine.
There is no second client, no remote caller, and no plan for one — the CLI,
the Qt widget, and the voice pipeline are all adapters running in the same
Python process as the core.

**Alternatives**: A local FastAPI server with adapters as HTTP clients would
let the core run as an independent process and be language-agnostic to
callers. It would also add a serialization boundary, a port to manage, a
process to keep alive, and a dependency (FastAPI) explicitly rejected in
CLAUDE.md.

**Tradeoff**: No horizontal scaling, no remote access, no multi-client
support — none of which this project needs. In exchange: one process to run,
no network layer to reason about or secure, and a function call instead of a
request as the unit of testing. If Rory ever needs a second machine or a
second concurrent client, this decision gets revisited then, not preemptively.

**Date**: 2026-08-26

---

## ADR-002: CLI is the primary development interface, not a debug afterthought

**Decision**: `rory/cli.py` is a first-class part of this feature, built
alongside `core.py`, not bolted on after voice/GUI work.

**Context**: `RoryCore.handle_text(str) -> Reply` is defined as audio- and
UI-agnostic specifically so it can be exercised without a microphone, a
speaker, or a Qt event loop. The CLI is the adapter that proves that
contract holds — if the core can only be driven through voice or the widget,
its audio-agnosticism is untested and easy to accidentally violate.

**Alternatives**: Treat the CLI as a throwaway debug script written after the
"real" (voice) interface exists. This is the more common approach in voice
projects, since voice is the actual product.

**Tradeoff**: Slightly more upfront structure (a real REPL loop, not a
scratch script) in exchange for a fast, deterministic, hardware-free feedback
loop for every feature built on top of the core — including all of this
feature's tests, which drive `RoryCore` directly with a fake LLM, never
through the CLI or a real API call.

**Date**: 2026-08-26

---

## ADR-003: Explicit whitelisted tools instead of shell access

**Decision**: Rory can launch exactly the applications named in
`rory/tools/desktop.py::APPS`. Each entry maps an enum key to a hardcoded argv
list. The model chooses a key; it never produces a command, a path, or a
string that reaches `subprocess`. There is no `run_command` tool and there
will not be one.

**Context**: A single `run_shell(command: str)` tool would make every desktop
capability available at once and would be about ten lines of code. It is also
the most direct prompt-injection target imaginable: Rory reads a personal
knowledge base and will later transcribe arbitrary audio, so text an attacker
controls can reach the model's context. If the model can emit a command
string, that text can emit a command string.

**Alternatives**: (a) `run_shell` with a regex/denylist validator — denylists
on shell syntax are famously leaky (`$()`, backticks, `;`, newline, encoding
tricks), and a validator that is wrong once is wrong permanently. (b) An
allowlist of full command strings parsed with `shlex` — better, but still puts
model output on the command line and requires the parser to be correct.
(c) The enum-key approach taken here.

**Tradeoff**: Rory can do dramatically less. Adding a capability is a code
change, not a conversation — the model cannot improvise its way to a new
ability, which is exactly the point. What is bought: the set of processes Rory
can ever start is finite, auditable by reading one dict, and fixed at import
time. Validation rejects an out-of-enum value before any tool body executes,
so the failure mode of a compromised or confused model is a rejected tool call
rather than an executed command.

**Date**: 2026-08-26

---

## ADR-004: Three tools, not eight

**Decision**: V1 ships `open_app`, `check_app_running`, and `get_datetime`.
Variation lives in an enum argument, not in the number of functions. The
initial sketch had roughly eight tools — `open_browser`, `open_terminal`,
`open_chatgpt`, `get_date`, `get_time`, and so on.

**Context**: "Open my browser" and "open ChatGPT" are the same action with
different arguments. `get_date` and `get_time` read the same clock. Splitting
them multiplies the tool count without adding a single capability.

**Alternatives**: One function per app reads more explicitly at the call site
and lets each carry a tailored description. It also means every new app is a
new function, a new schema, and a new test — and it inflates the tool list sent
on *every* request, which is both token cost and a documented source of model
confusion as the list grows.

**Tradeoff**: Individual tool descriptions are slightly more generic, so the
model leans on the enum values to pick correctly — which makes naming the keys
well (`browser`, not `chromium`) load-bearing. In exchange, adding an app is
one line in `APPS`, and because `AppName = Literal[*APPS]` derives the schema
enum from the whitelist keys, the schema and the whitelist cannot drift apart.
There is no second list to forget to update.

**Date**: 2026-08-26
