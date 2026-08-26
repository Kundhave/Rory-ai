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
