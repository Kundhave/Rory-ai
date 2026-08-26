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

---

## ADR-005: No vector database at this scale

**Decision**: Retrieval storage is a NumPy `.npz` file of normalised vectors
plus a `chunks.json` sidecar. There is no Qdrant, Chroma, FAISS, or any other
vector database, embedded or hosted.

**Context**: The entire knowledge base ingests to 183 chunks. Brute-force
cosine similarity over 183 vectors of 384 dimensions is one matrix multiply —
milliseconds, measured, not estimated. A vector database earns its cost at a
scale where linear scan actually becomes slow (tens of thousands of vectors
and up) or where features like filtering, sharding, or concurrent writes from
multiple processes are genuinely needed. None of that is true here: one user,
one process, a knowledge base measured in personal notes rather than a
corpus.

**Alternatives**: Qdrant (already used in some of the user's other projects,
per `knowledge/ABOUT_ME.md`) or Chroma would both work and would look more
like "real RAG infrastructure." Both were explicitly rejected in CLAUDE.md's
dependency list before this feature was even written.

**Tradeoff**: No approximate-nearest-neighbour speedup, no persistence
transaction log, no built-in filtering DSL — none of which matter yet. In
exchange: the entire retrieval storage layer is two NumPy arrays, readable by
opening the file, with no server process to run, no schema to migrate, and
nothing that can be "down." If the knowledge base someday grows to a size
where brute-force search shows up in a trace as slow, that's the trigger to
revisit this ADR — not before.

**Date**: 2026-08-26

---

## ADR-006: Retrieval as a tool, not always-on pre-retrieval

**Decision**: `search_notes` is registered through the same `@tool` decorator
and dispatched through the same registry as `open_app`, `check_app_running`,
and `get_datetime`. There is no separate code path that runs retrieval before
every turn and injects the top-K chunks into the prompt regardless of what
the model asked for.

**Context**: The more common RAG pattern retrieves unconditionally on every
turn — embed the incoming message, always fetch top-K, always stuff it into
context. That means paying an embedding call and a context-window cost on
every turn, including "thanks!", "open my terminal", and "what time is it,"
none of which need personal context at all. It also means the model never
gets to ask a *different* question than the one the user typed — it's stuck
with whatever the raw user message happened to retrieve.

**Alternatives**: Always-on pre-retrieval, injected either into every system
prompt or as a synthetic first turn. Simpler to reason about in one sense —
retrieval always happens — but it conflates "the model decided this needs a
knowledge-base lookup" with "the user's turn happened," which are different
things, and it can't be selective about *when* to look something up versus
when the profile card already covers it.

**Tradeoff**: Retrieval quality now depends on the model's tool-calling
judgment — it has to recognise a question needs `search_notes` and phrase a
sensible query, which the golden-question harness (`rory/eval.py`) exists
specifically to measure. In exchange: retrieval only costs an API round trip
when something is actually being looked up, it composes with the existing
agent loop and grounding rules instead of needing its own, and the model can
issue a query in its own words rather than being stuck with the user's exact
phrasing.

**Date**: 2026-08-26

---

## ADR-007: Local embeddings, not an embedding API

**Decision**: Embeddings are computed locally via `fastembed`
(`BAAI/bge-small-en-v1.5`), both at ingest time and for every query. There is
no call to an embeddings endpoint (Gemini's or anyone else's).

**Context**: Embedding sits on the hot path of every `search_notes` call —
the query has to be embedded before it can be compared to anything, on every
turn that triggers retrieval. A remote embedding API adds network latency and
a second point of failure to that path, on top of the LLM call the turn
already needs. It also means personal notes and every query about them leave
the machine twice (once to embed, once to generate) instead of once.

**Alternatives**: Gemini's embedding endpoint, or another hosted embeddings
API. Would remove the ~100-500MB local model download and keep the
dependency list shorter in spirit (though `fastembed` was already in
CLAUDE.md's tech stack before this feature existed). Would also tie
retrieval's availability and latency to a second external service beyond the
LLM provider already in the critical path.

**Tradeoff**: A first-run model download and the disk space to hold it, plus
being pinned to whatever quality `bge-small` offers rather than a
provider's frontier embedding model. In exchange: retrieval works offline
once ingested, has no per-query cost or rate limit distinct from the LLM
call, and its latency is local compute rather than a second network
round-trip stacked in front of every tool-calling turn.

**Date**: 2026-08-26

---

## ADR-008: Decoupled STT/LLM/TTS instead of a managed realtime voice API

**Decision**: Voice is three independent stages glued together in `cli.py` —
record → `STT.transcribe` → `RoryCore.handle_text` → `TTS.synthesize` → play —
rather than a single managed realtime voice API (e.g. a bidirectional
streaming voice session that handles turn-taking, transcription, and
synthesis as one product).

**Context**: A managed realtime voice API is a genuinely appealing shortcut —
one connection, built-in turn-taking, often lower latency. It also means the
"LLM" and the "voice" become one vendor-shaped black box: tool calling,
grounding rules, and the RAG/tools/agent-loop work already built in Features
2-3 would have to be re-expressed however that specific realtime product
allows, or abandoned. It would also make `RoryCore` no longer the true
single entry point — a chunk of the actual conversation logic would live
inside a provider's realtime session instead of in code this project owns
and tests.

**Alternatives**: A realtime voice API (several providers offer one). Faster
to a demo, and turn-taking/interruption would come for free. But it collapses
three independently-swappable Protocols (`LLM`, `STT`, `TTS`) into one
vendor-specific integration, and this project's whole reason to exist is
understanding each piece — see CLAUDE.md's Protocol-per-provider rule.

**Tradeoff**: No built-in interruption/barge-in, and the turns are visibly
sequential (record, then transcribe, then think, then speak) rather than
streaming. In exchange: `RoryCore.handle_text` stays the one real entry
point regardless of which adapter drives it, STT/LLM/TTS can each be
swapped or mocked independently (proven directly by this feature — `tests/
test_voice.py` never touches a real API), and the tool-calling/RAG work from
Features 2-3 works over voice with zero changes, because voice never talks
to the LLM directly at all.

**Date**: 2026-08-26

---

## ADR-009: Click-to-stop instead of voice-activity detection

**Decision**: Recording starts on Enter and stops on a second Enter. There is
no voice-activity detection (VAD), silence detection, endpointing, or
wake-word — `Recorder` captures exactly what happens between the two
keypresses, nothing more.

**Context**: VAD sounds like a small addition — "just stop recording after a
pause" — but it is a real subsystem: a threshold to tune per-microphone and
per-room-noise-floor, a decision about how much trailing silence is "done
talking" versus "thinking," and a failure mode (cutting someone off
mid-sentence, or never triggering in a noisy room) that is hard to debug
because it's probabilistic rather than a bug you can point at.

**Alternatives**: Energy-threshold VAD (simplest, but the noise-floor problem
above), a proper VAD model (accurate, but a new dependency and a new failure
surface for a V1), or streaming with server-side endpointing (ties this
project to a specific STT provider's streaming API, a much bigger commitment
than the current one-shot `POST` to Sarvam).

**Tradeoff**: The user has to explicitly press Enter twice per utterance —
less magical than "just start talking." In exchange: recording start/stop is
a deterministic keypress with zero tuning, zero per-environment calibration,
and zero probabilistic failure mode. Given CLAUDE.md's "a small codebase that
is fully understood beats a large one that works," an explicit boundary the
user controls is a better V1 trade than a VAD system nobody has tuned.

**Date**: 2026-08-26

---

## ADR-010: TTS cache + local fallback as a cost-control decision

**Decision**: Every synthesized phrase is cached to disk under
`data/tts_cache/`, keyed by `sha256(text + voice)`. Separately, `TTS_ENGINE`
defaults to `"local"` (espeak-ng, free, no network), and `FallbackTTS` drops
to local automatically if Sarvam fails for any reason (credit exhaustion, a
server error, or a timeout — all three occurred for real during this
feature's development).

**Context**: TTS is the one part of this system that, if called naively
during iterative development, keeps charging money for the same handful of
test phrases said over and over. The cache turns "re-run the CLI to check a
change" from a paid API call into a disk read. The `local`-by-default engine
means a fresh checkout of this repo never touches Sarvam credits until
someone deliberately opts in.

**Alternatives**: No cache, relying on developer discipline to not over-test
voice output — unrealistic in practice, this is exactly the kind of cost
that quietly adds up during normal iteration. Or caching only in a specific
"dev mode" flag — more moving parts for no real benefit, since caching
correct, cache-busted-by-voice-change audio is never wrong to do, even in
production (a returning user asking the same question gets an instant
cached answer).

**Tradeoff**: `data/tts_cache/` grows unboundedly with no eviction — accepted
for now since WAV files for short spoken replies are small and this is a
single-user local app, not a multi-tenant service; a size cap can be added
later if it ever matters. The cache key deliberately includes the voice
identifier specifically so changing `TTS_VOICE` (as happened live during
this feature — switching from `anushka` to `manisha`) can never silently
replay stale audio in the wrong voice.

**Date**: 2026-08-26
