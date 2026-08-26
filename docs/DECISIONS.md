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

---

## ADR-011: PySide6 over GTK/Tkinter for the desktop widget

**Decision**: The desktop widget is built with PySide6 (Qt), which was
already this project's declared dependency for the GUI band before this
feature was built.

**Context**: PySide6 was already in CLAUDE.md's fixed tech-stack list from
before Feature 1 was written, so this wasn't really an open choice made
during this feature — it's worth recording *why* it was the right call
regardless, since the alternatives were genuinely viable.

**Alternatives**: Tkinter (stdlib, zero install cost, but its threading story
is weaker — it has no equivalent of Qt's automatic cross-thread signal
marshaling, so a correct worker-thread design needs more manual plumbing
around `after()` polling). GTK (a fine toolkit, first-class on Linux, but a
second UI framework to learn with no reuse against Qt's very complete
`QSystemTrayIcon`/`QSocketNotifier`/threading primitives, all of which this
feature leans on directly).

**Tradeoff**: PySide6 is a heavier dependency (network of Qt shared
libraries) than Tkinter's zero-cost stdlib inclusion. In exchange: signals
crossing threads safely is a built-in primitive rather than something to
hand-build, `QSystemTrayIcon` and `QSocketNotifier` are both first-class and
already proven in this exact feature, and the whole threading model this
feature's correctness depends on ("every state change crosses back to the
main thread via a Qt signal") is a framework guarantee, not a convention
this project has to enforce by hand.

**Date**: 2026-08-26

---

## ADR-012: Tray + fixed-position popup + socket trigger, not a floating always-on-top window

**Decision**: The widget's primary UI is a system tray icon plus a small
popup shown at a fixed screen position on click, with an additional Unix
socket (`$XDG_RUNTIME_DIR/rory.sock`) that an external trigger (a compositor
keybind) can write to for the same toggle effect. There is no attempt at a
floating, always-on-top, click-to-talk window, and no attempt to register a
global hotkey from inside the application.

**Context**: This project's actual desktop environment is Hyprland under
Wayland (confirmed via `$XDG_SESSION_TYPE`/`$XDG_CURRENT_DESKTOP` before this
feature was built, not assumed). Under Wayland, a client cannot position its
own top-level window, and "always on top" is compositor policy rather than
something an application can request and have honored consistently. A
global-hotkey library mostly cannot hook keyboard input at all under
Wayland's security model, since that would mean one client eavesdropping on
every keystroke system-wide. Building against either assumption would work
by accident on some compositors and silently misbehave on others — exactly
what CLAUDE.md's "do not build something that silently misbehaves" was
warning against.

**Alternatives**: (a) A floating always-on-top widget positioned near the
cursor or screen corner — works on X11, unreliable-to-broken on Wayland
compositors depending on their specific window-management protocol support.
(b) An in-process global hotkey library — mostly non-functional under
Wayland for the security reason above, and would need an X11-specific
fallback path this project isn't trying to maintain. (c) A compositor
scripting integration (e.g. a Hyprland-specific IPC call) — ties this
project to one compositor, when the goal is a Linux voice assistant, not a
Hyprland-specific one.

**Tradeoff**: No exact "appears right next to the tray icon" positioning
(the popup opens at a fixed offset instead), and no self-contained global
hotkey — the user must bind a keypress in their own compositor/DE to write to
the socket (documented in README.md for Hyprland specifically, since that's
this project's actual environment). In exchange: the tray icon and the popup
both behave identically across any Wayland compositor with tray support, the
socket approach works identically under X11 too with zero special-casing,
and the keybinding stays exactly where it belongs — owned by the desktop
environment, which is the only thing actually allowed to own it under
Wayland's security model.

**Date**: 2026-08-26

---

## ADR-013: Persistent SVG sticky widget supersedes the tray+popup UI (amends ADR-012)

**Decision**: The desktop widget's primary UI is now a single persistent
window showing the user's own `assets/images/widget.svg`, always open for
Rory's whole run, with a state-colored border and a text panel that appears
only when there's a transcript/reply to show. Click the widget itself to
start/stop listening. There is no system tray icon anymore. The Unix socket
trigger from ADR-012 is unchanged and stays.

**Context**: ADR-012 chose tray+popup specifically because a tray icon is
the one piece of screen real estate Wayland reliably grants a client. That
reasoning still holds — nothing about Wayland changed. What changed is the
actual requirement: the owner wants a specific piece of artwork
permanently visible on the desktop as the primary interface, not hidden
behind a tray icon until clicked. A tray-first design cannot satisfy
"visible at all times" — a tray icon is small, generic, and the popup it
opens is explicitly transient in ADR-012's own design.

**What was tested, not assumed, before committing to this**: the window's
own `move()` call was confirmed to have zero effect on its Wayland-reported
position, in both the default (tiled) state and after toggling `floating`
on via a live `hyprctl` rule — position is Hyprland's decision every time,
matching ADR-012's underlying premise but now confirmed for a *persistent*
window too, not just a click-triggered popup. Live-tested three separate
`windowrule`/`windowrulev2` syntax variants against the running compositor
(Hyprland 0.56.0) attempting to force float+pin; every variant was rejected
by the compositor's config parser. Rather than document a rule as "correct"
that was never confirmed working, README.md tells the owner what shape the
rule needs (float + pin, matched by title `Rory`) and to verify the exact
syntax against their installed version, instead of asserting a specific
line with false confidence.

**Alternatives**: Keep the tray+popup and add the SVG as the popup's
content instead of the drawn-circle icon. Rejected because it still hides
the widget behind a click most of the time, which directly contradicts
"remain visible on the desktop at all times."

**Tradeoff**: No tray icon means no `QSystemTrayIcon.isSystemTrayAvailable()`
fallback path to worry about, but also no minimized/background-only mode —
the window is always on screen somewhere, which is exactly the point, but
means it takes up permanent space unless the user's compositor is configured
to make it float+pin+move to an out-of-the-way spot (a config change only
the user can make, per above). `VoiceWorker` — the actual state machine and
threading model from Feature 5 — did not change at all; only the
presentation layer wrapping it did, confirmed by the untouched
`tests/test_widget.py` state-transition tests passing unmodified against
the new UI.

**Date**: 2026-08-26

---

## ADR-014: Sarvam bulbul:v3 over bulbul:v2, and retry-before-degrade for TTS

**Decision**: Sarvam TTS uses `bulbul:v3` (speaker `shreya`) rather than
`bulbul:v2` (`manisha`); the per-request timeout is 5s rather than 60s; and
`FallbackTTS` retries the primary engine up to 3 times before degrading to
local espeak-ng.

**Context**: Two user-visible symptoms — replies taking ~30s to speak, and
the voice sometimes sounding robotic and noisy — turned out to share one
root cause. Measurement against the live API:

| model / speaker | success rate | timings |
|---|---|---|
| `bulbul:v2` / manisha | 2/5 | 1.2s, ✗30s, ✗30s, 1.1s, ✗30s |
| `bulbul:v3` / shreya | 5/5 | 1.7–2.7s |

`bulbul:v2` fails roughly half the time, and each failure hangs for ~30.3s
before returning a 500 (Sarvam's own server-side timeout). Our 60s httpx
timeout never fired, so every failure cost the full 30s — and then
`FallbackTTS` degraded to espeak-ng, whose formant synthesis is the
"noise" the user heard. So the 30s wait and the bad voice were the same bug
observed from two angles.

Successful calls barely scale with length (0.95s at 25 chars, 2.12s at 105,
2.13s at 300), so success and failure are cleanly separated populations —
which is what makes a short timeout safe.

**Alternatives**: (a) Retry `bulbul:v2` harder — measured at 12s timeout ×
3 attempts this still averaged 11.1s with 2/6 turns degrading to espeak-ng;
it treats the symptom while staying on the broken model. (b) Hedged
parallel requests (fire N, take the first success) — a real fix for this
failure profile, but multiplies API cost and adds concurrency complexity
for what turned out to be an unnecessary workaround once v3 was tested.
(c) Accept the local voice as normal — rejected; it's audibly much worse
and the user specifically objected to it.

**Tradeoff**: Speaker names are model-specific, so moving to v3 meant giving
up `manisha` (v2-only) and choosing a new voice. The retry and short timeout
are kept anyway despite v3's reliability — they're cheap insurance for any
future transient wobble, and cost nothing on the success path. Measured
result: mean TTS latency 11.13s → 2.06s, max 15.68s → 2.94s, and zero
degradations to the local voice across the verification run.

`FallbackTTS.last_engine` was added alongside this and is logged to the
trace, so "why does it sound bad?" is answerable from the trace file
instead of being invisible — that ambiguity is what made this take two
rounds to diagnose.

**Date**: 2026-08-26
