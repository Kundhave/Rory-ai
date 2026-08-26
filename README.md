# Rory - My Personal AI Voice Assistant

## What is Rory?

> A character from my favorite show, Gilmore Girls. 
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
- [Technology Stack](#technology-stack)
- [Security](#security)
- [Failure Handling](#failure-handling)
- [Testing](#testing)
- [Project Structure](#project-structure)

- [Known Limitations](#known-limitations)
- [Future Scope](#future-scope)

## What actually is rory? 

Rory is a personal, voice-first AI agent that combines **Speech-to-Text, LLM-based reasoning, RAG,memory, and controlled tool execution** to understand personal context, respond naturally, and interact with my Linux desktop.

The system follows a modular **STT → LLM → TTS** architecture, with RAG and memory providing context and an explicit tool layer enabling desktop actions.

## How it looks:

![desktop widget](assets/images/desktop_widget.png)

## Why I Built Rory?

I built Rory because I wanted a project that was **genuinely fun to use while forcing me to understand how AI agents actually work** - from voice pipelines and RAG to memory, tool calling, and system architecture.

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

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the component breakdown, and
[docs/DECISIONS.md](docs/DECISIONS.md) for the reasoning behind non-obvious choices.

## Request Lifecycle

![Rory Request Lifecycle](assets/images/rory-request-lifecycle.svg)

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Rory needs a Gemini API key:

1. Get one at https://aistudio.google.com/apikey
2. `cp .env.example .env`
3. Put the key in `.env` as `GEMINI_API_KEY=...`

Rory fails loudly at startup if `GEMINI_API_KEY` is missing — it will not run
half-configured.

## Running the CLI

The CLI is the primary way to develop and exercise Rory's core — no audio
hardware or desktop environment required.

```bash
python -m rory.cli
```

Type a message, get a reply, keep talking — it's a multi-turn conversation.
Type `exit` or press Ctrl-D to quit. Press Enter on an empty line to talk
instead of type — see Voice below.

## Voice

Rory speaks every answer, and can take spoken input instead of typed. Voice
is entirely an adapter over the same `RoryCore.handle_text` used for typed
input — nothing about tools, retrieval, or the agent loop changes based on
whether a turn started as text or speech.

### Setup

1. **Local voice (default, free)** — install espeak-ng:
   ```bash
   sudo pacman -S espeak-ng      # Arch
   # or: sudo apt install espeak-ng   (Debian/Ubuntu)
   ```
   `TTS_ENGINE=local` in `.env` (or unset — it's the default) uses this,
   never touches your Sarvam credits, and needs no network.

2. **Sarvam voice (optional, real neural TTS/STT)** — get a key at
   https://dashboard.sarvam.ai, put it in `.env` as `SARVAM_API_KEY=...`.
   Speech input (STT) always needs this key regardless of `TTS_ENGINE`;
   there is no local fallback for STT. Set `TTS_ENGINE=sarvam` to use the
   real Bulbul voice for output too — pick a speaker with `TTS_VOICE`
   (v3 female: `shreya`, `priya`, `ritu`, `kavya`, `ishita`; v3 male:
   `shubh`, `aditya`, `rahul`, `varun`. `espeak-ng --voices=en` lists local
   options for `TTS_LOCAL_VOICE`).
   If Sarvam fails, Rory retries a few times and then falls back to the
   local voice rather than losing the turn's spoken answer.

   **Stay on `bulbul:v3` (the default).** Measured against the live API,
   `bulbul:v2` succeeded on only 2 of 5 calls, with each failure hanging
   ~30s before erroring and then degrading to the robotic local voice;
   `bulbul:v3` succeeded 5 of 5 in ~2s. Speaker names are model-specific —
   v2 names like `manisha` do not exist on v3. See ADR-014 in
   `docs/DECISIONS.md`.

   If speech ever sounds robotic, check which engine actually spoke:
   ```bash
   grep '"stage": "tts"' logs/trace.jsonl | tail -5
   ```
   `"engine": "primary"` is the real Sarvam voice; `"fallback"` means you
   heard local espeak-ng because Sarvam failed.

### Using it

- **Typed in, spoken out**: type normally. The text answer is always printed
  first, then spoken — if TTS fails for any reason, you still see the answer.
- **Spoken in**: press Enter on an empty prompt to start recording, press
  Enter again to stop (click-to-stop — no voice-activity detection, no
  silence detection, no wake word: recording is exactly what happened
  between the two keypresses). What Rory *heard* is always printed before
  the answer — speech recognition mishearing a name or project ("Relay"
  becoming "railay") is the most common failure in a voice pipeline over
  personal notes, and seeing the transcript is the cheapest way to catch it.
  A recording that transcribes to nothing usable (silence, noise) is
  rejected before it ever reaches the LLM, with a prompt to try again.
- Repeated phrases are cached (`data/tts_cache/`, gitignored, regenerable) —
  hearing the same answer twice costs nothing the second time.

## Knowledge base and retrieval

Rory answers personal questions (projects, goals, ideas) by searching
`knowledge/*.md` through a `search_notes` tool — see
[knowledge/README.md](knowledge/README.md) for the file format (real
markdown headings matter a lot for retrieval quality) and a data-sensitivity
note before putting anything in there.

```bash
python -m rory.rag.ingest
```

Builds `data/index.npz` and `data/chunks.json` from whatever is in
`knowledge/`. First run downloads the local embedding model
(`BAAI/bge-small-en-v1.5`, ~100-500MB) — expect a delay. Both output files are
fully disposable; delete them and re-run this any time the notes change.

### Desktop tools

Rory can launch a small whitelisted set of desktop apps and check whether one
is running (`rory/tools/desktop.py::APPS`) and read the current date/time —
see `docs/ARCHITECTURE.md` for the security model.

## Evaluation: does RAG actually help at this scale?

`tests/golden.yaml` has ~20 hand-written questions against the real knowledge
base — which tool should fire, which source document should show up in the
top-3 retrieved chunks, what the answer must or must not say. `rory/eval.py`
runs them against the real Gemini API two ways:

```bash
python -m rory.eval             # RAG: search_notes retrieves on demand
python -m rory.eval --stuff     # comparison: whole knowledge base pasted into the prompt
```

Both runs used `gemini-flash-lite-latest` against the real knowledge base
(186 indexed chunks from `knowledge/*.md`), on 2026-08-26. Whatever these
numbers show is what's recorded here — this was not tuned after the fact.

| | RAG (`search_notes`) | `--stuff` (whole KB in prompt) |
|---|---|---|
| Golden cases passed | 16/20 | 19/20 |
| Avg. prompt tokens/call | **960** | **12,088** (12.6x more) |
| Calls per case | 1-2 (tool round trip when it fires) | 1 (2 for tool-only cases) |
| Latency per case | 7-155s (occasional free-tier retry backoff) | consistently ~7.5s |

**RAG's 4 failures**, and what they actually show:

- Two (`reloop-competition`, `techtrendgpt-purpose`) are **routing** failures,
  not retrieval failures: the always-present profile card
  (`agent/prompts.py::PROFILE_CARD`) already names these projects with a
  one-line description, so the model judged it already knew enough and never
  called `search_notes` — then missed a specific fact (an exact ranking
  number) that only lives in the full notes. This is a direct, measured
  consequence of the profile-card design (see ARCHITECTURE.md): it buys
  cheap identity on every turn at the cost of occasionally short-circuiting a
  lookup that would have gotten a more precise answer.
- One (`rag-projects`, "which of my projects use RAG?") is a genuine
  **retrieval** miss — `search_notes` fired, but the top-3 chunks were from
  `ABOUT_ME.md`/`IDEAS.md` rather than `PROJECTS.md`'s dedicated RAG-project
  list. A real gap in chunk ranking for broad/enumerative questions, not a
  routing problem.
- One (`engineering-specialization`) is a bug in the golden case itself, not
  the system: `expect_source: [GOALS.md, ABOUT_ME.md]` was written meaning
  "either is acceptable," but the harness checks that *every* listed source
  appears in top-3. Left as-is rather than quietly fixed and re-run — flagging
  it here for whoever reviews `tests/golden.yaml` next.

**`--stuff`'s 1 failure** (`internship-comp-target`, missing "25" from the
answer) had the entire knowledge base in context and still didn't reliably
extract one specific number — stuffing doesn't make answers dependable either.

**Conclusion, honestly**: at this knowledge-base size (186 chunks, roughly
15K tokens of raw notes), `--stuff` scored marginally higher on this run,
because it structurally can't have a routing failure — the content is just
already there. RAG's real, measured win is cost: **~92% fewer prompt tokens
per call**, since most turns only pay for a ~1K-token round trip instead of
pasting the whole KB every single time regardless of relevance. RAG's
weakness here isn't retrieval quality (`MIN_SCORE` calibration and chunking
held up fine in 15/16 fired cases) — it's that routing correctness depends on
the model's judgment about whether to call the tool at all, which the profile
card can undercut for exactly the facts it partially covers. That tradeoff
gets more favorable for RAG as the knowledge base grows past what comfortably
fits in a context window; at the current scale, both approaches are cheap and
fast in absolute terms, and the choice is really about future headroom versus
today's slightly higher pass rate.

## Desktop widget

A persistent on-screen widget showing `assets/images/widget.svg` — the same
`RoryCore` and voice pipeline as the CLI, wrapped in a PySide6 GUI. Nothing
about tools, retrieval, or the agent loop changes based on whether a turn
started in the CLI or the widget.

```bash
./run.sh
```

or install `rory.desktop` into your app launcher — copy or symlink it into
`~/.local/share/applications/`. **`rory.desktop`'s `Exec=` path is hardcoded
to this checkout's location; edit it if you clone Rory somewhere else.**

**Using it**: the widget stays open on screen for as long as Rory is
running. Click it to start listening; click again to stop and process. A
colored border around the image shows the current state (idle/listening/
processing/speaking/error) without altering the artwork itself. What Rory
heard and said appears in a small text panel below the image, only while
there's something to show — voice is the primary channel, but the text is
always readable, which is how you catch a misheard name. The **error state
shows the actual error** (a device error, a network failure, an API quota
message), not a generic "something went wrong." Right-click for a Quit
option, or just close the window.

### Known desktop-environment caveats (tested on Hyprland 0.56 / Wayland, `omarchy` session)

- **No custom window positioning, confirmed by testing, not assumed.**
  Wayland doesn't let a client set its own top-level window's screen
  position. I tested this directly: the widget's own `move()` call had no
  effect regardless of whether the window was floating or tiled — position
  is entirely Hyprland's call. Drag it wherever you want after it opens;
  that placement holds for the life of the window.
- **No always-on-top**, for the same reason — it's compositor policy, not
  something a client can request and have honored. The window just stays
  open; it does not hide or close itself between turns.
- **Making it float and stay put is a Hyprland config change only you can
  make** — I could not verify one exact working rule for your Hyprland
  0.56 install; live-tested three syntax variants against the running
  compositor and each was rejected (`windowrulev2` is deprecated in this
  version; the newer `windowrule = match:title ..., <action>` form rejected
  every variant I tried). The working form on Hyprland 0.56 — verified live,
  and now in `~/.config/hypr/hyprland.conf` — needs an explicit **value** on
  each action:
  ```
  windowrule = match:title ^Rory$, float 1
  windowrule = match:title ^Rory$, pin 1
  windowrule = match:title ^Rory$, move 1356 684
  ```
- **Coordinates are in logical, scale-divided pixels — not physical ones.**
  This display is 1920x1080 at `scale=1.25`, so the usable coordinate space
  is 1536x864. A `move` computed against 1920x1080 puts the widget
  off-screen, which looks exactly like "the widget isn't running." To
  recompute after any monitor/scale change:
  ```bash
  hyprctl monitors -j | python3 -c "import json,sys; m=json.load(sys.stdin)[0]; print(m['width']/m['scale'], m['height']/m['scale'])"
  # then: x = logical_width - widget_width - margin, same for y
  ```
- **`exec-once` only fires at Hyprland startup**, not on `hyprctl reload` —
  after adding the autostart line, log out and back in (or just run
  `./run.sh &` once for the current session).
- **No built-in global hotkey**, and none is attempted — hotkey libraries
  mostly can't hook keyboard input under Wayland's security model. Instead,
  Rory listens on a Unix socket at `$XDG_RUNTIME_DIR/rory.sock`; anything
  written to it toggles listening, exactly like clicking the widget. Bind
  your own keypress to it — for Hyprland, add to `~/.config/hypr/bindings.conf`:
  ```
  bind = SUPER, R, exec, echo -n "toggle" | socat - UNIX-SENDTO:$XDG_RUNTIME_DIR/rory.sock
  ```
  (needs `socat`, already on this machine; any tool that can write a
  datagram to a Unix socket works — verified live with exactly this command).

## Measured latency and cost per turn

From `logs/trace.jsonl` over 8 real turns (mix of RAG, tool, and plain
conversation) plus 5 real STT round trips, on 2026-08-27. Measured, not
estimated — and deliberately **not** optimised.

| stage | n | median | worst |
|---|---|---|---|
| `stt` (Sarvam Saaras) | 5 | 0.82s | 1.02s |
| `llm_generate` (per call) | 14 | 0.95s | 1.34s |
| `tool_dispatch` | 6 | 0.03s | 0.25s |
| `tts` (Sarvam Bulbul v3) | 8 | 2.35s | 5.52s |
| `turn_complete` (LLM+tools, excl. STT/TTS) | 8 | 1.99s | 2.10s |

**End-to-end for a spoken turn** ≈ STT + turn_complete + TTS ≈ **4.2s median**.
TTS dominates: it is the single largest stage and roughly half the wall clock.
A tool-calling or RAG turn costs ~2 LLM round trips (14 calls / 8 turns =
1.75 avg), which is why `turn_complete` is ~2x a single `llm_generate`.

**Token cost per turn** (measured): ~1,818 prompt + ~44 completion tokens.
At current Gemini 3.x Flash-Lite rates (~$0.25 / $1.50 per M in/out) that is
**~$0.0005 per turn — about 0.05 cents**, or roughly $1 per 2,000 turns. On
the free tier it is $0, subject to per-minute and per-day request caps.
Sarvam TTS/STT bill against prepaid credits by character/duration
(~116 characters of speech per turn median); check the Sarvam dashboard for
your plan's rate.

The single biggest cost lever already in place is the TTS cache — a repeated
phrase costs nothing the second time.

## Known limitations

Honest list. These are accepted V1 tradeoffs, not bugs to be surprised by.

- **The widget shows no transcript.** It renders only the artwork, so you
  cannot see what Rory *heard* — a misheard name looks like a wrong answer
  rather than a mis-transcription. The CLI prints `heard: ...` and remains
  the better surface for diagnosing that. (An unspoken *answer* is not lost:
  if TTS fails, the reply text appears on the widget — see below.)
- **Bare single-name retrieval queries can miss.** "who is Marcel" scores
  ~0.55 against a 0.58 threshold and returns nothing, while "what is my
  dog's name" scores 0.75. Short, low-signal queries under-retrieve. The
  threshold is calibrated so genuinely off-topic questions return nothing;
  lowering it to catch these would trade a real safety property for a
  convenience one.
- **Latency is not optimised.** ~4.2s median for a spoken turn, dominated by
  TTS. No streaming, no sentence-level pipelining — the whole reply is
  synthesised before any audio plays.
- **Conversation memory is a fixed 20-message window.** Older turns fall out
  silently with no summarisation; a fact mentioned early in a long session
  stops being "remembered" without any warning.
- **STT has no local fallback.** Speech input requires `SARVAM_API_KEY` and
  a working network; TTS degrades to espeak-ng offline, STT simply cannot run.
- **Wayland positioning is compositor policy.** Rory cannot place or pin its
  own window; that lives in `hyprland.conf` (see caveats above).
- **Gemini occasionally emits tool-call syntax as plain text** rather than
  through the structured API. The agent loop detects this against the real
  tool names and makes the model retry, so it should never be spoken — but
  the underlying model behaviour is non-deterministic and outside our control.
- **No authentication, no sandboxing.** Anything that can write a datagram to
  `$XDG_RUNTIME_DIR/rory.sock` can trigger listening. Fine for a single-user
  desktop; not a multi-user design.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

92 tests, all offline and free — no API keys needed. Tests marked `live` hit
real APIs and are excluded from the default run (`pytest -m live` to include
them). `tests/test_failure_modes.py` covers the failure matrix
deterministically with `FakeLLM`.

