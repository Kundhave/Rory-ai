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

## The RAG pipeline (Feature 3)

```
knowledge/*.md (gitignored, personal)
        ↓  python -m rory.rag.ingest
split_into_sections()   header-aware: one section per heading, full
                         heading-path breadcrumb kept as metadata
        ↓
chunk_section()          ~300 words/chunk (≈400 tokens), 15% overlap
                          for sections longer than one chunk
        ↓
Embedder.embed(texts)    FastEmbedder (fastembed, local, no network at
                          query time)
        ↓
normalize()               L2-normalise so search is one matmul
        ↓
data/index.npz (vectors) + data/chunks.json (text + heading + source)
                          both fully regenerable — delete and re-ingest

query at runtime:
  search_notes(query)  <- an ordinary @tool, dispatched through the same
        ↓                  registry as open_app/check_app_running/get_datetime
  embed query, normalize, vectors @ query_vector, top-K, filter < MIN_SCORE
        ↓
  {ok, results: [{source, heading, text, score}, ...]}   (possibly empty)
```

**`rory/rag/embed.py`** — the `Embedder` Protocol and `FastEmbedder`
(`BAAI/bge-small-en-v1.5` via fastembed, 384-dim, local). See DECISIONS.md for
why local rather than an embedding API.

**`rory/rag/ingest.py`** — `split_into_sections` walks the markdown line by
line, tracking a heading stack so a chunk under a level-3 heading still
carries its level-1/level-2 ancestors (e.g. `PROJECTS.md > 2. Relay > Core
Architecture`). `chunk_section` only splits a section that exceeds
`CHUNK_WORDS`, with `OVERLAP_WORDS` shared between consecutive pieces so a
fact near a boundary survives whole in at least one chunk. `README.md` is
excluded — it documents the format, it isn't personal knowledge. Word count
stands in for token count; there's no tokenizer in this project's
dependencies, and getting chunk size roughly right matters more than getting
it exactly right.

**`rory/rag/retrieve.py`** — owns `MIN_SCORE` and does the actual search
(`vectors @ query_vector`, since both sides are pre-normalised). It also
defines `search_notes`, the fourth tool, decorated with the same `@tool` used
by the desktop tools — there is no separate retrieval code path, no
always-on pre-retrieval step run before the model sees a message. If the
model doesn't ask, nothing is retrieved.

### Why returning zero results is correct, not a bug

`MIN_SCORE` was calibrated by hand against this knowledge base, not guessed:
off-topic queries ("what's the capital of France") top out around 0.55
cosine similarity against `bge-small`, genuinely relevant queries start
around 0.65 even when loosely phrased. `MIN_SCORE = 0.58` sits in that gap.
Below it, `search_notes` returns `results: []` rather than the
nearest-but-irrelevant chunk — this is the mechanism (not a prompt
instruction) that lets Rory say "that's not in my notes" instead of
confabulating from whatever scored highest regardless of whether it was
actually relevant.

### The profile card is not retrieved

`agent/prompts.py::PROFILE_CARD` is a small (~150 token) hand-written summary
— name, current focus, top projects — sent on *every* turn regardless of
whether `search_notes` fires or what it returns. Retrieval is precise but
conditional: it only surfaces content for a query that scores above
`MIN_SCORE`, so a turn that never calls `search_notes` (small talk, a
desktop-tool request, a phrasing that doesn't match the notes' wording) would
otherwise carry zero personal context. Core identity shouldn't be
retrieval-dependent; everything more specific than the card (project
internals, exact figures, brainstormed ideas) is left to `search_notes` on
purpose, which is what keeps the card small enough to afford on every turn.

## Where state lives

- Conversation history: in-memory, inside a `RoryCore` instance, bounded to
  `MAX_HISTORY_MESSAGES` (a sliding window — see `core.py`). Lost when the
  process exits; there is no persistence in this feature.
- Trace events: appended to `logs/trace.jsonl` (gitignored).
- Configuration: `.env`, loaded once at import via `Settings`.
- Knowledge base: `knowledge/*.md` (gitignored, personal, hand-written).
- The retrieval index: `data/index.npz` + `data/chunks.json` (gitignored,
  entirely derived from `knowledge/` — delete and re-run
  `python -m rory.rag.ingest` at any time).

## The voice layer (Feature 4)

Voice is purely an adapter on top of `cli.py`. `RoryCore`, `agent/*`,
`tools/*`, and `rag/*` did not change at all for this feature — voice input
becomes a `str` (a transcript) before it ever reaches `RoryCore.handle_text`,
and voice output starts from the plain `str` on `Reply.text`.

```
TTS (out):
  reply.text
      ↓
  TTS.synthesize(text) -> WAV bytes    (CachedTTS -> FallbackTTS -> Sarvam/local)
      ↓
  audio.play(wav_bytes)                 sounddevice, blocks until done

STT (in):
  Enter (start) ... Enter (stop)        click-to-stop, no VAD/endpointing
      ↓
  Recorder captures at the DEVICE'S OWN native rate
      ↓
  audio.resample(..., to_rate=16000)    linear interpolation (numpy only)
      ↓
  audio.to_wav_bytes(...)  ->  STT.transcribe(wav_bytes) -> str
      ↓
  is_usable(transcript)?    no -> "couldn't hear that clearly" and STOP,
      ↓ yes                        never reaches RoryCore
  print "heard: <transcript>"       <- shown before the answer, always
      ↓
  RoryCore.handle_text(transcript)  (same path text input always used)
```

**`rory/voice/audio.py`** — `to_wav_bytes`/`from_wav_bytes` are pure functions
(no device), so the WAV format itself is unit-tested without hardware.
`Recorder` opens the input stream at the device's *own* default sample rate
rather than assuming 16kHz — some ALSA hardware devices reject arbitrary
rates outright — then resamples what was captured down to 16kHz afterward.
Resampling is linear interpolation via numpy; there's no scipy in this
project's dependencies, and linear is more than sufficient for speech going
into an STT model.

**`rory/voice/tts.py`** — `SarvamTTS` (Bulbul), `LocalTTS` (espeak-ng via
`subprocess`, argv list, `shell=False` — same discipline as
`tools/desktop.py`), `FallbackTTS` (Sarvam → local on credit exhaustion, a
5xx, or a timeout — all three were observed for real against the live API
during development, not hypothetical), and `CachedTTS`
(`sha256(text + voice)` under `data/tts_cache/`, gitignored, regenerable).

**`rory/voice/stt.py`** — `SarvamSTT` (Saaras) and `is_usable(transcript)`,
the whole mechanism behind "empty or unusable transcripts must not reach the
LLM." Verified against a real 2-second recording of ambient silence during
development: Sarvam returned `""`, and `is_usable("")` correctly rejected it
before `RoryCore.handle_text` was ever called.

### Why the transcript is always shown before the answer

Mishearing a proper noun is the single most common failure mode in a voice
pipeline built on someone's personal notes — "Relay" becoming "railay,"
"CUSTOS" becoming "customs." There's no way to catch that from inside the
STT or LLM calls themselves; the cheapest fix that actually works is showing
the user what Rory *heard*, before showing what it answered, so a
mis-transcription is obvious immediately instead of surfacing as a
confusingly wrong answer three sentences later.

### Two turn_ids, not one, for a single voice turn

`RoryCore.handle_text` mints its own `turn_id` internally and cannot be
handed one from outside without changing `core.py` — which this feature was
explicitly not allowed to touch. So the `stt` trace event (which happens
*before* `handle_text` is ever called) gets its own `turn_id`, while
`llm_generate`/`tool_dispatch`/`tts` all share the one `RoryCore` generated.
A deliberate seam, not an oversight — correlating stt-to-answer requires
reading trace timestamps rather than a single shared id.
