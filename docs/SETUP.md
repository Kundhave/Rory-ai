# Setup and Running

Operational detail, split out of README.md to keep that document focused on
architecture and engineering decisions.

Tested on Arch Linux, Hyprland 0.56, Wayland, `omarchy` session.

## Install

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

Rory fails loudly at startup if `GEMINI_API_KEY` is missing. It will not run half
configured.

### Model choice matters on the free tier

`gemini-flash-latest` resolves to a model capped at 20 requests **per day** on
the free tier, which is easy to exhaust while developing.
`gemini-flash-lite-latest` has a per-minute cap instead, which recovers in under
a minute rather than blocking for the rest of the day. That is the default in
`.env.example` for this reason.

## Building the knowledge index

```bash
python -m rory.rag.ingest
```

Builds `data/index.npz` and `data/chunks.json` from `knowledge/*.md`. The first
run downloads the local embedding model (`BAAI/bge-small-en-v1.5`), so expect a
delay. Re-run it any time the notes change.

See [../knowledge/README.md](../knowledge/README.md) for the file format, which
matters a lot for retrieval quality, and a data sensitivity note. Short version:
use real markdown headings, one topic per heading. The chunker is header aware,
so headings are the primary retrieval signal, not decoration.

## CLI

```bash
python -m rory.cli
```

Type to chat. Press Enter on an empty line to start recording, Enter again to
stop. `exit` or Ctrl-D to quit.

## Desktop widget

```bash
./run.sh
```

Click the status button to start listening, click again to stop and process.
The button is the state indicator:

| state | appearance |
|---|---|
| idle | pink, mic glyph |
| listening | red |
| processing / speaking | rotating arc |
| error | `!`, real error in the tooltip |

Right-click for Quit.

To install into the app launcher, copy or symlink `rory.desktop` into
`~/.local/share/applications/`. Its `Exec=` path is hardcoded to this checkout,
so edit it if Rory lives somewhere else.

For autostart, add to `~/.config/hypr/autostart.conf`:

```
exec-once = uwsm-app -- /path/to/Rory-ai/run.sh
```

`exec-once` only fires at Hyprland startup, not on `hyprctl reload`. After adding
it, log out and back in, or just run `./run.sh &` for the current session.

## Voice configuration

`TTS_ENGINE=local` (the default) uses espeak-ng. It is free, needs no network,
and never touches Sarvam credits, which makes it the right default for
development.

```bash
sudo pacman -S espeak-ng        # Arch
# or: sudo apt install espeak-ng
```

`TTS_ENGINE=sarvam` uses the real neural voice. Speech **input** always needs
`SARVAM_API_KEY` regardless of `TTS_ENGINE`, since there is no local STT
fallback.

Stay on `bulbul:v3` (the default). See ADR-014 in
[DECISIONS.md](DECISIONS.md) and the problems section of the README: `bulbul:v2`
succeeded on only 2 of 5 measured calls, hanging ~30s per failure.

Speaker names are model specific, so v2 names like `manisha` do not exist on v3.

- v3 female: `shreya`, `priya`, `ritu`, `kavya`, `ishita`
- v3 male: `shubh`, `aditya`, `rahul`, `varun`
- local: `espeak-ng --voices=en` lists options for `TTS_LOCAL_VOICE`

If speech ever sounds robotic, check which engine actually spoke:

```bash
grep '"stage": "tts"' logs/trace.jsonl | tail -5
```

`"engine": "primary"` is the real Sarvam voice. `"fallback"` means espeak-ng
because Sarvam failed.

## Wayland and Hyprland notes

Under Wayland a client cannot position or pin its own window. That is compositor
policy, so it lives in `~/.config/hypr/hyprland.conf`:

```
windowrule = match:title ^Rory$, float 1
windowrule = match:title ^Rory$, pin 1
windowrule = match:title ^Rory$, move 1356 684
```

Each action needs an explicit value. `float` alone is rejected; `float 1` works.
`windowrulev2` is deprecated in 0.56.

**Coordinates are logical pixels, physical divided by scale.** This display is
1920x1080 at scale 1.25, so the usable space is 1536x864. A move computed against
the physical resolution puts the widget off screen, which looks exactly like the
widget not running. To recompute after a monitor or scale change:

```bash
hyprctl monitors -j | python3 -c "import json,sys; m=json.load(sys.stdin)[0]; print(m['width']/m['scale'], m['height']/m['scale'])"
# then: x = logical_width - widget_width - margin, same for y
```

There is no built-in global hotkey, because hotkey libraries mostly cannot hook
keyboard input under Wayland's security model. Instead Rory listens on a Unix
socket at `$XDG_RUNTIME_DIR/rory.sock` and lets the compositor own the keybinding.
Add to `~/.config/hypr/bindings.conf`:

```
bind = SUPER, R, exec, echo -n "toggle" | socat - UNIX-SENDTO:$XDG_RUNTIME_DIR/rory.sock
```

Needs `socat`. Anything that can write a datagram to a Unix socket works.

## Evaluation harness

```bash
python -m rory.eval             # RAG: search_notes retrieves on demand
python -m rory.eval --stuff     # whole knowledge base pasted into the prompt
```

Both hit the real Gemini API and cost quota. Useful flags:

- `--limit N` runs only the first N cases
- `--delay N` seconds between LLM calls, for free-tier per-minute caps
- `--model` overrides the model for one run

Results and interpretation are in the README.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

93 tests, all offline and free. No API keys needed. Tests marked `live` hit real
APIs and are excluded by default (`pytest -m live` to include them).
