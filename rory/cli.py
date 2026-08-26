"""Text REPL — the primary development interface for Rory's core.

Runs the same RoryCore.handle_text loop that voice and GUI adapters will use,
with no audio devices or Qt event loop involved. Type 'exit' or Ctrl-D to quit.
"""
from __future__ import annotations

from rory.config import settings
from rory.core import RoryCore
from rory.llm import GeminiLLM


def main() -> None:
    llm = GeminiLLM(api_key=settings.gemini_api_key, model=settings.gemini_model)
    core = RoryCore(llm)

    print("Rory CLI. Type 'exit' to quit.")
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            print()
            break

        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            break

        reply = core.handle_text(text)
        if reply.error:
            print(f"[error] {reply.error}")
        else:
            print(reply.text)


if __name__ == "__main__":
    main()
