"""Settings, validated eagerly so a missing key fails at import, not on first use."""
from __future__ import annotations

from typing import Literal

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str
    gemini_model: str = "gemini-flash-latest"
    rory_trace_verbose: bool = False

    # Voice. sarvam_api_key is optional at the Settings level (unlike Gemini's
    # key) because tts_engine defaults to "local" — a fresh checkout can run
    # the CLI's voice output without ever needing a Sarvam account. Something
    # that actually tries to use Sarvam (SarvamTTS/SarvamSTT) fails loudly at
    # construction if the key is missing, matching this project's "fail loud,
    # not on first use" rule at the point where the dependency is real.
    sarvam_api_key: str | None = None
    tts_engine: Literal["sarvam", "local"] = "local"
    tts_voice: str = "anushka"        # Sarvam Bulbul speaker name
    tts_local_voice: str = "en-us+f3"  # espeak-ng voice id


def _load() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        missing = [err["loc"][0] for err in exc.errors() if err["type"] == "missing"]
        raise SystemExit(
            "Rory is missing required configuration: "
            f"{', '.join(str(m) for m in missing)}.\n"
            "Copy .env.example to .env and fill in the values."
        ) from exc


settings = _load()
