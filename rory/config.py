"""Settings, validated eagerly so a missing key fails at import, not on first use."""
from __future__ import annotations

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str
    gemini_model: str = "gemini-flash-latest"
    rory_trace_verbose: bool = False


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
