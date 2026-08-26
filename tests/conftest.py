import os

# Must happen before anything imports rory.config, since Settings validates at
# import time. A real .env is never read here because we chdir tests below.
os.environ.setdefault("GEMINI_API_KEY", "test-key")

import pytest


@pytest.fixture(autouse=True)
def isolate_cwd(tmp_path, monkeypatch):
    """Keep trace.py's logs/ writes out of the real repo during tests."""
    monkeypatch.chdir(tmp_path)
