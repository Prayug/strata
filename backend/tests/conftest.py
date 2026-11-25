import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# tests never touch the operating system's keychain: credentials go to a file in the temporary data directory
os.environ["STRATA_SECRET_STORE"] = "file"
os.environ.pop("STRATA_API_TOKEN", None)
os.environ["STRATA_OFFLINE"] = "1"  # the suite never downloads anything; sources are exercised through a mocked transport


@pytest.fixture(autouse=True)
def _clean_secret_cache():
    from strata.api import settings
    settings._cache.clear()
    yield
    settings._cache.clear()
