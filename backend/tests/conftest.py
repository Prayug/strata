import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# tests never touch the operating system's keychain: credentials go to a file in the temporary data directory
os.environ["STRATA_SECRET_STORE"] = "file"
os.environ.pop("STRATA_API_TOKEN", None)
os.environ["STRATA_OFFLINE"] = "1"  # the suite never downloads anything; sources are exercised through a mocked transport
# every test has its own temporary data directory: the one-machine-at-a-time lease (strata.core.instance) is switched
# off here so that hundreds of directories do not each get a heartbeat thread. test_v6_instance.py switches it on.
os.environ["STRATA_INSTANCE_GUARD"] = "0"
os.environ.pop("STRATA_INSTANCE_KERNEL", None)


@pytest.fixture(autouse=True)
def _clean_secret_cache():
    from strata.api import settings
    settings._cache.clear()
    yield
    settings._cache.clear()
