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


@pytest.fixture(autouse=True, scope="session")
def _known_time_zone():
    """The tests written before V10 are those of a user whose time zone is known (the setting they were written under: a time
    with no zone is read in the zone of Settings, and said). V10 asks about such a time until the zone is known; the V10
    tests switch this off where that question is what they test."""
    from strata.api import settings
    old = settings.DEFAULTS.get("ui.timezone_confirmed")
    settings.DEFAULTS["ui.timezone_confirmed"] = True
    # the same for V10's go-ahead before a reading with several parts runs: the earlier tests post such a request and look
    # at what ran. test_v10.py switches it on where the waiting is what is tested.
    old_c = settings.DEFAULTS.get("ai.confirm_complex")
    settings.DEFAULTS["ai.confirm_complex"] = False
    yield
    settings.DEFAULTS["ui.timezone_confirmed"] = old
    settings.DEFAULTS["ai.confirm_complex"] = old_c
