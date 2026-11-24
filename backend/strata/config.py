"""Runtime settings, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ENGINE_VERSION = "1.0.0"


def _load_dotenv() -> None:
    """Minimal .env loader so the app works without python-dotenv."""
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"):
        if candidate.is_file():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            break


_load_dotenv()


REPO_ROOT = Path(__file__).resolve().parents[2]


def _data_dir() -> Path:
    """STRATA_DATA_DIR; relative paths are resolved against the repository root so the
    API, CLI and workers agree regardless of the current working directory."""
    raw = os.environ.get("STRATA_DATA_DIR") or "data"
    p = Path(raw).expanduser()
    return (p if p.is_absolute() else REPO_ROOT / p).resolve()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=_data_dir)
    max_workers: int = int(os.environ.get("STRATA_MAX_WORKERS") or max(1, min((os.cpu_count() or 2) - 2, 8)))
    risk_free_rate: float = float(os.environ.get("STRATA_RISK_FREE_RATE", "0.04"))
    cors_origins: tuple[str, ...] = tuple(o for o in os.environ.get("STRATA_CORS_ORIGINS", "http://localhost:5199,http://127.0.0.1:5199,http://localhost:5173").split(",") if o)
    thetadata_url: str = os.environ.get("THETADATA_BASE_URL", "http://127.0.0.1:25510")
    polygon_api_key: str = os.environ.get("POLYGON_API_KEY", "")
    polygon_base_url: str = os.environ.get("POLYGON_BASE_URL", "https://api.polygon.io")
    databento_api_key: str = os.environ.get("DATABENTO_API_KEY", "")
    orats_api_key: str = os.environ.get("ORATS_API_KEY", "")
    orats_base_url: str = os.environ.get("ORATS_BASE_URL", "https://api.orats.io/datav2")

    @property
    def market_dir(self) -> Path:
        return self.data_dir / "market"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def override_settings(**kw) -> Settings:
    """Used by tests and the CLI to point the app at another data directory."""
    global _settings
    base = get_settings()
    vals = {f: getattr(base, f) for f in base.__dataclass_fields__}
    vals.update(kw)
    if "data_dir" in kw:
        vals["data_dir"] = Path(kw["data_dir"]).expanduser().resolve()
    _settings = Settings(**vals)
    return _settings
