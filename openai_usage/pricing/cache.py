"""On-disk caching for remotely fetched pricing catalogs.

Catalogs are fetched at runtime and cached under ``./.cache/openai-usage/``
with a one-day TTL, so a process pays the download cost at most once a day.
Every failure mode degrades instead of raising: a failed fetch falls back to a
stale cache entry, and an unwritable cache directory is simply skipped.
"""

import logging
import os
import pathlib
import threading
import time
import typing

import requests

logger = logging.getLogger(__name__)

CACHE_DIR_ENV = "OPENAI_USAGE_CACHE_DIR"
CACHE_TTL_ENV = "OPENAI_USAGE_CACHE_TTL"
OFFLINE_ENV = "OPENAI_USAGE_OFFLINE"

DEFAULT_CACHE_DIR = pathlib.Path(".cache") / "openai-usage"
DEFAULT_TTL_SECONDS = 24 * 60 * 60

# How long a degraded catalog is memoized before another fetch is attempted.
# Short enough that a process recovers from a transient failure on its own,
# long enough that a lasting outage does not retry on every single lookup.
DEGRADED_TTL_SECONDS = 60

# Separate connect and read budgets. Fetching is opportunistic and always has a
# fallback, so an unreachable network must fail fast rather than stall a caller
# that would have been served fine by the bundled snapshot.
FETCH_TIMEOUT_SECONDS = (5, 15)

_T = typing.TypeVar("_T")


def cache_dir() -> pathlib.Path:
    """Directory holding cached catalogs, overridable per environment."""
    override = os.environ.get(CACHE_DIR_ENV)
    return pathlib.Path(override) if override else DEFAULT_CACHE_DIR


def ttl_seconds() -> float:
    """How long a cached catalog stays fresh, in seconds."""
    raw = os.environ.get(CACHE_TTL_ENV)
    if not raw:
        return float(DEFAULT_TTL_SECONDS)
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            f"Ignoring invalid {CACHE_TTL_ENV}={raw!r}, "
            + f"using {DEFAULT_TTL_SECONDS}s"
        )
        return float(DEFAULT_TTL_SECONDS)


def is_offline() -> bool:
    """Whether network fetches are disabled entirely."""
    return os.environ.get(OFFLINE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def fetch_text(url: str, filename: str, *, force_refresh: bool = False) -> str | None:
    """Return the catalog at ``url``, going through the on-disk cache.

    Resolution order: a fresh cache entry, then a network fetch, then a stale
    cache entry. Returns ``None`` when every option is exhausted, leaving the
    caller to fall back to whatever it bundles.
    """
    path = cache_dir() / filename
    age = _age_seconds(path)

    if is_offline():
        if age is not None:
            logger.info(f"{OFFLINE_ENV} is set, using cached {filename}")
            return _read(path)
        logger.info(f"{OFFLINE_ENV} is set and {filename} is not cached")
        return None

    if not force_refresh and age is not None and age < ttl_seconds():
        return _read(path)

    try:
        response = requests.get(url, timeout=FETCH_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        if age is not None:
            logger.warning(f"Failed to fetch {url} ({e}), using stale {filename}")
            return _read(path)
        logger.warning(f"Failed to fetch {url}: {e}")
        return None

    _write(path, response.text)
    return response.text


class TTLCache(typing.Generic[_T]):
    """In-process memo for a parsed catalog, expiring with the on-disk TTL.

    Without this the parsed catalog would either be re-validated on every
    lookup, or pinned for the lifetime of the process and never pick up a
    refreshed download.
    """

    def __init__(self) -> None:
        self._value: _T | None = None
        self._stored_at: float = 0.0
        self._ttl: float | None = None

    def get(self) -> _T | None:
        if self._value is None:
            return None
        ttl = ttl_seconds() if self._ttl is None else self._ttl
        if time.monotonic() - self._stored_at >= ttl:
            self._value = None
            return None
        return self._value

    def set(self, value: _T, *, ttl: float | None = None) -> _T:
        """Memoize ``value``, optionally for less than the full TTL.

        A degraded result is stored with a short ``ttl`` so one transient
        failure near startup does not pin a process to it for a whole day.
        """
        self._value = value
        self._stored_at = time.monotonic()
        self._ttl = ttl
        return value

    def clear(self) -> None:
        self._value = None
        self._stored_at = 0.0
        self._ttl = None


def _read(path: pathlib.Path) -> str | None:
    """Read a cache file, treating any filesystem error as a miss."""
    try:
        return path.read_text()
    except OSError as e:
        logger.debug(f"Could not read cache file {path}: {e}")
        return None


def _age_seconds(path: pathlib.Path) -> float | None:
    """Seconds since the cache file was written, or ``None`` if absent."""
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return None


def _write(path: pathlib.Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically, ignoring filesystem failures.

    Caching is an optimization, so a read-only or full filesystem must never
    turn into an error for the caller.
    """
    # Thread id as well as pid: two threads racing to populate the same file
    # would otherwise share a temp path, and the loser's replace would fail.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(text)
        os.replace(tmp, path)
    except OSError as e:
        logger.warning(f"Could not write cache file {path}: {e}")
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
