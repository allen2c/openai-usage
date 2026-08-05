import os
import pathlib
import time

import pytest
import requests

from openai_usage.pricing import cache

URL = "https://example.invalid/catalog.json"
FILENAME = "catalog.json"


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


@pytest.fixture
def cache_env(tmp_path, monkeypatch):
    """Point the cache at a scratch directory with no environment overrides."""
    monkeypatch.setenv(cache.CACHE_DIR_ENV, str(tmp_path))
    monkeypatch.delenv(cache.CACHE_TTL_ENV, raising=False)
    monkeypatch.delenv(cache.OFFLINE_ENV, raising=False)
    return tmp_path


@pytest.fixture
def fetches(monkeypatch):
    """Record every outbound fetch and serve a canned body."""
    calls: list[str] = []
    body = {"text": '{"a": 1}'}

    def fake_get(url, timeout=None):
        calls.append(url)
        return _Response(body["text"])

    monkeypatch.setattr(requests, "get", fake_get)
    return calls, body


def test_fetch_writes_cache_then_reuses_it(cache_env, fetches):
    calls, _ = fetches

    assert cache.fetch_text(URL, FILENAME) == '{"a": 1}'
    assert (cache_env / FILENAME).read_text() == '{"a": 1}'
    assert len(calls) == 1

    # Within the TTL the file is served from disk, without a second request.
    assert cache.fetch_text(URL, FILENAME) == '{"a": 1}'
    assert len(calls) == 1


def test_fetch_refetches_once_stale(cache_env, fetches, monkeypatch):
    calls, body = fetches
    cache.fetch_text(URL, FILENAME)

    monkeypatch.setenv(cache.CACHE_TTL_ENV, "0")
    body["text"] = '{"a": 2}'

    assert cache.fetch_text(URL, FILENAME) == '{"a": 2}'
    assert len(calls) == 2


def test_force_refresh_bypasses_a_fresh_cache(cache_env, fetches):
    calls, body = fetches
    cache.fetch_text(URL, FILENAME)

    body["text"] = '{"a": 3}'
    assert cache.fetch_text(URL, FILENAME, force_refresh=True) == '{"a": 3}'
    assert len(calls) == 2


def test_failed_fetch_falls_back_to_a_stale_entry(cache_env, fetches, monkeypatch):
    cache.fetch_text(URL, FILENAME)
    monkeypatch.setenv(cache.CACHE_TTL_ENV, "0")

    def boom(url, timeout=None):
        raise requests.exceptions.ConnectionError("no network")

    monkeypatch.setattr(requests, "get", boom)

    assert cache.fetch_text(URL, FILENAME) == '{"a": 1}'


def test_failed_fetch_without_a_cache_returns_none(cache_env, monkeypatch):
    def boom(url, timeout=None):
        raise requests.exceptions.ConnectionError("no network")

    monkeypatch.setattr(requests, "get", boom)

    assert cache.fetch_text(URL, FILENAME) is None


def test_offline_never_fetches(cache_env, fetches, monkeypatch):
    calls, _ = fetches
    cache.fetch_text(URL, FILENAME)
    monkeypatch.setenv(cache.OFFLINE_ENV, "1")
    monkeypatch.setenv(cache.CACHE_TTL_ENV, "0")

    # Stale, but offline means the cache is used rather than refreshed.
    assert cache.fetch_text(URL, FILENAME) == '{"a": 1}'
    assert len(calls) == 1


def test_offline_without_a_cache_returns_none(cache_env, monkeypatch):
    monkeypatch.setenv(cache.OFFLINE_ENV, "true")
    assert cache.fetch_text(URL, FILENAME) is None


def test_unwritable_cache_dir_still_returns_the_body(tmp_path, monkeypatch, fetches):
    unwritable = tmp_path / "readonly"
    unwritable.mkdir()
    monkeypatch.setenv(cache.CACHE_DIR_ENV, str(unwritable / "nested"))
    monkeypatch.delenv(cache.OFFLINE_ENV, raising=False)
    unwritable.chmod(0o500)
    try:
        assert cache.fetch_text(URL, FILENAME) == '{"a": 1}'
    finally:
        unwritable.chmod(0o700)


def test_write_leaves_no_temporary_file_behind(cache_env, fetches):
    cache.fetch_text(URL, FILENAME)
    assert [p.name for p in cache_env.iterdir()] == [FILENAME]


def test_cache_dir_defaults_below_the_working_directory(monkeypatch):
    monkeypatch.delenv(cache.CACHE_DIR_ENV, raising=False)
    assert cache.cache_dir() == pathlib.Path(".cache") / "openai-usage"


def test_cache_dir_honours_the_environment(monkeypatch):
    monkeypatch.setenv(cache.CACHE_DIR_ENV, "/somewhere/else")
    assert cache.cache_dir() == pathlib.Path("/somewhere/else")


def test_ttl_defaults_to_one_day(monkeypatch):
    monkeypatch.delenv(cache.CACHE_TTL_ENV, raising=False)
    assert cache.ttl_seconds() == 24 * 60 * 60


def test_invalid_ttl_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv(cache.CACHE_TTL_ENV, "not-a-number")
    assert cache.ttl_seconds() == 24 * 60 * 60


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_offline_flag_accepts_common_spellings(monkeypatch, value):
    monkeypatch.setenv(cache.OFFLINE_ENV, value)
    assert cache.is_offline() is True


@pytest.mark.parametrize("value", ["0", "false", "no", ""])
def test_offline_flag_is_off_otherwise(monkeypatch, value):
    monkeypatch.setenv(cache.OFFLINE_ENV, value)
    assert cache.is_offline() is False


def test_ttl_cache_expires(monkeypatch):
    monkeypatch.delenv(cache.CACHE_TTL_ENV, raising=False)
    ttl_cache: cache.TTLCache[str] = cache.TTLCache()

    assert ttl_cache.get() is None
    ttl_cache.set("value")
    assert ttl_cache.get() == "value"

    now = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: now + 24 * 60 * 60 + 1)
    assert ttl_cache.get() is None


def test_ttl_cache_clear():
    ttl_cache: cache.TTLCache[str] = cache.TTLCache()
    ttl_cache.set("value")
    ttl_cache.clear()
    assert ttl_cache.get() is None


def test_fetch_uses_a_bounded_timeout(cache_env, monkeypatch):
    """An unreachable network must fail fast; the snapshot fallback is fine."""
    seen: list[object] = []

    def record(url, timeout=None):
        seen.append(timeout)
        return _Response("{}")

    monkeypatch.setattr(requests, "get", record)
    cache.fetch_text(URL, FILENAME)

    connect, read = seen[0]  # type: ignore[misc]
    assert connect <= 5
    assert read <= 30


def test_env_var_names_are_stable():
    """These are documented, so renaming one is a breaking change."""
    assert cache.CACHE_DIR_ENV == "OPENAI_USAGE_CACHE_DIR"
    assert cache.CACHE_TTL_ENV == "OPENAI_USAGE_CACHE_TTL"
    assert cache.OFFLINE_ENV == "OPENAI_USAGE_OFFLINE"
    assert os.environ.get("PYTEST_IS_RUNNING") == "true"
