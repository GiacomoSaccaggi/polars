"""Tests for the connection_options parameter in read_database_uri."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import pytest

import polars as pl


@pytest.fixture
def _mock_connectorx(monkeypatch: pytest.MonkeyPatch):
    """Inject a fake connectorx module that captures the connection URI."""
    fake_cx = ModuleType("connectorx")
    fake_cx.__version__ = "0.4.6"  # type: ignore[attr-defined]

    def _read_sql(**kwargs: Any):
        raise ConnectionError(kwargs["conn"])

    fake_cx.read_sql = _read_sql  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "connectorx", fake_cx)


@pytest.mark.usefixtures("_mock_connectorx")
class TestConnectionOptions:
    """Verify that connection_options dict gets serialized into the URI."""

    def _get_uri(self, uri: str, **kwargs: Any) -> str:
        """Run read_database_uri and extract the URI from the raised error."""
        with pytest.raises(ConnectionError) as exc_info:
            pl.read_database_uri("SELECT 1", uri, engine="connectorx", **kwargs)
        return str(exc_info.value)

    def test_appends_to_uri(self) -> None:
        result = self._get_uri(
            "trino://user@host:8080/catalog",
            connection_options={"schema": "analytics", "source": "test"},
        )
        assert "schema=analytics" in result
        assert "source=test" in result

    def test_preserves_existing_params(self) -> None:
        result = self._get_uri(
            "trino://user@host:8080/catalog?verify=false",
            connection_options={"source": "test"},
        )
        assert "verify=false" in result
        assert "source=test" in result

    def test_url_encodes_values(self) -> None:
        result = self._get_uri(
            "trino://user@host:8080/catalog",
            connection_options={"client_info": "my app v2"},
        )
        assert "my%20app%20v2" in result

    def test_none_is_noop(self) -> None:
        result = self._get_uri(
            "trino://user@host:8080/catalog",
            connection_options=None,
        )
        assert result == "trino://user@host:8080/catalog"

    def test_rejected_for_adbc(self) -> None:
        with pytest.raises(ValueError, match="adbc.*does not support.*connection_options"):
            pl.read_database_uri(
                "SELECT 1",
                "sqlite:///:memory:",
                engine="adbc",
                connection_options={"key": "value"},
            )
