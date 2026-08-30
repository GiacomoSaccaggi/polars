"""Integration tests for Trino via ConnectorX.

These tests require a running Trino instance (e.g. Docker: trinodb/trino:latest).
They are skipped automatically if Trino is not reachable on localhost:8080.

Run manually:
    docker run -d --name trino -p 8080:8080 trinodb/trino:latest
    # wait ~15s for startup
    pytest tests/unit/io/database/test_trino.py -v
"""

from __future__ import annotations

import urllib.request

import pytest

import polars as pl
from polars.testing import assert_frame_equal

TRINO_URL = "trino://test@localhost:8080/tpch"


def trino_is_running() -> bool:
    """Check if Trino is reachable."""
    try:
        urllib.request.urlopen("http://localhost:8080/v1/info", timeout=2)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not trino_is_running(),
    reason="Trino not running on localhost:8080 (start with: docker run -d -p 8080:8080 trinodb/trino:latest)",
)


class TestTrinoBasic:
    """Basic Trino connectivity via ConnectorX."""

    def test_simple_query(self) -> None:
        df = pl.read_database_uri(
            "SELECT 1 AS x, 'hello' AS y",
            TRINO_URL,
            engine="connectorx",
        )
        assert df.shape == (1, 2)
        assert df["x"][0] == 1
        assert df["y"][0] == "hello"

    def test_tpch_tiny(self) -> None:
        df = pl.read_database_uri(
            "SELECT * FROM tiny.nation ORDER BY nationkey LIMIT 5",
            TRINO_URL,
            engine="connectorx",
        )
        assert df.shape == (5, 4)
        assert "nationkey" in df.columns
        assert "name" in df.columns

    def test_empty_result(self) -> None:
        df = pl.read_database_uri(
            "SELECT * FROM tiny.nation WHERE nationkey < 0",
            TRINO_URL,
            engine="connectorx",
        )
        assert df.shape[0] == 0
        assert "nationkey" in df.columns


class TestTrinoPartition:
    """Parallel query execution with partition_on."""

    def test_partition_on_integer(self) -> None:
        query = "SELECT * FROM tiny.nation"
        df_single = pl.read_database_uri(query, TRINO_URL, engine="connectorx")
        df_partitioned = pl.read_database_uri(
            query,
            TRINO_URL,
            engine="connectorx",
            partition_on="nationkey",
            partition_num=4,
        )
        # Same data regardless of partitioning
        assert df_single.shape == df_partitioned.shape
        assert_frame_equal(
            df_single.sort("nationkey"),
            df_partitioned.sort("nationkey"),
        )

    def test_partition_with_range(self) -> None:
        df = pl.read_database_uri(
            "SELECT * FROM tiny.nation",
            TRINO_URL,
            engine="connectorx",
            partition_on="nationkey",
            partition_range=(0, 25),
            partition_num=5,
        )
        assert df.shape[0] == 25  # tpch tiny.nation has 25 rows


class TestTrinoTypes:
    """Type mapping from Trino to Polars."""

    def test_integer_types(self) -> None:
        df = pl.read_database_uri(
            "SELECT CAST(1 AS TINYINT) AS ti, CAST(2 AS SMALLINT) AS si, "
            "CAST(3 AS INTEGER) AS i, CAST(4 AS BIGINT) AS bi",
            TRINO_URL,
            engine="connectorx",
        )
        assert df["ti"][0] == 1
        assert df["si"][0] == 2
        assert df["i"][0] == 3
        assert df["bi"][0] == 4

    def test_float_types(self) -> None:
        df = pl.read_database_uri(
            "SELECT CAST(1.5 AS REAL) AS r, CAST(2.5 AS DOUBLE) AS d",
            TRINO_URL,
            engine="connectorx",
        )
        assert abs(df["r"][0] - 1.5) < 0.01
        assert abs(df["d"][0] - 2.5) < 0.001

    def test_string_types(self) -> None:
        df = pl.read_database_uri(
            "SELECT CAST('hello' AS VARCHAR) AS v",
            TRINO_URL,
            engine="connectorx",
        )
        assert df["v"][0] == "hello"

    def test_boolean_type(self) -> None:
        df = pl.read_database_uri(
            "SELECT true AS t, false AS f",
            TRINO_URL,
            engine="connectorx",
        )
        assert df["t"][0] is True
        assert df["f"][0] is False

    def test_date_type(self) -> None:
        df = pl.read_database_uri(
            "SELECT DATE '2026-08-18' AS d",
            TRINO_URL,
            engine="connectorx",
        )
        assert str(df["d"][0]) == "2026-08-18"
