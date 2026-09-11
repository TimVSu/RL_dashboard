from __future__ import annotations

import pandas as pd
import pytest

from dashboard.data.schema import SchemaError, VEHICLE_COLUMNS, validate_columns


def test_validate_columns_passes_when_all_present() -> None:
    df = pd.DataFrame({c: [] for c in VEHICLE_COLUMNS})
    validate_columns(df, VEHICLE_COLUMNS, table="vehicle_states", run_id="r1")


def test_validate_columns_fails_clearly_when_missing() -> None:
    df = pd.DataFrame({"vehicle_id": [1], "soc": [0.5]})
    with pytest.raises(SchemaError) as exc_info:
        validate_columns(df, VEHICLE_COLUMNS, table="vehicle_states", run_id="r1")
    message = str(exc_info.value)
    assert "vehicle_states" in message
    assert "r1" in message
    assert "allocation" in message  # one of the missing columns is named
