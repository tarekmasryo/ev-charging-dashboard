import numpy as np
import pandas as pd
import pytest

from ev_charging_dashboard.data import REQUIRED_COLS, load_main


def _make_df(**overrides):
    row = {
        "id": 1,
        "name": "Station A",
        "city": "cairo",
        "country_code": "eg",
        "latitude": 30.0444,
        "longitude": 31.2357,
        "ports": 2,
        "power_kw": 50.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_missing_columns_raises(tmp_path) -> None:
    df = _make_df().drop(columns=["power_kw"])
    p = tmp_path / "bad.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError) as e:
        load_main(str(p))
    assert "missing required columns" in str(e.value).lower()


def test_cleaning_and_types(tmp_path) -> None:
    df = _make_df(ports="-3", power_kw="120", country_code="us", city=None)
    p = tmp_path / "ok.csv"
    df.to_csv(p, index=False)

    out = load_main(str(p))
    assert set(REQUIRED_COLS).issubset(out.columns)
    assert out.loc[0, "ports"] == 1
    assert out.loc[0, "power_kw"] == 120.0
    assert out.loc[0, "country_code"] == "US"
    assert out.loc[0, "city"] == "Unknown"
    assert bool(out.loc[0, "is_fast_dc"]) in {True, False}
    assert "kw_per_port" in out.columns
    assert np.isfinite(out.loc[0, "kw_per_port"]) or np.isnan(out.loc[0, "kw_per_port"])


def test_drops_missing_lat_lon(tmp_path) -> None:
    df = pd.concat([_make_df(latitude=np.nan), _make_df(id=2)], ignore_index=True)
    p = tmp_path / "mix.csv"
    df.to_csv(p, index=False)
    out = load_main(str(p))
    assert len(out) == 1
    assert int(out.iloc[0]["id"]) == 2
