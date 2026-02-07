import io

import pytest

from ev_charging_dashboard.data import load_main


def test_load_main_standardizes_schema_and_derives_fields() -> None:
    csv_text = (
        "country,city,station_id,latitude,longitude,ports,power_kw\n"
        "US,New York,1,40.7128,-74.0060,4,200\n"
        "US,New York,2,40.7128,-74.0060,2,60\n"
        "CA,Toronto,3,43.6532,-79.3832,3,150\n"
    )
    df = load_main(io.StringIO(csv_text))

    expected_cols = {
        "country_code",
        "city",
        "id",
        "ports",
        "power_kw",
        "kw_per_port",
        "is_fast_dc",
        "power_class",
    }
    assert expected_cols.issubset(df.columns)

    assert df["country_code"].tolist() == ["US", "US", "CA"]
    assert df["ports"].tolist() == [4, 2, 3]

    kw_per_port = df["kw_per_port"].round(2).tolist()
    assert kw_per_port == [50.00, 30.00, 50.00]

    assert df["is_fast_dc"].tolist() == [True, False, True]
    assert set(df["power_class"].tolist()) == {"DC (Slow)", "DC (Fast)"}


def test_load_main_raises_on_missing_required_columns() -> None:
    csv_text = "country,city,ports\nUS,New York,3\n"
    with pytest.raises(ValueError):
        load_main(io.StringIO(csv_text))
