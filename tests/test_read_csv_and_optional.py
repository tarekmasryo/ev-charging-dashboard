import io

import pandas as pd

from ev_charging_dashboard.data import load_optional, read_csv


def test_read_csv_supports_file_like() -> None:
    csv_text = "a,b\n1,2\n3,4\n"
    df = read_csv(io.StringIO(csv_text))
    assert df.shape == (2, 2)
    assert df["a"].tolist() == [1, 3]


def test_load_optional_returns_empty_on_missing_source(tmp_path) -> None:
    missing = tmp_path / "nope.csv"
    df = load_optional(str(missing))
    assert isinstance(df, pd.DataFrame)
    assert df.empty
