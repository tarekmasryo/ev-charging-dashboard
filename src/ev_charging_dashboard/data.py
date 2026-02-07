"""Data loading and normalization."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

DATA_REPO_URL = "https://github.com/tarekmasryo/Global-EV-Charging-Stations"
RAW_DEFAULT_URL = "https://raw.githubusercontent.com/tarekmasryo/Global-EV-Charging-Stations/main/data/charging_station.csv"
KAGGLE_DEFAULT_PATH = "/kaggle/input/global-ev-charging-stations/charging_station.csv"

REQUIRED_COLS = {
    "id",
    "country_code",
    "city",
    "latitude",
    "longitude",
    "ports",
    "power_kw",
}

POP_PATH = "world_population.csv"
REG_PATH = "country_region.csv"


def normalize_url(url: str) -> str:
    if not isinstance(url, str):
        return url

    u = url.split("?", 1)[0]
    if "github.com" in u and "/blob/" in u:
        u = u.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return u


def read_csv(source: Any) -> pd.DataFrame:
    if not isinstance(source, str):
        return pd.read_csv(source)

    src = normalize_url(source)
    return pd.read_csv(src)


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases: dict[str, list[str]] = {
        "id": ["id", "station_id", "stationid", "location_id"],
        "name": ["name", "station_name", "location_name"],
        "country_code": ["country_code", "country", "iso2", "country_iso2", "cc"],
        "city": ["city", "municipality", "town"],
        "latitude": ["latitude", "lat"],
        "longitude": ["longitude", "lon", "lng"],
        "ports": ["ports", "num_ports", "connectors", "num_connectors"],
        "power_kw": ["power_kw", "power", "max_power_kw", "max_kw", "powerkW"],
        "power_class": ["power_class", "charger_class", "class"],
        "is_fast_dc": ["is_fast_dc", "fast_dc", "is_fast", "dc_fast"],
    }

    lower_map = {c.lower(): c for c in df.columns}
    rename: dict[str, str] = {}

    for canonical, candidates in aliases.items():
        if canonical in df.columns:
            continue
        for cand in candidates:
            key = cand.lower()
            if key in lower_map:
                rename[lower_map[key]] = canonical
                break

    if not rename:
        return df

    return df.rename(columns=rename)


def _derive_power_class(power_kw: pd.Series) -> pd.Series:
    s = pd.to_numeric(power_kw, errors="coerce").fillna(0.0)
    s = s.clip(lower=0.0)
    bins = [0.0, 50.0, 150.0, 350.0, 1e12]
    labels = ["AC", "DC (Slow)", "DC (Fast)", "Ultra-fast"]
    out = pd.cut(s, bins=bins, labels=labels, right=True, include_lowest=False)
    return out.astype("object").where(s > 0.0, other="Unknown")


def _derive_is_fast_dc(power_kw: pd.Series) -> pd.Series:
    s = pd.to_numeric(power_kw, errors="coerce").fillna(0.0)
    return (s >= 150.0).astype(bool)


def load_main(source: Any) -> pd.DataFrame:
    df = read_csv(source)
    df = _canonicalize_columns(df)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()

    out["id"] = out["id"].astype(str)
    out["name"] = out.get("name", pd.Series(index=out.index, dtype="object")).astype("object")
    out["country_code"] = out["country_code"].astype(str).str.upper()
    out["city"] = out["city"].astype(str).replace({"nan": "Unknown"}).fillna("Unknown")

    for col in ["latitude", "longitude", "ports", "power_kw"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    out["ports"] = out["ports"].fillna(1).clip(lower=1).astype(int)
    out["power_kw"] = out["power_kw"].fillna(0.0).clip(lower=0.0)

    if "is_fast_dc" not in out.columns:
        out["is_fast_dc"] = _derive_is_fast_dc(out["power_kw"])
    else:
        out["is_fast_dc"] = out["is_fast_dc"].astype(bool)

    if "power_class" not in out.columns:
        out["power_class"] = _derive_power_class(out["power_kw"])
    else:
        out["power_class"] = out["power_class"].fillna(_derive_power_class(out["power_kw"]))

    out["kw_per_port"] = out["power_kw"] / out["ports"]
    return out


def load_optional(path: str) -> pd.DataFrame:
    try:
        if not isinstance(path, str) or not os.path.exists(path):
            return pd.DataFrame()
        return read_csv(path)
    except Exception:
        return pd.DataFrame()
