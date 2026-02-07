"""Analytics helpers for the EV charging dashboard.

This module is intentionally UI-agnostic: it transforms dataframes into
computed signals that the Streamlit app can display.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import zscore


def _integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))
    yy = np.asarray(y, dtype=float)
    xx = np.asarray(x, dtype=float)
    if yy.size < 2:
        return 0.0
    return float(np.sum((yy[:-1] + yy[1:]) * np.diff(xx) * 0.5))


@dataclass(frozen=True)
class LorenzResult:
    x: np.ndarray
    y: np.ndarray
    gini: float

    def __iter__(self):
        return iter((self.x, self.y, self.gini))


def aggregate_by_country(stations: pd.DataFrame) -> pd.DataFrame:
    required = {"country_code", "id", "ports", "power_kw", "is_fast_dc", "kw_per_port"}
    missing = required.difference(stations.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    d = stations.copy()
    d["country_code"] = d["country_code"].astype(str).str.upper()

    g = (
        d.groupby("country_code", dropna=False)
        .agg(
            stations=("id", "count"),
            total_ports=("ports", "sum"),
            fast_share=("is_fast_dc", "mean"),
            avg_power_kw=("power_kw", "mean"),
            avg_kw_per_port=("kw_per_port", "mean"),
        )
        .reset_index()
    )

    g["fast_share"] = (pd.to_numeric(g["fast_share"], errors="coerce") * 100.0).round(1)
    denom = pd.to_numeric(g["stations"], errors="coerce").replace(0, np.nan)
    g["ports_per_station"] = (pd.to_numeric(g["total_ports"], errors="coerce") / denom).round(2)

    return g.sort_values("total_ports", ascending=False).reset_index(drop=True)


def add_population(country_df: pd.DataFrame, pop_df: pd.DataFrame) -> pd.DataFrame:
    out = country_df.copy()

    if pop_df is None or pop_df.empty:
        out["ports_per_100k"] = np.nan
        return out

    if "country_code" not in pop_df.columns:
        raise ValueError("Population table missing column: country_code")

    pop = pop_df.copy()
    pop["country_code"] = pop["country_code"].astype(str).str.upper()

    if "population" in pop.columns:
        pop["population"] = pd.to_numeric(pop["population"], errors="coerce")
    elif "population_m" in pop.columns:
        pop["population_m"] = pd.to_numeric(pop["population_m"], errors="coerce")
        pop["population"] = pop["population_m"] * 1_000_000.0
    else:
        raise ValueError("Population table missing column: population or population_m")

    left = out.copy()
    left["country_code"] = left["country_code"].astype(str).str.upper()

    merged = left.merge(pop[["country_code", "population"]], on="country_code", how="left")
    denom = pd.to_numeric(merged["population"], errors="coerce")

    merged["ports_per_100k"] = (
        pd.to_numeric(merged["total_ports"], errors="coerce") / denom * 100_000.0
    ).replace([np.inf, -np.inf], np.nan)

    merged["ports_per_100k"] = merged["ports_per_100k"].round(3)
    return merged


def add_region(country_df: pd.DataFrame, region_df: pd.DataFrame) -> pd.DataFrame:
    out = country_df.copy()

    if region_df is None or region_df.empty:
        out["region"] = "Global"
        return out

    required = {"country_code", "region"}
    missing = required.difference(region_df.columns)
    if missing:
        raise ValueError(f"Region table missing columns: {sorted(missing)}")

    left = out.copy()
    left["country_code"] = left["country_code"].astype(str).str.upper()

    right = region_df.copy()
    right["country_code"] = right["country_code"].astype(str).str.upper()

    merged = left.merge(right[["country_code", "region"]], on="country_code", how="left")
    merged["region"] = merged["region"].fillna("Global")
    return merged


def lorenz_curve(values: pd.Series | np.ndarray) -> LorenzResult:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    v = np.clip(v, 0.0, None)

    if v.size == 0 or float(v.sum()) <= 0.0:
        return LorenzResult(x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0]), gini=0.0)

    v = np.sort(v)
    cum = np.cumsum(v)
    y = np.insert(cum / cum[-1], 0, 0.0)
    x = np.linspace(0.0, 1.0, y.size)
    gini = float(1.0 - 2.0 * _integrate_trapezoid(y, x))
    return LorenzResult(x=x, y=y, gini=gini)


def pareto_table(df: pd.DataFrame, value_col: str = "total_ports") -> pd.DataFrame:
    if "country_code" not in df.columns or value_col not in df.columns:
        raise ValueError(f"Missing required columns: country_code, {value_col}")

    t = df[["country_code", value_col]].copy()
    t[value_col] = pd.to_numeric(t[value_col], errors="coerce").fillna(0.0)
    t = t.sort_values(value_col, ascending=False).reset_index(drop=True)

    total = float(t[value_col].sum())
    if total <= 0.0:
        t["share"] = 0.0
        t["cum_share"] = 0.0
        t["rank"] = np.arange(1, len(t) + 1)
        return t

    t["share"] = 100.0 * t[value_col] / total
    t["cum_share"] = 100.0 * t[value_col].cumsum() / total
    t["rank"] = np.arange(1, len(t) + 1)
    return t


def minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if not s.notna().any():
        return pd.Series(50.0, index=series.index)

    lo = float(np.nanmin(s.to_numpy()))
    hi = float(np.nanmax(s.to_numpy()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(50.0, index=series.index)

    return (s - lo) / (hi - lo) * 100.0


def impact_index(country_df: pd.DataFrame) -> pd.DataFrame:
    d = country_df.copy()
    for col in ("fast_share", "ports_per_station", "avg_kw_per_port", "total_ports"):
        if col not in d.columns:
            d[col] = np.nan

    fast = pd.to_numeric(d["fast_share"], errors="coerce").fillna(0.0)
    pps = minmax(d["ports_per_station"].fillna(0.0))
    kwpp = minmax(d["avg_kw_per_port"].fillna(0.0))
    vol = minmax(d["total_ports"].fillna(0.0))

    score = 0.4 * fast + 0.3 * pps + 0.2 * kwpp + 0.1 * vol

    out = d.copy()
    out["impact_score"] = pd.to_numeric(score, errors="coerce").round(1)
    return out.sort_values("impact_score", ascending=False).reset_index(drop=True)


def opportunity_index(country_df: pd.DataFrame) -> pd.DataFrame:
    d = country_df.copy()
    for col in ("fast_share", "ports_per_station", "avg_kw_per_port", "stations"):
        if col not in d.columns:
            d[col] = np.nan

    fast = pd.to_numeric(d["fast_share"], errors="coerce").fillna(0.0)
    gap_fast = 100.0 - fast

    pps = minmax(d["ports_per_station"].fillna(0.0))
    kwpp = minmax(d["avg_kw_per_port"].fillna(0.0))

    gap_density = 100.0 - pps
    gap_kw = 100.0 - kwpp

    scale = minmax(pd.to_numeric(d["stations"], errors="coerce").fillna(0.0))

    score = 0.50 * gap_fast + 0.30 * gap_density + 0.10 * gap_kw + 0.10 * scale

    out = d.copy()
    out["opportunity_score"] = pd.to_numeric(score, errors="coerce").round(1)
    return out.sort_values("opportunity_score", ascending=False).reset_index(drop=True)


def city_outliers(stations: pd.DataFrame, z_threshold: float = 2.0) -> pd.DataFrame:
    required = {"country_code", "city", "id", "ports", "kw_per_port"}
    missing = required.difference(stations.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    d = stations.copy()
    d["country_code"] = d["country_code"].astype(str).str.upper()
    d["city"] = d["city"].astype(str)

    g = (
        d.groupby(["country_code", "city"], dropna=False)
        .agg(
            stations=("id", "count"),
            total_ports=("ports", "sum"),
            avg_kw_per_port=("kw_per_port", "mean"),
        )
        .reset_index()
    )

    if g.empty:
        g["z_kw_per_port"] = pd.Series(dtype=float)
        return g

    center = pd.to_numeric(g["avg_kw_per_port"], errors="coerce")
    fill = float(center.median()) if center.notna().any() else 0.0
    z = zscore(center.fillna(fill).to_numpy(), nan_policy="omit")

    g["z_kw_per_port"] = pd.Series(z, index=g.index).astype(float)
    out = g[g["z_kw_per_port"].abs() >= float(z_threshold)]
    return out.sort_values("z_kw_per_port", ascending=False).reset_index(drop=True)


def humanize(value: object) -> str:
    try:
        x = float(value)
    except Exception:
        return str(value)

    if not np.isfinite(x):
        return "n/a"
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.2f}K"
    if float(x).is_integer():
        return f"{x:.0f}"
    return f"{x:.2f}"


def humanize_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(humanize)
    return out
