from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass
class DashboardData:
    stations: pd.DataFrame
    population_by_country: pd.DataFrame
    regions_by_country: pd.DataFrame

    def population_lookup(self) -> dict[str, float]:
        if self.population_by_country.empty:
            return {}

        required = {"country_code", "population"}
        if not required.issubset(self.population_by_country.columns):
            return {}

        series = self.population_by_country.set_index("country_code")["population"]
        series = pd.to_numeric(series, errors="coerce")
        return series.astype(float).to_dict()

    def region_lookup(self) -> dict[str, str]:
        if self.regions_by_country.empty:
            return {}

        required = {"country_code", "region"}
        if not required.issubset(self.regions_by_country.columns):
            return {}

        series = self.regions_by_country.set_index("country_code")["region"]
        return series.astype(str).to_dict()


@dataclass(frozen=True)
class DataService:
    load_main: Callable[[object], pd.DataFrame]
    load_optional: Callable[[object], pd.DataFrame]
    population_path: str
    regions_path: str

    def load(self, source: object) -> DashboardData:
        stations = self.load_main(source)
        population = self.load_optional(self.population_path)
        regions = self.load_optional(self.regions_path)
        return DashboardData(
            stations=stations,
            population_by_country=population,
            regions_by_country=regions,
        )
