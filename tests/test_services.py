import io

from ev_charging_dashboard.data import load_main, load_optional
from ev_charging_dashboard.services import DataService


def test_data_service_load_builds_dashboard_data(tmp_path) -> None:
    pop_path = tmp_path / "population.csv"
    pop_path.write_text("country_code,population\nUS,333000000\nCA,40000000\n", encoding="utf-8")

    reg_path = tmp_path / "regions.csv"
    reg_path.write_text(
        "country_code,region\nUS,North America\nCA,North America\n", encoding="utf-8"
    )

    service = DataService(
        load_main=load_main,
        load_optional=load_optional,
        population_path=str(pop_path),
        regions_path=str(reg_path),
    )

    csv_text = (
        "country,city,station_id,lat,lon,ports,power_kw\n"
        "US,New York,1,40.7128,-74.0060,4,200\n"
        "CA,Toronto,2,43.6532,-79.3832,2,60\n"
    )

    data = service.load(io.StringIO(csv_text))

    assert data.stations.shape[0] == 2
    assert set(["country_code", "latitude", "longitude", "is_fast_dc", "power_class"]).issubset(
        data.stations.columns
    )

    pop = data.population_lookup()
    reg = data.region_lookup()
    assert pop["US"] == 333000000
    assert reg["US"] == "North America"
