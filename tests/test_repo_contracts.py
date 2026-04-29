from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_FILE = "EV-Charging-Analytics.py"


def test_streamlit_entrypoint_exists_and_is_documented() -> None:
    assert (ROOT / APP_FILE).is_file()

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    devcontainer = (ROOT / ".devcontainer" / "devcontainer.json").read_text(encoding="utf-8")

    assert APP_FILE in readme
    assert APP_FILE in makefile
    assert APP_FILE in devcontainer
    assert "streamlit run app.py" not in readme
    assert "streamlit run app.py" not in makefile


def test_readme_does_not_claim_map_clustering_without_cluster_layer() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    case_study = (ROOT / "CASE_STUDY.md").read_text(encoding="utf-8").lower()

    assert "map with clustering" not in readme
    assert "clustered world map" not in case_study


def test_live_preview_asset_is_lightweight() -> None:
    preview = ROOT / "assets" / "Analytics.gif"
    if preview.exists():
        assert preview.stat().st_size <= 1_000_000
