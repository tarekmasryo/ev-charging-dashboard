from ev_charging_dashboard.data import normalize_url


def test_normalize_url_converts_github_blob_to_raw() -> None:
    url = "https://github.com/acme/repo/blob/main/data/file.csv"
    out = normalize_url(url)
    assert out == "https://raw.githubusercontent.com/acme/repo/main/data/file.csv"


def test_normalize_url_keeps_non_github_urls() -> None:
    url = "https://example.com/data.csv"
    assert normalize_url(url) == url


def test_normalize_url_strips_query_params() -> None:
    url = "https://example.com/data.csv?download=1"
    assert normalize_url(url) == "https://example.com/data.csv"
