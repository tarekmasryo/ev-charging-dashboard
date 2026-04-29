# Case Study — EV Charging Analytics Dashboard

## Problem

EV charging infrastructure data is large, fragmented, and difficult to evaluate quickly. A practical dashboard should help answer operational planning questions:

- Where is charging capacity concentrated?
- Which countries or cities have stronger fast-DC coverage?
- How unequal is port distribution across markets?
- Which countries look like stronger expansion opportunities?
- How would a simple allocation rule distribute new fast-DC capacity?

## Approach

The dashboard combines a Streamlit interface with a small analytics package under `src/`:

- **Data layer:** schema normalization, column aliases, robust boolean parsing, numeric coercion, and optional enrichment files.
- **Analytics layer:** country aggregation, Lorenz/Gini concentration metrics, impact scoring, opportunity scoring, outlier detection, and scenario allocation.
- **UI layer:** KPI cards, filters, map view, charts, comparison views, and exportable tables.

The main Streamlit entrypoint is:

```text
EV-Charging-Analytics.py
```

## Key Decisions

- **Stable app entrypoint:** the filename is kept unchanged to match the deployed Streamlit configuration.
- **Portable data loading:** the dataset source can be controlled through query params, Streamlit secrets, or environment variables.
- **Decision-ready metrics:** Lorenz curve and Gini score quantify charging-port concentration instead of relying only on bar charts.
- **Transparent scoring:** impact and opportunity scores are rule-based and easy to inspect.
- **Scenario workflow:** the optimizer shows how additional fast-DC capacity could be distributed under clear assumptions.
- **Tested data parsing:** boolean values such as `False`, `0`, `true`, and `yes` are parsed explicitly to avoid KPI distortion.

## Results

The dashboard supports:

- Country, city, power-class, fast-DC, and port-range filtering
- Global infrastructure KPIs
- Charging mix analysis
- Port concentration analysis
- Interactive station-level map markers
- Impact and opportunity ranking
- Fast-DC allocation scenarios
- Selected-slice vs baseline comparison

## Production Notes

This is a portfolio-grade analytics dashboard, not a full SaaS product. The current scope intentionally stays lightweight:

- No authentication layer
- No database persistence
- No scheduled ingestion jobs
- No live API refresh loop

The app is ready for Streamlit-style deployment after setting the correct main file path:

```text
EV-Charging-Analytics.py
```

## Next Steps

- Add a data-quality tab for missingness, duplicates, coordinate ranges, and schema drift.
- Add scenario exports for optimizer outputs.
- Add cost-weighted allocation using estimated kW, port count, and regional constraints.
- Add sensitivity analysis for opportunity-score weights.
- Add a lightweight monitoring note for dataset refresh date and row-count drift.
