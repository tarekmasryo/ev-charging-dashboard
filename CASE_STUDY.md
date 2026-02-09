# Case Study — EV Charging Analytics Dashboard (2025)

## Problem
Global EV charging data is large, fragmented, and hard to explore consistently. The goal was a single interactive dashboard that answers practical planning questions fast:

- Where are charging stations concentrated, and how unequal is the distribution?
- What share of infrastructure is fast-DC, and how does it vary by country/power class?
- Which countries look “high impact / high opportunity” when comparing coverage vs demand signals?
- If we add new fast-DC capacity, how should we allocate it under simple constraints?

## Approach
- Streamlit UI for fast iteration and a clean exploration flow.
- Plotly for interactive insights (KPIs, distributions, Lorenz curve + Gini).
- PyDeck for a global map with clustering and smooth filtering.
- Structured as a package (`src/`) with tests, Ruff, and pre-commit for production hygiene.

## Key Decisions
- **Portable data loading:** default public raw CSV auto-load with overrides via query param `?csv=...`, Streamlit secrets, or env vars.
- **Graceful enrichment:** optional files (e.g., population / region mappings) are used when present; related metrics are skipped when missing.
- **Decision-ready inequality metrics:** Lorenz curve + Gini to quantify how concentrated infrastructure is across countries.
- **Optimizer as a “what-if” tool:** simulate allocation of new fast-DC ports using **transparent, rule-based allocation** (easy to explain and audit).
- **Resilience by design:** fail fast on invalid inputs, but avoid crashing the UI on non-critical missing fields.

## Results
A decision-friendly dashboard that supports:
- Country/city/power-class/fast-DC filtering
- Global KPIs (stations, ports, avg kW, fast-DC share)
- Distribution insights (including Lorenz curve + Gini)
- Interactive clustered world map
- Compare mode (selected slice vs global)
- Allocation optimizer for fast-DC expansion scenarios

## Next Steps
- Add scenario presets (budget, target fast-DC share, priority regions) and exportable “scenario reports”.
- Add data validation summary (missingness/duplicates/range checks) with a small diagnostics panel.
- Add an `artifacts/` export path for filtered snapshots and charts.
- Extend the optimizer with cost weights (kW, ports, regional constraints) and sensitivity analysis.
