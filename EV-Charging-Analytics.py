import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import zscore

# ============ App config ============
st.set_page_config(
    page_title="EV Charging Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============ Constants / Theme ============
PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

DATA_REPO_URL = "https://github.com/tarekmasryo/Global-EV-Charging-Stations"
RAW_DEFAULT_URL = "https://raw.githubusercontent.com/tarekmasryo/Global-EV-Charging-Stations/main/data/charging_stations_2025_world.csv"

REQUIRED_COLS = {
    "id","name","city","country_code","latitude","longitude",
    "ports","power_kw","power_class","is_fast_dc"
}

CSS = """
<style>
:root{
  --bg:#0e1117; --panel:#11141c; --ink:#e7eaf1; --muted:#9aa1ac; --stroke:#2a2f3a; --primary:#2b5cff;
}
html, body, [class*="css"] { font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif; }
.block-container { padding-top: .6rem; }
.section { background:var(--panel); border:1px solid var(--stroke); border-radius:14px; padding:16px; margin:10px 0 18px; }
.card { background:var(--panel); border:1px solid var(--stroke); border-radius:14px; padding:14px 16px; }
.card h4 { margin:0; font-size:13px; color:var(--muted); font-weight:600; }
.card .v { font-size:26px; font-weight:800; margin-top:4px; color:var(--ink); }
h1.title { margin:0 0 8px; color:var(--ink); font-size:28px }
.subtitle { color:var(--muted); font-size:12px; margin-bottom:8px; }
.badge { background:#1a2240; color:#bcd0ff; border:1px solid #2b3a73; border-radius:999px; padding:3px 10px; font-size:12px; display:inline-block; margin:0 6px 6px 0; }
hr { border:none; border-top:1px solid var(--stroke); margin:12px 0; }
.progress {height:12px;border-radius:8px;background:#1b1f2a;overflow:hidden;border:1px solid var(--stroke); position:relative}
.progress .center {position:absolute; left:50%; top:0; bottom:0; width:1px; background:#3a4050}
.progress .bar {position:absolute; top:0; bottom:0;}
.small { font-size:12px; color:#9aa1ac }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ============ URL normalization ============
def _normalize_url(path_or_url: str) -> str:
    if not isinstance(path_or_url, str):
        return path_or_url
    url = path_or_url.strip()
    if not url:
        return url
    if url.startswith(("http://", "https://")):
        m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)", url)
        if m:
            u, r, b, tail = m.groups()
            return f"https://raw.githubusercontent.com/{u}/{r}/{b}/{tail}"
        return url
    return url  # local path

@st.cache_data(show_spinner=True)
def _read_csv(src):
    if hasattr(src, "read"):  # uploaded file-like
        return pd.read_csv(src)
    return pd.read_csv(_normalize_url(src))

@st.cache_data(show_spinner=True)
def load_main(src):
    df = _read_csv(src)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {', '.join(sorted(missing))}")
    for c in ["ports", "power_kw", "latitude", "longitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ports"] = df["ports"].fillna(0).clip(lower=0)
    df["power_kw"] = df["power_kw"].fillna(np.nan)
    df["country_code"] = df["country_code"].astype(str).str.upper()
    df["city"] = df["city"].fillna("Unknown")
    df["is_fast_dc"] = df["is_fast_dc"].astype(str).str.lower().isin(["1","true","t","y","yes"])
    df["power_class"] = df["power_class"].fillna("UNKNOWN")
    df = df.dropna(subset=["latitude","longitude"])
    df["kw_per_port"] = (df["power_kw"] / df["ports"].replace(0, np.nan)).round(2)
    return df

@st.cache_data
def _load_optional(filename: str):
    try:
        import os
        if filename and os.path.exists(filename):
            return pd.read_csv(filename)
    except Exception:
        pass
    return pd.DataFrame()

POP_PATH = "world_population.csv"
REG_PATH = "country_region.csv"
pop_df = _load_optional(POP_PATH)
region_df = _load_optional(REG_PATH)

# ============ Analytics helpers ============
def aggregate_by_country(df):
    g = (df.groupby("country_code", as_index=False)
           .agg(stations=("id","count"),
                total_ports=("ports","sum"),
                fast_share=("is_fast_dc","mean"),
                avg_power_kw=("power_kw","mean"),
                avg_kw_per_port=("kw_per_port","mean")))
    g["fast_share"] = (g["fast_share"]*100).round(1)
    g["ports_per_station"] = (g["total_ports"]/g["stations"]).replace([np.inf,-np.inf], np.nan).round(2)
    return g.sort_values("total_ports", ascending=False)

def add_population(g, pop_df_in):
    if pop_df_in is None or pop_df_in.empty:
        g["ports_per_100k"] = np.nan
        return g
    p = pop_df_in.copy()
    p["country_code"] = p["country_code"].astype(str).str.upper()
    p["population"] = pd.to_numeric(p["population"], errors="coerce")
    g = g.merge(p, on="country_code", how="left")
    g["ports_per_100k"] = (g["total_ports"]/g["population"]*100000).replace([np.inf,-np.inf], np.nan).round(3)
    return g

def add_region(g, reg_df_in):
    if reg_df_in is None or reg_df_in.empty:
        g["region"] = "Global"
        return g
    r = reg_df_in.copy()
    r["country_code"] = r["country_code"].astype(str).str.upper()
    g = g.merge(r, on="country_code", how="left")
    g["region"] = g["region"].fillna("Global")
    return g

def lorenz_curve(values: pd.Series):
    x = np.array(values.fillna(0), dtype=float)
    if x.sum() == 0:
        return np.array([0,1]), np.array([0,1]), 0.0
    xs = np.sort(x)
    cum = np.cumsum(xs)
    L = np.insert(cum/cum[-1], 0, 0)
    X = np.linspace(0, 1, len(L))
    gini = 1 - 2*np.trapz(L, X)
    return X, L, gini

def pareto_table(g, col="total_ports"):
    t = g[["country_code", col]].sort_values(col, ascending=False).reset_index(drop=True)
    s = t[col].sum() or 1
    t["share"] = 100 * t[col] / s
    t["cum_share"] = 100 * t[col].cumsum() / s
    t["rank"] = np.arange(1, len(t)+1)
    return t

def minmax(s):
    a, b = s.min(), s.max()
    return pd.Series(50.0, index=s.index) if a == b else 100*(s-a)/(b-a)

def impact_index(g):
    z = pd.DataFrame(index=g.index)
    z["fast"] = g["fast_share"].fillna(0)
    z["pps"]  = minmax(g["ports_per_station"].fillna(0))
    z["kwpp"] = minmax(g["avg_kw_per_port"].fillna(0))
    z["vol"]  = minmax(g["total_ports"].fillna(0))
    score = 0.4*z["fast"] + 0.3*z["pps"] + 0.2*z["kwpp"] + 0.1*z["vol"]
    out = g.copy()
    out["impact_score"] = score.round(1)
    return out.sort_values("impact_score", ascending=False)

def opportunity_index(g):
    z = pd.DataFrame(index=g.index)
    z["gap_fast"]   = 100 - g["fast_share"].fillna(0)
    z["gap_density"]= 100 - minmax(g["ports_per_station"].fillna(0))
    z["gap_kw"]     = 100 - minmax(g["avg_kw_per_port"].fillna(0))
    z["scale"]      = minmax(g["stations"].fillna(0))
    score = 0.50*z["gap_fast"] + 0.30*z["gap_density"] + 0.10*z["gap_kw"] + 0.10*z["scale"]
    out = g.copy()
    out["opportunity_score"] = score.round(1)
    return out.sort_values("opportunity_score", ascending=False)

def city_outliers(df):
    c = (df.groupby(["country_code","city"], as_index=False)
           .agg(stations=("id","count"),
                total_ports=("ports","sum"),
                avg_kw_per_port=("kw_per_port","mean")))
    if c.empty:
        c["z_kw_per_port"] = []
        return c
    c["z_kw_per_port"] = zscore(c["avg_kw_per_port"].fillna(c["avg_kw_per_port"].median()), nan_policy="omit")
    return c.loc[c["z_kw_per_port"].abs()>=2.0].sort_values("z_kw_per_port", ascending=False)

def humanize(x):
    try:
        x = float(x)
        if abs(x) >= 1_000_000: return f"{x/1_000_000:.2f}M"
        if abs(x) >= 1_000:     return f"{x/1_000:.2f}K"
        return f"{x:.0f}" if float(x).is_integer() else f"{x:.2f}"
    except Exception:
        return x

def humanize_df(df_, cols):
    dd = df_.copy()
    for c in cols:
        if c in dd.columns:
            dd[c] = dd[c].apply(humanize)
    return dd

def rel_bar(label: str, rel_pct: float):
    mag = min(50.0, abs(rel_pct))
    left = 50.0 if rel_pct >= 0 else 50.0 - mag
    color = "#2b5cff" if rel_pct >= 0 else "#ff4d4d"
    st.markdown(f"{label}: {rel_pct:+.1f}%")
    st.markdown(
        f"""
        <div class="progress">
          <div class="center"></div>
          <div class="bar" style="left:{left}%; width:{mag}%; background:{color};"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============ Data source UX ============
with st.sidebar:
    st.header("Data")
    # Only two modes: default GitHub OR upload
    mode = st.radio("Load dataset from", ["File path", "Upload"], horizontal=True, index=0)
    st.caption("File path uses the GitHub dataset automatically. Upload lets you replace it with your own CSV.")

# Auto-load from GitHub (or env/secrets override) when mode == "File path"
def _default_source() -> str:
    # allow ?csv= or secrets/env overrides silently (no UI)
    try:
        q = st.query_params.get("csv", None)
        if q: return q
    except Exception:
        try:
            qp = st.experimental_get_query_params()
            if "csv" in qp and len(qp["csv"]) > 0:
                return qp["csv"][0]
        except Exception:
            pass
    try:
        if "DATA_URL" in st.secrets:
            return st.secrets["DATA_URL"]
    except Exception:
        pass
    for k in ("CSV_URL", "CSV_PATH", "DATA_URL"):
        v = os.getenv(k)
        if v: return v
    return RAW_DEFAULT_URL

df = None
auto_error = None

if mode == "File path":
    try:
        df = load_main(_default_source())
    except Exception as e:
        auto_error = str(e)
        st.error(f"Auto-load from GitHub failed.\n{auto_error}")
        st.link_button("Open dataset repo", DATA_REPO_URL)

else:  # Upload
    upload = st.file_uploader("Upload CSV", type="csv")
    if upload is not None:
        try:
            df = load_main(upload)
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")

# If still no data, stop.
if df is None:
    st.stop()

# ============ Filters ============
st.sidebar.header("Filters")
countries = df["country_code"].dropna().astype(str).str.upper().sort_values().unique().tolist()
sel_countries = st.sidebar.multiselect("Countries", countries, default=countries or [])
classes = sorted(df["power_class"].unique().tolist())
sel_classes = st.sidebar.multiselect("Power class", classes, default=classes)
fast_only = st.sidebar.toggle("Fast DC only", False)

pmin_raw, pmax_raw = df["ports"].min(), df["ports"].max()
pmin = int(pmin_raw) if pd.notna(pmin_raw) else 0
pmax = int(pmax_raw) if pd.notna(pmax_raw) else 1
if pmin == pmax:
    pmax = pmin + 1
sel_ports = st.sidebar.slider("Ports range", pmin, pmax, (pmin, pmax))
city_q = st.sidebar.text_input("City contains").strip().lower()

f = df.copy()
if sel_countries: f = f[f["country_code"].isin(sel_countries)]
if sel_classes:   f = f[f["power_class"].isin(sel_classes)]
if fast_only:     f = f[f["is_fast_dc"]]
f = f[f["ports"].between(sel_ports[0], sel_ports[1], inclusive="both")]
if city_q:        f = f[f["city"].str.lower().str.contains(city_q)]

# ============ Views ============
st.sidebar.header("Views")
view = st.sidebar.selectbox("Preset", ["Current filters","Global","Fast-DC only","Top-10 by ports"])
def _view_df():
    if view == "Global":
        return df.copy()
    if view == "Fast-DC only":
        t = df.copy()
        return t[t["is_fast_dc"]]
    if view == "Top-10 by ports":
        top = (df.groupby("country_code")["ports"].sum().sort_values(ascending=False).head(10).index.tolist())
        return df[df["country_code"].isin(top)]
    return f.copy()
view_df = _view_df()

# ============ Header ============
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("<h1 class='title'>⚡ EV Charging Analytics</h1>", unsafe_allow_html=True)
sub = f"{view_df['country_code'].nunique()} countries • {int(view_df['ports'].sum()):,} ports • Fast-DC {(100*view_df['is_fast_dc'].mean() if len(view_df) else 0):.1f}%"
st.markdown(f"<div class='subtitle'>{sub}</div>", unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
k1.markdown(f"<div class='card'><h4>Total Stations</h4><div class='v'>{view_df.shape[0]:,}</div></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='card'><h4>Total Ports</h4><div class='v'>{int(view_df['ports'].sum()):,}</div></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='card'><h4>Avg Power (kW)</h4><div class='v'>{(view_df['power_kw'].mean() if len(view_df) else 0):.1f}</div></div>", unsafe_allow_html=True)
k4.markdown(f"<div class='card'><h4>Countries</h4><div class='v'>{view_df['country_code'].nunique()}</div></div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ============ Pages ============
def page_overview(d):
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Summary")
    g = add_region(add_population(aggregate_by_country(d), pop_df), region_df)
    if g.empty:
        st.info("No data in view."); st.markdown('</div>', unsafe_allow_html=True); return
    t = pareto_table(g)
    X, L, gini = lorenz_curve(g["total_ports"])
    k80 = (t["cum_share"] <= 80).sum()
    badges = [
        f"{k80} countries hold 80% of total ports.",
        f"Top-3 share: {t['share'].head(3).sum():.1f}%",
        f"Top-5 share: {t['share'].head(5).sum():.1f}%",
        f"Gini (ports): {gini:.3f}",
    ]
    st.markdown("".join([f"<span class='badge'>{b}</span>" for b in badges]), unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        donut = pd.DataFrame({"kind": ["Fast-DC", "Non-Fast"], "value": [d["is_fast_dc"].sum(), (~d["is_fast_dc"]).sum()]})
        fig_donut = px.pie(donut, values="value", names="kind", hole=0.6, title="Charging mix")
        fig_donut.update_layout(height=320, showlegend=True)
        st.plotly_chart(fig_donut, use_container_width=True, config=PLOTLY_CONFIG)
    with c2:
        figp = go.Figure()
        figp.add_trace(go.Bar(x=t["country_code"], y=t["share"], name="Share (%)"))
        figp.add_trace(go.Scatter(x=t["country_code"], y=t["cum_share"], name="Cumulative (%)", yaxis="y2"))
        figp.update_layout(height=320, yaxis=dict(title="Share (%)"),
                           yaxis2=dict(title="Cumulative (%)", overlaying="y", side="right", range=[0, 100]))
        st.plotly_chart(figp, use_container_width=True, config=PLOTLY_CONFIG)
    st.markdown('</div>', unsafe_allow_html=True)

def page_mix(d):
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Power Class Mix")
    mix = (d.groupby(["country_code", "power_class"], as_index=False).agg(stations=("id", "count")))
    if mix.empty:
        st.info("No data in view.")
    else:
        fig = px.bar(mix, x="country_code", y="stations", color="power_class", title="Stations by power class (stacked)")
        fig.update_layout(height=520, bargap=0.15, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    st.markdown('</div>', unsafe_allow_html=True)

def page_map(d):
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Advanced Map")
    d = d.copy()
    d["_size"] = np.sqrt(np.clip(d["ports"], 1, None)) * 600
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=d,
        get_position=["longitude", "latitude"],
        get_radius="_size",
        get_fill_color=[30, 136, 229, 160],
        opacity=0.35,
        stroked=False,
        radius_min_pixels=1,
        radius_max_pixels=70,
        pickable=True,
        blend=True,
    )
    view = pdk.ViewState(
        latitude=d["latitude"].mean() if len(d) else df["latitude"].mean(),
        longitude=d["longitude"].mean() if len(d) else df["longitude"].mean(),
        zoom=2.2,
    )
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v11",
        initial_view_state=view,
        layers=[layer],
        tooltip={"text": "{name}\n{country_code} {city}\nPorts: {ports}\nPower: {power_kw}"}
    ), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def page_insights(d):
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Insights")
    g = add_region(add_population(aggregate_by_country(d), pop_df), region_df)
    if g.empty:
        st.info("No data in view."); st.markdown('</div>', unsafe_allow_html=True); return
    imp = impact_index(g); opp = opportunity_index(g)
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Impact Index (Top 10)**")
        st.dataframe(imp.head(10)[["country_code","impact_score","fast_share","ports_per_station","avg_kw_per_port","total_ports"]],
                     use_container_width=True, height=360)
    with c2:
        st.write("**Opportunity Score (Top 10)**")
        st.dataframe(opp.head(10)[["country_code","opportunity_score","fast_share","ports_per_station","avg_kw_per_port","stations"]],
                     use_container_width=True, height=360)
    st.markdown("<hr/>", unsafe_allow_html=True)
    t = pareto_table(g, "total_ports")
    figp = go.Figure()
    figp.add_trace(go.Bar(x=t["country_code"], y=t["share"], name="Share (%)"))
    figp.add_trace(go.Scatter(x=t["country_code"], y=t["cum_share"], name="Cumulative (%)", yaxis="y2"))
    figp.update_layout(height=360, yaxis=dict(title="Share (%)"),
                       yaxis2=dict(title="Cumulative (%)", overlaying="y", side="right", range=[0, 100]))
    st.plotly_chart(figp, use_container_width=True, config=PLOTLY_CONFIG)
    X, L, gini = lorenz_curve(g["total_ports"])
    figL = go.Figure()
    figL.add_trace(go.Scatter(x=X, y=L, mode="lines", name="Lorenz"))
    figL.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Equality", line=dict(dash="dash")))
    figL.update_layout(height=320, annotations=[dict(x=0.82, y=0.18, xref="paper", yref="paper", text=f"Gini={gini:.3f}", showarrow=False)])
    st.plotly_chart(figL, use_container_width=True, config=PLOTLY_CONFIG)
    st.markdown("<hr/>", unsafe_allow_html=True)
    out = city_outliers(d)
    st.write("**City outliers (|z| ≥ 2)**")
    if out.empty: st.info("No outliers detected.")
    else: st.dataframe(out.head(120), use_container_width=True, height=300)
    st.markdown('</div>', unsafe_allow_html=True)

def page_optimizer(d):
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Allocation Optimizer")
    g = add_region(add_population(aggregate_by_country(d), pop_df), region_df)
    if g.empty:
        st.info("No data in view."); st.markdown('</div>', unsafe_allow_html=True); return
    opp = opportunity_index(g)
    targets = st.multiselect("Target countries", opp["country_code"].tolist(), default=opp["country_code"].head(8).tolist())
    add_ports = st.slider("Total fast-DC ports to allocate", 0, 50000, 10000, step=1000)
    if not targets or add_ports == 0:
        st.info("Select targets and >0 ports to allocate."); st.markdown('</div>', unsafe_allow_html=True); return
    pool = opp[opp["country_code"].isin(targets)].copy()
    w = pool["opportunity_score"].clip(lower=1).astype(float)
    w_np = w.to_numpy()
    alloc_np = np.floor(add_ports * (w_np / w_np.sum())).astype(int)
    rem = int(add_ports - alloc_np.sum())
    if rem > 0:  alloc_np[np.argsort(-w_np)[:rem]] += 1
    elif rem < 0: alloc_np[np.argsort(w_np)[:(-rem)]] -= 1
    alloc = pd.Series(alloc_np, index=pool.index)
    plan = pool[["country_code","fast_share","ports_per_station","avg_kw_per_port","stations","total_ports"]].copy()
    plan["alloc_ports"] = alloc
    plan["new_total_ports"] = plan["total_ports"] + plan["alloc_ports"]
    plan["new_ports_per_station"] = (plan["new_total_ports"]/plan["stations"]).replace([np.inf,-np.inf], np.nan)
    plan["new_fast_share"] = np.clip(plan["fast_share"] + 5.0*(plan["alloc_ports"]>0), 0, 100)
    st.write("**Allocation plan**")
    st.dataframe(humanize_df(plan.sort_values("alloc_ports", ascending=False),
                             ["total_ports","alloc_ports","new_total_ports","stations"]),
                 use_container_width=True, height=420)
    st.markdown('</div>', unsafe_allow_html=True)

def page_compare(d):
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Compare — Selected vs Global/Region")
    g_sel = add_region(add_population(aggregate_by_country(d), pop_df), region_df)
    g_all = add_region(add_population(aggregate_by_country(df), pop_df), region_df)
    if g_sel.empty:
        st.info("No data in view."); st.markdown('</div>', unsafe_allow_html=True); return
    regions = ["Global"] + sorted(g_all["region"].dropna().unique().tolist())
    ref_region = st.selectbox("Reference region", regions, index=0)
    def agg_stats(g):
        cols = ["fast_share","ports_per_station","avg_kw_per_port","ports_per_100k"]
        return g[cols].mean(numeric_only=True)
    sel_stats = agg_stats(g_sel)
    ref_stats = agg_stats(g_all if ref_region == "Global" else g_all[g_all["region"]==ref_region])
    labels = ["Fast-DC (%)","Ports/Station","kW/port","Ports/100k"]
    keys   = ["fast_share","ports_per_station","avg_kw_per_port","ports_per_100k"]
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Selected (current view)**")
        for lab, k in zip(labels, keys):
            v = float(sel_stats.get(k, np.nan))
            st.markdown(f"{lab}: **{v:.2f}**")
    with c2:
        st.write(f"**Reference — {ref_region}**")
        for lab, k in zip(labels, keys):
            v = float(ref_stats.get(k, np.nan))
            st.markdown(f"{lab}: **{v:.2f}**")
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.subheader("Relative difference")
    for lab, k in zip(labels, keys):
        a = float(sel_stats.get(k, np.nan))
        b = float(ref_stats.get(k, np.nan))
        if np.isnan(a) or np.isnan(b) or b == 0:
            st.text(f"{lab}: n/a")
            continue
        rel = (a/b - 1.0) * 100.0
        rel_bar(lab, rel)
    st.markdown('</div>', unsafe_allow_html=True)

def page_stations(d):
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Stations")
    cols = ["id","name","city","country_code","ports","power_kw","power_class","is_fast_dc","latitude","longitude","kw_per_port"]
    st.dataframe(d[cols].head(2000), use_container_width=True, height=520)
    st.download_button(" Download CSV", d[cols].to_csv(index=False).encode("utf-8"),
                       "ev_charging_filtered.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

# ============ Router ============
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview","Mix","Map","Insights","Optimizer","Compare","Stations"], index=0)

# compute view_df once per render
def _get_view_df():
    if view == "Global":
        return df.copy()
    if view == "Fast-DC only":
        t = df.copy(); return t[t["is_fast_dc"]]
    if view == "Top-10 by ports":
        top = (df.groupby("country_code")["ports"].sum().sort_values(ascending=False).head(10).index.tolist())
        return df[df["country_code"].isin(top)]
    return f.copy()
view_df = _get_view_df()

if page == "Overview":     page_overview(view_df)
elif page == "Mix":        page_mix(view_df)
elif page == "Map":        page_map(view_df)
elif page == "Insights":   page_insights(view_df)
elif page == "Optimizer":  page_optimizer(view_df)
elif page == "Compare":    page_compare(view_df)
elif page == "Stations":   page_stations(view_df)
