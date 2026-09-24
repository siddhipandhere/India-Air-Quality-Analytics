"""
Dataset : Air Quality Data in India (2015-2020), file: city_day.csv
          https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="India City Air Quality", layout="wide")

DATA_FILE = "city_day.csv"
REQUIRED_COLUMNS = ["City", "Date", "AQI"]
POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
              "Benzene", "Toluene", "Xylene"]
FORM_POLLUTANTS = {"PM2.5": 500.0, "PM10": 800.0, "NO2": 300.0,
                   "CO": 50.0, "SO2": 200.0, "O3": 300.0}  # name -> slider max

# CPCB National AQI categories: name, upper bound, colour, health advisory
BUCKETS = [
    ("Good", 50, "#3a9d5d", "Minimal impact."),
    ("Satisfactory", 100, "#9cc35a",
     "Minor breathing discomfort for sensitive people."),
    ("Moderate", 200, "#e3b93b",
     "Discomfort for people with lung or heart disease, children and older adults."),
    ("Poor", 300, "#e98b3a", "Breathing discomfort for most people on prolonged exposure."),
    ("Very Poor", 400, "#d6453d", "Respiratory illness on prolonged exposure."),
    ("Severe", np.inf, "#8e2138",
     "Affects healthy people; serious impact on those with existing disease."),
]
BUCKET_ORDER = [b[0] for b in BUCKETS]
BUCKET_COLORS = {b[0]: b[2] for b in BUCKETS}
BUCKET_ADVICE = {b[0]: b[3] for b in BUCKETS}
BUCKET_BINS = [-np.inf, 50, 100, 200, 300, 400, np.inf]

SEASON_OF_MONTH = {12: "Winter", 1: "Winter", 2: "Winter",
                   3: "Summer", 4: "Summer", 5: "Summer",
                   6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
                   10: "Post-monsoon", 11: "Post-monsoon"}
SEASON_ORDER = ["Winter", "Summer", "Monsoon", "Post-monsoon"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

NUM_FEATURES = POLLUTANTS + ["AQI_today", "AQI_7d_avg", "Month"]
CAT_FEATURES = ["City"]
SPLIT_DATE = pd.Timestamp("2019-07-01")  # train before, test from this date on

# Visual identity: a hazy-sky base, ink text, and the CPCB AQI colours as the
# only strong colours, so colour on screen always means "air quality level".
HAZE, PANEL, INK, MUTED, RULE, TEAL = "#EDF0F1", "#E1E7E9", "#1D2A33", "#5B6770", "#CDD5D8", "#2E5E6E"
BODY_FONT = "IBM Plex Sans, Segoe UI, Helvetica, Arial, sans-serif"
HEAD_FONT = "IBM Plex Sans Condensed, Arial Narrow, Segoe UI, sans-serif"

pio.templates["haze"] = go.layout.Template(layout=dict(
    font=dict(family=BODY_FONT, size=13, color=INK),
    title=dict(font=dict(family=HEAD_FONT, size=17,
               color=INK), x=0, xanchor="left"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=[TEAL, "#8C6A3F", "#7A8C96", "#B5838D", "#4F772D", "#6D597A"],
    xaxis=dict(showgrid=False, linecolor=RULE, tickcolor=RULE, ticks="outside",
               zeroline=False, title_font=dict(color=MUTED)),
    yaxis=dict(gridcolor=RULE, linecolor=RULE,
               zeroline=False, title_font=dict(color=MUTED)),
    legend=dict(orientation="h", yanchor="top", y=-0.18, x=0, title_text=""),
    margin=dict(l=8, r=8, t=56, b=8),
    hoverlabel=dict(font_family=BODY_FONT, bgcolor="white"),
    coloraxis=dict(colorbar=dict(outlinewidth=0, thickness=12)),
))
PLOT_TEMPLATE = "haze"

st.markdown(
    """<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600;700&display=swap');
:root {--haze:#EDF0F1; --panel:#E1E7E9; --ink:#1D2A33; --muted:#5B6770; --rule:#CDD5D8; --teal:#2E5E6E;}
.stApp {background: var(--haze); color: var(--ink);}
html, body, .stApp, .stMarkdown, button, input, textarea, select, label,
[data-testid="stWidgetLabel"] {font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;}
h1, h2, h3 {font-family: 'IBM Plex Sans Condensed', 'Arial Narrow', sans-serif !important;
            color: var(--ink) !important; letter-spacing: -0.01em;}
h1 {font-weight: 700 !important; font-size: 2.5rem !important; line-height: 1.08 !important;
    padding-bottom: 0.2rem !important;}
h3 {font-weight: 600 !important; font-size: 1.35rem !important;}
header[data-testid="stHeader"] {background: transparent;}
[data-testid="stDecoration"], #MainMenu, footer {display: none;}
[data-testid="stHeaderActionElements"], .stMarkdown h1 a, .stMarkdown h3 a {display: none !important;}
.block-container {padding-top: 2.4rem; max-width: 1240px;}
[data-testid="stSidebar"] {background: var(--panel); border-right: 1px solid var(--rule);}
[data-testid="stSidebar"] [role="radiogroup"] label {padding: 0.22rem 0;}
.brand {font-family: 'IBM Plex Sans Condensed', sans-serif; font-weight: 700; font-size: 1.35rem;
        line-height: 1.15; color: var(--ink); margin-bottom: 0.2rem;}
.brand-sub {font-size: 0.85rem; color: var(--muted); margin-bottom: 1.2rem;}
.lede {color: var(--muted); font-size: 1.02rem; max-width: 68ch; margin: 0 0 1.8rem;}
.kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
       gap: 1.2rem 1.6rem; margin: 0.2rem 0 2rem;}
.kpi {border-left: 4px solid; padding: 0.05rem 0 0.1rem 0.8rem; min-width: 0;}
.kpi-label {font-size: 0.84rem; color: var(--muted);}
.kpi-value {font-family: 'IBM Plex Sans Condensed', sans-serif; font-weight: 600; font-size: 1.85rem;
            line-height: 1.15; color: var(--ink); overflow-wrap: anywhere;}
.kpi-note {font-size: 0.8rem; color: var(--muted);}
.hero {display: grid; grid-template-columns: auto 1fr; gap: 2.4rem; align-items: end; margin: 0.4rem 0 2.2rem;}
@media (max-width: 760px) {.hero {grid-template-columns: 1fr; gap: 1rem;}}
.hero-num {font-family: 'IBM Plex Sans Condensed', sans-serif; font-weight: 700; font-size: 5.6rem;
           line-height: 0.9; color: var(--ink);}
.hero-cap {font-size: 0.9rem; color: var(--muted); margin-top: 0.5rem;}
.scale {position: relative; padding-top: 1.7rem;}
.scale-bar {display: flex; height: 14px; border-radius: 2px; overflow: hidden;}
.scale-labels {display: flex; font-size: 0.75rem; color: var(--muted); margin-top: 0.4rem;}
.scale-labels div, .scale-bar div {min-width: 0;}
.scale-marker {position: absolute; top: 0; transform: translateX(-50%); font-size: 0.8rem;
               font-weight: 600; white-space: nowrap; color: var(--ink);}
.scale-marker::after {content: ""; display: block; width: 2px; height: 30px; background: var(--ink);
                      margin: 3px auto 0;}
.col-head {font-family: 'IBM Plex Sans Condensed', sans-serif; font-weight: 600; font-size: 1.3rem;
           border-left: 4px solid; padding-left: 0.6rem; margin: 0.4rem 0 0.6rem;}
.forecast {border-left: 6px solid; padding: 0.7rem 1.1rem; background: rgba(255,255,255,0.6);
           margin: 0.8rem 0; max-width: 640px;}
.forecast-num {font-family: 'IBM Plex Sans Condensed', sans-serif; font-weight: 700; font-size: 3rem;
               line-height: 1;}
.forecast-cat {font-weight: 600; font-size: 1.05rem;}
.forecast-note {color: var(--muted); font-size: 0.92rem; margin-top: 0.3rem;}
.footnote {color: var(--muted); font-size: 0.8rem; margin-top: 2.4rem;}
</style>""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_bucket(aqi_values):
    """Map AQI numbers to CPCB category names (ordered categorical)."""
    return pd.cut(aqi_values, bins=BUCKET_BINS, labels=BUCKET_ORDER)


def bucket_of(value):
    return str(to_bucket(pd.Series([value])).iloc[0])


def fmt_pct(x):
    return f"{x:+.1f}%"


def color_of(value):
    return BUCKET_COLORS[bucket_of(value)]


def page_header(title, lede):
    st.markdown(f"<h1>{title}</h1><p class='lede'>{lede}</p>",
                unsafe_allow_html=True)


def kpi_row(items):
    """items: list of (label, value, note, border colour)."""
    cells = "".join(
        f"<div class='kpi' style='border-color:{c}'><div class='kpi-label'>{l}</div>"
        f"<div class='kpi-value'>{v}</div><div class='kpi-note'>{n}</div></div>"
        for l, v, n, c in items)
    st.markdown(f"<div class='kpis'>{cells}</div>", unsafe_allow_html=True)


def aqi_scale(value, caption):
    """Headline AQI number with its position on the CPCB colour scale."""
    widths = [50, 50, 100, 100, 100, 100]  # scale drawn from 0 to 500
    bar = "".join(
        f"<div style='flex:{w};background:{b[2]}'></div>" for w, b in zip(widths, BUCKETS))
    labels = "".join(
        f"<div style='flex:{w}'>{b[0]}</div>" for w, b in zip(widths, BUCKETS))
    pos = min(max(value / 500 * 100, 4), 96)
    st.markdown(
        f"<div class='hero'><div><div class='hero-num'>{value:.0f}</div>"
        f"<div class='hero-cap'>{caption}</div></div>"
        f"<div class='scale'><div class='scale-marker' style='left:{pos:.1f}%'>{bucket_of(value)}</div>"
        f"<div class='scale-bar'>{bar}</div><div class='scale-labels'>{labels}</div></div></div>",
        unsafe_allow_html=True)


def section(title):
    st.markdown(f"### {title}")


# ---------------------------------------------------------------------------
# 1. Data loading and cleaning
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data...")
def load_data(source):
    return pd.read_csv(source)


@st.cache_data(show_spinner="Cleaning data...")
def clean_data(raw: pd.DataFrame):
    """Clean the raw city-day table and return (clean_df, cleaning_report)."""
    df = raw.copy()
    present = [c for c in POLLUTANTS if c in df.columns]
    for c in POLLUTANTS:
        if c not in df.columns:
            df[c] = np.nan

    missing_before = df[POLLUTANTS + ["AQI"]].isna().mean().mul(100).round(1)
    raw_rows = len(df)

    # Types
    df["City"] = df["City"].astype(str).str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    for c in POLLUTANTS + ["AQI"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        # negative concentrations are sensor errors
        df.loc[df[c] < 0, c] = np.nan

    # Duplicates
    dups = int(df.duplicated(subset=["City", "Date"]).sum())
    df = (df.drop_duplicates(subset=["City", "Date"])
            .sort_values(["City", "Date"])
            .reset_index(drop=True))

    # Fill short sensor gaps (up to 3 days) within each city by interpolation.
    # The AQI target itself is never imputed.
    df[POLLUTANTS] = df.groupby("City")[POLLUTANTS].transform(
        lambda s: s.interpolate(limit=3, limit_direction="both"))

    # Derived columns (AQI category recomputed from AQI for consistency)
    df["AQI_Bucket"] = to_bucket(df["AQI"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Season"] = df["Month"].map(SEASON_OF_MONTH)

    missing_after = df[POLLUTANTS + ["AQI"]].isna().mean().mul(100).round(1)
    report = {
        "raw_rows": raw_rows,
        "clean_rows": len(df),
        "duplicates": dups,
        "cities": df["City"].nunique(),
        "start": df["Date"].min(),
        "end": df["Date"].max(),
        "rows_with_aqi": int(df["AQI"].notna().sum()),
        "pollutants_found": present,
        "missing": pd.DataFrame({"Missing before (%)": missing_before,
                                 "Missing after (%)": missing_after}),
    }
    return df, report


# ---------------------------------------------------------------------------
# 2. Forecast model: predict tomorrow's AQI from today's readings
# ---------------------------------------------------------------------------
def build_forecast_frame(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["City", "Date"]).copy()
    g = d.groupby("City")
    d["AQI_today"] = d["AQI"]
    d["AQI_7d_avg"] = g["AQI"].transform(
        lambda s: s.rolling(7, min_periods=3).mean())
    d["AQI_next_day"] = g["AQI"].shift(-1)
    d["Next_date"] = g["Date"].shift(-1)
    d = d[(d["Next_date"] - d["Date"]).dt.days == 1]
    return d.dropna(subset=["AQI_today", "AQI_next_day"]).reset_index(drop=True)


@st.cache_resource(show_spinner="Training the next-day AQI model...")
def train_model(df: pd.DataFrame):
    data = build_forecast_frame(df)
    train = data[data["Date"] < SPLIT_DATE]
    test = data[data["Date"] >= SPLIT_DATE]
    if len(train) < 500 or len(test) < 100:  # fall back to a chronological 80/20 split
        cut = data["Date"].quantile(0.8)
        train, test = data[data["Date"] < cut], data[data["Date"] >= cut]

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])
    model = Pipeline([
        ("prep", pre),
        ("rf", RandomForestRegressor(n_estimators=200, max_depth=16,
                                     min_samples_leaf=3, n_jobs=-1,
                                     random_state=42)),
    ])
    X_cols = NUM_FEATURES + CAT_FEATURES
    model.fit(train[X_cols], train["AQI_next_day"])

    test = test.copy()
    test["Predicted"] = model.predict(test[X_cols])
    y, p = test["AQI_next_day"], test["Predicted"]
    persistence_mae = mean_absolute_error(
        y, test["AQI_today"])  # "tomorrow = today"
    mae = mean_absolute_error(y, p)

    actual_alert, pred_alert = y > 200, p > 200  # Poor or worse
    metrics = {
        "MAE": mae,
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": r2_score(y, p),
        "Baseline MAE": persistence_mae,
        "Improvement": (persistence_mae - mae) / persistence_mae * 100,
        "Category accuracy": float((to_bucket(y).astype(str).values
                                    == to_bucket(p).astype(str).values).mean() * 100),
        "Alert recall": float((pred_alert & actual_alert).sum()
                              / max(actual_alert.sum(), 1) * 100),
        "Train rows": len(train),
        "Test rows": len(test),
        "Test start": test["Date"].min(),
    }

    names = model.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(model.named_steps["rf"].feature_importances_, index=names)
    imp.index = ["City" if n.startswith(
        "cat__") else n.replace("num__", "") for n in imp.index]
    importance = imp.groupby(level=0).sum().sort_values(ascending=True)

    medians = train.groupby("City")[NUM_FEATURES].median()
    return model, metrics, test, importance, medians


# ---------------------------------------------------------------------------
# 3. Analysis helpers
# ---------------------------------------------------------------------------
def lockdown_change(df: pd.DataFrame):
    """Average AQI in the 2020 lockdown window vs the same window in 2019."""
    def window(year):
        m = (df["Date"] >= f"{year}-03-25") & (df["Date"] <= f"{year}-05-31")
        return df[m].groupby("City")["AQI"].mean()

    w19, w20 = window(2019), window(2020)
    both = pd.concat([w19.rename("2019"), w20.rename("2020")], axis=1).dropna()
    if both.empty:
        return None, None
    both["Change (%)"] = (both["2020"] - both["2019"]) / both["2019"] * 100
    overall = (both["2020"].mean() - both["2019"].mean()) / \
        both["2019"].mean() * 100
    return both.sort_values("Change (%)"), overall


def compute_insights(a: pd.DataFrame, full_for_cities: pd.DataFrame):
    city_avg = a.groupby("City")["AQI"].mean().sort_values(ascending=False)
    season_avg = a.groupby("Season")["AQI"].mean().reindex(
        SEASON_ORDER).dropna()
    month_avg = a.groupby("Month")["AQI"].mean()
    corr = (a[POLLUTANTS + ["AQI"]].corr(numeric_only=True)["AQI"]
            .drop("AQI").dropna().sort_values(ascending=False))
    city_poor = (a.assign(poor=a["AQI"] > 200).groupby(
        "City")["poor"].mean() * 100)

    yearly = a.groupby("Year")["AQI"].mean()
    full_years = [y for y in yearly.index if a.loc[a["Year"]
                                                   == y, "Month"].nunique() == 12]
    trend = None
    if len(full_years) >= 2:
        y0, y1 = full_years[0], full_years[-1]
        # Compare only cities reporting in both years, so monitoring stations
        # added later do not distort the trend.
        common = (set(a.loc[a["Year"] == y0, "City"])
                  & set(a.loc[a["Year"] == y1, "City"]))
        if common:
            m0 = a[(a["Year"] == y0) & a["City"].isin(common)]["AQI"].mean()
            m1 = a[(a["Year"] == y1) & a["City"].isin(common)]["AQI"].mean()
            trend = (y0, y1, (m1 - m0) / m0 * 100, len(common))

    _, lock_overall = lockdown_change(full_for_cities)
    return {
        "city_avg": city_avg,
        "season_avg": season_avg,
        "worst_month": MONTH_NAMES[int(month_avg.idxmax()) - 1],
        "best_month": MONTH_NAMES[int(month_avg.idxmin()) - 1],
        "corr": corr,
        "poor_share": (a["AQI"] > 200).mean() * 100,
        "city_poor": city_poor.sort_values(ascending=False),
        "trend": trend,
        "lockdown": lock_overall,
    }


# ---------------------------------------------------------------------------
# 4. App: data source and filters
# ---------------------------------------------------------------------------
st.sidebar.markdown("<div class='brand'>India city air quality</div>"
                    "<div class='brand-sub'>Daily CPCB readings, 2015 to 2020</div>",
                    unsafe_allow_html=True)

source = DATA_FILE if os.path.exists(DATA_FILE) else st.sidebar.file_uploader(
    "Upload city_day.csv", type="csv")
if source is None:
    page_header("India city air quality", "Add the dataset to start.")
    st.info("Place **city_day.csv** next to app.py, or upload it from the sidebar. "
            "Download it from https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india")
    st.stop()

raw = load_data(source)
missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
if missing_cols:
    st.error(f"The file is missing required columns: {', '.join(missing_cols)}. "
             "Use city_day.csv from the Kaggle dataset.")
    st.stop()

df, report = clean_data(raw)

page = st.sidebar.radio("Page", ["Executive overview", "Trends & seasons",
                                 "Pollution drivers", "Next-day AQI forecast",
                                 "Insights & actions"])

all_cities = sorted(df["City"].unique())
if st.sidebar.toggle("All cities", value=True):
    cities = all_cities
else:
    starter = [c for c in ["Delhi", "Mumbai", "Kolkata",
                           "Chennai", "Bengaluru"] if c in all_cities]
    cities = st.sidebar.multiselect(
        "Cities", all_cities, default=starter or all_cities[:3])
y_min, y_max = int(df["Year"].min()), int(df["Year"].max())
years = st.sidebar.slider("Years", y_min, y_max,
                          (y_min, y_max)) if y_min < y_max else (y_min, y_max)

st.sidebar.markdown("**AQI categories**")
legend = "".join(
    f"<div><span style='background:{c}'></span>{n} "
    f"({'401+' if np.isinf(hi) else f'{lo}–{int(hi)}'})</div>"
    for (n, hi, c, _), lo in zip(BUCKETS, [0, 51, 101, 201, 301, 401]))
st.sidebar.markdown(
    f"<div class='aqi-legend'>{legend}</div>", unsafe_allow_html=True)

f = df[df["City"].isin(cities) & df["Year"].between(years[0], years[1])]
a = f.dropna(subset=["AQI"])  # rows with an official AQI reading
if a.empty:
    st.warning(
        "No AQI readings match these filters. Select at least one city or widen the year range.")
    st.stop()


# ---------------------------------------------------------------------------
# Page: Executive overview
# ---------------------------------------------------------------------------
if page == "Executive overview":
    page_header("How clean is the air in India's cities?",
                f"{len(a):,} daily AQI readings from {a['City'].nunique()} cities, "
                f"{a['Date'].min():%B %Y} to {a['Date'].max():%B %Y}.")

    city_avg = a.groupby("City")["AQI"].mean().sort_values()
    avg = a["AQI"].mean()
    aqi_scale(avg, "average AQI across the selection")
    poor = (a["AQI"] > 200).mean() * 100
    kpi_row([
        ("Days Poor or worse", f"{poor:.1f}%",
         "AQI above 200", BUCKET_COLORS["Poor"]),
        ("Severe days", f"{int((a['AQI'] > 400).sum()):,}",
         "AQI above 400", BUCKET_COLORS["Severe"]),
        ("Most polluted city", city_avg.index[-1], f"average AQI {city_avg.iloc[-1]:.0f}",
         color_of(city_avg.iloc[-1])),
        ("Cleanest city", city_avg.index[0], f"average AQI {city_avg.iloc[0]:.0f}",
         color_of(city_avg.iloc[0])),
    ])

    c1, c2 = st.columns([3, 2])
    with c1:
        ranking = city_avg.reset_index()
        ranking["Category"] = to_bucket(ranking["AQI"]).astype(str)
        fig = px.bar(ranking, x="AQI", y="City", orientation="h", color="Category",
                     color_discrete_map=BUCKET_COLORS,
                     category_orders={"Category": BUCKET_ORDER},
                     title="Average AQI by city", template=PLOT_TEMPLATE,
                     height=max(350, 24 * len(ranking)))
        fig.update_layout(yaxis_title=None, legend_title=None)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        dist = (a["AQI_Bucket"].astype(str).value_counts(normalize=True)
                .mul(100).reindex(BUCKET_ORDER).fillna(0).reset_index())
        dist.columns = ["Category", "Share of days (%)"]
        fig = px.bar(dist, x="Category", y="Share of days (%)", color="Category",
                     color_discrete_map=BUCKET_COLORS, template=PLOT_TEMPLATE,
                     title="How often the air falls in each category")
        fig.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Data cleaning summary"):
        kpi_row([
            ("Raw rows", f"{report['raw_rows']:,}", "as downloaded", RULE),
            ("Duplicates removed",
             f"{report['duplicates']:,}", "same city and date", RULE),
            ("Rows with AQI",
             f"{report['rows_with_aqi']:,}", "used for AQI analysis", RULE),
            ("Cities", f"{report['cities']}", "in the dataset", RULE),
        ])
        st.markdown(
            "- Parsed dates and converted all readings to numbers\n"
            "- Treated negative concentrations as sensor errors (set to missing)\n"
            "- Removed duplicate city-date rows\n"
            "- Filled pollutant gaps of up to 3 days by linear interpolation within each city\n"
            "- Never imputed AQI itself: days without an official AQI are excluded from AQI analysis\n"
            "- Recomputed the AQI category from AQI using CPCB breakpoints; added year, month and season")
        st.dataframe(report["missing"], use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Trends & seasons
# ---------------------------------------------------------------------------
elif page == "Trends & seasons":
    page_header("When is the air worst?",
                "Pollution in Indian cities follows the calendar. The dotted lines mark the "
                "Moderate (100) and Poor (200) thresholds.")

    m = a.assign(Month_start=a["Date"].dt.to_period("M").dt.to_timestamp())
    if a["City"].nunique() <= 6:
        monthly = m.groupby(["Month_start", "City"])[
            "AQI"].mean().reset_index()
        fig = px.line(monthly, x="Month_start", y="AQI", color="City",
                      template=PLOT_TEMPLATE, title="Monthly average AQI by city")
    else:
        monthly = m.groupby("Month_start")["AQI"].mean().reset_index()
        fig = px.line(monthly, x="Month_start", y="AQI", template=PLOT_TEMPLATE,
                      title="Monthly average AQI across selected cities "
                            "(select 6 or fewer cities to compare them)")
    for lo, colour in [(200, BUCKET_COLORS["Poor"]), (100, BUCKET_COLORS["Moderate"])]:
        fig.add_hline(y=lo, line_dash="dot", line_color=colour, opacity=0.6)
    fig.update_layout(xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        heat = a.pivot_table(index="Year", columns="Month",
                             values="AQI", aggfunc="mean")
        heat = heat.reindex(columns=range(1, 13))
        fig = px.imshow(heat, x=MONTH_NAMES, y=[str(y) for y in heat.index],
                        color_continuous_scale=[BUCKET_COLORS[b]
                                                for b in BUCKET_ORDER],
                        aspect="auto", text_auto=".0f",
                        title="Average AQI by month and year", template=PLOT_TEMPLATE)
        fig.update_layout(coloraxis_colorbar_title="AQI")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        season = a.groupby("Season")["AQI"].mean().reindex(
            SEASON_ORDER).dropna().reset_index()
        season["Category"] = to_bucket(season["AQI"]).astype(str)
        fig = px.bar(season, x="Season", y="AQI", color="Category",
                     color_discrete_map=BUCKET_COLORS, template=PLOT_TEMPLATE,
                     title="Average AQI by season")
        fig.update_layout(xaxis_title=None, legend_title=None)
        st.plotly_chart(fig, use_container_width=True)

    section("The 2020 lockdown as a natural experiment")
    lock, overall = lockdown_change(df[df["City"].isin(cities)])
    if lock is None:
        st.info(
            "The selected cities have no readings for both 25 Mar–31 May 2019 and 2020.")
    else:
        st.markdown(f"<p class='lede'>Between 25 March and 31 May 2020, average AQI changed by "
                    f"<b>{fmt_pct(overall)}</b> across {len(lock)} cities compared with the same "
                    f"weeks in 2019. With traffic and industry largely shut, this is a measure of "
                    f"how much of the pollution comes from human activity.</p>",
                    unsafe_allow_html=True)
        fig = px.bar(lock.reset_index(), x="Change (%)", y="City", orientation="h",
                     template=PLOT_TEMPLATE, color_discrete_sequence=[
                         BUCKET_COLORS["Good"]],
                     title="Change in average AQI, lockdown weeks 2020 against 2019",
                     height=max(300, 24 * len(lock)))
        fig.update_layout(yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Pollution drivers
# ---------------------------------------------------------------------------
elif page == "Pollution drivers":
    page_header("What drives the AQI?",
                "Which pollutants move with the index, how they change through the year, "
                "and how each city's pollution mix differs.")

    corr = (a[POLLUTANTS + ["AQI"]].corr(numeric_only=True)["AQI"]
            .drop("AQI").dropna().sort_values())
    c1, c2 = st.columns(2)
    with c1:
        corr_df = corr.rename("Correlation").rename_axis(
            "Pollutant").reset_index()
        fig = px.bar(corr_df, x="Correlation", y="Pollutant", orientation="h",
                     template=PLOT_TEMPLATE, color_discrete_sequence=[
                         BUCKET_COLORS["Poor"]],
                     title="Correlation of each pollutant with AQI")
        fig.update_layout(xaxis_title="Correlation with AQI", yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        top = corr.index[-1] if len(corr) else "PM2.5"
        sample = a.dropna(subset=[top])
        sample = sample.sample(min(5000, len(sample)), random_state=1)
        sample = sample.assign(Category=sample["AQI_Bucket"].astype(str))
        fig = px.scatter(sample, x=top, y="AQI", color="Category",
                         color_discrete_map=BUCKET_COLORS,
                         category_orders={"Category": BUCKET_ORDER}, opacity=0.5,
                         template=PLOT_TEMPLATE, title=f"{top} vs AQI (sample of days)")
        fig.update_layout(legend_title=None)
        st.plotly_chart(fig, use_container_width=True)

    key = [p for p in ["PM2.5", "PM10", "NO2",
                       "SO2", "O3", "CO"] if a[p].notna().any()]
    seas = a.groupby("Season")[key].mean().reindex(
        SEASON_ORDER).dropna(how="all")
    fig = go.Figure()
    for p in key:
        fig.add_bar(name=p, x=seas.index, y=seas[p])
    fig.update_layout(barmode="group", template=PLOT_TEMPLATE,
                      title="Average pollutant concentration by season (CO in mg/m³, others µg/m³)")
    st.plotly_chart(fig, use_container_width=True)

    city_profile = a.groupby("City")[key].mean()
    fig = px.imshow((city_profile / city_profile.max()).round(2), aspect="auto",
                    color_continuous_scale=["#F4F6F6", "#E3B93B", "#E98B3A", "#8E2138"], template=PLOT_TEMPLATE,
                    title="Pollution profile by city (1.0 = highest city for that pollutant)")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Next-day AQI forecast
# ---------------------------------------------------------------------------
elif page == "Next-day AQI forecast":
    page_header("Forecasting tomorrow's AQI",
                "A random forest model predicts tomorrow's AQI from today's pollutant readings, "
                "today's AQI, the 7-day average, the month and the city. It learns from earlier "
                "years and is scored on later dates it has never seen.")

    model, mt, test, importance, medians = train_model(df)

    kpi_row([
        ("Average error", f"{mt['MAE']:.1f}", "AQI points (MAE)", TEAL),
        ("Better than guessing", f"{mt['Improvement']:.0f}%",
         f"vs 'same as today' (MAE {mt['Baseline MAE']:.1f})", TEAL),
        ("Correct category",
         f"{mt['Category accuracy']:.0f}%", "of test days", TEAL),
        ("Unsafe days caught", f"{mt['Alert recall']:.0f}%",
         "of days above AQI 200", BUCKET_COLORS["Poor"]),
        ("R²", f"{mt['R2']:.2f}", f"RMSE {mt['RMSE']:.1f}", RULE),
    ])
    st.markdown(f"<p class='lede'>Trained on {mt['Train rows']:,} city-days and tested on "
                f"{mt['Test rows']:,} city-days from {mt['Test start']:%d %B %Y} onward.</p>",
                unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        test_cities = sorted(test["City"].unique())
        default = test_cities.index("Delhi") if "Delhi" in test_cities else 0
        city = st.selectbox("City to inspect", test_cities, index=default)
        t = test[test["City"] == city]
        fig = go.Figure()
        fig.add_scatter(x=t["Next_date"], y=t["AQI_next_day"], name="Actual", mode="lines",
                        line=dict(color=MUTED, width=1.5))
        fig.add_scatter(x=t["Next_date"], y=t["Predicted"], name="Forecast", mode="lines",
                        line=dict(color=TEAL, width=2))
        fig.update_layout(template=PLOT_TEMPLATE,
                          title=f"Actual vs forecast AQI, {city}")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        imp_df = importance.tail(10).rename(
            "Importance").rename_axis("Feature").reset_index()
        fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                     template=PLOT_TEMPLATE, color_discrete_sequence=[TEAL],
                     title="What the model relies on most")
        fig.update_layout(xaxis_title="Importance", yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    section("Try a forecast")
    with st.form("forecast"):
        c1, c2, c3, c4 = st.columns(4)
        f_city = c1.selectbox("City", list(medians.index))
        f_month = c2.selectbox("Month", MONTH_NAMES, index=10)
        med = medians.loc[f_city]
        today = c3.number_input("Today's AQI", 0.0, 1000.0,
                                float(round(med["AQI_today"], 0)) if pd.notna(med["AQI_today"]) else 150.0)
        week = c4.number_input("Average AQI over last 7 days", 0.0, 1000.0,
                               float(round(med["AQI_7d_avg"], 0)) if pd.notna(med["AQI_7d_avg"]) else 150.0)
        cols = st.columns(len(FORM_POLLUTANTS))
        values = {}
        for col, (p, pmax) in zip(cols, FORM_POLLUTANTS.items()):
            default_v = float(med[p]) if pd.notna(med[p]) else 0.0
            values[p] = col.number_input(
                f"Today's {p}", 0.0, pmax, round(min(default_v, pmax), 1))
        submitted = st.form_submit_button("Forecast tomorrow's AQI")

    if submitted:
        # unspecified pollutants use training medians
        row = {c: np.nan for c in NUM_FEATURES}
        row.update(values)
        row.update({"AQI_today": today, "AQI_7d_avg": week,
                    "Month": MONTH_NAMES.index(f_month) + 1, "City": f_city})
        pred = float(model.predict(pd.DataFrame([row])[
                     NUM_FEATURES + CAT_FEATURES])[0])
        cat = bucket_of(pred)
        alert = ("<div class='forecast-note'><b>Alert level.</b> Advise schools to limit outdoor "
                 "activity and tighten construction and traffic controls for the day.</div>"
                 if pred > 200 else "")
        st.markdown(
            f"<div class='forecast' style='border-color:{BUCKET_COLORS[cat]}'>"
            f"<div class='kpi-label'>Tomorrow in {f_city}</div>"
            f"<div class='forecast-num'>{pred:.0f}</div>"
            f"<div class='forecast-cat' style='color:{BUCKET_COLORS[cat]}'>{cat}</div>"
            f"<div class='forecast-note'>{BUCKET_ADVICE[cat]}</div>{alert}</div>",
            unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: Insights & actions
# ---------------------------------------------------------------------------
elif page == "Insights & actions":
    page_header("Findings and recommended actions",
                "Written from the cities and years selected in the sidebar, so the "
                "recommendations change with the selection.")
    ins = compute_insights(a, df[df["City"].isin(cities)])
    ca, sa, corr = ins["city_avg"], ins["season_avg"], ins["corr"]

    section("What the data shows")
    facts = [
        f"Average AQI across the selection is **{a['AQI'].mean():.0f} ({bucket_of(a['AQI'].mean())})**, "
        f"and **{ins['poor_share']:.1f}%** of city-days were Poor or worse.",
        f"**{ca.index[0]}** is the most polluted city (average AQI {ca.iloc[0]:.0f}); "
        f"**{ca.index[-1]}** is the cleanest ({ca.iloc[-1]:.0f}).",
    ]
    if len(sa) >= 2:
        facts.append(f"**{sa.idxmax()}** is the worst season (average AQI {sa.max():.0f}), about "
                     f"**{sa.max() / sa.min():.1f}×** the cleanest season, **{sa.idxmin()}**. "
                     f"The worst month is {ins['worst_month']}; the cleanest is {ins['best_month']}.")
    if len(corr):
        facts.append(f"**{corr.index[0]}** tracks AQI most closely (correlation {corr.iloc[0]:.2f}), "
                     f"followed by {', '.join(corr.index[1:3])}.")
    if ins["trend"]:
        y0, y1, ch, n = ins["trend"]
        facts.append(
            f"Across the {n} cities reporting in both years, average AQI "
            f"changed by **{fmt_pct(ch)}** from {y0} to {y1}.")
    if ins["lockdown"] is not None:
        facts.append(f"During the 2020 lockdown, AQI changed by **{fmt_pct(ins['lockdown'])}** "
                     f"versus the same weeks in 2019.")
    for fact in facts:
        st.markdown(f"- {fact}")

    worst3 = ", ".join(ins["city_poor"].head(3).index)
    top_pollutant = corr.index[0] if len(corr) else "particulate matter"
    worst_season = sa.idxmax() if len(sa) else "winter"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='col-head' style='border-color:{BUCKET_COLORS["Very Poor"]}'>Risks</div>",
                    unsafe_allow_html=True)
        st.markdown(
            f"- **{worst3}** spend the largest share of days at Poor or worse, exposing residents "
            f"to sustained health risk.\n"
            f"- Pollution peaks every **{worst_season}**, so hospitals and schools face a "
            f"predictable surge in respiratory cases each year.\n"
            f"- Heavy reliance on **{top_pollutant}** means dust, burning and vehicle exhaust "
            f"control failures quickly push AQI into unsafe ranges.")
    with c2:
        st.markdown(f"<div class='col-head' style='border-color:{BUCKET_COLORS["Good"]}'>Opportunities</div>",
                    unsafe_allow_html=True)
        lock_txt = (f"The lockdown cut AQI by {abs(ins['lockdown']):.0f}%, proving large gains are "
                    f"possible when traffic and industry emissions fall."
                    if ins["lockdown"] is not None and ins["lockdown"] < 0 else
                    "Cleaner cities show which local policies are worth copying.")
        st.markdown(
            f"- {lock_txt}\n"
            f"- Seasonality is highly predictable, so interventions can be scheduled in advance "
            f"instead of reacting after AQI spikes.\n"
            f"- The next-day forecast gives authorities a one-day warning window for alerts.")
    with c3:
        st.markdown(f"<div class='col-head' style='border-color:{TEAL}'>Recommended actions</div>",
                    unsafe_allow_html=True)
        st.markdown(
            f"- Launch a **pre-{worst_season.lower()} action plan** in {worst3}: restrict "
            f"construction dust, enforce crop-residue and waste burning bans, and step up road "
            f"cleaning.\n"
            f"- Use the forecast to issue **next-day health alerts** when AQI is predicted above "
            f"200, with guidance for schools, outdoor workers and hospitals.\n"
            f"- Target **{top_pollutant}** sources first: vehicle emission checks, public transport "
            f"incentives and odd-even schemes on forecast Poor days.")

    st.markdown("<p class='footnote'>Data: Air Quality Data in India (2015 to 2020), CPCB via "
                "Kaggle. AQI categories follow the CPCB National Air Quality Index.</p>",
                unsafe_allow_html=True)
