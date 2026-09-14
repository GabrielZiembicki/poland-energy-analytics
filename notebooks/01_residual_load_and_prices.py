# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: poland-energy-analytics (3.12.4.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 1. Residual Load and Day-Ahead Prices
#
# Explore how residual load (demand minus wind and solar) drives day-ahead prices on the Polish power market.
#

# %% [markdown]
# ## Load data
#

# %%
from pathlib import Path
import sys

from elasticvolt import ElasticvoltClient
import calendar
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

for candidate_root in (Path.cwd(), Path.cwd().parent):
    if (candidate_root / "utils" / "plotting.py").exists():
        root_str = str(candidate_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        break
else:
    raise ModuleNotFoundError(
        "Could not locate utils/plotting.py from the current notebook directory."
    )

from utils.plotting import style_figure

# %%
# Constants

DATA_START_DATE = "2020-01-01"
DATA_END_DATE = "2025-12-31"
ROW_LIMIT = 1_000_000
LOCAL_TIMEZONE = "Europe/Warsaw"
RANDOM_SEED = 0
MAX_SCATTER_POINTS_PER_YEAR = 8_000
MAX_SCATTER_POINTS_SEASONAL = 30_000
LOWESS_FRAC_YEARLY = 0.3
LOWESS_FRAC_OVERALL = 0.2
BALANCING_MARKET_REFORM_DATE = pd.Timestamp("2024-06-14", tz="UTC")

# Output variables for the blog template (mirrors the capture-factor analysis).
OUTPUT_DIR = Path("./outputs/residual-load-and-prices")
INSTALLED_CAPACITY_CSV = (
    "/Users/gziembicki/repo/clarwise/clarwise/data/"
    "installed_capacity/installed_capacity_2026_03.csv"
)
TEMPLATE_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
NEG_PRICE_THRESHOLD_DEFAULT_GW = 10.0
MIDDAY_HOURS = range(11, 15)
EVENING_HOURS = range(17, 22)

# %% [markdown]
# ## Download data
#

# %%
client = ElasticvoltClient()

generation = client.get_dataset(
    "generation_by_fuel_15min",
    start_date=DATA_START_DATE,
    end_date=DATA_END_DATE,
    limit=ROW_LIMIT,
    columns=[
        "interval_start_utc",
        "biomass",
        "fossil_brown_coal_lignite",
        "fossil_coal_derived_gas",
        "fossil_gas",
        "fossil_hard_coal",
        "fossil_oil",
        "hydro_pumped_storage",
        "hydro_run_of_river_and_poundage",
        "hydro_water_reservoir",
        "other_renewable",
        "other",
    ],
)
print(
    f"Generation rows: {len(generation):,}; columns: {len(generation.columns)}"
)
generation.head()

# %%
prices = client.get_dataset(
    "prices_15min",
    start_date=DATA_START_DATE,
    end_date=DATA_END_DATE,
    limit=ROW_LIMIT,
    columns=["interval_start_utc", "rce"],
)

print(f"Prices rows: {len(prices):,}; columns: {len(prices.columns)}")
prices.head()

# %% [markdown]
# ## Build 15-minute merged dataset
#
# Merge the 15-minute generation and price series at native resolution. Residual load is
# computed directly from `generation_by_fuel_15min` as the sum of all dispatchable fuel
# categories (biomass, fossil, hydro, other) — i.e., total generation minus solar and wind —
# and converted to GW. Hour-of-day and month are taken in Europe/Warsaw local time.
#

# %%
dispatchable_cols = [
    "biomass",
    "fossil_brown_coal_lignite",
    "fossil_coal_derived_gas",
    "fossil_gas",
    "fossil_hard_coal",
    "fossil_oil",
    "hydro_pumped_storage",
    "hydro_run_of_river_and_poundage",
    "hydro_water_reservoir",
    "other_renewable",
    "other",
]


for dataset in (generation, prices):
    dataset["interval_start_utc"] = pd.to_datetime(
        dataset["interval_start_utc"], utc=True
    )

gen = generation.copy()
gen["residual_load_mw"] = gen[dispatchable_cols].fillna(0).sum(axis=1)

df = (
    gen[["interval_start_utc", "residual_load_mw"]]
    .merge(prices[["interval_start_utc", "rce"]], on="interval_start_utc")
    .dropna(subset=["residual_load_mw", "rce"])
    .sort_values("interval_start_utc")
    .reset_index(drop=True)
)
if df.empty:
    raise ValueError("Merged 15-minute dataset is empty after preprocessing.")

df["residual_load_gw"] = df["residual_load_mw"] / 1000.0

ts_local = df["interval_start_utc"].dt.tz_convert(LOCAL_TIMEZONE)
df["year"] = ts_local.dt.year
df["month"] = ts_local.dt.month
df["hour"] = ts_local.dt.hour
df["date_local"] = ts_local.dt.normalize().dt.tz_localize(None)

season_map = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
}
df["season"] = df["month"].map(season_map)

years_sorted = sorted(df["year"].unique())
print(f"15-minute observations: {len(df):,}")
print(f"Years available: {years_sorted}")
df.head()

# %% [markdown]
# ## LOWESS smoother
#
# Thin wrapper around `statsmodels.nonparametric.smoothers_lowess.lowess`. Returns the
# smoothed curve on a sorted-x grid plus an R² evaluated at the original data points.
#

# %%
from statsmodels.nonparametric.smoothers_lowess import lowess


def lowess_smooth(x, y, frac=0.3):
    """LOWESS fit. Returns (grid_x, grid_y, r2_at_data_points)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 30:
        return None, None, np.nan
    sm = lowess(y, x, frac=frac, it=1, return_sorted=True)
    grid_x, grid_y = sm[:, 0], sm[:, 1]
    fit = np.interp(x, grid_x, grid_y)
    ss_res = float(((y - fit) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return grid_x, grid_y, r2


# %% [markdown]
# ## Figure 1 - Year-by-year scatter with LOWESS fit
#
# 15-minute market price (PLN/MWh) vs residual load (GW). The
# black line is a LOWESS smoother.
#

# %%
n_years = len(years_sorted)
cols = min(3, n_years) if n_years else 1
rows = max(1, (n_years + cols - 1) // cols)
fig1 = make_subplots(
    rows=rows,
    cols=cols,
    subplot_titles=[str(y) for y in years_sorted],
    shared_yaxes=True,
    shared_xaxes=False,
    horizontal_spacing=0.04,
    vertical_spacing=0.10,
)

year_stats = []
for i, year in enumerate(years_sorted):
    row, col = i // cols + 1, i % cols + 1
    sub = df[df["year"] == year]
    sample = sub.sample(
        min(len(sub), MAX_SCATTER_POINTS_PER_YEAR), random_state=RANDOM_SEED
    )

    fig1.add_trace(
        go.Scattergl(
            x=sample["residual_load_gw"],
            y=sample["rce"],
            mode="markers",
            marker=dict(size=3, opacity=0.25, color="#0052CC"),
            showlegend=False,
            name=str(year),
        ),
        row=row,
        col=col,
    )

    ex, ey, r2 = lowess_smooth(
        sub["residual_load_gw"].values,
        sub["rce"].values,
        frac=LOWESS_FRAC_YEARLY,
    )
    if ex is not None:
        fig1.add_trace(
            go.Scatter(
                x=ex,
                y=ey,
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,
                name=f"{year} LOWESS",
            ),
            row=row,
            col=col,
        )
        fig1.add_annotation(
            text=f"R² = {r2:.2f}",
            xref=f"x{i+1} domain" if i else "x domain",
            yref=f"y{i+1} domain" if i else "y domain",
            x=0.03,
            y=0.97,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            font=dict(size=11, color="black"),
        )

    year_stats.append(
        {
            "year": int(year),
            "r2_lowess": r2,
            "median_price_pln": float(sub["rce"].median()),
            "pct_intervals_le_0": float((sub["rce"] <= 0).mean() * 100),
            "neg_price_intervals": int((sub["rce"] <= 0).sum()),
            "n_intervals": int(len(sub)),
        }
    )

fig1.update_yaxes(range=[-500, 3000])
style_figure(
    fig1,
    title="How residual load shapes Polish power prices",
    subtitle="15-min RCE vs residual load, by year (LOWESS in black)",
    height=640 * rows,
    axis_caption="Residual Load [GW]",
)
fig1.show()

# %% [markdown]
# ## Per-year statistics
#
# Compute LOWESS R², median price, % intervals at price ≤ 0, and the absolute
# count of negative-price intervals.
#

# %%
year_stats_df = (
    pd.DataFrame(year_stats).sort_values("year").reset_index(drop=True)
)
year_stats_df["r2_lowess"] = year_stats_df["r2_lowess"].round(3)
year_stats_df["median_price_pln"] = year_stats_df["median_price_pln"].round(1)
year_stats_df["pct_intervals_le_0"] = year_stats_df["pct_intervals_le_0"].round(
    2
)
year_stats_df

# %% [markdown]
# ## Overall variance explained
#
# Single-number R² for the residual-load → price relationship over the full sample, plus
# the latest year and the worst year
#

# %%
_, _, overall_r2 = lowess_smooth(
    df["residual_load_gw"].values,
    df["rce"].values,
    frac=LOWESS_FRAC_OVERALL,
)
valid_r2 = year_stats_df.dropna(subset=["r2_lowess"])
if valid_r2.empty:
    raise ValueError("LOWESS R² could not be computed for any year.")
best_year_row = valid_r2.loc[valid_r2["r2_lowess"].idxmax()]
worst_year_row = valid_r2.loc[valid_r2["r2_lowess"].idxmin()]
latest_year_row = year_stats_df.iloc[-1]

print(
    f"Overall LOWESS R² (all years pooled): {overall_r2:.3f}"
    f"  → residual load alone explains ~{overall_r2*100:.0f}% of hourly price variance"
)
print(
    f'Worst year:   {int(worst_year_row["year"])}  R² = {worst_year_row["r2_lowess"]:.2f}'
)
print(
    f'Latest year:  {int(latest_year_row["year"])}  R² = {latest_year_row["r2_lowess"]:.2f}'
)
print(
    f'Best year:    {int(best_year_row["year"])}  R² = {best_year_row["r2_lowess"]:.2f}'
)

# %% [markdown]
# ## Negative and zero prices
#
# 15-min intervals clearing at zero or below — concentrated at the low-residual-load tail and
# growing year-over-year as RES penetration rises.
#

# %%
neg = year_stats_df[
    ["year", "neg_price_intervals", "pct_intervals_le_0"]
].copy()
neg.columns = ["year", "intervals_le_0", "pct_intervals_le_0"]
print(neg.to_string(index=False))

low_resid_threshold = 10.0  # GW
low = df[df["residual_load_gw"] < low_resid_threshold]
if len(low):
    print(
        f"\nBelow {low_resid_threshold:.0f} GW residual load ({len(low):,} intervals): "
        f'{(low["rce"] <= 0).mean()*100:.1f}% clear ≤ 0, '
        f'mean price {low["rce"].mean():.1f} PLN/MWh.'
    )

# %% [markdown]
# ## Figure 2: Seasonal view (Winter vs Summer)
#
# Same scatter restricted to Winter and Summer with per-season LOWESS curves overlaid. The
# winter range extends further to the right (heating); the summer curve sits visibly above
# winter at equivalent residual loads.
#

# %%
df_post_reform = df[
    df["interval_start_utc"] >= BALANCING_MARKET_REFORM_DATE
].copy()
df_ws = df_post_reform[
    df_post_reform["season"].isin(["Winter", "Summer"])
].copy()

season_colors = {"Winter": "#0052CC", "Summer": "#FFA600"}

sample = df_ws.sample(
    min(len(df_ws), MAX_SCATTER_POINTS_SEASONAL), random_state=RANDOM_SEED
)
fig2 = px.scatter(
    sample,
    x="residual_load_gw",
    y="rce",
    color="season",
    color_discrete_map=season_colors,
    labels={
        "residual_load_gw": "Residual Load [GW]",
        "rce": "Market price [PLN/MWh]",
        "season": "Season",
    },
)
fig2.update_traces(mode="markers", marker=dict(size=4, opacity=0.25))

for season in ["Winter", "Summer"]:
    sub = df_ws[df_ws["season"] == season]
    ex, ey, _ = lowess_smooth(
        sub["residual_load_gw"].values,
        sub["rce"].values,
        frac=LOWESS_FRAC_YEARLY,
    )
    if ex is not None:
        fig2.add_trace(
            go.Scatter(
                x=ex,
                y=ey,
                mode="lines",
                line=dict(color=season_colors[season], width=3),
                name=f"{season} LOWESS",
                showlegend=True,
            )
        )

style_figure(
    fig2,
    title="Winter and summer face different stacks",
    subtitle="Residual load vs RCE - Winter vs Summer",
    height=620,
)
fig2.update_yaxes(range=[-500, 2000])
fig2.show()

for season in ["Winter", "Summer"]:
    sub = df[df["season"] == season]
    print(
        f"{season}: residual P01-P99 = "
        f'[{sub["residual_load_gw"].quantile(0.01):.1f}, '
        f'{sub["residual_load_gw"].quantile(0.99):.1f}] GW, '
        f'% intervals <= 0: {(sub["rce"] <= 0).mean()*100:.2f}%, '
        f'median price: {sub["rce"].median():.1f}'
    )


# %% [markdown]
# ## Figure 3: Sensitivity surface (∂Price / ∂ResidualLoad)
#
# Local linear slope of price on residual load fitted _within each (month, hour) cell_ across
# all available years. Units: PLN/MWh per GW. Cells with fewer than 50 observations are
# masked.
#


# %%
def cell_slope(group, xcol="residual_load_gw", ycol="rce", min_n=50):
    x = group[xcol].to_numpy(dtype=float)
    y = group[ycol].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < min_n:
        return np.nan
    lo, hi = np.quantile(x, [0.005, 0.995])
    mid_mask = (x >= lo) & (x <= hi)
    x, y = x[mid_mask], y[mid_mask]
    if len(x) < min_n or x.std() == 0:
        return np.nan
    slope, _intercept = np.polyfit(x, y, 1)
    return float(slope)


def build_sensitivity_surface(data, min_n=50):
    rows = []
    for (month, hour), group in data.groupby(["month", "hour"]):
        rows.append(
            {
                "month": int(month),
                "hour": int(hour),
                "slope_pln_per_gw": cell_slope(group, min_n=min_n),
                "n": len(group),
                "mean_residual_gw": float(group["residual_load_gw"].mean()),
            }
        )
    return (
        pd.DataFrame(rows).sort_values(["month", "hour"]).reset_index(drop=True)
    )


# Restrict to the period where the system is structurally comparable to today
df_sens = df[df["interval_start_utc"] >= BALANCING_MARKET_REFORM_DATE]
print(
    f"Sensitivity surface fitted on {len(df_sens):,} intervals "
    f'from {df_sens["interval_start_utc"].min().date()} '
    f'to {df_sens["interval_start_utc"].max().date()}.'
)

sens = build_sensitivity_surface(df_sens, min_n=25)
month_names = [calendar.month_name[m] for m in range(1, 13)]
heat = sens.pivot(
    index="month", columns="hour", values="slope_pln_per_gw"
).reindex(
    index=range(1, 13),
    columns=range(0, 24),
)

fig3 = go.Figure(
    data=go.Heatmap(
        z=heat.values,
        x=list(heat.columns),
        y=[month_names[m - 1] for m in heat.index],
        colorscale="YlGnBu",
        zmin=10,
        zmax=160,
        colorbar=dict(
            title=(
                "Sensitivity<br>"
                "<span style='font-size:11px;font-style:italic'>[PLN/MWh per GW]</span>"
            ),
            ticks="outside",
        ),
        xgap=1,
        ygap=1,
    )
)
fig3.update_layout(margin=dict(t=100, b=80, l=80, r=40))
fig3.update_xaxes(
    title="Hour of day",
    tickmode="array",
    tickvals=[0, 4, 8, 12, 16, 20],
    showgrid=False,
)
fig3.update_yaxes(autorange="reversed", showgrid=False)
style_figure(
    fig3,
    title="When does residual load move prices the most?",
    subtitle="Estimated price sensitivity [PLN/MWh per GW] - month x hour",
    height=520,
    source_y=-0.14,
)
fig3.show()

heat.round(0)

# %%
df_sens.groupby("year").size()


# %% [markdown]
# ## Sensitivity table: key regimes
#
# Pulls out the specific (month, hour) cells the blog quotes.
#


# %%
def sens_at(month, hour):
    row = sens[(sens["month"] == month) & (sens["hour"] == hour)]
    if row.empty:
        return np.nan
    return float(row["slope_pln_per_gw"].iloc[0])


regimes = pd.DataFrame(
    [
        ("Summer midday trough", 7, 12),
        ("Summer evening peak", 7, 19),
        ("Winter morning ramp", 1, 7),
        ("Winter evening peak", 1, 18),
        ("Summer night trough", 7, 3),
        ("Shoulder afternoon", 4, 15),
    ],
    columns=["regime", "month", "hour"],
)
regimes["sensitivity_pln_mwh_per_gw"] = regimes.apply(
    lambda r: sens_at(int(r["month"]), int(r["hour"])),
    axis=1,
).round(0)
regimes


# %% [markdown]
# ## Marginal sensitivity by time-of-day band
#
# Average sensitivity in the midday solar window (10–15), evening peak (17–21), and night
# trough (00–05).
#


# %%
def band_mean(months, hours):
    sel = sens[sens["month"].isin(months) & sens["hour"].isin(hours)]
    if sel.empty:
        return np.nan
    return float(sel["slope_pln_per_gw"].mean())


all_months = list(range(1, 13))
summer_months = [6, 7, 8]
winter_months = [12, 1, 2]

bands = pd.DataFrame(
    [
        ("All year", "Midday (10–15)", band_mean(all_months, range(10, 16))),
        ("All year", "Evening (17–21)", band_mean(all_months, range(17, 22))),
        ("All year", "Night (00–05)", band_mean(all_months, range(0, 6))),
        ("Summer", "Midday (10–15)", band_mean(summer_months, range(10, 16))),
        ("Summer", "Evening (17–21)", band_mean(summer_months, range(17, 22))),
        ("Winter", "Evening (17–21)", band_mean(winter_months, range(17, 22))),
    ],
    columns=["season", "band", "avg_sensitivity_pln_per_gw"],
)
bands["avg_sensitivity_pln_per_gw"] = bands["avg_sensitivity_pln_per_gw"].round(
    0
)
bands

# %% [markdown]
# ## Out-of-sample validation
#
# Train the sensitivity surface on all-but-the-last available year, evaluate on the held-out
# last year, and report the cell-wise correlation.
#

# %%
if len(years_sorted) >= 2:
    train_years = years_sorted[:-1]
    test_year = years_sorted[-1]

    train_surf = build_sensitivity_surface(
        df[df["year"].isin(train_years)],
        min_n=20,
    ).rename(columns={"slope_pln_per_gw": "train"})
    test_surf = build_sensitivity_surface(
        df[df["year"] == test_year],
        min_n=20,
    ).rename(columns={"slope_pln_per_gw": "test"})

    merged = (
        train_surf[["month", "hour", "train"]]
        .merge(
            test_surf[["month", "hour", "test"]],
            on=["month", "hour"],
        )
        .dropna()
    )
    corr = float(merged["train"].corr(merged["test"]))

    print(f"Train years: {train_years}")
    print(f"Test year:   {test_year}")
    print(f"Cells compared: {len(merged)} / 288")
    print(f"Cell-wise correlation (train vs test slopes): {corr:.3f}")
    merged.head()
else:
    print("Need at least 2 years of data for out-of-sample validation.")

# %% [markdown]
# ## BESS value concentration
#
# Approximation: hourly arbitrage value ≈ |price − daily mean|. Compute what share of total
# annual arbitrage value is concentrated in the most extreme hours of the year.
#

# %%
df_arb = df.copy()
daily_mean = df_arb.groupby("date_local")["rce"].transform("mean")
df_arb["abs_spread"] = (df_arb["rce"] - daily_mean).abs()
total_value = float(df_arb["abs_spread"].sum())
if total_value <= 0:
    raise ValueError(
        "Total arbitrage proxy value is zero; cannot compute concentration shares."
    )

rows__ = []
for top_pct in [1, 5, 10, 20, 50]:
    thr = df_arb["abs_spread"].quantile(1 - top_pct / 100.0)
    sub = df_arb[df_arb["abs_spread"] >= thr]
    rows__.append(
        {
            "top_pct_hours": top_pct,
            "share_of_total_value_pct": round(
                sub["abs_spread"].sum() / total_value * 100, 1
            ),
            "min_abs_spread_pln": round(float(thr), 1),
        }
    )
concentration = pd.DataFrame(rows__)
concentration

# %% [markdown]
# ## Template values
#

# %%
print("=== TL;DR figures ===")
print(f"Overall R² (all years pooled):        {overall_r2*100:.0f}%")
print(
    f'Worst year R² ({int(worst_year_row["year"])}):                {worst_year_row["r2_lowess"]:.2f}'
)
print(
    f'Latest year R² ({int(latest_year_row["year"])}):               {latest_year_row["r2_lowess"]:.2f}'
)
print()
print("Sensitivity range (PLN/MWh per GW):")
print(
    f'  min cell: {sens["slope_pln_per_gw"].min():.0f}'
    f'  median: {sens["slope_pln_per_gw"].median():.0f}'
    f'  max cell: {sens["slope_pln_per_gw"].max():.0f}'
)
print()
print("Negative-price intervals by year:")
for _, row in year_stats_df.iterrows():
    print(
        f'  {int(row["year"])}: {int(row["neg_price_intervals"]):>5d} intervals '
        f'({row["pct_intervals_le_0"]:.2f}%)  median price {row["median_price_pln"]:.1f} PLN/MWh'
    )
print()
print("Seasonal residual-load range (P01–P99, GW):")
for s in ["Winter", "Summer"]:
    sub = df[df["season"] == s]
    print(
        f'  {s}: [{sub["residual_load_gw"].quantile(0.01):.1f}, '
        f'{sub["residual_load_gw"].quantile(0.99):.1f}]'
    )

# %% [markdown]
# ## Output variables for the blog template
#
# Assemble the `variables.json` consumed by the landing-page template, mirroring the
# capture-factor analysis. `TEMPLATE_VARS` documents every key the post expects; the
# `variables` dict below holds the computed values, and a sanity check asserts the two
# key sets match exactly.
#

# %%
TEMPLATE_VARS = {
    # ============================================================
    # Year-by-year statistics — used in TL;DR and the table after Figure 1
    # ============================================================
    # R² of the LOWESS smoother fit of price on residual load, within each year.
    "R2_2020": "R² of the smoothed price ~ residual-load fit, 2020. Format: 0.XX.",
    "R2_2021": "R² of the smoothed price ~ residual-load fit, 2021. Format: 0.XX.",
    "R2_2022": "R² of the smoothed price ~ residual-load fit, 2022 (lowest in series — fuel crisis). Format: 0.XX.",
    "R2_2023": "R² of the smoothed price ~ residual-load fit, 2023. Format: 0.XX.",
    "R2_2024": "R² of the smoothed price ~ residual-load fit, 2024. Format: 0.XX.",
    "R2_2025": "R² of the smoothed price ~ residual-load fit, 2025. Format: 0.XX. Used in TL;DR — should reflect recovery.",
    # Median of all hourly day-ahead wholesale prices (RCE) in the given year, PLN/MWh.
    "MEDIAN_2020": "Median hourly day-ahead price for 2020, PLN/MWh.",
    "MEDIAN_2021": "Median hourly day-ahead price for 2021, PLN/MWh.",
    "MEDIAN_2022": "Median hourly day-ahead price for 2022, PLN/MWh (highest of series).",
    "MEDIAN_2023": "Median hourly day-ahead price for 2023, PLN/MWh.",
    "MEDIAN_2024": "Median hourly day-ahead price for 2024, PLN/MWh.",
    "MEDIAN_2025": "Median hourly day-ahead price for 2025, PLN/MWh.",
    # Mean of all hourly day-ahead wholesale prices (RCE) in the given year, PLN/MWh.
    # Sensitive to the negative-price tail and crisis spikes — read alongside the median.
    "MEAN_2020": "Mean hourly day-ahead price for 2020, PLN/MWh.",
    "MEAN_2021": "Mean hourly day-ahead price for 2021, PLN/MWh.",
    "MEAN_2022": "Mean hourly day-ahead price for 2022, PLN/MWh (highest of series).",
    "MEAN_2023": "Mean hourly day-ahead price for 2023, PLN/MWh.",
    "MEAN_2024": "Mean hourly day-ahead price for 2024, PLN/MWh.",
    "MEAN_2025": "Mean hourly day-ahead price for 2025, PLN/MWh.",
    # Hours in the year with day-ahead price ≤ 0 PLN/MWh (integer count).
    "NEG_HOURS_2020": "Hours with price ≤ 0 PLN/MWh, 2020. Likely 0 or very few.",
    "NEG_HOURS_2021": "Hours with price ≤ 0 PLN/MWh, 2021.",
    "NEG_HOURS_2022": "Hours with price ≤ 0 PLN/MWh, 2022.",
    "NEG_HOURS_2023": "Hours with price ≤ 0 PLN/MWh, 2023.",
    "NEG_HOURS_2024": "Hours with price ≤ 0 PLN/MWh, 2024. Also used in TL;DR.",
    "NEG_HOURS_2025": "Hours with price ≤ 0 PLN/MWh, full year 2025. Used in TL;DR, year-by-year table, AND the negative-prices section narrative. Original post quoted '>350' mid-year.",
    # Installed RES capacity (wind + solar), year-end value in GW.
    "RES_2020": "Installed RES capacity (wind + solar), year-end 2020, GW.",
    "RES_2021": "Installed RES capacity (wind + solar), year-end 2021, GW.",
    "RES_2022": "Installed RES capacity (wind + solar), year-end 2022, GW.",
    "RES_2023": "Installed RES capacity (wind + solar), year-end 2023, GW.",
    "RES_2024": "Installed RES capacity (wind + solar), year-end 2024, GW.",
    "RES_2025": "Installed RES capacity (wind + solar), year-end 2025, GW.",
    # ============================================================
    # Negative-prices section
    # ============================================================
    "NEG_PRICE_THRESHOLD": "Residual load (GW) below which the trend line bends downward into negative-price territory. ~10 GW in the original post.",
    # ============================================================
    # Seasonal section
    # ============================================================
    "WINTER_P99": "99th percentile of residual load on winter weekdays, GW. ~24 GW in original prose, refine to actual P99.",
    "SUMMER_P99": "99th percentile of residual load on summer weekdays, GW. ~20 GW in original prose, refine to actual P99.",
    # ============================================================
    # Sensitivity surface section (Figure 3 heatmap), PLN/MWh per GW
    # ============================================================
    "SENSITIVITY_MIN": "Lowest month × hour sensitivity on the heatmap (PLN/MWh per GW). Typically summer nights.",
    "SENSITIVITY_MAX": "Highest month × hour sensitivity on the heatmap (PLN/MWh per GW). Typically winter evening peak.",
    "SENSITIVITY_MIDDAY": "Sensitivity (PLN/MWh per GW) in the darkest midday cell of the heatmap — the solar-dip core.",
    "PEAK_SENSITIVITY_MONTH": "Calendar month name where the midday sensitivity peaks. E.g. 'June' or 'July'.",
    "SENSITIVITY_EVENING": "Sensitivity (PLN/MWh per GW) in the evening peak band of the heatmap.",
    "SENSITIVITY_RATIO": "Ratio of SENSITIVITY_MAX to SENSITIVITY_MIN. Dimensionless. Used as 'on the order of {X}×'.",
    # ============================================================
    # Methodology section
    # ============================================================
    "LOWESS_BW": "LOWESS bandwidth parameter used in the year-by-year fit. Format depending on library: e.g. 'frac=0.3' or '0.3'.",
}


# %%
def _round_or_none(value, ndigits):
    """Round a scalar, mapping NaN/None to JSON null."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return round(float(value), ndigits)


# Year-by-year R² and median price.
# R² comes from the per-year LOWESS table (15-min resolution). Median is taken on
# hourly-mean prices so it matches the "hourly day-ahead price" framing.
r2_by_year = dict(zip(year_stats_df["year"], year_stats_df["r2_lowess"]))

hourly_price = (
    df[["interval_start_utc", "rce"]]
    .dropna()
    .set_index("interval_start_utc")
    .resample("1h")["rce"]
    .mean()
    .dropna()
)
hourly_price = pd.DataFrame(
    {
        "rce": hourly_price.to_numpy(),
        "year": hourly_price.index.tz_convert(LOCAL_TIMEZONE).year,
    }
)
years_with_data = set(int(y) for y in hourly_price["year"].unique())
median_by_year = hourly_price.groupby("year")["rce"].median().to_dict()
mean_by_year = hourly_price.groupby("year")["rce"].mean().to_dict()
neg_hours_by_year = (
    hourly_price[hourly_price["rce"] <= 0].groupby("year").size().to_dict()
)

# Installed RES capacity (year-end wind + solar, GW) from the shared capacity CSV.
installed_capacity = pd.read_csv(INSTALLED_CAPACITY_CSV)
installed_capacity["date"] = pd.to_datetime(installed_capacity["date"])
installed_capacity["cap_year"] = installed_capacity["date"].dt.year
res_by_year_gw = (
    installed_capacity.sort_values("date")
    .groupby("cap_year")[["solar", "wind"]]
    .last()
    .sum(axis=1)
    .div(1000.0)  # MW → GW
    .to_dict()
)

# Negative-price threshold: residual load below which the pooled LOWESS curve bends
# into negative territory (highest grid point where smoothed price ≤ 0).
grid_x, grid_y, _ = lowess_smooth(
    df["residual_load_gw"].values,
    df["rce"].values,
    frac=LOWESS_FRAC_OVERALL,
)
if grid_x is not None and np.any(grid_y <= 0):
    neg_price_threshold = round(float(grid_x[grid_y <= 0].max()), 1)
else:
    neg_price_threshold = NEG_PRICE_THRESHOLD_DEFAULT_GW

# Seasonal residual-load P99 on weekdays (Mon–Fri, Europe/Warsaw).
weekday_mask = (
    df["interval_start_utc"].dt.tz_convert(LOCAL_TIMEZONE).dt.dayofweek < 5
)
df_weekday = df[weekday_mask]
winter_p99 = _round_or_none(
    df_weekday.loc[
        df_weekday["season"] == "Winter", "residual_load_gw"
    ].quantile(0.99),
    1,
)
summer_p99 = _round_or_none(
    df_weekday.loc[
        df_weekday["season"] == "Summer", "residual_load_gw"
    ].quantile(0.99),
    1,
)

# Sensitivity-surface anchors (PLN/MWh per GW) from the Figure-3 surface `sens`.
sens_valid = sens.dropna(subset=["slope_pln_per_gw"])
sensitivity_min = round(float(sens_valid["slope_pln_per_gw"].min()))
sensitivity_max = round(float(sens_valid["slope_pln_per_gw"].max()))
sensitivity_ratio = (
    round(sensitivity_max / sensitivity_min, 1) if sensitivity_min else None
)
# Midday "dark core": the steepest (most extreme) midday cell and its month.
midday_cells = sens_valid[sens_valid["hour"].isin(MIDDAY_HOURS)]
midday_core = midday_cells.loc[midday_cells["slope_pln_per_gw"].idxmax()]
sensitivity_midday = round(float(midday_core["slope_pln_per_gw"]))
peak_sensitivity_month = calendar.month_name[int(midday_core["month"])]
# Evening band: mean sensitivity across the evening peak hours, all months.
sensitivity_evening = round(float(band_mean(all_months, EVENING_HOURS)))

# %%
variables: dict = {}

for _y in TEMPLATE_YEARS:
    variables[f"R2_{_y}"] = _round_or_none(r2_by_year.get(_y), 2)
for _y in TEMPLATE_YEARS:
    variables[f"MEDIAN_{_y}"] = _round_or_none(median_by_year.get(_y), 1)
for _y in TEMPLATE_YEARS:
    variables[f"MEAN_{_y}"] = _round_or_none(mean_by_year.get(_y), 1)
for _y in TEMPLATE_YEARS:
    variables[f"NEG_HOURS_{_y}"] = (
        int(neg_hours_by_year.get(_y, 0)) if _y in years_with_data else None
    )
for _y in TEMPLATE_YEARS:
    variables[f"RES_{_y}"] = _round_or_none(res_by_year_gw.get(_y), 1)

variables.update(
    {
        # Negative prices
        "NEG_PRICE_THRESHOLD": neg_price_threshold,
        # Seasonal
        "WINTER_P99": winter_p99,
        "SUMMER_P99": summer_p99,
        # Sensitivity surface
        "SENSITIVITY_MIN": sensitivity_min,
        "SENSITIVITY_MAX": sensitivity_max,
        "SENSITIVITY_MIDDAY": sensitivity_midday,
        "PEAK_SENSITIVITY_MONTH": peak_sensitivity_month,
        "SENSITIVITY_EVENING": sensitivity_evening,
        "SENSITIVITY_RATIO": sensitivity_ratio,
        # Methodology
        "LOWESS_BW": LOWESS_FRAC_YEARLY,
    }
)

# Keep the computed values and the documented spec in lock-step.
_missing = set(TEMPLATE_VARS) - set(variables)
_extra = set(variables) - set(TEMPLATE_VARS)
assert not _missing and not _extra, f"missing={_missing} extra={_extra}"

# %%
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
variables_path = OUTPUT_DIR / "variables.json"
with open(variables_path, "w") as f:
    json.dump(variables, f, indent=2, default=str)

print(f"Wrote {len(variables)} variables → {variables_path}")
print(json.dumps(variables, indent=2, default=str))
