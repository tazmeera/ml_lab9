"""
Lab Sheet-09 | Project 1: COVID-19 Data Analysis Dashboard
MCA III Semester (Session 2026-2027)

Steps
  1. Load data + EDA (shape, dtypes, describe, missing-value report)
  2. Clean data (negative corrections, missing values) - every decision is logged
  3. Static charts (Matplotlib / Seaborn)  -> charts/*.png
  4. Interactive dashboard (Plotly)        -> covid_dashboard.html
  5. Auto-generated findings               -> findings.md, eda_report.txt

Run:  python generate_covid_dataset.py   (once, creates covid_data.csv)
      python covid_dashboard.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.join(HERE, "charts")
os.makedirs(CH, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
log = []          # EDA / cleaning log written to eda_report.txt


def say(text=""):
    print(text)
    log.append(str(text))


# =============================================================================
# 1. LOAD + EDA
# =============================================================================
raw = pd.read_csv(os.path.join(HERE, "covid_data.csv"), parse_dates=["date"])
say("=" * 70 + "\n1. EXPLORATORY DATA ANALYSIS\n" + "=" * 70)
say(f"Shape: {raw.shape[0]:,} rows x {raw.shape[1]} columns")
say(f"Period: {raw.date.min().date()} -> {raw.date.max().date()} | Countries: {raw.country.nunique()}")
say("\nData types:\n" + raw.dtypes.to_string())
say("\nSummary statistics:\n" + raw.describe().T.round(1).to_string())
miss = pd.DataFrame({"missing": raw.isna().sum(), "pct": (raw.isna().mean() * 100).round(2)})
say("\nMissing values BEFORE cleaning:\n" + miss[miss.missing > 0].to_string())
neg_n = int((raw.new_cases < 0).sum())
say(f"\nNegative new_cases (data corrections): {neg_n}")
say(f"Duplicate (country,date) rows: {int(raw.duplicated(['country', 'date']).sum())}")

# =============================================================================
# 2. CLEANING
# =============================================================================
df = raw.sort_values(["country", "date"]).reset_index(drop=True)
g = df.groupby("country")

say("\n" + "=" * 70 + "\n2. DATA CLEANING DECISIONS\n" + "=" * 70)
# a) negative daily counts are reporting corrections -> treat as missing
for c in ["new_cases", "new_deaths", "new_recoveries"]:
    df.loc[df[c] < 0, c] = np.nan
say(f"- Negative daily values set to NaN: {neg_n} new_cases rows")

# b) daily counts: linear interpolation inside each country, remaining gaps -> 0
for c in ["new_cases", "new_deaths", "new_recoveries"]:
    df[c] = df.groupby("country")[c].transform(lambda s: s.interpolate(limit_direction="both")).fillna(0).round()
say("- Daily counts (new_*): linear interpolation within each country; leftover gaps = 0")

# c) cumulative columns never decrease -> forward fill within country, 0 before first value
for c in ["total_recoveries", "people_vaccinated", "people_fully_vaccinated", "total_vaccinations"]:
    df[c] = df.groupby("country")[c].transform(lambda s: s.ffill()).fillna(0)
say("- Cumulative columns: forward-fill within country (values cannot fall); 0 before first report")
say("- UK stopped reporting recoveries in Feb-2022 -> carried forward (recovery totals for UK are a LOWER BOUND)")

# d) derived metrics
df["cases_7d"] = df.groupby("country").new_cases.transform(lambda s: s.rolling(7, min_periods=1).mean())
df["deaths_7d"] = df.groupby("country").new_deaths.transform(lambda s: s.rolling(7, min_periods=1).mean())
df["vacc_pct"] = df.people_fully_vaccinated / df.population * 100
df["cases_per_million"] = df.total_cases / df.population * 1e6
df["deaths_per_million"] = df.total_deaths / df.population * 1e6
df["month"] = df.date.dt.to_period("M").dt.to_timestamp()
assert df[["new_cases", "new_deaths", "new_recoveries", "total_recoveries", "people_vaccinated"]].isna().sum().sum() == 0
say(f"\nMissing values AFTER cleaning: {int(df.isna().sum().sum())}")
df.to_csv(os.path.join(HERE, "covid_data_clean.csv"), index=False)

latest = df[df.date == df.date.max()].set_index("country")
glob = df.groupby("date")[["new_cases", "new_deaths", "new_recoveries", "total_cases", "total_deaths",
                           "total_recoveries", "people_fully_vaccinated", "population"]].sum()
glob["cases_7d"] = glob.new_cases.rolling(7, min_periods=1).mean()
glob["deaths_7d"] = glob.new_deaths.rolling(7, min_periods=1).mean()

# =============================================================================
# 3. STATIC CHARTS
# =============================================================================
def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(CH, name), dpi=150)
    plt.close()


# 1 global trend (line - change over time)
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.fill_between(glob.index, glob.cases_7d, color="#4c78a8", alpha=0.25)
ax.plot(glob.index, glob.cases_7d, color="#4c78a8", label="New cases (7-day avg)")
ax2 = ax.twinx()
ax2.plot(glob.index, glob.deaths_7d, color="#e45756", label="New deaths (7-day avg)")
ax.set(title="Global daily cases and deaths (20 countries)", ylabel="Cases / day")
ax2.set_ylabel("Deaths / day"); ax2.grid(False)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left")
save("01_global_trend.png")

# 2 confirmed vs recovered vs deaths (cumulative)
fig, ax = plt.subplots(figsize=(11, 4.5))
for c, col, lab in [("total_cases", "#4c78a8", "Confirmed"), ("total_recoveries", "#54a24b", "Recovered"),
                    ("total_deaths", "#e45756", "Deaths")]:
    ax.plot(glob.index, glob[c] / 1e6, color=col, label=lab, lw=2)
ax.set(title="Cumulative confirmed, recovered and deaths", ylabel="Million people"); ax.legend()
save("02_cumulative_confirmed_recovered_deaths.png")

# 3 top-10 countries (bar - ranking)
top = latest.total_cases.sort_values(ascending=False).head(10) / 1e6
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=top.values, y=top.index, color="#4c78a8", ax=ax)
for i, v in enumerate(top.values):
    ax.text(v, i, f" {v:.1f}M", va="center", fontsize=9)
ax.set(title="Top 10 countries by confirmed cases", xlabel="Million cases", ylabel="")
save("03_top10_countries_cases.png")

# 4 per-capita view
pc = latest[["cases_per_million", "deaths_per_million"]].sort_values("cases_per_million", ascending=False)
fig, ax = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
sns.barplot(x=pc.cases_per_million / 1000, y=pc.index, color="#4c78a8", ax=ax[0])
sns.barplot(x=pc.deaths_per_million, y=pc.index, color="#e45756", ax=ax[1])
ax[0].set(title="Cases per 1,000 people", xlabel="", ylabel="")
ax[1].set(title="Deaths per million people", xlabel="", ylabel="")
save("04_per_capita_comparison.png")

# 5 vaccination progress (multi-line)
fig, ax = plt.subplots(figsize=(11, 5))
sel = latest.vacc_pct.sort_values(ascending=False).iloc[[0, 1, 4, 8, 12, 16, 19]].index
for c in sel:
    d = df[df.country == c]
    ax.plot(d.date, d.vacc_pct, label=c, lw=2)
ax.set(title="Fully vaccinated share of population", ylabel="% of population"); ax.legend(ncol=2)
save("05_vaccination_trends.png")

# 6 heatmap - monthly cases per million, top countries
top_c = latest.total_cases.sort_values(ascending=False).head(10).index
mm = (df[df.country.isin(top_c)].groupby(["country", "month"]).new_cases.sum().unstack("month")
      .div(latest.population[top_c], axis=0) * 1e6).loc[top_c]
fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(mm, cmap="YlOrRd", ax=ax, cbar_kws={"label": "New cases per million / month"},
            xticklabels=[d.strftime("%b-%y") if i % 3 == 0 else "" for i, d in enumerate(mm.columns)])
ax.set(title="Monthly new cases per million - top 10 countries (waves visible as dark bands)", xlabel="", ylabel="")
save("06_monthly_heatmap.png")

# 7 continent comparison (box plot)
fig, ax = plt.subplots(figsize=(9, 5))
cont = latest.reset_index()
sns.boxplot(data=cont, x="continent", y="deaths_per_million", color="#9ecae9", ax=ax)
sns.stripplot(data=cont, x="continent", y="deaths_per_million", color="#1f3b73", size=6, ax=ax)
ax.set(title="Deaths per million by continent", xlabel="", ylabel="Deaths per million")
plt.xticks(rotation=20)
save("07_continent_boxplot.png")

# 8 case-fatality rate over time
m = glob.resample("ME").sum(numeric_only=True)
cfr = (m.new_deaths / m.new_cases.shift(1) * 100).replace([np.inf, -np.inf], np.nan)
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(cfr.index, cfr.values, marker="o", color="#b279a2")
ax.set(title="Monthly case fatality rate (deaths / previous month's cases)", ylabel="%")
save("08_case_fatality_rate.png")

# 9 vaccination vs. death-rate scatter
sc = latest.reset_index()
fig, ax = plt.subplots(figsize=(9, 5.5))
sns.scatterplot(data=sc, x="vacc_pct", y="deaths_per_million", hue="continent", size="population",
                sizes=(60, 600), alpha=.8, ax=ax)
for _, r in sc.iterrows():
    ax.annotate(r.country, (r.vacc_pct, r.deaths_per_million), fontsize=7, xytext=(4, 4), textcoords="offset points")
ax.set(title="Vaccination coverage vs cumulative deaths per million", xlabel="Fully vaccinated (%)")
ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
save("09_vaccination_vs_deaths.png")

# 10 correlation heatmap
cols = ["new_cases", "new_deaths", "new_recoveries", "people_fully_vaccinated", "total_vaccinations"]
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation between key variables")
save("10_correlation_heatmap.png")
say(f"\nSaved 10 static charts in {CH}")

# =============================================================================
# 4. INTERACTIVE DASHBOARD (Plotly)
# =============================================================================
ISO = {"India": "IND", "USA": "USA", "Brazil": "BRA", "Russia": "RUS", "France": "FRA", "UK": "GBR",
       "Germany": "DEU", "Italy": "ITA", "Spain": "ESP", "Turkey": "TUR", "Japan": "JPN",
       "South Korea": "KOR", "Indonesia": "IDN", "Mexico": "MEX", "Argentina": "ARG",
       "South Africa": "ZAF", "Nigeria": "NGA", "Egypt": "EGY", "Australia": "AUS", "Canada": "CAN"}
LAY = dict(template="plotly_white", margin=dict(l=40, r=20, t=60, b=40), height=430,
           font=dict(family="Segoe UI, Arial", size=13))

# A - global timeline with range slider
fA = make_subplots(specs=[[{"secondary_y": True}]])
fA.add_trace(go.Scatter(x=glob.index, y=glob.cases_7d, name="New cases (7d avg)", fill="tozeroy",
                        line=dict(color="#4c78a8")), secondary_y=False)
fA.add_trace(go.Scatter(x=glob.index, y=glob.deaths_7d, name="New deaths (7d avg)",
                        line=dict(color="#e45756")), secondary_y=True)
fA.update_layout(title="Global waves - cases and deaths", xaxis=dict(rangeslider=dict(visible=True)),
                 legend=dict(orientation="h", y=1.12), **LAY)
fA.update_yaxes(title_text="Cases / day", secondary_y=False); fA.update_yaxes(title_text="Deaths / day", secondary_y=True)

# B - country explorer with dropdown
countries = sorted(df.country.unique())
fB = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.6, .4], vertical_spacing=.06,
                   subplot_titles=("Daily new cases (7d avg)", "Fully vaccinated (%) and deaths (7d avg)"))
for i, c in enumerate(countries):
    d = df[df.country == c]
    vis = i == 0
    fB.add_trace(go.Scatter(x=d.date, y=d.cases_7d, name="Cases", line=dict(color="#4c78a8"), visible=vis), 1, 1)
    fB.add_trace(go.Scatter(x=d.date, y=d.deaths_7d, name="Deaths", line=dict(color="#e45756"), visible=vis), 2, 1)
    fB.add_trace(go.Scatter(x=d.date, y=d.vacc_pct, name="Vaccinated %", line=dict(color="#54a24b", dash="dot"),
                            yaxis="y3", visible=vis), 2, 1)
buttons = []
for i, c in enumerate(countries):
    v = [False] * (3 * len(countries)); v[3 * i:3 * i + 3] = [True] * 3
    buttons.append(dict(label=c, method="update", args=[{"visible": v}, {"title": f"Country explorer - {c}"}]))
fB.update_layout(title=f"Country explorer - {countries[0]}", updatemenus=[dict(buttons=buttons, x=0, y=1.25, xanchor="left")],
                 yaxis3=dict(overlaying="y2", side="right", title="Vaccinated %", range=[0, 100], showgrid=False),
                 legend=dict(orientation="h", y=-0.1), **{**LAY, "height": 520})

# C - top countries per-capita (bar)
pcs = latest.sort_values("cases_per_million", ascending=True).tail(15)
fC = go.Figure(go.Bar(x=pcs.cases_per_million / 1000, y=pcs.index, orientation="h", marker_color="#4c78a8",
                      hovertemplate="%{y}: %{x:.0f} cases per 1,000 people<extra></extra>"))
fC.update_layout(title="Cases per 1,000 people (top 15)", xaxis_title="Cases per 1,000", **LAY)

# D - vaccination curves
fD = go.Figure()
for c in countries:
    d = df[df.country == c]
    fD.add_trace(go.Scatter(x=d.date, y=d.vacc_pct, name=c, mode="lines", line=dict(width=2)))
fD.update_layout(title="Vaccination rollout (click legend to isolate countries)", yaxis_title="% fully vaccinated", **LAY)

# E - choropleth
ll = latest.reset_index()
ll["iso"] = ll.country.map(ISO)
fE = go.Figure(go.Choropleth(locations=ll.iso, z=ll.deaths_per_million, text=ll.country, colorscale="Reds",
                             colorbar_title="Deaths / M", marker_line_color="white"))
fE.update_layout(title="Deaths per million (cumulative)", geo=dict(showframe=False, projection_type="natural earth"),
                 **{**LAY, "height": 460})

# F - cases vs deaths by month (stacked per continent)
mc = df.groupby(["month", "continent"]).new_cases.sum().reset_index()
fF = go.Figure()
for cont_name, d in mc.groupby("continent"):
    fF.add_trace(go.Bar(x=d.month, y=d.new_cases, name=cont_name))
fF.update_layout(barmode="stack", title="Monthly new cases by continent", yaxis_title="New cases", **LAY)

kpi = dict(cases=glob.total_cases.iloc[-1], deaths=glob.total_deaths.iloc[-1], rec=glob.total_recoveries.iloc[-1],
           vacc=glob.people_fully_vaccinated.iloc[-1] / glob.population.iloc[-1] * 100,
           cfr=glob.total_deaths.iloc[-1] / glob.total_cases.iloc[-1] * 100)
figs = [fA, fB, fE, fC, fD, fF]
parts = [f.to_html(full_html=False, include_plotlyjs=(True if i == 0 else False)) for i, f in enumerate(figs)]
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>COVID-19 Analysis Dashboard</title>
<style>
body{{font-family:'Segoe UI',Arial,sans-serif;background:#f4f6f9;margin:0;color:#1d2433}}
header{{background:#1f3b73;color:#fff;padding:22px 40px}} header h1{{margin:0;font-size:26px}}
header p{{margin:4px 0 0;opacity:.8}}
.kpis{{display:flex;gap:16px;padding:20px 40px;flex-wrap:wrap}}
.kpi{{flex:1;min-width:170px;background:#fff;border-radius:10px;padding:16px 20px;box-shadow:0 1px 4px #0001}}
.kpi b{{display:block;font-size:26px;margin-top:4px}} .kpi span{{font-size:13px;color:#667}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:0 40px 30px}}
.card{{background:#fff;border-radius:10px;padding:8px;box-shadow:0 1px 4px #0001}} .full{{grid-column:1/3}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}} .full{{grid-column:1}}}}
footer{{padding:10px 40px 30px;color:#667;font-size:13px}}
</style></head><body>
<header><h1>COVID-19 Data Analysis Dashboard</h1>
<p>20 countries | {df.date.min():%d %b %Y} - {df.date.max():%d %b %Y} | Lab Sheet-09, MCA III Sem</p></header>
<div class="kpis">
<div class="kpi"><span>Confirmed cases</span><b>{kpi['cases']/1e6:,.1f} M</b></div>
<div class="kpi"><span>Recoveries (reported)</span><b>{kpi['rec']/1e6:,.1f} M</b></div>
<div class="kpi"><span>Deaths</span><b>{kpi['deaths']/1e6:,.2f} M</b></div>
<div class="kpi"><span>Crude fatality rate</span><b>{kpi['cfr']:.2f} %</b></div>
<div class="kpi"><span>Fully vaccinated</span><b>{kpi['vacc']:.1f} %</b></div></div>
<div class="grid">
<div class="card full">{parts[0]}</div><div class="card full">{parts[1]}</div>
<div class="card full">{parts[2]}</div><div class="card">{parts[3]}</div><div class="card">{parts[5]}</div>
<div class="card full">{parts[4]}</div></div>
<footer>Data: simulated COVID-19-style dataset (see generate_covid_dataset.py). Hover for values, drag to zoom, use the dropdown in the country explorer.</footer>
</body></html>"""
with open(os.path.join(HERE, "covid_dashboard.html"), "w", encoding="utf-8") as f:
    f.write(html)
say("Saved interactive dashboard -> covid_dashboard.html")

# =============================================================================
# 5. FINDINGS (numbers computed from the data)
# =============================================================================
peak_day = glob.cases_7d.idxmax()
top3 = latest.total_cases.sort_values(ascending=False).head(3)
hi_pc = latest.cases_per_million.idxmax()
lo_pc = latest.cases_per_million.idxmin()
v_hi, v_lo = latest.vacc_pct.idxmax(), latest.vacc_pct.idxmin()
cfr_early = cfr.loc[:"2020-12-31"].mean()
cfr_late = cfr.loc["2022-01-01":].mean()
corr_vd = np.corrcoef(latest.vacc_pct, latest.deaths_per_million)[0, 1]
cont_d = latest.groupby("continent").deaths_per_million.mean().sort_values(ascending=False)

findings = f"""# COVID-19 Analysis - Findings and Chart Explanations

*Dataset: simulated COVID-19-style data, 20 countries, {df.date.min():%d %b %Y} to {df.date.max():%d %b %Y}
({len(df):,} rows). Conclusions describe THIS dataset; to analyse real data, replace `covid_data.csv`
with a real export that uses the same column names.*

## Key findings
1. **Scale** - {kpi['cases']/1e6:,.0f} M confirmed cases, {kpi['deaths']/1e6:,.2f} M deaths, crude fatality rate {kpi['cfr']:.2f} %.
2. **Peak** - the global 7-day average peaked on {peak_day:%d %b %Y} at about {glob.cases_7d.max()/1e6:.2f} M cases/day.
3. **Largest absolute counts** - {', '.join(f'{c} ({v/1e6:.0f} M)' for c, v in top3.items())}.
   Per capita the picture changes: highest burden is **{hi_pc}** ({latest.cases_per_million[hi_pc]/1000:.0f} per 1,000 people),
   lowest is **{lo_pc}** ({latest.cases_per_million[lo_pc]/1000:.0f} per 1,000).
4. **Fatality fell over time** - monthly case fatality rate averaged {cfr_early:.2f} % in 2020 vs {cfr_late:.2f} % from 2022 onwards.
5. **Vaccination** - highest coverage: {v_hi} ({latest.vacc_pct[v_hi]:.0f} %), lowest: {v_lo} ({latest.vacc_pct[v_lo]:.0f} %).
   Correlation between vaccination coverage and deaths per million across countries is {corr_vd:.2f}
   (cross-country correlation only; it does not prove cause. The positive sign here arises because, in this dataset, countries with heavier outbreaks also rolled out vaccines faster and have older populations - a confounding effect, not evidence that vaccines raise deaths).
6. **Continents** - average deaths per million: {', '.join(f'{k} {v:.0f}' for k, v in cont_d.items())}.

## Data quality notes
- {neg_n} negative daily case counts (reporting corrections) set to missing then interpolated.
- Daily columns interpolated within each country; cumulative columns forward-filled.
- UK recoveries stop in Feb 2022, so UK/global recovery totals are lower bounds.

## What each chart shows (and why that chart type)
| Chart | Type | Explanation |
|---|---|---|
| 01 global trend | line + area | Time series is best shown as a line; 7-day average removes weekend reporting dips. Deaths trail cases by ~10 days. |
| 02 cumulative | line | Confirmed vs recovered vs deaths; the gap between confirmed and recovered is the active-case backlog. |
| 03 top-10 | horizontal bar | Ranking categories; long country names fit on a horizontal axis. |
| 04 per-capita | paired bars | Same countries ranked per head of population, avoiding the "big country" bias in chart 03. |
| 05 vaccination | multi-line | Rollout speed differs by country: slope = speed, plateau = ceiling. |
| 06 heatmap | heatmap | Country x month grid shows waves as dark bands and whether waves were synchronous. |
| 07 continent box | box + points | Spread of deaths per million inside each continent, outliers visible. |
| 08 fatality rate | line + markers | Deaths / cases over time - falls as treatment and vaccination improve. |
| 09 vaccination vs deaths | bubble scatter | Relationship between two numeric variables; bubble size = population. |
| 10 correlation | heatmap | Strength of linear relationships between variables. |

Interactive dashboard (`covid_dashboard.html`): global wave explorer with range slider, country dropdown, choropleth map,
per-capita ranking, vaccination curves and stacked continent chart.
"""
with open(os.path.join(HERE, "findings.md"), "w", encoding="utf-8") as f:
    f.write(findings)
with open(os.path.join(HERE, "eda_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log))
say("Saved findings.md and eda_report.txt")
