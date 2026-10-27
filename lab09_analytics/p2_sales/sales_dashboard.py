"""
Lab Sheet-09 | Project 2: Sales Data Analysis Dashboard
MCA III Semester (Session 2026-2027)

  1. Load + EDA                   -> eda_report.txt
  2. Clean (duplicates, missing)  -> sales_data_clean.csv
  3. Static charts                -> charts/*.png
  4. RFM customer segmentation    -> customer_segments.csv
  5. Interactive Plotly dashboard -> sales_dashboard.html
  6. Business insights            -> findings.md
(An interactive filterable app is in sales_app.py:  streamlit run sales_app.py)

Run:  python generate_sales_dataset.py   (once)   then   python sales_dashboard.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.join(HERE, "charts")
os.makedirs(CH, exist_ok=True)
sns.set_theme(style="whitegrid")
BLUE, RED, GREEN, GREY = "#4c78a8", "#e45756", "#54a24b", "#9aa5b1"
log = []


def say(t=""):
    print(t)
    log.append(str(t))


def inr(x):
    """Format rupees in crore / lakh."""
    return f"Rs {x / 1e7:.2f} Cr" if abs(x) >= 1e7 else f"Rs {x / 1e5:.1f} L"


# =============================================================================
# 1. LOAD + EDA
# =============================================================================
raw = pd.read_csv(os.path.join(HERE, "sales_data.csv"), parse_dates=["Order Date", "Ship Date"])
say("=" * 70 + "\n1. EXPLORATORY DATA ANALYSIS\n" + "=" * 70)
say(f"Shape: {raw.shape[0]:,} rows x {raw.shape[1]} columns")
say(f"Period: {raw['Order Date'].min():%d %b %Y} -> {raw['Order Date'].max():%d %b %Y}")
say(f"Customers: {raw['Customer ID'].nunique()} | Products: {raw['Product Name'].nunique()} | States: {raw.State.nunique()}")
say("\nData types:\n" + raw.dtypes.to_string())
say("\nNumeric summary:\n" + raw[["Sales", "Quantity", "Discount", "Profit"]].describe().T.round(2).to_string())
miss = pd.DataFrame({"missing": raw.isna().sum(), "pct": (raw.isna().mean() * 100).round(2)})
say("\nMissing values BEFORE cleaning:\n" + miss[miss.missing > 0].to_string())
n_dup = int(raw.duplicated().sum())
say(f"\nExact duplicate rows: {n_dup}")
say("\nSales skew: mean %.0f vs median %.0f (right-skewed -> a few big orders dominate)" % (raw.Sales.mean(), raw.Sales.median()))

# =============================================================================
# 2. CLEANING
# =============================================================================
say("\n" + "=" * 70 + "\n2. DATA CLEANING DECISIONS\n" + "=" * 70)
df = raw.drop_duplicates().copy()
say(f"- Dropped {n_dup} exact duplicate rows -> {len(df):,} rows")

# Segment: a customer has one segment -> recover from the customer's other orders
seg_map = df.dropna(subset=["Segment"]).groupby("Customer ID").Segment.first()
n = df.Segment.isna().sum()
df["Segment"] = df.Segment.fillna(df["Customer ID"].map(seg_map)).fillna("Unknown")
say(f"- Segment: {n} missing -> recovered from the same customer's other orders (rest = 'Unknown')")

# Region: a state belongs to one region -> recover from State
reg_map = df.dropna(subset=["Region"]).groupby("State").Region.first()
n = df.Region.isna().sum()
df["Region"] = df.Region.fillna(df.State.map(reg_map))
say(f"- Region: {n} missing -> recovered from State lookup")

# Quantity / Discount: median of the sub-category
for col in ["Quantity", "Discount"]:
    n = df[col].isna().sum()
    # mode (most common value) keeps Discount on the real price-list values (0, 5, 10 ... %)
    df[col] = df[col].fillna(df.groupby("Sub-Category")[col].transform(lambda s: s.mode().iloc[0]))
    say(f"- {col}: {n} missing -> most common value (mode) of the product Sub-Category")

# Ship date: order date + median delay
delay = (df["Ship Date"] - df["Order Date"]).dt.days
n = df["Ship Date"].isna().sum()
df["Ship Date"] = df["Ship Date"].fillna(df["Order Date"] + pd.to_timedelta(delay.median(), unit="D"))
say(f"- Ship Date: {n} missing -> Order Date + median delay ({delay.median():.0f} days)")

# Profit: estimate from the median margin of the (Sub-Category, Discount) group; flag it
df["Profit_Imputed"] = df.Profit.isna()
margin = (df.Profit / df.Sales)
grp_margin = margin.groupby([df["Sub-Category"], df.Discount]).transform("median")
sub_margin = margin.groupby(df["Sub-Category"]).transform("median")
n = df.Profit.isna().sum()
df["Profit"] = df.Profit.fillna(df.Sales * grp_margin.fillna(sub_margin)).round(2)
say(f"- Profit: {n} missing -> Sales x median margin of (Sub-Category, Discount); flagged in 'Profit_Imputed'")

df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
df["MonthNum"] = df["Order Date"].dt.month
df["Margin"] = df.Profit / df.Sales
df["Ship_Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
assert df.drop(columns=[]).isna().sum().sum() == 0
say(f"\nMissing values AFTER cleaning: {int(df.isna().sum().sum())}")
df.to_csv(os.path.join(HERE, "sales_data_clean.csv"), index=False)

TOTAL_S, TOTAL_P = df.Sales.sum(), df.Profit.sum()
last_year = df.Year.max()
cut = df[df.Year == last_year]["Order Date"].max()          # last year is partial -> compare same period
ytd = lambda y: df[(df.Year == y) & (df["Order Date"] <= cut.replace(year=y))]
say(f"\nNOTE: {last_year} is a partial year (to {cut:%d %b}); year-on-year comparisons use the same Jan-{cut:%b} window.")


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(CH, name), dpi=150)
    plt.close()


# =============================================================================
# 3. STATIC CHARTS
# =============================================================================
monthly = df.groupby("Month")[["Sales", "Profit"]].sum()
# 01 monthly sales + profit
fig, ax = plt.subplots(figsize=(12, 4.8))
ax.bar(monthly.index, monthly.Sales / 1e5, width=22, color=BLUE, alpha=.55, label="Sales")
ax.set_ylabel("Sales (Rs lakh)")
ax2 = ax.twinx()
ax2.plot(monthly.index, monthly.Profit / 1e5, color=RED, marker="o", ms=4, label="Profit")
ax2.axhline(0, color="k", lw=.6); ax2.set_ylabel("Profit (Rs lakh)"); ax2.grid(False)
ax.set_title("Monthly sales and profit (festive Oct-Dec peaks every year)")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left")
save("01_monthly_sales_profit.png")

# 02 year-on-year by month
pv = df.pivot_table(index="MonthNum", columns="Year", values="Sales", aggfunc="sum") / 1e5
fig, ax = plt.subplots(figsize=(10, 4.5))
for y in pv.columns:
    ax.plot(pv.index, pv[y], marker="o", label=str(y), lw=2)
ax.set_xticks(range(1, 13)); ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax.set(title=f"Seasonality and growth: monthly sales by year ({last_year} partial)", ylabel="Sales (Rs lakh)"); ax.legend()
save("02_yearly_seasonality.png")

# 03 category + sub-category profit
sub = df.groupby(["Category", "Sub-Category"])[["Sales", "Profit"]].sum().reset_index().sort_values("Profit")
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
cat = df.groupby("Category")[["Sales", "Profit"]].sum() / 1e5
cat.plot.bar(ax=ax[0], color=[BLUE, GREEN], rot=0)
ax[0].set(title="Sales vs profit by category", ylabel="Rs lakh", xlabel="")
ax[1].barh(sub["Sub-Category"], sub.Profit / 1e5, color=[RED if v < 0 else GREEN for v in sub.Profit])
ax[1].set(title="Profit by sub-category (red = loss-making)", xlabel="Profit (Rs lakh)")
save("03_category_subcategory_profit.png")

# 04 top / bottom products
prod = df.groupby("Product Name")[["Sales", "Profit"]].sum().sort_values("Profit")
sel = pd.concat([prod.head(8), prod.tail(8)])
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(sel.index, sel.Profit / 1e5, color=[RED if v < 0 else GREEN for v in sel.Profit])
ax.axvline(0, color="k", lw=.7)
ax.set(title="Best 8 and worst 8 products by total profit", xlabel="Profit (Rs lakh)")
save("04_top_bottom_products.png")

# 05 discount vs profit
dd = df.groupby("Discount")[["Sales", "Profit"]].sum()
dd["Margin %"] = dd.Profit / dd.Sales * 100
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].bar([f"{int(d * 100)}%" for d in dd.index], dd.Profit / 1e5, color=[RED if v < 0 else GREEN for v in dd.Profit])
ax[0].set(title="Total profit by discount level", xlabel="Discount", ylabel="Rs lakh"); ax[0].axhline(0, color="k", lw=.6)
ax[1].plot([f"{int(d * 100)}%" for d in dd.index], dd["Margin %"], marker="o", color=RED)
ax[1].axhline(0, color="k", lw=.6); ax[1].set(title="Profit margin by discount level", xlabel="Discount", ylabel="Margin %")
save("05_discount_vs_profit.png")

# 06 regional + state
reg = df.groupby("Region")[["Sales", "Profit"]].sum().sort_values("Sales", ascending=False) / 1e5
st = df.groupby("State")[["Sales", "Profit"]].sum().sort_values("Sales").tail(10) / 1e5
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
reg.plot.bar(ax=ax[0], color=[BLUE, GREEN], rot=0); ax[0].set(title="Region: sales vs profit", ylabel="Rs lakh", xlabel="")
st.Sales.plot.barh(ax=ax[1], color=BLUE); ax[1].set(title="Top 10 states by sales", xlabel="Rs lakh", ylabel="")
save("06_regional_analysis.png")

# 07 region x category margin heatmap
hm = df.pivot_table(index="Region", columns="Category", values=["Profit", "Sales"], aggfunc="sum")
hm = (hm["Profit"] / hm["Sales"] * 100)
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.heatmap(hm, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=ax, cbar_kws={"label": "Profit margin %"})
ax.set_title("Profit margin % - Region x Category"); ax.set_ylabel("")
save("07_region_category_margin.png")

# 08 customer segment (Consumer / Corporate / Home Office)
sg = df[df.Segment != "Unknown"].groupby("Segment").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"),
                                                        Customers=("Customer ID", "nunique"), Orders=("Order ID", "nunique"))
sg["Sales per customer (Rs L)"] = sg.Sales / sg.Customers / 1e5
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
ax[0].pie(sg.Sales, labels=sg.index, autopct="%1.0f%%", colors=[BLUE, GREEN, "#f58518"], startangle=90)
ax[0].set_title("Share of sales")
sns.barplot(x=sg.index, y=sg["Sales per customer (Rs L)"], color=BLUE, ax=ax[1]); ax[1].set(title="Sales per customer", ylabel="Rs lakh", xlabel="")
sns.barplot(x=sg.index, y=sg.Profit / sg.Sales * 100, color=GREEN, ax=ax[2]); ax[2].set(title="Profit margin %", ylabel="%", xlabel="")
save("08_customer_segments.png")

# 09 distributions + correlations
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
sns.histplot(np.log10(df.Sales), bins=40, color=BLUE, ax=ax[0]); ax[0].set(title="Order value distribution (log10 Rs)", xlabel="log10(Sales)")
sns.histplot(df.Ship_Days, discrete=True, color=GREEN, ax=ax[1]); ax[1].set(title="Shipping time (days)", xlabel="days")
sns.heatmap(df[["Sales", "Quantity", "Discount", "Profit"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax[2])
ax[2].set_title("Correlation matrix")
save("09_distributions_correlation.png")

# =============================================================================
# 4. RFM CUSTOMER SEGMENTATION (K-Means)
# =============================================================================
snap = df["Order Date"].max() + pd.Timedelta(days=1)
rfm = df.groupby("Customer ID").agg(Recency=("Order Date", lambda s: (snap - s.max()).days),
                                     Frequency=("Order ID", "nunique"), Monetary=("Sales", "sum"),
                                     Profit=("Profit", "sum"))
X = StandardScaler().fit_transform(np.log1p(rfm[["Recency", "Frequency", "Monetary"]]))
km = KMeans(n_clusters=4, n_init=20, random_state=42).fit(X)
rfm["Cluster"] = km.labels_
cs = rfm.groupby("Cluster").agg(Customers=("Recency", "size"), Recency=("Recency", "mean"),
                                Frequency=("Frequency", "mean"), Monetary=("Monetary", "mean"))
names, left = {}, set(cs.index)
champ = cs.loc[list(left)].Monetary.idxmax(); names[champ] = "Champions"; left.remove(champ)
lapsed = cs.loc[list(left)].Recency.idxmax(); names[lapsed] = "At-risk / lapsed"; left.remove(lapsed)
loyal = cs.loc[list(left)].Frequency.idxmax(); names[loyal] = "Loyal regulars"; left.remove(loyal)
names[left.pop()] = "Occasional buyers"
rfm["Segment_RFM"] = rfm.Cluster.map(names)
rfm.drop(columns="Cluster").to_csv(os.path.join(HERE, "customer_segments.csv"))
rs = rfm.groupby("Segment_RFM").agg(Customers=("Recency", "size"), Avg_Recency_days=("Recency", "mean"),
                                    Avg_Orders=("Frequency", "mean"), Avg_Spend=("Monetary", "mean"),
                                    Total_Sales=("Monetary", "sum"), Total_Profit=("Profit", "sum")).sort_values("Total_Sales", ascending=False)
rs["Sales_share_%"] = rs.Total_Sales / rs.Total_Sales.sum() * 100
say("\nRFM segments:\n" + rs.round(1).to_string())
order = ["Champions", "Loyal regulars", "Occasional buyers", "At-risk / lapsed"]
pal = dict(zip(order, [GREEN, BLUE, "#f58518", RED]))
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(data=rfm, x="Frequency", y="Monetary", hue="Segment_RFM", hue_order=order, palette=pal, alpha=.75, ax=ax[0])
ax[0].set(yscale="log", title="RFM clusters: orders vs spend", ylabel="Total spend (Rs, log)")
sns.scatterplot(data=rfm, x="Recency", y="Monetary", hue="Segment_RFM", hue_order=order, palette=pal, alpha=.75, ax=ax[1], legend=False)
ax[1].set(yscale="log", title="RFM clusters: days since last order vs spend", xlabel="Recency (days since last order)")
save("10_rfm_segmentation.png")
say(f"\nSaved 10 static charts in {CH}")

# =============================================================================
# 5. INTERACTIVE DASHBOARD
# =============================================================================
LAY = dict(template="plotly_white", margin=dict(l=40, r=20, t=55, b=40), height=420, font=dict(family="Segoe UI, Arial", size=13))
yrs = sorted(df.Year.unique())

# A monthly trend with year dropdown
mt = df.groupby(["Year", "MonthNum"])[["Sales", "Profit"]].sum().reset_index()
fA = make_subplots(specs=[[{"secondary_y": True}]])
for i, y in enumerate([None] + yrs):
    d = monthly.reset_index() if y is None else mt[mt.Year == y].assign(Month=lambda x: pd.to_datetime(dict(year=y, month=x.MonthNum, day=1)))
    fA.add_trace(go.Bar(x=d.Month, y=d.Sales, name="Sales", marker_color=BLUE, opacity=.6, visible=y is None), secondary_y=False)
    fA.add_trace(go.Scatter(x=d.Month, y=d.Profit, name="Profit", line=dict(color=RED), mode="lines+markers", visible=y is None), secondary_y=True)
btn = []
for i, y in enumerate([None] + yrs):
    v = [False] * (2 * (len(yrs) + 1)); v[2 * i] = v[2 * i + 1] = True
    btn.append(dict(label="All years" if y is None else str(y), method="update", args=[{"visible": v}]))
fA.update_layout(title="Monthly sales and profit (choose a year)", updatemenus=[dict(buttons=btn, x=0, y=1.18, xanchor="left")],
                 legend=dict(orientation="h", y=1.15, x=.6), **LAY)
fA.update_yaxes(title_text="Sales (Rs)", secondary_y=False); fA.update_yaxes(title_text="Profit (Rs)", secondary_y=True)

# B category -> sub-category treemap coloured by margin
tm = df.groupby(["Category", "Sub-Category"])[["Sales", "Profit"]].sum().reset_index()
tm["Margin %"] = tm.Profit / tm.Sales * 100
fB = px.treemap(tm, path=["Category", "Sub-Category"], values="Sales", color="Margin %", color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0, title="Product performance - size = sales, colour = profit margin")
fB.update_layout(**LAY)

# C region
rg = df.groupby("Region")[["Sales", "Profit"]].sum().reset_index().sort_values("Sales", ascending=False)
fC = go.Figure([go.Bar(x=rg.Region, y=rg.Sales, name="Sales", marker_color=BLUE),
                go.Bar(x=rg.Region, y=rg.Profit, name="Profit", marker_color=GREEN)])
fC.update_layout(barmode="group", title="Regional analysis", **LAY)

# D discount vs profit
fD = go.Figure(go.Bar(x=[f"{int(d * 100)}%" for d in dd.index], y=dd.Profit, marker_color=[RED if v < 0 else GREEN for v in dd.Profit],
                      hovertemplate="Discount %{x}<br>Profit Rs %{y:,.0f}<extra></extra>"))
fD.update_layout(title="Profit by discount level - deep discounts destroy profit", xaxis_title="Discount", yaxis_title="Profit (Rs)", **LAY)

# E RFM scatter
fE = px.scatter(rfm.reset_index(), x="Frequency", y="Monetary", color="Segment_RFM", category_orders={"Segment_RFM": order},
                color_discrete_map=pal, log_y=True, hover_data=["Customer ID", "Recency"], title="Customer segmentation (RFM, K-Means)")
fE.update_layout(**LAY)

# F top products
tp = df.groupby("Product Name")[["Sales", "Profit"]].sum().sort_values("Profit").pipe(lambda d: pd.concat([d.head(6), d.tail(6)]))
fF = go.Figure(go.Bar(x=tp.Profit, y=tp.index, orientation="h", marker_color=[RED if v < 0 else GREEN for v in tp.Profit]))
fF.update_layout(title="Best and worst products by profit", xaxis_title="Profit (Rs)", **LAY)

kp = [("Total sales", inr(TOTAL_S)), ("Total profit", inr(TOTAL_P)), ("Profit margin", f"{TOTAL_P / TOTAL_S * 100:.1f} %"),
      ("Orders", f"{df['Order ID'].nunique():,}"), ("Customers", f"{df['Customer ID'].nunique():,}"),
      ("Avg order value", f"Rs {TOTAL_S / df['Order ID'].nunique():,.0f}")]
cards = "".join(f'<div class="kpi"><span>{a}</span><b>{b}</b></div>' for a, b in kp)
parts = [f.to_html(full_html=False, include_plotlyjs=(True if i == 0 else False)) for i, f in enumerate([fA, fB, fC, fD, fE, fF])]
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Sales Analysis Dashboard</title>
<style>body{{font-family:'Segoe UI',Arial,sans-serif;background:#f4f6f9;margin:0;color:#1d2433}}
header{{background:#1f3b73;color:#fff;padding:22px 40px}} header h1{{margin:0;font-size:26px}} header p{{margin:4px 0 0;opacity:.8}}
.kpis{{display:flex;gap:16px;padding:20px 40px;flex-wrap:wrap}} .kpi{{flex:1;min-width:150px;background:#fff;border-radius:10px;padding:16px 20px;box-shadow:0 1px 4px #0001}}
.kpi b{{display:block;font-size:24px;margin-top:4px}} .kpi span{{font-size:13px;color:#667}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:0 40px 30px}} .card{{background:#fff;border-radius:10px;padding:8px;box-shadow:0 1px 4px #0001}} .full{{grid-column:1/3}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}} .full{{grid-column:1}}}} footer{{padding:10px 40px 30px;color:#667;font-size:13px}}</style></head><body>
<header><h1>Sales Analysis Dashboard</h1><p>{df['Order Date'].min():%b %Y} - {df['Order Date'].max():%b %Y} | Lab Sheet-09, MCA III Sem</p></header>
<div class="kpis">{cards}</div>
<div class="grid"><div class="card full">{parts[0]}</div><div class="card">{parts[1]}</div><div class="card">{parts[2]}</div>
<div class="card">{parts[3]}</div><div class="card">{parts[5]}</div><div class="card full">{parts[4]}</div></div>
<footer>Data: simulated retail dataset (generate_sales_dataset.py). Hover for details; use the year dropdown on the first chart. For full filters run: streamlit run sales_app.py</footer></body></html>"""
with open(os.path.join(HERE, "sales_dashboard.html"), "w", encoding="utf-8") as f:
    f.write(html)
say("Saved interactive dashboard -> sales_dashboard.html")

# =============================================================================
# 6. FINDINGS
# =============================================================================
yoy = {y: ytd(y).Sales.sum() for y in yrs}
g_last = (yoy[yrs[-1]] / yoy[yrs[-2]] - 1) * 100
cagr_txt = " -> ".join(f"{y}: {inr(v)}" for y, v in yoy.items())
festive = df[df.MonthNum.isin([10, 11, 12])].Sales.sum() / df[df.Year < last_year].Sales.sum() * 100 / 1  # share of full years
full = df[df.Year < last_year]
festive = full[full.MonthNum.isin([10, 11, 12])].Sales.sum() / full.Sales.sum() * 100
catm = df.groupby("Category")[["Sales", "Profit"]].sum()
loss_sub = sub[sub.Profit < 0]
disc_loss = df[df.Discount >= 0.15]
reg_m = df.groupby("Region")[["Sales", "Profit"]].sum().assign(m=lambda x: x.Profit / x.Sales * 100).sort_values("m")
champ_row = rs.loc["Champions"]
top_share = rfm.sort_values("Monetary", ascending=False).head(int(len(rfm) * .2)).Monetary.sum() / rfm.Monetary.sum() * 100
profit_leak = -disc_loss[disc_loss.Profit < 0].Profit.sum()

findings = f"""# Sales Analysis - Findings, Insights and Recommendations

*Dataset: simulated retail sales data (India, INR), {len(df):,} orders after cleaning, {df['Order Date'].min():%b %Y} - {df['Order Date'].max():%b %Y}.*

## Headline numbers
- Total sales **{inr(TOTAL_S)}**, profit **{inr(TOTAL_P)}** (margin **{TOTAL_P / TOTAL_S * 100:.1f} %**).
- Same-period (Jan-{cut:%b}) sales by year: {cagr_txt}. Latest growth vs previous year: **{g_last:+.0f} %**.

## Key insights
1. **Seasonality** - Oct-Dec account for {festive:.0f} % of sales in complete years (festive season); stock and staffing should peak before October.
2. **Revenue is not profit** - Technology brings {catm.loc['Technology', 'Sales'] / TOTAL_S * 100:.0f} % of sales but only a {catm.loc['Technology', 'Profit'] / catm.loc['Technology', 'Sales'] * 100:.1f} % margin; Office Supplies is small ({catm.loc['Office Supplies', 'Sales'] / TOTAL_S * 100:.0f} % of sales) but earns a {catm.loc['Office Supplies', 'Profit'] / catm.loc['Office Supplies', 'Sales'] * 100:.0f} % margin.
3. **Loss-making sub-categories** - {', '.join(f"{r['Sub-Category']} ({inr(r.Profit)})" for _, r in loss_sub.iterrows())}.
4. **Discounting hurts** - orders discounted 15 % or more lose {inr(profit_leak)} in total; profit is positive only up to ~10 % discount (chart 05).
5. **Regions** - best margin: {reg_m.index[-1]} ({reg_m.m.iloc[-1]:.1f} %), weakest: {reg_m.index[0]} ({reg_m.m.iloc[0]:.1f} %).
6. **Customers** - the top 20 % of customers generate {top_share:.0f} % of sales. RFM clustering found: {', '.join(f"{k} ({int(v.Customers)})" for k, v in rs.iterrows())}.
   Champions average {int(champ_row.Avg_Orders)} orders and spend {inr(champ_row.Avg_Spend)} each.

## Recommendations
- Cap discounts at 10 % on Tables, Printers and East-region orders until margins recover.
- Re-price or renegotiate supplier cost on loss-making sub-categories; bundle low-margin items with high-margin accessories.
- Run a retention campaign for the *At-risk / lapsed* segment and a loyalty programme for *Champions*.
- Build festive-season inventory from September; run slow-season (Jun-Aug) promotions on high-margin Office Supplies only.

## What each chart shows
| Chart | Type | Why this type |
|---|---|---|
| 01 monthly sales/profit | bars + line, dual axis | trend over time; bars = volume, line = profit |
| 02 yearly seasonality | multi-line by month | compares the same month across years (growth + seasonality) |
| 03 category / sub-category | grouped + sorted bars | compare categories; sorted bars expose loss-makers |
| 04 top/bottom products | diverging horizontal bars | ranks products; colour marks loss vs profit |
| 05 discount vs profit | bars + line | shows the threshold where discounting turns unprofitable |
| 06 regional | grouped bars | sales vs profit per region + state ranking |
| 07 region x category | heatmap | two categorical dimensions and one measure (margin) |
| 08 segments | pie + bars | share of sales, spend per customer, margin by segment |
| 09 distributions | histograms + correlation | skew of order values, shipping time, variable relationships |
| 10 RFM | scatter by cluster | K-Means customer groups on Recency, Frequency, Monetary |
"""
with open(os.path.join(HERE, "findings.md"), "w", encoding="utf-8") as f:
    f.write(findings)
with open(os.path.join(HERE, "eda_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log))
say("Saved findings.md and eda_report.txt")
