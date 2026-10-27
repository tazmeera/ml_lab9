"""
Generates a retail sales dataset (sales_data.csv) for Lab 09 Project 2.

SIMULATED data in the style of the well-known "Superstore" dataset, localised to India (INR).
Built-in patterns: yearly growth, festive-season peak (Oct-Dec), category margins, losses at high
discount, regional differences. Deliberate data problems: missing values and duplicate rows.
sales_dashboard.py also works on any real CSV with the same column names.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(11)
N = 9000

REGIONS = {
    "North": ["Delhi", "Punjab", "Uttar Pradesh", "Haryana", "Uttarakhand"],
    "South": ["Karnataka", "Tamil Nadu", "Kerala", "Telangana"],
    "West": ["Maharashtra", "Gujarat", "Rajasthan", "Goa"],
    "East": ["West Bengal", "Odisha", "Bihar", "Assam"],
    "Central": ["Madhya Pradesh", "Chhattisgarh"],
}
REGION_W = {"North": 0.27, "South": 0.25, "West": 0.30, "East": 0.12, "Central": 0.06}
REGION_MARGIN = {"North": 0.0, "South": 0.03, "West": 0.02, "East": -0.04, "Central": -0.02}

# category -> sub-category -> (list of products, avg unit price INR, base margin)
CATALOG = {
    "Technology": {
        "Phones": (["Smartphone X1", "Smartphone Lite", "Smartphone Pro Max", "Feature Phone"], 24000, 0.14),
        "Laptops": (["Laptop Air 14", "Laptop Gaming 15", "Laptop Business 13"], 62000, 0.10),
        "Accessories": (["Wireless Mouse", "Bluetooth Headset", "USB-C Hub", "Power Bank"], 1800, 0.30),
        "Printers": (["Inkjet Printer", "Laser Printer", "All-in-One Printer"], 12000, 0.02),
    },
    "Furniture": {
        "Chairs": (["Ergo Office Chair", "Mesh Chair", "Executive Chair"], 7500, 0.12),
        "Tables": (["Standing Desk", "Conference Table", "Study Table"], 11000, -0.04),
        "Bookcases": (["Wooden Bookcase", "Metal Shelf Unit"], 6500, 0.08),
        "Furnishings": (["Desk Lamp", "Wall Clock", "Curtains Set"], 1200, 0.25),
    },
    "Office Supplies": {
        "Paper": (["A4 Paper Ream", "Notebook Pack", "Sticky Notes"], 350, 0.38),
        "Binders": (["Ring Binder", "Lever Arch File"], 280, 0.34),
        "Storage": (["File Cabinet", "Storage Box", "Desk Organizer"], 2200, 0.15),
        "Art & Stationery": (["Pen Set", "Marker Pack", "Highlighters"], 400, 0.30),
    },
}
CATEGORY_W = {"Technology": 0.30, "Furniture": 0.25, "Office Supplies": 0.45}

# ------------------------------------------------------------ customers
n_cust = 700
cust = pd.DataFrame({
    "Customer ID": [f"CU-{i:04d}" for i in range(1, n_cust + 1)],
    "Customer Name": [f"Customer {i:04d}" for i in range(1, n_cust + 1)],
    "Segment": rng.choice(["Consumer", "Corporate", "Home Office"], n_cust, p=[0.52, 0.30, 0.18]),
})
# heavy-tailed purchase frequency -> a few loyal, many occasional customers
cust["weight"] = rng.pareto(1.6, n_cust) + 0.3
cust["weight"] /= cust.weight.sum()
seg_boost = {"Consumer": 1.0, "Corporate": 1.6, "Home Office": 1.2}

# ------------------------------------------------------------ order dates (growth + festive season)
days = pd.date_range("2023-01-01", "2026-09-30")
doy_w = np.where(days.month.isin([10, 11, 12]), 1.7, np.where(days.month.isin([3, 4]), 1.15, 1.0))
year_w = np.select([days.year == 2023, days.year == 2024, days.year == 2025], [1.0, 1.25, 1.55], 1.85)
p = doy_w * year_w
p /= p.sum()
order_dates = rng.choice(days, N, p=p)

rows = []
for i in range(N):
    c = cust.iloc[rng.choice(n_cust, p=cust.weight.values)]
    cat = rng.choice(list(CATEGORY_W), p=list(CATEGORY_W.values()))
    sub = rng.choice(list(CATALOG[cat]))
    prods, price, margin = CATALOG[cat][sub]
    prod = rng.choice(prods)
    region = rng.choice(list(REGION_W), p=list(REGION_W.values()))
    state = rng.choice(REGIONS[region])
    qty = int(np.clip(rng.poisson(2.2) + 1, 1, 12) * (1 if price > 20000 else 1 + (rng.random() < 0.3)))
    qty = max(1, min(qty, 12 if price < 20000 else 4))
    discount = rng.choice([0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
                          p=[0.38, 0.12, 0.15, 0.10, 0.12, 0.07, 0.04, 0.02])
    unit = price * rng.lognormal(0, 0.18) * seg_boost[c.Segment] ** 0.15
    sales = round(unit * qty * (1 - discount), 2)
    # profit margin: base - discount penalty + regional effect + noise
    m = margin + REGION_MARGIN[region] - 0.9 * discount + 0.35 * margin * (discount == 0) + rng.normal(0, 0.04)
    profit = round(sales * m, 2)
    od = pd.Timestamp(order_dates[i])
    ship = od + pd.Timedelta(days=int(rng.choice([2, 3, 4, 5, 6, 7], p=[.12, .24, .28, .18, .1, .08])))
    rows.append((f"ORD-{od:%Y}-{i + 1:05d}", od, ship, c["Customer ID"], c["Customer Name"], c.Segment, region, state,
                 cat, sub, prod, sales, qty, discount, profit))

df = pd.DataFrame(rows, columns=["Order ID", "Order Date", "Ship Date", "Customer ID", "Customer Name", "Segment",
                                 "Region", "State", "Category", "Sub-Category", "Product Name", "Sales", "Quantity",
                                 "Discount", "Profit"]).sort_values("Order Date").reset_index(drop=True)

# ------------------------------------------------------------ inject data problems
n = len(df)
df.loc[rng.choice(n, int(n * 0.020), replace=False), "Discount"] = np.nan
df.loc[rng.choice(n, int(n * 0.015), replace=False), "Profit"] = np.nan
df.loc[rng.choice(n, int(n * 0.012), replace=False), "Segment"] = np.nan
df.loc[rng.choice(n, int(n * 0.010), replace=False), "Region"] = np.nan
df.loc[rng.choice(n, int(n * 0.020), replace=False), "Ship Date"] = pd.NaT
df.loc[rng.choice(n, int(n * 0.008), replace=False), "Quantity"] = np.nan
dups = df.sample(40, random_state=1)
df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)

df.to_csv("sales_data.csv", index=False)
print(f"sales_data.csv written: {len(df):,} rows x {df.shape[1]} cols | "
      f"{df['Order Date'].min():%Y-%m-%d} -> {df['Order Date'].max():%Y-%m-%d}")
