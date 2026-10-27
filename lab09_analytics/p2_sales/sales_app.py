"""Interactive sales dashboard with filters.  Run:  streamlit run sales_app.py
(needs sales_data_clean.csv - created by sales_dashboard.py)"""
import os

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Sales Dashboard", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def load():
    return pd.read_csv(os.path.join(HERE, "sales_data_clean.csv"), parse_dates=["Order Date", "Month"])


df = load()
st.title("Sales Analysis Dashboard")
st.sidebar.header("Filters")
years = st.sidebar.multiselect("Year", sorted(df.Year.unique()), default=sorted(df.Year.unique()))
regions = st.sidebar.multiselect("Region", sorted(df.Region.unique()), default=sorted(df.Region.unique()))
cats = st.sidebar.multiselect("Category", sorted(df.Category.unique()), default=sorted(df.Category.unique()))
segs = st.sidebar.multiselect("Segment", sorted(df.Segment.unique()), default=sorted(df.Segment.unique()))
f = df[df.Year.isin(years) & df.Region.isin(regions) & df.Category.isin(cats) & df.Segment.isin(segs)]
if f.empty:
    st.warning("No data for the selected filters.")
    st.stop()

c = st.columns(5)
c[0].metric("Sales (Rs lakh)", f"{f.Sales.sum() / 1e5:,.1f}")
c[1].metric("Profit (Rs lakh)", f"{f.Profit.sum() / 1e5:,.1f}")
c[2].metric("Margin", f"{f.Profit.sum() / f.Sales.sum() * 100:.1f}%")
c[3].metric("Orders", f"{f['Order ID'].nunique():,}")
c[4].metric("Customers", f"{f['Customer ID'].nunique():,}")

m = f.groupby("Month")[["Sales", "Profit"]].sum().reset_index()
st.plotly_chart(px.line(m, x="Month", y=["Sales", "Profit"], title="Monthly sales and profit"), width="stretch")
a, b = st.columns(2)
sub = f.groupby(["Category", "Sub-Category"])[["Sales", "Profit"]].sum().reset_index()
a.plotly_chart(px.bar(sub.sort_values("Profit"), x="Profit", y="Sub-Category", color="Category", orientation="h",
                      title="Profit by sub-category"), width="stretch")
r = f.groupby("Region")[["Sales", "Profit"]].sum().reset_index()
b.plotly_chart(px.bar(r, x="Region", y=["Sales", "Profit"], barmode="group", title="Regional performance"), width="stretch")
d = f.groupby("Discount").Profit.sum().reset_index()
a.plotly_chart(px.bar(d, x="Discount", y="Profit", title="Profit by discount level"), width="stretch")
p = f.groupby("Product Name")[["Sales", "Profit"]].sum().sort_values("Profit")
b.plotly_chart(px.bar(pd.concat([p.head(5), p.tail(5)]).reset_index(), x="Profit", y="Product Name", orientation="h",
                      title="Best / worst products"), width="stretch")
with st.expander("Filtered data"):
    st.dataframe(f.head(500))
