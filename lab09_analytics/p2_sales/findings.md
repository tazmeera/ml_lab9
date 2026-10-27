# Sales Analysis - Findings, Insights and Recommendations

*Dataset: simulated retail sales data (India, INR), 9,000 orders after cleaning, Jan 2023 - Sep 2026.*

## Headline numbers
- Total sales **Rs 25.71 Cr**, profit **Rs 87.4 L** (margin **3.4 %**).
- Same-period (Jan-Sep) sales by year: 2023: Rs 3.20 Cr -> 2024: Rs 4.50 Cr -> 2025: Rs 5.16 Cr -> 2026: Rs 6.10 Cr. Latest growth vs previous year: **+18 %**.

## Key insights
1. **Seasonality** - Oct-Dec account for 34 % of sales in complete years (festive season); stock and staffing should peak before October.
2. **Revenue is not profit** - Technology brings 73 % of sales but only a 4.2 % margin; Office Supplies is small (5 % of sales) but earns a 17 % margin.
3. **Loss-making sub-categories** - Tables (Rs -29.9 L), Printers (Rs -16.2 L).
4. **Discounting hurts** - orders discounted 15 % or more lose Rs 90.5 L in total; profit is positive only up to ~10 % discount (chart 05).
5. **Regions** - best margin: South (5.4 %), weakest: East (-0.6 %).
6. **Customers** - the top 20 % of customers generate 65 % of sales. RFM clustering found: Champions (88), Loyal regulars (204), Occasional buyers (269), At-risk / lapsed (118).
   Champions average 52 orders and spend Rs 15.5 L each.

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
