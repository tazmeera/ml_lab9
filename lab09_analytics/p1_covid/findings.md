# COVID-19 Analysis - Findings and Chart Explanations

*Dataset: simulated COVID-19-style data, 20 countries, 01 Mar 2020 to 31 Dec 2022
(20,720 rows). Conclusions describe THIS dataset; to analyse real data, replace `covid_data.csv`
with a real export that uses the same column names.*

## Key findings
1. **Scale** - 852 M confirmed cases, 7.46 M deaths, crude fatality rate 0.88 %.
2. **Peak** - the global 7-day average peaked on 17 Feb 2022 at about 2.94 M cases/day.
3. **Largest absolute counts** - India (317 M), USA (131 M), Brazil (79 M).
   Per capita the picture changes: highest burden is **Italy** (399 per 1,000 people),
   lowest is **Nigeria** (43 per 1,000).
4. **Fatality fell over time** - monthly case fatality rate averaged 2.29 % in 2020 vs 0.55 % from 2022 onwards.
5. **Vaccination** - highest coverage: Spain (80 %), lowest: Nigeria (25 %).
   Correlation between vaccination coverage and deaths per million across countries is 0.57
   (cross-country correlation only; it does not prove cause. The positive sign here arises because, in this dataset, countries with heavier outbreaks also rolled out vaccines faster and have older populations - a confounding effect, not evidence that vaccines raise deaths).
6. **Continents** - average deaths per million: Europe 3172, South America 2699, North America 2152, Asia 1732, Oceania 1233, Africa 796.

## Data quality notes
- 15 negative daily case counts (reporting corrections) set to missing then interpolated.
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
