"""
Generates a COVID-19-style dataset (covid_data.csv) for Lab 09 Project 1.

NOTE: this is a SIMULATED dataset (epidemic waves, reporting noise, vaccination rollout and
deliberately injected missing values). It has the same column layout as the real
"Our World in Data" / JHU style files, so covid_dashboard.py also works on a real CSV:
replace covid_data.csv with a real export that has the same column names.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)


def lag(a, k):
    """Shift an array k days into the future, filling the start with zeros (no wrap-around)."""
    out = np.zeros_like(a)
    out[k:] = a[:-k]
    return out


# country, continent, population (millions), rough severity, vaccination speed
COUNTRIES = [
    ("India", "Asia", 1400, 1.0, 0.9), ("USA", "North America", 331, 1.9, 0.8),
    ("Brazil", "South America", 214, 1.7, 0.85), ("Russia", "Europe", 146, 1.2, 0.6),
    ("France", "Europe", 67, 1.8, 1.0), ("UK", "Europe", 68, 1.9, 1.05),
    ("Germany", "Europe", 83, 1.5, 1.0), ("Italy", "Europe", 59, 1.7, 1.0),
    ("Spain", "Europe", 47, 1.6, 1.1), ("Turkey", "Asia", 85, 1.3, 0.8),
    ("Japan", "Asia", 125, 0.6, 1.0), ("South Korea", "Asia", 52, 0.9, 1.0),
    ("Indonesia", "Asia", 274, 0.7, 0.8), ("Mexico", "North America", 128, 1.0, 0.7),
    ("Argentina", "South America", 45, 1.5, 0.9), ("South Africa", "Africa", 60, 0.8, 0.55),
    ("Nigeria", "Africa", 211, 0.2, 0.3), ("Egypt", "Africa", 104, 0.3, 0.5),
    ("Australia", "Oceania", 26, 0.5, 1.0), ("Canada", "North America", 38, 1.1, 1.1),
]
dates = pd.date_range("2020-03-01", "2022-12-31", freq="D")
t = np.arange(len(dates))

# epidemic waves (peak day index, width, relative height)
WAVES = [(60, 25, 0.5), (250, 40, 0.9), (330, 30, 0.6), (480, 35, 1.0),
         (600, 30, 0.8), (700, 25, 2.2), (760, 20, 1.6), (900, 40, 0.7)]

frames = []
for name, cont, pop_m, sev, vspeed in COUNTRIES:
    pop = pop_m * 1e6
    shift = rng.integers(-25, 25)
    base = sum(h * np.exp(-0.5 * ((t - (c + shift)) / w) ** 2) for c, w, h in WAVES)
    base *= rng.uniform(0.7, 1.3, len(WAVES)).mean()
    daily_rate = 0.0004 * sev * base                         # fraction of population / day
    new_cases = rng.poisson(np.maximum(pop * daily_rate / 3, 0)) * 3
    # weekend reporting dips
    new_cases = (new_cases * np.where(dates.dayofweek >= 5, 0.7, 1.0)).astype(int)
    cfr = 0.02 * np.exp(-t / 500) + 0.002            # fatality falls over time (vaccines/treatment)
    new_deaths = rng.poisson(lag(new_cases, 10) * cfr * rng.uniform(0.8, 1.2))
    new_rec = rng.poisson(lag(new_cases, 14) * 0.97)
    
    # vaccination: starts Dec 2020 (day ~275), logistic rollout
    start = 275 + rng.integers(0, 40)
    v_frac = np.where(t > start, 1 / (1 + np.exp(-(t - (start + 220 / vspeed)) / (60 / vspeed))), 0)
    ceiling = min(0.92, 0.45 + 0.4 * vspeed)
    one_dose = pop * ceiling * v_frac
    full = pop * ceiling * 0.9 * lag(v_frac, 40)
    total_vacc = one_dose + full * 0.95 + pop * 0.15 * lag(v_frac, 220)

    df = pd.DataFrame({
        "date": dates, "country": name, "continent": cont, "population": int(pop),
        "new_cases": new_cases, "new_deaths": new_deaths, "new_recoveries": new_rec,
    })
    df["total_cases"] = df.new_cases.cumsum()
    df["total_deaths"] = df.new_deaths.cumsum()
    df["total_recoveries"] = df.new_recoveries.cumsum()
    df["people_vaccinated"] = one_dose.astype("int64")
    df["people_fully_vaccinated"] = full.astype("int64")
    df["total_vaccinations"] = total_vacc.astype("int64")
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

# ---- inject realistic data problems -----------------------------------------
n = len(data)
for col, frac in [("new_cases", 0.02), ("new_deaths", 0.03), ("new_recoveries", 0.06),
                  ("total_recoveries", 0.05), ("people_vaccinated", 0.04),
                  ("people_fully_vaccinated", 0.05), ("total_vaccinations", 0.04)]:
    data.loc[rng.choice(n, int(n * frac), replace=False), col] = np.nan
# one country stopped reporting recoveries entirely in 2022
data.loc[(data.country == "UK") & (data.date >= "2022-02-01"), ["new_recoveries", "total_recoveries"]] = np.nan
# a handful of negative data corrections (happens in real data)
neg = rng.choice(data.index[data.new_cases.notna()], 15, replace=False)
data.loc[neg, "new_cases"] = -rng.integers(10, 500, 15)

data.to_csv("covid_data.csv", index=False)
print(f"covid_data.csv written: {data.shape[0]:,} rows x {data.shape[1]} cols, "
      f"{data.country.nunique()} countries, {dates[0].date()} -> {dates[-1].date()}")
