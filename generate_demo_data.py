"""
Generate a synthetic demo dataset that looks statistically similar to the
original (private) hackathon dataset, without copying any real record.

How it works:
- Each column is sampled from the original column's value distribution.
- A few realistic links are kept: month -> season, month -> workload/target,
  reason for absence -> absence category, and age -> BMI (with random noise).
- Weight is recalculated from the synthetic BMI and height.
- A final check makes sure no synthetic row is identical to a real row.

This script needs the private file data/absent_cleaned.csv, which is NOT
uploaded to GitHub (see .gitignore). Only the output, demo_data.csv, is public.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
real = pd.read_csv("data/absent_cleaned.csv")
real["Work load"] = real["Work load"].astype(str).str.replace(",", "").astype(float)
n = len(real)


def sample(col, size=n):
    """Draw values from a column's real frequency distribution."""
    freq = real[col].value_counts(normalize=True)
    return rng.choice(freq.index.to_numpy(), size=size, p=freq.to_numpy())


demo = pd.DataFrame({"ID": np.arange(1, n + 1)})

# Independent columns
for col in ["Reason for absence", "Month of absence", "Day of the week",
            "Transportation expense", "Distance from Residence to Work", "Service time",
            "Age", "Disciplinary failure", "Education", "Son", "Social drinker",
            "Social smoker", "Pet", "Height"]:
    demo[col] = sample(col)

# Month -> season (most common season recorded for each month)
season_of_month = real.groupby("Month of absence")["Seasons"].agg(lambda s: s.mode()[0])
demo["Seasons"] = demo["Month of absence"].map(season_of_month).fillna(1).astype(int)

# Month -> workload and hit target (drawn from the same month in the original data)
wl, ht = [], []
for m in demo["Month of absence"]:
    pool = real[real["Month of absence"] == m]
    wl.append(rng.choice(pool["Work load"].to_numpy()))
    ht.append(rng.choice(pool["Hit target"].to_numpy()))
demo["Work load"] = [f"{int(w):,}" for w in wl]
demo["Hit target"] = ht

# Age -> BMI (linear trend from the original data plus random noise)
slope, intercept = np.polyfit(real["Age"], real["Body mass index"], 1)
resid_sd = (real["Body mass index"] - (slope * real["Age"] + intercept)).std()
bmi = slope * demo["Age"] + intercept + rng.normal(0, resid_sd, n)
demo["Body mass index"] = np.clip(bmi.round(), real["Body mass index"].min(),
                                  real["Body mass index"].max()).astype(int)
demo["Weight"] = (demo["Body mass index"] * (demo["Height"] / 100) ** 2).round()

# Reason for absence -> absence category (1 = Time Slip, 2 = MC, 3 = Abnormal)
label_probs = pd.crosstab(real["Reason for absence"], real["Absenteeism time in hours"],
                          normalize="index")
demo["Absenteeism time in hours"] = [
    rng.choice(label_probs.columns.to_numpy(), p=label_probs.loc[r].to_numpy())
    for r in demo["Reason for absence"]
]

demo = demo[real.columns]

# Privacy check - no synthetic row may match a real row
cols = [c for c in real.columns if c not in ("ID", "Work load")]
matches = demo[cols].merge(real[cols].drop_duplicates(), how="inner")
if len(matches):
    demo = demo.drop(demo.index[demo[cols].apply(tuple, axis=1).isin(matches.apply(tuple, axis=1))])
print(f"Rows removed because they matched a real record: {len(matches)}")

demo.to_csv("demo_data.csv", index=False)
print(f"Saved demo_data.csv with {len(demo)} synthetic rows.")
