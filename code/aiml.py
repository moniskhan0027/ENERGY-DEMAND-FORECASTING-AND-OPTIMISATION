# aiml.py
# Forecast + Fast Ensemble + Simple optimizer + Report
# Uses the fast ensemble method (Option A: median + sampled residuals)
#
# Place at:
# C:\Users\sazeb\Desktop\AI ENERGY FORECATING OPTIMISATION\code\aiml.py

import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# PuLP for optimizer (toy LP)
try:
    import pulp
except Exception:
    pulp = None  # optimizer is optional; code will guard

# -----------------------
# Paths (relative)
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # ...\code
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

CSV_IN = os.path.join(DATA_DIR, "india_daily_consumption_2014_2024.csv")
PRED_2025_CSV = os.path.join(RESULTS_DIR, "predicted_2025_consumption.csv")
PRED_ENSEMBLE_CSV = os.path.join(RESULTS_DIR, "predicted_2025_ensembles.csv")
TREND_IMG = os.path.join(RESULTS_DIR, "trend_2014_2025.png")
PDF_REPORT = os.path.join(RESULTS_DIR, "consumption_report_2014_2025.pdf")
MODEL_FILE = os.path.join(MODELS_DIR, "rf_energy_model.pkl")
DISPATCH_SAMPLE = os.path.join(RESULTS_DIR, "dispatch_schedule_sample_day.csv")

# -----------------------
# Settings
# -----------------------
ANNUAL_GROWTH_TARGET = 0.05   # +5% smooth growth throughout 2025
RF_ESTIMATORS = 200
RANDOM_STATE = 42
ENSEMBLE_SIZE = 200  # used for number of residual samples; ensemble generation is vectorized and fast
SAMPLE_ENSEMBLE_MEMBERS = 10  # how many ensemble members to save in CSV

# -----------------------
# Load data
# -----------------------
if not os.path.exists(CSV_IN):
    raise FileNotFoundError(f"Input CSV missing at {CSV_IN}. Place india_daily_consumption_2014_2024.csv in {DATA_DIR}")

df_all = pd.read_csv(CSV_IN, parse_dates=["date"]).set_index("date").sort_index()
if "consumption_mwh" not in df_all.columns:
    raise RuntimeError("CSV must contain 'consumption_mwh' column.")

print("Loaded data:", df_all.shape, "range:", df_all.index.min().date(), "->", df_all.index.max().date())

# -----------------------
# Feature engineering
# -----------------------
df = df_all.copy()
df["year"] = df.index.year
df["month"] = df.index.month
df["dayofyear"] = df.index.dayofyear
df["target"] = df["consumption_mwh"].shift(-1)
df = df.dropna()

FEATURES = ["year", "month", "dayofyear", "consumption_mwh"]

# train/test: train on <2024, test = 2024
train = df[df.index.year < 2024].copy()
test = df[df.index.year == 2024].copy()
if train.empty or test.empty:
    raise RuntimeError("Train or test split empty. Ensure CSV covers 2014-2024.")

X_train = train[FEATURES]; y_train = train["target"]
X_test = test[FEATURES]; y_test = test["target"]

# -----------------------
# Train RandomForest
# -----------------------
print("Training RandomForest...")
model = RandomForestRegressor(n_estimators=RF_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
model.fit(X_train, y_train)
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

y_test_pred = model.predict(X_test)
mae_test = mean_absolute_error(y_test, y_test_pred)
print(f"Model MAE (2024): {mae_test:.2f} MWh")

# residuals for bootstrap
residuals = (y_test.values - y_test_pred).astype(float)
residuals_centered = residuals - np.mean(residuals) if residuals.size > 0 else np.array([0.0])

# -----------------------
# Iterative median forecast for 2025
# -----------------------
dates_2025 = pd.date_range("2025-01-01", "2025-12-31", freq="D")
history_val = float(df_all["consumption_mwh"].iloc[-1])

# clipping bounds to prevent blow-up
q_low = df_all["consumption_mwh"].quantile(0.005)
q_high = df_all["consumption_mwh"].quantile(0.995)
clip_low = q_low * 0.8
clip_high = q_high * 1.2

median_preds = []
print("Computing iterative median predictions for 2025 (with smooth growth ramp)...")
for ts in dates_2025:
    x_row = [ts.year, ts.month, ts.timetuple().tm_yday, history_val]
    x_df = pd.DataFrame([x_row], columns=FEATURES)
    base = float(model.predict(x_df)[0])
    base = float(np.clip(base, clip_low, clip_high))
    frac = ts.timetuple().tm_yday / 365.0
    growth_mult = 1.0 + ANNUAL_GROWTH_TARGET * frac
    final = float(np.clip(base * growth_mult, clip_low * 0.9, clip_high * 1.25))
    median_preds.append(final)
    history_val = final

# -----------------------
# FAST ensemble: vectorized residual sampling around median predictions
# (Option A) — very fast compared to iterative per-member feed-forward
# -----------------------
print("Generating fast ensemble by sampling residuals and adding to median (vectorized)...")
rng = np.random.RandomState(RANDOM_STATE)

median_array = np.array(median_preds)  # shape (365,)
# If residuals array is tiny, create small Gaussian residuals fallback
if residuals_centered.size == 0:
    residuals_centered = np.random.normal(0.0, np.std(df_all["consumption_mwh"]) * 0.01, size=500)

# Draw residual samples (days x ensemble_size)
samples = rng.choice(residuals_centered, size=(len(median_array), ENSEMBLE_SIZE), replace=True)
ensemble_preds = median_array.reshape(-1, 1) + samples
# Clip ensemble to reasonable band
ensemble_preds = np.clip(ensemble_preds, clip_low * 0.85, clip_high * 1.3)

# Compute percentiles (10th, 50th, 90th)
q10 = np.percentile(ensemble_preds, 10, axis=1)
q50 = np.percentile(ensemble_preds, 50, axis=1)
q90 = np.percentile(ensemble_preds, 90, axis=1)

# Save median + intervals
pred_df = pd.DataFrame({
    "date": dates_2025,
    "median_pred_mwh": median_array,
    "p10_mwh": q10,
    "p50_mwh": q50,
    "p90_mwh": q90
}).set_index("date")
pred_df.to_csv(PRED_2025_CSV)
print("Saved median + 10/50/90 percentiles to:", PRED_2025_CSV)

# Save a subset of ensemble members for inspection
n_save = min(SAMPLE_ENSEMBLE_MEMBERS, ENSEMBLE_SIZE)
ens_df = pd.DataFrame(ensemble_preds[:, :n_save], index=dates_2025,
                      columns=[f"ens_{k+1}" for k in range(n_save)])
ens_df.to_csv(PRED_ENSEMBLE_CSV)
print("Saved sample ensemble members to:", PRED_ENSEMBLE_CSV)

# -----------------------
# Optional: create a toy hourly profile for a sample day and run a simple optimizer (if pulp available)
# -----------------------
def make_hourly_from_daily(daily_val, date):
    idx = pd.date_range(date, periods=24, freq="H")
    hourly_demand = np.repeat(daily_val / 24.0, 24)
    dfh = pd.DataFrame({"demand": hourly_demand}, index=idx)
    # toy solar profile
    hours = dfh.index.hour
    solar = np.where((hours >= 6) & (hours <= 18), 0.2 * np.sin((hours - 6) / 12.0 * np.pi) * daily_val * 0.02, 0.0)
    dfh["solar"] = solar
    # price profile: night cheap, evening peak
    dfh["price"] = 0.10
    dfh.loc[dfh.index.hour.isin([17,18,19,20]), "price"] = 0.20
    return dfh

def optimize_day_simple(hourly_df, battery_capacity=500000.0, battery_power=50000.0, eff=0.95):
    if pulp is None:
        print("PuLP not installed — skipping optimizer.")
        return None
    hours = list(range(len(hourly_df)))
    prob = pulp.LpProblem("dispatch", pulp.LpMinimize)
    grid = pulp.LpVariable.dicts("grid", hours, lowBound=0)
    charge = pulp.LpVariable.dicts("charge", hours, lowBound=0, upBound=battery_power)
    discharge = pulp.LpVariable.dicts("discharge", hours, lowBound=0, upBound=battery_power)
    soc = pulp.LpVariable.dicts("soc", hours, lowBound=0, upBound=battery_capacity)
    prob += pulp.lpSum([grid[h] * float(hourly_df["price"].iloc[h]) for h in hours])
    for h in hours:
        demand = float(hourly_df["demand"].iloc[h])
        solar = float(hourly_df["solar"].iloc[h])
        demand_to_meet = max(0.0, demand - solar)
        prob += grid[h] + discharge[h] >= demand_to_meet + charge[h] - 1e-6
        if h == 0:
            prob += soc[h] == 0.5 * battery_capacity + (charge[h] * eff - discharge[h] / eff)
        else:
            prob += soc[h] == soc[h-1] + (charge[h] * eff - discharge[h] / eff)
    prob += soc[hours[-1]] >= 0.5 * battery_capacity
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    sol = {"grid": [], "charge": [], "discharge": [], "soc": [], "price": []}
    for h in hours:
        sol["grid"].append(pulp.value(grid[h]) or 0.0)
        sol["charge"].append(pulp.value(charge[h]) or 0.0)
        sol["discharge"].append(pulp.value(discharge[h]) or 0.0)
        sol["soc"].append(pulp.value(soc[h]) or 0.0)
        sol["price"].append(float(hourly_df["price"].iloc[h]))
    sol_df = pd.DataFrame(sol, index=hourly_df.index)
    sol_df["demand"] = hourly_df["demand"]
    sol_df["solar"] = hourly_df["solar"]
    return sol_df

# run sample day optimizer on mid-year day
sample_date = pred_df.index[len(pred_df)//2]
hourly_profile = make_hourly_from_daily(pred_df.loc[sample_date, "median_pred_mwh"], sample_date)
dispatch = optimize_day_simple(hourly_profile)
if dispatch is not None:
    dispatch.to_csv(DISPATCH_SAMPLE)
    print("Saved sample dispatch schedule to:", DISPATCH_SAMPLE)

# -----------------------
# Plot trend (historical yearly averages) + predicted 2025 with shaded PI
# -----------------------
hist_yearly = df_all["consumption_mwh"].resample("Y").mean()
hist_years = hist_yearly.index.year.tolist()
first_year = hist_years[0]
all_years = list(range(first_year, 2025 + 1))
plot_vals = []
for y in all_years:
    if y in hist_years:
        # year-end index like '2014-12-31'
        plot_vals.append(float(hist_yearly.loc[f"{y}-12-31"]))
    elif y == 2025:
        plot_vals.append(float(pred_df["median_pred_mwh"].mean()))
    else:
        plot_vals.append(np.nan)

plt.figure(figsize=(12,6))
# historical line
years_hist = all_years[:-1]
vals_hist = plot_vals[:-1]
plt.plot(years_hist, vals_hist, marker="o", color="#2c7fb8", label="Historical (2014-2024)")
# connector to 2025
plt.plot([2024, 2025], [vals_hist[-1], plot_vals[-1]], marker="o", color="#de2d26", linewidth=2, label="Predicted (2025)")
# shaded PI band for 2025 (use mean of p10/p90 across days for display)
pi10_mean = np.mean(pred_df["p10_mwh"])
pi90_mean = np.mean(pred_df["p90_mwh"])
plt.fill_between([2025-0.2, 2025+0.2], [pi10_mean, pi10_mean], [pi90_mean, pi90_mean], color="#fdd0a2", alpha=0.6, label="10-90% PI (mean)")

plt.title("Avg Daily Consumption: Historical (2014-2024) vs Predicted 2025")
plt.xlabel("Year")
plt.ylabel("Avg Daily Consumption (MWh/day)")
plt.xticks(all_years, rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(TREND_IMG, dpi=200)
plt.close()
print("Saved trend image to:", TREND_IMG)

# -----------------------
# PDF report
# -----------------------
styles = getSampleStyleSheet()
doc = SimpleDocTemplate(PDF_REPORT, pagesize=letter)
story = []
story.append(Paragraph("<b>AI Energy Forecasting & Optimisation — Report</b>", styles["Title"]))
story.append(Spacer(1,6))
story.append(Paragraph(f"Historical range: {df_all.index.min().date()} to {df_all.index.max().date()}", styles["BodyText"]))
story.append(Spacer(1,6))
story.append(Paragraph(f"Model MAE (2024): {mae_test:.2f} MWh", styles["BodyText"]))
story.append(Spacer(1,6))
story.append(Paragraph(f"Predicted avg daily consumption 2025 (median): {np.mean(median_array):,.2f} MWh/day", styles["BodyText"]))
story.append(Spacer(1,12))
story.append(Paragraph("<b>Trend (Historical vs Predicted)</b>", styles["Heading2"]))
story.append(Spacer(1,8))
story.append(Image(TREND_IMG, width=480, height=320))
story.append(Spacer(1,12))
if dispatch is not None:
    story.append(Paragraph("<b>Sample dispatch schedule (one day)</b>", styles["Heading2"]))
    story.append(Spacer(1,6))
    story.append(Paragraph(f"Dispatch saved at: {DISPATCH_SAMPLE}", styles["BodyText"]))
doc.build(story)
print("Saved PDF report to:", PDF_REPORT)

# -----------------------
# Done
# -----------------------
print("\n=== OUTPUTS ===")
print("predicted daily 2025 (median + intervals):", PRED_2025_CSV)
print("sample ensemble CSV:", PRED_ENSEMBLE_CSV)
print("trend image:", TREND_IMG)
print("pdf report:", PDF_REPORT)
print("model file:", MODEL_FILE)
if dispatch is not None:
    print("sample dispatch schedule:", DISPATCH_SAMPLE)
print("All done.")
