#!/usr/bin/env python3
"""
Corrected analysis: Platform Maturity and Research-to-Patent Translation
========================================================================
Omidi et al. — Polymer Drug-Delivery Research (2015–2025)

Changes from the original notebook:
  1. Uses Publication Year consistently (manuscript must say "publication year")
  2. Drops polymeric_micelle from regression, lag, and forecast analyses
  3. Keeps polymeric_micelle in descriptive Table 1
  4. Fixes lag interpretation (correctly identifies peak direction)
  5. Runs on the current CSV (6,684 patents) — all numbers updated
  6. Removes duplicate/stale code
  7. Cleans chi-square test (excludes NaN rows)
  8. Fixes forecast description (uses recent 3-year medians, not future data)
"""

import pandas as pd
import numpy as np
import os, json, warnings
from scipy.stats import pearsonr, spearmanr, chi2_contingency
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 0. Setup
# ─────────────────────────────────────────────────────────────
BASE_OUT = "outputs"
FIG_DIR  = os.path.join(BASE_OUT, "figures")
TAB_DIR  = os.path.join(BASE_OUT, "tables")
for d in (FIG_DIR, TAB_DIR):
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────
patents = pd.read_csv("all_patents_cleaned.csv")
nih     = pd.read_csv("nih_polymer_drug_delivery_cleaned.csv")

# Ensure correct types
nih["total_cost"]   = pd.to_numeric(nih["total_cost"], errors="coerce")
nih["fiscal_year"]  = pd.to_numeric(nih["fiscal_year"], errors="coerce").astype(int)
patents["Publication Year"] = pd.to_numeric(patents["Publication Year"], errors="coerce")

# Standardize polymer column
if "Polymer" in patents.columns and "polymer" not in patents.columns:
    patents.rename(columns={"Polymer": "polymer"}, inplace=True)

# Filter patents to 2015-2025 by Publication Year
patents = patents[patents["Publication Year"].between(2015, 2025)].copy()

print(f"NIH projects: {len(nih)}")
print(f"Patents (2015–2025 by publication year): {len(patents)}")
print(f"NIH polymers: {sorted(nih['polymer'].unique())}")
print(f"Patent polymers: {sorted(patents['polymer'].unique())}")

# ─────────────────────────────────────────────────────────────
# 2. Table 1 — Descriptive overview (ALL polymers incl. PM)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TABLE 1: NIH Activity and Patent Output by Polymer (2015–2025)")
print("="*70)

# NIH by polymer
nih_by_poly = (
    nih.groupby("polymer", as_index=False)
       .agg(nih_projects=("application_id", "nunique"),
            nih_funding=("total_cost", "sum"))
)
nih_by_poly["nih_funding_m"] = nih_by_poly["nih_funding"] / 1e6

# Patents by polymer (using unique Lens ID)
pat_by_poly = (
    patents.dropna(subset=["Lens ID"])
           .groupby("polymer")["Lens ID"]
           .nunique()
           .reset_index(name="patents_n")
)

# Merge
tbl1 = nih_by_poly.merge(pat_by_poly, on="polymer", how="outer").fillna(0)
tbl1["patents_per_project"] = np.where(
    tbl1["nih_projects"] > 0, tbl1["patents_n"] / tbl1["nih_projects"], np.nan
)
tbl1["patents_per_$m"] = np.where(
    tbl1["nih_funding_m"] > 0, tbl1["patents_n"] / tbl1["nih_funding_m"], np.nan
)

# Clean display
tbl1 = tbl1[["polymer", "nih_projects", "nih_funding_m", "patents_n",
             "patents_per_project", "patents_per_$m"]].copy()
tbl1.columns = ["Polymer", "NIH Projects (n)", "NIH Funding ($M)",
                 "Patents (n)", "Patents/Project", "Patents/$M"]
tbl1 = tbl1.sort_values("Polymer").reset_index(drop=True)

for c in ["NIH Funding ($M)", "Patents/Project", "Patents/$M"]:
    tbl1[c] = tbl1[c].round(2)

print(tbl1.to_string(index=False))
tbl1.to_csv(os.path.join(TAB_DIR, "table1_polymer_overview.csv"), index=False)

# Total counts for manuscript
total_nih = int(tbl1["NIH Projects (n)"].sum())
total_funding_m = tbl1["NIH Funding ($M)"].sum()
total_patents = int(tbl1["Patents (n)"].sum())
print(f"\nTotals: {total_nih} NIH projects, ${total_funding_m:.1f}M funding, {total_patents} patents")

# ─────────────────────────────────────────────────────────────
# 3. Chi-square test (6 polymers with NIH data only)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("CHI-SQUARE TESTS")
print("="*70)

# 6-polymer test (excluding polymeric_micelle which has 0 NIH projects)
tbl_6 = tbl1[tbl1["NIH Projects (n)"] > 0].copy()
contingency_6 = np.array([
    tbl_6["NIH Projects (n)"].values.astype(int),
    tbl_6["Patents (n)"].values.astype(int)
])
chi2_6, p_6, dof_6, _ = chi2_contingency(contingency_6)
print(f"6 polymers (excl. polymeric micelle): χ²={chi2_6:.2f}, df={dof_6}, p={p_6:.2e}")

# 7-polymer test (as in original — keep for comparison)
contingency_7 = np.array([
    tbl1["NIH Projects (n)"].values.astype(int),
    tbl1["Patents (n)"].values.astype(int)
])
chi2_7, p_7, dof_7, _ = chi2_contingency(contingency_7)
print(f"7 polymers (incl. polymeric micelle): χ²={chi2_7:.2f}, df={dof_7}, p={p_7:.2e}")

# ─────────────────────────────────────────────────────────────
# 4. Aggregate trends & same-year correlations
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("AGGREGATE TEMPORAL TRENDS")
print("="*70)

# Annual patent counts (publication year, all polymers)
patent_year_counts = (
    patents.groupby("Publication Year").size()
    .reset_index(name="patent_count")
    .rename(columns={"Publication Year": "year"})
)

# Annual NIH counts
nih_year_agg = (
    nih.groupby("fiscal_year", as_index=False)
       .agg(nih_count=("project_number", "count"),
            nih_funding=("total_cost", "sum"))
       .rename(columns={"fiscal_year": "year"})
)

# Merge
year_trends = (
    patent_year_counts.merge(nih_year_agg, on="year", how="outer")
    .fillna(0)
    .sort_values("year")
    .reset_index(drop=True)
)
year_trends["year"] = year_trends["year"].astype(int)
year_trends["nih_funding_m"] = year_trends["nih_funding"] / 1e6

# Peaks
pat_peak_yr  = year_trends.loc[year_trends["patent_count"].idxmax(), "year"]
pat_peak_val = year_trends["patent_count"].max()
proj_peak_yr = year_trends.loc[year_trends["nih_count"].idxmax(), "year"]
proj_peak_val= year_trends["nih_count"].max()
fund_peak_yr = year_trends.loc[year_trends["nih_funding_m"].idxmax(), "year"]

print(f"Patent peak: {pat_peak_yr} (n={int(pat_peak_val)})")
print(f"NIH project peak: {proj_peak_yr} (n={int(proj_peak_val)})")
print(f"NIH funding peak: {fund_peak_yr}")

# Same-year correlations
def corr_report(x, y, name):
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    print(f"\n  {name}:")
    print(f"    Pearson  r = {pr:.2f}, p = {pp:.4f}")
    print(f"    Spearman ρ = {sr:.2f}, p = {sp:.4f}")
    return pr, pp, sr, sp

print("\nSame-year correlations:")
proj_corr = corr_report(year_trends["nih_count"], year_trends["patent_count"],
                        "NIH Projects vs Patents")
fund_corr = corr_report(year_trends["nih_funding"], year_trends["patent_count"],
                        "NIH Funding vs Patents")

year_trends.to_csv(os.path.join(TAB_DIR, "year_trends.csv"), index=False)

# ─────────────────────────────────────────────────────────────
# 5. Lag correlation analysis (CORRECTED direction labels)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("LAG CORRELATION ANALYSIS (Corrected)")
print("="*70)

def lag_corr_df(x, y, method="pearson", lags=range(-3, 4)):
    """
    Lag convention:
      Positive lag (e.g., +1): x leads y (NIH at t compared to Patents at t+1)
      Negative lag (e.g., -1): y leads x (Patents at t compared to NIH at t+1)
    """
    x = pd.Series(x).astype(float).reset_index(drop=True)
    y = pd.Series(y).astype(float).reset_index(drop=True)
    out = []
    for l in lags:
        if l > 0:
            x_l, y_l = x.iloc[:-l], y.iloc[l:]
        elif l < 0:
            x_l, y_l = x.iloc[-l:], y.iloc[:l]
        else:
            x_l, y_l = x, y
        n = min(len(x_l), len(y_l))
        x_l, y_l = x_l.iloc[:n], y_l.iloc[:n]
        if n < 3:
            out.append({"lag": l, "n": n, "r": np.nan, "p": np.nan})
            continue
        if method.lower().startswith("pear"):
            r, p = pearsonr(x_l, y_l)
        else:
            r, p = spearmanr(x_l, y_l)
        out.append({"lag": l, "n": n, "r": r, "p": p})
    df = pd.DataFrame(out).sort_values("lag").reset_index(drop=True)
    df["abs_r"] = df["r"].abs()
    return df

# Compute all lag tables
proj_pear  = lag_corr_df(year_trends["nih_count"],   year_trends["patent_count"], "pearson")
proj_spear = lag_corr_df(year_trends["nih_count"],   year_trends["patent_count"], "spearman")
fund_pear  = lag_corr_df(year_trends["nih_funding"], year_trends["patent_count"], "pearson")
fund_spear = lag_corr_df(year_trends["nih_funding"], year_trends["patent_count"], "spearman")

# Merged table
lag_table = pd.DataFrame({
    "Lag": proj_pear["lag"],
    "N": proj_pear["n"],
    "Pearson r (Projects)":  proj_pear["r"].round(3),
    "Pearson p (Projects)":  proj_pear["p"].round(4),
    "Spearman ρ (Projects)": proj_spear["r"].round(3),
    "Spearman p (Projects)": proj_spear["p"].round(4),
    "Pearson r (Funding)":   fund_pear["r"].round(3),
    "Pearson p (Funding)":   fund_pear["p"].round(4),
    "Spearman ρ (Funding)":  fund_spear["r"].round(3),
    "Spearman p (Funding)":  fund_spear["p"].round(4),
})

print("\nFull lag table (positive = NIH leads, negative = Patents lead):")
print(lag_table.to_string(index=False))

# Identify peaks correctly
fund_spear_peak_idx = fund_spear["abs_r"].idxmax()
fund_spear_peak_lag = int(fund_spear.loc[fund_spear_peak_idx, "lag"])
fund_spear_peak_r   = fund_spear.loc[fund_spear_peak_idx, "r"]
fund_spear_peak_p   = fund_spear.loc[fund_spear_peak_idx, "p"]

fund_pear_peak_idx = fund_pear["abs_r"].idxmax()
fund_pear_peak_lag = int(fund_pear.loc[fund_pear_peak_idx, "lag"])
fund_pear_peak_r   = fund_pear.loc[fund_pear_peak_idx, "r"]
fund_pear_peak_p   = fund_pear.loc[fund_pear_peak_idx, "p"]

print(f"\n  Peak Spearman ρ (Funding→Patents): ρ={fund_spear_peak_r:.3f}, p={fund_spear_peak_p:.4f} at lag={fund_spear_peak_lag}")
if fund_spear_peak_lag < 0:
    print(f"    → Patents LEAD NIH funding by {abs(fund_spear_peak_lag)} year(s)")
elif fund_spear_peak_lag > 0:
    print(f"    → NIH funding LEADS patents by {fund_spear_peak_lag} year(s)")
else:
    print(f"    → Same-year alignment")

print(f"  Peak Pearson r (Funding→Patents): r={fund_pear_peak_r:.3f}, p={fund_pear_peak_p:.4f} at lag={fund_pear_peak_lag}")
if fund_pear_peak_lag < 0:
    print(f"    → Patents LEAD NIH funding by {abs(fund_pear_peak_lag)} year(s)")
elif fund_pear_peak_lag > 0:
    print(f"    → NIH funding LEADS patents by {fund_pear_peak_lag} year(s)")
else:
    print(f"    → Same-year alignment")

# Also report the best positive-lag (NIH leads) for completeness
pos_fund_spear = fund_spear[fund_spear["lag"] > 0]
if not pos_fund_spear.empty:
    best_pos = pos_fund_spear.loc[pos_fund_spear["abs_r"].idxmax()]
    print(f"\n  Best POSITIVE lag (NIH → Patents): Spearman ρ={best_pos['r']:.3f}, p={best_pos['p']:.4f} at lag=+{int(best_pos['lag'])}")

lag_table.to_csv(os.path.join(TAB_DIR, "lag_correlation_table_corrected.csv"), index=False)

# ─────────────────────────────────────────────────────────────
# 6. Polymer-specific lag correlations (EXCLUDING polymeric_micelle)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("POLYMER-SPECIFIC LAG CORRELATIONS (Excluding polymeric_micelle)")
print("="*70)

# Build polymer–year panel
nih_poly_year = (
    nih.groupby(["polymer", "fiscal_year"], as_index=False)
       .agg(nih_projects=("application_id", "nunique"),
            funding_usd=("total_cost", "sum"))
       .rename(columns={"fiscal_year": "year"})
)

pat_poly_year = (
    patents.groupby(["polymer", "Publication Year"]).size()
    .reset_index(name="patent_count")
    .rename(columns={"Publication Year": "year"})
)
pat_poly_year["year"] = pat_poly_year["year"].astype(int)

panel_full = (
    nih_poly_year.merge(pat_poly_year, on=["polymer", "year"], how="outer")
    .fillna({"nih_projects": 0, "funding_usd": 0, "patent_count": 0})
    .sort_values(["polymer", "year"])
    .reset_index(drop=True)
)

def lag_corr_single(x, y, lag=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if lag > 0:
        x, y = x[:-lag], y[lag:]
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    # Check for constant input
    if np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return np.nan, np.nan
    return pearsonr(x[mask], y[mask])

rows = []
# Exclude polymeric_micelle (zero NIH across all years)
analysis_polymers = [p for p in sorted(panel_full["polymer"].unique()) if p != "polymeric_micelle"]

for poly in analysis_polymers:
    sub = panel_full[panel_full["polymer"] == poly].sort_values("year")
    if sub["year"].nunique() < 3:
        continue

    proj_corrs = [lag_corr_single(sub["nih_projects"], sub["patent_count"], lag=L) for L in (0,1,2,3)]
    fund_corrs = [lag_corr_single(sub["funding_usd"],  sub["patent_count"], lag=L) for L in (0,1,2,3)]

    def best_lag(corrs):
        valid = [(i, abs(c[0])) for i, c in enumerate(corrs) if not np.isnan(c[0])]
        return max(valid, key=lambda x: x[1])[0] if valid else 0

    bp = best_lag(proj_corrs)
    bf = best_lag(fund_corrs)

    row = {"polymer": poly}
    for i in range(4):
        row[f"proj_r{i}"] = round(proj_corrs[i][0], 3) if not np.isnan(proj_corrs[i][0]) else np.nan
        row[f"proj_p{i}"] = round(proj_corrs[i][1], 4) if not np.isnan(proj_corrs[i][1]) else np.nan
        row[f"fund_r{i}"] = round(fund_corrs[i][0], 3) if not np.isnan(fund_corrs[i][0]) else np.nan
        row[f"fund_p{i}"] = round(fund_corrs[i][1], 4) if not np.isnan(fund_corrs[i][1]) else np.nan
    row["proj_best_lag"] = bp
    row["proj_best_r"]   = round(proj_corrs[bp][0], 3) if not np.isnan(proj_corrs[bp][0]) else np.nan
    row["fund_best_lag"] = bf
    row["fund_best_r"]   = round(fund_corrs[bf][0], 3) if not np.isnan(fund_corrs[bf][0]) else np.nan
    rows.append(row)

poly_lag_summary = pd.DataFrame(rows)
print(poly_lag_summary[["polymer", "proj_r0", "proj_p0", "proj_best_lag", "proj_best_r",
                         "fund_r0", "fund_p0", "fund_best_lag", "fund_best_r"]].to_string(index=False))
poly_lag_summary.to_csv(os.path.join(TAB_DIR, "polymer_lag_correlations_corrected.csv"), index=False)

# ─────────────────────────────────────────────────────────────
# 7. Regression modeling (EXCLUDING polymeric_micelle)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("REGRESSION MODELING (Excluding polymeric_micelle)")
print("="*70)

# Build panel for modeling (6 polymers only)
nih_yr = (
    nih.groupby(["polymer", "fiscal_year"], as_index=False)
       .agg(nih_count=("application_id", "nunique"),
            nih_funding=("total_cost", "sum"))
       .rename(columns={"fiscal_year": "year"})
)
pat_yr = (
    patents[patents["polymer"] != "polymeric_micelle"]
    .groupby(["polymer", "Publication Year"]).size()
    .reset_index(name="patent_count")
    .rename(columns={"Publication Year": "year"})
)
pat_yr["year"] = pat_yr["year"].astype(int)

panel = (
    nih_yr.merge(pat_yr, on=["polymer", "year"], how="outer")
    .fillna(0)
    .sort_values(["polymer", "year"])
    .reset_index(drop=True)
)

# Create t+1 target
panel["patent_next"] = panel.groupby("polymer", sort=False)["patent_count"].shift(-1)
panel_ml = panel.dropna(subset=["patent_next"]).copy()

# Features
panel_ml["log_funding"] = np.log1p(panel_ml["nih_funding"])
panel_ml["year_trend"]  = panel_ml["year"] - panel_ml["year"].min()

X_base = pd.get_dummies(
    panel_ml[["log_funding", "nih_count", "year_trend", "polymer"]],
    columns=["polymer"], drop_first=True
).astype(float)

y = panel_ml["patent_next"].astype(int)

# Train/test split (chronological)
train_mask = panel_ml["year"] <= 2022
X_train, X_test = X_base.loc[train_mask].copy(), X_base.loc[~train_mask].copy()
y_train, y_test = y.loc[train_mask], y.loc[~train_mask]

# Drop constant columns
keep_cols = X_train.nunique(dropna=False) > 1
X_train = X_train.loc[:, keep_cols]
X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

X_train_c = sm.add_constant(X_train, has_constant="add")
X_test_c  = sm.add_constant(X_test, has_constant="add")

print(f"Training: {len(y_train)} obs (≤2022), Test: {len(y_test)} obs (2023–2024)")
print(f"Polymers in model: {sorted(panel_ml['polymer'].unique())}")

# ── Poisson ──
poi_res = sm.GLM(y_train, X_train_c, family=sm.families.Poisson()).fit(cov_type="HC1")
disp = poi_res.deviance / poi_res.df_resid
pred_poi = np.clip(poi_res.predict(X_test_c), 0, None)
rmse_poi = np.sqrt(mean_squared_error(y_test, pred_poi))
mae_poi  = mean_absolute_error(y_test, pred_poi)

print(f"\nPoisson: dispersion={disp:.3f}, RMSE={rmse_poi:.3f}, MAE={mae_poi:.3f}")
print(poi_res.summary())

# ── Negative Binomial ──
nb_res = NegativeBinomial(y_train, X_train_c).fit(method="bfgs", maxiter=1000, disp=False)
pred_nb = np.clip(nb_res.predict(X_test_c), 0, None)
rmse_nb = np.sqrt(mean_squared_error(y_test, pred_nb))
mae_nb  = mean_absolute_error(y_test, pred_nb)

print(f"\nNeg Binomial: α={nb_res.params.iloc[-1]:.4f}, RMSE={rmse_nb:.3f}, MAE={mae_nb:.3f}, Pseudo R²={nb_res.prsquared:.4f}")
print(nb_res.summary())

# Choose best model
if rmse_nb <= rmse_poi:
    best_model = nb_res
    best_name  = "Negative Binomial"
    best_rmse, best_mae = rmse_nb, mae_nb
else:
    best_model = poi_res
    best_name  = "Poisson"
    best_rmse, best_mae = rmse_poi, mae_poi

print(f"\nSelected model: {best_name}")

# ── Coefficient table for manuscript ──
print("\n--- Table 3 (Corrected): NB Regression Coefficients ---")
coef_rows = []
exog_names = nb_res.model.exog_names
for i in range(len(nb_res.params) - 1):  # exclude alpha from main table
    name = exog_names[i]
    coef = nb_res.params.iloc[i]
    se   = nb_res.bse.iloc[i]
    pv   = nb_res.pvalues.iloc[i]
    ci   = nb_res.conf_int().iloc[i]
    sig  = "***" if pv<0.001 else "**" if pv<0.01 else "*" if pv<0.05 else "ns"
    coef_rows.append({
        "Variable": name, "β": round(coef, 4), "SE": round(se, 4),
        "p": round(pv, 4), "CI_low": round(ci.iloc[0], 3), "CI_high": round(ci.iloc[1], 3),
        "Sig": sig
    })

# Add alpha
coef_rows.append({
    "Variable": "alpha (dispersion)",
    "β": round(nb_res.params.iloc[-1], 4),
    "SE": round(nb_res.bse.iloc[-1], 4),
    "p": round(nb_res.pvalues.iloc[-1], 4),
    "CI_low": round(nb_res.conf_int().iloc[-1].iloc[0], 3),
    "CI_high": round(nb_res.conf_int().iloc[-1].iloc[1], 3),
    "Sig": "**" if nb_res.pvalues.iloc[-1] < 0.01 else "*" if nb_res.pvalues.iloc[-1] < 0.05 else "ns"
})

coef_df = pd.DataFrame(coef_rows)
print(coef_df.to_string(index=False))
coef_df.to_csv(os.path.join(TAB_DIR, "table3_nb_regression_corrected.csv"), index=False)

# Model fit summary
fit_summary = {
    "poisson_dispersion": round(disp, 3),
    "poisson_rmse": round(rmse_poi, 3),
    "poisson_mae": round(mae_poi, 3),
    "nb_alpha": round(float(nb_res.params.iloc[-1]), 4),
    "nb_alpha_p": round(float(nb_res.pvalues.iloc[-1]), 4),
    "nb_rmse": round(rmse_nb, 3),
    "nb_mae": round(mae_nb, 3),
    "nb_pseudo_r2": round(nb_res.prsquared, 4),
    "n_train": len(y_train),
    "n_test": len(y_test),
}
print(f"\nModel fit summary: {json.dumps(fit_summary, indent=2)}")
with open(os.path.join(TAB_DIR, "model_fit_summary.json"), "w") as f:
    json.dump(fit_summary, f, indent=2)

# ─────────────────────────────────────────────────────────────
# 8. Forecasting (2027–2029 from 2026–2028 features)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FORECASTING (Predicted 2027–2029)")
print("="*70)

forecast_years = [2026, 2027, 2028]

# Baseline: median NIH inputs from most recent 3 years of actual data
recent_cutoff = panel_ml["year"].max() - 2
recent_window = panel_ml[panel_ml["year"] >= recent_cutoff]
median_counts  = recent_window.groupby("polymer")["nih_count"].median().to_dict()
median_funding = recent_window.groupby("polymer")["nih_funding"].median().to_dict()

print(f"Baseline window: {int(recent_cutoff)}–{int(panel_ml['year'].max())} (median NIH values)")
print(f"  Median counts: {median_counts}")

forecast_rows = []
for poly in sorted(panel_ml["polymer"].unique()):
    for yr in forecast_years:
        forecast_rows.append({
            "polymer": poly,
            "year": yr,
            "nih_count": float(median_counts.get(poly, 0)),
            "nih_funding": float(median_funding.get(poly, 0)),
            "log_funding": np.log1p(float(median_funding.get(poly, 0))),
            "year_trend": yr - panel_ml["year"].min(),
        })

forecast_df = pd.DataFrame(forecast_rows)

# Design matrix (match training columns)
model_cols = [c for c in best_model.model.exog_names if c != "const"]
X_forecast = pd.get_dummies(
    forecast_df[["log_funding", "nih_count", "year_trend", "polymer"]],
    columns=["polymer"], drop_first=True
).astype(float).reindex(columns=model_cols, fill_value=0)

X_forecast_c = sm.add_constant(X_forecast, has_constant="add")

forecast_df["predicted_patents"] = best_model.predict(X_forecast_c)
forecast_df["predicted_patents"] = forecast_df["predicted_patents"].clip(lower=0).round().astype(int)
forecast_df["predicted_year"] = forecast_df["year"] + 1

pivot = forecast_df.pivot(index="polymer", columns="predicted_year", values="predicted_patents")
print(f"\nForecasted patent counts (predicted year):")
print(pivot.to_string())

forecast_df.to_csv(os.path.join(TAB_DIR, "forecast_patents_2027_2029_corrected.csv"), index=False)
pivot.to_csv(os.path.join(TAB_DIR, "forecast_pivot_corrected.csv"))

# ─────────────────────────────────────────────────────────────
# 9. Summary of all manuscript numbers (CORRECTED)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("CORRECTED NUMBERS FOR MANUSCRIPT")
print("="*70)

print(f"""
ABSTRACT / RESULTS:
  NIH projects:        {total_nih}
  Total NIH funding:   ${total_funding_m:.1f}M
  Patent applications: {total_patents} (by publication year)

TABLE 1:
  (See table1_polymer_overview.csv)

¶63 — Overall Trends:
  Hydrogel projects: {int(tbl1[tbl1['Polymer']=='hydrogel']['NIH Projects (n)'].values[0])}
  PEG projects:      {int(tbl1[tbl1['Polymer']=='PEG']['NIH Projects (n)'].values[0])}
  Hydrogel patents:  {int(tbl1[tbl1['Polymer']=='hydrogel']['Patents (n)'].values[0])}
  PEG patents:       {int(tbl1[tbl1['Polymer']=='PEG']['Patents (n)'].values[0])}
  PLA patents/proj:  {tbl1[tbl1['Polymer']=='PLA']['Patents/Project'].values[0]:.1f}
  PLA patents/$M:    {tbl1[tbl1['Polymer']=='PLA']['Patents/$M'].values[0]:.1f}
  Chitosan pat/proj: {tbl1[tbl1['Polymer']=='chitosan']['Patents/Project'].values[0]:.1f}
  Chitosan pat/$M:   {tbl1[tbl1['Polymer']=='chitosan']['Patents/$M'].values[0]:.1f}
  PCL pat/proj:      {tbl1[tbl1['Polymer']=='PCL']['Patents/Project'].values[0]:.1f}
  PLGA pat/proj:     {tbl1[tbl1['Polymer']=='PLGA']['Patents/Project'].values[0]:.1f}
  Chi-sq (6 poly):   χ²={chi2_6:.2f}, df={dof_6}, p<0.001

¶67 — Temporal Trends:
  Patent peak:  {pat_peak_yr} (n={int(pat_peak_val)})
  NIH peak:     {proj_peak_yr} (n={int(proj_peak_val)})
  Same-year r (funding-patents):  Pearson r={fund_corr[0]:.2f}, p={fund_corr[1]:.4f}
  Same-year ρ (funding-patents):  Spearman ρ={fund_corr[2]:.2f}, p={fund_corr[3]:.4f}
  Same-year r (projects-patents): Pearson r={proj_corr[0]:.2f}, p={proj_corr[1]:.4f}

¶68 — Lag Analysis (CORRECTED):
  Peak Spearman ρ (Funding): {fund_spear_peak_r:.3f}, p={fund_spear_peak_p:.4f} at lag={fund_spear_peak_lag}
    Direction: {"Patents lead NIH" if fund_spear_peak_lag < 0 else "NIH leads patents" if fund_spear_peak_lag > 0 else "Same year"}
  Peak Pearson r (Funding): {fund_pear_peak_r:.3f}, p={fund_pear_peak_p:.4f} at lag={fund_pear_peak_lag}

¶77–78 — Regression:
  Poisson dispersion: {disp:.1f}
  Poisson RMSE: {rmse_poi:.1f}
  Poisson MAE:  {mae_poi:.1f}
  NB α:         {float(nb_res.params.iloc[-1]):.3f}, p={float(nb_res.pvalues.iloc[-1]):.3f}
  NB RMSE:      {rmse_nb:.1f}
  NB MAE:       {mae_nb:.1f}
  NB Pseudo R²: {nb_res.prsquared:.2f}
  (See table3_nb_regression_corrected.csv for all coefficients)
""")

print("✅ All corrected outputs saved to outputs/tables/")
