# Platform Maturity and the NIH–Patent Nexus: Evidence from Polymer Drug-Delivery Research (2015–2025)

## Overview

This repository contains the data processing scripts, analysis code, and output tables for a study examining how NIH-funded polymer drug-delivery research aligns with downstream patent publications. The study integrates NIH RePORTER records (2015–2025) with patent data from The Lens to construct a polymer–year panel covering six major drug-delivery polymers.

## Data Sources

- **NIH RePORTER** (https://reporter.nih.gov/) — Fiscal years 2015–2025
- **The Lens** (https://www.lens.org/) — Patent publications 2015–2025

## Polymers Studied

Six polymer classes were analyzed: hydrogel, polyethylene glycol (PEG), chitosan, polylactic acid (PLA), poly(lactic-co-glycolic acid) (PLGA), and polycaprolactone (PCL). Polymeric micelles are included in descriptive statistics (Table 1) but excluded from time-series, regression, and forecast analyses due to zero NIH projects meeting the inclusion criteria.

## Repository Contents

| File | Description |
|------|-------------|
| `corrected_analysis.py` | Main analysis script (Python). Produces all tables, statistics, and forecasts reported in the manuscript. |
| `all_patents_cleaned.csv` | Cleaned patent records from The Lens (2015–2025), indexed by publication year. |
| `nih_polymer_drug_delivery_cleaned.csv` | Cleaned NIH RePORTER records (2015–2025) with polymer classification. |
| `corrected_analysis_tables.xlsx` | Formatted Excel workbook containing all output tables (Table 1, Table 3, lag correlations, forecasts). |

## Running the Analysis

```bash
pip install pandas numpy scipy statsmodels scikit-learn
python corrected_analysis.py
```

The script reads the two CSV files and produces all numerical results, summary tables, and forecast outputs reported in the manuscript.

## Key Results Summary

- **627** NIH-funded projects totaling **$246.6 million** corresponded to **6,684** patent publications
- Patent distribution differed significantly across polymers (χ² = 82.94, df = 5, p < 0.001)
- Polymer identity was the dominant predictor of next-year patent publications in Negative Binomial regression
- Strongest temporal alignment between NIH funding and patent publications occurred at lag = −1 (Spearman ρ = 0.794, p = 0.006)
- Patent data indexed by **publication year** (date patent was made publicly available by the patent office)

## Notes

- Patent publication year reflects the date on which a patent application was made publicly available, which typically follows the original filing date by approximately 18 months.
- All analyses use publication year as the temporal variable for patent records.
- Polymeric micelles (0 NIH projects, 42 patent publications) are reported in Table 1 but excluded from all polymer-specific analyses.
