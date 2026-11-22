# Machine-Learning Stock Ranking on NEPSE

Code and data for the MSc thesis **A Machine Learning Approach to Stock Ranking in Nepal's Stock Market: A Backtesting Study on Banks and Financial Institutions**.

## Summary

The study checks whether a firm-level machine-learning ranking model can beat simple investment strategies for banks and financial institutions (BFIs) listed on the Nepal Stock Exchange (NEPSE).

- **Data:** ten years of quarterly filings, daily prices, corporate actions, index values and Treasury-bill rates (FY2015/16 to FY2024/25), rebuilt **point-in-time**. Prices are adjusted for bonus and rights issues, survivorship bias is measured, and a 45-day reporting lag (90 days for Q4) prevents look-ahead bias.
- **Model:** XGBoost ranks eligible stocks every month on 38 fundamental and market features. It is refitted every fiscal year (walk-forward) and tuned with Optuna inside each training window.
- **Evaluation:** a long-only top-20% portfolio is backtested over FY2018/19 to FY2024/25, net of 0.40% one-way costs. It is compared with NEPSE buy-and-hold, a price-to-book (P/B) value screen, 6-month momentum and an equal-weight holding of the universe. The model is explained with SHAP.
- **Result:** the model's net Sharpe ratio was **0.245**, against **0.725** for the P/B screen, **0.636** for momentum and **0.724** for equal weight. It was beaten in every robustness variant.

## Folder layout

```
project/
├── README.md
├── *.py              Data pipeline, model, backtest, figures (below)
├── data/             Raw pulls, intermediate CSVs, predictions, results
│   └── models/       Saved XGBoost boosters (one per fiscal year and seed)
└── figures/          Diagnostic plots from backtest.py and explain.py
    └── thesis/       Numbered figures used in the report
```

## Requirements

Python 3.12 (3.10 or later should work).

```bash
pip install pandas numpy scipy scikit-learn xgboost optuna shap matplotlib openpyxl requests
```

## Pipeline

Run the scripts in this order. Each one reads from `data/` and writes to it.

| Step | Script | Produces |
|---|---|---|
| 1 | `api-call-procedure.py` | Quarterly filings → `bfi_fin_reports_2015_2025.csv` |
| 2 | `fetch-prices-full.py` | Daily prices from 2010, including absorbed symbols → `raw_prices_full.json`, `bfi_prices_full_2010_2025.csv` |
| 3 | `fetch-index.py` | Index history from the API (starts 2016-09-25) → `nepse_index_history.csv` |
| 4 | `build-index.py` | Joins API index data to `index_old_data.xlsx` → `index_daily_2011_2025.csv` |
| 5 | `build-riskfree.py` | 91-day T-bill rate (NRB via IMF/DBnomics) → `riskfree_daily.csv`, `riskfree_monthly_91d.csv` |
| 6 | `load-corporate-actions.py` | Bonus, rights, dividends and mergers from `bonus.xlsx` and `merger-acquisition.xlsx` → `corporate_actions_*.csv`, `merger_acquisition.csv` |
| 7 | `build-adjusted-prices.py` | Ex-dates and back-adjusted total-return prices → `ca_exdates.csv`, `bfi_prices_adjusted_2010_2025.csv` |
| 8 | `build-shares.py` | Shares outstanding and market cap → `bfi_shares_outstanding.csv`, `bfi_shares_marketcap.csv` |
| 9 | `build-pit-fundamentals.py` | Reporting-lag availability dates → `bfi_fundamentals_pit.csv` |
| 10 | `build-universe.py` | Point-in-time investable universe and survivorship bound → `universe_monthly.csv`, `survivorship_bound.csv` |
| 11 | `build-features.py` | Feature matrix and forward targets → `features_monthly.csv` |
| 12 | `build-model.py` | Walk-forward training and out-of-sample predictions → `model_predictions.csv`, `models/` |
| 13 | `backtest.py` | Portfolios, metrics, Sharpe tests and cost grid → `backtest_*.csv`, `figures/` |
| 14 | `explain.py` | SHAP values, importance and stability → `shap_*.csv`, `figures/shap_*` |
| 15 | `make-figures.py` | Report figures → `figures/thesis/` |

`fetch-prices.py` is the earlier 2015–2025 price pull for surviving symbols only. `fetch-prices-full.py` replaces it; it is kept because it holds the only copy of the unadjusted price capture.

Steps 1–3 call the data vendor's API. That endpoint rate-limits hard and reports the limit as `"Unauthorized"` with HTTP 200. If you see that message, wait a few minutes and try again. Every later step works offline from the files in `data/`.

### Robustness variants

Every main-result CSV in `data/` has a suffixed twin for each robustness variant, e.g. `_lag15`, `_top10`, `_sectorneutral`.

```bash
# Reporting lag of 15 or 30 days (rebuild universe and features, then retrain)
REPORTING_LAG_DAYS=15 PIPELINE_TAG=_lag15 python build-universe.py
REPORTING_LAG_DAYS=15 PIPELINE_TAG=_lag15 python build-features.py
python build-model.py --features features_monthly_lag15.csv --tag _lag15
python backtest.py --predictions model_predictions_lag15.csv --tag _lag15

# Backtest-only variants
python backtest.py --top 0.10 --tag _top10
python backtest.py --sector-neutral --tag _sectorneutral
python backtest.py --hold 3 --tag _quarterly
python backtest.py --rf 182 --tag _rf182
python backtest.py --exclude-undated window --tag _undated_window

# Model variants
python build-model.py --window 36 --tag _roll36            # rolling 36-month window instead of expanding
python build-model.py --target fwdExcessRet --tag _excess  # raw excess return instead of rank
# then: python backtest.py --predictions model_predictions_<tag>.csv --tag <tag>
```

## Key design choices

- **No look-ahead.** A filing can be used only from quarter end + 45 days (Q4 + 90 days). The only forward-looking columns are the `fwd*` targets.
- **Survivorship bias.** The vendor holds no prices for 167 absorbed institutions, so the bias could not be removed. It is measured as an upper bound instead (`survivorship_bound.csv`).
- **Corporate actions.** NEPSE prices are unadjusted, so the vendor's adjusted series was rebuilt from 571 bonus and 81 rights issues. Each ex-date was checked against the exchange's own price-change field.
- **Target.** The model predicts each stock's cross-sectional percentile rank of next-month total return, not the raw return.
- **Costs.** The base cost is 0.40% one-way (broker commission, SEBON fee, depository charge), with sensitivity from 0% to 1%.

Each script's docstring explains its own decisions in detail.

## Notes

- The data is used for academic research only. Prices, filings and corporate actions belong to their providers (NEPSE, the data vendor, Nepal Rastra Bank and the IMF) and must not be redistributed.
- Nothing in this repository is investment advice. The model studied here underperformed simple benchmarks.
