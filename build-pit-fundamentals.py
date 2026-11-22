"""Attach a point-in-time availability date to each fundamental observation.

THE PROBLEM. Fundamentals carry a fiscal quarter-end date, not a publication date (the
API exposes no publication field -- verified: the only date-like column is reportType).
Joining fundamentals to prices on the quarter-end date would let the model trade on
figures that were not yet public, inflating backtest performance with information that
did not exist at the decision point. This is look-ahead bias and it is the single
failure mode most likely to invalidate the study.

THE ASSUMPTION. Nepali BFIs begin publishing roughly 15 days after quarter end and are
generally complete by 45 days. NRB requires quarterly disclosure within 35 days; the
observed practice runs slightly longer. A fundamental is therefore treated as usable
only from:

    availableDate = quarterEndDate + REPORTING_LAG_DAYS

WHY THE CONSERVATIVE END. The lag must hold for EVERY company at EVERY rebalancing
date, not for the typical one. Assuming 30 days when a company files on day 42 leaks
future information, and the leak is not random: late filers skew toward distressed
institutions, which are exactly the observations a ranking model keys on. Being early
costs some signal freshness; being late manufactures alpha. 45 days is therefore the
default, with 15 and 30 retained as sensitivity cases spanning the real publication
window rather than arbitrary offsets.

Q4 CAVEAT. The fourth quarter carries annual audited results, which in practice take
longer than interim quarters. Q4_EXTRA_DAYS makes that explicit rather than silently
applying the interim lag to a filing that cannot meet it.

Outputs:
    data/bfi_fundamentals_pit.csv   one row per company-quarter-item with availableDate
    data/pit_lag_sensitivity.csv    availability calendar under each lag scenario
"""

import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# Nepali FY starts mid-July; approximate AD date of each fiscal quarter end.
# Shared with build-shares.py -- keep in sync.
QUARTER_END = {'Q1': (10, 17), 'Q2': (1, 14), 'Q3': (4, 13), 'Q4': (7, 15)}

# Baseline assumption; see module docstring. The environment variable overrides it for
# the reporting-lag robustness runs (Q4 keeps its extra days on top).
REPORTING_LAG_DAYS = int(os.environ.get('REPORTING_LAG_DAYS', 45))

# Annual audited results take longer than interim statements.
Q4_EXTRA_DAYS = 45

# Spans the observed publication window (15 days earliest, 45 days typical completion).
SENSITIVITY_LAGS = [15, 30, 45]


def quarter_end_date(fy_start, quarter_no):
    month, day = QUARTER_END[quarter_no]
    year = fy_start if quarter_no == 'Q1' else fy_start + 1
    return pd.Timestamp(year=year, month=month, day=day)


def add_availability(df, lag_days=REPORTING_LAG_DAYS, q4_extra=Q4_EXTRA_DAYS):
    df = df.copy()
    df['quarterEndDate'] = [quarter_end_date(fy, q)
                            for fy, q in zip(df.fyStart, df.quarterNo)]
    extra = df.quarterNo.eq('Q4').map({True: q4_extra, False: 0})
    df['lagDays'] = lag_days + extra
    df['availableDate'] = df.quarterEndDate + pd.to_timedelta(df.lagDays, unit='D')
    return df


def build():
    f = pd.read_csv(os.path.join(DATA_DIR, 'bfi_fin_reports_2015_2025.csv'))
    return add_availability(f)


def sensitivity(df):
    """Availability calendar under each candidate lag, for the robustness check."""
    keys = df[['sector', 'stockSymbol', 'fiscalYearAD', 'fyStart', 'quarterNo']]
    keys = keys.drop_duplicates().reset_index(drop=True)
    keys = add_availability(keys)

    out = keys[['sector', 'stockSymbol', 'fiscalYearAD', 'fyStart', 'quarterNo',
                'quarterEndDate']].copy()
    for lag in SENSITIVITY_LAGS:
        k = add_availability(keys, lag_days=lag)
        out[f'available_lag{lag}'] = k.availableDate
    return out


def as_of_join(prices, fundamentals, value_col='valueAbs'):
    """Join fundamentals onto a price panel using ONLY data public at each date.

    This is the function feature construction must go through. Joining on
    quarterEndDate instead would reintroduce the look-ahead bias this module exists to
    prevent, so the merge is keyed on availableDate and nothing else.

    A backward merge_asof takes, for each (symbol, date), the most recent filing whose
    availableDate has already passed -- never a future one. No tolerance is set: an old
    filing is still the best information genuinely available, and capping it would
    silently drop early-window rows rather than represent what a trader knew.
    """
    p = prices[['symbol', 'date']].drop_duplicates().sort_values('date')
    f = fundamentals.sort_values('availableDate')

    merged = pd.merge_asof(
        p, f,
        left_on='date', right_on='availableDate',
        left_by='symbol', right_by='stockSymbol',
        direction='backward',
    )
    # how stale the figure is at the point of use -- a feature in its own right,
    # and the diagnostic for spotting symbols that stopped filing
    merged['dataAgeDays'] = (merged.date - merged.availableDate).dt.days
    return merged


def main():
    df = build()
    cols = ['sector', 'stockSymbol', 'fiscalYearAD', 'fiscalYearBS', 'fyStart',
            'quarterNo', 'quarterEndDate', 'lagDays', 'availableDate',
            'reportType', 'itemName', 'itemValue', 'precision', 'valueAbs']
    df[cols].to_csv(os.path.join(DATA_DIR, 'bfi_fundamentals_pit.csv'), index=False)

    sens = sensitivity(df)
    sens.to_csv(os.path.join(DATA_DIR, 'pit_lag_sensitivity.csv'), index=False)

    print(f'rows={len(df)} company-quarters={len(sens)} '
          f'symbols={df.stockSymbol.nunique()}')
    print(f'baseline lag={REPORTING_LAG_DAYS}d (+{Q4_EXTRA_DAYS}d for Q4)')
    print('\navailability lag by quarter (days after quarter end):')
    print(df.groupby('quarterNo').lagDays.first().to_string())
    print('\nquarter-end -> available, one fiscal year:')
    sample = sens[sens.fyStart == 2019].drop_duplicates('quarterNo')
    print(sample[['quarterNo', 'quarterEndDate'] +
                 [f'available_lag{l}' for l in SENSITIVITY_LAGS]].to_string(index=False))


if __name__ == '__main__':
    main()
