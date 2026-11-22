"""Risk-free rate for Sharpe and Sortino: Nepal Treasury bills.

PRIMARY: 91-day T-bill, monthly weighted average (Nepal Rastra Bank, via IMF
International Financial Statistics, series IFS/M.NP.FITB_PA, retrieved from DBnomics).
The NRB website was unavailable, so this IMF copy is used. It is identified as NRB's
91-day weighted-average rate because it matches every NRB 91-day figure available
exactly: mid-Jul 2022 10.66, mid-Jul 2023 6.35, mid-Jun 2024 2.99, mid-Jul 2024 3.00,
mid-Apr 2025 3.06, mid-Jun 2025 2.94.

MONTH LABELS. Each IMF month is the Nepali month ENDING mid-way through it: the value
labelled July covers mid-June to mid-July (Asar). So a fiscal year (mid-July to
mid-July) is labels August..July, and a calendar day on or before the 15th belongs to
that month's label, after the 15th to the next month's.

ROBUSTNESS: 182-day T-bill, annual weighted average by fiscal year (NRB via
Investopaper, 2024-03-10), transcribed below and checked against a screenshot. It ends
at FY2022/23. Its fiscal-year averages correlate 0.973 with the 91-day series and sit
0.4-1.0 pp above it in most years, as expected for the longer tenor.

REJECTED SOURCE. An unsourced table of "annual" 91/364-day rates was checked and not
used: it mixes fiscal-year averages with mid-July year-end values (e.g. 2021/22 =
10.76 matches the mid-July 10.66, not the 6.08 average) and puts the NRB 182-day
average outside its own 91-364 range in 7 of 14 years.

GAP. The IMF series ends 2025-06 (period ending mid-June 2025). The mid-July 2025 value,
2.95, comes from NRB's annual 2024/25 macroeconomic report and is added by hand. Days
after mid-July 2025 have no rate and are left missing.

Outputs:
    data/riskfree_monthly_91d.csv   monthly 91-day rate, 2010 onward
    data/tbill_annual.csv           fiscal-year averages, 91-day and 182-day
    data/riskfree_daily.csv         daily rate per calendar day 2010-2025, both tenors
"""

import json
import os

import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

DBNOMICS_URL = 'https://api.db.nomics.world/v22/series/IMF/IFS/M.NP.FITB_PA'
RAW_PATH = os.path.join(DATA_DIR, 'raw_tbill_91d_imf.json')

# Not yet in the IMF series. Source: NRB, Current Macroeconomic and Financial
# Situation (annual data of 2024/25), Table 1.12, mid-July 2025.
MANUAL_91D = {'2025-07': 2.95}

# Weighted average 182-day T-bill rate, % per annum. Source: NRB via Investopaper.
TBILL_182D = {
    '2003/2004': 3.44, '2004/2005': 3.14, '2005/2006': 3.40, '2006/2007': 3.29,
    '2007/2008': 4.60, '2008/2009': 5.44, '2009/2010': 7.58, '2010/2011': 8.31,
    '2011/2012': 2.17, '2012/2013': 2.10, '2013/2014': 0.51, '2014/2015': 0.58,
    '2015/2016': 1.18, '2016/2017': 2.11, '2017/2018': 3.99, '2018/2019': 3.81,
    '2019/2020': 4.00, '2020/2021': 2.59, '2021/2022': 7.30, '2022/2023': 8.53,
}

# Nepali FY starts mid-July (Shrawan 1). Shared convention with build-shares.py.
FY_START_MONTH_DAY = (7, 16)

TRADING_DAYS = 252


def load_91d():
    """Fetch the IMF series, falling back to the saved copy if DBnomics is down."""
    try:
        r = requests.get(DBNOMICS_URL, params={'observations': 1}, timeout=60)
        r.raise_for_status()
        doc = r.json()['series']['docs'][0]
        raw = {'period': doc['period'], 'value': doc['value'],
               'name': doc.get('series_name')}
        with open(RAW_PATH, 'w') as f:
            json.dump(raw, f)
    except Exception:
        with open(RAW_PATH) as f:
            raw = json.load(f)

    s = pd.Series(raw['value'], index=raw['period'], dtype='float')
    for period, value in MANUAL_91D.items():
        s.loc[period] = value
    df = s.rename('rate91dPct').rename_axis('period').reset_index()
    df['period'] = pd.PeriodIndex(df.period, freq='M')
    df['source'] = ['NRB annual report' if str(p) in MANUAL_91D else 'NRB via IMF IFS'
                    for p in df.period]
    # label month M closes mid-M, so August..July is one fiscal year
    df['fyStart'] = [p.year if p.month >= 8 else p.year - 1 for p in df.period]
    return df.sort_values('period').reset_index(drop=True)


def fiscal_year_label(fy_start):
    return f'{fy_start}/{fy_start + 1}'


def annual(m91):
    a91 = m91.groupby('fyStart').agg(rate91dPct=('rate91dPct', 'mean'),
                                     months91d=('rate91dPct', 'size')).reset_index()
    a182 = pd.DataFrame({'fiscalYearAD': list(TBILL_182D),
                         'rate182dPct': list(TBILL_182D.values())})
    a182['fyStart'] = a182.fiscalYearAD.str[:4].astype(int)
    a = a91.merge(a182[['fyStart', 'rate182dPct']], on='fyStart', how='outer')
    a['fiscalYearAD'] = a.fyStart.map(fiscal_year_label)
    a = a[a.fyStart >= 2003].sort_values('fyStart').reset_index(drop=True)
    return a[['fiscalYearAD', 'fyStart', 'rate91dPct', 'months91d', 'rate182dPct']]


def to_daily(rate_annual):
    return (1 + rate_annual) ** (1 / TRADING_DAYS) - 1


def daily(m91, ann, start='2010-01-01', end='2025-12-31'):
    d = pd.DataFrame({'date': pd.date_range(start, end, freq='D')})

    # a day after the 15th belongs to the Nepali month that closes next month
    label = d.date.dt.to_period('M')
    d['period'] = label.where(d.date.dt.day <= 15, label + 1)
    d = d.merge(m91[['period', 'rate91dPct']], on='period', how='left')

    month, day = FY_START_MONTH_DAY
    ann = ann.copy()
    ann['fyStartDate'] = pd.to_datetime(ann.fyStart.astype(str) + f'-{month:02d}-{day:02d}')
    d = pd.merge_asof(d, ann[['fyStartDate', 'rate182dPct']],
                      left_on='date', right_on='fyStartDate', direction='backward')
    # from the date itself, so days past the last tabulated year are still labelled
    after_start = (d.date.dt.month > month) | ((d.date.dt.month == month) &
                                                (d.date.dt.day >= day))
    d['fiscalYearAD'] = (d.date.dt.year - (~after_start).astype(int)).map(fiscal_year_label)

    d['rf91Annual'] = d.rate91dPct / 100
    d['rf91Daily'] = to_daily(d.rf91Annual)
    d['rf182Annual'] = d.rate182dPct / 100
    d['rf182Daily'] = to_daily(d.rf182Annual)
    return d[['date', 'fiscalYearAD', 'rf91Annual', 'rf91Daily',
              'rf182Annual', 'rf182Daily']]


def main():
    m91 = load_91d()
    m91[m91.period >= pd.Period('2010-01', 'M')].assign(
        period=lambda x: x.period.astype(str)).to_csv(
        os.path.join(DATA_DIR, 'riskfree_monthly_91d.csv'), index=False)

    ann = annual(m91)
    ann.to_csv(os.path.join(DATA_DIR, 'tbill_annual.csv'), index=False)

    d = daily(m91, ann)
    d.to_csv(os.path.join(DATA_DIR, 'riskfree_daily.csv'), index=False)

    print(ann.round(2).to_string(index=False))
    gap = d[d.rf91Annual.isna()]
    print(f'\ndaily rows={len(d)}; 91-day missing={len(gap)} days'
          + (f' ({gap.date.min().date()} -> {gap.date.max().date()})' if len(gap) else ''))


if __name__ == '__main__':
    main()
