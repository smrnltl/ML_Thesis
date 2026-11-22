"""Fetch NEPSE index history (sector sub-indices + the overall NEPSE benchmark).

Provides the buy-and-hold benchmark for the BFI cross-section: the Banking,
Development Bank, Finance and Microfinance sub-indices, plus Nepse itself.

Note on coverage: index history starts 2016-09-25 regardless of fromDate (hard floor
in the API), so it does NOT span the full 2015-2025 fundamentals window. Microfinance
starts later still, 2017-11-01.

Outputs:
    data/raw_index.json             raw API response (all indices)
    data/nepse_index_history.csv    tidy panel with daily returns
"""

import json
import os

import pandas as pd
import requests

headers = {
    "AccessId": "Test",
    "ApiKey": "test_key",
    "ApiSecret": "b15bab187453449b91d77c97123348da",
    "ApiVersion": "1.0",
}

api_url = 'https://mdpapi.smartwealthpro.com/api/nepse/index-history'

FROM_DATE, TO_DATE = '2015-01-01', '2025-12-31'

BFI_INDICES = ['Banking', 'Development Bank', 'Finance', 'Microfinance']

# The feed renamed some indices mid-history. These pairs are contiguous with zero
# overlapping dates, so they are one continuous series under two labels.
# 'Insurance' is NOT included: it genuinely split into Life / Non Life in July 2018.
RENAMES = {
    'NEPSE': 'Nepse',
    'Hotels': 'Hotels And Tourism',
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def fetch_all():
    """A blank indexName returns every index in one call."""
    params = {'fromDate': FROM_DATE, 'toDate': TO_DATE, 'indexName': ''}
    response = requests.get(api_url, headers=headers, params=params, timeout=900)
    response.raise_for_status()
    result = response.json().get('result')
    if not isinstance(result, list):
        raise ValueError(f"no result: {response.json().get('message')!r}")
    return result


def build_panel(raw):
    df = pd.DataFrame(raw)
    # timestamps are mixed-precision (some carry fractional seconds)
    df['date'] = pd.to_datetime(df.asOf, format='ISO8601').dt.normalize()

    df['indexName'] = df.name.replace(RENAMES)
    df = df.drop_duplicates(['indexName', 'date'], keep='first')
    df = df.sort_values(['indexName', 'date']).reset_index(drop=True)

    df['ret'] = df.groupby('indexName').close.pct_change()
    df['ohlcFlag'] = ((df.close > df.high) | (df.close < df.low) | (df.high < df.low)).astype(int)

    return df[['indexName', 'date', 'open', 'high', 'low', 'close',
               'change', 'percentChange', 'turnover', 'volume', 'ret', 'ohlcFlag']]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    raw = fetch_all()
    with open(os.path.join(DATA_DIR, 'raw_index.json'), 'w') as f:
        json.dump(raw, f)

    df = build_panel(raw)
    df.to_csv(os.path.join(DATA_DIR, 'nepse_index_history.csv'), index=False)

    print(f"rows={len(df)} indices={df.indexName.nunique()}")
    coverage = df.groupby('indexName').agg(n=('close', 'size'),
                                           start=('date', 'min'),
                                           end=('date', 'max'))
    print(coverage.to_string())


if __name__ == '__main__':
    main()
