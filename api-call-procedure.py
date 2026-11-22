"""Fetch BFI quarterly financial report history (FY2015/16 - FY2024/25) from the MDP NEPSE API.

Outputs:
    data/raw_bfi.json                  raw API response per sector
    data/bfi_fin_reports_2015_2025.csv tidy long-format table (one row per line item)
"""

import json
import os

import requests
import pandas as pd

headers = {
    "AccessId": "Test",
    "ApiKey": "test_key",
    "ApiSecret": "b15bab187453449b91d77c97123348da",
    "ApiVersion": "1.0",  # required, otherwise the route 404s
}

base_url = 'https://mdpapi.smartwealthpro.com/api/financial/'

api_url = base_url + "fin-report-history"

# sector must match the API's labels exactly; timeline is case-sensitive ("Quarterly")
SECTORS = ['Commercial Banks', 'Development Banks', 'Finance', 'Microfinance']

# Nepali FY spans two AD years: 2015/2016 .. 2024/2025 == "2015 to 2025"
FY_START, FY_END = 2015, 2024

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def fetch_sector(sector):
    params = {
        'sector': sector,
        'companies': '',  # stock symbols; blank = all in sector
        'timeline': 'Quarterly',
        'fiscalYear': '',  # blank = full history, filtered client-side below
    }
    response = requests.get(api_url, headers=headers, params=params, timeout=600)
    response.raise_for_status()
    return response.json()['result']


def to_rows(raw):
    """Flatten nested reportData into long format.

    `precision` varies per record (1 or 1000), so raw itemValue is not comparable
    across rows. valueAbs restores absolute NPR.
    """
    rows = []
    for sector, records in raw.items():
        for r in records:
            fy_start = int(r['fiscalYearAD'].split('/')[0])
            if not (FY_START <= fy_start <= FY_END):
                continue
            precision = r.get('precision') or 1
            for item in r['reportData']:
                value = item.get('itemValue')
                rows.append({
                    'sector': sector,
                    'stockSymbol': r['stockSymbol'],
                    'fiscalYearAD': r['fiscalYearAD'],
                    'fiscalYearBS': r['fiscalYearBS'],
                    'fyStart': fy_start,
                    'quarterNo': r['quarterNo'],
                    'reportType': r['reportType'],
                    'itemName': item['itemName'],
                    'itemValue': value,
                    'precision': precision,
                    'valueAbs': value * precision if isinstance(value, (int, float)) else None,
                })
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    raw = {}
    for sector in SECTORS:
        raw[sector] = fetch_sector(sector)
        print(f"{sector:<20} records={len(raw[sector])}")

    with open(os.path.join(OUT_DIR, 'raw_bfi.json'), 'w') as f:
        json.dump(raw, f)

    df = pd.DataFrame(to_rows(raw))
    df.sort_values(
        ['sector', 'stockSymbol', 'fyStart', 'quarterNo', 'reportType', 'itemName'],
        inplace=True,
    )
    df.to_csv(os.path.join(OUT_DIR, 'bfi_fin_reports_2015_2025.csv'), index=False)
    print(f"\nrows={len(df)} companies={df.stockSymbol.nunique()}")


if __name__ == '__main__':
    main()
