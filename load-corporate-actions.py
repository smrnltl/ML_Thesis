"""Load corporate actions (bonus, cash dividend, rights) and M&A history from the
vendor spreadsheets into tidy CSVs.

Three things here are not obvious and are the reason this file exists:

1. RIGHTS RATIOS ARE CORRUPTED IN THE SOURCE WORKBOOK. Excel coerced the text
   ratios ("1:2", "1:1.25", "100:50") into time/duration values. The original text
   is unrecoverable from the displayed value alone, but the cell's number_format
   records which coercion was applied, so the pair is recoverable exactly:

       h:mm        time(1, 2)             -> 1:2
       mmss.0      time(0, 1, 1, 250000)  -> 1:1.25      (minutes:seconds.frac)
       [h]:mm:ss   timedelta(4d, 17400s)  -> 100:50      (overflowed past 24h)

   All 166 rows carry one of these three formats, so every row decodes unambiguously.
   This is why the sheet is read with openpyxl rather than pandas: pandas discards
   number_format and the three cases become indistinguishable.

2. RIGHTS LAND IN THE DECLARED YEAR; ABOUT HALF ARE COMPLETED. Reconciled against
   filing-derived share counts, rights agree far better at lag 0 (36.4% within 5%,
   42.9% with take-up >= 90%) than at lag 1 (5.6%, 12.2%). Of 92 rights events with
   share data, 49 (53%) show take-up >= 80% at lag 0 or 1; the rest were announced
   and not completed in the window. KRBL (1:2.25) and GRDBL (1:4), FY2017/18, both
   show full take-up in the declared year.

   (An earlier version tested rights at lag 1 only, found 3.3% agreement, and
   wrongly concluded rights were mostly unexecuted announcements.)

   Whether a given rights issue happened is settled by its price adjustment in
   build-adjusted-prices.py, not by this table alone.

3. BONUS AND CASH LAND ONE FISCAL YEAR AFTER DECLARATION. Declared for FY X, approved
   at the AGM and issued during FY X+1. Exact agreement for bonus is 60.9% at lag 1
   versus 15.8% at lag 0 and 14.6% at lag 2. `fyStartEffective` carries the applied
   lag for bonus/cash, `rightsFyStartEffective` for rights.

Outputs:
    data/corporate_actions_bonus.csv   bonus + cash dividend per company-FY
    data/corporate_actions_rights.csv  decoded rights ANNOUNCEMENTS per company-FY
    data/corporate_actions_merged.csv  one row per company-FY, all actions
    data/merger_acquisition.csv        tidy long-form M&A: one row per constituent
    data/ca_reconciliation.csv         per company-year check vs filing share counts
"""

import datetime
import os

import openpyxl
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

PAR_VALUE = 100.0

# Bonus/cash declared for FY X are issued during FY X+1; rights land in FY X itself
# (see module docstring).
ISSUE_LAG_YEARS = 1
RIGHTS_LAG_YEARS = 0


def decode_ratio(value, number_format):
    """Recover the (a, b) of a rights ratio "a:b" from Excel's time coercion.

    The number_format is what disambiguates; see module docstring case 1.
    """
    if isinstance(value, datetime.timedelta):
        # [h]:mm:ss -- hours overflowed past 24 and became days
        total = value.total_seconds()
        return total // 3600, (total % 3600) // 60
    if isinstance(value, datetime.time):
        if number_format == 'mmss.0':
            # minutes:seconds.fraction -- the fractional part is a real part of the ratio
            return float(value.minute), value.second + value.microsecond / 1e6
        # h:mm
        return float(value.hour), float(value.minute)
    raise ValueError(f'unrecognised rights ratio cell: {value!r} ({number_format})')


def load_rights():
    wb = openpyxl.load_workbook(os.path.join(DATA_DIR, 'bonus.xlsx'))
    # the sheet was renamed when book-closure dates were added
    ws = wb['right'] if 'right' in wb.sheetnames else wb['Sheet2']
    rows = []
    for row in ws.iter_rows(min_row=2, values_only=False):
        symbol, ratio_cell, fy_ad, fy_bs = row[0], row[1], row[2], row[3]
        if symbol.value is None:
            continue
        a, b = decode_ratio(ratio_cell.value, ratio_cell.number_format)
        rows.append({
            'symbol': symbol.value,
            'rightsFor': a,          # held
            'rightsGet': b,          # entitled
            'rightsRatio': b / a,    # new shares per existing share, IF fully taken up
            'ratioText': f'{a:g}:{b:g}',
            'fiscalYearAD': fy_ad.value,
            'fiscalYearBS': fy_bs.value,
            'bookClosureDate': row[4].value if len(row) > 4 else None,
        })
    df = pd.DataFrame(rows)
    df['bookClosureDate'] = pd.to_datetime(df.bookClosureDate)
    df['fyStartDeclared'] = df.fiscalYearAD.str[:4].astype(int)
    df['fyStartEffective'] = df.fyStartDeclared + RIGHTS_LAG_YEARS
    return df.sort_values(['symbol', 'fyStartDeclared']).reset_index(drop=True)


def load_bonus():
    df = pd.read_excel(os.path.join(DATA_DIR, 'bonus.xlsx'), sheet_name='bonus_cash')
    df = df.rename(columns={'Symbol': 'symbol', 'Bonus': 'bonusPct', 'Cash': 'cashPct',
                            'FiscalYearAD': 'fiscalYearAD', 'FiscalYearBS': 'fiscalYearBS',
                            'BookClosureDateAD': 'bookClosureDate'})
    if 'bookClosureDate' not in df:
        df['bookClosureDate'] = pd.NaT
    df['bookClosureDate'] = pd.to_datetime(df.bookClosureDate)
    # bonus % is of par value, so a 20% bonus multiplies the share count by 1.20
    df['bonusMult'] = 1 + df.bonusPct / 100
    # cash dividend per share in NPR (declared as a % of the NPR 100 par value)
    df['cashPerShare'] = df.cashPct / 100 * PAR_VALUE
    df['fyStartDeclared'] = df.fiscalYearAD.str[:4].astype(int)
    df['fyStartEffective'] = df.fyStartDeclared + ISSUE_LAG_YEARS
    return df.sort_values(['symbol', 'fyStartDeclared']).reset_index(drop=True)


def merge_actions(bonus, rights):
    b = bonus.groupby(['symbol', 'fiscalYearAD', 'fiscalYearBS', 'fyStartDeclared',
                       'fyStartEffective'], as_index=False).agg(
        bonusPct=('bonusPct', 'sum'), cashPct=('cashPct', 'sum'),
        cashPerShare=('cashPerShare', 'sum'),
        bonusBookClosure=('bookClosureDate', 'first'))
    r = rights.groupby(['symbol', 'fiscalYearAD', 'fyStartDeclared', 'fyStartEffective'],
                       as_index=False).agg(
        rightsRatio=('rightsRatio', 'sum'), ratioText=('ratioText', ' + '.join),
        nRights=('rightsRatio', 'size'),
        rightsBookClosure=('bookClosureDate', 'first'))
    r = r.rename(columns={'fyStartEffective': 'rightsFyStartEffective'})

    # one row per company and DECLARED year; bonus and rights take effect in
    # different years, so each keeps its own effective-year column
    m = b.merge(r, on=['symbol', 'fiscalYearAD', 'fyStartDeclared'], how='outer')
    m['fyStartEffective'] = m.fyStartDeclared + ISSUE_LAG_YEARS
    m['rightsFyStartEffective'] = m.fyStartDeclared + RIGHTS_LAG_YEARS
    for col, fill in [('bonusPct', 0.0), ('cashPct', 0.0), ('cashPerShare', 0.0),
                      ('rightsRatio', 0.0), ('nRights', 0)]:
        m[col] = m[col].fillna(fill)

    m['bonusMult'] = 1 + m.bonusPct / 100
    m['hasBonus'] = m.bonusPct > 0
    m['hasCash'] = m.cashPct > 0
    # declared; whether completed is settled by the price adjustment (docstring case 2)
    m['rightsAnnounced'] = m.rightsRatio > 0
    return m.sort_values(['symbol', 'fyStartDeclared']).reset_index(drop=True)


def resolve_survivors(long):
    """Correct deals whose recorded survivor is not the ticker the merged company
    trades under.

    The table records NMFBS + SMFDB -> "NLBS" (2023-07-15), but NLBS never appears in
    the price feed or the filings, while NMFBS keeps trading and filing as the merged
    company to this day (confirmed). Rule: if the recorded survivor never trades and
    exactly one constituent keeps trading after the deal, that constituent is the
    survivor. Needs the price panel; skipped if it is not there yet.
    """
    path = os.path.join(DATA_DIR, 'bfi_prices_full_2010_2025.csv')
    if not os.path.exists(path):
        long['recordedSurvivor'] = long.survivingSymbol
        return long
    last = pd.read_csv(path, usecols=['symbol', 'date'],
                       parse_dates=['date']).groupby('symbol').date.max()

    long['recordedSurvivor'] = long.survivingSymbol
    done = long[long.completed]
    for (surv, d), g in done.groupby(['survivingSymbol', 'jointOperationDate']):
        if surv in last.index:
            continue
        keeps = [s for s in g.constituentSymbol
                 if s in last.index and last[s] > d + pd.Timedelta(days=60)]
        if len(keeps) == 1:
            rows = g.index
            long.loc[rows, 'survivingSymbol'] = keeps[0]
            long.loc[rows, 'isSurvivor'] = long.loc[rows, 'constituentSymbol'] == keeps[0]
    return long


def load_merger_acquisition():
    df = pd.read_excel(os.path.join(DATA_DIR, 'merger-acquisition.xlsx'))
    company_cols = ['Company1', 'Company2', 'Company3', 'Company4', 'Company5']

    long = df.melt(
        id_vars=['NewCompanySymbol', 'Action', 'AddedOn', 'JointOperationDateAd'],
        value_vars=company_cols, var_name='slot', value_name='constituentSymbol',
    ).dropna(subset=['constituentSymbol'])

    long = long.rename(columns={'NewCompanySymbol': 'survivingSymbol',
                                'Action': 'action',
                                'AddedOn': 'recordAddedOn',
                                'JointOperationDateAd': 'jointOperationDate'})
    # a constituent equal to the surviving symbol is the acquirer continuing on
    long['isSurvivor'] = long.constituentSymbol == long.survivingSymbol

    # A MISSING jointOperationDate MEANS THE DEAL NEVER COMPLETED. Like the rights
    # sheet, this table mixes announcements with completed deals. Treating every
    # non-surviving constituent as delisted wrongly kills 8 companies that are still
    # trading today -- e.g. HBL appears as absorbed into NIB (undated, abandoned),
    # yet HBL went on to acquire CBBL in 2023; ADBL and NBL appear under an undated
    # SBL acquisition that never happened. Every one of those false positives is an
    # undated row, so the date is the discriminator.
    long['completed'] = long.jointOperationDate.notna()
    long = resolve_survivors(long)
    # only a completed deal actually removes a symbol from the tradable universe
    long['delisted'] = (~long.isSurvivor) & long.completed

    long = long.drop_duplicates(
        ['survivingSymbol', 'constituentSymbol', 'action', 'jointOperationDate'])
    return long.sort_values(['jointOperationDate', 'survivingSymbol',
                             'slot']).reset_index(drop=True)


def reconcile(actions):
    """Check the action table against filing-derived share counts.

    This is the evidence behind docstring cases 2 and 3; it is written out so the
    thesis can cite the agreement rates rather than assert them.
    """
    path = os.path.join(DATA_DIR, 'bfi_shares_outstanding.csv')
    if not os.path.exists(path):
        return None

    sh = pd.read_csv(path)
    q4 = sh[sh.quarterNo == 'Q4'][['stockSymbol', 'fyStart', 'sharesOutstanding']]
    q4 = q4.sort_values(['stockSymbol', 'fyStart'])
    q4['sharesPrev'] = q4.groupby('stockSymbol').sharesOutstanding.shift()
    q4['actualMult'] = q4.sharesOutstanding / q4.sharesPrev
    q4 = q4.dropna(subset=['actualMult'])

    # bonus and rights effective in the filing year, each at its own lag
    bonus = actions.groupby(['symbol', 'fyStartEffective'], as_index=False).bonusPct.sum()
    rights = actions.groupby(['symbol', 'rightsFyStartEffective'], as_index=False).agg(
        rightsRatio=('rightsRatio', 'sum'),
        ratioText=('ratioText', lambda s: ' + '.join(s.dropna())))

    c = q4.merge(bonus, left_on=['stockSymbol', 'fyStart'],
                 right_on=['symbol', 'fyStartEffective'], how='left')
    c = c.merge(rights, left_on=['stockSymbol', 'fyStart'],
                right_on=['symbol', 'rightsFyStartEffective'], how='left')
    c = c.fillna({'bonusPct': 0.0, 'rightsRatio': 0.0})
    c = c[(c.bonusPct > 0) | (c.rightsRatio > 0)]

    c['bonusMult'] = 1 + c.bonusPct / 100
    c['expectedBonusOnly'] = c.bonusMult
    c['expectedWithRights'] = c.bonusMult * (1 + c.rightsRatio)
    c['errBonusOnly'] = c.actualMult / c.expectedBonusOnly
    c['errWithRights'] = c.actualMult / c.expectedWithRights
    # what fraction of the announced rights entitlement actually materialised
    c['impliedRightsTakeup'] = (
        (c.actualMult / c.bonusMult - 1) / c.rightsRatio.where(c.rightsRatio > 0))

    return c[['stockSymbol', 'fyStart', 'sharesPrev', 'sharesOutstanding', 'actualMult',
              'bonusPct', 'rightsRatio', 'ratioText', 'expectedBonusOnly',
              'expectedWithRights', 'errBonusOnly', 'errWithRights',
              'impliedRightsTakeup']]


def main():
    bonus = load_bonus()
    rights = load_rights()
    actions = merge_actions(bonus, rights)
    ma = load_merger_acquisition()

    bonus.to_csv(os.path.join(DATA_DIR, 'corporate_actions_bonus.csv'), index=False)
    rights.to_csv(os.path.join(DATA_DIR, 'corporate_actions_rights.csv'), index=False)
    actions.to_csv(os.path.join(DATA_DIR, 'corporate_actions_merged.csv'), index=False)
    ma.to_csv(os.path.join(DATA_DIR, 'merger_acquisition.csv'), index=False)

    print(f'bonus/cash : rows={len(bonus):5} symbols={bonus.symbol.nunique()}')
    print(f'rights     : rows={len(rights):5} symbols={rights.symbol.nunique()}')
    print(f'merged     : rows={len(actions):5} symbols={actions.symbol.nunique()}')
    print(f'M&A        : rows={len(ma):5} deals={ma.survivingSymbol.nunique()} '
          f'completed-delistings={int(ma.delisted.sum())} '
          f'undated/abandoned={int((~ma.completed).sum())}')

    rec = reconcile(actions)
    if rec is not None:
        rec.to_csv(os.path.join(DATA_DIR, 'ca_reconciliation.csv'), index=False)
        bonus_only = rec[rec.rightsRatio == 0]
        with_rights = rec[rec.rightsRatio > 0]
        exact = lambda s: s.between(0.99, 1.01).mean()
        print(f'\nreconciliation vs filing share counts '
              f'(bonus lag={ISSUE_LAG_YEARS}, rights lag={RIGHTS_LAG_YEARS}):')
        print(f'  bonus-only      n={len(bonus_only):4} '
              f'exact={exact(bonus_only.errBonusOnly):.1%}')
        print(f'  rights-involved n={len(with_rights):4} '
              f'within 5%={with_rights.errWithRights.between(0.95, 1.05).mean():.1%} '
              f'median take-up={with_rights.impliedRightsTakeup.median():.3f}')
        print(f'  rights with take-up >= 0.9: '
              f'{int((with_rights.impliedRightsTakeup >= 0.9).sum())}/{len(with_rights)}')


if __name__ == '__main__':
    main()
