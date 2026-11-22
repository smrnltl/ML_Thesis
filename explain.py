"""SHAP explanations of the walk-forward model's out-of-sample predictions.

For each fiscal-year refit, TreeSHAP values are computed for that year's test rows with
each seed's booster and averaged over seeds. SHAP is additive, so the average equals the
SHAP value of the seed-averaged prediction the backtest used.

Reported:
  importance   mean |SHAP| per feature, overall and per fiscal year
  direction    Spearman correlation between a feature's rank and its SHAP value:
               +1 = higher rank pushes the predicted return rank up
  stability    Spearman correlation of the importance ordering between consecutive
               refits, and the share of years in which a feature's direction agrees
               with its overall direction (an explanation that changes every year is
               itself information for the investor; see H2)

Outputs:
    data/shap_values_<model>.csv        date, symbol, fyTest, prediction, SHAP per feature
    data/shap_importance_<model>.csv    feature x (overall, each FY, direction, signAgree)
    data/shap_stability_<model>.csv     consecutive-refit rank correlation of importance
    figures/shap_bar_<model>.png, figures/shap_beeswarm_<model>.png,
    figures/shap_fy_heatmap_<model>.png
"""

import argparse
import importlib.util
import os

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data')
FIG_DIR = os.path.join(BASE, 'figures')

_spec = importlib.util.spec_from_file_location('bm', os.path.join(BASE, 'build-model.py'))
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)
FEATURES, SEEDS = bm.FEATURES, bm.SEEDS


def label(c):
    return c.replace('_rank', '').replace('sec_', 'sector: ')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='xgb')
    args = ap.parse_args()
    name = args.model

    f = bm.load()
    p = pd.read_csv(os.path.join(DATA_DIR, 'model_predictions.csv'), parse_dates=['date'])
    rows = []
    for fy, keys in p.groupby('fyTest')[['date', 'symbol']]:
        test = keys.merge(f, on=['date', 'symbol'], how='left')
        X = test[FEATURES]
        sv = np.zeros(X.shape)
        for seed in SEEDS:
            booster = xgb.Booster()
            booster.load_model(os.path.join(bm.MODEL_DIR, f'{name}_{fy}_{seed}.json'))
            sv += shap.TreeExplainer(booster).shap_values(X) / len(SEEDS)
        out = pd.DataFrame(sv, columns=FEATURES)
        out.insert(0, 'date', test.date.values)
        out.insert(1, 'symbol', test.symbol.values)
        out.insert(2, 'fyTest', fy)
        rows.append((out, X.reset_index(drop=True)))
        print(f'FY{fy}: {len(out)} rows', flush=True)

    S = pd.concat([r[0] for r in rows], ignore_index=True)
    X = pd.concat([r[1] for r in rows], ignore_index=True)
    S = S.merge(p[['date', 'symbol', name]], on=['date', 'symbol'], how='left')
    S.to_csv(os.path.join(DATA_DIR, f'shap_values_{name}.csv'), index=False)

    absS = S[FEATURES].abs()
    imp = pd.DataFrame({'overall': absS.mean()})
    by_fy = absS.groupby(S.fyTest).mean().T
    imp = imp.join(by_fy)
    imp['share'] = imp.overall / imp.overall.sum()

    def direction(mask):
        d = {}
        for c in FEATURES:
            ok = mask & X[c].notna()
            d[c] = spearmanr(X.loc[ok, c], S.loc[ok, c]).statistic if ok.sum() > 20 \
                and X.loc[ok, c].nunique() > 1 else np.nan
        return pd.Series(d)

    allrows = pd.Series(True, index=S.index)
    imp['direction'] = direction(allrows)
    fy_dir = pd.DataFrame({fy: direction(S.fyTest == fy) for fy in S.fyTest.unique()})
    imp['signAgree'] = (np.sign(fy_dir).eq(np.sign(imp.direction), axis=0)).mean(axis=1)
    imp = imp.sort_values('overall', ascending=False)
    imp.to_csv(os.path.join(DATA_DIR, f'shap_importance_{name}.csv'))

    fys = list(by_fy.columns)
    stab = pd.DataFrame([dict(fromFY=a, toFY=b,
                              rankCorr=spearmanr(by_fy[a], by_fy[b]).statistic)
                         for a, b in zip(fys[:-1], fys[1:])])
    stab.to_csv(os.path.join(DATA_DIR, f'shap_stability_{name}.csv'), index=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    top = imp.head(15)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh([label(c) for c in top.index[::-1]], top.overall[::-1])
    ax.set_xlabel('Mean |SHAP| (effect on predicted return rank)')
    ax.set_title(f'Top 15 features, {name}, out-of-sample')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'shap_bar_{name}.png'), dpi=150)
    plt.close(fig)

    shap.summary_plot(S[FEATURES].values, X.values, feature_names=[label(c) for c in FEATURES],
                      max_display=15, show=False)
    plt.gcf().tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'shap_beeswarm_{name}.png'), dpi=150)
    plt.close('all')

    hm = by_fy.loc[top.index]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(hm.values, aspect='auto', cmap='viridis')
    ax.set_yticks(range(len(hm)), [label(c) for c in hm.index])
    ax.set_xticks(range(len(fys)), fys, rotation=45)
    ax.set_title('Mean |SHAP| by refit year (top 15 features)')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'shap_fy_heatmap_{name}.png'), dpi=150)
    plt.close(fig)

    pd.set_option('display.width', 200)
    print(imp[['overall', 'share', 'direction', 'signAgree']].round(4).head(20).to_string())
    print('\nimportance-ordering stability between consecutive refits:')
    print(stab.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
