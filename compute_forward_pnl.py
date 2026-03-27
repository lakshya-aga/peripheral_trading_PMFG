#!/usr/bin/env python3
import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


def load_selections(outputs_dir: Path):
    files = sorted(glob.glob(str(outputs_dir / 'selection_*.json')))
    selections = []
    all_tickers = set()
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        signal_date = pd.Timestamp(data['date'])
        tickers = [t.replace('.', '-') for t in data['selected_assets']]
        selections.append({'signal_date': signal_date, 'tickers': tickers})
        all_tickers.update(tickers)
    return sorted(selections, key=lambda x: x['signal_date']), sorted(all_tickers)


def first_trading_day_after(index, dt):
    future_idx = index[index > dt]
    return future_idx[0] if len(future_idx) else None


def compute_forward_pnl(outputs_dir: Path, analysis_dir: Path):
    selections, all_tickers = load_selections(outputs_dir)
    if not selections:
        raise SystemExit('No selection_*.json files found')

    start = selections[0]['signal_date'] - pd.offsets.BDay(3)
    end = pd.Timestamp.today().normalize() + pd.offsets.BDay(2)
    prices = yf.download(
        all_tickers,
        start=start.strftime('%Y-%m-%d'),
        end=end.strftime('%Y-%m-%d'),
        auto_adjust=True,
        progress=False,
        group_by='column',
        threads=True,
    )['Close'].sort_index()

    trades = []
    components = []
    equity = []
    cum = 1.0

    for i, sel in enumerate(selections):
        signal_date = sel['signal_date']
        tickers = sel['tickers']
        entry_date = first_trading_day_after(prices.index, signal_date)
        if entry_date is None:
            continue

        if i + 1 < len(selections):
            next_signal = selections[i + 1]['signal_date']
            exit_date = first_trading_day_after(prices.index, next_signal)
            if exit_date is None:
                exit_date = prices.index[-1]
        else:
            exit_date = prices.index[-1]

        if exit_date <= entry_date:
            continue

        sub = prices.loc[[entry_date, exit_date], tickers]
        if isinstance(sub, pd.Series):
            sub = sub.to_frame()

        valid_cols = list(sub.columns[sub.loc[entry_date].notna() & sub.loc[exit_date].notna()])
        if not valid_cols:
            continue

        entry = sub.loc[entry_date, valid_cols]
        exitp = sub.loc[exit_date, valid_cols]
        indiv = (exitp / entry - 1.0).sort_values(ascending=False)
        port_ret = indiv.mean()
        cum *= (1 + port_ret)

        trades.append({
            'signal_date': str(signal_date.date()),
            'entry_date': str(pd.Timestamp(entry_date).date()),
            'exit_date': str(pd.Timestamp(exit_date).date()),
            'n_names': int(len(valid_cols)),
            'period_return_pct': round(port_ret * 100, 4),
            'cum_return_pct': round((cum - 1) * 100, 4),
        })
        equity.append({'date': pd.Timestamp(exit_date), 'equity': cum})

        for ticker, ret in indiv.items():
            components.append({
                'signal_date': str(signal_date.date()),
                'entry_date': str(pd.Timestamp(entry_date).date()),
                'exit_date': str(pd.Timestamp(exit_date).date()),
                'ticker': ticker,
                'return_pct': round(ret * 100, 4),
            })

    analysis_dir.mkdir(parents=True, exist_ok=True)
    trades_df = pd.DataFrame(trades)
    components_df = pd.DataFrame(components)
    equity_df = pd.DataFrame(equity)

    trades_csv = analysis_dir / 'pmfg_trade_pnl.csv'
    components_csv = analysis_dir / 'pmfg_component_breakdown.csv'
    summary_json = analysis_dir / 'pmfg_summary.json'
    img_path = analysis_dir / 'pmfg_pnl_curve.png'

    trades_df.to_csv(trades_csv, index=False)
    components_df.to_csv(components_csv, index=False)

    summary = {
        'trade_periods': int(len(trades_df)),
        'latest_mark_date': str(prices.index[-1].date()),
        'cumulative_return_pct': round((cum - 1) * 100, 4),
        'best_trade_pct': round(float(trades_df['period_return_pct'].max()), 4) if not trades_df.empty else None,
        'worst_trade_pct': round(float(trades_df['period_return_pct'].min()), 4) if not trades_df.empty else None,
        'method': 'equal-weight; enter next trading day close after signal; exit next signal next trading day close; mark final basket to latest close',
        'note': 'This matches the notebook selection logic, but the notebook itself does not define a backtest execution rule.',
    }
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity_df['date'], (equity_df['equity'] - 1) * 100, linewidth=2, color='#2563eb')
    ax.axhline(0, color='gray', linewidth=1, linestyle='--')
    ax.set_title('PMFG Forward P&L')
    ax.set_ylabel('Cumulative Return %')
    ax.set_xlabel('Exit Date')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(img_path, dpi=180)

    print(json.dumps({
        'image': str(img_path),
        'trades_csv': str(trades_csv),
        'components_csv': str(components_csv),
        'summary_json': str(summary_json),
        'summary': summary,
    }, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs-dir', default='outputs')
    parser.add_argument('--analysis-dir', default='analysis_outputs')
    args = parser.parse_args()
    compute_forward_pnl(Path(args.outputs_dir), Path(args.analysis_dir))
