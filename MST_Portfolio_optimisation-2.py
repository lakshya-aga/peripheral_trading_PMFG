#!/usr/bin/env python3
"""
PMFG Peripheral Portfolio – Forward Test Script
Paper: Pozzi, Di Matteo, Aste (2013)

Designed to run daily via GitHub Actions
"""

import time
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf
import networkx as nx

# ------------------
# CONFIG
# ------------------
LOOKBACK_YEARS = 2
EW_WINDOW = 125
TOP_PERC = 0.5
PORTFOLIO_SIZE = 10
OUTPUT_DIR = "outputs"

# ------------------
# UTILITIES
# ------------------
def normalize_tickers(tickers: List[str]) -> List[str]:
    return [t.replace(".", "-").strip() for t in tickers if isinstance(t, str)]

def get_sp500_universe() -> List[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500 = pd.read_html(url)[0]
    return normalize_tickers(sp500["Symbol"].tolist())

def download_prices(tickers: List[str], start, end) -> pd.DataFrame:
    prices = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=True,
    )
    return prices["Close"].dropna(axis=1)

# ------------------
# PMFG
# ------------------
def build_pmfg(corr: pd.DataFrame) -> nx.Graph:
    tickers = corr.columns.tolist()
    n = len(tickers)
    dist = np.sqrt(2 * (1 - corr.values))

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((corr.values[i, j], i, j, dist[i, j]))

    edges.sort(reverse=True, key=lambda x: x[0])
    G = nx.Graph()
    G.add_nodes_from(tickers)

    for corr_ij, i, j, d_ij in edges:
        if G.number_of_edges() >= 3 * (n - 2):
            break
        u, v = tickers[i], tickers[j]
        G.add_edge(u, v, corr=corr_ij, dist=d_ij)
        if not nx.check_planarity(G)[0]:
            G.remove_edge(u, v)

    return G

# ------------------
# CENTRALITY
# ------------------
def compute_centralities(G: nx.Graph):
    # Weighted (corr-based)
    Gw = G.copy()
    for u, v, d in Gw.edges(data=True):
        d["weight"] = d["corr"] + 1

    # Unweighted
    Gu = nx.Graph()
    Gu.add_nodes_from(G.nodes())
    Gu.add_edges_from(G.edges())

    return {
        "D_w": pd.Series(dict(Gw.degree(weight="weight"))),
        "D_u": pd.Series(dict(Gu.degree())),
        "BC_w": pd.Series(nx.betweenness_centrality(G, weight="dist")),
        "BC_u": pd.Series(nx.betweenness_centrality(Gu)),
        "E_w": pd.Series(nx.eccentricity(G, weight="dist")),
        "E_u": pd.Series(nx.eccentricity(Gu)),
        "C_w": pd.Series(nx.closeness_centrality(G, distance="dist")),
        "C_u": pd.Series(nx.closeness_centrality(Gu)),
        "EC_w": pd.Series(nx.eigenvector_centrality_numpy(Gw, weight="weight")),
        "EC_u": pd.Series(nx.eigenvector_centrality_numpy(Gu)),
    }

# ------------------
# MAIN
# ------------------
def main():
    today = datetime.now().date()
    start = today - relativedelta(years=LOOKBACK_YEARS)

    tickers = get_sp500_universe()
    prices = download_prices(tickers, start, today)
    returns = prices.pct_change().dropna()

    # 1y Sharpe proxy filter
    perf = returns.rolling(250).mean() / returns.rolling(250).std()
    latest_perf = perf.iloc[-1].dropna()
    cutoff = latest_perf.quantile(1 - TOP_PERC)
    universe = latest_perf[latest_perf >= cutoff].index.tolist()

    # EW correlations
    corr = (
        returns[universe]
        .ewm(span=EW_WINDOW)
        .corr()
        .loc[returns.index[-1]]
    )

    pmfg = build_pmfg(corr)

    # Centralities
    C = compute_centralities(pmfg)
    N = len(pmfg)

    # Rank-based aggregation (paper-faithful)
    ranks = {k: v.rank() for k, v in C.items()}

    X = (ranks["D_w"] + ranks["D_u"] + ranks["BC_w"] + ranks["BC_u"] - 4) / (4 * (N - 1))
    Y = (ranks["E_w"] + ranks["E_u"] + ranks["C_w"] + ranks["C_u"] + ranks["EC_w"] + ranks["EC_u"] - 6) / (6 * (N - 1))

    score = X + Y
    selected = score.sort_values(ascending=False).head(PORTFOLIO_SIZE)

    # OUTPUT
    output = {
        "date": str(today),
        "selected_assets": selected.index.tolist(),
        "scores": selected.to_dict(),
    }

    pd.Series(score).to_csv(f"{OUTPUT_DIR}/scores_{today}.csv")
    with open(f"{OUTPUT_DIR}/selection_{today}.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Selected assets:", selected.index.tolist())

if __name__ == "__main__":
    main()
