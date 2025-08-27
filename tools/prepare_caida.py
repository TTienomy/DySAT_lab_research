#!/usr/bin/env python3
"""
Prepare CAIDA-like edge time-series into DySAT-ready inputs.

Inputs (choose one):
A) A folder with daily CSV files (filenames containing YYYYMMDD), columns: source,target[,weight]
B) A single CSV with columns: source,target,time,weight (time in YYYYMMDD / YYYY-MM-DD / timestamp)

Outputs:
- out_dir/
    graphs/
        adj_ts_000.npz, adj_ts_001.npz, ...
    features/ (optional)
        feats_ts_000.npz, feats_ts_001.npz, ...
    nodes.npy                # index -> node_id
    meta.json                # time bins, counts, etc.

Each adj_ts_* is a scipy.sparse CSR matrix saved via scipy.sparse.save_npz
Each feats_ts_* is scipy.sparse CSR (N x D). Default D=2 with [in_degree, out_degree]
"""

import os
import re
import json
import math
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp

DATE_RE = re.compile(r"(\d{8})")

def infer_time_from_filename(p):
    m = DATE_RE.findall(os.path.basename(p))
    return m[0] if m else None

def normalize_time_value(x, granularity):
    """
    Return normalized time key string by granularity: day | month | year
    Accepts int YYYYMMDD / str date / pandas Timestamp
    """
    if isinstance(x, (int, np.integer)):
        s = str(int(x))
        # if just YYYYMMDD
        if len(s) == 8:
            y, m, d = s[:4], s[4:6], s[6:8]
            ts = pd.Timestamp(f"{y}-{m}-{d}")
        else:
            # maybe epoch seconds
            ts = pd.to_datetime(x, unit='s', errors='coerce')
    else:
        # try parse
        ts = pd.to_datetime(x, errors='coerce', utc=False)
        if ts is pd.NaT:
            # try to parse YYYYMMDD string
            if isinstance(x, str) and len(x) == 8 and x.isdigit():
                y, m, d = x[:4], x[4:6], x[6:8]
                ts = pd.Timestamp(f"{y}-{m}-{d}")
            else:
                raise ValueError(f"Cannot parse time: {x}")

    if granularity == 'day':
        return ts.strftime("%Y-%m-%d")
    elif granularity == 'month':
        return ts.strftime("%Y-%m")
    elif granularity == 'year':
        return ts.strftime("%Y")
    else:
        # raw day by default
        return ts.strftime("%Y-%m-%d")

def read_folder(folder, has_header=True, weight_default=1.0):
    """
    Read multiple CSV files from folder. Expect columns at least: source,target[,weight]
    Time will be inferred from filename (YYYYMMDD).
    Returns DataFrame with columns: source,target,weight,time
    """
    folder = Path(folder)
    files = sorted([str(p) for p in folder.glob("*.csv")])
    if not files:
        raise FileNotFoundError(f"No CSV files in {folder}")

    rows = []
    for f in files:
        t = infer_time_from_filename(f)
        if not t:
            warnings.warn(f"Skip {f}: cannot infer time from filename")
            continue
        df = pd.read_csv(f, header=0 if has_header else None)
        if has_header:
            cols = [c.lower().strip() for c in df.columns]
            df.columns = cols
        else:
            df.columns = ['source', 'target', 'weight'][:df.shape[1]]

        if 'source' not in df.columns or 'target' not in df.columns:
            raise ValueError(f"{f} must have columns source,target (and optional weight).")

        if 'weight' not in df.columns:
            df['weight'] = weight_default

        df['time'] = t
        rows.append(df[['source','target','weight','time']])

    out = pd.concat(rows, ignore_index=True)
    return out

def read_single_csv(path):
    """
    Read single CSV with columns: source,target,time,weight
    time can be YYYYMMDD / YYYY-MM-DD / timestamp
    """
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    for c in ['source','target','time']:
        if c not in cols:
            raise ValueError(f"Missing column '{c}' in {path}")
    # standardize
    df = df.rename(columns={cols.get('source'):'source',
                            cols.get('target'):'target',
                            cols.get('time'):'time'})
    if 'weight' in cols:
        df = df.rename(columns={cols.get('weight'):'weight'})
    else:
        df['weight'] = 1.0
    return df[['source','target','weight','time']]

def build_index_mapping(edges_df):
    nodes = pd.Index(pd.unique(edges_df[['source','target']].values.ravel()))
    node2idx = {n:i for i,n in enumerate(nodes)}
    return nodes.to_numpy(), node2idx

def make_snapshots(edges_df, granularity='day'):
    """
    edges_df: columns source,target,weight,time (time is raw)
    returns: dict time_key -> DataFrame (source_idx, target_idx, weight)
    """
    # normalize time
    edges_df = edges_df.copy()
    edges_df['tkey'] = edges_df['time'].apply(lambda x: normalize_time_value(x, granularity))

    # map nodes to indices (global consistent index across all time)
    nodes_array, node2idx = build_index_mapping(edges_df)

    grouped = {}
    for tkey, g in edges_df.groupby('tkey'):
        s = g['source'].map(node2idx).to_numpy()
        d = g['target'].map(node2idx).to_numpy()
        w = g['weight'].astype(float).to_numpy()

        # aggregate multi-edges by (s,d)
        df = pd.DataFrame({'s': s, 'd': d, 'w': w})
        df = df.groupby(['s','d'], as_index=False)['w'].sum()
        grouped[tkey] = (df, len(nodes_array))

    # sort by time key
    keys_sorted = sorted(grouped.keys())
    return keys_sorted, grouped, nodes_array

def degrees_as_features(adj_csr):
    """
    Return an (N x 2) sparse CSR with columns [in_degree, out_degree]
    """
    out_deg = np.asarray(adj_csr.sum(axis=1)).ravel()
    in_deg  = np.asarray(adj_csr.sum(axis=0)).ravel()
    feats = np.vstack([in_deg, out_deg]).T
    # normalize (log1p) to reduce skew
    feats = np.log1p(feats)
    return sp.csr_matrix(feats)

def save_outputs(out_dir, keys_sorted, grouped, nodes_array, make_feats=True):
    out_dir = Path(out_dir)
    (out_dir / "graphs").mkdir(parents=True, exist_ok=True)
    if make_feats:
        (out_dir / "features").mkdir(parents=True, exist_ok=True)

    # Save node list
    np.save(out_dir / "nodes.npy", nodes_array)

    # Save graphs (+ optional features)
    for i, tkey in enumerate(keys_sorted):
        df, n = grouped[tkey]
        adj = sp.coo_matrix((df['w'].to_numpy(),
                             (df['s'].to_numpy(), df['d'].to_numpy())),
                             shape=(n, n),
                             dtype=np.float32).tocsr()
        sp.save_npz(out_dir / "graphs" / f"adj_ts_{i:03d}.npz", adj)

        if make_feats:
            feats = degrees_as_features(adj)
            sp.save_npz(out_dir / "features" / f"feats_ts_{i:03d}.npz", feats)

    # Save meta
    meta = {
        "num_snapshots": len(keys_sorted),
        "time_keys": keys_sorted,
        "has_features": bool(make_feats),
        "node_count": int(len(nodes_array)),
        "adj_pattern": "graphs/adj_ts_{:03d}.npz",
        "feats_pattern": "features/feats_ts_{:03d}.npz" if make_feats else None
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Saved {len(keys_sorted)} snapshots to: {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv_folder", type=str, help="Folder with multiple CSV files (filenames contain YYYYMMDD)")
    src.add_argument("--single_csv", type=str, help="Single CSV with columns source,target,time,weight")

    ap.add_argument("--granularity", type=str, default="day",
                    choices=["day","month","year"],
                    help="How to bin time into snapshots")
    ap.add_argument("--out_dir", type=str, default="data/processed/CAIDA")
    ap.add_argument("--no_features", action="store_true", help="Do not generate node features")
    ap.add_argument("--has_header", action="store_true", help="Folder CSVs have header row (default: detect minimal)")
    args = ap.parse_args()

    if args.csv_folder:
        df = read_folder(args.csv_folder, has_header=args.has_header)
    else:
        df = read_single_csv(args.single_csv)

    keys_sorted, grouped, nodes_array = make_snapshots(df, granularity=args.granularity)
    save_outputs(args.out_dir, keys_sorted, grouped, nodes_array, make_feats=(not args.no_features))

if __name__ == "__main__":
    main()
