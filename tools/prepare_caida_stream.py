#!/usr/bin/env python3
import os, re, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

DATE_RE = re.compile(r"(\d{8})")

def list_csvs(folder):
    files = []
    for f in sorted(os.listdir(folder)):
        if not f.endswith(".csv"):
            continue
        if f.endswith("ip_to_id_mapping.csv") or f.endswith("time_to_id_mapping.csv"):
            continue
        m = DATE_RE.search(f)
        if not m:
            # 跳過沒有日期的檔案
            continue
        files.append((f, m.group(1)))
    return files

def degrees_as_features(adj_csr):
    out_deg = np.asarray(adj_csr.sum(axis=1)).ravel()
    in_deg  = np.asarray(adj_csr.sum(axis=0)).ravel()
    feats = np.vstack([np.log1p(in_deg), np.log1p(out_deg)]).T
    return sp.csr_matrix(feats)

def pass1_collect_nodes(folder, files, chunksize):
    nodes = set()
    for fname, _ in files:
        path = os.path.join(folder, fname)
        for chunk in pd.read_csv(path, usecols=["source","target"], chunksize=chunksize):
            nodes.update(chunk["source"].unique().tolist())
            nodes.update(chunk["target"].unique().tolist())
    nodes = np.array(sorted(nodes))
    node2idx = {n:i for i,n in enumerate(nodes)}
    return nodes, node2idx

def build_one_snapshot(folder, fname, node2idx, chunksize):
    # 逐塊讀一天，聚合 (s,d)->w
    agg = {}
    path = os.path.join(folder, fname)
    for chunk in pd.read_csv(path, chunksize=chunksize):
        cols = {c.lower(): c for c in chunk.columns}
        chunk = chunk.rename(columns={cols.get('source','source'):'source',
                                      cols.get('target','target'):'target',
                                      cols.get('weight','weight'):'weight'})
        if 'weight' not in chunk.columns:
            chunk['weight'] = 1.0
        s = chunk['source'].map(node2idx).to_numpy()
        d = chunk['target'].map(node2idx).to_numpy()
        w = chunk['weight'].astype(float).to_numpy()
        for si, di, wi in zip(s, d, w):
            key = (int(si), int(di))
            agg[key] = agg.get(key, 0.0) + float(wi)
    if not agg:
        return None
    # 組稀疏矩陣
    n = len(node2idx)
    rows, cols, data = zip(*[(k[0], k[1], v) for k, v in agg.items()])
    adj = sp.coo_matrix((np.array(data, dtype=np.float32),
                         (np.array(rows), np.array(cols))),
                        shape=(n, n), dtype=np.float32).tocsr()
    return adj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_folder", required=True, help="Folder containing daily CSVs with YYYYMMDD in filename")
    ap.add_argument("--out_dir", default="data/processed/CAIDA")
    ap.add_argument("--chunksize", type=int, default=1_000_000, help="rows per chunk for streaming")
    ap.add_argument("--no_features", action="store_true")
    args = ap.parse_args()

    folder = args.csv_folder
    out_dir = Path(args.out_dir)
    (out_dir / "graphs").mkdir(parents=True, exist_ok=True)
    if not args.no_features:
        (out_dir / "features").mkdir(parents=True, exist_ok=True)

    files = list_csvs(folder)
    if not files:
        raise SystemExit(f"No dated CSV files found under {folder}")

    print(f"[Pass1] scanning nodes from {len(files)} files ...")
    nodes, node2idx = pass1_collect_nodes(folder, files, args.chunksize)
    np.save(out_dir / "nodes.npy", nodes)
    print(f"[Pass1] total unique nodes = {len(nodes)}")

    print("[Pass2] building snapshots ...")
    tkeys = []
    for i, (fname, ymd) in enumerate(files):
        print(f"  -> {fname} ({i+1}/{len(files)})")
        adj = build_one_snapshot(folder, fname, node2idx, args.chunksize)
        if adj is None:
            print("     (empty day)")
            continue
        sp.save_npz(out_dir / "graphs" / f"adj_ts_{i:03d}.npz", adj)
        if not args.no_features:
            feats = degrees_as_features(adj)
            sp.save_npz(out_dir / "features" / f"feats_ts_{i:03d}.npz", feats)
        tkeys.append(ymd)

    meta = {
        "num_snapshots": len(tkeys),
        "time_keys": tkeys,
        "has_features": (not args.no_features),
        "node_count": int(len(nodes)),
        "adj_pattern": "graphs/adj_ts_{:03d}.npz",
        "feats_pattern": "features/feats_ts_{:03d}.npz" if not args.no_features else None
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] snapshots={len(tkeys)} out={out_dir}")

if __name__ == "__main__":
    main()
