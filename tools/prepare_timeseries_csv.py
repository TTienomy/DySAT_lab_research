#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

def degrees_as_features(adj_csr):
    out_deg = np.asarray(adj_csr.sum(axis=1)).ravel()
    in_deg  = np.asarray(adj_csr.sum(axis=0)).ravel()
    feats = np.vstack([in_deg, out_deg]).T  # [in, out]
    feats = np.log1p(feats)
    return sp.csr_matrix(feats)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with columns: source,target,time,weight")
    ap.add_argument("--out_dir", default="data/processed/CAIDA", help="Output dir")
    ap.add_argument("--no_features", action="store_true", help="Do not emit node features")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "graphs").mkdir(parents=True, exist_ok=True)
    if not args.no_features:
        (out_dir / "features").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    need = {"source","target","time"}
    if not need.issubset({c.lower() for c in df.columns}):
        raise ValueError("CSV must have columns: source,target,time[,weight]")

    # normalize column names
    df.columns = [c.lower() for c in df.columns]
    if "weight" not in df.columns:
        df["weight"] = 1.0

    # make sure types are sane
    for col in ["source","target","time"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0)

    # node indexing across all snapshots (consistent)
    nodes = pd.Index(pd.unique(df[["source","target"]].values.ravel()))
    node2idx = {n:i for i,n in enumerate(nodes)}
    np.save(out_dir / "nodes.npy", nodes.to_numpy())

    # sort unique timesteps (integers are fine)
    tkeys = sorted(df["time"].unique().tolist())

    for i, t in enumerate(tkeys):
        g = df[df["time"] == t]
        s = g["source"].map(node2idx).to_numpy()
        d = g["target"].map(node2idx).to_numpy()
        w = g["weight"].astype(float).to_numpy()

        # aggregate multi-edges within the snapshot
        gg = pd.DataFrame({"s": s, "d": d, "w": w}).groupby(["s","d"], as_index=False)["w"].sum()

        n = len(nodes)
        adj = sp.coo_matrix((gg["w"].to_numpy(), (gg["s"].to_numpy(), gg["d"].to_numpy())),
                            shape=(n, n), dtype=np.float32).tocsr()
        sp.save_npz(out_dir / "graphs" / f"adj_ts_{i:03d}.npz", adj)

        if not args.no_features:
            feats = degrees_as_features(adj)
            sp.save_npz(out_dir / "features" / f"feats_ts_{i:03d}.npz", feats)

    meta = {
        "num_snapshots": len(tkeys),
        "time_keys": [int(t) for t in tkeys],
        "has_features": (not args.no_features),
        "node_count": int(len(nodes)),
        "adj_pattern": "graphs/adj_ts_{:03d}.npz",
        "feats_pattern": "features/feats_ts_{:03d}.npz" if not args.no_features else None
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] snapshots={len(tkeys)} nodes={len(nodes)} out={out_dir}")

if __name__ == "__main__":
    main()
