from __future__ import print_function
import os, glob, re
import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp

SRC_DIR = "/mnt/kingston/caida_dataset"    # 你的來源資料夾
OUT_DIR = os.path.expanduser("~/dysat_test/DySAT/data/CAIDA")
UNDIRECTED = False      # traceroute 多半是有向；若想當無向圖改 True
USE_WEIGHT = True       # 是否使用 weight 權重

os.makedirs(OUT_DIR, exist_ok=True)

# 1) 找出所有 YYYYMMDD.csv 並排序
files = sorted(glob.glob(os.path.join(SRC_DIR, "*.csv")),
               key=lambda p: re.findall(r"(\d{8})", os.path.basename(p))[0])

print("Found", len(files), "daily CSVs")

# 2) 第一次掃描：收集所有節點 ID，確保各 snapshot 用同一套索引
all_nodes = set()
for fp in files:
    # 兼容有/無表頭：若沒表頭就指定欄名
    try:
        df = pd.read_csv(fp, usecols=[0,1], nrows=5)
        has_header = set(df.columns) >= {"source","target"}
    except Exception:
        has_header = False

    if has_header:
        df = pd.read_csv(fp, usecols=["source","target"])
    else:
        df = pd.read_csv(fp, header=None, names=["source","target","time","weight"], usecols=[0,1])

    all_nodes.update(df["source"].unique().tolist())
    all_nodes.update(df["target"].unique().tolist())

id2idx = {nid:i for i, nid in enumerate(sorted(all_nodes))}
N = len(id2idx)
print("Total unique nodes:", N)

# 3) 第二次掃描：建 MultiGraph snapshots
graphs = []
for fp in files:
    # 讀完整欄位（source,target,time,weight），兼容有/無表頭
    try:
        df = pd.read_csv(fp)
        cols = [c.lower() for c in df.columns]
        df.columns = cols
        need = {"source","target"}
        assert need.issubset(set(cols))
    except Exception:
        df = pd.read_csv(fp, header=None, names=["source","target","time","weight"])

    if "weight" not in df.columns:
        df["weight"] = 1

    G = nx.MultiDiGraph() if not UNDIRECTED else nx.MultiGraph()
    G.add_nodes_from(range(N))  # 固定節點集合

    # 加邊（映射到 0..N-1）
    for s, t, w in df[["source","target","weight"]].itertuples(index=False):
        si = id2idx[s]; ti = id2idx[t]
        if UNDIRECTED and si == ti:  # 無向情境通常不留 self-loop
            continue
        if USE_WEIGHT:
            G.add_edge(si, ti, weight=float(w))
        else:
            G.add_edge(si, ti)

    graphs.append(G)
    print("Built", os.path.basename(fp), "nodes=", G.number_of_nodes(), "edges=", G.number_of_edges())

# 4) 存成 DySAT 需要的 graphs.npz（key 必須叫 'graph'）
np.savez(os.path.join(OUT_DIR, "graphs.npz"), graph=np.array(graphs, dtype=object))
print("Saved graphs.npz with", len(graphs), "snapshots to", OUT_DIR)

# 5) （可選）存 features.npz：用 one-hot（N×N 稀疏）當特徵
make_feats = True
if make_feats:
    feats = [sp.identity(N, dtype=np.float32) for _ in graphs]
    np.savez(os.path.join(OUT_DIR, "features.npz"), feats=np.array(feats, dtype=object))
    print("Saved features.npz (one-hot)")
