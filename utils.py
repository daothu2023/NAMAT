import os
import re
import math
import random
import pickle
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def set_seed(seed: int = 43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_gene_list(path: str) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            s = re.split(r"[\s,;\t]+", s)[0].strip()
            if s:
                out.append(s)

    seen = set()
    out2 = []
    for g in out:
        if g not in seen:
            seen.add(g)
            out2.append(g)
    return out2


def _auto_gene_col(df: pd.DataFrame, preferred: str = "gene") -> str:
    cols = [str(c).strip() for c in df.columns]
    if preferred in cols:
        return preferred

    candidates = [
        "gene", "Gene", "symbol", "Symbol", "gene_symbol", "hgnc_symbol",
        "HGNC", "GeneName", "ENSEMBL", "Ensembl",
    ]
    for c in candidates:
        if c in cols:
            return c
    return cols[0]


def load_features(csv_path: str, gene_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    if gene_col not in df.columns:
        gcol = _auto_gene_col(df, preferred=gene_col)
        if gcol != "gene":
            df = df.rename(columns={gcol: "gene"})
            print(f"[INFO] Auto-detected omics gene column '{gcol}'. Using 'gene'.")
        else:
            first_col = df.columns[0]
            df = df.rename(columns={first_col: "gene"})
            print(f"[WARN] No gene col; use '{first_col}' as 'gene'")
    elif gene_col != "gene":
        df = df.rename(columns={gene_col: "gene"})

    return df


def load_ppi_edges(ppi_path: str) -> pd.DataFrame:
    """
    Load PPI edges as an unweighted undirected graph.

    Returned columns:
      - u: source node
      - v: target node
    """

    def _prefer(x: str) -> str:
        if pd.isna(x):
            return ""

        s = str(x).strip()
        toks = re.findall(r"[bB]?'([^'\]]+)'", s)

        if len(toks) >= 1:
            for t in toks:
                if not t.upper().startswith(("ENSG", "ENST", "ENSP")):
                    return t.strip()
            return toks[0].split(".")[0]

        for sep in ["|", ",", ";", "\t", " "]:
            if sep in s:
                parts = [p.strip() for p in s.split(sep) if p.strip()]
                for p in parts:
                    if not p.upper().startswith(("ENSG", "ENST", "ENSP")):
                        return p
                return parts[0].split(".")[0]

        if s.upper().startswith(("ENSG", "ENST", "ENSP")):
            return s.split(".")[0]
        return s

    df = None
    for enc in ["utf-8", "latin-1"]:
        for sep in [None, "\t", ",", ";"]:
            try:
                df = pd.read_csv(
                    ppi_path,
                    sep=sep,
                    engine="python",
                    encoding=enc,
                    comment="#",
                )
                break
            except Exception:
                df = None
        if df is not None:
            break

    if df is None:
        raise ValueError(f"Cannot read PPI file: {ppi_path}")
    if df.shape[1] < 2:
        raise ValueError(f"PPI file must have >=2 columns: {ppi_path}")

    cols = list(df.columns)
    if not ("u" in cols and "v" in cols):
        df = df.rename(columns={cols[0]: "u", cols[1]: "v"})

    df["u"] = df["u"].astype(str).map(_prefer).str.upper().str.strip()
    df["v"] = df["v"].astype(str).map(_prefer).str.upper().str.strip()

    df = df[(df["u"] != "") & (df["v"] != "") & (df["u"] != df["v"])].reset_index(drop=True)

    a = np.minimum(df["u"].values, df["v"].values)
    b = np.maximum(df["u"].values, df["v"].values)
    df["_a"] = a
    df["_b"] = b
    df = df.drop_duplicates(subset=["_a", "_b"], keep="last")
    df = df.drop(columns=["_a", "_b"])

    return df[["u", "v"]].reset_index(drop=True)


def print_intersection_check(ppi_edgelists_all, features_df, pos_genes, neg_genes):
    feat_nodes = set(features_df["gene"].astype(str).str.upper().str.strip().tolist())

    ppi_nodes = set()
    for edf in ppi_edgelists_all.values():
        ppi_nodes.update(edf["u"].tolist())
        ppi_nodes.update(edf["v"].tolist())

    node_set = ppi_nodes & feat_nodes
    pos_set = set(g.upper().strip() for g in pos_genes)
    neg_set = set(g.upper().strip() for g in neg_genes)

    print("\n========== INTERSECTION CHECK (PPI ∩ OMICS FEATURE) ==========")
    print(f"[CHECK] #PPI nodes union: {len(ppi_nodes):,}")
    print(f"[CHECK] #Omics genes:     {len(feat_nodes):,}")
    print(f"[CHECK] #Intersection:    {len(node_set):,}")
    print("\n========== LABEL COUNTS ON INTERSECTION NODES ==========")
    print(f"[CHECK] POS input={len(pos_set):,} kept={len(pos_set & node_set):,} missing={len(pos_set - node_set):,}")
    print(f"[CHECK] NEG input={len(neg_set):,} kept={len(neg_set & node_set):,} missing={len(neg_set - node_set):,}")


def compute_global_nodes(ppi_edgelists_all, features_df, cache_path: str, use_cache: bool):
    if use_cache and cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            nodes = pickle.load(f)
        if isinstance(nodes, list) and len(nodes) > 0:
            print(f"[CACHE] loaded universe_nodes: {len(nodes):,} from {cache_path}")
            return nodes

    feat_nodes = set(features_df["gene"].astype(str).str.upper().str.strip().tolist())

    ppi_nodes = set()
    for edf in ppi_edgelists_all.values():
        ppi_nodes.update(edf["u"].tolist())
        ppi_nodes.update(edf["v"].tolist())

    nodes = sorted(list(ppi_nodes & feat_nodes))
    if len(nodes) == 0:
        raise ValueError("No overlap between PPI nodes and omics feature genes.")

    if use_cache and cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(nodes, f)
        print(f"[CACHE] saved universe_nodes: {len(nodes):,} -> {cache_path}")

    return nodes


def build_X_on_nodes(nodes, features_df):
    fdf = features_df.copy()
    fdf["__key__"] = fdf["gene"].astype(str).str.upper().str.strip()

    feat_cols = [c for c in fdf.columns if c not in ("gene", "__key__")]
    X = fdf.set_index("__key__").loc[nodes, feat_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)

    return torch.tensor(X, dtype=torch.float32), feat_cols


def align_edges_to_nodes(ppi_edgelists, nodes):
    nodes_set = set(nodes)
    aligned = {}
    for name, edf in ppi_edgelists.items():
        df = edf.copy()
        m = df["u"].isin(nodes_set) & df["v"].isin(nodes_set) & (df["u"] != df["v"])
        aligned[name] = df.loc[m, ["u", "v"]].reset_index(drop=True)
    return aligned


def edgelist_to_sparse(edge_df, node_index, self_loops: bool = True, device: str = "cpu"):
    """
    Convert an unweighted edge list into a sparse adjacency matrix.

    All edges are treated as binary connections; each edge is assigned
    a value of 1.0 in the resulting sparse matrix.
    """
    uu, vv, vals = [], [], []

    for a, b in zip(edge_df["u"], edge_df["v"]):
        ia = node_index.get(a)
        ib = node_index.get(b)
        if ia is None or ib is None or ia == ib:
            continue
        uu += [ia, ib]
        vv += [ib, ia]
        # Unweighted graph: assign value 1.0 to each directed edge entry.
        vals += [1.0, 1.0]

    if self_loops:
        N = len(node_index)
        uu += list(range(N))
        vv += list(range(N))
        vals += [1.0] * N

    idx = torch.tensor([uu, vv], dtype=torch.long, device=device)
    val = torch.tensor(vals, dtype=torch.float32, device=device)
    N = len(node_index)
    return torch.sparse_coo_tensor(idx, val, (N, N)).coalesce()


def normalize_adj(A: torch.Tensor) -> torch.Tensor:
    A = A.coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp_min(1e-12)
    inv = deg.pow(-0.5)
    r, c = A.indices()
    val = A.values() * inv[r] * inv[c]
    return torch.sparse_coo_tensor(A.indices(), val, A.size(), device=A.device).coalesce()


def build_A_list(aligned_edges, nodes, device: str):
    node_to_idx = {g: i for i, g in enumerate(nodes)}
    A_list = []
    for _, edf in aligned_edges.items():
        A = edgelist_to_sparse(edf, node_to_idx, self_loops=True, device=device)
        A_list.append(normalize_adj(A))
    return A_list


def build_deg_mask_and_logdeg_fast(aligned_edges, nodes, deg_feat_clip: float = 8.0):
    node_to_idx = {g: i for i, g in enumerate(nodes)}
    N = len(nodes)
    masks, logdeg, deg0_pct = [], [], []

    for _, edf in aligned_edges.items():
        iu = edf["u"].map(node_to_idx).to_numpy(dtype=np.int64, na_value=-1)
        iv = edf["v"].map(node_to_idx).to_numpy(dtype=np.int64, na_value=-1)
        m = (iu >= 0) & (iv >= 0) & (iu != iv)
        iu = iu[m]
        iv = iv[m]

        deg = np.bincount(iu, minlength=N) + np.bincount(iv, minlength=N)
        mask = (deg > 0).astype(np.float32)
        ld = np.log1p(deg.astype(np.float32))
        if deg_feat_clip is not None:
            ld = np.clip(ld, 0.0, float(deg_feat_clip))

        masks.append(torch.from_numpy(mask))
        logdeg.append(torch.from_numpy(ld.astype(np.float32)))
        deg0_pct.append(float((deg == 0).mean()))

    return masks, logdeg, deg0_pct


def cosine_warmup_lr(epoch: int, max_epochs: int, base_lr: float, warmup: int = 5) -> float:
    if epoch < warmup:
        return base_lr * (epoch + 1) / max(1, warmup)
    t = (epoch - warmup) / max(1, max_epochs - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * t))


def make_splits(nodes, pos_genes, neg_genes, cfg):
    N = len(nodes)
    y = np.full(N, -1, dtype=np.int64)

    pos_set = set(g.upper().strip() for g in pos_genes)
    neg_set = set(g.upper().strip() for g in neg_genes)

    for i, g in enumerate(nodes):
        if g in pos_set:
            y[i] = 1
        elif g in neg_set:
            y[i] = 0

    labeled_idx = np.where(y >= 0)[0]
    if len(labeled_idx) == 0:
        raise ValueError("No labeled nodes after intersection.")

    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    splits = []

    for fold, (tr, te) in enumerate(skf.split(labeled_idx, y[labeled_idx])):
        tr_nodes = labeled_idx[tr]
        te_nodes = labeled_idx[te]
        y_tr = y[tr_nodes]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=cfg.SEED + fold)
        tr_i, va_i = next(sss.split(tr_nodes, y_tr))
        train_sel = tr_nodes[tr_i]
        val_sel = tr_nodes[va_i]
        splits.append((train_sel, val_sel, te_nodes))

    return y, splits


def safe_auprc_from_logits(logits_1d: torch.Tensor, y_true_np_1d: np.ndarray) -> float:
    prob = torch.sigmoid(logits_1d).detach().cpu().numpy()
    if not np.isfinite(prob).all():
        prob = np.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)
    if len(np.unique(y_true_np_1d)) < 2:
        return 0.0
    return float(average_precision_score(y_true_np_1d, prob))


def bce_ls(logits, targets_float, pos_weight=None, eps: float = 0.06):
    if eps > 0:
        targets_float = targets_float * (1 - eps) + 0.5 * eps
    return F.binary_cross_entropy_with_logits(logits, targets_float, pos_weight=pos_weight)


def zscore_columns_torch(X: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mu = torch.mean(X, dim=0, keepdim=True)
    sd = torch.std(X, dim=0, keepdim=True, unbiased=False).clamp_min(eps)
    return (X - mu) / sd
