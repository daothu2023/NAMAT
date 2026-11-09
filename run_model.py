import argparse
import itertools
import os
import warnings
from dataclasses import dataclass, replace
from typing import Dict, List

import numpy as np
import torch

warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")

from model import NAMAT
from utils import (
    align_edges_to_nodes,
    bce_ls,
    build_A_list,
    build_X_on_nodes,
    build_deg_mask_and_logdeg_fast,
    compute_global_nodes,
    cosine_warmup_lr,
    load_features,
    load_ppi_edges,
    make_splits,
    read_gene_list,
    safe_auprc_from_logits,
    set_seed,
    zscore_columns_torch,
)


@dataclass
class Config:
    POS_TXT: str
    NEG_TXT: str
    FEATURES_CSV: str
    FEATURES_GENE_COL: str = "gene"
    PPI_PATHS: Dict[str, str] = None

    CACHE_UNIVERSE: bool = True
    UNIVERSE_CACHE_PATH: str = ""
    DO_INTERSECTION_CHECK: bool = True

    SEED: int = 42
    N_FOLDS: int = 5

    MAX_EPOCHS: int = 380
    PATIENCE: int = 80
    LR: float = 1e-3
    WEIGHT_DECAY: float = 4e-4
    DROPOUT: float = 0.40
    LABEL_SMOOTHING: float = 0.06
    WARMUP_EPOCHS: int = 5

    HIDS: List[int] = None
    FEATDROP_P: float = 0.20

    FUSION_TEMP_START: float = 0.8
    FUSION_TEMP_END: float = 0.6
    GATE_WARMUP_EPOCHS: int = 30

    PPI_DROPOUT_P: float = 0.25
    MASK_DEG0: bool = True

    USE_DEG_AWARE: bool = True
    DEG_FEAT_CLIP: float = 8.0

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    OUTDIR: str = ""

    LIN_IN_LAYERS: int = 3
    LIN_IN_HIDDEN: int = 64
    LIN_IN_OUT_DIM: int = 64

# ===================== Hyperparameter search space (mode="grid") =====================
# Every key is a Config field. A list with one value keeps that hyperparameter fixed;
# a list with several values adds it to the grid search. To search more values,
# add them to the corresponding list. All combinations are evaluated.
# In grid mode these values are used instead of the corresponding command-line arguments.
SEARCH_SPACE = {
    "LR": [1e-3, 1e-4],
    "LIN_IN_LAYERS": [2, 3],
    "DROPOUT": [0.4, 0.3],
    "WEIGHT_DECAY": [4e-4],
    "LIN_IN_HIDDEN": [64],
    "LIN_IN_OUT_DIM": [64],
    "HIDS": [[64, 128]],
    "FEATDROP_P": [0.20],
    "PPI_DROPOUT_P": [0.25],
    "LABEL_SMOOTHING": [0.06],
    "FUSION_TEMP_START": [0.8],
    "FUSION_TEMP_END": [0.6],
    "GATE_WARMUP_EPOCHS": [30],
}


def build_candidate_configs(cfg) -> List[Dict]:
    """All combinations of SEARCH_SPACE (combinations with FUSION_TEMP_START < FUSION_TEMP_END are skipped)."""
    for k in SEARCH_SPACE:
        if not hasattr(cfg, k):
            raise KeyError(f"SEARCH_SPACE key '{k}' is not a Config field")
    keys = list(SEARCH_SPACE)
    candidates = []
    for values in itertools.product(*(SEARCH_SPACE[k] for k in keys)):
        hp = dict(zip(keys, values))
        if hp.get("FUSION_TEMP_START", cfg.FUSION_TEMP_START) < hp.get("FUSION_TEMP_END", cfg.FUSION_TEMP_END):
            continue
        candidates.append(hp)
    if len(candidates) == 0:
        raise ValueError("SEARCH_SPACE produced no valid configuration.")
    return candidates


def format_hp(hp: Dict) -> str:
    """One-line description of the searched hyperparameters (those with more than one value)."""
    searched = [k for k in SEARCH_SPACE if len(SEARCH_SPACE[k]) > 1] or list(SEARCH_SPACE)
    return ", ".join(f"{k}={hp[k]}" for k in searched)


def build_ppi_paths(root: str) -> Dict[str, str]:
    ppi_dir = f"{root}/PPI"
    return {
        "CPDB": f"{ppi_dir}/CPDB_PPI.csv",
        "STRING": f"{ppi_dir}/STRING_PPI.csv",
        "MULTINET": f"{ppi_dir}/MULTINET_PPI.csv",
        "PCNet": f"{ppi_dir}/PCNET_PPI.csv",
        "IRefIndex": f"{ppi_dir}/IREF_PPI.csv",
        "IRefIndex_2015": f"{ppi_dir}/IREF_2015_PPI.csv",
    }


def make_cfg(args) -> Config:
    ppi_paths = build_ppi_paths(args.root)

    return Config(
        POS_TXT=f"{args.root}/labels/{args.cancer}true.txt",
        NEG_TXT=args.neg_txt if args.neg_txt else f"{args.root}/labels/2187false.txt",
        FEATURES_CSV=args.features_csv if args.features_csv else f"{args.root}/node_features/multiomics_features_{args.cancer}.csv",
        FEATURES_GENE_COL=args.features_gene_col,
        PPI_PATHS=ppi_paths,
        UNIVERSE_CACHE_PATH=f"{args.outdir}/{args.cancer}_cache_universe_nodes.pkl",
        OUTDIR=f"{args.outdir}/MULTI6PPI_OMIC_ONLY_WITHMLP_WITHRES_{args.cancer}",
        SEED=args.seed,
        N_FOLDS=args.n_folds,
        MAX_EPOCHS=args.max_epochs,
        PATIENCE=args.patience,
        LR=args.lr,
        WEIGHT_DECAY=args.weight_decay,
        DROPOUT=args.dropout,
        LABEL_SMOOTHING=args.label_smoothing,
        WARMUP_EPOCHS=args.warmup_epochs,
        HIDS=args.hids,
        FEATDROP_P=args.featdrop_p,
        FUSION_TEMP_START=args.fusion_temp_start,
        FUSION_TEMP_END=args.fusion_temp_end,
        GATE_WARMUP_EPOCHS=args.gate_warmup_epochs,
        PPI_DROPOUT_P=args.ppi_dropout_p,
        MASK_DEG0=not args.no_mask_deg0,
        USE_DEG_AWARE=not args.no_deg_aware,
        DEG_FEAT_CLIP=args.deg_feat_clip,
        LIN_IN_LAYERS=args.lin_in_layers,
        LIN_IN_HIDDEN=args.lin_in_hidden,
        LIN_IN_OUT_DIM=args.lin_in_out_dim,
    )


def train_one_fold(Xd, Ad, mask_list, logdeg_list, y_np, y_full, train_sel, val_sel, test_sel, cfg: Config, fold: int):
    """Train NAMAT on one fold (early stopping on validation loss) and return (val AUPRC, test AUPRC)
    of the best-validation-loss checkpoint, or None if no valid checkpoint was obtained."""
    device = cfg.DEVICE
    N = Xd.shape[0]

    train_sel = np.asarray(train_sel, dtype=np.int64)
    val_sel = np.asarray(val_sel, dtype=np.int64)
    test_sel = np.asarray(test_sel, dtype=np.int64)

    train_mask = torch.zeros(N, dtype=torch.bool, device=device)
    val_mask = torch.zeros(N, dtype=torch.bool, device=device)
    train_mask[torch.tensor(train_sel, device=device)] = True
    val_mask[torch.tensor(val_sel, device=device)] = True

    model = NAMAT(
        in_dim=Xd.shape[1],
        hids=cfg.HIDS,
        dropout=cfg.DROPOUT,
        featdrop_p=cfg.FEATDROP_P,
        use_deg_aware=cfg.USE_DEG_AWARE,
        lin_in_layers=cfg.LIN_IN_LAYERS,
        lin_in_hidden=cfg.LIN_IN_HIDDEN,
        lin_in_out_dim=cfg.LIN_IN_OUT_DIM,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    best_val_loss = float("inf")
    best_state = None
    no_imp = 0

    for ep in range(cfg.MAX_EPOCHS):
        for group in opt.param_groups:
            group["lr"] = cosine_warmup_lr(ep, cfg.MAX_EPOCHS, cfg.LR, warmup=cfg.WARMUP_EPOCHS)

        temp = (
            (cfg.FUSION_TEMP_START - cfg.FUSION_TEMP_END)
            * max(0, cfg.MAX_EPOCHS - 1 - ep)
            / max(1, cfg.MAX_EPOCHS - 1)
            + cfg.FUSION_TEMP_END
        )

        force_uniform = ep < cfg.GATE_WARMUP_EPOCHS
        ppi_dropout = 0.0 if force_uniform else cfg.PPI_DROPOUT_P

        model.train()
        opt.zero_grad(set_to_none=True)

        logits, _ = model(
            Xd,
            Ad,
            mask_list=mask_list,
            logdeg_list=logdeg_list,
            temp=temp,
            force_uniform=force_uniform,
            ppi_dropout_p=ppi_dropout,
            mask_deg0=cfg.MASK_DEG0,
        )

        if not torch.isfinite(logits).all():
            print(f"[WARN] non-finite logits at ep={ep} -> break")
            break

        ytr = y_full[train_mask]
        npos = int((ytr == 1).sum().item())
        nneg = int((ytr == 0).sum().item())

        posw = None
        if npos > 0 and nneg > 0:
            posw = torch.tensor(
                [min(nneg / max(1, npos), 10.0)],
                dtype=torch.float32,
                device=device,
            )

        loss = bce_ls(
            logits[train_mask],
            y_full[train_mask],
            pos_weight=posw,
            eps=cfg.LABEL_SMOOTHING,
        )

        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if not torch.isfinite(total_norm):
            opt.zero_grad(set_to_none=True)
            continue

        opt.step()

        model.eval()
        with torch.no_grad():
            v_logits, _ = model(
                Xd,
                Ad,
                mask_list=mask_list,
                logdeg_list=logdeg_list,
                temp=cfg.FUSION_TEMP_END,
                force_uniform=False,
                ppi_dropout_p=0.0,
                mask_deg0=cfg.MASK_DEG0,
            )

        v_loss = bce_ls(
            v_logits[val_mask],
            y_full[val_mask],
            pos_weight=posw,
            eps=cfg.LABEL_SMOOTHING,
        )
        val_loss = v_loss.item()

        val_ap = safe_auprc_from_logits(v_logits[val_sel], y_np[val_sel].astype(np.int64))

        if ep % 40 == 0 or ep == cfg.MAX_EPOCHS - 1:
            print(
                f"[Fold {fold} | ep={ep:03d}] "
                f"val_AUPRC={val_ap:.4f}"
            )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= cfg.PATIENCE:
                break

    if best_state is None:
        return None

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    with torch.no_grad():
        logits_best, _ = model(
            Xd,
            Ad,
            mask_list=mask_list,
            logdeg_list=logdeg_list,
            temp=cfg.FUSION_TEMP_END,
            force_uniform=False,
            ppi_dropout_p=0.0,
            mask_deg0=cfg.MASK_DEG0,
        )

    val_ap_final = safe_auprc_from_logits(logits_best[val_sel], y_np[val_sel].astype(np.int64))
    test_ap_final = safe_auprc_from_logits(logits_best[test_sel], y_np[test_sel].astype(np.int64))
    return float(val_ap_final), float(test_ap_final)


def train_namat(nodes, X_final, A_list, mask_list, logdeg_list, y_np, splits, cfg: Config, run_id: int,
                mode: str = "grid"):
    """mode="fixed": one hyperparameter configuration (cfg) for every fold.
    mode="grid": for every fold, train each configuration in SEARCH_SPACE on the training split with
    early stopping on validation loss, select the configuration with the highest validation AUPRC,
    retrain it once, and evaluate the retrained model on the held-out test fold."""
    device = cfg.DEVICE

    Xd = X_final.to(device)
    Ad = [A.to(device) for A in A_list]
    y_full = torch.tensor(y_np, dtype=torch.float32, device=device)

    mask_list = [m.to(device) for m in mask_list]
    logdeg_list = [d.to(device) for d in logdeg_list]

    candidates = build_candidate_configs(cfg) if mode == "grid" else None

    fold_test = []

    for fold, (train_sel, val_sel, test_sel) in enumerate(splits, start=1):
        args_fold = (Xd, Ad, mask_list, logdeg_list, y_np, y_full, train_sel, val_sel, test_sel)

        if mode == "grid":
            best_hp, best_val_auprc = None, -1.0
            for i, hp in enumerate(candidates, start=1):
                print(f"[Fold {fold}/{cfg.N_FOLDS}] Candidate configuration {i}/{len(candidates)}: {format_hp(hp)}")
                res = train_one_fold(*args_fold, replace(cfg, **hp), fold)
                if res is None:
                    print(f"[WARN] Fold {fold}: best_state None -> skip candidate")
                    continue
                print(f"[Fold {fold}/{cfg.N_FOLDS}] Validation: AUPRC={res[0]:.4f}")
                if res[0] > best_val_auprc:
                    best_val_auprc, best_hp = res[0], hp
            if best_hp is None:
                print(f"[WARN] Fold {fold}: no valid candidate -> skip fold")
                continue
            print(f"[Fold {fold}/{cfg.N_FOLDS}] Selected configuration: {format_hp(best_hp)}, "
                  f"validation AUPRC={best_val_auprc:.4f}")
            cfg_fold = replace(cfg, **best_hp)          # retrain once with the selected configuration
        else:
            cfg_fold = cfg

        res = train_one_fold(*args_fold, cfg_fold, fold)
        if res is None:
            print(f"[WARN] Fold {fold}: best_state None -> skip fold")
            continue
        val_ap_final, test_ap_final = res
        fold_test.append(float(test_ap_final))

        print(
            f"[Fold {fold}/{cfg.N_FOLDS}] "
            f"FINAL val={val_ap_final:.4f} "
            f"test={test_ap_final:.4f}"
        )

    return fold_test


def main_one(cfg: Config, cancer: str, run_id: int = 1, mode: str = "grid"):
    set_seed(cfg.SEED)
    os.makedirs(cfg.OUTDIR, exist_ok=True)

    print(f"\n========== RUN {run_id} (SEED={cfg.SEED}) ==========")
    print(f"[CANCER] {cancer} | [OUTDIR] {cfg.OUTDIR}")

    pos = read_gene_list(cfg.POS_TXT)
    neg = read_gene_list(cfg.NEG_TXT)

    feat_omic = load_features(cfg.FEATURES_CSV, cfg.FEATURES_GENE_COL)
    feat_omic["gene"] = feat_omic["gene"].astype(str).str.upper().str.strip()

    ppi_raw_all = {}
    for nm, path in cfg.PPI_PATHS.items():
        if not os.path.exists(path):
            print(f"[WARN] missing {path}; skip {nm}")
            continue
        ppi_raw_all[nm] = load_ppi_edges(path)

    if len(ppi_raw_all) == 0:
        raise ValueError("No PPI loaded.")

    nodes = compute_global_nodes(ppi_raw_all, feat_omic, cfg.UNIVERSE_CACHE_PATH, cfg.CACHE_UNIVERSE)

    y_np, splits = make_splits(nodes, pos, neg, cfg)

    aligned_multi = align_edges_to_nodes(ppi_raw_all, nodes)
    ppi_order = ["CPDB", "STRING", "MULTINET", "PCNet", "IRefIndex", "IRefIndex_2015"]
    aligned_multi = {k: aligned_multi[k] for k in ppi_order if k in aligned_multi}
    ppi_names = list(aligned_multi.keys())

    assert len(ppi_names) == 6, f"Expected 6 PPIs but got {len(ppi_names)}: {ppi_names}"

    A_list = build_A_list(aligned_multi, nodes, cfg.DEVICE)
    masks_cpu, logdeg_cpu, deg0_pct = build_deg_mask_and_logdeg_fast(
        aligned_multi,
        nodes,
        deg_feat_clip=cfg.DEG_FEAT_CLIP,
    )

    X_omic, feat_cols = build_X_on_nodes(nodes, feat_omic)
    X_omic = torch.nan_to_num(X_omic, nan=0.0, posinf=0.0, neginf=0.0)
    X_final = zscore_columns_torch(X_omic, eps=1e-6)
    X_final = torch.nan_to_num(X_final, nan=0.0, posinf=0.0, neginf=0.0)

    au_6ppi = train_namat(
        nodes,
        X_final,
        A_list,
        masks_cpu,
        logdeg_cpu,
        y_np,
        splits,
        cfg,
        run_id,
        mode=mode,
    )

    print(
        f"\n[SUMMARY] Run {run_id} NAMAT: "
        f"mean={float(np.mean(au_6ppi)):.4f} ± "
        f"{float(np.std(au_6ppi)):.4f} "
        f"(n={len(au_6ppi)})"
    )

    return au_6ppi


def main_n_runs(cfg: Config, cancer: str, n_runs: int = 1, mode: str = "grid"):
    all_6 = []
    run_means = []   # one value per run: test AUPRC averaged over its folds

    for r in range(n_runs):
        cfg_r = replace(cfg, SEED=cfg.SEED + r)
        au6 = main_one(cfg_r, cancer=cancer, run_id=r + 1, mode=mode)
        all_6.extend(au6)
        if len(au6):
            run_means.append(float(np.mean(au6)))

    print("\n" + "=" * 80)
    print("[ALL RUNS SUMMARY] Test AUPRC (mean ± std across run-level means)")
    if len(run_means):
        print(
            f"NAMAT : "
            f"{float(np.mean(run_means)):.4f} ± "
            f"{float(np.std(run_means)):.4f} "
            f"(n_runs={len(run_means)}, n_folds_total={len(all_6)})"
        )
        print("   per-run means: " + ", ".join(f"{m:.4f}" for m in run_means))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run NAMAT 6PPI gated GCN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--root", type=str, default="./Data",
                         help="Path to the Data folder containing PPI/, labels/, node_features/ subfolders")
    parser.add_argument("--outdir", type=str, default="./outputs",
                         help="Directory for the universe-nodes cache")
    parser.add_argument("--cancer", type=str, default="BLCA", help="Cancer type code (e.g. BLCA, BRCA, LUAD)")
    parser.add_argument("--n_runs", type=int, default=10, help="Number of repeated runs with different seeds")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--mode", type=str, default="grid", choices=["grid", "fixed"],
                        help="grid: per-fold grid search over SEARCH_SPACE (selection by validation AUPRC, then retraining); "
                             "fixed: a single configuration given by the command-line arguments")

    parser.add_argument("--features_csv", type=str, default=None, help="Override path to the features CSV file")
    parser.add_argument("--features_gene_col", type=str, default="gene", help="Column name for gene identifiers in the features CSV")
    parser.add_argument("--neg_txt", type=str, default=None, help="Override path to the negative gene list")

    parser.add_argument("--n_folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--max_epochs", type=int, default=380, help="Maximum number of training epochs per fold")
    parser.add_argument("--patience", type=int, default=80, help="Early stopping patience (epochs without validation-loss improvement)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate")
    parser.add_argument("--weight_decay", type=float, default=4e-4, help="AdamW weight decay")
    parser.add_argument("--dropout", type=float, default=0.40, help="Dropout probability")
    parser.add_argument("--label_smoothing", type=float, default=0.06, help="Label smoothing coefficient")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Learning-rate warm-up epochs")

    parser.add_argument("--hids", type=int, nargs="+", default=[64, 128], help="Hidden dimensions of the message-passing layers")
    parser.add_argument("--featdrop_p", type=float, default=0.20, help="Feature-wise dropout probability")

    parser.add_argument("--fusion_temp_start", type=float, default=0.8, help="Starting attention temperature")
    parser.add_argument("--fusion_temp_end", type=float, default=0.6, help="Final attention temperature")
    parser.add_argument("--gate_warmup_epochs", type=int, default=30, help="Epochs with uniform (non-learned) attention before gating starts")
    parser.add_argument("--ppi_dropout_p", type=float, default=0.25, help="PPI-network dropout probability during attention computation")

    parser.add_argument("--no_mask_deg0", action="store_true", help="Disable masking of degree-0 nodes")
    parser.add_argument("--no_deg_aware", action="store_true", help="Disable degree-aware features")
    parser.add_argument("--deg_feat_clip", type=float, default=8.0, help="Clip value for the log-degree feature")

    parser.add_argument("--lin_in_layers", type=int, default=3, help="Number of layers in the input MLP encoder")
    parser.add_argument("--lin_in_hidden", type=int, default=64, help="Hidden size of the input MLP encoder")
    parser.add_argument("--lin_in_out_dim", type=int, default=64, help="Output dimension of the input MLP encoder")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = make_cfg(args)
    print("Using device:", cfg.DEVICE)
    if args.mode == "grid":
        n_cand = len(build_candidate_configs(cfg))
        print(f"Mode: grid search over {n_cand} configurations per fold (+1 retraining)")
    else:
        print("Mode: fixed configuration")
    main_n_runs(cfg, cancer=args.cancer, n_runs=args.n_runs, mode=args.mode)
