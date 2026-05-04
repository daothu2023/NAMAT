import argparse
import hashlib
import os
from dataclasses import dataclass, replace
from typing import Dict, List

import numpy as np
import torch

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
    print_intersection_check,
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


def build_ppi_paths(root: str) -> Dict[str, str]:
    return {
        "CPDB": f"{root}/CPDB_PPI_FULL.csv",
        "STRING": f"{root}/STRINGdb_PPI_FULL.csv",
        "MULTINET": f"{root}/MULTINET_PPI_FULL.csv",
        "PCNet": f"{root}/PCNET_PPI_FULL.csv",
        "IRefIndex": f"{root}/IREF_PPI_FULL.csv",
        "IRefIndex_2015": f"{root}/IREF_2015_PPI_FULL.csv",
    }


def make_cfg(args) -> Config:
    ppi_paths = build_ppi_paths(args.root)

    return Config(
        POS_TXT=f"{args.root}/{args.cancer}true.txt",
        NEG_TXT=args.neg_txt if args.neg_txt else f"{args.root}/2187false.txt",
        FEATURES_CSV=args.features_csv if args.features_csv else f"{args.root}/multiomics_features_{args.cancer}.csv",
        FEATURES_GENE_COL=args.features_gene_col,
        PPI_PATHS=ppi_paths,
        UNIVERSE_CACHE_PATH=f"{args.root}/{args.cancer}_cache_universe_nodes.pkl",
        OUTDIR=f"{args.root}/MULTI6PPI_OMIC_ONLY_WITHMLP_WITHRES_{args.cancer}",
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


def train_namat(nodes, X_final, A_list, mask_list, logdeg_list, y_np, splits, cfg: Config, run_id: int):
    device = cfg.DEVICE
    N = len(nodes)

    Xd = X_final.to(device)
    Ad = [A.to(device) for A in A_list]
    y_full = torch.tensor(y_np, dtype=torch.float32, device=device)

    mask_list = [m.to(device) for m in mask_list]
    logdeg_list = [d.to(device) for d in logdeg_list]

    fold_test = []

    for fold, (train_sel, val_sel, test_sel) in enumerate(splits, start=1):
        train_sel = np.asarray(train_sel, dtype=np.int64)
        val_sel = np.asarray(val_sel, dtype=np.int64)
        test_sel = np.asarray(test_sel, dtype=np.int64)

        print(
            f"\n[NAMAT (OMIC ONLY | 6PPI | WITH MLP | WITH RES) | Run {run_id} | Fold {fold}] "
            f"train(pos={(y_np[train_sel] == 1).sum()},neg={(y_np[train_sel] == 0).sum()}) "
            f"val(pos={(y_np[val_sel] == 1).sum()},neg={(y_np[val_sel] == 0).sum()}) "
            f"test(pos={(y_np[test_sel] == 1).sum()},neg={(y_np[test_sel] == 0).sum()})"
        )

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
            test_ap = safe_auprc_from_logits(v_logits[test_sel], y_np[test_sel].astype(np.int64))

            if ep % 40 == 0 or ep == cfg.MAX_EPOCHS - 1:
                print(
                    f"[Fold {fold} | ep={ep:03d}] "
                    f"val_loss={val_loss:.4f} "
                    f"val_AUPRC={val_ap:.4f} "
                    f"test_AUPRC={test_ap:.4f}"
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
            print(f"[WARN] Fold {fold}: best_state None -> skip fold")
            continue

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
        fold_test.append(float(test_ap_final))

        print(
            f"[Fold {fold}/{cfg.N_FOLDS}] "
            f"FINAL val={val_ap_final:.4f} "
            f"test={test_ap_final:.4f} "
            f"best_vloss={best_val_loss:.4f}"
        )

    return fold_test


def main_one(cfg: Config, cancer: str, run_id: int = 1):
    set_seed(cfg.SEED)
    os.makedirs(cfg.OUTDIR, exist_ok=True)

    print(f"\n========== RUN {run_id} (SEED={cfg.SEED}) ==========")
    print(f"[CANCER] {cancer} | [OUTDIR] {cfg.OUTDIR}")
    print("[MODE] NAMAT | OMIC ONLY | 6PPI | WITH lin_in MLP | WITH residual")
    print("Using device:", cfg.DEVICE)

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

    if cfg.DO_INTERSECTION_CHECK:
        print_intersection_check(ppi_raw_all, feat_omic, pos, neg)

    nodes = compute_global_nodes(ppi_raw_all, feat_omic, cfg.UNIVERSE_CACHE_PATH, cfg.CACHE_UNIVERSE)
    print(f"[GLOBAL] Nodes={len(nodes):,}")
    print(
        "[DEBUG:NODES]",
        "md5=",
        hashlib.md5((",".join(nodes)).encode()).hexdigest()[:12],
        "head=",
        nodes[:5],
    )

    y_np, splits = make_splits(nodes, pos, neg, cfg)
    labeled = y_np >= 0
    pos_all = int((y_np[labeled] == 1).sum())
    neg_all = int((y_np[labeled] == 0).sum())
    print(
        f"[LABELS] labeled={int(labeled.sum()):,} "
        f"pos={pos_all:,} "
        f"neg={neg_all:,} "
        f"pos_rate={pos_all / max(1, pos_all + neg_all):.4f}"
    )

    aligned_multi = align_edges_to_nodes(ppi_raw_all, nodes)
    ppi_order = ["CPDB", "STRING", "MULTINET", "PCNet", "IRefIndex", "IRefIndex_2015"]
    aligned_multi = {k: aligned_multi[k] for k in ppi_order if k in aligned_multi}
    ppi_names = list(aligned_multi.keys())

    print("[DEBUG] PPI order used:", ppi_names)
    assert len(ppi_names) == 6, f"Expected 6 PPIs but got {len(ppi_names)}: {ppi_names}"

    A_list = build_A_list(aligned_multi, nodes, cfg.DEVICE)
    masks_cpu, logdeg_cpu, deg0_pct = build_deg_mask_and_logdeg_fast(
        aligned_multi,
        nodes,
        deg_feat_clip=cfg.DEG_FEAT_CLIP,
    )

    print("[DEBUG] %deg==0 per PPI (real edges, no self-loop):")
    print("   " + " | ".join([f"{ppi_names[i]}:{deg0_pct[i] * 100:.2f}%" for i in range(len(ppi_names))]))

    X_omic, feat_cols = build_X_on_nodes(nodes, feat_omic)
    X_omic = torch.nan_to_num(X_omic, nan=0.0, posinf=0.0, neginf=0.0)
    X_final = zscore_columns_torch(X_omic, eps=1e-6)
    X_final = torch.nan_to_num(X_final, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"[X_FINAL] shape={tuple(X_final.shape)} (omic_only={len(feat_cols)})")

    print("\n" + "=" * 80)
    print("[RUN] NAMAT (OMIC ONLY | 6PPI) | WITH MLP | WITH RESIDUAL")

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
    )

    print(
        f"\n[SUMMARY] Run {run_id} NAMAT: "
        f"mean={float(np.mean(au_6ppi)):.4f} ± "
        f"{float(np.std(au_6ppi)):.4f} "
        f"(n={len(au_6ppi)})"
    )

    return au_6ppi


def main_n_runs(cfg: Config, cancer: str, n_runs: int = 1):
    all_6 = []

    for r in range(n_runs):
        cfg_r = replace(cfg, SEED=cfg.SEED + r)
        au6 = main_one(cfg_r, cancer=cancer, run_id=r + 1)
        all_6.extend(au6)

    print("\n" + "=" * 80)
    print("[ALL RUNS SUMMARY] Test AUPRC (mean ± std)")
    if len(all_6):
        print(
            f"NAMAT : "
            f"{float(np.mean(all_6)):.4f} ± "
            f"{float(np.std(all_6)):.4f} "
            f"(n={len(all_6)})"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run NAMAT 6PPI gated GCN.")

    parser.add_argument("--root", type=str, default="/content/drive/MyDrive/PPI_STRING")
    parser.add_argument("--cancer", type=str, default="BLCA")
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--features_csv", type=str, default=None)
    parser.add_argument("--features_gene_col", type=str, default="gene")
    parser.add_argument("--neg_txt", type=str, default=None)

    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=380)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=4e-4)
    parser.add_argument("--dropout", type=float, default=0.40)
    parser.add_argument("--label_smoothing", type=float, default=0.06)
    parser.add_argument("--warmup_epochs", type=int, default=5)

    parser.add_argument("--hids", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--featdrop_p", type=float, default=0.20)

    parser.add_argument("--fusion_temp_start", type=float, default=0.8)
    parser.add_argument("--fusion_temp_end", type=float, default=0.6)
    parser.add_argument("--gate_warmup_epochs", type=int, default=30)
    parser.add_argument("--ppi_dropout_p", type=float, default=0.25)

    parser.add_argument("--no_mask_deg0", action="store_true")
    parser.add_argument("--no_deg_aware", action="store_true")
    parser.add_argument("--deg_feat_clip", type=float, default=8.0)

    parser.add_argument("--lin_in_layers", type=int, default=3)
    parser.add_argument("--lin_in_hidden", type=int, default=64)
    parser.add_argument("--lin_in_out_dim", type=int, default=64)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = make_cfg(args)
    print("Using device:", cfg.DEVICE)
    main_n_runs(cfg, cancer=args.cancer, n_runs=args.n_runs)
