# NAMAT: Node-wise Message-Aware Multi-Graph Attention for Cancer Driver Gene Prioritization

NAMAT is a graph neural network for cancer driver gene prioritization that integrates multiple protein-protein interaction (PPI) networks by estimating graph reliability separately for each gene, instead of applying a single fixed weighting scheme across all genes and networks.

## Model Framework

![NAMAT framework overview](NAMAT_architecture.png)

1. **Graph-specific Message Passing** – node representations are propagated independently across each of the six PPI networks, producing a separate message per network for every gene.
2. **Message-aware Reliability Learning** – a shared MLP scores each (gene, network) pair from the gene's current embedding, that network's propagated message, and the gene's log-degree in that network; a temperature-scaled softmax turns the scores into per-network attention weights.
3. **Attention-weighted Message Aggregation** – the per-network messages are combined into a single update using the learned attention weights.
4. **Node Update** – the aggregated message is linearly projected, added to the previous representation through a residual connection, normalized with PairNorm, and passed through a nonlinearity to produce the next-layer node representation.

## Requirements

- python 3.9+
- torch >= 2.0
- numpy >= 1.23
- pandas >= 1.5
- scikit-learn >= 1.1

Install with:

```bash
git clone https://github.com/daothu2023/NAMAT.git
cd NAMAT
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended; training also runs on CPU, just considerably slower.

## Data

All required data files are included in the `Data/` folder of this repository, so no manual download is needed; just clone the repository and run.

- `PPI/{NAME}_PPI.csv`: six PPI network edge lists (`u, v, weight`) — CPDB, STRING, MultiNet, PCNet, IRefIndex, IRefIndex 2015.
- `node_features/multiomics_features_<CANCER>.csv`: cancer-specific multi-omics node feature matrix (mutation frequency, DNA methylation, gene expression, copy-number alteration).
- `labels/<CANCER>true.txt`: driver gene (positive) labels for each cancer type, one gene symbol per line.
- `labels/2187false.txt`: shared passenger-gene (negative) set, 2,187 genes, used across all cancer types.

`<CANCER>` must match one of: `BRCA, BLCA, LUAD, LIHC, THCA, LUSC, ESCA, PRAD, STAD, COAD, UCEC, CESC`.

## Repository Structure

```
NAMAT/
├── README.md
├── requirements.txt
├── model.py       # NAMAT architecture (PairNorm, message-aware attention, NAMAT module)
├── utils.py        # data loading, graph construction, splitting, metrics
├── run_model.py     # main entry point (CLI, grid search, training / evaluation loop)
└── Data/
    ├── PPI/               # six PPI network edge lists
    ├── labels/            # driver / passenger gene labels
    └── node_features/     # multi-omics node features, one CSV per cancer type
```

## Usage

```bash
# Run NAMAT for BRCA with default settings (10 runs x 5-fold cross-validation with grid search in each fold)
python run_model.py --root ./Data --outdir ./outputs --cancer BRCA

# Run NAMAT for BRCA with a single fixed hyperparameter configuration (no grid search)
python run_model.py --root ./Data --outdir ./outputs --cancer BRCA --mode fixed

# Run for LUAD with fewer runs/epochs, e.g. for a quick sanity check
python run_model.py --root ./Data --outdir ./outputs --cancer LUAD --n_runs 1 --max_epochs 30 --patience 15
```

`--max_epochs 30 --patience 15` caps training to 30 epochs and stops early if validation loss doesn't improve for 15 epochs — to quickly check the pipeline. Omit these two flags to use the full training schedule (380 epochs, patience 80).

The script will:

- Load the six PPI networks and the cancer-specific multi-omics feature matrix from `--root`.
- Build the shared gene universe from the intersection of the omics features and the union of all PPI networks (cached for reuse across runs).
- Run stratified 5-fold cross-validation, repeated over `--n_runs` random seeds, training NAMAT with node-wise message-aware attention over all six PPI networks. In each fold, the training part is further split into a training set and a validation set (85/15, stratified).
- With `--mode grid` (default): in each fold, train every hyperparameter configuration in `SEARCH_SPACE` with early stopping on validation loss, select the configuration with the highest validation AUPRC, retrain it once, and evaluate the retrained model on the held-out test fold.
- With `--mode fixed`: train a single configuration, given by the command-line arguments, in each fold and evaluate it on the held-out test fold.
- Print per-fold validation/test AUPRC, a per-run summary, and an overall mean ± std AUPRC across all folds and runs.

Run `python run_model.py --help` for the full list of configurable hyperparameters (learning rate, dropout, attention temperature schedule, PPI dropout, warm-up epochs, etc.).

### Grid search space

The search space is defined by `SEARCH_SPACE` at the top of `run_model.py`. Each hyperparameter is given as a list: a list with one value keeps it fixed, and a list with several values adds it to the search. All combinations are evaluated (8 configurations by default). In grid mode, the values in `SEARCH_SPACE` are used instead of the corresponding command-line arguments. The default space is (searched hyperparameters in bold):

| Hyperparameter | Key in `SEARCH_SPACE` | Values |
| --- | --- | --- |
| **Learning rate** | `LR` | **1e-3, 1e-4** |
| **Input MLP layers** | `LIN_IN_LAYERS` | **2, 3** |
| **Dropout** | `DROPOUT` | **0.4, 0.3** |
| Weight decay (AdamW) | `WEIGHT_DECAY` | 4e-4 |
| Input MLP hidden size | `LIN_IN_HIDDEN` | 64 |
| Input MLP output size | `LIN_IN_OUT_DIM` | 64 |
| Message-passing hidden sizes | `HIDS` | [64, 128] |
| Feature dropout | `FEATDROP_P` | 0.20 |
| PPI dropout | `PPI_DROPOUT_P` | 0.25 |
| Label smoothing | `LABEL_SMOOTHING` | 0.06 |
| Attention temperature, start and end | `FUSION_TEMP_START`, `FUSION_TEMP_END` | 0.8, 0.6 |
| Gate warm-up epochs | `GATE_WARMUP_EPOCHS` | 30 |

Settings outside the search space are set from the command line and are the same in both modes:

| Setting | Argument | Value |
| --- | --- | --- |
| Maximum epochs | `--max_epochs` | 380 |
| Early-stopping patience (validation loss) | `--patience` | 80 |
| Learning-rate warm-up epochs | `--warmup_epochs` | 5 |
| Folds and runs | `--n_folds`, `--n_runs` | 5 folds, 10 runs |
