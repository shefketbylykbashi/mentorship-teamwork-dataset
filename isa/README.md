# Instance Space Analysis

The input, settings and output of the Instance Space Analysis of the 131 benchmark instances and the ten solvers. The analysis was run in [MATILDA](https://matilda.unimelb.edu.au), which runs the MATLAB InstanceSpace toolkit.

| File | Content |
|---|---|
| `FEATURES.md` | The 58 candidate features: definition and range of each |
| `extract_features.py` | Computes the features from the instance files |
| `build_metadata.py` | Adds the relative performance of the ten solvers |
| `metadata.csv` | The MATILDA input: 131 rows, `Instances`, `Source`, 58 `feature_*` columns, 10 `algo_*` columns |
| `matilda/options.json` | The MATILDA settings |
| `matilda/*.csv` | The MATILDA output tables |

## Rebuilding the input

```bash
python3 isa/extract_features.py     # instances -> isa/features.csv
python3 isa/build_metadata.py       # + results/results.jsonl -> isa/metadata.csv
```

Both use only the Python standard library and reproduce `metadata.csv` exactly.

**Features.** Raw values: no normalisation, bounding or transformation is applied here. `Instances` is `<family>/<file name>` and `Source` is the family (`google`, `type2` … `type7`).

**Performance.** Let U be the sum of the validated scores of a solver on an instance divided by its number of planned runs (10 for the randomised solvers, 1 for the deterministic ones). The `algo_*` value is R = U / max U over the ten solvers, so the best solver on each instance has R = 1.

## Settings

`matilda/options.json` holds the options of the run. The main ones:

| Stage | Setting |
|---|---|
| PRELIM | a solver is good on an instance if R ≥ 0.95 (`perf.epsilon` = 0.05, `MaxPerf`, relative); Box-Cox and z-transformation of the features; outlier bounding off |
| SIFTED | correlation threshold 0.3, K = 10 clusters, random forests of 50 trees |
| PILOT | numerical solution, 30 restarts |
| CLOISTER | p = 0.05, correlation threshold 0.7 |
| PYTHIA | polynomial kernel, 10-fold cross-validation |
| TRACE | purity threshold 0.75 |
| β-hardness | an instance is β-easy if more than `betaThreshold` = 0.55 of the solvers are good on it |

## Output tables

| File | Content |
|---|---|
| `coordinates.csv` | Z₁ and Z₂ of each instance |
| `projection_matrix.csv` | The projection matrix A over the ten selected features |
| `feature_raw.csv`, `feature_process.csv` | The selected features before and after normalisation |
| `algorithm_raw.csv`, `algorithm_process.csv`, `algorithm_bin.csv` | Performance: raw, processed, and the good/not good labels |
| `bounds.csv`, `bounds_prunned.csv` | The CLOISTER boundary |
| `footprint_<solver>_good.csv`, `footprint_<solver>_best.csv` | The TRACE footprint polygons |
| `footprint_performance.csv` | Area, density and purity of every footprint |
| `svm_table.csv`, `algorithm_svm.csv`, `SVM_model_prediction.csv` | PYTHIA: accuracy, precision and recall of each SVM, and its predictions |
| `portfolio.csv`, `portfolio_svm.csv` | The best solver of each instance, observed and selected, as the position of the solver among the `algo_*` columns (1 = `evolved_b` … 10 = `original_f`; 0 in `portfolio_svm.csv` = no solver predicted good) |
| `good_algos.csv`, `beta_easy.csv` | The number of good solvers per instance, and the β-easy label |
| `*_color.csv`, `color_table.csv` | The same quantities scaled for MATILDA's plots |

MATILDA was run with the solver names `refactored_b` … `refactored_f` for the original solvers. In the tables here they are renamed to `original_b` … `original_f`; nothing else was changed.
