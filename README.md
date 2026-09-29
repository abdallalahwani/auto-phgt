# Auto-PHGT

Auto-PHGT is a research implementation of semantic-token discovery for
heterogeneous-graph node classification. It enumerates schema-valid
target-to-target meta-paths, samples complete concrete instances, tokenizes
their node features, and fuses semantic tokens with HGT representations.

The canonical structural-volume score for a path of \(h\) edges is
\[
S(p)=\left(N_{\mathrm{target}}\prod_{r\in p}
        \frac{E_r}{N_{\mathrm{src}(r)}}\right)^{1/h}.
\]
Here \(E_r/N_{\mathrm{src}(r)}\) is the relation's mean out-degree, not a
transition probability.

## Source layout and research campaigns

* `auto_phgt/` contains the discovery, CSR instance sampling, tokenization,
  HGT/transformer fusion, heterogeneous neighbor sampling, training and
  evaluation used by every campaign.
* `experiments/run_acm.py`, `run_mag.py`, `baseline_suite.py`, `aggregate.py`:
  initial ACM/MAG experiments and summaries.
* `experiments/selector_controls.py`, `selector_controls_summary.py`:
  controls, manually specified paths, and multi-seed selector comparisons.
* `experiments/residual_selection/` and `residual_campaign.py`:
  dataset splits, HGB-style HGT trainer, RCMS, protocol, training tasks and
  aggregation for ACM, DBLP, Freebase and OGBN-MAG.
* `experiments/lightweight_selection/`, `lightweight_campaign.py`, and
  `lightweight_recovery.py`: Transition-Info, FastPath and set-aware selection,
  frozen protocol, experiment task graph and recovery.
* `experiments/method_comparison/`: EdgeOverlap, multi-label IMDB training,
  candidate scoring/selection, frozen matrix, worker, resuming controller and
  completion-aware aggregation for ACM, DBLP and IMDB.
* `experiments/freebase_comparison.py`: separately frozen Freebase EdgeOverlap
  extension with matched historical reference checks.
* `tests/`: synthetic-graph tests and campaign-level protocol/aggregation
  checks. The Freebase integration tests require locally generated, ignored
  campaign artifacts and skip explicitly when those are absent.

The EdgeOverlap rule selects the next path \(p\) to maximize
\(\mathrm{TI}_{\mathrm{cov}}(p)(1-\max_{q\in S}O(p,q))\), where \(O\) is the mean
per-source Jaccard of typed directed edges from complete sampled paths.

## Install and validate

Use Python 3.12 and install an appropriate PyTorch build for your CPU or GPU,
then `pip install -r requirements.txt`. For an RTX 2080 Ti or Titan Xp, the
PyTorch 2.9.0 CUDA 12.6 build supports both GPUs:

```sh
pip install 'torch==2.9.0+cu126' --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pytest -q tests
```

GPU-heavy jobs must run inside an authorized compute allocation, not on a
login node. The Python modules expose `--help` for entry points:

```sh
python -m experiments.run_acm --help
python -m experiments.run_mag --help
python -m experiments.residual_campaign --help
python -m experiments.lightweight_campaign --help
python -m experiments.method_comparison.worker --help
python -m experiments.method_comparison.controller --help
python -m experiments.freebase_comparison --help
```

## Reproduction and provenance

This is a **source-only publication**, not a copy of the original experiment
workspace. Campaign output IDs and ignored `artifacts/v2/`, `artifacts/v3/`,
`artifacts/v4_hedge/`, and `artifacts/v5/` paths retain their historical names
so existing results can still be identified. Source modules have descriptive
names here rather than the old `experiments.v3`, `experiments.v4_hedge`, and
`experiments.v5` imports. Historical protocol locks hash the **original**
source filenames and contents: they cannot be applied directly to renamed
files in this release. Fresh execution needs its own compatible protocol
freeze and locally generated data/results; the later comparison additionally
requires earlier compatible result files. Do not infer that running a reference
entry point reproduces a past published table.

Datasets, frozen locks, raw results, checkpoints, logs, smoke output, Slurm
launch scripts, notebooks and the project proposal are intentionally not
included. No results or official benchmark claims are stored in this repo.
