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

## Source layout

* `auto_phgt/`: path discovery, CSR instance sampling, tokenization, HGT and
  transformer fusion, node/neighbor sampling, training, and evaluation.
* `experiments/run_acm.py` and `experiments/run_mag.py`: independent ACM and
  OGBN-MAG reference entry points.
* `experiments/hgb_data.py`, `hgb_training.py`, and `multilabel.py`: HGB dataset
  splits, shared full-graph model/training utilities, and IMDB-style
  multi-label loss and F1.
* `experiments/transition_info.py`, `path_ranking.py`, and `edge_overlap.py`:
  label-free transition statistics, training-only FastPath scores, and
  sampled-edge redundancy with a deterministic greedy selector.
* `tests/`: focused unit and synthetic-graph integration tests.

The EdgeOverlap rule selects the next path \(p\) to maximize
\(\mathrm{TI}_{\mathrm{cov}}(p)(1-\max_{q\in S}O(p,q))\), where \(O\) is the mean
per-source Jaccard of typed directed edges from complete sampled paths.
The raw selected paths depend on the dataset and discovery space; they are
not packaged as universal constants.

## Install and test

Use Python 3.12 and install an appropriate PyTorch build for your CPU or GPU,
then `pip install -r requirements.txt`. For an RTX 2080 Ti or Titan Xp, the
PyTorch 2.9.0 CUDA 12.6 build supports both GPUs:

```sh
pip install 'torch==2.9.0+cu126' --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pytest -q tests
```

Dataset downloads and runtime outputs are generated locally and ignored by
Git. For example, from the project root:

```sh
python -m experiments.run_acm --mode auto_phgt --seed 0 --device cuda
python -m experiments.run_mag --mode hgt --seed 0 --device cuda
```

The original research campaigns used additional frozen experiment plans,
hardware-specific launch scripts, saved dataset checksums, and cached results.
Those historical campaigns are **not** reproduced by these two reference
entry points; do not interpret their output as a reproduction of any prior
reported comparison. This source-only release intentionally excludes datasets,
results, checkpoints, hardware launch scripts, historical campaign
orchestrators, notebooks, and private project documents.
