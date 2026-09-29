"""ACM experiment entry point. Run only when explicitly requested."""

from __future__ import annotations

import argparse
import json
import time

from auto_phgt.data import load_acm
from auto_phgt.evaluation import evaluate_full_graph
from auto_phgt.model import AutoPHGT
from auto_phgt.runtime import resolve_device, to_device
from auto_phgt.tokenization import MetaPathInstanceExtractor, impute_missing_features
from auto_phgt.training import fit_full_graph, set_seed, split_ids

from .common import (PATH_SELECTIONS, Stages, add_common_args, build_result, now,
                     path_statistics, resolve_checkpoint, save_result, select_templates,
                     split_record, summary)
from .protocol import hyperparameters as protocol_hyperparameters

SPLIT_METHOD = ("official HGB train/test masks; acm_validation_size training papers held out "
                "for validation with torch.Generator(seed)")


def run(*, mode: str = "auto_phgt", seed: int = 0, epochs: int | None = None,
        patience: int | None = None, d_model: int | None = None, k: int | None = None,
        path_selection: str = "discovered", root: str = "data/acm", device: str = "auto",
        output=None, checkpoint=None, no_checkpoint: bool = False, restart: bool = False,
        template_dir="artifacts/discovery", extra=None, path_seed=None, paths=None,
        ranks=None, token_control=None):
    start = time.perf_counter()
    started_at = now()
    device = resolve_device(device)
    hp = protocol_hyperparameters("acm")
    hp.update({key: value for key, value in
               {"epochs": epochs, "patience": patience, "d_model": d_model, "k": k}.items()
               if value is not None})
    stages = Stages()
    set_seed(seed)
    with stages("load"):
        graph = load_acm(root)
    train_ids, val_ids, test_ids = split_ids(graph, seed=seed,
                                             acm_validation_size=hp["acm_validation_size"])
    with stages("discovery"):
        templates, discovery = select_templates(
            graph, dataset="acm", selection=path_selection, k=hp["k"],
            max_hops=hp["max_hops"], seed=seed, template_dir=template_dir,
            path_seed=path_seed, paths=paths, ranks=ranks, train_ids=train_ids,
            labels=graph["paper"].y)
    # ACM is small, so its features live on the training device for the whole run.
    with stages("impute"):
        features = to_device(impute_missing_features(graph), device)
    extractor = MetaPathInstanceExtractor(graph, templates,
                                          instances_per_path=hp["instances_per_path"], seed=seed)
    stats = path_statistics(extractor, templates, train_ids) if mode != "hgt" else None
    model = AutoPHGT.build(
        graph, templates, int(graph["paper"].y.max()) + 1, x_dict=features,
        d_model=hp["d_model"], pooling=hp["pooling"], token_dropout=hp["token_dropout"],
        hgt_layers=hp["hgt_layers"], hgt_heads=hp["hgt_heads"],
        fusion_layers=hp["fusion_layers"], fusion_heads=hp["fusion_heads"],
        ffn_mult=hp["ffn_mult"], dropout=hp["dropout"], mode=mode,
        token_control=token_control,
        num_dummy_tokens=hp["k"] * hp["instances_per_path"] if token_control == "dummy" else None,
    ).to(device)
    checkpoint = resolve_checkpoint("acm", mode, seed, checkpoint=checkpoint,
                                    no_checkpoint=no_checkpoint, restart=restart)
    config = {"dataset": "acm", "mode": mode, "seed": seed, "path_selection": path_selection,
              "hyperparameters": hp, "templates": [list(t.schema) for t in templates]}
    # Optional fields enter the config only when used, preserving checkpoint compatibility.
    config.update({key: value for key, value in (("token_control", token_control),
                                                 ("path_seed", path_seed)) if value is not None})
    fit = fit_full_graph(model, graph, features, extractor, train_ids, val_ids,
                         epochs=hp["epochs"], patience=hp["patience"], lr=hp["lr"],
                         weight_decay=hp["weight_decay"], seed=seed,
                         checkpoint_path=checkpoint, config=config, progress=True)
    test = evaluate_full_graph(model, features, graph.edge_index_dict, extractor,
                               graph["paper"].y, test_ids)
    result = build_result(
        dataset="acm", mode=mode, seed=seed, hyperparameters=hp, discovery=discovery,
        device=device, model=model, fit=fit, test=test, checkpoint=checkpoint,
        started_at=started_at, setup=stages,
        extra={"path_selection": path_selection, "k": hp["k"],
               "token_control": token_control, "path_seed": path_seed,
               "split": split_record(train_ids, val_ids, test_ids, SPLIT_METHOD),
               "path_statistics": stats, "total_runtime_s": time.perf_counter() - start,
               **(extra or {})})
    result["output"] = str(save_result(result, output))
    return result


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--k", type=int, default=None, help="number of meta-paths (default 5)")
    parser.add_argument("--path-selection", choices=PATH_SELECTIONS, default="discovered")
    parser.add_argument("--root", default="data/acm")
    args = parser.parse_args()
    result = run(mode=args.mode, seed=args.seed, epochs=args.epochs, patience=args.patience,
                 d_model=args.d_model, k=args.k, path_selection=args.path_selection,
                 root=args.root, device=args.device, output=args.output,
                 checkpoint=args.checkpoint, no_checkpoint=args.no_checkpoint,
                 restart=args.restart)
    print(json.dumps(summary(result) | {"output": result["output"]}, indent=2))


if __name__ == "__main__":
    main()
