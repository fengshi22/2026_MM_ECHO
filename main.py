import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from ECHO_model import MultiViewNet
from semi_dataset_load import DatasetLoader


ROOT = Path(__file__).resolve().parent

DATASETS = (
    "20newsgroups", "esp_game", "Hdigit", "NUSWIDE",
    "NUSWide20k", "OutdoorScene", "prokaryotic", "UCI",
)

METRICS = ("ACC", "Precision", "Recall", "Macro_F1", "Micro_F1")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def training(conf, data):
    features, labels, train_ids, test_ids, adj_list, relations = data

    set_seed(conf.seed)
    model = MultiViewNet(conf, adj_list)

    if conf.device.startswith("cuda"):
        torch.cuda.synchronize(conf.device)

    start = time.perf_counter()
    scores = model.fit(features, relations, labels, train_ids, test_ids)

    if conf.device.startswith("cuda"):
        torch.cuda.synchronize(conf.device)

    elapsed = time.perf_counter() - start
    values = np.asarray(scores, dtype=float)

    if values.size != len(METRICS) or not np.isfinite(values).all():
        raise RuntimeError("Invalid evaluation results.")

    return values


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate ECHO.")
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="20newsgroups")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    os.chdir(ROOT)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        parser.error("Use cpu or cuda:N.")

    base = OmegaConf.load("ECHO_config.yaml")
    common = {key: base[key] for key in base if key not in DATASETS}
    selected = DATASETS if args.dataset == "all" else [args.dataset]

    for name in selected:
        conf = OmegaConf.merge(common, base[name])
        conf.dataset_name = name
        conf.device = str(device)
        conf.data_dir = str(ROOT / "data" / f"{name}_{conf.knn}")
        conf.log_dir = str(ROOT / "logs" / f"{name}_{conf.knn}")

        if args.epochs is not None:
            conf.max_epoch = args.epochs

        conf.seed = 22
        set_seed(conf.seed)
        data = DatasetLoader(conf, load_from_file=True)
        features, labels, train_ids, test_ids, _, relations = data

        conf.num_nodes = int(labels.numel())
        conf.num_classes = int(torch.unique(labels).numel())
        conf.num_views = len(features)
        conf.num_view_feature_dims = [x.shape[1] for x in features]

        Path(conf.data_dir).mkdir(parents=True, exist_ok=True)
        for view, edges in enumerate(relations):
            graph_path = Path(conf.data_dir) / f"view{view}_graph.txt"
            if not graph_path.exists():
                np.savetxt(
                    graph_path,
                    edges.detach().cpu().numpy(),
                    fmt="%d",
                )

        conf.seed = args.seed
        values = training(conf, data)

        print(f"Dataset: {name}")
        for metric, value in zip(METRICS, values):
            scale, unit = (100, "%")
            print(f"{metric}: {value * scale:.2f} {unit}")


if __name__ == "__main__":
    main()