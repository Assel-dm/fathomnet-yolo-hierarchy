#!/usr/bin/env python3
"""Run a sequential multi-seed Megalodon campaign with one native control per seed.

Run from the YOLO repository. --dry-run prints the plan without writing or training.
Only report metadata are copied; original checkpoints and datasets stay in place.
"""

import argparse
import csv
import hashlib
import json
import math
import pickle
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

import yaml


def configuration(folder):
    value = yaml.safe_load((folder / "args.yaml").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Invalid args.yaml: {folder}")
    return value


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def signature(cfg):
    ignored = {
        "name",
        "project",
        "save_dir",
        "cfg",
        "data",
        "hxe_alpha",
        "hxe_weight",
        "hxe_taxonomy",
        "seed",
    }
    if not float(cfg.get("soft_weight", 0)):
        ignored.update(key for key in cfg if key.startswith("soft_"))
    return {key: value for key, value in cfg.items() if key not in ignored}


def last_epoch(folder):
    """Read the actual final training epoch, not the best checkpoint's epoch."""
    path = folder / "results.csv"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, skipinitialspace=True))
    if not rows:
        return None
    try:
        value = float(rows[-1]["epoch"])
        return (
            int(value)
            if math.isfinite(value) and value.is_integer() and value > 0
            else None
        )
    except (KeyError, TypeError, ValueError):
        return None


def completion_hashes(folder):
    return {
        name: sha256(folder / name)
        for name in ("args.yaml", "results.csv", "weights/best.pt")
    }


def record_completion(folder, cfg, reason="normal_exit"):
    """Called only after the training subprocess returned successfully."""
    epoch = last_epoch(folder)
    if epoch is None or not (folder / "weights/best.pt").is_file():
        raise ValueError(f"Successful training did not produce valid outputs: {folder}")
    if not (folder / "weights/best.pt").stat().st_size:
        raise ValueError(f"The best checkpoint is empty: {folder}")
    receipt = {
        "schema_version": 1,
        "status": "completed",
        "reason": reason,
        "last_epoch": epoch,
        "configured_epochs": int(cfg["epochs"]),
        "sha256": completion_hashes(folder),
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=folder, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
        temporary.replace(folder / ".campaign_completed.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _finalized_checkpoint(folder, cfg, epoch):
    """Recognize a pre-receipt run finalized by Ultralytics, on CPU only.

    These are the user's own training checkpoints, loaded in the training
    environment. Ultralytics finalizes last.pt for both normal and early stops.
    An intermediate last.pt keeps its epoch and optimizer state.
    """
    path = folder / "weights/last.pt"
    if not path.is_file():
        return False
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("epoch") != -1
            or "optimizer" not in checkpoint
            or checkpoint["optimizer"] is not None
            or checkpoint.get("model") is None
        ):
            return False
        saved = checkpoint.get("train_args", {})
        if not isinstance(saved, dict):
            return False
        if int(saved.get("seed", -1)) != int(cfg["seed"]) or int(
            saved.get("epochs", -1)
        ) != int(cfg["epochs"]):
            return False
        if (
            Path(str(saved.get("data", ""))).expanduser().resolve()
            != Path(str(cfg["data"])).expanduser().resolve()
        ):
            return False
        for key in ("hxe_weight", "soft_weight"):
            if not math.isclose(float(saved.get(key, 0)), float(cfg.get(key, 0))):
                return False
        if float(cfg.get("hxe_weight", 0)) and not math.isclose(
            float(saved.get("hxe_alpha", -1)), float(cfg["hxe_alpha"])
        ):
            return False
        history = checkpoint.get("train_results")
        return not (
            isinstance(history, dict)
            and "epoch" in history
            and (not history["epoch"] or int(float(history["epoch"][-1])) != epoch)
        )
    except (
        ImportError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        RuntimeError,
        EOFError,
        pickle.UnpicklingError,
    ) as error:
        print(f"Cannot verify finalized checkpoint {path}: {error}", file=sys.stderr)
        return False


def completed(folder, cfg):
    if (
        not (folder / "weights/best.pt").is_file()
        or not (folder / "results.csv").is_file()
    ):
        return False
    epoch = last_epoch(folder)
    if epoch is None or not (folder / "weights/best.pt").stat().st_size:
        return False
    if epoch >= int(cfg["epochs"]):
        return True
    receipt = folder / ".campaign_completed.json"
    if receipt.is_file():
        try:
            status = json.loads(receipt.read_text(encoding="utf-8"))
            if (
                isinstance(status, dict)
                and status.get("schema_version") == 1
                and status.get("status") == "completed"
                and status.get("last_epoch") == epoch
                and status.get("configured_epochs") == int(cfg["epochs"])
                and status.get("sha256") == completion_hashes(folder)
            ):
                return True
        except (OSError, ValueError, TypeError):
            pass
    return _finalized_checkpoint(folder, cfg, epoch)


def execute(command, dry):
    print("$ " + shlex.join(map(str, command)), flush=True)
    if not dry:
        subprocess.run(list(map(str, command)), check=True)


def copy_metadata(source, target, filenames, dry):
    for filename in filenames:
        old, new = source / filename, target / filename
        if not old.is_file():
            raise ValueError(f"Missing metadata: {old}")
        if old.resolve() == new.resolve():
            continue
        if new.exists() and old.read_bytes() != new.read_bytes():
            raise ValueError(
                f"Conflicting campaign metadata: {new}. Choose a new --campaign-dir."
            )
        print(f"Metadata: {old} -> {new}")
        if not dry:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, new)


def cached_evaluation(folder, run, data_hash, tax_hash, image_size, version):
    summary_path = folder / "evaluation_summary.json"
    if not summary_path.is_file() or not (folder / "hierarchy_matches.json").is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("ultralytics_version") != version:
        return False
    checkpoint = summary.get("checkpoint", {})
    if (
        PurePosixPath(checkpoint.get("path", "").replace("\\", "/")).parent.parent.name
        != run.name
    ):
        return False
    if checkpoint.get("sha256") != sha256(run / "weights/best.pt"):
        return False
    if summary.get("dataset_yaml", {}).get("sha256") != data_hash:
        return False
    if (
        summary.get("taxonomy_file", {}).get("sha256") != tax_hash
        or summary.get("split") != "val"
    ):
        return False
    expected = {
        "batch": 16,
        "imgsz": image_size,
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "rect": True,
        "single_cls": False,
        "agnostic_nms": False,
        "augment": False,
    }
    if any(
        summary.get("validation_args", {}).get(key) != value
        for key, value in expected.items()
    ):
        return False
    hierarchy = summary.get("hierarchy", {})
    expected = {
        "confidence": 0.25,
        "matching_iou": 0.5,
        "nms_iou": 0.7,
        "max_det": 300,
        "scope": "geometrically matched objects, independent of class",
        "matching": "greedy descending IoU, then confidence",
        "nms": "class-agnostic, single-label diagnostic; native AP unchanged",
    }
    return hierarchy.get("taxonomy", {}).get("distance_mode") == "edge_count" and all(
        hierarchy.get("protocol", {}).get(key) == value
        for key, value in expected.items()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--ce-run", required=True, type=Path)
    parser.add_argument(
        "--ce-eval", type=Path, help="Existing CE evaluation directory, if available"
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--train-root", type=Path, help="Existing Megalodon training project"
    )
    parser.add_argument(
        "--campaign-dir", type=Path, default=Path("runs/megalodon_hxe_campaign")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/megalodon_hxe_campaign")
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.2, 0.4, 0.6])
    parser.add_argument("--lambdas", nargs="+", type=float, default=[1.0])
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Seeds shared by native and HXE; defaults to the seed in --ce-run",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Retrain every requested configuration, including CE, in an empty campaign directory",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(not math.isfinite(v) or v < 0 for v in args.alphas):
        parser.error("alphas must be finite and nonnegative")
    if any(not math.isfinite(v) or v <= 0 for v in args.lambdas):
        parser.error("lambdas must be finite and positive")
    if len(set(args.alphas)) != len(args.alphas) or len(set(args.lambdas)) != len(
        args.lambdas
    ):
        parser.error("Do not repeat alpha or lambda values")
    if args.seeds is not None and (
        any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds)
    ):
        parser.error("Seeds must be distinct nonnegative integers")
    dataset, ce, campaign = [
        p.expanduser().resolve()
        for p in (args.dataset_dir, args.ce_run, args.campaign_dir)
    ]
    runs = args.runs_dir.expanduser().resolve()
    data, taxonomy = dataset / "data.yaml", dataset / "taxonomy_megalodon.json"
    base = configuration(ce)
    if float(base.get("hxe_weight", 0)) or float(base.get("soft_weight", 0)):
        raise ValueError(
            "--ce-run must be the native baseline (hxe_weight=0 and soft_weight=0)"
        )
    if not (ce / "weights/best.pt").is_file() or not (ce / "results.csv").is_file():
        raise ValueError(
            "The completed CE run requires args.yaml, results.csv and weights/best.pt"
        )
    if Path(str(base["data"])).expanduser().resolve() != data:
        raise ValueError("The CE data.yaml path differs from --dataset-dir/data.yaml")
    if not data.is_file() or not taxonomy.is_file():
        raise ValueError(
            "The finalized data.yaml and taxonomy_megalodon.json are required"
        )
    seeds = args.seeds if args.seeds is not None else [int(base["seed"])]
    image_size = int(base.get("imgsz", 640))
    if args.fresh and campaign.exists() and any(campaign.iterdir()):
        raise ValueError("--fresh requires an empty or new --campaign-dir")
    initial_model = str(base["model"])
    if Path(initial_model).name in {"best.pt", "last.pt"}:
        raise ValueError(
            "CE args.yaml must identify the original pretrained model, not a trained checkpoint"
        )
    supports_soft = "soft_weight" in base
    version = "8.4.183"  # Project version; dry-run does not import the training stack.
    if not args.dry_run:
        from ultralytics import __version__
        from ultralytics.cfg import DEFAULT_CFG_DICT

        version = __version__
        supports_soft = "soft_weight" in DEFAULT_CFG_DICT
    train_roots = [runs / "detect/fathomnet-megalodon", campaign / "detect"]
    if args.train_root:
        train_roots.append(args.train_root.expanduser().resolve())
    candidates = sorted(
        {ce} | {p.parent for root in train_roots for p in root.glob("*/args.yaml")}
    )
    evaluation_roots = [runs / "megalodon-hierarchy", campaign / "fathomnet-hierarchy"]
    evaluation_folders = sorted(
        {
            p.parent
            for root in evaluation_roots
            for p in root.rglob("evaluation_summary.json")
        }
    )
    if args.ce_eval:
        evaluation_folders.insert(0, args.ce_eval.expanduser().resolve())
    data_hash, tax_hash = sha256(data), sha256(taxonomy)

    def evaluate(run, prospective=False):
        name = run.name + "_val"
        if not prospective:
            copy_metadata(
                run,
                campaign / "detect" / run.name,
                ["args.yaml", "results.csv"],
                args.dry_run,
            )
            for old in evaluation_folders:
                if cached_evaluation(
                    old, run, data_hash, tax_hash, image_size, version
                ):
                    target = campaign / "fathomnet-hierarchy" / name
                    copy_metadata(
                        old,
                        target,
                        ["evaluation_summary.json", "hierarchy_matches.json"],
                        args.dry_run,
                    )
                    print(f"Reusing evaluation: {old}")
                    return
        target = campaign / "fathomnet-hierarchy" / name
        if target.exists():
            raise ValueError(
                f"Incomplete or incompatible evaluation: {target}. Choose a new --campaign-dir."
            )
        command = [
            sys.executable,
            "-m",
            "fathomnet_yolo.evaluate_hierarchy",
            "--model",
            run / "weights/best.pt",
            "--data",
            data,
            "--taxonomy",
            taxonomy,
            "--split",
            "val",
            "--imgsz",
            image_size,
            "--batch",
            16,
            "--device",
            0,
            "--workers",
            4,
            "--diagnostic-conf",
            0.25,
            "--match-iou",
            0.5,
            "--iou",
            0.7,
            "--max-det",
            300,
            "--distance-mode",
            "edge_count",
            "--project",
            campaign / "fathomnet-hierarchy",
            "--name",
            name,
        ]
        execute(command, args.dry_run)
        if not args.dry_run and not cached_evaluation(
            target, run, data_hash, tax_hash, image_size, version
        ):
            raise ValueError(f"Evaluation could not be validated: {target}")

    def train_one(seed, alpha, weight):
        matching = []
        if not args.fresh:
            for folder in candidates:
                cfg = configuration(folder)
                if (
                    Path(str(cfg.get("data", ""))).expanduser().resolve() == data
                    and signature(cfg) == signature(base)
                    and int(cfg.get("seed", -1)) == seed
                    and not float(cfg.get("soft_weight", 0))
                    and math.isclose(float(cfg.get("hxe_weight", 0)), weight)
                    and (
                        not weight
                        or math.isclose(float(cfg.get("hxe_alpha", 0)), alpha)
                    )
                    and completed(folder, cfg)
                ):
                    matching.append(folder)
        if matching:
            folder = min(
                matching,
                key=lambda item: (
                    not item.is_relative_to(campaign / "detect"),
                    item != ce,
                    str(item),
                ),
            )
            label = f"alpha={alpha:g}, lambda={weight:g}" if weight else "native CE"
            print(f"Reusing trained run for {label}, seed={seed}: {folder}")
            evaluate(folder)
            return
        tag = lambda value: str(float(value)).replace(".", "p")
        name = (
            f"megalodon_grid_hxe_a{tag(alpha)}_w{tag(weight)}_seed{seed}"
            if weight
            else f"megalodon_grid_native_seed{seed}"
        )
        folder = campaign / "detect" / name
        if folder.exists():
            raise ValueError(f"Incomplete or incompatible training directory: {folder}")
        command = [
            "yolo",
            "detect",
            "train",
            f"cfg={ce / 'args.yaml'}",
            f"model={initial_model}",
            f"data={data}",
            f"seed={seed}",
            "resume=False",
            f"save_dir={folder}",
            "exist_ok=False",
            f"project={base.get('project') or 'fathomnet-megalodon'}",
            f"name={name}",
            f"hxe_weight={weight}",
            f"hxe_alpha={alpha}",
            f"hxe_taxonomy={taxonomy if weight else 'None'}",
        ]
        if supports_soft:
            command.append("soft_weight=0")
        execute(command, args.dry_run)
        if args.dry_run:
            evaluate(folder, prospective=True)
        else:
            cfg = configuration(folder)
            record_completion(folder, cfg)
            if not completed(folder, cfg):
                raise ValueError(f"Training output could not be verified: {folder}")
            evaluate(folder)

    count = len(seeds) * (1 + len(args.alphas) * len(args.lambdas))
    print(f"Seeds: {seeds}; {count} configurations including one native CE per seed.")
    for seed in seeds:
        train_one(seed, 0.0, 0.0)
        for weight in args.lambdas:
            for alpha in args.alphas:
                train_one(seed, alpha, weight)

    command = [
        sys.executable,
        "-m",
        "fathomnet_yolo.visualization",
        "--runs-dir",
        campaign,
        "--taxonomy",
        taxonomy,
        "--sweep",
        "alpha",
        "--distance-mode",
        "edge_count",
        "--output",
        args.output,
    ]
    if len(args.lambdas) == 1:
        command.extend(["--fixed-lambda", args.lambdas[0]])
    execute(command, args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.CalledProcessError,
    ) as error:
        raise SystemExit(f"Error: {error}") from error
