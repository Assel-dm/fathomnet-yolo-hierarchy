"""Evaluate a native YOLO11 checkpoint without changing its training or detection AP

Run from the repository root, for example::

    python -m fathomnet_yolo.evaluate_hierarchy --model runs/detect/baseline/weights/best.pt \
        --data prepared/fathomnet/data.yaml --taxonomy metadata/taxonomy_fathomnet2025.json

The separate hierarchy diagnostic matches boxes independently of their class.
Its scores describe matched objects and must be read with the reported coverage.
"""

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

from fathomnet_yolo.metrics import DetectionHierarchyAccumulator
from fathomnet_yolo.taxonomy import Taxonomy
from ultralytics import __version__ as ultralytics_version
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import RANK, nms
from ultralytics.utils.files import increment_path


class HierarchyDetectionValidator(DetectionValidator):
    """Keep native AP and add an independent class-agnostic hierarchy diagnostic"""

    def __init__(self, *args, taxonomy, distance_mode="edge_count", match_iou=0.5, diagnostic_conf=0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.taxonomy_path = Path(taxonomy).expanduser().resolve()
        self.distance_mode = distance_mode
        self.match_iou = match_iou
        self.diagnostic_conf = diagnostic_conf

    def init_metrics(self, model):
        super().init_metrics(model)
        if self.end2end or self.args.single_cls:
            raise ValueError("Use an ordinary multiclass YOLO11 detection checkpoint.")
        self.taxonomy = Taxonomy.from_json(self.taxonomy_path, self.names, distance_mode=self.distance_mode)
        # A matching checkpoint alone does not guarantee that this dataset uses its class order
        Taxonomy.from_json(self.taxonomy_path, self.data["names"], distance_mode=self.distance_mode)
        alternative_mode = "lca_height" if self.distance_mode == "edge_count" else "edge_count"
        self.alternative_taxonomy = Taxonomy.from_json(self.taxonomy_path, self.names, distance_mode=alternative_mode)
        self.hierarchy = DetectionHierarchyAccumulator(
            self.taxonomy, iou_threshold=self.match_iou, conf_threshold=self.diagnostic_conf
        )
        self.hierarchy_rows = []

    def postprocess(self, preds):
        raw = preds[0] if isinstance(preds, (tuple, list)) else preds
        diagnostic_raw = raw.clone()
        outputs = super().postprocess(preds)
        if diagnostic_raw.ndim != 3 or diagnostic_raw.shape[1] != self.nc + 4:
            raise ValueError("Expected YOLO11 detection output shaped (batch, 4 + classes, candidates).")
        full_scores = diagnostic_raw[:, 4 : 4 + self.nc].transpose(1, 2)
        diagnostic, indices = nms.non_max_suppression(
            diagnostic_raw,
            conf_thres=self.hierarchy.conf_threshold,
            iou_thres=self.args.iou,
            nc=self.nc,
            multi_label=False,
            agnostic=True,
            max_det=self.args.max_det,
            return_idxs=True,
        )
        self.hierarchy_predictions = [
            (diagnostic[index][:, :4], full_scores[index, indices[index].long().reshape(-1)])
            for index in range(len(outputs))
        ]
        return outputs

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        for index, (diagnostic_boxes, diagnostic_scores) in enumerate(self.hierarchy_predictions):
            truth = self._prepare_batch(index, batch)
            boxes = diagnostic_boxes.detach().cpu().float().numpy()
            scores = diagnostic_scores.detach().cpu().float().numpy()
            targets = truth["cls"].detach().cpu().numpy().astype("int64")
            matches = self.hierarchy.update(truth["bboxes"].detach().cpu().float().numpy(), targets, boxes, scores)
            for gt_index, pred_index, overlap in matches:
                ranking = (-scores[pred_index]).argsort(kind="stable")
                self.hierarchy_rows.append(
                    {
                        "image": str(truth["im_file"]),
                        "gt_index": gt_index,
                        "target_class": int(targets[gt_index]),
                        "predicted_class": int(ranking[0]),
                        "iou": overlap,
                        "confidence": float(scores[pred_index].max()),
                        "top20_classes": ranking[:20].tolist(),
                    }
                )

    def get_stats(self):
        stats = super().get_stats()
        self.hierarchy_report = {
            "taxonomy": self.taxonomy.metadata(),
            "protocol": {
                "scope": "geometrically matched objects, independent of class",
                "confidence": self.hierarchy.conf_threshold,
                "matching_iou": self.hierarchy.iou_threshold,
                "matching": "greedy descending IoU, then confidence",
                "nms": "class-agnostic, single-label diagnostic; native AP unchanged",
                "nms_iou": self.args.iou,
                "max_det": self.args.max_det,
                "ahd_at_k": "mean distance over every one of the top min(k, num_classes) labels",
                "mistake_severity": "mean top-1 distance over exact-label mistakes among matched objects",
                "undefined_metrics": "null; no matched objects is not a zero error rate",
            },
            "metrics": self.hierarchy.compute(),
            "alternative_distance": {
                "mode": self.alternative_taxonomy.distance_mode,
                "metrics": self.hierarchy.compute(self.alternative_taxonomy),
            },
        }
        for filename, payload in (
            ("hierarchy_metrics.json", self.hierarchy_report),
            ("hierarchy_matches.json", self.hierarchy_rows),
        ):
            (self.save_dir / filename).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        stats.update(
            {
                f"hierarchy/{key}": float(value) if value is not None else math.nan
                for key, value in self.hierarchy_report["metrics"].items()
            }
        )
        mode = self.alternative_taxonomy.distance_mode
        stats.update(
            {
                f"hierarchy_{mode}/{key}": float(value) if value is not None else math.nan
                for key, value in self.hierarchy_report["alternative_distance"]["metrics"].items()
                if key == "mistake_severity" or key.startswith("avg_hierarchical_distance")
            }
        )
        return stats


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path, help="Native trained YOLO11 .pt checkpoint")
    parser.add_argument("--data", required=True, type=Path, help="Prepared YOLO dataset YAML")
    parser.add_argument("--taxonomy", required=True, type=Path, help="Taxonomy generated from the same category order")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="One device, e.g. 0 or cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=Path("runs/fathomnet-hierarchy"))
    parser.add_argument("--name", help="New output directory name; a suffix is added if it already exists")
    parser.add_argument("--diagnostic-conf", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--distance-mode", choices=("edge_count", "lca_height"), default="edge_count")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU, shared by native and diagnostic NMS")
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args(argv)
    if RANK != -1 or "," in str(args.device):
        parser.error("Run this diagnostic in one process on one device, without torchrun/DDP.")
    if args.batch < 1 or args.imgsz < 1 or args.workers < 0 or args.max_det < 1:
        parser.error("batch, imgsz and max-det must be positive; workers must be nonnegative.")
    if not 0 <= args.diagnostic_conf <= 1 or not 0 < args.match_iou <= 1 or not 0 < args.iou <= 1:
        parser.error("diagnostic-conf must be in [0, 1]; match-iou and iou must be in (0, 1].")
    for key in ("model", "data", "taxonomy"):
        path = getattr(args, key).expanduser().resolve()
        if not path.is_file():
            parser.error(f"{key} file not found: {path}")
        setattr(args, key, path)
    if args.model.suffix.lower() != ".pt":
        parser.error("Use a native .pt checkpoint so complete class-score vectors remain available.")
    name = args.name or f"hierarchy-{args.split}"
    if Path(name).name != name or name in {".", ".."}:
        parser.error("name must be a single directory name.")

    # Validate the local taxonomy before loading a model or creating an output directory.
    Taxonomy.from_json(args.taxonomy, distance_mode=args.distance_mode)
    checkpoint_hash = _sha256(args.model)
    output = increment_path(args.project.expanduser().resolve() / name, mkdir=True)
    validation_args = {
        "task": "detect",
        "mode": "val",
        "model": str(args.model),
        "data": str(args.data),
        "split": args.split,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "conf": 0.001,  # Native detection-validation default; separate from diagnostic-conf.
        "iou": args.iou,
        "max_det": args.max_det,
        "rect": True,  # Same default used by YOLO(...).val().
        "single_cls": False,
        "agnostic_nms": False,
        "augment": False,
        "plots": True,
        "save_json": False,
        "save_txt": False,
        "project": str(output.parent),
        "name": output.name,
    }
    validator = HierarchyDetectionValidator(
        taxonomy=args.taxonomy,
        distance_mode=args.distance_mode,
        match_iou=args.match_iou,
        diagnostic_conf=args.diagnostic_conf,
        args=validation_args,
        save_dir=output,
    )
    result = validator(model=str(args.model))
    result = {key: float(value) if math.isfinite(float(value)) else None for key, value in result.items()}
    summary = {
        "ultralytics_version": ultralytics_version,
        "python_version": platform.python_version(),
        "checkpoint": {"path": str(args.model), "sha256": checkpoint_hash},
        "dataset_yaml": {"path": str(args.data), "sha256": _sha256(args.data)},
        "taxonomy_file": {"path": str(args.taxonomy), "sha256": _sha256(args.taxonomy)},
        "split": args.split,
        "validation_args": validation_args,
        "native_detection_metrics": {key: value for key, value in result.items() if not key.startswith("hierarchy")},
        "hierarchy": validator.hierarchy_report,
    }
    (output / "evaluation_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Evaluation saved to {output}")
    return result


if __name__ == "__main__":
    main()
