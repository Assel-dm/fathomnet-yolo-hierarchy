"""Add AHD@k and mistake severity metrics."""

import numpy as np


def _targets(values, count):
    values = np.asarray(values)
    if values.ndim != 1 or (values.size and (not np.issubdtype(values.dtype, np.integer))):
        raise ValueError("Targets must be a one-dimensional integer array")
    values = values.astype(np.int64)
    if ((values < 0) | (values >= count)).any():
        raise ValueError("Class ID outside taxonomy class order")
    return values


def classification_metrics(targets, scores, taxonomy, topk=(1, 5, 20)):
    """Compute metrics in the detector class order, preserving unknown ranks.

    Empty populations yield ``None`` rates. For nonempty populations with no mistakes, severity is zero, matching the
    preceding classification protocol. At a rank only known ground-truth ancestors enter the denominator; a prediction
    without this ancestor counts as incorrect. Score ties are resolved by increasing class ID. ``@k`` uses ``min(k,
    num_classes)`` labels
    """
    count = len(taxonomy.class_names)
    targets = _targets(targets, count)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(targets), count) or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite with shape (num_objects, num_classes)")
    topk = tuple(sorted({1, *topk}))
    if any(type(k) is not int or k < 1 for k in topk):
        raise ValueError("topk values must be positive integers")
    ranking = np.argsort(-scores, axis=1, kind="stable")
    predictions = ranking[:, 0]
    mistakes = predictions != targets
    distances = taxonomy.distance_matrix[targets, predictions]
    result = {
        "num_objects": len(targets),
        "num_errors": int(mistakes.sum()),
        "top1_accuracy": float((~mistakes).mean()) if len(targets) else None,
        "top1_error": float(mistakes.mean()) if len(targets) else None,
        "mistake_severity": float(distances[mistakes].mean()) if mistakes.any() else (0.0 if len(targets) else None),
    }
    for k in topk:
        top = ranking[:, : min(k, count)]
        result[f"top{k}_accuracy"] = float((top == targets[:, None]).any(axis=1).mean()) if len(targets) else None
        result[f"avg_hierarchical_distance@{k}"] = (
            float(taxonomy.distance_matrix[targets[:, None], top].mean()) if len(targets) else None
        )
    for rank, mapping in taxonomy.ancestors_by_rank.items():
        true_ancestors, predicted_ancestors = mapping[targets], mapping[predictions]
        known = np.asarray([ancestor is not None for ancestor in true_ancestors], dtype=bool)
        result[f"{rank}_support"] = int(known.sum())
        result[f"{rank}_accuracy_top1_ancestor"] = (
            float((true_ancestors[known] == predicted_ancestors[known]).mean()) if known.any() else None
        )
    return result


def _boxes(values):
    boxes = np.asarray(values, dtype=np.float64)
    if boxes.shape == (0,):
        boxes = boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError("Boxes must be finite xyxy arrays with shape (N, 4)")
    if (boxes[:, 2:] <= boxes[:, :2]).any():
        raise ValueError("Boxes must have positive width and height")
    return boxes


def match_boxes(gt_boxes, pred_boxes, confidences, iou_threshold=0.5):
    """Greedily match by decreasing IoU, independently of classes.

    Both arrays must use the same coordinate system. Ties use decreasing prediction confidence, then increasing
    prediction index and GT index. Returns ``(gt_index, pred_index, iou)`` triples. This diagnostic assignment is
    deliberately separate from AP matching, and does not maximize cardinality
    """
    gt_boxes, pred_boxes = _boxes(gt_boxes), _boxes(pred_boxes)
    confidences = np.asarray(confidences, dtype=np.float64)
    if confidences.shape != (len(pred_boxes),) or not np.isfinite(confidences).all():
        raise ValueError("Confidences must align with prediction boxes")
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    lower = np.maximum(gt_boxes[:, None, :2], pred_boxes[None, :, :2])
    upper = np.minimum(gt_boxes[:, None, 2:], pred_boxes[None, :, 2:])
    intersection = np.maximum(upper - lower, 0).prod(axis=-1)
    gt_area = (gt_boxes[:, 2:] - gt_boxes[:, :2]).prod(axis=1)
    pred_area = (pred_boxes[:, 2:] - pred_boxes[:, :2]).prod(axis=1)
    iou = intersection / (gt_area[:, None] + pred_area[None, :] - intersection)
    candidates = [(int(gt), int(pred), float(iou[gt, pred])) for gt, pred in np.argwhere(iou >= iou_threshold)]
    candidates.sort(key=lambda pair: (-pair[2], -confidences[pair[1]], pair[1], pair[0]))
    matches, used_gt, used_pred = [], set(), set()
    for gt, pred, overlap in candidates:
        if gt not in used_gt and pred not in used_pred:
            matches.append((gt, pred, overlap))
            used_gt.add(gt)
            used_pred.add(pred)
    return matches


class DetectionHierarchyAccumulator:
    """Aggregate conditional taxonomy scores and box coverage over images.

    Call ``update`` with predictions *after NMS* and their complete sigmoid score vectors; apply identical NMS settings
    to every compared experiment The common confidence threshold is applied here to each vector's maximum Unmatched
    prediction counts concern geometry only: a localized object with a wrong label is matched and included as a
    classification error
    """

    def __init__(self, taxonomy, iou_threshold=0.5, conf_threshold=0.25, topk=(1, 5, 20)):
        if not 0 < iou_threshold <= 1 or not 0 <= conf_threshold <= 1:
            raise ValueError("Invalid IoU or confidence threshold")
        self.taxonomy = taxonomy
        self.iou_threshold = float(iou_threshold)
        self.conf_threshold = float(conf_threshold)
        self.topk = topk
        self.num_images = self.num_gt = self.num_predictions = self.num_matched = 0
        self._targets, self._scores, self._ious = [], [], []

    def update(self, gt_boxes, gt_classes, pred_boxes, pred_scores):
        """Add one image and return matches in the original prediction indexing."""
        gt_boxes, pred_boxes = _boxes(gt_boxes), _boxes(pred_boxes)
        gt_classes = _targets(gt_classes, len(self.taxonomy.class_names))
        scores = np.asarray(pred_scores, dtype=np.float64)
        if len(gt_boxes) != len(gt_classes):
            raise ValueError("Ground-truth boxes and classes are misaligned")
        if scores.shape != (len(pred_boxes), len(self.taxonomy.class_names)) or not np.isfinite(scores).all():
            raise ValueError("Full prediction score vectors must align with boxes and classes")
        if ((scores < 0) | (scores > 1)).any():
            raise ValueError("Detection scores must be sigmoid scores in [0, 1]")
        confidence = scores.max(axis=1)
        kept = np.flatnonzero(confidence >= self.conf_threshold)
        matches = match_boxes(gt_boxes, pred_boxes[kept], confidence[kept], self.iou_threshold)
        self.num_images += 1
        self.num_gt += len(gt_boxes)
        self.num_predictions += len(kept)
        self.num_matched += len(matches)
        for gt, pred, overlap in matches:
            self._targets.append(int(gt_classes[gt]))
            self._scores.append(scores[kept[pred]].copy())
            self._ious.append(overlap)
        return [(gt, int(kept[pred]), overlap) for gt, pred, overlap in matches]

    def compute(self, taxonomy=None):
        """Return matched-object metrics, optionally using a second distance convention."""
        taxonomy = taxonomy or self.taxonomy
        if (taxonomy.class_names, taxonomy.class_node_ids) != (
            self.taxonomy.class_names,
            self.taxonomy.class_node_ids,
        ):
            raise ValueError("Alternative taxonomy must use the same ordered class labels and taxon IDs")
        scores = np.asarray(self._scores, dtype=np.float64).reshape(-1, len(self.taxonomy.class_names))
        result = classification_metrics(np.asarray(self._targets, dtype=np.int64), scores, taxonomy, self.topk)
        result.update(
            {
                "num_images": self.num_images,
                "num_gt": self.num_gt,
                "num_predictions": self.num_predictions,
                "num_matched": self.num_matched,
                "num_missed": self.num_gt - self.num_matched,
                "num_unmatched_predictions": self.num_predictions - self.num_matched,
                "matched_rate": self.num_matched / self.num_gt if self.num_gt else None,
                "prediction_matched_rate": self.num_matched / self.num_predictions if self.num_predictions else None,
                "mean_matched_iou": float(np.mean(self._ious)) if self._ious else None,
            }
        )
        return result
