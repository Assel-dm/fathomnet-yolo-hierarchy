"""Build a portable confusion explorer from exported hierarchy diagnostics.

Every confusion statistic is conditional on class-independent geometric matches.
Unmatched detections and missed ground-truth objects are reference counts only;
these diagnostics do not replace native detection precision, recall, or AP.
No checkpoints, pickle objects, source images, or network resources are loaded.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile


_RELATIONS = {"exact", "predicted_ancestor", "predicted_descendant", "other_branch"}
_DISTANCE_MODES = {"edge_count", "lca_height"}
_TOLERANCE = 1e-9


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, label, minimum=0):
    _require(type(value) is int and value >= minimum, f"{label} must be an integer >= {minimum}")
    return value


def _number(value, label, minimum=None, maximum=None):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    _require(finite, f"{label} must be a finite number")
    _require(minimum is None or value >= minimum, f"{label} must be >= {minimum}")
    _require(maximum is None or value <= maximum, f"{label} must be <= {maximum}")
    return value


def _text(value, label):
    _require(isinstance(value, str) and bool(value.strip()), f"{label} must be a nonempty string")
    return value


def _mapping(value, label):
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_strict_pairs, parse_constant=_bad_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON in {path.name}: {exc}") from exc


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_text(value, label):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None, f"{label} must be a SHA-256 digest")
    return value.lower()


class _Taxonomy:
    """Validated tree with explicit ranks; missing ranks are never inferred."""

    def __init__(self, data, metadata):
        _mapping(data, "Taxonomy")
        _mapping(metadata, "Summary taxonomy metadata")
        _require(type(data.get("schema_version")) is int and data["schema_version"] == 1, "Taxonomy schema_version must be 1")
        _require(type(metadata.get("schema_version")) is int and metadata["schema_version"] == 1, "Summary taxonomy schema_version must be 1")
        self.mode = metadata.get("distance_mode")
        _require(isinstance(self.mode, str) and self.mode in _DISTANCE_MODES, "Recorded taxonomy distance_mode must be edge_count or lca_height")
        source_mode = data.get("distance_mode", "edge_count")
        _require(isinstance(source_mode, str) and source_mode in _DISTANCE_MODES, "Invalid taxonomy distance_mode")
        _require(isinstance(data.get("nodes"), list) and data["nodes"], "Taxonomy requires nonempty nodes")
        self.nodes = {}
        for raw_node in data["nodes"]:
            node = _mapping(raw_node, "Taxonomy node")
            node_id = _text(node.get("id"), "Taxonomy node ID")
            _require(node_id not in self.nodes, f"Duplicate taxonomy node ID: {node_id}")
            _require("parent" in node, f"Missing parent field for node {node_id}")
            parent = node["parent"]
            _require(parent is None or isinstance(parent, str) and parent.strip(), f"Invalid parent for node {node_id}")
            rank = node.get("rank")
            _require(rank is None or isinstance(rank, str) and rank.strip(), f"Invalid rank for node {node_id}")
            self.nodes[node_id] = {"id": node_id, "parent": parent, "name": _text(node.get("name"), f"Node {node_id} name"), "rank": rank}
        roots = [node_id for node_id, node in self.nodes.items() if node["parent"] is None]
        _require(len(roots) == 1, "Taxonomy must contain exactly one root")
        self.root_id = roots[0]
        self.paths = {}
        for node_id in self.nodes:
            path, visited, current = [], set(), node_id
            while current is not None:
                _require(current in self.nodes, f"Unknown parent node: {current}")
                _require(current not in visited, f"Taxonomy cycle at node {current}")
                visited.add(current)
                path.append(current)
                current = self.nodes[current]["parent"]
            path.reverse()
            _require(path[0] == self.root_id, f"Node {node_id} is disconnected from the root")
            path_ranks = [self.nodes[key]["rank"] for key in path if self.nodes[key]["rank"] is not None]
            _require(len(path_ranks) == len(set(path_ranks)), f"Repeated rank on path to {node_id}")
            self.paths[node_id] = path
        self.heights = dict.fromkeys(self.nodes, 0)
        for path in self.paths.values():
            for depth, node_id in enumerate(path):
                self.heights[node_id] = max(self.heights[node_id], len(path) - depth - 1)
        raw_classes = data.get("classes")
        _require(isinstance(raw_classes, list) and raw_classes, "Taxonomy requires nonempty classes")
        for entry in raw_classes:
            _mapping(entry, "Taxonomy class")
            _integer(entry.get("class_id"), "Taxonomy class_id")
        raw_classes = sorted(raw_classes, key=lambda item: item["class_id"])
        _require([entry["class_id"] for entry in raw_classes] == list(range(len(raw_classes))), "Taxonomy class IDs must be contiguous from zero")
        self.classes = []
        for entry in raw_classes:
            node_id = entry.get("node_id")
            _require(isinstance(node_id, str) and node_id in self.nodes, f"Unknown class node_id: {node_id}")
            name = _text(entry.get("name"), "Taxonomy class name")
            accepted_name = _text(entry.get("accepted_name", self.nodes[node_id]["name"]), f"Accepted name for class {name}")
            _require(accepted_name == self.nodes[node_id]["name"], f"Accepted name for class {name} differs from its mapped taxonomy node name")
            self.classes.append({"id": entry["class_id"], "name": name, "accepted_name": accepted_name, "node_id": node_id, "rank": self.nodes[node_id]["rank"], "path": list(self.paths[node_id])})
        self.names = [entry["name"] for entry in self.classes]
        self.class_nodes = [entry["node_id"] for entry in self.classes]
        _require(len(set(self.names)) == len(self.names), "Taxonomy class names must be unique")
        _require(len(set(self.class_nodes)) == len(self.class_nodes), "Taxonomy class nodes must be unique")
        _integer(metadata.get("num_nodes"), "Recorded num_nodes", 1)
        _integer(metadata.get("num_classes"), "Recorded num_classes", 1)
        _require(metadata["num_nodes"] == len(self.nodes), "Taxonomy node count differs from summary")
        _require(metadata["num_classes"] == len(self.classes), "Taxonomy class count differs from summary")
        _require(metadata.get("class_names") == self.names, "Taxonomy class names/order differ from summary")
        _require(metadata.get("class_node_ids") == self.class_nodes, "Taxonomy class node IDs/order differ from summary")
        self.ranks = sorted({node["rank"] for node in self.nodes.values() if node["rank"] is not None})
        self.ancestors = {rank: [next((node_id for node_id in self.paths[class_node] if self.nodes[node_id]["rank"] == rank), None) for class_node in self.class_nodes] for rank in self.ranks}
        self._pairs = {}

    def pair(self, target, prediction):
        key = (target, prediction)
        if key in self._pairs:
            return self._pairs[key]
        left_node, right_node = self.class_nodes[target], self.class_nodes[prediction]
        left, right = self.paths[left_node], self.paths[right_node]
        common = 0
        for a, b in zip(left, right):
            if a != b:
                break
            common += 1
        lca_id = left[common - 1]
        if target == prediction:
            relation = "exact"
        elif right_node in left:
            relation = "predicted_ancestor"
        elif left_node in right:
            relation = "predicted_descendant"
        else:
            relation = "other_branch"
        result = {"edge_count": len(left) + len(right) - 2 * common, "lca_height": 0 if target == prediction else self.heights[lca_id], "relation": relation, "lca_id": lca_id}
        self._pairs[key] = result
        return result


def aggregate_rows(rows, classes_count):
    """Aggregate prepared matches; rows are true classes, columns predictions.

    Row and column pair fractions include exact-label matches in denominators.
    Incoming error fractions use every matched prediction assigned to that class.
    Undefined rates and severity with no mistakes are returned as ``None``.
    """
    _integer(classes_count, "classes_count")
    matrix = [[0] * classes_count for _ in range(classes_count)]
    pair_metadata, pair_counts = {}, Counter()
    row_support, predicted_support = Counter(), Counter()
    sources = defaultdict(set)
    all_distances, error_distances = [], []
    population = 0
    for row in rows:
        _mapping(row, "Prepared match")
        target = _integer(row.get("target"), "Prepared target")
        prediction = _integer(row.get("prediction"), "Prepared prediction")
        _require(target < classes_count and prediction < classes_count, "Prepared class ID outside taxonomy")
        distance = _number(row.get("distance"), "Prepared distance", 0)
        relation = row.get("relation")
        _require(isinstance(relation, str) and relation in _RELATIONS, "Invalid prepared relation")
        _require((target == prediction) == (relation == "exact"), "Prepared exact relation disagrees with class IDs")
        _require(target != prediction or distance == 0, "Exact predictions must have zero distance")
        lca_id = _text(row.get("lca_id"), "Prepared lca_id")
        key = (target, prediction)
        metadata = (distance, relation, lca_id)
        _require(key not in pair_metadata or pair_metadata[key] == metadata, "Inconsistent metadata for the same confusion pair")
        pair_metadata[key] = metadata
        pair_counts[key] += 1
        matrix[target][prediction] += 1
        row_support[target] += 1
        predicted_support[prediction] += 1
        population += 1
        all_distances.append(distance)
        if target != prediction:
            sources[prediction].add(target)
            error_distances.append(distance)
    errors = len(error_distances)
    receivers = []
    for class_id in range(classes_count):
        support = predicted_support[class_id]
        correct = matrix[class_id][class_id]
        incorrect = support - correct
        receivers.append({"class_id": class_id, "incoming_errors": incorrect, "distinct_sources": len(sources[class_id]), "predicted_support": support, "correct": correct, "incorrect_fraction": incorrect / support if support else None})
    pairs = []
    for (target, prediction), count in sorted(pair_counts.items(), key=lambda item: (-item[1], item[0])):
        distance, relation, lca_id = pair_metadata[(target, prediction)]
        pairs.append({"target": target, "prediction": prediction, "count": count, "row_fraction": count / row_support[target], "column_fraction": count / predicted_support[prediction], "distance": distance, "relation": relation, "lca_id": lca_id})
    return {"population": population, "errors": errors, "correct": population - errors, "accuracy": (population - errors) / population if population else None, "ahd1": math.fsum(all_distances) / population if population else None, "severity": math.fsum(error_distances) / errors if errors else None, "matrix": matrix, "receivers": receivers, "pairs": pairs}


def _classification_metrics(rows, taxonomy, mode):
    population = len(rows)
    errors = sum(row["target"] != row["prediction"] for row in rows)
    distances = [taxonomy.pair(row["target"], row["prediction"])[mode] for row in rows]
    metrics = {"num_objects": population, "num_errors": errors, "top1_accuracy": (population - errors) / population if population else None, "top1_error": errors / population if population else None, "mistake_severity": math.fsum(distances) / errors if errors else (0.0 if population else None)}
    for k in (1, 5, 20):
        width = min(k, len(taxonomy.classes))
        metrics[f"top{k}_accuracy"] = sum(row["target"] in row["top_classes"][:width] for row in rows) / population if population else None
        metrics[f"avg_hierarchical_distance@{k}"] = math.fsum(taxonomy.pair(row["target"], prediction)[mode] for row in rows for prediction in row["top_classes"][:width]) / (population * width) if population else None
    for rank, ancestor_ids in taxonomy.ancestors.items():
        supported = [row for row in rows if ancestor_ids[row["target"]] is not None]
        metrics[f"{rank}_support"] = len(supported)
        metrics[f"{rank}_accuracy_top1_ancestor"] = sum(ancestor_ids[row["target"]] == ancestor_ids[row["prediction"]] for row in supported) / len(supported) if supported else None
    return metrics


def _compare_metrics(computed, expected, label):
    _mapping(expected, label)
    for key, actual in computed.items():
        _require(key in expected, f"{label} is missing {key}")
        recorded = expected[key]
        if actual is None:
            equal = recorded is None
        elif type(actual) is int:
            equal = type(recorded) is int and recorded == actual
        else:
            try:
                equal = type(recorded) in (int, float) and math.isfinite(recorded) and math.isclose(actual, recorded, rel_tol=_TOLERANCE, abs_tol=_TOLERANCE)
            except OverflowError:
                equal = False
        _require(equal, f"{label}.{key} does not reproduce: computed {actual!r}, recorded {recorded!r}")


def _training_args(path, summary):
    try:
        import yaml
    except ImportError as exc:
        raise ValueError("PyYAML is required to read training arguments") from exc
    try:
        args = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid YAML in {path.name}: {exc}") from exc
    _mapping(args, "Training arguments")
    weight = _number(args.get("hxe_weight", 0), "Training hxe_weight", 0)
    alpha = _number(args.get("hxe_alpha", 0 if weight == 0 else None), "Training hxe_alpha", 0)
    _require(type(args.get("seed")) is int, "Training seed must be an integer")
    checkpoint = _mapping(summary.get("checkpoint"), "Summary checkpoint")
    checkpoint_path = PurePosixPath(_text(checkpoint.get("path"), "Checkpoint path").replace("\\", "/"))
    _require(checkpoint_path.parent.name == "weights", "Checkpoint path must identify a weights file inside the run directory")
    run_name = checkpoint_path.parent.parent.name
    _require(_text(args.get("name"), "Training name") == run_name, "Training args name differs from checkpoint run directory")
    encoded_hxe = re.fullmatch(r"(.+)_hxe_a([0-9]+(?:p[0-9]+)?)_w([0-9]+(?:p[0-9]+)?)_seed([0-9]+)", run_name)
    encoded_native = re.fullmatch(r"(.+)_native_seed([0-9]+)", run_name)
    parameters_reconciled = encoded_hxe is not None or encoded_native is not None
    if encoded_hxe:
        encoded_alpha = float(encoded_hxe.group(2).replace("p", "."))
        encoded_weight = float(encoded_hxe.group(3).replace("p", "."))
        _require(math.isclose(alpha, encoded_alpha, rel_tol=1e-12, abs_tol=1e-12), "Training hxe_alpha differs from the value encoded in the run name")
        _require(math.isclose(weight, encoded_weight, rel_tol=1e-12, abs_tol=1e-12), "Training hxe_weight differs from the value encoded in the run name")
        _require(args["seed"] == int(encoded_hxe.group(4)), "Training seed differs from the value encoded in the run name")
    elif encoded_native:
        _require(weight == 0, "A native run name requires hxe_weight=0")
        _require(args["seed"] == int(encoded_native.group(2)), "Training seed differs from the value encoded in the native run name")
    method = "native" if weight == 0 else ("auxiliary" if alpha == 0 else "hxe")
    run = {"name": run_name, "model": _text(args.get("model"), "Training model"), "method": method, "alpha": alpha, "lambda": weight, "seed": args["seed"], "split": _text(summary.get("split"), "Evaluation split"), "checkpoint_sha256": _sha_text(checkpoint.get("sha256"), "Checkpoint SHA-256")}
    return run, parameters_reconciled


def load_data(matches_path, summary_path, taxonomy_path, args_path, allow_compatible_taxonomy=False):
    """Load, validate and aggregate one exported evaluation without model loads.

    Taxonomy bytes must match the evaluation's recorded SHA-256 by default.
    The explicit compatibility override is accepted only after class metadata,
    every exported top-k/rank/distance metric, and population identities pass.
    It never certifies a different file as the exact evaluation taxonomy.
    """
    _require(type(allow_compatible_taxonomy) is bool, "allow_compatible_taxonomy must be a boolean")
    paths = {"matches": Path(matches_path), "evaluation_summary": Path(summary_path), "taxonomy": Path(taxonomy_path), "training_args": Path(args_path)}
    for role, path in paths.items():
        _require(path.is_file(), f"Missing {role} file: {path}")
    summary = _mapping(_read_json(paths["evaluation_summary"]), "Evaluation summary")
    hierarchy = _mapping(summary.get("hierarchy"), "Summary hierarchy")
    taxonomy_record = _mapping(summary.get("taxonomy_file"), "Summary taxonomy_file")
    expected_taxonomy_sha = _sha_text(taxonomy_record.get("sha256"), "Recorded taxonomy SHA-256")
    actual_taxonomy_sha = _sha256(paths["taxonomy"])
    hash_matches = actual_taxonomy_sha == expected_taxonomy_sha
    _require(hash_matches or allow_compatible_taxonomy, "Taxonomy SHA-256 differs from the evaluation record. Use --allow-compatible-taxonomy only to permit a file that passes all strict compatibility checks.")
    taxonomy = _Taxonomy(_read_json(paths["taxonomy"]), hierarchy.get("taxonomy"))
    run, parameters_reconciled = _training_args(paths["training_args"], summary)
    protocol = _mapping(hierarchy.get("protocol"), "Hierarchy protocol")
    minimum_confidence = _number(protocol.get("confidence"), "Diagnostic confidence threshold", 0, 1)
    minimum_iou = _number(protocol.get("matching_iou"), "Diagnostic matching IoU", 0, 1)
    for key in ("nms_iou",):
        if key in protocol:
            _number(protocol[key], f"Diagnostic {key}", 0, 1)
    if "max_det" in protocol:
        _integer(protocol["max_det"], "Diagnostic max_det", 1)
    raw_rows = _read_json(paths["matches"])
    _require(isinstance(raw_rows, list), "Hierarchy matches must be an array")
    rows, identities = [], set()
    count = len(taxonomy.classes)
    ranking_width = min(20, count)
    for index, raw in enumerate(raw_rows):
        raw = _mapping(raw, f"Match {index}")
        image = _text(raw.get("image"), f"Match {index} image")
        gt_index = _integer(raw.get("gt_index"), f"Match {index} gt_index")
        identity = (image, gt_index)
        _require(identity not in identities, f"Duplicate image/gt_index match at row {index}")
        identities.add(identity)
        target = _integer(raw.get("target_class"), f"Match {index} target_class")
        prediction = _integer(raw.get("predicted_class"), f"Match {index} predicted_class")
        _require(target < count and prediction < count, f"Match {index} class ID outside taxonomy")
        ranking = raw.get("top20_classes")
        _require(isinstance(ranking, list) and ranking_width <= len(ranking) <= count, f"Match {index} must export at least the first {ranking_width} ranked labels")
        for class_id in ranking:
            _integer(class_id, f"Match {index} ranked class ID")
            _require(class_id < count, f"Match {index} ranked class ID outside taxonomy")
        _require(len(ranking) == len(set(ranking)), f"Match {index} ranking contains duplicate classes")
        _require(ranking[0] == prediction, f"Match {index} first ranked label differs from predicted_class")
        confidence = _number(raw.get("confidence"), f"Match {index} confidence", 0, 1)
        iou = _number(raw.get("iou"), f"Match {index} IoU", 0, 1)
        _require(confidence >= minimum_confidence, f"Match {index} confidence is below the diagnostic threshold")
        _require(iou >= minimum_iou, f"Match {index} IoU is below the matching threshold")
        pair = taxonomy.pair(target, prediction)
        rows.append({"id": index, "image": image, "gt_index": gt_index, "target": target, "prediction": prediction, "confidence": confidence, "iou": iou, "distance": pair[taxonomy.mode], "relation": pair["relation"], "lca_id": pair["lca_id"], "top_classes": list(ranking)})
    reference_metrics = _mapping(hierarchy.get("metrics"), "Hierarchy metrics")
    required_counts = ("num_images", "num_gt", "num_predictions", "num_matched", "num_missed", "num_unmatched_predictions")
    for key in required_counts:
        _integer(reference_metrics.get(key), f"Hierarchy metrics.{key}")
    population = len(rows)
    _require(reference_metrics["num_matched"] == population, "Match row count differs from num_matched")
    _require(reference_metrics["num_gt"] == population + reference_metrics["num_missed"], "num_gt must equal num_matched + num_missed")
    _require(reference_metrics["num_predictions"] == population + reference_metrics["num_unmatched_predictions"], "num_predictions must equal num_matched + num_unmatched_predictions")
    _require(reference_metrics["num_images"] >= len({row["image"] for row in rows}), "num_images is smaller than the number of matched image identities")
    computed = _classification_metrics(rows, taxonomy, taxonomy.mode)
    computed["mean_matched_iou"] = math.fsum(row["iou"] for row in rows) / population if population else None
    computed["matched_rate"] = population / reference_metrics["num_gt"] if reference_metrics["num_gt"] else None
    computed["prediction_matched_rate"] = population / reference_metrics["num_predictions"] if reference_metrics["num_predictions"] else None
    _compare_metrics(computed, reference_metrics, "Hierarchy metrics")
    alternative = hierarchy.get("alternative_distance")
    if alternative is not None:
        _mapping(alternative, "Alternative distance")
        alternative_mode = alternative.get("mode")
        _require(isinstance(alternative_mode, str) and alternative_mode in _DISTANCE_MODES and alternative_mode != taxonomy.mode, "Alternative distance must identify the other supported distance mode")
        alternative_computed = _classification_metrics(rows, taxonomy, alternative_mode)
        alternative_computed.update({key: reference_metrics[key] for key in required_counts})
        for key in ("mean_matched_iou", "matched_rate", "prediction_matched_rate"):
            alternative_computed[key] = computed[key]
        _compare_metrics(alternative_computed, alternative.get("metrics"), "Alternative distance metrics")
    aggregates = aggregate_rows(rows, count)
    checks = ["Taxonomy schema, node IDs, rooted acyclic tree, explicit ranks and class mappings validated.", "Taxonomy class names, node IDs, ordering and counts match the evaluation metadata.", "Match image/ground-truth identities are unique; class IDs, rankings, confidences, IoUs and protocol thresholds validated.", "Matched population, exact errors, top-k accuracy, AHD@1/5/20, mistake severity, rank support/accuracy and mean IoU reproduce the evaluation within 1e-9.", "Ground-truth and prediction count identities and matched rates reproduce the evaluation.", "Training arguments use safe YAML; run name agrees with the checkpoint directory."]
    if alternative is not None:
        checks.append("The alternative distance mode and its exported metrics also reproduce within 1e-9.")
    notes = ["Confusions and filtered statistics describe geometrically matched objects only; native detection AP uses its separate evaluation protocol.", "Complete image, missed-object and unmatched-prediction totals are taken from the evaluation summary; exported match rows cannot independently reconstruct those totals.", "Ranking scores, score ties and the geometric matching procedure cannot be rechecked because full score vectors, boxes and prediction identities are not exported.", "Mistake severity is undefined when a selected population contains no exact-label errors."]
    provenance_notes = ["Only exported JSON/YAML records and the local taxonomy were read. No checkpoint, pickle, source image or network request was used.", "Image paths are retained as case identifiers; images are not embedded."]
    if parameters_reconciled:
        checks.append("Training arguments agree with the parameters encoded in the recognized run naming convention.")
    else:
        notes.append("The custom run name does not encode parameters in a recognized convention; alpha, weight and seed are taken from training arguments and cannot be independently reconciled against the run name.")
    if hash_matches:
        checks.append("Taxonomy file SHA-256 matches the exact file recorded by the evaluation.")
    else:
        note = f"Taxonomy SHA-256 differs: recorded {expected_taxonomy_sha}; supplied {actual_taxonomy_sha}. The explicit compatibility override was used after all strict metadata and metric checks passed. This file is compatible with these exported diagnostics but is not certified as the exact evaluation taxonomy."
        notes.append(note)
        provenance_notes.append(note)
    return {"title": "FathomNet confusion explorer", "run": run, "classes": taxonomy.classes, "taxonomy": {"nodes": list(taxonomy.nodes.values()), "root_id": taxonomy.root_id, "distance_mode": taxonomy.mode}, "matches": rows, "reference": {"protocol": protocol, "metrics": reference_metrics, "native_metrics": _mapping(summary.get("native_detection_metrics"), "Native detection metrics")}, "validation": {"checks": checks, "notes": notes}, "provenance": {"sources": [{"role": role, "name": path.name, "sha256": _sha256(path)} for role, path in paths.items()], "notes": provenance_notes}, "aggregates": aggregates}


def _safe_json(payload):
    """A JSON literal safe inside an inline script, including hostile labels."""
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _protect_inputs(output_path, input_paths):
    destination = output_path.resolve()
    for path in input_paths:
        _require(destination != path.resolve(), f"Output must not overwrite input file {path.name}")
        if output_path.exists() and path.exists():
            _require(not os.path.samefile(output_path, path), f"Output must not overwrite an input hard link: {path.name}")


def generate(matches_path, summary_path, taxonomy_path, args_path, output_path, assets_dir=None, allow_compatible_taxonomy=False):
    """Validate inputs and atomically write one self-contained offline HTML file.

    The assets directory contains ``template.html`` and ``d3.v7.9.0.min.js``.
    The template must contain ``__D3_SOURCE__`` and ``__CONFUSION_DATA__`` once
    each. Returns the absolute output :class:`pathlib.Path`.
    """
    output = Path(output_path)
    assets = Path(assets_dir) if assets_dir is not None else Path(__file__).with_name("confusion_explorer_assets")
    template_path, d3_path = assets / "template.html", assets / "d3.v7.9.0.min.js"
    inputs = [Path(matches_path), Path(summary_path), Path(taxonomy_path), Path(args_path), template_path, d3_path]
    _protect_inputs(output, inputs)
    payload = load_data(matches_path, summary_path, taxonomy_path, args_path, allow_compatible_taxonomy=allow_compatible_taxonomy)
    for path in (template_path, d3_path):
        _require(path.is_file(), f"Missing explorer asset: {path}")
    template = template_path.read_text(encoding="utf-8-sig")
    for token in ("__D3_SOURCE__", "__CONFUSION_DATA__"):
        _require(template.count(token) == 1, f"Template must contain {token} exactly once")
    d3_source = d3_path.read_text(encoding="utf-8-sig")
    d3_source = re.sub(r"</script", lambda match: "<\\/" + match.group(0)[2:], d3_source, flags=re.IGNORECASE)
    replacements = {"__D3_SOURCE__": d3_source, "__CONFUSION_DATA__": _safe_json(payload)}
    html = re.sub(r"__D3_SOURCE__|__CONFUSION_DATA__", lambda match: replacements[match.group(0)], template)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", prefix=".confusion-explorer-", suffix=".tmp", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(html)
            stream.flush()
            os.fsync(stream.fileno())
        _protect_inputs(output, inputs)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output.resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path, help="Directory containing hierarchy_matches.json and evaluation_summary.json")
    parser.add_argument("--taxonomy", required=True, type=Path, help="Taxonomy JSON used by the exported evaluation")
    parser.add_argument("--args", required=True, type=Path, help="Training args.yaml for the checkpoint run")
    parser.add_argument("--output", required=True, type=Path, help="Self-contained HTML output path")
    parser.add_argument("--allow-compatible-taxonomy", action="store_true", help="Allow a taxonomy SHA mismatch only after every strict metadata and metric check passes")
    options = parser.parse_args(argv)
    try:
        output = generate(options.evaluation / "hierarchy_matches.json", options.evaluation / "evaluation_summary.json", options.taxonomy, options.args, options.output, allow_compatible_taxonomy=options.allow_compatible_taxonomy)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
