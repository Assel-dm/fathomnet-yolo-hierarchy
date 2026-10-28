"""Build an offline YOLO/FathomNet report from saved evaluations.

Adapted from the save_figure / draw_tradeoff / build_html / build_reports pattern
in Assel-dm/hierarchical-classification/common/visualization.py (local reference).
The report only reads completed exports; it neither trains nor evaluates a model.

Copy this file to fathomnet_yolo/visualization.py, alongside taxonomy.py, and run::

    python -m fathomnet_yolo.visualization --runs-dir runs \
        --taxonomy metadata/taxonomy_fathomnet2025.json \
        --sweep alpha --fixed-lambda 1 --output reports/alpha_lambda1

Dependencies: numpy, matplotlib, PyYAML, and the existing Taxonomy helper.
"""
import argparse
import base64
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from html import escape
from pathlib import Path, PurePosixPath

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

try:
    from .taxonomy import Taxonomy
except ImportError:
    from fathomnet_yolo.taxonomy import Taxonomy

METRICS = {
    "map95": "Native mAP50–95", "map50": "Native mAP50",
    "precision": "Native precision", "recall": "Native recall",
    "coverage": "Geometric coverage", "exact_coverage": "Coverage with exact labels",
    "accuracy": "Accuracy on matched objects", "top1_error": "Top-1 error on matched objects",
    "severity": "Mistake severity", "ahd1": "AHD@1", "ahd5": "AHD@5", "ahd20": "AHD@20",
    "macro_accuracy": "Mean accuracy per represented class", "macro_ahd1": "Mean AHD@1 per represented class",
    "matched": "Matched objects", "correct": "Exact labels", "gt": "Annotations",
    "missed": "Missed annotations", "unmatched": "Unmatched predictions",
    "native_delta_map95": "Δ mAP50–95 vs native, same seed",
    "native_delta_ahd1": "Δ AHD@1 vs native, common objects",
    "control_delta_ahd1": "Δ AHD@1 vs α=0, common objects",
}
RATES = {"map95", "map50", "precision", "recall", "coverage", "exact_coverage", "accuracy", "macro_accuracy", "top1_error", "native_delta_map95"}
COLORS = ("#007d79", "#cc6b18", "#6554a4", "#c83f5d", "#2369aa", "#596b38")


def finite(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def read_csv(path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for record in csv.DictReader(stream):
            out = {}
            for key, value in record.items():
                try:
                    out[key.strip()] = float(value)
                except (ValueError, TypeError):
                    out[key.strip()] = value
            rows.append(out)
    return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row if not k.startswith("_")))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def run_label(row, include_seed=True):
    if row["method"] == "native":
        label = "Native YOLO"
    elif row["alpha"] == 0:
        label = f"Auxiliary α=0 · λ={row['lambda']:g}"
    else:
        label = f"HXE α={row['alpha']:g} · λ={row['lambda']:g}"
    return label + (f" · seed {row['seed']}" if include_seed else "")


def group_key(row):
    return row["protocol"], row["model"], row["method"], row["alpha"], row["lambda"], row["split"]


def aggregate_seeds(records):
    """One independent value per seed; sample SD is undefined for one seed."""
    grouped = defaultdict(list)
    for row in records:
        grouped[group_key(row)].append(row)
    result = []
    for key, rows in grouped.items():
        seeds = [row["seed"] for row in rows]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Multiple checkpoints for the same configuration and seed: {run_label(rows[0], False)}. Select one run per seed.")
        first = rows[0]
        out = {k: first[k] for k in ("protocol", "model", "method", "alpha", "lambda", "split")}
        out.update(label=run_label(first, False), n_seeds=len(seeds), seeds=", ".join(map(str, sorted(seeds))))
        for metric in METRICS:
            values = [r[metric] for r in rows if finite(r.get(metric))]
            out[f"{metric}_n"] = len(values)
            out[f"{metric}_mean"] = statistics.mean(values) if values else None
            out[f"{metric}_std"] = statistics.stdev(values) if len(values) >= 2 else None
        result.append(out)
    return sorted(result, key=lambda r: (r["protocol"], r["lambda"], -1 if r["alpha"] is None else r["alpha"]))


def load_records(args):
    root = args.runs_dir.expanduser().resolve()
    config_paths = list((root / "detect").rglob("args.yaml"))
    configs_by_name = defaultdict(list)
    for path in config_paths:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(cfg, dict):
            configs_by_name[path.parent.name].append((path, cfg))
    records, duplicates, seen = [], [], {}
    for path in sorted((root / "fathomnet-hierarchy").rglob("evaluation_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        train_name = PurePosixPath(summary["checkpoint"]["path"].replace("\\", "/")).parent.parent.name
        candidates = configs_by_name.get(train_name, [])
        if len(candidates) != 1:
            raise ValueError(f"Missing or ambiguous args.yaml for {train_name} ({path})")
        config_path, cfg = candidates[0]
        weight = float(cfg.get("hxe_weight", 0))
        alpha = float(cfg.get("hxe_alpha", 0)) if weight else None
        if weight:
            if args.fixed_lambda is not None and not math.isclose(weight, args.fixed_lambda):
                continue
            if args.fixed_alpha is not None and not math.isclose(alpha, args.fixed_alpha):
                continue
        report = summary["hierarchy"]
        mode = args.distance_mode
        if report["taxonomy"]["distance_mode"] == mode:
            scores = report["metrics"]
        elif report.get("alternative_distance", {}).get("mode") == mode:
            scores = report["alternative_distance"]["metrics"]
        else:
            raise ValueError(f"Distance mode {mode} is unavailable in {path}")
        tax = Taxonomy.from_json(args.taxonomy, report["taxonomy"]["class_names"], distance_mode=mode)
        if list(tax.class_node_ids) != report["taxonomy"]["class_node_ids"]:
            raise ValueError(f"Taxonomy incompatible with {path}")
        ignored = {"name", "save_dir", "project", "seed", "hxe_alpha", "hxe_weight", "hxe_taxonomy", "data", "cfg"}
        inference_keys = ("batch", "imgsz", "conf", "iou", "max_det", "rect", "single_cls", "agnostic_nms", "augment")
        protocol = fingerprint({
            "data": summary["dataset_yaml"]["sha256"], "taxonomy": summary["taxonomy_file"]["sha256"],
            "split": summary["split"], "version": summary["ultralytics_version"],
            "distance": mode, "hierarchy": report["protocol"],
            "validation": {k: summary["validation_args"].get(k) for k in inference_keys},
            "training": {k: v for k, v in cfg.items() if k not in ignored},
        })
        identity = protocol, summary["checkpoint"]["sha256"]
        if identity in seen:
            if seen[identity] != int(cfg["seed"]):
                raise ValueError("The same checkpoint is assigned to multiple seeds; these are not independent repetitions.")
            duplicates.append(str(path))
            continue
        seen[identity] = int(cfg["seed"])
        matches_path = path.parent / "hierarchy_matches.json"
        matches = json.loads(matches_path.read_text(encoding="utf-8"))
        index = {}
        for m in matches:
            key = (Path(m["image"]).name, m["gt_index"])
            if key in index:
                raise ValueError(f"Duplicate annotation identity: {key}")
            target, pred = m["target_class"], m["predicted_class"]
            m = dict(m, correct=(target == pred), distance=float(tax.distance_matrix[target, pred]))
            index[key] = m
        n, gt = len(matches), int(scores["num_gt"])
        correct = sum(m["correct"] for m in index.values())
        distances = [m["distance"] for m in index.values()]
        class_groups = defaultdict(list)
        for m in index.values():
            class_groups[m["target_class"]].append(m)
        actual_ahd = statistics.mean(distances) if n else None
        if n != scores["num_matched"] or (n and not math.isclose(actual_ahd, scores["avg_hierarchical_distance@1"], abs_tol=1e-9)):
            raise ValueError(f"Matches and taxonomy do not reproduce the metrics in {path}")
        det = summary["native_detection_metrics"]
        row = {"protocol": protocol, "model": Path(str(cfg["model"])).name,
               "method": "native" if not weight else ("auxiliary" if alpha == 0 else "hxe"),
               "alpha": alpha, "lambda": weight, "seed": int(cfg["seed"]), "split": summary["split"],
               "run": train_name, "checkpoint_sha256": summary["checkpoint"]["sha256"],
               "map95": det["metrics/mAP50-95(B)"], "map50": det["metrics/mAP50(B)"],
               "precision": det["metrics/precision(B)"], "recall": det["metrics/recall(B)"],
               "coverage": n / gt if gt else None, "exact_coverage": correct / gt if gt else None,
               "accuracy": scores["top1_accuracy"], "top1_error": 1 - correct / n if n else None,
               "severity": scores["mistake_severity"], "ahd1": scores["avg_hierarchical_distance@1"],
               "ahd5": scores.get("avg_hierarchical_distance@5"), "ahd20": scores.get("avg_hierarchical_distance@20"),
               "macro_accuracy": statistics.mean(statistics.mean(m["correct"] for m in pop) for pop in class_groups.values()) if class_groups else None,
               "macro_ahd1": statistics.mean(statistics.mean(m["distance"] for m in pop) for pop in class_groups.values()) if class_groups else None,
               "matched": n, "correct": correct, "gt": gt, "missed": scores["num_missed"],
               "unmatched": scores["num_unmatched_predictions"],
               "_index": index, "_tax": tax, "_scores": scores,
               "_config": cfg, "_train_dir": config_path.parent, "_eval_dir": path.parent,
               "_history": read_csv(config_path.parent / "results.csv"), "_summary": summary}
        records.append(row)
    if not records:
        raise ValueError("No compatible evaluations found: check runs-dir/detect and runs-dir/fathomnet-hierarchy.")
    return records, duplicates


def paired_objects(reference, row):
    keys = reference["_index"].keys() & row["_index"].keys()
    if not keys:
        return {"objects": 0, "delta_accuracy": None, "delta_ahd1": None}
    left = [reference["_index"][k] for k in keys]
    right = [row["_index"][k] for k in keys]
    if any(l["target_class"] != r["target_class"] for l, r in zip(left, right)):
        raise ValueError("Different target classes for the same annotation.")
    return {"objects": len(keys),
            "delta_accuracy": statistics.mean(r["correct"] - l["correct"] for l, r in zip(left, right)),
            "delta_ahd1": statistics.mean(r["distance"] - l["distance"] for l, r in zip(left, right))}


def add_references(records):
    natives, controls = {}, {}
    for row in records:
        if row["method"] == "native":
            key, target = (row["protocol"], row["seed"]), natives
        elif row["method"] == "auxiliary":
            key, target = (row["protocol"], row["seed"], row["lambda"]), controls
        else:
            continue
        if key in target:
            raise ValueError("Ambiguous reference: multiple runs for the same protocol and seed.")
        target[key] = row
    for row in records:
        native = natives.get((row["protocol"], row["seed"]))
        control = controls.get((row["protocol"], row["seed"], row["lambda"]))
        row["native_delta_map95"] = row["map95"] - native["map95"] if native else None
        for label, reference in (("native", native), ("control", control)):
            pair = paired_objects(reference, row) if reference else {"objects": None, "delta_accuracy": None, "delta_ahd1": None}
            row[f"{label}_common_objects"] = pair["objects"]
            row[f"{label}_delta_ahd1"] = pair["delta_ahd1"]
            row[f"{label}_delta_accuracy"] = pair["delta_accuracy"]


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)


def place_labels(ax, points):
    """Try nearby offsets and avoid text collisions without an extra dependency."""
    from matplotlib.transforms import Bbox, offset_copy

    renderer = ax.figure.canvas.get_renderer()
    bounds = ax.get_window_extent(renderer).padded(-3)
    marks = [Bbox.from_bounds(*((ax.transData.transform((x, y)) - 5).tolist()), 10, 10) for x, y, text in points]
    occupied = []
    offsets = ((6, 7), (6, -14), (-6, 7), (-6, -14), (8, 21), (-8, 21), (8, -26), (-8, -26), (15, 0), (-15, 0), (8, 34), (-8, 34))
    for x, y, text in points:
        chosen = None
        for dx, dy in offsets:
            transform = offset_copy(ax.transData, fig=ax.figure, x=dx, y=dy, units="points")
            label = ax.text(x, y, text, transform=transform, ha="left" if dx > 0 else "right", fontsize=8)
            box = label.get_window_extent(renderer).expanded(1.03, 1.16)
            if (bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1)
                    and not any(box.overlaps(other) for other in occupied + marks)):
                occupied.append(box)
                chosen = label
                break
            label.remove()
        if chosen is None:
            label = ax.annotate(text, (x, y), xytext=(6, 7), textcoords="offset points", fontsize=8)
            occupied.append(label.get_window_extent(renderer))


def draw_tradeoffs(summary, output, sweep):
    result = []
    for protocol in sorted({r["protocol"] for r in summary}):
        rows = [r for r in summary if r["protocol"] == protocol]
        fig, axes = plt.subplots(3, 2, figsize=(12.4, 12))
        specs = (("top1_error", "severity"), ("top1_error", "ahd1"), ("top1_error", "ahd5"), ("top1_error", "ahd20"), ("map95", "ahd1"), ("exact_coverage", "ahd1"))
        label_sets = []
        fixed_lambda = len({r["lambda"] for r in rows if r["method"] != "native"}) == 1
        fixed_alpha = len({r["alpha"] for r in rows if r["method"] != "native"}) == 1
        for ax, (x, y) in zip(axes.flat, specs):
            points = []
            for i, row in enumerate(rows):
                xx, yy = row.get(f"{x}_mean"), row.get(f"{y}_mean")
                if not finite(xx) or not finite(yy):
                    continue
                xf = 100 if x in RATES else 1
                dx, dy = row.get(f"{x}_std"), row.get(f"{y}_std")
                native = row["method"] == "native"
                color = "#303c47" if native else COLORS[i % len(COLORS)]
                ax.errorbar(xx * xf, yy, xerr=dx * xf if finite(dx) else None, yerr=dy if finite(dy) else None,
                            marker="*" if native else "o", markersize=10 if native else 6,
                            color=color, capsize=3, linewidth=1, linestyle="none")
                annotation = "Native" if native else (f"α={row['alpha']:g}" if fixed_lambda else f"λ={row['lambda']:g}" if fixed_alpha else f"α={row['alpha']:g} λ={row['lambda']:g}")
                points.append((xx * xf, yy, annotation))
            ax.set(xlabel=METRICS[x] + (" (%)" if x in RATES else ""), ylabel=METRICS[y])
            ax.grid(alpha=.2)
            ax.margins(x=.22, y=.24)
            label_sets.append((ax, points))
        parameter = "α" if sweep == "alpha" else "λ"
        fig.suptitle(f"{parameter} sweep · protocol {protocol}\nPoints = means across seeds; error bars = standard deviations when n ≥ 2", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .95))
        fig.canvas.draw()
        for ax, points in label_sets:
            place_labels(ax, points)
        path = output / "figures" / f"tradeoffs_{protocol}.png"
        save_figure(fig, path)
        result.append(path)
    return result


def plot_learning_curves(row, output):
    history = [r for r in row["_history"] if finite(r.get("epoch"))]
    if not history:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7))
    for ax, component in zip(axes.flat, ("box", "cls", "dfl")):
        for split, label in (("train", "Training"), ("val", "Internal validation")):
            key = f"{split}/{component}_loss"
            selected = [r for r in history if finite(r.get(key))]
            if selected:
                ax.plot([r["epoch"] for r in selected], [r[key] for r in selected], label=label)
        ax.set(xlabel="Epoch", ylabel={"box": "Box loss", "cls": "Classification loss", "dfl": "DFL loss"}[component])
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    ax = axes.flat[3]
    for key, label in (("metrics/mAP50-95(B)", "mAP50–95"), ("metrics/mAP50(B)", "mAP50")):
        selected = [r for r in history if finite(r.get(key))]
        ax.plot([r["epoch"] for r in selected], [100 * r[key] for r in selected], label=label)
    ax.set(xlabel="Epoch", ylabel="Internal mAP (%)")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle(run_label(row))
    fig.tight_layout()
    path = output / "figures" / f"learning_{row['_slug']}.png"
    save_figure(fig, path)
    return path


def per_class_data(row):
    tax, index = row["_tax"], row["_index"]
    records = []
    for c, name in enumerate(tax.class_names):
        pop = [m for m in index.values() if m["target_class"] == c]
        mistakes = [m for m in pop if not m["correct"]]
        records.append({"class_id": c, "class": name, "matched_support": len(pop),
                        "correct": sum(m["correct"] for m in pop),
                        "accuracy": statistics.mean(m["correct"] for m in pop) if pop else None,
                        "ahd1": statistics.mean(m["distance"] for m in pop) if pop else None,
                        "severity": statistics.mean(m["distance"] for m in mistakes) if mistakes else None})
    return records


def plot_per_class(row, classes, output):
    selected = sorted((c for c in classes if c["matched_support"]), key=lambda r: (r["accuracy"], -r["matched_support"]))[:20]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, max(4.5, len(selected) * .23)))
    if selected:
        axes[0].barh(range(len(selected)), [100 * c["accuracy"] for c in selected], color="#007d79")
        axes[0].set_yticks(range(len(selected)), [f"{c['class']} (n={c['matched_support']})" for c in selected], fontsize=7)
        axes[0].invert_yaxis()
    axes[0].set(xlabel="Accuracy on matched objects (%)", xlim=(0, 100), title="20 lowest conditional class accuracies")
    counts = Counter(m["distance"] for m in row["_index"].values() if not m["correct"])
    if counts:
        axes[1].bar(sorted(counts), [counts[d] for d in sorted(counts)], color="#cc6b18")
    else:
        axes[1].text(.5, .5, "No errors among matched objects", ha="center", transform=axes[1].transAxes)
    axes[1].set(xlabel="Taxonomic distance", ylabel="Classification errors on matched objects", title="Error distance distribution")
    fig.suptitle(run_label(row)); fig.tight_layout()
    path = output / "figures" / f"classes_{row['_slug']}.png"
    save_figure(fig, path)
    return path


def formatted(value, rate=False):
    if value is None or (isinstance(value, (int, float)) and not finite(value)):
        return "—"
    if isinstance(value, (int, float)):
        return f"{100 * value:.2f} %" if rate else f"{value:.4g}"
    return str(value)


def html_table(rows, columns):
    def cell(row, key):
        value = row.get(key)
        if key in {"native_delta_map95", "native_delta_accuracy", "control_delta_accuracy"}:
            return f"{100 * value:+.2f} pp" if finite(value) else "—"
        return formatted(value, key in RATES)

    header = "".join(f"<th>{escape(label)}</th>" for key, label in columns)
    body = "".join("<tr>" + "".join(f"<td>{escape(cell(row, key))}</td>" for key, label in columns) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


def picture(path, caption):
    if path is None:
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<figure><figcaption>{escape(caption)}</figcaption><img loading="lazy" alt="{escape(caption)}" src="data:image/png;base64,{data}"></figure>'


def build_html(records, summary, figures, output, args, duplicates):
    parts = ["<h1>YOLO / FathomNet: detection and hierarchy</h1>",
             '<nav><a href="#comparison">Comparisons</a> · <a href="#seeds">Across seeds</a> · <a href="#runs">Run details</a></nav>',
             f"<p>Distinct checkpoints: {len(records)} · Seeds observed: {len(set(r['seed'] for r in records))} · Distance mode: {escape(args.distance_mode)}.</p>",
             '<p class="note">Hierarchical distances and accuracies apply only to matched objects. Read them alongside coverage, missed annotations and unmatched predictions. Native YOLO (λ=0) is distinct from the auxiliary control α=0, λ&gt;0.</p>',
             '<h2 id="comparison">Metrics by run</h2>',
             '<p>mAP comes from a separate native evaluation of the selected checkpoint. Learning curves use internal validation during training; these two evaluations are kept separate.</p>']
    identity_columns = [("label", "Run")]
    if len({r["protocol"] for r in records}) > 1:
        identity_columns.append(("protocol", "Protocol"))
    labeled = [dict(r, label=run_label(r)) for r in records]
    parts.append("<h3>Detection and coverage</h3>")
    parts.append(html_table(labeled, identity_columns + [
        ("map95", METRICS["map95"]), ("map50", "mAP50"), ("precision", "Native precision"), ("recall", "Native recall"),
        ("coverage", "Geometric coverage"), ("exact_coverage", "Coverage with exact labels"), ("matched", "Matched"), ("missed", "Missed"), ("unmatched", "Unmatched predictions")]))
    parts.append("<h3>Hierarchy on matched objects</h3>")
    parts.append(html_table(labeled, identity_columns + [
        ("accuracy", "Accuracy on matched objects"), ("severity", "Severity"), ("ahd1", "AHD@1"), ("ahd5", "AHD@5"), ("ahd20", "AHD@20"),
        ("macro_accuracy", "Mean accuracy per class"), ("macro_ahd1", "Mean AHD@1 per class"), ("matched", "Matched")]))
    parts.append("<h3>Comparisons on common annotations within each seed</h3>")
    parts.append(html_table([dict(r, label=run_label(r)) for r in records], [
        ("label", "Run"), ("native_common_objects", "Common objects vs native"),
        ("native_delta_map95", "Δ mAP vs native"), ("native_delta_accuracy", "Δ accuracy vs native"),
        ("native_delta_ahd1", "Δ AHD@1 vs native"), ("control_common_objects", "Common objects vs α=0"),
        ("control_delta_accuracy", "Δ accuracy vs α=0"), ("control_delta_ahd1", "Δ AHD@1 vs α=0")]))
    parts.extend(picture(path, "Means and standard deviations across seeds within the same protocol") for path in figures)
    parts.append('<p>Lower distances and top-1 error are preferable. Higher mAP and coverage are preferable. Class averages include only classes with at least one matched object.</p>')
    parts += ['<h2 id="seeds">Means and standard deviations across seeds</h2>',
              '<p>Each seed contributes once. Sample standard deviation uses n−1; for a single seed it is shown as “—”, never zero. Comparisons against native YOLO or α=0 use the same seed and common annotations. The valid seed count can be smaller than the total when a metric is undefined.</p>']
    compact_metrics = ("map95", "exact_coverage", "accuracy", "severity", "ahd1", "ahd5", "ahd20")
    compact_rows = []
    for group in summary:
        out = {k: group[k] for k in ("label", "protocol", "n_seeds", "seeds")}
        for metric in compact_metrics:
            avg, sd = group[f"{metric}_mean"], group[f"{metric}_std"]
            factor = 100 if metric in RATES else 1
            avg_label = f"{factor * avg:.3f}" if finite(avg) else "—"
            std_label = f"{factor * sd:.3f}" if finite(sd) else "—"
            suffix = f" (n={group[f'{metric}_n']})" if group[f"{metric}_n"] != group["n_seeds"] else ""
            out[f"summary_{metric}"] = f"{avg_label} ± {std_label}{suffix}"
        compact_rows.append(out)
    compact_columns = [("label", "Configuration"), ("seeds", "Seeds"), ("n_seeds", "n seeds")]
    if len({r["protocol"] for r in summary}) > 1:
        compact_columns.append(("protocol", "Protocol"))
    compact_columns += [(f"summary_{m}", METRICS[m] + (" (%)" if m in RATES else "")) for m in compact_metrics]
    parts.append(html_table(compact_rows, compact_columns))
    parts.append("<details><summary>Complete statistics and differences from reference models</summary>")
    stats_rows = []
    selected_metrics = ("map95", "exact_coverage", "accuracy", "macro_accuracy", "severity", "ahd1", "ahd5", "ahd20", "macro_ahd1", "native_delta_map95", "native_delta_ahd1", "control_delta_ahd1")
    for group in summary:
        for metric in selected_metrics:
            avg, sd = group[f"{metric}_mean"], group[f"{metric}_std"]
            mean_label = f"{100 * avg:+.2f} pp" if metric == "native_delta_map95" and finite(avg) else formatted(avg, metric in RATES)
            std_label = f"{100 * sd:.2f} pp" if metric == "native_delta_map95" and finite(sd) else formatted(sd, metric in RATES)
            stats_rows.append({"label": group["label"], "protocol": group["protocol"], "seeds": group["seeds"], "n_seeds": group["n_seeds"],
                               "metric": METRICS[metric] + (" (percentage points)" if metric.startswith("native_delta_map") else ""),
                               "mean": mean_label, "std": std_label, "n_metric": group[f"{metric}_n"]})
    parts.append(html_table(stats_rows, [("label", "Configuration"), ("protocol", "Protocol"), ("seeds", "Seeds"), ("n_seeds", "n seeds"), ("metric", "Metric"), ("mean", "Mean"), ("std", "Standard deviation"), ("n_metric", "Valid seeds")]))
    parts.append("</details>")
    parts += ['<h2 id="runs">Learning curves, classes and confusions</h2>',
              '<p>Classes may represent different taxonomic ranks. Supports below count matched objects, not images. The current exports do not provide enough information to compute detection F1 or AP per class here.</p>']
    for row in records:
        parts.append(f"<details><summary>{escape(run_label(row))}</summary>")
        parts.append(f"<p>Protocol {escape(row['protocol'])} · split {escape(row['split'])} · {row['gt']} annotations · {row['matched']} matched · {row['missed']} missed · {row['unmatched']} unmatched predictions.</p>")
        parts.append(picture(row["_learning_plot"], "Unsmoothed internal learning curves; loss scales depend on the objective"))
        parts.append(picture(row["_classes_plot"], "Conditional classification and error distances"))
        parts.append('<p>“—” denotes an undefined metric, including no matched objects, or no classification errors when computing severity.</p>')
        classes = sorted(row["_classes"], key=lambda r: (r["accuracy"] is None, r["accuracy"] or 0, -r["matched_support"]))
        parts.append(html_table(classes, [("class", "Target class"), ("matched_support", "Matched support"), ("correct", "Exact labels"), ("accuracy", "Accuracy"), ("ahd1", "AHD@1"), ("severity", "Severity")]))
        rank_rows = []
        for key, support in row["_scores"].items():
            if key.endswith("_support"):
                rank = key[:-8]
                accuracy = row["_scores"].get(f"{rank}_accuracy_top1_ancestor")
                rank_rows.append({"rank": rank, "eligible": support, "accuracy": accuracy})
        parts.append("<h3>Accuracy by rank on eligible objects</h3>")
        parts.append(html_table(rank_rows, [("rank", "Rank"), ("eligible", "Eligible support"), ("accuracy", "Accuracy of the top-1 ancestor")]))
        pairs = Counter((m["target_class"], m["predicted_class"]) for m in row["_index"].values() if not m["correct"])
        confusions = [{"target": row["_tax"].class_names[t], "prediction": row["_tax"].class_names[p], "count": count, "distance": float(row["_tax"].distance_matrix[t, p])} for (t, p), count in pairs.most_common(25)]
        parts.append("<h3>25 most frequent confusions on matched objects</h3>")
        parts.append(html_table(confusions, [("target", "Target"), ("prediction", "Prediction"), ("count", "Count"), ("distance", "Distance")]))
        parts.append("</details>")
    parts += ["<h2>Protocol notes</h2>",
              '<p>Incompatible protocols are neither aggregated nor plotted together. Their identifiers account for the data, taxonomy, split, validation settings and other training parameters. Statistics across seeds are not computed by pooling all annotations from every seed.</p>',
              '<p>AHD@k averages the distances of all labels in the top-k. Severity averages only exact-label classification errors. Increasing α also changes the scale of HXE in the unnormalised formulation.</p>',
              f"<p>Repeated evaluations of the same checkpoint ignored: {len(duplicates)}. Complete CSV exports accompany this report.</p>"]
    css = """body{font:15px/1.5 system-ui,sans-serif;color:#203340;background:#f5f7f9;margin:0}
    main{max-width:1250px;margin:auto;padding:24px}h1{overflow-wrap:anywhere}h2{margin-top:32px}
    nav,a{color:#007d79}details,figure,.table-wrap{background:white;border:1px solid #dce4e8;border-radius:8px}
    details{padding:16px;margin:12px 0}summary{cursor:pointer;font-weight:600}figure{margin:16px 0;padding:12px}
    figcaption{font-weight:600;margin-bottom:8px}img{display:block;width:100%;height:auto}
    .table-wrap{overflow:auto;max-height:510px}table{border-collapse:collapse;width:100%;font-size:13px}
    th,td{padding:8px;white-space:nowrap;border-bottom:1px solid #e2e8ec}th{position:sticky;top:0;background:#eef3f5}
    .note{padding:12px;background:#fff2d9}@media(max-width:600px){main{padding:12px}details{padding:10px}}
    """
    (output / "report.html").write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FathomNet: detection and hierarchy</title><style>' + css + "</style></head><body><main>" + "".join(parts) + "</main></body></html>", encoding="utf-8")


def build_reports(args):
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, duplicates = load_records(args)
    # Validate distinct seeds before any aggregation or reference selection.
    aggregate_seeds(records)
    add_references(records)
    summary = aggregate_seeds(records)
    class_rows = []
    for i, row in enumerate(records):
        row["_slug"] = f"{i:03d}_{row['checkpoint_sha256'][:12]}"
        row["_classes"] = per_class_data(row)
        row["_learning_plot"] = plot_learning_curves(row, output)
        row["_classes_plot"] = plot_per_class(row, row["_classes"], output)
        class_rows.extend(dict(c, run=row["run"], seed=row["seed"], alpha=row["alpha"], lambda_weight=row["lambda"], protocol=row["protocol"]) for c in row["_classes"])
    figures = draw_tradeoffs(summary, output, args.sweep)
    write_csv(output / "comparison.csv", records)
    write_csv(output / "seed_summary.csv", summary)
    write_csv(output / "per_class_comparison.csv", class_rows)
    manifest = {"runs": len(records), "groups": len(summary), "distance_mode": args.distance_mode,
                "taxonomy_used_sha256": hashlib.sha256(args.taxonomy.read_bytes()).hexdigest(),
                "reported_taxonomy_sha256": sorted({r["_summary"]["taxonomy_file"]["sha256"] for r in records}),
                "ignored_duplicate_evaluations": duplicates,
                "protocols": sorted({r["protocol"] for r in records}),
                "evaluations": [str(r["_eval_dir"]) for r in records]}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    build_html(records, summary, figures, output, args, duplicates)
    print(f"Report: {output / 'report.html'}")
    return records, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/fathomnet"))
    parser.add_argument("--sweep", choices=("alpha", "lambda"), default="alpha")
    parser.add_argument("--fixed-lambda", type=float)
    parser.add_argument("--fixed-alpha", type=float)
    parser.add_argument("--distance-mode", choices=("edge_count", "lca_height"), default="edge_count")
    args = parser.parse_args(argv)
    build_reports(args)


if __name__ == "__main__":
    main()
