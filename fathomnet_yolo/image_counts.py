"""Count distinct image support from a finalized Megalodon YOLO export.

Only the three final split lists, their label files, and the final class mapping
are read. Image files are checked for presence; their contents are never opened.
"""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath
from uuid import UUID

SPLITS = ("train", "val", "test")


def _source(role, name, contents):
    return {
        "role": role,
        "name": name,
        "sha256": hashlib.sha256(contents).hexdigest(),
    }


def _read_bytes(path, label):
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"Cannot read {label}: {error.strerror or error}") from error


def _decode(contents, label):
    try:
        return contents.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8 text") from error


def _read_json_bytes(contents, label):
    try:
        return json.loads(_decode(contents, label))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {label}: {error.msg}") from error


def _class_mapping(records, label):
    if not isinstance(records, list):
        raise TypeError(f"{label} must be a list of class records")
    mapping = {}
    names = set()
    categories = set()
    for record in records:
        if not isinstance(record, dict):
            raise TypeError(f"{label} contains a non-object class record")
        class_id = record.get("yolo_id")
        category_id = record.get("category_id")
        name = record.get("name")
        if type(class_id) is not int or class_id < 0 or class_id in mapping:
            raise ValueError(f"{label} has an invalid or repeated yolo_id")
        if type(category_id) is not int or category_id in categories:
            raise ValueError(f"{label} has an invalid or repeated category_id")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError(f"{label} has an invalid or repeated class name")
        mapping[class_id] = (category_id, name)
        categories.add(category_id)
        names.add(name)
    if not mapping or set(mapping) != set(range(len(mapping))):
        raise ValueError(f"{label} requires consecutive yolo_id values from zero")
    return mapping


def _nonnegative_int(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _parse_labels(contents, label, class_ids):
    records = []
    for line_number, line in enumerate(_decode(contents, label).splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(
                f"{label}:{line_number}: expected five YOLO detection fields"
            )
        try:
            class_id = int(fields[0])
            values = tuple(float(value) for value in fields[1:])
        except ValueError as error:
            raise ValueError(
                f"{label}:{line_number}: malformed YOLO detection"
            ) from error
        if class_id not in class_ids:
            raise ValueError(f"{label}:{line_number}: unknown class ID {class_id}")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError(
                f"{label}:{line_number}: coordinates must be finite and normalized"
            )
        x, y, width, height = values
        if width <= 0 or height <= 0:
            raise ValueError(
                f"{label}:{line_number}: box width and height must be positive"
            )
        # The exporter writes twelve significant digits, so accept roundoff at
        # image borders while rejecting boxes that substantially cross them.
        tolerance = 1e-6
        if (
            x - width / 2 < -tolerance
            or y - height / 2 < -tolerance
            or x + width / 2 > 1 + tolerance
            or y + height / 2 > 1 + tolerance
        ):
            raise ValueError(f"{label}:{line_number}: box extends beyond the image")
        records.append((class_id, *(0.0 if value == 0 else value for value in values)))
    if not records:
        raise ValueError(
            f"{label}: a finalized annotated image requires at least one box"
        )
    # Sorting retains duplicate boxes but ignores annotation line order. Floats
    # are parsed before hashing so equivalent textual forms have equal meaning.
    signature = hashlib.sha256(
        json.dumps(sorted(records), separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()
    return records, signature


def _protect_sources(output_path, paths):
    if output_path is None:
        return
    destination = Path(output_path).expanduser().resolve()
    for source in paths:
        resolved = source.resolve()
        same_file = destination == resolved
        if not same_file and destination.exists() and resolved.exists():
            try:
                same_file = destination.samefile(resolved)
            except OSError:
                pass
        if same_file:
            raise ValueError("The HTML output must not overwrite an input source file")


def attach_export_image_counts(
    data, dataset_dir, class_map=None, export_summary=None, output_path=None
):
    """Attach exact distinct-image counts and return audit sources and notes.

    ``data`` is the shared explorer tree with ``originals`` records containing
    ``yolo_class_id``, ``coco_id``, and ``original_name``. Validation finishes
    before any values in ``data`` are changed. The final export's class_map.json
    is always checked, even when an external class mapping was also supplied.

    Split-list entries may contain the original server's absolute paths: only
    their basenames are used, under images/{split} in ``dataset_dir``. An image
    UUID may occur in multiple splits only if its complete annotation multiset
    agrees. All-split counts deduplicate such UUIDs and report their overlap.
    """
    directory = Path(dataset_dir).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError("dataset_dir must be a finalized YOLO export directory")
    mapping_path = directory / "class_map.json"
    mapping_bytes = _read_bytes(mapping_path, "dataset class_map.json")
    mapping = _class_mapping(
        _read_json_bytes(mapping_bytes, "dataset class_map.json"), "Dataset class map"
    )
    if (
        class_map is not None
        and _class_mapping(class_map, "Provided class map") != mapping
    ):
        raise ValueError("Dataset class map disagrees with the provided class map")

    sources = [_source("Final YOLO class mapping", "class_map.json", mapping_bytes)]
    input_paths = [mapping_path]
    summary_path = directory / "export_summary.json"
    if summary_path.exists():
        input_paths.append(summary_path)
    if export_summary is None:
        summary_bytes = _read_bytes(summary_path, "dataset export_summary.json")
        export_summary = _read_json_bytes(summary_bytes, "dataset export_summary.json")
        sources.append(
            _source("Final export summary", "export_summary.json", summary_bytes)
        )
    if not isinstance(export_summary, dict) or export_summary.get("status") not in {
        "ready",
        "ready_with_skipped_images",
    }:
        raise ValueError("Image counts require a completed export summary")
    if _nonnegative_int(export_summary.get("classes"), "Export summary classes") != len(
        mapping
    ):
        raise ValueError("Export summary class count disagrees with the final mapping")

    tree = data.get("tree") if isinstance(data, dict) else None
    if not isinstance(tree, dict) or not isinstance(data.get("summary"), dict):
        raise TypeError("Expected a prepared explorer tree and summary")
    node_list = []
    seen_nodes = set()
    tree_mapping = {}

    def inspect(node):
        node_id = node.get("id")
        if not isinstance(node_id, str) or node_id in seen_nodes:
            raise ValueError("Tree node IDs must be unique strings")
        seen_nodes.add(node_id)
        node_list.append(node)
        originals = node.get("originals", [])
        children = node.get("children", [])
        if not isinstance(originals, list) or not isinstance(children, list):
            raise TypeError("Tree originals and children must be lists")
        for record in originals:
            if not isinstance(record, dict):
                raise TypeError("Tree original classes must be objects")
            class_id = record.get("yolo_class_id")
            if type(class_id) is not int or class_id in tree_mapping:
                raise ValueError("Tree original class IDs must be unique integers")
            tree_mapping[class_id] = (
                record.get("coco_id"),
                record.get("original_name"),
            )
        for child in children:
            if not isinstance(child, dict):
                raise TypeError("Tree children must be objects")
            inspect(child)

    inspect(tree)
    if tree_mapping != mapping:
        raise ValueError("Dataset class map disagrees with the prepared taxonomy")

    class_images = {
        class_id: {split: set() for split in SPLITS} for class_id in mapping
    }
    selected_images = {split: set() for split in SPLITS}
    signatures = {}
    class_ids = set(mapping)
    split_stats = export_summary.get("splits")
    if not isinstance(split_stats, dict):
        raise TypeError("Export summary requires final split statistics")
    for split in SPLITS:
        stats = split_stats.get(split)
        if not isinstance(stats, dict):
            raise TypeError(f"Export summary is missing the {split} split")
        expected_images = _nonnegative_int(
            stats.get("exported_images"), f"{split} exported_images"
        )
        expected_boxes = _nonnegative_int(
            stats.get("exported_boxes"), f"{split} exported_boxes"
        )
        expected_counts = stats.get("exported_class_counts")
        if not isinstance(expected_counts, dict) or set(expected_counts) - {
            name for _, name in mapping.values()
        }:
            raise ValueError(f"{split} requires valid final exported_class_counts")
        for name, count in expected_counts.items():
            _nonnegative_int(count, f"{split} exported count for {name}")
        if sum(expected_counts.values()) != expected_boxes:
            raise ValueError(
                f"{split} exported box total disagrees with its class counts"
            )
        list_name = f"splits/{split}.txt"
        list_path = directory / list_name
        input_paths.append(list_path)
        list_bytes = _read_bytes(list_path, list_name)
        sources.append(_source(f"Final {split} image selection", list_name, list_bytes))
        entries = []
        for line_number, line in enumerate(
            _decode(list_bytes, list_name).splitlines(), 1
        ):
            if not line.strip():
                continue
            basename = PurePosixPath(line.strip().replace("\\", "/")).name
            identifier = Path(basename).stem
            try:
                canonical = str(UUID(identifier))
            except ValueError as error:
                raise ValueError(
                    f"{list_name}:{line_number}: expected an image UUID filename"
                ) from error
            if identifier != canonical:
                raise ValueError(
                    f"{list_name}:{line_number}: image UUID must use its canonical spelling"
                )
            if identifier in selected_images[split]:
                raise ValueError(
                    f"{list_name}:{line_number}: repeated image UUID in one split"
                )
            image_path = directory / "images" / split / basename
            if not image_path.is_file():
                raise ValueError(f"Missing selected image images/{split}/{basename}")
            selected_images[split].add(identifier)
            entries.append((identifier, image_path))
        if len(entries) != expected_images:
            raise ValueError(
                f"{split} split-list image count disagrees with export_summary"
            )

        actual_counts = Counter()
        label_digest = hashlib.sha256(b"fathomnet-yolo-label-stream-v1\0")
        for identifier, image_path in sorted(entries):
            label_name = f"labels/{split}/{identifier}.txt"
            label_path = directory / label_name
            input_paths.extend((image_path, label_path))
            label_bytes = _read_bytes(label_path, label_name)
            records, signature = _parse_labels(label_bytes, label_name, class_ids)
            previous = signatures.get(identifier)
            if previous is not None and previous != signature:
                raise ValueError(
                    "The same image UUID has inconsistent annotations across splits"
                )
            signatures[identifier] = signature
            actual_counts.update(mapping[record[0]][1] for record in records)
            for class_id in {record[0] for record in records}:
                class_images[class_id][split].add(identifier)
            encoded_name = label_name.encode("utf-8")
            label_digest.update(len(encoded_name).to_bytes(8, "big"))
            label_digest.update(encoded_name)
            label_digest.update(len(label_bytes).to_bytes(8, "big"))
            label_digest.update(label_bytes)
        if sum(actual_counts.values()) != expected_boxes or actual_counts != Counter(
            expected_counts
        ):
            raise ValueError(
                f"{split} label box counts disagree with exported_class_counts"
            )
        sources.append(
            {
                "role": f"Final {split} labels; framed name and byte stream, sorted by UUID",
                "name": f"labels/{split}/ (selected files only)",
                "sha256": label_digest.hexdigest(),
            }
        )

    _protect_sources(output_path, input_paths)
    # Keep all computed state separate so malformed input never partially changes
    # the prepared explorer data, including failures in a later split.
    attachments = {}
    original_counts = []

    def collect(node):
        direct = {split: set() for split in SPLITS}
        for record in node.get("originals", []):
            support = class_images[record["yolo_class_id"]]
            for split in SPLITS:
                direct[split].update(support[split])
            original_counts.append(
                (
                    record,
                    {
                        "images": len(set().union(*support.values())),
                        "images_by_split": {
                            split: len(support[split]) for split in SPLITS
                        },
                    },
                )
            )
        subtree = {split: direct[split].copy() for split in SPLITS}
        for child in node.get("children", []):
            child_support = collect(child)
            for split in SPLITS:
                subtree[split].update(child_support[split])
        attachments[node["id"]] = {
            "direct_images": len(set().union(*direct.values())),
            "subtree_images": len(set().union(*subtree.values())),
            "direct_images_by_split": {split: len(direct[split]) for split in SPLITS},
            "subtree_images_by_split": {split: len(subtree[split]) for split in SPLITS},
        }
        return subtree

    root_support = collect(tree)
    if any(root_support[split] != selected_images[split] for split in SPLITS):
        raise ValueError(
            "Taxonomy image support disagrees with the final selected images"
        )
    all_images = set().union(*selected_images.values())
    overlapping = sum(map(len, selected_images.values())) - len(all_images)
    for node in node_list:
        node.update(attachments[node["id"]])
    for record, counts in original_counts:
        record.update(counts)
    data["summary"].update(
        images=len(all_images),
        annotated_images=len(all_images),
        images_by_split={split: len(selected_images[split]) for split in SPLITS},
    )
    notes = [
        (
            "Distinct image counts use only the finalized YOLO split lists and their final labels; "
            "image files are checked for presence without reading their pixels."
        ),
        (
            "An image contributes once per directly annotated class. Subtree image counts are "
            "unions across that taxon and its descendants, so child image counts are not additive."
        ),
        (
            "All-split image counts deduplicate image UUIDs across splits. Label audit digests "
            "hash a versioned stream of sorted selected label names and their exact bytes, "
            "with each name and content prefixed by its eight-byte big-endian length."
        ),
    ]
    if overlapping:
        notes.append(
            f"The final split lists contain {overlapping} repeated cross-split image memberships. "
            "Their annotation multisets agree; All counts each image UUID once. "
            "Cross-split image reuse should be reviewed when interpreting evaluation results."
        )
    return {"sources": sources, "notes": notes}
