"""Check images and anontations 
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlparse

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")
SPLITS = ("train", "val", "test")


class DatasetValidationError(ValueError):
    """An input failed validation; details are recorded in the output audit.json"""


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _positive_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _resolve_image(image, images_dir, naming):
    filename = Path(str(image.get("file_name", "")).replace("\\", "/"))
    if filename.is_absolute() or ".." in filename.parts:
        raise ValueError("file_name must be a relative path within images_dir")
    candidates = []
    # The official downloader accepts a parent directory and creates images/ and rois/
    # Only full-image locations are considered; cropped ROI files are never searched
    for directory in (images_dir, images_dir / "images"):
        if naming in ("auto", "file_name") and filename.name:
            candidates.extend((directory / filename, directory / filename.name))
        if naming in ("auto", "image_id"):
            candidates.extend(directory / (str(image["id"]) + suffix) for suffix in IMAGE_SUFFIXES)
    found = {candidate.resolve() for candidate in candidates if candidate.is_file()}
    if len(found) != 1:
        raise ValueError(
            "no matching full image" if not found else "ambiguous full image; select --image-naming explicitly"
        )
    result = found.pop()
    if result.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"unsupported image extension: {result.suffix}")
    return result


def _group_key(image, group_by):
    if group_by == "image":
        return f"image:{image['id']}", "image"
    for field in ("sequence_id", "video_id", "dive_id"):
        if image.get(field) is not None:
            return f"{field}:{image[field]}", field
    url = image.get("coco_url") or image.get("flickr_url") or ""
    parts = unquote(urlparse(url).path).strip("/").split("/")
    if "framegrabs" in parts:
        index = parts.index("framegrabs")
        if len(parts) > index + 4 and parts[index + 2] == "images":
            return "dive:" + "/".join(parts[index : index + 4]), "url_dive"
    # NOAA names explicitly preserve the video identifier before the frame suffix
    video = re.fullmatch(r"(.+_VID_\d{8}T\d{6}Z_[A-Za-z0-9]+)_frame_\d+\.[A-Za-z0-9]+", parts[-1])
    if video:
        return f"video:{video[1]}", "url_video"
    if group_by == "dive":
        raise ValueError("no explicit sequence, dive or video group could be inferred")
    date = str(image.get("date_captured", ""))[:10]
    if len(date) == 10 and date[4] == "-" and date[7] == "-":
        return f"capture_day:{date}", "capture_day"
    return f"image:{image['id']}", "image_fallback"


def _load_csv_labels(path, categories, id_column, label_column, errors):
    if path is None:
        return {}
    names = {category["name"]: category["id"] for category in categories.values()}
    result = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {id_column, label_column}.issubset(reader.fieldnames):
            errors.append(f"CSV must contain exact columns {id_column!r} and {label_column!r}")
            return result
        for line, row in enumerate(reader, 2):
            key, label = row[id_column].strip(), row[label_column].strip()
            category_id = names.get(label)
            if category_id is None:
                try:
                    category_id = int(label)
                except ValueError:
                    pass
            if not key or category_id not in categories:
                errors.append(f"CSV line {line}: missing annotation id or unknown category {label!r}")
            elif key in result:
                errors.append(f"CSV line {line}: duplicate annotation id {key!r}")
            else:
                result[key] = category_id
    return result


def _assign_splits(records, annotations, class_ids, fractions, seed):
    """Balance groups, then prioritize class coverage over exact split proportions

    A class represented by enough groups should occur in every active split.
    Coverage repair moves whole groups and never removes a donor's required
    class coverage. Classes with too few groups prioritize the training split.
    Conflicting coverage constraints can still leave a class absent; the audit
    reports the actual coverage rather than claiming exact stratification.
    """
    grouped = defaultdict(list)
    for record in records.values():
        grouped[record["group"]].append(record["id"])
    counts = {}
    for group, image_ids in grouped.items():
        counts[group] = Counter(ann["category_id"] for image_id in image_ids for ann in annotations[image_id])
    totals = sum(counts.values(), Counter())
    active = [name for name, fraction in zip(SPLITS, fractions) if fraction > 0]
    if len(grouped) < len(active):
        raise ValueError(f"{len(grouped)} source groups cannot populate {len(active)} nonempty splits")
    groups = sorted(grouped)
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda key: (len(grouped[key]), sum(counts[key].values())), reverse=True)
    assigned, sizes, class_counts = {}, Counter(), {name: Counter() for name in active}
    split_groups = Counter()
    total_images = len(records)
    fraction_by_name = dict(zip(SPLITS, fractions))
    for index, group in enumerate(groups):
        empty = [name for name in active if not split_groups[name]]
        candidates = empty if len(groups) - index == len(empty) else active

        def cost(name):
            fraction = fraction_by_name[name]
            target = total_images * fraction
            old = (sizes[name] - target) ** 2
            new = (sizes[name] + len(grouped[group]) - target) ** 2
            score = (new - old) / max(target, 1)
            for category_id, count in counts[group].items():
                target = totals[category_id] * fraction
                before = class_counts[name][category_id] - target
                score += ((before + count) ** 2 - before**2) / max(target, 1) / max(len(class_ids), 1)
            return score

        chosen = min(candidates, key=cost)
        assigned[group] = chosen
        sizes[chosen] += len(grouped[group])
        class_counts[chosen].update(counts[group])
        split_groups[chosen] += 1
    class_groups = {category_id: [group for group in groups if counts[group][category_id]] for category_id in class_ids}
    required = {
        category_id: set(active) if len(members) >= len(active) else {"train"}
        for category_id, members in class_groups.items()
    }
    # A size-only greedy choice can put all of a rare taxon's groups in train
    # Restore coverage with the smallest admissible provenance-preserving moves
    for category_id in sorted(class_ids, key=lambda value: (len(class_groups[value]), value)):
        for destination in active:
            if destination not in required[category_id] or class_counts[destination][category_id]:
                continue
            candidates = []
            for group in class_groups[category_id]:
                donor = assigned[group]
                if donor == destination or split_groups[donor] <= 1:
                    continue
                if any(
                    donor in required[other] and class_counts[donor][other] <= count
                    for other, count in counts[group].items()
                ):
                    continue
                candidates.append(group)
            if not candidates:
                continue
            group = min(candidates, key=lambda value: (len(grouped[value]), value))
            donor = assigned[group]
            assigned[group] = destination
            sizes[donor] -= len(grouped[group])
            sizes[destination] += len(grouped[group])
            class_counts[donor].subtract(counts[group])
            class_counts[destination].update(counts[group])
            split_groups[donor] -= 1
            split_groups[destination] += 1
    return {image_id: assigned[record["group"]] for image_id, record in records.items()}


def prepare_dataset(
    coco_path,
    images_dir,
    output_dir,
    *,
    labels_csv=None,
    csv_id_column="annotation_id",
    csv_label_column="concept_name",
    image_naming="auto",
    group_by="auto",
    fractions=(0.7, 0.15, 0.15),
    seed=42,
    link_mode="auto",
    verify_images=True,
    box_policy="error",
    duplicate_policy="error",
):
    """Create an audited, grouped YOLO dataset in a new output directory.

    Args:
        coco_path (str | Path): Training JSON with images, annotations and categories
        images_dir (str | Path): Existing directory containing full images, not ROI crops
        output_dir (str | Path): New destination; existing directories are refused
        labels_csv (str | Path | None): Optional labels joined by annotation id, never row order
        csv_id_column (str): Exact CSV column containing COCO annotation ids
        csv_label_column (str): Exact CSV column containing category ids or exact names
        image_naming (str): auto, file_name or image_id (official FGVC downloader naming)
        group_by (str): auto, dive or image. Dive requires explicit dive/video
            groups; auto also uses capture days or image-level groups with warnings
        fractions (tuple): Desired train, validation and test image fractions
        seed (int): Seed used for group ordering and split reproducibility
        link_mode (str): auto, symlink, hardlink or copy. Auto uses hardlinks on Windows
        verify_images (bool): Read image headers and verify files with Pillow
        box_policy (str): error rejects out-of-image boxes; clip intersects them
            with the image and explicitly excludes intersections of zero area
        duplicate_policy (str): error rejects exact class/box duplicates; drop
            retains their first annotation and records every excluded id

    Returns:
        (dict): Audit including actual split distributions and missing classes.

    Raises:
        DatasetValidationError: Invalid annotations, images, joins or unusable splits.
        FileExistsError: Destination already exists; no files are overwritten.
    """
    if image_naming not in ("auto", "file_name", "image_id"):
        raise ValueError("image_naming must be auto, file_name or image_id")
    if group_by not in ("auto", "dive", "image"):
        raise ValueError("group_by must be auto, dive or image")
    if link_mode not in ("auto", "symlink", "hardlink", "copy"):
        raise ValueError("link_mode must be auto, symlink, hardlink or copy")
    if box_policy not in ("error", "clip") or duplicate_policy not in ("error", "drop"):
        raise ValueError("box_policy must be error or clip; duplicate_policy must be error or drop")
    if len(fractions) != 3 or any(not math.isfinite(f) or f < 0 for f in fractions):
        raise ValueError("fractions must contain three finite nonnegative values")
    if fractions[0] <= 0 or not math.isclose(sum(fractions), 1, abs_tol=1e-8):
        raise ValueError("fractions must sum to one and include a nonempty training fraction")
    coco_path, images_dir, output_dir = (
        Path(path).expanduser().absolute() for path in (coco_path, images_dir, output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "invalid",
        "coco_path": str(coco_path),
        "images_dir": str(images_dir),
        "seed": seed,
        "image_naming": image_naming,
        "requested_fractions": dict(zip(SPLITS, fractions)),
        "group_by": group_by,
        "box_policy": box_policy,
        "duplicate_policy": duplicate_policy,
        "annotation_changes": [],
        "excluded_annotations": [],
        "excluded_images": [],
        "errors": [],
        "warnings": [],
    }
    errors, warnings = report["errors"], report["warnings"]

    def fail():
        _write_json(output_dir / "audit.json", report)
        raise DatasetValidationError(
            f"Dataset validation failed ({len(errors)} errors); see {output_dir / 'audit.json'}"
        )

    try:
        raw = coco_path.read_bytes()
        data = json.loads(raw)
        report["coco_sha256"] = hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError) as error:
        errors.append(f"Cannot read COCO JSON: {error}")
        fail()
    if not isinstance(data, dict) or any(
        not isinstance(data.get(key), list) for key in ("images", "annotations", "categories")
    ):
        errors.append("COCO JSON must contain images, annotations and categories lists")
        fail()
    report["source_images"] = len(data["images"])
    categories = {}
    for category in data["categories"]:
        if (
            not isinstance(category, dict)
            or not _positive_integer(category.get("id"))
            or not isinstance(category.get("name"), str)
            or not category["name"].strip()
        ):
            errors.append(f"Invalid category: {category!r}")
        elif category["id"] in categories:
            errors.append(f"Duplicate category id: {category['id']}")
        else:
            categories[category["id"]] = category
    if not categories or len({c["name"] for c in categories.values()}) != len(categories):
        errors.append("Categories must be nonempty and have distinct names")
    if errors:
        fail()
    try:
        csv_labels = _load_csv_labels(labels_csv, categories, csv_id_column, csv_label_column, errors)
        if labels_csv is not None:
            report["labels_csv"] = {
                "path": str(Path(labels_csv).expanduser().absolute()),
                "sha256": hashlib.sha256(Path(labels_csv).read_bytes()).hexdigest(),
                "id_column": csv_id_column,
                "label_column": csv_label_column,
            }
    except OSError as error:
        errors.append(f"Cannot read labels CSV: {error}")
        fail()
    records, group_sources = {}, Counter()
    path_groups, url_groups = {}, {}
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        while key != parent[key]:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for image in data["images"]:
        try:
            if not isinstance(image, dict) or not _positive_integer(image.get("id")):
                raise ValueError("image must have a positive integer id")
            image_id = image["id"]
            if image_id in records:
                raise ValueError("duplicate image id")
            if not all(_positive_integer(image.get(key)) for key in ("width", "height")):
                raise ValueError("width and height must be positive integers")
            source = _resolve_image(image, images_dir, image_naming)
            if verify_images:
                from PIL import Image

                with Image.open(source) as opened:
                    if opened.size != (image["width"], image["height"]):
                        raise ValueError(
                            f"JSON dimensions {(image['width'], image['height'])} differ from file {opened.size}"
                        )
                    if opened.getexif().get(274, 1) != 1:
                        raise ValueError("EXIF orientation needs explicit reconciliation with annotation coordinates")
                    opened.load()
                if source.suffix.lower() in (".jpg", ".jpeg"):
                    with source.open("rb") as handle:
                        handle.seek(-2, os.SEEK_END)
                        if handle.read() != b"\xff\xd9":
                            raise ValueError(
                                "JPEG lacks end marker; reject before training can attempt an in-place repair"
                            )
            group, group_source = _group_key(image, group_by)
            find(group)
            # Identical source files or URLs must never leak across splits, even in image mode.
            url = image.get("coco_url") or image.get("flickr_url")
            for mapping, key in ((path_groups, str(source)), (url_groups, url)):
                if key:
                    if key in mapping:
                        parent[find(group)] = find(mapping[key])
                    mapping[key] = group
            records[image_id] = {
                "id": image_id,
                "source": source,
                "width": image["width"],
                "height": image["height"],
                "group": group,
                "group_source": group_source,
            }
            group_sources[group_source] += 1
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors.append(f"Image {image.get('id') if isinstance(image, dict) else '?'}: {error}")
    for record in records.values():
        record["group"] = find(record["group"])
    if group_sources["image_fallback"]:
        warnings.append(
            f"No source group inferred for {group_sources['image_fallback']} images; "
            "auto used image-level groups. Review leakage before training."
        )
    if group_sources["capture_day"]:
        warnings.append(
            f"{group_sources['capture_day']} images are grouped only by capture day; "
            "these groups do not establish complete dive-level separation."
        )
    if group_by == "image":
        warnings.append("Explicit image-level split: nearby video frames may leak across partitions.")
    report["group_sources"] = dict(group_sources)
    annotations, seen_ids, duplicate_rows = defaultdict(list), set(), {}
    for annotation in data["annotations"]:
        try:
            if not isinstance(annotation, dict) or not _positive_integer(annotation.get("id")):
                raise ValueError("annotation must have a positive integer id")
            annotation_id = annotation["id"]
            if annotation_id in seen_ids:
                raise ValueError("duplicate annotation id")
            seen_ids.add(annotation_id)
            image_id = annotation.get("image_id")
            if not _positive_integer(image_id) or image_id not in records:
                raise ValueError("unknown or invalid image_id")
            category_id = annotation.get("category_id")
            csv_category = csv_labels.get(str(annotation_id))
            if csv_category is not None:
                if category_id in categories and category_id != csv_category:
                    raise ValueError("CSV category conflicts with the labeled COCO annotation")
                category_id = csv_category
            if not _positive_integer(category_id) or category_id not in categories:
                raise ValueError("unknown or masked category_id; provide exact CSV join if labels exist separately")
            if annotation.get("iscrowd", 0):
                raise ValueError("crowd annotation unsupported; it cannot silently become background")
            box = annotation.get("bbox")
            if (
                not isinstance(box, list)
                or len(box) != 4
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)
            ):
                raise ValueError("bbox must contain four finite numeric xywh values")
            x, y, width, height = box
            image = records[image_id]
            if width <= 0 or height <= 0:
                raise ValueError("bbox must have strictly positive width and height")
            if x < 0 or y < 0 or x + width > image["width"] or y + height > image["height"]:
                if box_policy == "error":
                    raise ValueError("bbox extends beyond image; no silent clipping is performed")
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(image["width"], x + width), min(image["height"], y + height)
                if x2 <= x1 or y2 <= y1:
                    report["excluded_annotations"].append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "reason": "outside_image_no_intersection",
                            "original_bbox": box,
                        }
                    )
                    continue
                corrected = [x1, y1, x2 - x1, y2 - y1]
                report["annotation_changes"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "reason": "clipped_to_image",
                        "original_bbox": box,
                        "corrected_bbox": corrected,
                    }
                )
                box = corrected
            row_key = (image_id, category_id, *box)
            if row_key in duplicate_rows:
                if duplicate_policy == "error":
                    raise ValueError("duplicate class/box annotation for the same image")
                report["excluded_annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "reason": "duplicate_class_box",
                        "kept_annotation_id": duplicate_rows[row_key],
                        "original_bbox": annotation["bbox"],
                        "corrected_bbox": box,
                    }
                )
                continue
            duplicate_rows[row_key] = annotation_id
            annotations[image_id].append({"id": annotation_id, "category_id": category_id, "bbox": box})
        except (ValueError, TypeError) as error:
            errors.append(f"Annotation {annotation.get('id') if isinstance(annotation, dict) else '?'}: {error}")
    unknown_csv = sorted(set(csv_labels) - {str(value) for value in seen_ids})
    if unknown_csv:
        errors.append(f"CSV contains {len(unknown_csv)} ids absent from COCO annotations, e.g. {unknown_csv[:5]}")
    if not records or not seen_ids:
        errors.append("Labeled training images and annotations must be nonempty")
    if errors:
        fail()
    empty_images = [image_id for image_id in records if not annotations[image_id]]
    if empty_images:
        warnings.append(
            f"Excluded {len(empty_images)} images without retained annotations; not treated as verified negatives."
        )
        for image_id in empty_images:
            report["excluded_images"].append({"image_id": image_id, "reason": "no_retained_annotations"})
            del records[image_id]
    if not records:
        errors.append("No labeled images remain after applying the preparation policies")
        fail()
    report["group_sources"] = dict(Counter(record["group_source"] for record in records.values()))
    try:
        splits = _assign_splits(records, annotations, categories, fractions, seed)
    except ValueError as error:
        errors.append(str(error))
        fail()
    class_map = [
        {"yolo_id": index, "category_id": category_id, "name": categories[category_id]["name"]}
        for index, category_id in enumerate(sorted(categories))
    ]
    yolo_ids = {item["category_id"]: item["yolo_id"] for item in class_map}
    split_counts = {
        name: Counter(
            ann["category_id"] for image_id in records if splits[image_id] == name for ann in annotations[image_id]
        )
        for name in SPLITS
    }
    report["splits"] = {}
    for name in SPLITS:
        selected = [record for image_id, record in records.items() if splits[image_id] == name]
        missing = sorted(set(categories) - set(split_counts[name]))
        report["splits"][name] = {
            "images": len(selected),
            "groups": len({record["group"] for record in selected}),
            "annotations": sum(split_counts[name].values()),
            "category_counts": dict(sorted(split_counts[name].items())),
            "missing_category_ids": missing,
        }
        if fractions[SPLITS.index(name)] and missing:
            warnings.append(
                f"{name} has no examples of {len(missing)} categories; inspect per-class evaluation coverage"
            )
    missing_train = set(categories) - set(split_counts["train"])
    if missing_train:
        errors.append(
            f"Training split misses categories {sorted(missing_train)}; revise grouping/splits before training"
        )
        fail()
    mode = ("hardlink" if os.name == "nt" else "symlink") if link_mode == "auto" else link_mode
    report["link_mode"] = mode
    report["verified_image_files"] = verify_images
    for directory in ("images", "labels", "splits"):
        (output_dir / directory).mkdir()
    manifest = []
    try:
        for image_id, record in sorted(records.items()):
            target = output_dir / "images" / (str(image_id) + record["source"].suffix.lower())
            if mode == "symlink":
                target.symlink_to(record["source"])
            elif mode == "hardlink":
                os.link(record["source"], target)
            else:
                shutil.copyfile(record["source"], target)
            rows = []
            for annotation in sorted(annotations[image_id], key=lambda a: a["id"]):
                x, y, width, height = annotation["bbox"]
                values = (
                    (x + width / 2) / record["width"],
                    (y + height / 2) / record["height"],
                    width / record["width"],
                    height / record["height"],
                )
                rows.append(
                    str(yolo_ids[annotation["category_id"]]) + " " + " ".join(format(value, ".10g") for value in values)
                )
            (output_dir / "labels" / f"{image_id}.txt").write_text(
                "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
            )
            manifest.append(
                {
                    "image_id": image_id,
                    "source": str(record["source"]),
                    "image": str(target),
                    "split": splits[image_id],
                    "group": record["group"],
                    "annotation_ids": [ann["id"] for ann in annotations[image_id]],
                }
            )
        for name in SPLITS:
            (output_dir / "splits" / f"{name}.txt").write_text(
                "".join(item["image"] + "\n" for item in manifest if item["split"] == name), encoding="utf-8"
            )
    except OSError as error:
        errors.append(f"Cannot create image {mode}: {error}; select another --link-mode and a new output directory")
        fail()
    _write_json(output_dir / "class_map.json", class_map)
    _write_json(output_dir / "split_manifest.json", manifest)
    # JSON is valid YAML; writing it avoids importing any training dependencies.
    dataset = {
        "path": str(output_dir),
        "train": "splits/train.txt",
        "val": "splits/val.txt",
        "names": {item["yolo_id"]: item["name"] for item in class_map},
    }
    if fractions[2]:
        dataset["test"] = "splits/test.txt"
    _write_json(output_dir / "data.yaml", dataset)
    report["status"] = "valid"
    report["images"] = len(records)
    report["source_annotations"] = len(data["annotations"])
    report["annotations"] = sum(len(annotations[image_id]) for image_id in records)
    report["categories"] = len(categories)
    warnings.append(
        "Annotation completeness is not established automatically; "
        "audit visible unannotated organisms before interpreting detector AP"
    )
    _write_json(output_dir / "audit.json", report)
    return report


def main():
    """Run the preparation CLI without importing Ultralytics or downloading images"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--labels-csv")
    parser.add_argument("--csv-id-column", default="annotation_id")
    parser.add_argument("--csv-label-column", default="concept_name")
    parser.add_argument("--image-naming", choices=("auto", "file_name", "image_id"), default="auto")
    parser.add_argument("--group-by", choices=("auto", "dive", "image"), default="auto")
    parser.add_argument("--fractions", type=float, nargs=3, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link-mode", choices=("auto", "symlink", "hardlink", "copy"), default="auto")
    parser.add_argument("--skip-image-verification", action="store_true")
    parser.add_argument(
        "--box-policy",
        choices=("error", "clip"),
        default="error",
        help="clip intersects boxes with images and records exclusions when the intersection is empty",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=("error", "drop"),
        default="error",
        help="drop records duplicate class/box annotation IDs and retains the first occurrence",
    )
    args = vars(parser.parse_args())
    args["coco_path"] = args.pop("coco")
    args["verify_images"] = not args.pop("skip_image_verification")
    report = prepare_dataset(**args)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
