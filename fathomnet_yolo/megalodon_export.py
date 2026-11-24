"""Select fully labeled Megalodon images and export the original splits for YOLO."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
from uuid import UUID

from PIL import Image

SPLITS = ("train", "val", "test")
EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tif"}
S3_URL = "https://fathomnet-pds.s3.us-west-2.amazonaws.com/"


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_text(path, content):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


@contextmanager
def lock_output(output):
    """Prevent concurrent writes to one dataset. Closing the file releases the lock."""
    with (output / ".megalodon_export.lock").open("a+b") as lock:
        try:
            if sys.platform == "win32":
                import msvcrt

                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError(f"Cannot lock {output}. Use one export per output directory.") from error
        yield


def _unique_json_object(entries):
    result = {}
    for key, value in entries:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def read_concept_map(raw):
    mapping = json.loads(raw, object_pairs_hook=_unique_json_object)
    if not isinstance(mapping, dict) or any(
        not name.strip()
        or (target is not None and (not isinstance(target, str) or not target.strip() or target != target.strip()))
        for name, target in mapping.items()
    ):
        raise ValueError("The concept map must map original names to accepted names or null")
    return mapping


def _checked_box(box, width, height):
    if (
        not isinstance(box, list)
        or len(box) != 4
        or any(type(value) not in (float, int) or not math.isfinite(value) for value in box)
    ):
        raise ValueError(f"Malformed bbox {box!r}")
    x, y, w, h = box
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
        raise ValueError(f"Invalid bbox {box!r} for {width}x{height}")
    return box


def plan_export(manifest, mapping, limit):
    """Select complete images, keeping labels predictable from the selected train split."""
    input_counts = {split: Counter() for split in SPLITS}
    excluded = {split: Counter() for split in SPLITS}
    selected = {split: [] for split in SPLITS}
    seen = set()
    with manifest.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                image = json.loads(line)
                split = image["split"]
                if not isinstance(image["uuid"], str):
                    raise TypeError("Expected an image UUID string")
                identifier = str(UUID(image["uuid"]))
                if identifier != image["uuid"] or identifier in seen or split not in SPLITS:
                    raise ValueError(f"Unexpected split or repeated/invalid UUID {split!r} {identifier}")
                seen.add(identifier)
                boxes = image["annotations"]
                if not isinstance(boxes, list):
                    raise TypeError("annotations must be a list")
                input_counts[split].update({"images": 1, "boxes": len(boxes)})
                if image["image_status"] == "dimensions_mismatch":
                    excluded[split]["dimensions_mismatch"] += 1
                    continue
                if not boxes:
                    excluded[split]["no_boxes"] += 1
                    continue
                if any(box["status"] != "matched" for box in boxes):
                    excluded[split]["some_boxes_unmatched"] += 1
                    continue
                names = [mapping.get(box["concept"]) for box in boxes]
                if any(name is None for name in names):
                    excluded[split]["some_concepts_not_selected"] += 1
                    continue
                if not (image.get("s3_key") or image.get("source_url")):
                    excluded[split]["no_image_source"] += 1
                    continue
                width, height = image["width"], image["height"]
                if any(type(value) is not int or value <= 0 for value in (width, height)):
                    raise ValueError("Expected positive integer image dimensions")
                image["labels"] = [(name, _checked_box(box["bbox"], width, height)) for name, box in zip(names, boxes)]
                selected[split].append(image)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{manifest}:{line_number}: {error}") from error

    for images in selected.values():
        images.sort(key=lambda image: image["uuid"])

    def trim(split):
        if limit is not None and len(selected[split]) > limit:
            excluded[split]["pilot_limit"] += len(selected[split]) - limit
            selected[split] = selected[split][:limit]

    trim("train")
    train_classes = {name for image in selected["train"] for name, _ in image["labels"]}
    outside_train = {}
    for split in ("val", "test"):
        retained = []
        outside = Counter()
        for image in selected[split]:
            missing = [name for name, _ in image["labels"] if name not in train_classes]
            if missing:
                excluded[split]["some_classes_absent_from_train"] += 1
                outside.update(missing)
            else:
                retained.append(image)
        selected[split] = retained
        outside_train[split] = dict(sorted(outside.items()))
        trim(split)

    class_map = [
        {"yolo_id": index, "category_id": index + 1, "name": name} for index, name in enumerate(sorted(train_classes))
    ]
    summary = {"classes": len(class_map), "splits": {}}
    for split in SPLITS:
        images = selected[split]
        classes = Counter(name for image in images for name, _ in image["labels"])
        known_sizes = [image["size_bytes"] for image in images if image.get("size_bytes") is not None]
        summary["splits"][split] = {
            "input_images": input_counts[split]["images"],
            "input_boxes": input_counts[split]["boxes"],
            "selected_images": len(images),
            "selected_boxes": sum(classes.values()),
            "class_counts": dict(sorted(classes.items())),
            "boxes_of_classes_absent_from_selected_train": outside_train.get(split, {}),
            "excluded_images": dict(sorted(excluded[split].items())),
            "selected_images_with_known_size": len(known_sizes),
            "selected_known_bytes": sum(known_sizes),
        }
    return selected, class_map, summary


def image_extension(path, image):
    """Verify format, orientation, dimensions and complete decoding with Pillow."""
    with Image.open(path) as source:
        extension = EXTENSIONS.get(source.format)
        if extension is None:
            raise ValueError(f"Unsupported image format {source.format}")
        if source.size != (image["width"], image["height"]):
            raise ValueError(f"Dimensions differ: decoded {source.size}, expected {(image['width'], image['height'])}")
        if source.getexif().get(274, 1) != 1 or getattr(source, "n_frames", 1) != 1:
            raise ValueError("Image orientation or multiple frames require manual inspection against the COCO boxes")
        source.load()
    return extension


def source_candidates(image):
    key = image.get("s3_key")
    if key:
        yield "s3", S3_URL + quote(key, safe="/")
    link = image.get("source_url")
    if link and urlparse(link).scheme in ("http", "https") and urlparse(link).netloc:
        yield "source_url", link


def get_image(image, directory, receipt, attempts, timeout):
    """Return a receipt; validate any existing image before reusing it."""
    identifier = image["uuid"]
    expected_sha = image.get("sha256")
    if expected_sha and not re.fullmatch(r"[a-f0-9]{64}", expected_sha):
        raise ValueError(f"Malformed manifest SHA-256 for {identifier}")
    for suffix in EXTENSIONS.values():
        path = directory / (identifier + suffix)
        if not path.exists():
            continue
        try:
            file_hash = file_sha256(path)
            if expected_sha and file_hash == expected_sha:
                source_kind = receipt["source"] if receipt and receipt["sha256"] == file_hash else "s3_checksum"
            elif receipt and receipt["file_name"] == path.name and receipt["sha256"] == file_hash:
                source_kind = receipt["source"]
            else:
                continue
            if image_extension(path, image) == suffix:
                return {"uuid": identifier, "file_name": path.name, "sha256": file_hash, "source": source_kind}
        except (OSError, ValueError, Image.DecompressionBombError):
            continue

    failures = []
    for kind, link in source_candidates(image):
        for attempt in range(attempts):
            try:
                request = Request(link, headers={"User-Agent": "FathomNet-Megalodon-YOLO/1.0"})
                with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as stream:
                    temporary = Path(stream.name)
                    try:
                        with urlopen(request, timeout=timeout) as response:
                            shutil.copyfileobj(response, stream)
                        stream.close()
                        file_hash = file_sha256(temporary)
                        if kind == "s3" and expected_sha and file_hash != expected_sha:
                            raise ValueError("S3 image SHA-256 mismatch")
                        suffix = image_extension(temporary, image)
                        path = directory / (identifier + suffix)
                        temporary.replace(path)
                        return {"uuid": identifier, "file_name": path.name, "sha256": file_hash, "source": kind}
                    finally:
                        stream.close()
                        temporary.unlink(missing_ok=True)
            except (OSError, ValueError, HTTPException, Image.DecompressionBombError) as error:
                if (
                    attempt + 1 == attempts
                    or isinstance(error, ValueError)
                    or (isinstance(error, HTTPError) and error.code < 500 and error.code not in (408, 429))
                ):
                    failures.append(f"{kind}: {error}")
                    break
                time.sleep(min(2**attempt, 16))
    raise ValueError(" / ".join(failures) or "No supported image source")


def write_labels(output, image, by_name):
    width, height = image["width"], image["height"]
    lines = []
    for name, (x, y, w, h) in image["labels"]:
        values = ((x + w / 2) / width, (y + h / 2) / height, w / width, h / height)
        coordinates = " ".join(f"{value:.12g}" for value in values)
        lines.append(f"{by_name[name]} {coordinates}\n")
    atomic_text(output / "labels" / image["split"] / (image["uuid"] + ".txt"), "".join(lines))


def read_receipts(path):
    if not path.exists():
        return {}
    result = {}
    with path.open("r+b") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                stream.truncate(offset)
                break
            receipt = json.loads(line)
            result[receipt["uuid"]] = receipt
    return result


def export_images(output, selected, class_map, summary, fingerprint, workers, attempts, timeout, skip_failed=False):
    """Resume a selection and optionally finalize the validated images after failures."""
    marker = output / "export_inputs.json"
    report = output / "export_summary.json"
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != fingerprint:
            raise ValueError("This directory belongs to a different manifest or selection. Choose a new --output-dir.")
    else:
        if any(path.name not in (report.name, ".megalodon_export.lock") for path in output.iterdir()):
            raise ValueError("Choose an empty --output-dir for a new export.")
        write_json(marker, fingerprint)

    data_yaml = output / "data.yaml"
    if data_yaml.exists():
        data_yaml.replace(output / "data.yaml.previous")
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output / "splits").mkdir(exist_ok=True)
    receipt_path = output / "download_receipts.jsonl"
    prior_receipts = read_receipts(receipt_path)
    available = {split: [] for split in SPLITS}
    download_sources = {split: Counter() for split in SPLITS}
    failures = []
    total = sum(map(len, selected.values()))
    processed = 0
    with (
        receipt_path.open("a", encoding="utf-8", newline="\n", buffering=1) as receipts_file,
        concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        for split in SPLITS:
            images = selected[split]
            directory = output / "images" / split
            for start in range(0, len(images), workers * 4):
                futures = {
                    pool.submit(
                        get_image, image, directory, prior_receipts.get(image["uuid"]), attempts, timeout
                    ): image
                    for image in images[start : start + workers * 4]
                }
                for future in concurrent.futures.as_completed(futures):
                    image = futures[future]
                    try:
                        receipt = future.result()
                    except (OSError, ValueError, Image.DecompressionBombError) as error:
                        failures.append({"uuid": image["uuid"], "split": split, "error": str(error)})
                    else:
                        if prior_receipts.get(image["uuid"]) != receipt:
                            receipts_file.write(json.dumps(receipt, ensure_ascii=False) + "\n")
                        download_sources[split][receipt["source"]] += 1
                        available[split].append((image, directory / receipt["file_name"]))
                    processed += 1
                    if processed % 1000 == 0 or processed == total:
                        print(f"Images checked: {processed}/{total}. Failures: {len(failures)}", flush=True)

    summary["status"] = "downloads_failed" if failures else "ready"
    summary["skip_failed"] = skip_failed
    summary["download_failures"] = len(failures)
    for split in SPLITS:
        summary["splits"][split]["download_sources"] = dict(download_sources[split])
    write_json(output / "download_failures.json", failures)
    if failures and not skip_failed:
        write_json(report, summary)
        raise ValueError(
            f"{len(failures)} images failed. See download_failures.json and rerun the same command, "
            "or add --skip-failed to finalize the available images."
        )

    # A failed training image may have been the only example of a class.
    # Rebuild class IDs and all labels from the validated train membership.
    train_classes = {name for image, _ in available["train"] for name, _ in image["labels"]}
    class_map = [
        {"yolo_id": index, "category_id": index + 1, "name": name} for index, name in enumerate(sorted(train_classes))
    ]
    exclusions = []
    for split in ("val", "test"):
        retained = []
        for image, path in available[split]:
            missing_classes = sorted({name for name, _ in image["labels"]} - train_classes)
            if missing_classes:
                exclusions.append({"uuid": image["uuid"], "split": split, "classes_absent_from_train": missing_classes})
            else:
                retained.append((image, path))
        available[split] = retained
    summary["classes_before_download"] = summary["classes"]
    summary["classes"] = len(class_map)
    summary["excluded_after_download"] = exclusions
    for split in SPLITS:
        counts = Counter(name for image, _ in available[split] for name, _ in image["labels"])
        summary["splits"][split].update(
            exported_images=len(available[split]),
            exported_boxes=sum(counts.values()),
            exported_class_counts=dict(sorted(counts.items())),
        )
    empty = [split for split in SPLITS if not available[split]]
    if empty:
        summary["status"] = "empty_split_after_download"
        write_json(report, summary)
        raise ValueError(f"No usable images in {', '.join(empty)} after download. See export_summary.json.")
    by_name = {row["name"]: row["yolo_id"] for row in class_map}
    for split in SPLITS:
        for image, _ in available[split]:
            write_labels(output, image, by_name)
        paths = sorted(str(path) for _, path in available[split])
        atomic_text(output / "splits" / f"{split}.txt", "".join(path + "\n" for path in paths))
    write_json(output / "class_map.json", class_map)
    write_json(
        output / "categories_coco.json",
        {"categories": [{"id": row["category_id"], "name": row["name"]} for row in class_map]},
    )
    # JSON is also valid YAML. File lists give YOLO the exact validated selection.
    data = {"path": str(output), **{split: f"splits/{split}.txt" for split in SPLITS}}
    data["names"] = [row["name"] for row in class_map]
    write_json(data_yaml, data)
    summary["status"] = "ready_with_skipped_images" if failures else "ready"
    write_json(report, summary)
    if failures:
        print(f"Finalized available images; {len(failures)} failed downloads are listed in download_failures.json.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="manifest.jsonl from megalodon_audit")
    parser.add_argument(
        "--concept-map", type=Path, required=True, help="JSON mapping original concepts to names or null"
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination of the planned or exported dataset")
    parser.add_argument("--download", action="store_true", help="Download images and export YOLO files")
    parser.add_argument(
        "--skip-failed", action="store_true", help="Finalize the validated images if some downloads fail"
    )
    parser.add_argument("--limit-per-split", type=int, help="Deterministic pilot cap; use a separate output directory")
    parser.add_argument("--workers", type=int, default=8, help="Number of simultaneous downloads")
    parser.add_argument("--attempts", type=int, default=3, help="Attempts per image source")
    parser.add_argument("--timeout", type=float, default=30, help="Network timeout in seconds")
    args = parser.parse_args()
    if args.workers < 1 or args.attempts < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--workers, --attempts and --timeout must be positive")
    if args.limit_per_split is not None and args.limit_per_split < 1:
        parser.error("--limit-per-split must be positive")

    try:
        raw_map = args.concept_map.read_bytes()
        fingerprint = {
            "schema_version": 1,
            "manifest_sha256": file_sha256(args.manifest),
            "concept_map_sha256": hashlib.sha256(raw_map).hexdigest(),
            "limit_per_split": args.limit_per_split,
        }
        selected, class_map, summary = plan_export(args.manifest, read_concept_map(raw_map), args.limit_per_split)
        if fingerprint["manifest_sha256"] != file_sha256(args.manifest):
            raise ValueError("The audit manifest changed during selection. Rerun the command.")
        summary.update({"status": "plan", "inputs": fingerprint})
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        for split, stats in summary["splits"].items():
            print(
                f"{split}: {stats['selected_images']}/{stats['input_images']} images, {stats['selected_boxes']} boxes"
            )
        print(f"Training classes: {summary['classes']}")
        with lock_output(output):
            marker = output / "export_inputs.json"
            if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != fingerprint:
                raise ValueError(
                    "This directory belongs to a different manifest or selection. Choose a new --output-dir."
                )
            if args.download:
                empty = [split for split, images in selected.items() if not images]
                if empty:
                    write_json(output / "export_summary.json", summary)
                    raise ValueError(
                        f"Empty selected splits: {', '.join(empty)}. Review the plan or raise the pilot limit."
                    )
                export_images(
                    output,
                    selected,
                    class_map,
                    summary,
                    fingerprint,
                    args.workers,
                    args.attempts,
                    args.timeout,
                    skip_failed=args.skip_failed,
                )
            elif marker.exists():
                raise ValueError("Use a new output directory to plan a selection separately from an existing download.")
            else:
                write_json(output / "export_summary.json", summary)
        print(f"Summary: {output / 'export_summary.json'}")
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
