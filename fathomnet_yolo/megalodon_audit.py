"""Audit exact Megalodon box matches against a pinned FathomNet Open Data release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import sys
import tempfile
import uuid
from collections import Counter
from datetime import date
from pathlib import Path


SPLITS = ("train", "val", "test")
BUCKET = "s3://fathomnet-pds"
OBJECTS = f"{BUCKET}/manifests/objects/*.parquet"


def clean_text(value):
    return value.strip() or None if isinstance(value, str) else None


def integer(record, key, context, minimum=0):
    value = record.get(key)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{context}: expected integer {key} >= {minimum}")
    return value


def write_line(file, record):
    file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def stage_coco(coco_dir, images_path, boxes_path):
    """Validate the COCO identifiers and write flat input tables for DuckDB."""
    seen = {}
    input_files = {}
    with images_path.open("w", encoding="utf-8") as images_file, boxes_path.open("w", encoding="utf-8") as boxes_file:
        for split in SPLITS:
            path = coco_dir / f"coco_{split}.json"
            raw = path.read_bytes()
            input_files[split] = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
            coco = json.loads(raw)
            del raw
            if not isinstance(coco, dict) or not all(
                isinstance(coco.get(key), list) for key in ("images", "annotations", "categories")
            ):
                raise ValueError(f"{path}: expected COCO image, annotation and category arrays")
            categories = coco["categories"]
            if (
                len(categories) != 1 or not isinstance(categories[0], dict)
                or categories[0].get("name") != "object"
            ):
                raise ValueError(f"{path}: expected the Megalodon single category 'object'")
            object_category = integer(categories[0], "id", str(path))
            by_id = {}
            for image in coco["images"]:
                if not isinstance(image, dict):
                    raise ValueError(f"{path}: invalid image record")
                image_id = integer(image, "id", str(path))
                context = f"{path}: image {image_id}"
                if image_id in by_id:
                    raise ValueError(f"{context}: duplicate image id")
                if image.get("split") not in (None, split):
                    raise ValueError(f"{context}: record split differs from file split")
                try:
                    image_uuid = str(uuid.UUID(image["fathomnet_uuid"]))
                except (KeyError, AttributeError, TypeError, ValueError) as error:
                    raise ValueError(f"{context}: invalid or missing fathomnet_uuid") from error
                if image_uuid in seen:
                    raise ValueError(f"{context}: duplicate UUID {image_uuid} already in {seen[image_uuid]}")
                seen[image_uuid] = split
                width = integer(image, "width", context, minimum=1)
                height = integer(image, "height", context, minimum=1)
                by_id[image_id] = (image_uuid, width, height)
                write_line(images_file, {
                    "split": split, "uuid": image_uuid, "image_id": image_id,
                    "width": width, "height": height,
                    "source_url": clean_text(image.get("source_url")) or clean_text(image.get("coco_url")),
                    "coco_image_license": image.get("license"),
                })
            annotation_ids = set()
            for annotation in coco["annotations"]:
                if not isinstance(annotation, dict):
                    raise ValueError(f"{path}: invalid annotation record")
                box_id = integer(annotation, "id", str(path))
                context = f"{path}: annotation {box_id}"
                if box_id in annotation_ids:
                    raise ValueError(f"{context}: duplicate annotation id")
                annotation_ids.add(box_id)
                image_id = integer(annotation, "image_id", context)
                if image_id not in by_id:
                    raise ValueError(f"{context}: unknown image id {image_id}")
                if integer(annotation, "category_id", context) != object_category:
                    raise ValueError(f"{context}: unknown category id")
                bbox = annotation.get("bbox")
                if (
                    not isinstance(bbox, list) or len(bbox) != 4
                    or not all(type(value) in (int, float) and math.isfinite(value) for value in bbox)
                ):
                    raise ValueError(f"{context}: expected four finite bbox coordinates")
                x, y, w, h = map(float, bbox)
                image_uuid, width, height = by_id[image_id]
                write_line(boxes_file, {
                    "uuid": image_uuid, "id": box_id, "x": x, "y": y, "w": w, "h": h,
                    "valid": x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= width and y + h <= height,
                    "coco_review_state": clean_text(annotation.get("review_state")),
                    "coco_annotation_license": annotation.get("license"),
                })
            input_files[split].update(images=len(by_id), boxes=len(annotation_ids))
            del coco
    return input_files


def match_query(connection, images_path, boxes_path, images_parquet, boxes_parquet, objects_parquet):
    connection.execute("""
        CREATE TEMP TABLE benchmark_images AS
        SELECT * FROM read_json(?, format='newline_delimited', columns={
            split:'VARCHAR', uuid:'VARCHAR', image_id:'BIGINT', width:'INTEGER', height:'INTEGER',
            source_url:'VARCHAR', coco_image_license:'JSON'
        })
    """, [str(images_path)])
    connection.execute("""
        CREATE TEMP TABLE benchmark_boxes AS
        SELECT *, count(*) OVER (PARTITION BY uuid, x, y, w, h) AS benchmark_count
        FROM read_json(?, format='newline_delimited', columns={
            uuid:'VARCHAR', id:'BIGINT', x:'DOUBLE', y:'DOUBLE', w:'DOUBLE', h:'DOUBLE', valid:'BOOLEAN',
            coco_review_state:'VARCHAR', coco_annotation_license:'JSON'
        })
    """, [str(boxes_path)])
    connection.execute("""
        CREATE TEMP TABLE source_images AS
        SELECT p.image_uuid, count(*) AS n, max(p.width) AS width, max(p.height) AS height,
               max(p.s3_key) AS s3_key, lower(max(p.sha256)) AS sha256,
               max(p.source_url) AS source_url, max(p.image_license) AS image_license
        FROM read_parquet(?) p
        JOIN benchmark_images b ON p.image_uuid = b.uuid
        GROUP BY p.image_uuid
    """, [images_parquet])
    connection.execute("""
        CREATE TEMP TABLE source_matches AS
        SELECT b.uuid, b.id, count(*) AS n, max(trim(p.concept)) AS concept,
               max(p.box_uuid) AS box_uuid, max(upper(trim(p.review_state))) AS review_state,
               max(p.annotation_license) AS annotation_license
        FROM read_parquet(?) p
        JOIN benchmark_boxes b ON p.image_uuid = b.uuid AND p.x = b.x AND p.y = b.y
                               AND p.width = b.w AND p.height = b.h
        WHERE b.valid AND b.benchmark_count = 1
        GROUP BY b.uuid, b.id
    """, [boxes_parquet])
    if objects_parquet:
        connection.execute("""
            CREATE TEMP TABLE object_sizes AS
            SELECT s.image_uuid, min(p.size_bytes) AS size_bytes
            FROM read_parquet(?) p
            JOIN source_images s ON p.image_uuid = s.image_uuid AND p.s3_key = s.s3_key
                                 AND lower(p.sha256) = s.sha256
            WHERE s.n = 1
            GROUP BY s.image_uuid
            HAVING count(DISTINCT p.size_bytes) = 1 AND min(p.size_bytes) > 0
        """, [objects_parquet])
    else:
        connection.execute("CREATE TEMP TABLE object_sizes(image_uuid VARCHAR, size_bytes BIGINT)")
    return connection.execute("""
        WITH image_info AS (
            SELECT i.*,
                   CASE WHEN p.n IS NULL THEN 'missing_metadata'
                        WHEN p.n > 1 THEN 'ambiguous_metadata'
                        WHEN (p.width IS NOT NULL AND p.width <> i.width)
                          OR (p.height IS NOT NULL AND p.height <> i.height) THEN 'dimensions_mismatch'
                        ELSE 'available'
                   END AS image_status,
                   coalesce(i.source_url, CASE WHEN p.n = 1 THEN p.source_url END) AS download_url,
                   p.s3_key, p.sha256, p.image_license, o.size_bytes
            FROM benchmark_images i
            LEFT JOIN source_images p ON i.uuid = p.image_uuid
            LEFT JOIN object_sizes o ON i.uuid = o.image_uuid
        )
        SELECT i.split, i.uuid, i.image_id, i.width, i.height, i.download_url AS source_url,
               i.image_status, i.coco_image_license,
               CASE WHEN i.image_status = 'available' THEN i.s3_key END AS s3_key,
               CASE WHEN i.image_status = 'available' THEN i.sha256 END AS sha256,
               CASE WHEN i.image_status = 'available' THEN i.image_license END AS image_license,
               CASE WHEN i.image_status = 'available' THEN i.size_bytes END AS size_bytes,
               b.id AS box_id, b.x, b.y, b.w, b.h, b.coco_review_state, b.coco_annotation_license,
               s.concept,
               CASE WHEN s.n = 1 THEN s.box_uuid END AS box_uuid,
               CASE WHEN s.n = 1 THEN s.review_state END AS review_state,
               CASE WHEN s.n = 1 THEN s.annotation_license END AS annotation_license,
               CASE WHEN b.id IS NULL THEN NULL
                    WHEN NOT b.valid THEN 'invalid_geometry'
                    WHEN upper(b.coco_review_state) = 'REJECTED' THEN 'benchmark_rejected'
                    WHEN b.benchmark_count > 1 THEN 'duplicate_benchmark_bbox'
                    WHEN i.image_status = 'dimensions_mismatch' THEN 'dimensions_mismatch'
                    WHEN s.n IS NULL THEN 'no_exact_match'
                    WHEN s.n > 1 THEN 'ambiguous_exact_match'
                    WHEN s.review_state = 'REJECTED' THEN 'rejected_source_box'
                    WHEN s.concept IS NULL OR s.concept = '' THEN 'empty_concept'
                    ELSE 'matched'
               END AS box_status
        FROM image_info i
        LEFT JOIN benchmark_boxes b ON i.uuid = b.uuid
        LEFT JOIN source_matches s ON b.uuid = s.uuid AND b.id = s.id
        ORDER BY CASE i.split WHEN 'train' THEN 1 WHEN 'val' THEN 2 ELSE 3 END, i.image_id, b.id
    """)


def read_rows(cursor):
    fields = [column[0] for column in cursor.description]
    while rows := cursor.fetchmany(4096):
        for row in rows:
            yield dict(zip(fields, row))


def decode_json_value(value):
    return json.loads(value) if value is not None else None


def write_outputs(cursor, staging, sources):
    image_fields = (
        "split", "uuid", "image_id", "width", "height", "source_url", "image_status",
        "s3_key", "sha256", "size_bytes", "image_license",
    )
    counts = {split: Counter() for split in SPLITS}
    concepts = {split: Counter() for split in SPLITS}
    box_statuses = {split: Counter() for split in SPLITS}
    image_statuses = {split: Counter() for split in SPLITS}
    urls = {}
    cross_split_urls = set()
    with (staging / "manifest.jsonl").open("w", encoding="utf-8") as output:
        for _, group in itertools.groupby(read_rows(cursor), key=lambda row: row["uuid"]):
            rows = list(group)
            first = rows[0]
            image = {field: first[field] for field in image_fields}
            image["coco_image_license"] = decode_json_value(first["coco_image_license"])
            split = image["split"]
            stats = counts[split]
            annotations = []
            for row in rows:
                if row["box_id"] is None:
                    continue
                status = row["box_status"]
                concept = row["concept"] if status == "matched" else None
                annotations.append({
                    "id": row["box_id"], "bbox": [row["x"], row["y"], row["w"], row["h"]],
                    "status": status, "concept": concept, "box_uuid": row["box_uuid"],
                    "review_state": row["review_state"], "annotation_license": row["annotation_license"],
                    "coco_review_state": row["coco_review_state"],
                    "coco_annotation_license": decode_json_value(row["coco_annotation_license"]),
                })
                box_statuses[split][status] += 1
                if concept:
                    concepts[split][concept] += 1
            image["annotations"] = annotations
            stats["images"] += 1
            stats["boxes"] += len(annotations)
            if annotations and all(box["status"] == "matched" for box in annotations):
                stats["images_with_all_boxes_matched"] += 1
            if not annotations:
                stats["images_without_annotations"] += 1
            if image["size_bytes"] is not None:
                stats["images_with_known_size"] += 1
                stats["known_size_bytes"] += image["size_bytes"]
            if not image["s3_key"] and not image["source_url"]:
                stats["images_without_download_source"] += 1
            if source_url := image["source_url"]:
                if urls.setdefault(source_url, split) != split:
                    cross_split_urls.add(source_url)
            image_statuses[split][image["image_status"]] += 1
            write_line(output, image)

    all_concepts = set().union(*concepts.values())
    with (staging / "concepts.tsv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t")
        writer.writerow(("concept", *SPLITS, "total"))
        for concept in sorted(all_concepts, key=str.casefold):
            support = [concepts[split][concept] for split in SPLITS]
            writer.writerow((concept, *support, sum(support)))

    by_split = {}
    for split in SPLITS:
        stats = counts[split]
        n_matched = box_statuses[split]["matched"]
        by_split[split] = {
            "images": stats["images"], "boxes": stats["boxes"],
            "matched_boxes": n_matched,
            "matched_box_fraction": n_matched / stats["boxes"] if stats["boxes"] else None,
            "images_with_all_boxes_matched": stats["images_with_all_boxes_matched"],
            "images_without_annotations": stats["images_without_annotations"],
            "images_without_download_source": stats["images_without_download_source"],
            "distinct_matched_concepts": len(concepts[split]),
            "images_with_known_size": stats["images_with_known_size"],
            "known_size_bytes": stats["known_size_bytes"],
            "image_status_counts": dict(sorted(image_statuses[split].items())),
            "box_status_counts": dict(sorted(box_statuses[split].items())),
        }
    report = {
        "schema_version": 1, "sources": sources, "by_split": by_split,
        "cross_split_source_urls": len(cross_split_urls),
        "evaluation_concepts_absent_from_train": {
            split: sorted(concepts[split].keys() - concepts["train"].keys(), key=str.casefold)
            for split in ("val", "test")
        },
        "known_total_bytes": sum(stats["known_size_bytes"] for stats in counts.values()),
        "images_without_known_size": sum(stats["images"] - stats["images_with_known_size"] for stats in counts.values()),
    }
    with (staging / "audit.json").open("w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return report


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-dir", required=True, type=Path, help="Directory with coco_train.json, coco_val.json and coco_test.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--release", help="Pinned release date YYYY-MM-DD (required with default public S3 source)")
    parser.add_argument("--images-parquet", help="Local FathomNet images Parquet file or glob")
    parser.add_argument("--boxes-parquet", help="Local FathomNet boxes Parquet file or glob")
    size_options = parser.add_mutually_exclusive_group()
    size_options.add_argument("--objects-parquet", help="Local objects manifest Parquet file or glob")
    size_options.add_argument("--skip-size-estimate", action="store_true", help="Do not read the public S3 objects manifest")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    local = bool(args.images_parquet and args.boxes_parquet)
    if not local and (args.images_parquet or args.boxes_parquet or args.objects_parquet):
        parser.error("local mode requires both --images-parquet and --boxes-parquet")
    if args.release:
        try:
            if date.fromisoformat(args.release).isoformat() != args.release:
                raise ValueError
        except ValueError:
            parser.error("--release must be in YYYY-MM-DD format")
    if local:
        images, boxes, objects = (
            str(Path(args.images_parquet).resolve()), str(Path(args.boxes_parquet).resolve()),
            str(Path(args.objects_parquet).resolve()) if args.objects_parquet else None,
        )
    else:
        if not args.release:
            parser.error("pass a dated --release for S3 or supply two local Parquet inputs")
        images = f"{BUCKET}/metadata/images/release={args.release}/*.parquet"
        boxes = f"{BUCKET}/metadata/boxes/release={args.release}/*.parquet"
        objects = None if args.skip_size_estimate else OBJECTS
    try:
        import duckdb
    except ImportError:
        parser.exit(2, "Error: install DuckDB in the current environment with python -m pip install duckdb\n")

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".megalodon-audit-", dir=args.output_dir) as temp:
            staging = Path(temp)
            input_images, input_boxes = staging / "input_images.jsonl", staging / "input_boxes.jsonl"
            inputs = stage_coco(args.coco_dir, input_images, input_boxes)
            with duckdb.connect() as connection:
                if not local:
                    connection.execute("INSTALL httpfs")
                    connection.execute("LOAD httpfs")
                    connection.execute("SET s3_region = 'us-west-2'")
                connection.execute("SET temp_directory = ?", [str((staging / "duckdb_spill").resolve())])
                cursor = match_query(connection, input_images, input_boxes, images, boxes, objects)
                report = write_outputs(cursor, staging, {
                    "coco": inputs, "release": args.release,
                    "images_parquet": images, "boxes_parquet": boxes, "objects_parquet": objects,
                })
            for name in ("manifest.jsonl", "concepts.tsv", "audit.json"):
                os.replace(staging / name, args.output_dir / name)
    except (OSError, ValueError, OverflowError, duckdb.Error) as error:
        parser.exit(2, f"Error: {error}\n")

    for split, stats in report["by_split"].items():
        print(f"{split}: {stats['matched_boxes']}/{stats['boxes']} boxes matched; "
              f"{stats['images_with_all_boxes_matched']}/{stats['images']} images with all boxes matched")
    print(f"Audit written to {args.output_dir / 'audit.json'}")


if __name__ == "__main__":
    sys.exit(main())
