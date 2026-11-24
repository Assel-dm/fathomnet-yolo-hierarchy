"""Resolve Megalodon concepts with WoRMS and build the existing YOLO taxonomy format."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote

from fathomnet_yolo.build_taxonomy import (
    WormsClient,
    build_taxonomy,
    flatten_classification,
    resolve_name,
    validate_class_map,
)

SPLITS = ("train", "val", "test")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def count_concepts(manifest):
    """Count candidate boxes on fully matched images, before applying class filters."""
    counts = {split: Counter() for split in SPLITS}
    checksum = hashlib.sha256()
    with manifest.open("rb") as lines:
        for line in lines:
            checksum.update(line)
            image = json.loads(line)
            boxes = image["annotations"]
            if image["image_status"] == "dimensions_mismatch" or not boxes:
                continue
            if not (image.get("s3_key") or image.get("source_url")):
                continue
            if all(box["status"] == "matched" for box in boxes):
                counts[image["split"]].update(box["concept"] for box in boxes)
    return counts, checksum.hexdigest()


def lookup_concept(name, client, override):
    """Record unresolved names while allowing service failures to stop the run."""
    try:
        accepted = resolve_name(name, client, override)
    except ValueError as error:
        if str(error).startswith(("No unique exact WoRMS taxon", "WoRMS accepted record could not be verified")):
            return {"status": "unresolved", "reason": str(error)}
        raise
    except HTTPError as error:
        name_endpoint = f"AphiaRecordsByName/{quote(name, safe='')}?"
        if override is None and error.code == 404 and name_endpoint in error.url:
            return {"status": "unresolved", "reason": "No record at the WoRMS name endpoint"}
        raise
    return {
        "status": "resolved",
        "accepted_name": accepted["scientificname"],
        "aphia_id": accepted["AphiaID"],
        "rank": accepted["rank"].lower(),
        "reason": override["reason"] if override else "Exact source name with one accepted AphiaID",
    }


def resolve_concepts(manifest, output_dir, client, overrides, minimum):
    counts, manifest_sha = count_concepts(manifest)
    names = sorted(set().union(*counts.values()), key=lambda name: (-counts["train"][name], name))
    if not names:
        raise ValueError("No concepts on fully matched images. Check audit.json before proceeding.")
    if not isinstance(overrides, dict):
        raise ValueError("Overrides must map original concepts to a null or {aphia_id, reason}")
    unexpected = set(overrides) - set(names)
    if unexpected:
        raise ValueError(f"Overrides absent from the concept inventory: {sorted(unexpected)}")

    records = {}
    identifiers_by_name = defaultdict(set)
    train_by_id = Counter()
    for index, name in enumerate(names, 1):
        if name in overrides and overrides[name] is None:
            record = {"status": "excluded", "reason": "Explicit null in the overrides file"}
        else:
            record = lookup_concept(name, client, overrides.get(name))
        record["boxes"] = {split: counts[split][name] for split in SPLITS}
        records[name] = record
        if record["status"] == "resolved":
            identifiers_by_name[record["accepted_name"].casefold()].add(record["aphia_id"])
            train_by_id[record["aphia_id"]] += counts["train"][name]
        if index % 100 == 0 or index == len(names):
            print(f"Checked {index}/{len(names)} WoRMS concepts", flush=True)

    mapping = {}
    has_lineage = {}
    for name, record in records.items():
        mapping[name] = None
        if record["status"] != "resolved":
            continue
        identifier = record["aphia_id"]
        if len(identifiers_by_name[record["accepted_name"].casefold()]) > 1:
            record["status"] = "ambiguous_accepted_name"
        elif train_by_id[identifier] < minimum:
            record["status"] = "insufficient_train_support"
        else:
            if identifier not in has_lineage:
                try:
                    lineage = client.get(f"AphiaClassificationByAphiaID/{identifier}")
                except HTTPError as error:
                    if error.code != 404:
                        raise
                    lineage = None
                if lineage:
                    flatten_classification(lineage, identifier)
                has_lineage[identifier] = bool(lineage)
            if has_lineage[identifier]:
                mapping[name] = record["accepted_name"]
            else:
                record["status"] = "missing_lineage"
    report = {
        "manifest": manifest.name,
        "manifest_sha256": manifest_sha,
        "minimum_train_boxes": minimum,
        "raw_concepts": len(records),
        "selected_classes": len(set(mapping.values()) - {None}),
        "statuses": dict(Counter(record["status"] for record in records.values())),
        "concepts": records,
    }
    write_json(output_dir / "concept_resolution.json", report)
    write_json(output_dir / "concept_map.json", mapping)
    print(f"Selected {report['selected_classes']} canonical classes. Review {output_dir / 'concept_resolution.json'}.")
    if not report["selected_classes"]:
        raise ValueError("No train-supported taxonomic classes. Check the concept resolution report.")


def build_export_taxonomy(dataset_dir, resolution_path, client, output):
    """Reuse the same rank policy and class order as the FGVC2025 taxonomy builder."""
    categories_path = dataset_dir / "categories_coco.json"
    class_map_path = dataset_dir / "class_map.json"
    categories = read_json(categories_path)["categories"]
    class_map = read_json(class_map_path)
    resolution = read_json(resolution_path)
    export_inputs = read_json(dataset_dir / "export_inputs.json")
    if resolution["manifest_sha256"] != export_inputs["manifest_sha256"]:
        raise ValueError("The resolution and YOLO dataset come from different audit manifests.")
    records = resolution["concepts"]
    ids_by_name = defaultdict(set)
    for record in records.values():
        if record["status"] == "resolved":
            ids_by_name[record["accepted_name"]].add(record["aphia_id"])
    overrides = {}
    for category in categories:
        name = category["name"]
        identifiers = ids_by_name[name]
        if len(identifiers) != 1:
            raise ValueError(
                f"Expected one selected AphiaID for class {name!r}; check the resolution and dataset paths."
            )
        overrides[name] = {"aphia_id": next(iter(identifiers)), "reason": "Accepted AphiaID from the concept audit"}
    result = build_taxonomy(
        categories,
        client,
        overrides=overrides,
        source={"file_name": categories_path.name, "sha256": hashlib.sha256(categories_path.read_bytes()).hexdigest()},
    )
    validate_class_map(result, class_map)
    for key, path in (("class_map", class_map_path), ("concept_resolution", resolution_path)):
        result["provenance"][key] = {"file_name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    write_json(output, result)
    print(f"Saved {len(result['classes'])} classes and {len(result['nodes'])} nodes to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    resolve = commands.add_parser("resolve", help="Map observed FathomNet concepts to accepted WoRMS names")
    resolve.add_argument("--manifest", type=Path, required=True)
    resolve.add_argument("--output-dir", type=Path, required=True)
    resolve.add_argument("--overrides", type=Path)
    resolve.add_argument("--min-train-boxes", type=int, default=1)
    taxonomy = commands.add_parser("taxonomy", help="Generate the taxonomy for an exported YOLO dataset")
    taxonomy.add_argument("--dataset-dir", type=Path, required=True)
    taxonomy.add_argument("--resolution", type=Path, required=True)
    taxonomy.add_argument("--output", type=Path, required=True)
    for command in (resolve, taxonomy):
        command.add_argument("--cache", type=Path, required=True)
        command.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    client = WormsClient(args.cache, offline=args.offline)
    if args.command == "resolve":
        if args.min_train_boxes < 1:
            parser.error("--min-train-boxes must be at least 1")
        overrides = read_json(args.overrides) if args.overrides else {}
        resolve_concepts(args.manifest, args.output_dir, client, overrides, args.min_train_boxes)
    else:
        build_export_taxonomy(args.dataset_dir, args.resolution, client, args.output)


if __name__ == "__main__":
    main()
