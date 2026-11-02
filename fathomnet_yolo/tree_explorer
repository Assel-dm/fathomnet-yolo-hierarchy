"""Create a standalone, offline taxonomy explorer from COCO and frozen WoRMS JSON"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile


def read_json(path):
    """Read UTF-8 JSON and retain its exact byte hash for provenance."""
    path = Path(path)
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid UTF-8 JSON in {path.name}: {error}") from error
    return value, hashlib.sha256(raw).hexdigest()


def indexed_records(records, field, label, string_ids=False):
    """Reject duplicate IDs and implicit conversions such as bool to int."""
    if not isinstance(records, list):
        raise ValueError(f"{label} must be a JSON list")
    indexed = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Each {label} entry must be an object")
        identifier = record.get(field)
        valid = (
            isinstance(identifier, str) and bool(identifier.strip())
            if string_ids else type(identifier) is int
        )
        if not valid or identifier in indexed:
            raise ValueError(f"{label}: invalid or duplicate {field}={identifier!r}")
        indexed[identifier] = record
    return indexed


def prepare_data(coco, taxonomy, class_map=None, title="FathomNet taxonomy explorer"):
    """Join original classes by source_category_id; never infer names or parents."""
    if not isinstance(coco, dict) or not isinstance(taxonomy, dict):
        raise ValueError("COCO and taxonomy must both be JSON objects")
    if type(taxonomy.get("schema_version")) is not int or taxonomy["schema_version"] != 1:
        raise ValueError("Expected taxonomy schema_version=1")
    categories = indexed_records(coco.get("categories"), "id", "COCO categories")
    images = indexed_records(coco.get("images"), "id", "COCO images")
    annotations = indexed_records(coco.get("annotations"), "id", "COCO annotations")
    source_nodes = indexed_records(taxonomy.get("nodes"), "id", "Taxonomy nodes", True)
    classes = indexed_records(taxonomy.get("classes"), "class_id", "Taxonomy classes")
    if not categories or not source_nodes or not classes:
        raise ValueError("Categories, taxonomy nodes and classes must be nonempty")
    if sorted(classes) != list(range(len(classes))):
        raise ValueError("Taxonomy class_id values must be contiguous from zero")
    for category in categories.values():
        if not isinstance(category.get("name"), str) or not category["name"].strip():
            raise ValueError("COCO categories require nonempty names")

    children = defaultdict(list)
    roots = []
    for node_id, node in source_nodes.items():
        if not isinstance(node.get("name"), str) or not node["name"].strip():
            raise ValueError(f"Missing taxon name: {node_id}")
        if node.get("rank") is not None and not isinstance(node["rank"], str):
            raise ValueError(f"Invalid taxonomic rank: {node_id}")
        parent = node.get("parent")
        if parent is None:
            roots.append(node_id)
        elif not isinstance(parent, str) or parent not in source_nodes:
            raise ValueError(f"Unknown parent {parent!r} for {node_id}")
        else:
            children[parent].append(node_id)
    if len(roots) != 1:
        raise ValueError("Taxonomy must have exactly one root")
    for node_id in source_nodes:
        seen = set()
        current = node_id
        while current is not None:
            if current in seen:
                raise ValueError(f"Cycle in taxonomy at {current}")
            seen.add(current)
            current = source_nodes[current].get("parent")
        if roots[0] not in seen:
            raise ValueError(f"Disconnected taxonomy node: {node_id}")

    by_category = {}
    for class_id, entry in classes.items():
        category_id = entry.get("source_category_id")
        if type(category_id) is not int:
            raise ValueError(
                f"Class {class_id} has no integer source_category_id. "
                "Rebuild the taxonomy with the project builder; name guessing is disabled."
            )
        if category_id not in categories or category_id in by_category:
            raise ValueError(f"Unmapped or repeated source_category_id={category_id}")
        if entry.get("name") != categories[category_id]["name"]:
            raise ValueError(f"Original class name disagrees for COCO category {category_id}")
        node_id = entry.get("node_id")
        if not isinstance(node_id, str) or node_id not in source_nodes:
            raise ValueError(f"Unknown taxon for class {class_id}: {node_id}")
        if entry.get("accepted_name", source_nodes[node_id]["name"]) != source_nodes[node_id]["name"]:
            raise ValueError(f"Accepted name disagrees with taxon {node_id}")
        if entry.get("accepted_rank", source_nodes[node_id].get("rank")) != source_nodes[node_id].get("rank"):
            raise ValueError(f"Accepted rank disagrees with taxon {node_id}")
        by_category[category_id] = entry
    if set(by_category) != set(categories):
        raise ValueError("COCO categories and taxonomy source_category_id sets differ")

    if class_map is not None:
        mapping = indexed_records(class_map, "yolo_id", "Class map")
        if set(mapping) != set(classes):
            raise ValueError("Class map and taxonomy class IDs differ")
        for class_id, entry in classes.items():
            if type(mapping[class_id].get("category_id")) is not int:
                raise ValueError(f"Class map category_id must be an integer at class {class_id}")
            expected = (entry["source_category_id"], entry["name"])
            actual = (mapping[class_id].get("category_id"), mapping[class_id].get("name"))
            if actual != expected:
                raise ValueError(f"Class map disagrees with taxonomy at class {class_id}")

    counts = Counter()
    category_images = defaultdict(set)
    for annotation in annotations.values():
        category_id, image_id = annotation.get("category_id"), annotation.get("image_id")
        if type(category_id) is not int or category_id not in categories:
            raise ValueError(f"Unknown annotation category_id={category_id!r}")
        if type(image_id) is not int or image_id not in images:
            raise ValueError(f"Unknown annotation image_id={image_id!r}")
        counts[category_id] += 1
        category_images[category_id].add(image_id)

    originals = defaultdict(list)
    direct_images = defaultdict(set)
    for category_id in sorted(categories):
        category, entry = categories[category_id], by_category[category_id]
        node_id = entry["node_id"]
        accepted = source_nodes[node_id]["name"]
        originals[node_id].append({
            "coco_id": category_id,
            "yolo_class_id": entry["class_id"],
            "original_name": category["name"],
            "retained_name": accepted,
            "renamed": category["name"] != accepted,
            "annotations": counts[category_id],
            "images": len(category_images[category_id]),
        })
        direct_images[node_id].update(category_images[category_id])

    def build(node_id):
        source = source_nodes[node_id]
        child_ids = sorted(children[node_id], key=lambda key: (source_nodes[key]["name"].casefold(), key))
        branches = []
        image_support = set(direct_images[node_id])
        direct_count = sum(entry["annotations"] for entry in originals[node_id])
        total = direct_count
        for child_id in child_ids:
            child, child_images = build(child_id)
            branches.append(child)
            image_support.update(child_images)
            total += child["subtree_annotations"]
        return {
            "id": node_id,
            "name": source["name"],
            "rank": source.get("rank") or "",
            "aphia_id": source.get("aphia_id"),
            "originals": originals[node_id],
            "direct_annotations": direct_count,
            "subtree_annotations": total,
            "direct_images": len(direct_images[node_id]),
            "subtree_images": len(image_support),
            "children": branches,
        }, image_support

    tree, annotated_images = build(roots[0])
    original_ids = set(by_category[category_id]["node_id"] for category_id in categories)
    return {
        "title": title,
        "summary": {
            "nodes": len(source_nodes),
            "original_classes": len(classes),
            "original_nodes": len(original_ids),
            "added_nodes": len(source_nodes) - len(original_ids),
            "renamed_classes": sum(item["renamed"] for entries in originals.values() for item in entries),
            "internal_original_nodes": sum(bool(children[node_id]) for node_id in original_ids),
            "annotations": len(annotations),
            "images": len(images),
            "annotated_images": len(annotated_images),
        },
        "tree": tree,
    }


def embedded_json(value):
    """Keep dataset names from terminating the inert JSON script element."""
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    for char, escaped in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                          ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        text = text.replace(char, escaped)
    return text


def generate(coco_path, taxonomy_path, output_path, class_map_path=None,
             title="FathomNet taxonomy explorer", assets_dir=None):
    """Generate one portable HTML document; inputs are only read."""
    coco_path, taxonomy_path, output_path = map(Path, (coco_path, taxonomy_path, output_path))
    assets_dir = Path(assets_dir) if assets_dir else Path(__file__).with_name("tree_explorer_assets")
    template_path = assets_dir / "template.html"
    library_path = assets_dir / "d3.v7.9.0.min.js"
    input_paths = [coco_path, taxonomy_path, template_path, library_path]
    if class_map_path:
        input_paths.append(Path(class_map_path))
    if output_path.resolve() in {path.resolve() for path in input_paths}:
        raise ValueError("Output path must differ from every input and asset path")
    coco, coco_hash = read_json(coco_path)
    taxonomy, taxonomy_hash = read_json(taxonomy_path)
    class_map, map_hash = read_json(class_map_path) if class_map_path else (None, None)
    data = prepare_data(coco, taxonomy, class_map, title)
    data["provenance"] = {
        "coco_name": coco_path.name,
        "coco_sha256": coco_hash,
        "taxonomy_name": taxonomy_path.name,
        "taxonomy_sha256": taxonomy_hash,
        "class_map_sha256": map_hash,
        "count_note": "Raw COCO records before detection preprocessing; duplicates and out-of-bounds boxes are not filtered here.",
    }
    template = template_path.read_text(encoding="utf-8")
    library = library_path.read_text(encoding="utf-8")
    if "https://d3js.org v7.9.0" not in library[:200]:
        raise ValueError("Expected the bundled D3 7.9.0 library")
    if "</script" in library.lower():
        raise ValueError("Unexpected closing script tag in D3 source")
    for token in ("__D3_SOURCE__", "__TREE_DATA__"):
        if template.count(token) != 1:
            raise ValueError(f"Template must contain exactly one {token} placeholder")
    html = template.replace("__D3_SOURCE__", library).replace("__TREE_DATA__", embedded_json(data))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=output_path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(html)
        os.replace(temporary, output_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return data["summary"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", required=True, type=Path, help="Original annotated COCO JSON")
    parser.add_argument("--taxonomy", required=True, type=Path, help="Frozen project taxonomy JSON")
    parser.add_argument("--class-map", type=Path, help="Optional prepared class_map.json consistency check")
    parser.add_argument("--output", type=Path, default=Path("reports/taxonomy/tree.html"), help="Output HTML file")
    parser.add_argument("--title", default="FathomNet taxonomy explorer")
    parser.add_argument("--assets-dir", type=Path, help="Override the sibling tree_explorer_assets directory")
    args = parser.parse_args(argv)
    try:
        summary = generate(args.coco, args.taxonomy, args.output, args.class_map, args.title, args.assets_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"HTML written to {args.output.resolve()}")
    print(f"{summary['original_classes']} COCO classes; {summary['nodes']} taxonomic nodes; "
          f"{summary['added_nodes']} added ancestors; {summary['renamed_classes']} different accepted names.")


if __name__ == "__main__":
    main()
