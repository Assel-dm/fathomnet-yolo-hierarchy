"""Generate an offline Megalodon tree from the final frozen taxonomy and export."""

import argparse
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from fathomnet_yolo.tree_explorer import embedded_json, prepare_data, read_json

SPLITS = ("train", "val", "test")


def count_value(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def prepare_megalodon(
    taxonomy,
    class_map=None,
    export_summary=None,
    resolution=None,
    title="Megalodon taxonomy explorer",
):
    """Reuse the original explorer's topology checks and attach final box counts."""
    if not isinstance(taxonomy, dict) or not isinstance(taxonomy.get("classes"), list):
        raise TypeError("Expected a frozen taxonomy with classes")
    for value, label in (
        (export_summary, "export_summary"),
        (resolution, "concept_resolution"),
    ):
        if value is not None and not isinstance(value, dict):
            raise TypeError(f"{label} must be a JSON object")
    categories = [
        {"id": entry.get("source_category_id"), "name": entry.get("name")}
        for entry in taxonomy["classes"]
    ]
    data = prepare_data(
        {"categories": categories, "images": [], "annotations": []},
        taxonomy,
        class_map,
        title,
    )
    classes = taxonomy["classes"]
    by_name = {entry["name"]: entry for entry in classes}
    if len(by_name) != len(classes) or len(
        {entry["node_id"] for entry in classes}
    ) != len(classes):
        raise ValueError("Each dataset class requires a unique name and taxon node")
    source_nodes = {node["id"]: node for node in taxonomy["nodes"]}
    counts = None
    images_by_split = None
    notes = [
        "Blue nodes are the classes retained in the prepared dataset. Grey nodes are added taxonomic ancestors.",
        "The graph is the frozen training taxonomy, with its retained ranks and target taxa.",
        "Distinct image counts by class or subtree are unavailable in these metadata files.",
    ]
    if export_summary is not None:
        if export_summary.get("status") not in {"ready", "ready_with_skipped_images"}:
            raise ValueError("Export summary must describe a completed dataset export")
        if export_summary.get("classes") != len(classes):
            raise ValueError(
                "Export summary class count disagrees with the frozen taxonomy"
            )
        counts = {name: {split: 0 for split in SPLITS} for name in by_name}
        images_by_split = {}
        for split in SPLITS:
            stats = export_summary["splits"][split]
            exported_counts = stats.get("exported_class_counts")
            if not isinstance(exported_counts, dict):
                raise TypeError(f"Final exported_class_counts are required for {split}")
            if set(exported_counts) - set(by_name):
                raise ValueError(
                    f"Exported {split} classes are absent from the taxonomy"
                )
            for name, value in exported_counts.items():
                counts[name][split] = count_value(value, f"{split} count of {name}")
            total = count_value(stats.get("exported_boxes"), f"{split} exported_boxes")
            if sum(exported_counts.values()) != total:
                raise ValueError(
                    f"{split} exported box total disagrees with its class counts"
                )
            images_by_split[split] = count_value(
                stats.get("exported_images"), f"{split} exported_images"
            )
        if any(counts[name]["train"] == 0 for name in counts):
            raise ValueError(
                "Every predicted class must occur in the exported training split"
            )
        notes.append(
            "All displayed box counts come from the final exported dataset, after selection and download."
        )
        failures = count_value(
            export_summary.get("download_failures", 0), "download_failures"
        )
        if failures:
            notes.append(
                f"The final export excludes {failures} failed image downloads."
            )
    else:
        notes.append("Box counts were not provided.")

    aliases = defaultdict(set)
    if resolution is not None:
        concepts = resolution.get("concepts")
        if not isinstance(concepts, dict):
            raise ValueError(
                "Expected the concept resolution report's concepts mapping"
            )
        if export_summary is not None:
            manifest = export_summary.get("inputs", {}).get("manifest_sha256")
            if not manifest or resolution.get("manifest_sha256") != manifest:
                raise ValueError(
                    "Concept resolution and export summary use different audit manifests"
                )
        for original, record in concepts.items():
            if (
                record.get("status") != "resolved"
                or record.get("accepted_name") not in by_name
            ):
                continue
            entry = by_name[record["accepted_name"]]
            node = source_nodes[entry["node_id"]]
            if record.get("aphia_id") != node.get("aphia_id"):
                raise ValueError(f"Conflicting accepted AphiaID for {original!r}")
            if not isinstance(original, str) or not original.strip():
                raise ValueError("Source concept names must be nonempty strings")
            aliases[entry["name"]].add(original)
        notes.append(
            "Source aliases come from the concept audit. Their pre-export support counts are not used."
        )

    def attach(node):
        class_counts = (
            counts[node["originals"][0]["original_name"]].copy()
            if counts is not None and node["originals"]
            else None
        )
        if counts is not None and class_counts is None:
            class_counts = {split: 0 for split in SPLITS}
        total_counts = class_counts.copy() if class_counts is not None else None
        for record in node["originals"]:
            name = record["original_name"]
            record["aliases"] = sorted(aliases[name] - {name}, key=str.casefold)
            record["renamed"] = bool(record["aliases"])
            record["boxes_by_split"] = (
                class_counts.copy() if class_counts is not None else None
            )
            record["annotations"] = (
                sum(class_counts.values()) if class_counts is not None else None
            )
            record["images"] = None
        for child in node["children"]:
            attach(child)
            if total_counts is not None:
                for split in SPLITS:
                    total_counts[split] += child["subtree_boxes_by_split"][split]
        node["boxes_by_split"] = class_counts
        node["subtree_boxes_by_split"] = total_counts
        node["direct_annotations"] = (
            sum(class_counts.values()) if class_counts is not None else None
        )
        node["subtree_annotations"] = (
            sum(total_counts.values()) if total_counts is not None else None
        )
        node["direct_images"] = None
        node["subtree_images"] = None

    attach(data["tree"])
    summary = data["summary"]
    summary["renamed_classes"] = sum(bool(aliases[name] - {name}) for name in by_name)
    summary["annotations"] = data["tree"]["subtree_annotations"]
    summary["images"] = (
        sum(images_by_split.values()) if images_by_split is not None else None
    )
    summary["annotated_images"] = None
    summary["images_by_split"] = images_by_split
    summary["boxes_by_split"] = data["tree"]["subtree_boxes_by_split"]
    summary["rank_counts"] = dict(
        Counter(node.get("rank") or "unranked" for node in source_nodes.values())
    )
    summary["classes_by_rank"] = dict(
        Counter(
            source_nodes[entry["node_id"]].get("rank") or "unranked"
            for entry in classes
        )
    )
    return data, notes


def generate(
    taxonomy_path,
    output_path,
    class_map_path=None,
    export_summary_path=None,
    concept_resolution_path=None,
    title="Megalodon taxonomy explorer",
    assets_dir=None,
):
    """Write one self-contained document while preserving all input files."""
    assets = (
        Path(assets_dir)
        if assets_dir
        else Path(__file__).with_name("megalodon_tree_explorer_assets")
    )
    paths = {
        "taxonomy": Path(taxonomy_path),
        **({"class_map": Path(class_map_path)} if class_map_path else {}),
        **(
            {"export_summary": Path(export_summary_path)} if export_summary_path else {}
        ),
        **(
            {"concept_resolution": Path(concept_resolution_path)}
            if concept_resolution_path
            else {}
        ),
    }
    output = Path(output_path)
    template_path, library_path = assets / "template.html", assets / "d3.v7.9.0.min.js"
    protected = list(paths.values()) + [template_path, library_path]
    if any(
        output.resolve() == path.resolve()
        or (output.exists() and os.path.samefile(output, path))
        for path in protected
    ):
        raise ValueError("Output must differ from every input and resource")
    contents, sources = {}, []
    for role, path in paths.items():
        contents[role], digest = read_json(path)
        sources.append({"role": role, "name": path.name, "sha256": digest})
    taxonomy = contents["taxonomy"]
    provenance = taxonomy.get("provenance", {})
    for role in ("class_map", "concept_resolution"):
        if role in contents and provenance.get(role, {}).get("sha256"):
            actual = next(
                source["sha256"] for source in sources if source["role"] == role
            )
            if actual != provenance[role]["sha256"]:
                raise ValueError(
                    f"The {role} file differs from the file used to build this taxonomy"
                )
    data, notes = prepare_megalodon(
        taxonomy,
        contents.get("class_map"),
        contents.get("export_summary"),
        contents.get("concept_resolution"),
        title,
    )
    data["provenance"] = {"sources": sources, "notes": notes}
    template = template_path.read_text(encoding="utf-8")
    library = library_path.read_text(encoding="utf-8")
    if "https://d3js.org v7.9.0" not in library[:200] or "</script" in library.lower():
        raise ValueError("Expected the bundled D3 7.9.0 source")
    for token in ("__D3_SOURCE__", "__TREE_DATA__"):
        if template.count(token) != 1:
            raise ValueError(f"Expected one {token} placeholder")
    html = template.replace("__D3_SOURCE__", library).replace(
        "__TREE_DATA__", embedded_json(data)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=output.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(html)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return data["summary"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", required=True, type=Path)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--export-summary", type=Path)
    parser.add_argument("--concept-resolution", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="Megalodon taxonomy explorer")
    args = parser.parse_args()
    try:
        summary = generate(
            args.taxonomy,
            args.output,
            args.class_map,
            args.export_summary,
            args.concept_resolution,
            args.title,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"HTML written to {args.output.resolve()}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
