"""Generate standalone hierarchical sunburst charts for FGVC2025 and Megalodon."""

import argparse
import os
import tempfile
from collections import Counter
from pathlib import Path

from fathomnet_yolo.megalodon_tree_explorer import prepare_megalodon
from fathomnet_yolo.tree_explorer import embedded_json, prepare_data, read_json


def prepare_sunburst(
    dataset,
    taxonomy,
    class_map=None,
    coco=None,
    export_summary=None,
    concept_resolution=None,
    title=None,
):
    """Attach source-appropriate counts to the shared biological tree format."""
    if dataset == "fgvc":
        title = title or "FathomNet FGVC2025 · Hierarchical distribution"
        if coco is None:
            data, notes = prepare_megalodon(taxonomy, class_map, title=title)
            notes = [
                "The chart follows the frozen FGVC2025 training taxonomy.",
                "One unit represents one learned category in Categories mode, including classes on internal nodes.",
                "No annotation counts were supplied, so Boxes mode is unavailable.",
            ]
        else:
            data = prepare_data(coco, taxonomy, class_map, title)
            notes = [
                "Box counts are raw COCO annotation records before YOLO preprocessing.",
                "Duplicate or out-of-bounds annotations are retained in these raw counts.",
                "One unit represents one learned category in Categories mode, including classes on internal nodes.",
            ]
        data["dataset"] = "fgvc2025"
    elif dataset == "megalodon":
        title = title or "Megalodon · Hierarchical distribution"
        data, notes = prepare_megalodon(
            taxonomy, class_map, export_summary, concept_resolution, title
        )
        # The shared adapter's blue/grey legend belongs to the node-link explorer.
        notes = [note for note in notes if not note.startswith("Blue nodes are")]
        data["dataset"] = "megalodon"
    else:
        raise ValueError("dataset must be fgvc or megalodon")
    classes_by_rank = Counter()

    def annotate(node):
        learned = node["originals"]
        if len(learned) > 1:
            raise ValueError(
                "Multiple detector classes mapped to one taxon must be resolved before plotting"
            )
        for record in learned:
            record.setdefault("aliases", [])
            classes_by_rank[node["rank"] or "unranked"] += 1
        node["direct_classes"] = len(learned)
        node["subtree_classes"] = len(learned)
        for child in node["children"]:
            annotate(child)
            node["subtree_classes"] += child["subtree_classes"]

    annotate(data["tree"])
    if data["tree"]["subtree_classes"] != data["summary"]["original_classes"]:
        raise ValueError("Class totals disagree with the tree membership")
    data["summary"]["classes_by_rank"] = dict(classes_by_rank)
    notes.append(
        "An internal learned class contributes its own direct sector, separately from descendant classes. "
        "Direct sectors are display elements; the frozen biological taxonomy is unchanged."
    )
    notes.append(
        "Colours identify phyla; the shade varies along their taxonomic branches."
    )
    notes.append(
        "Rings follow tree depth. Missing or additional taxonomic ranks are preserved as recorded."
    )
    return data, notes


def generate(
    dataset,
    taxonomy_path,
    output_path,
    class_map_path=None,
    coco_path=None,
    export_summary_path=None,
    concept_resolution_path=None,
    title=None,
    assets_dir=None,
):
    """Write one portable HTML document without reading any image or checkpoint."""
    assets = (
        Path(assets_dir) if assets_dir else Path(__file__).with_name("sunburst_assets")
    )
    paths = {"taxonomy": Path(taxonomy_path)}
    optional = {
        "class_map": class_map_path,
        "coco": coco_path,
        "export_summary": export_summary_path,
        "concept_resolution": concept_resolution_path,
    }
    paths.update(
        {name: Path(path) for name, path in optional.items() if path is not None}
    )
    if dataset == "fgvc" and (export_summary_path or concept_resolution_path):
        raise ValueError(
            "Use --coco for FGVC counts. Export summaries and concept resolution apply to Megalodon."
        )
    if dataset == "megalodon" and coco_path:
        raise ValueError(
            "Use the final export summary for Megalodon rather than the original single-class COCO"
        )
    output = Path(output_path)
    template_path, library_path = assets / "template.html", assets / "d3.v7.9.0.min.js"
    for path in list(paths.values()) + [template_path, library_path]:
        if output.resolve() == path.resolve() or (
            output.exists() and path.exists() and os.path.samefile(output, path)
        ):
            raise ValueError("Output must differ from every source and resource")
    contents, sources = {}, []
    for role, path in paths.items():
        contents[role], checksum = read_json(path)
        sources.append({"role": role, "name": path.name, "sha256": checksum})
    provenance = contents["taxonomy"].get("provenance", {})
    for role in ("class_map", "concept_resolution"):
        recorded = provenance.get(role, {}).get("sha256")
        if role in contents and recorded:
            actual = next(
                source["sha256"] for source in sources if source["role"] == role
            )
            if actual != recorded:
                raise ValueError(
                    f"The {role} file differs from the source used to construct this taxonomy"
                )
    data, notes = prepare_sunburst(
        dataset,
        contents["taxonomy"],
        contents.get("class_map"),
        contents.get("coco"),
        contents.get("export_summary"),
        contents.get("concept_resolution"),
        title,
    )
    data["provenance"] = {"sources": sources, "notes": notes}
    template = template_path.read_text(encoding="utf-8")
    library = library_path.read_text(encoding="utf-8")
    if "https://d3js.org v7.9.0" not in library[:200] or "</script" in library.lower():
        raise ValueError("Expected bundled D3 7.9.0")
    for token in ("__D3_SOURCE__", "__TREE_DATA__"):
        if template.count(token) != 1:
            raise ValueError(f"Expected exactly one {token}")
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
    commands = parser.add_subparsers(dest="dataset", required=True)
    fgvc = commands.add_parser(
        "fgvc", help="FGVC2025 taxonomy with optional raw COCO annotation counts"
    )
    fgvc.add_argument("--coco", type=Path)
    megalodon = commands.add_parser(
        "megalodon", help="Final multiclass Megalodon dataset"
    )
    megalodon.add_argument("--export-summary", type=Path)
    megalodon.add_argument("--concept-resolution", type=Path)
    for command in (fgvc, megalodon):
        command.add_argument("--taxonomy", type=Path, required=True)
        command.add_argument("--class-map", type=Path)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--title")
    args = parser.parse_args()
    try:
        summary = generate(
            args.dataset,
            args.taxonomy,
            args.output,
            args.class_map,
            getattr(args, "coco", None),
            getattr(args, "export_summary", None),
            getattr(args, "concept_resolution", None),
            args.title,
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"HTML written to {args.output.resolve()}")
    print(
        f"{summary['original_classes']} learned classes, {summary['nodes']} taxonomic nodes."
    )


if __name__ == "__main__":
    main()
