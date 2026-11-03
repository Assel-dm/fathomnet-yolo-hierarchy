"""Generate offline iNaturalist taxonomy HTML without loading images or models
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile


RANKS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")


def unique_object(pairs):
    """Do not silently change scientific labels or IDs on duplicate JSON keys."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Repeated JSON object key: {key}")
        result[key] = value
    return result


class JsonStream:
    """Bounded-buffer JSON reader for category arrays inside large split files.

    Large images/annotations arrays are decoded one record at a time and discarded.
    Only categories are retained. This avoids loading a 200 MB split into memory.
    """

    def __init__(self, stream, chunk_size=65536):
        self.stream = stream
        self.chunk_size = chunk_size
        self.buffer = ""
        self.position = 0
        self.eof = False
        self.decoder = json.JSONDecoder(object_pairs_hook=unique_object)

    def fill(self):
        self.buffer = self.buffer[self.position:]
        self.position = 0
        chunk = self.stream.read(self.chunk_size)
        if chunk:
            self.buffer += chunk
        else:
            self.eof = True

    def peek(self):
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if self.eof:
                return ""
            self.fill()

    def take(self, expected):
        actual = self.peek()
        if actual != expected:
            raise ValueError(f"Invalid JSON structure: expected {expected!r}, got {actual!r}")
        self.position += 1

    def value(self):
        if not self.peek():
            raise ValueError("Unexpected end of JSON")
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError as error:
                if self.eof:
                    raise ValueError(f"Invalid JSON value: {error.msg}") from error
                if len(self.buffer) - self.position > 16 * 1024 * 1024:
                    raise ValueError("A JSON record exceeds 16 MB; check the metadata format") from error
                self.fill()
                continue
            # A number may be split between chunks, e.g. 12 | 34 or 1 | e3.
            if type(value) in (int, float) and not self.eof:
                if end == len(self.buffer) or self.buffer[end] not in " \t\r\n,]}:":
                    self.fill()
                    continue
            self.position = end
            return value

    def array(self):
        self.take("[")
        if self.peek() == "]":
            self.take("]")
            return
        while True:
            yield self.value()
            next_char = self.peek()
            if next_char == "]":
                self.take("]")
                return
            self.take(",")

    def skip_value(self):
        if self.peek() == "[":
            for _ in self.array():
                pass
        else:
            self.value()


def read_categories(path):
    """Accept either categories.json (a list) or a full COCO-style split JSON."""
    with Path(path).open(encoding="utf-8-sig") as stream:
        reader = JsonStream(stream)
        if reader.peek() == "[":
            categories = list(reader.array())
        else:
            reader.take("{")
            seen = set()
            categories = None
            if reader.peek() != "}":
                while True:
                    key = reader.value()
                    if not isinstance(key, str) or key in seen:
                        raise ValueError("JSON object keys must be unique strings")
                    seen.add(key)
                    reader.take(":")
                    if key == "categories":
                        categories = list(reader.array())
                    else:
                        reader.skip_value()
                    if reader.peek() == "}":
                        break
                    reader.take(",")
            reader.take("}")
            if categories is None:
                raise ValueError("No categories array found in metadata JSON")
        if reader.peek():
            raise ValueError("Unexpected content after the JSON document")
    return validate_categories(categories)


def validate_categories(categories):
    if not isinstance(categories, list) or not categories:
        raise ValueError("Categories must be a nonempty list")
    result = {}
    for category in categories:
        if not isinstance(category, dict):
            raise ValueError("Each category must be an object")
        identifier = category.get("id")
        if type(identifier) is not int or identifier in result:
            raise ValueError(f"Invalid or duplicate category ID: {identifier!r}")
        for rank in RANKS:
            field = "name" if rank == "species" else rank
            value = category.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Category {identifier}: missing scientific label {field}")
        if category.get("common_name") is not None and not isinstance(category["common_name"], str):
            raise ValueError(f"Category {identifier}: common_name must be a string")
        result[identifier] = category
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_record(role, path):
    return {"role": role, "name": Path(path).name, "sha256": sha256_file(path)}


def finish_tree(tree):
    """Sort and summarize; preserve scientific names and raw category IDs."""
    ranks = Counter()
    counts = {"nodes": 0, "classes": 0, "internal_nodes": 0, "leaves": 0, "max_depth": 0}
    identifiers, class_ids = set(), set()
    labels = defaultdict(set)
    stack = [(tree, 0, ())]
    while stack:
        node, depth, parent_path = stack.pop()
        identifier = node["id"]
        if identifier in identifiers:
            raise ValueError(f"Repeated structural node ID: {identifier}")
        identifiers.add(identifier)
        counts["nodes"] += 1
        counts["max_depth"] = max(counts["max_depth"], depth)
        if depth:
            ranks[node["rank"]] += 1
            labels[(node["rank"], node["name"])].add(parent_path)
        node["children"].sort(key=lambda child: (child["name"].casefold(), child["id"]))
        if node["class_id"] is not None:
            if node["children"] or node["rank"] != "species" or node["class_id"] in class_ids:
                raise ValueError("Each species class must identify exactly one leaf")
            class_ids.add(node["class_id"])
            counts["classes"] += 1
        if node["children"]:
            counts["internal_nodes"] += 1
        else:
            counts["leaves"] += 1
            if node["class_id"] is None:
                raise ValueError("Every leaf must map to a species category")
        path = parent_path + ((node["rank"], node["name"]),)
        stack.extend((child, depth + 1, path) for child in node["children"])
    counts["ranks"] = [{"rank": rank, "nodes": ranks[rank]} for rank in RANKS]
    counts["homonymous_labels"] = sum(len(parents) > 1 for parents in labels.values())
    return counts


def build_inat19h(isa_path, categories):
    """Preserve all ISA node IDs and edges, attaching names from descendant species."""
    children, parents, node_ids = defaultdict(list), {}, set()
    for number, line in enumerate(Path(isa_path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"ISA line {number}: expected parent and child IDs")
        parent, child = parts
        if parent == child or child in parents:
            raise ValueError(f"ISA line {number}: repeated child or self-reference {child}")
        parents[child] = parent
        children[parent].append(child)
        node_ids.update((parent, child))
    roots = node_ids - set(parents)
    if len(roots) != 1:
        raise ValueError("ISA must define one root")
    root = next(iter(roots))
    paths = {}
    for identifier in node_ids:
        seen, chain, current = set(), [], identifier
        while current is not None:
            if current in seen:
                raise ValueError(f"Cycle in ISA at {current}")
            seen.add(current)
            chain.append(current)
            current = parents.get(current)
        if chain[-1] != root:
            raise ValueError(f"Disconnected ISA node: {identifier}")
        paths[identifier] = list(reversed(chain))
    leaves = node_ids - set(children)
    expected = {f"nat{identifier:04d}": identifier for identifier in categories}
    if leaves != set(expected):
        missing = sorted(set(expected) - leaves)
        extra = sorted(leaves - set(expected))
        raise ValueError(f"ISA/category leaves differ; missing={missing[:5]}, extra={extra[:5]}")
    names = {}
    for leaf in sorted(leaves):
        branch = paths[leaf][1:]
        if len(branch) != len(RANKS):
            raise ValueError(f"ISA species {leaf} must have seven ranks, found {len(branch)}")
        category = categories[expected[leaf]]
        for identifier, rank in zip(branch, RANKS):
            label = category["name" if rank == "species" else rank]
            value = (rank, label)
            if identifier in names and names[identifier] != value:
                raise ValueError(f"Category scientific labels disagree for ISA node {identifier}")
            names[identifier] = value
    if set(names) != node_ids - {root}:
        raise ValueError("ISA contains nodes outside the species paths")

    def build(identifier):
        rank, name = ("root", "All taxa") if identifier == root else names[identifier]
        category = categories[expected[identifier]] if identifier in leaves else None
        return {
            "id": identifier,
            "name": name,
            "rank": rank,
            "class_id": category["id"] if category is not None else None,
            "common_name": category.get("common_name") or "" if category is not None else "",
            "children": [build(child) for child in children[identifier]],
        }
    tree = build(root)
    return tree, finish_tree(tree)


def build_inat21(categories):
    """Keep homonymous taxa separate with parent-path-qualified structural IDs."""
    root = {"id": "inat21:root", "name": "All taxa", "rank": "root", "class_id": None,
            "common_name": "", "children": []}
    branches = {}
    identifier_paths = {}
    for category_id in sorted(categories):
        category = categories[category_id]
        parent, path = root, ()
        for rank in RANKS:
            label = category["name" if rank == "species" else rank]
            component = (rank, category_id, label) if rank == "species" else (rank, label)
            path += (component,)
            if path not in branches:
                if rank == "species":
                    identifier = f"inat21:species:{category_id}"
                else:
                    encoded = json.dumps(path, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                    identifier = f"inat21:{rank}:" + hashlib.sha256(encoded).hexdigest()[:20]
                if identifier in identifier_paths and identifier_paths[identifier] != path:
                    raise ValueError("Structural ID collision; input must be reviewed")
                identifier_paths[identifier] = path
                node = {"id": identifier, "name": label, "rank": rank,
                        "class_id": category_id if rank == "species" else None,
                        "common_name": category.get("common_name") or "" if rank == "species" else "",
                        "children": []}
                parent["children"].append(node)
                branches[path] = node
            parent = branches[path]
    return root, finish_tree(root)


def embedded_json(value):
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    for char, escaped in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                          ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        text = text.replace(char, escaped)
    return text


def generate(dataset, categories_path, output_path, isa_path=None, title=None, assets_dir=None):
    categories_path, output_path = Path(categories_path), Path(output_path)
    assets = Path(assets_dir) if assets_dir else Path(__file__).with_name("tree_explorer_assets")
    template_path, d3_path = assets / "template.html", assets / "d3.v7.9.0.min.js"
    inputs = [categories_path, template_path, d3_path]
    if isa_path:
        inputs.append(Path(isa_path))
    if output_path.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("Output must differ from all metadata and asset paths")
    categories = read_categories(categories_path)
    records = [source_record("Category metadata", categories_path)]
    if dataset == "inat19h":
        if not isa_path:
            raise ValueError("iNat19-H requires --isa")
        tree, summary = build_inat19h(isa_path, categories)
        label = "iNaturalist 2019-H"
        records.insert(0, source_record("ISA hierarchy", isa_path))
        notes = ["ISA edges and node IDs are preserved. Scientific labels are checked against every descendant species."]
    elif dataset == "inat21":
        tree, summary = build_inat21(categories)
        label = "iNaturalist 2021"
        notes = ["The seven rank fields define this tree. Homonymous labels remain separate when their parent paths differ.",
                 "Structural node IDs are local display IDs, not external taxon identifiers."]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    notes += ["The root is a navigation node. No missing rank or biological relationship is inferred.",
              "Only category metadata is retained; images, coordinates and annotation records are not embedded.",
              "Descendant species counts describe taxonomy coverage, not image counts or experimental results."]
    data = {"title": title or f"{label} taxonomy explorer", "dataset_label": label,
            "summary": summary, "provenance": {"sources": records, "notes": notes}, "tree": tree}
    template, d3 = template_path.read_text(encoding="utf-8"), d3_path.read_text(encoding="utf-8")
    if "https://d3js.org v7.9.0" not in d3[:200] or "</script" in d3.lower():
        raise ValueError("Expected the bundled D3 7.9.0 source")
    for token in ("__D3_SOURCE__", "__TREE_DATA__"):
        if template.count(token) != 1:
            raise ValueError(f"Template must contain exactly one {token} placeholder")
    html = template.replace("__D3_SOURCE__", d3).replace("__TREE_DATA__", embedded_json(data))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=output_path.parent,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(html)
        os.replace(temporary, output_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="dataset", required=True)
    for dataset in ("inat19h", "inat21"):
        command = subparsers.add_parser(dataset)
        command.add_argument("--categories", "--annotations", dest="categories", required=True, type=Path,
                             help="Categories list JSON or complete split JSON; only categories are retained")
        if dataset == "inat19h":
            command.add_argument("--isa", required=True, type=Path, help="Exact inaturalist19_isa.txt used by the project")
        command.add_argument("--output", required=True, type=Path, help="Standalone offline HTML output")
        command.add_argument("--title")
        command.add_argument("--assets-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        summary = generate(args.dataset, args.categories, args.output, getattr(args, "isa", None), args.title, args.assets_dir)
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"HTML written to {args.output.resolve()}")
    print(f"{summary['classes']} species; {summary['nodes']} nodes; "
          f"{summary['homonymous_labels']} labels under distinct parent paths.")


if __name__ == "__main__":
    main()
