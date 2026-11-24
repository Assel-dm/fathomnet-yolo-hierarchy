"""Build and freeze a WoRMS taxonomy from COCO category names."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from fathomnet_yolo.taxonomy import Taxonomy

BASE_URL = "https://www.marinespecies.org/rest"
MAIN_RANKS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")


class WormsClient:
    """Cache source responses with URLs, retrieval timestamps and content hashes."""

    def __init__(self, cache_dir, offline=False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.sources = {}

    def get(self, endpoint):
        """Read one cached response or query the official HTTPS API."""
        url = f"{BASE_URL}/{endpoint}"
        key = hashlib.sha256(url.encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["url"] != url:
                raise ValueError(f"Cache URL mismatch: {path}")
            if hashlib.sha256(json.dumps(record["response"], sort_keys=True).encode()).hexdigest() != record["sha256"]:
                raise ValueError(f"Cache content hash mismatch: {path}")
        else:
            if self.offline:
                raise ValueError(f"Offline cache is missing {url}")
            request = Request(url, headers={"Accept": "application/json", "User-Agent": "FathomNet-YOLO-research/1.0"})
            for attempt in range(3):
                try:
                    with urlopen(request, timeout=45) as response:
                        payload = response.read()
                        data = json.loads(payload) if payload else None
                    break
                except HTTPError as error:
                    if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                        raise
                    time.sleep(2 ** (attempt + 1))
                except (URLError, TimeoutError):
                    if attempt == 2:
                        raise
                    time.sleep(2 ** (attempt + 1))
            record = {
                "url": url,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
                "response": data,
            }
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            time.sleep(0.1)
        self.sources[url] = {field: record[field] for field in ("url", "retrieved_at_utc", "sha256")}
        self.sources[url]["cache_file"] = path.name
        return record["response"]


def resolve_name(name, client, override=None):
    """Return an accepted record, rejecting uncertain matches and silent aliases."""
    if override is not None:
        if not isinstance(override, dict) or type(override.get("aphia_id")) is not int or not override.get("reason"):
            raise ValueError(f"Override for {name!r} needs aphia_id and a documented reason")
        records = [client.get(f"AphiaRecordByAphiaID/{override['aphia_id']}")]
    else:
        records = (
            client.get(f"AphiaRecordsByName/{quote(name, safe='')}?like=false&marine_only=false&extant_only=false")
            or []
        )
        records = [record for record in records if record.get("scientificname", "").casefold() == name.casefold()]
    accepted_ids = {record.get("valid_AphiaID") for record in records if record}
    if len(accepted_ids) != 1 or None in accepted_ids:
        candidates = [
            {key: record.get(key) for key in ("AphiaID", "scientificname", "status", "valid_AphiaID", "valid_name")}
            for record in records
            if record
        ]
        raise ValueError(f"No unique exact WoRMS taxon for {name!r}; provide an override. Candidates: {candidates}")
    accepted_id = accepted_ids.pop()
    accepted = next(
        (record for record in records if record["AphiaID"] == accepted_id and record.get("status") == "accepted"), None
    )
    if accepted is None:
        accepted = client.get(f"AphiaRecordByAphiaID/{accepted_id}")
    if not accepted or accepted.get("status") != "accepted" or accepted["AphiaID"] != accepted_id:
        raise ValueError(f"WoRMS accepted record could not be verified for {name!r}")
    return accepted


def flatten_classification(classification, accepted_id):
    """Flatten the nested WoRMS ``child`` chain into a list and verify its terminal taxon."""
    lineage, seen = [], set()
    node = classification
    while node:
        identifier = node.get("AphiaID")
        if type(identifier) is not int or identifier in seen or not node.get("scientificname") or not node.get("rank"):
            raise ValueError(f"Malformed WoRMS classification for {accepted_id}")
        seen.add(identifier)
        lineage.append(node)
        node = node.get("child")
    if not lineage or lineage[-1]["AphiaID"] != accepted_id:
        raise ValueError(f"WoRMS lineage does not end at accepted AphiaID {accepted_id}")
    return lineage


def build_taxonomy(categories, client, overrides=None, source=None):
    """Resolve COCO categories, then retain seven principal ranks and target taxa.

    Extra-rank targets (for example subfamilies) remain real nodes and are kept in every relevant descendant path. Such
    paths extend the seven-rank FGVC convention and may change edge distances; this policy is saved explicitly Original
    dataset labels and category IDs remain in ``classes``
    """
    overrides = overrides or {}
    if not categories or any(type(category.get("id")) is not int for category in categories):
        raise ValueError("COCO categories must have integer IDs")
    categories = sorted(categories, key=lambda category: category["id"])
    if len({category["id"] for category in categories}) != len(categories):
        raise ValueError("COCO category IDs must be unique")
    names = [category.get("name") for category in categories]
    if any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("COCO category names must be unique nonempty strings")
    if set(overrides) - set(names):
        raise ValueError("Overrides contain labels absent from the COCO categories")
    resolved, lineages = [], []
    for category in categories:
        accepted = resolve_name(category["name"], client, overrides.get(category["name"]))
        resolved.append(accepted)
        lineages.append(
            flatten_classification(
                client.get(f"AphiaClassificationByAphiaID/{accepted['AphiaID']}"), accepted["AphiaID"]
            )
        )
    target_ids = {record["AphiaID"] for record in resolved}
    retained = target_ids | {
        node["AphiaID"] for lineage in lineages for node in lineage if node["rank"].lower() in MAIN_RANKS
    }
    retained.update(lineage[0]["AphiaID"] for lineage in lineages)
    nodes = {}
    for lineage in lineages:
        parent = None
        for taxon in lineage:
            identifier = taxon["AphiaID"]
            if identifier not in retained:
                continue
            node_id = f"worms:{identifier}"
            node = {
                "id": node_id,
                "parent": parent,
                "name": taxon["scientificname"],
                "rank": taxon["rank"].lower(),
                "aphia_id": identifier,
            }
            if node_id in nodes and nodes[node_id] != node:
                raise ValueError(f"Conflicting WoRMS ancestry while constructing {node_id}")
            nodes[node_id] = node
            parent = node_id
    result = {
        "schema_version": 1,
        "distance_mode": "edge_count",
        "nodes": sorted(nodes.values(), key=lambda node: node["aphia_id"]),
        "classes": [
            {
                "class_id": index,
                "name": category["name"],
                "node_id": f"worms:{accepted['AphiaID']}",
                "source_category_id": category["id"],
                "accepted_name": accepted["scientificname"],
                "accepted_rank": accepted["rank"].lower(),
            }
            for index, (category, accepted) in enumerate(zip(categories, resolved))
        ],
        "provenance": {
            "provider": "World Register of Marine Species (WoRMS), VLIZ",
            "api_documentation": f"{BASE_URL}/api-docs/openapi.yaml",
            "source": source,
            "rank_policy": "seven_principal_ranks_plus_all_target_taxa_and_source_root",
            "principal_ranks": list(MAIN_RANKS),
            "extra_target_ranks": sorted({record["rank"].lower() for record in resolved} - set(MAIN_RANKS)),
            "distance_note": (
                "Edge distances extend FGVC's seven-rank convention where an extra-rank target is retained"
            ),
            "manual_overrides": overrides,
            "responses": sorted(client.sources.values(), key=lambda record: record["url"]),
        },
    }
    Taxonomy(result, class_names=names)
    return result


def validate_class_map(taxonomy, class_map):
    """Require the exact YOLO IDs, source category IDs and names from preparation."""
    if not isinstance(class_map, list) or not class_map:
        raise ValueError("class_map must be a nonempty list produced by data_prepare")
    for entry in class_map:
        if (
            not isinstance(entry, dict)
            or type(entry.get("yolo_id")) is not int
            or type(entry.get("category_id")) is not int
            or not isinstance(entry.get("name"), str)
            or not entry["name"].strip()
        ):
            raise ValueError("Each class_map entry requires integer yolo_id/category_id and a nonempty name")
    actual = [
        {field: entry[field] for field in ("yolo_id", "category_id", "name")}
        for entry in sorted(class_map, key=lambda entry: entry["yolo_id"])
    ]
    expected = [
        {
            "yolo_id": entry["class_id"],
            "category_id": entry["source_category_id"],
            "name": entry["name"],
        }
        for entry in sorted(taxonomy["classes"], key=lambda entry: entry["class_id"])
    ]
    if actual != expected:
        raise ValueError(
            "class_map does not match taxonomy classes: YOLO IDs, source category IDs and names must match exactly "
            "Use dataset_train.json and the class_map.json generated from that same source"
        )


def main():
    """Build a frozen taxonomy without downloading images or training a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, required=True, help="COCO JSON containing all named categories")
    parser.add_argument("--class-map", type=Path, required=True, help="class_map.json produced by data_prepare")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="Directory retaining raw source responses")
    parser.add_argument("--overrides", type=Path, help="JSON mapping labels to {aphia_id: integer, reason: string}")
    parser.add_argument("--offline", action="store_true", help="Require a complete cache; never access the network")
    args = parser.parse_args()
    raw = args.coco.read_bytes()
    data = json.loads(raw)
    class_map_raw = args.class_map.read_bytes()
    class_map = json.loads(class_map_raw)
    overrides = json.loads(args.overrides.read_text(encoding="utf-8")) if args.overrides else None
    taxonomy = build_taxonomy(
        data["categories"],
        WormsClient(args.cache, offline=args.offline),
        overrides=overrides,
        source={"file_name": args.coco.name, "sha256": hashlib.sha256(raw).hexdigest()},
    )
    validate_class_map(taxonomy, class_map)
    taxonomy["provenance"]["class_map"] = {
        "file_name": args.class_map.name,
        "sha256": hashlib.sha256(class_map_raw).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(taxonomy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(taxonomy['classes'])} classes and {len(taxonomy['nodes'])} nodes to {args.output}")


if __name__ == "__main__":
    main()
