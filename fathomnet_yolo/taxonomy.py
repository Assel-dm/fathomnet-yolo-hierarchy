"""Validated, local taxonomies for labels at different taxonomic depths"""

import json
from pathlib import Path

import numpy as np


class Taxonomy:
    """Store a rooted tree and the exact class order used by a detector.

    The JSON contains ``nodes`` with ``id``, ``parent``, ``name`` and optional
    ``rank``, and ``classes`` with contiguous ``class_id``, ``name``, ``node_id``.
    Classes may identify internal nodes. No missing rank is inferred.

    ``edge_count`` measures the shortest path length. ``lca_height`` measures
    the maximum number of edges below the lowest common ancestor, with zero
    on the diagonal. It reproduces the usual LCA height on a balanced tree,
    but is not numerically equivalent to a fixed-rank taxonomy on uneven trees.
    """

    @classmethod
    def from_json(cls, path, class_names=None, distance_mode=None):
        """Load a local JSON and optionally verify the detector's class order"""
        with Path(path).open(encoding="utf-8") as stream:
            return cls(json.load(stream), class_names=class_names, distance_mode=distance_mode)

    def __init__(self, data, class_names=None, distance_mode=None):
        if data.get("schema_version") != 1:
            raise ValueError("Taxonomy schema_version must be 1")
        self.distance_mode = distance_mode or data.get("distance_mode", "edge_count")
        if self.distance_mode not in {"edge_count", "lca_height"}:
            raise ValueError("distance_mode must be edge_count or lca_height")
        self.nodes = {}
        for node in data.get("nodes", []):
            identifier = node.get("id")
            if not isinstance(identifier, str) or not identifier.strip() or identifier in self.nodes:
                raise ValueError("Taxonomy node IDs must be unique nonempty strings")
            if not isinstance(node.get("name"), str) or not node["name"].strip():
                raise ValueError(f"Missing node name: {identifier}")
            rank = node.get("rank")
            if rank is not None and (not isinstance(rank, str) or not rank.strip()):
                raise ValueError(f"Invalid rank: {identifier}")
            self.nodes[identifier] = dict(node)
        roots = [key for key, node in self.nodes.items() if node.get("parent") is None]
        if len(roots) != 1:
            raise ValueError("Taxonomy must contain exactly one root")
        self.root = roots[0]
        self.paths = {}
        for identifier in self.nodes:
            path, current = [], identifier
            while current is not None:
                if current not in self.nodes:
                    raise ValueError(f"Unknown parent node: {current}")
                if current in path:
                    raise ValueError(f"Cycle in taxonomy at {current}")
                path.append(current)
                current = self.nodes[current].get("parent")
            self.paths[identifier] = tuple(reversed(path))
            ranks = [self.nodes[key].get("rank") for key in path if self.nodes[key].get("rank")]
            if len(ranks) != len(set(ranks)):
                raise ValueError(f"Repeated rank along the path to {identifier}")
        entries = data.get("classes", [])
        if not entries or any(type(entry.get("class_id")) is not int for entry in entries):
            raise ValueError("classes require integer class_id values")
        entries = sorted(entries, key=lambda entry: entry["class_id"])
        if [entry["class_id"] for entry in entries] != list(range(len(entries))):
            raise ValueError("class_id values must be contiguous from zero")
        self.class_names = tuple(entry.get("name") for entry in entries)
        self.class_node_ids = tuple(entry.get("node_id") for entry in entries)
        if any(not isinstance(name, str) or not name.strip() for name in self.class_names):
            raise ValueError("Class names must be nonempty strings")
        if len(set(self.class_names)) != len(entries) or len(set(self.class_node_ids)) != len(entries):
            raise ValueError("Each class must have a unique name and taxon node")
        if any(identifier not in self.nodes for identifier in self.class_node_ids):
            raise ValueError("Each class node_id must refer to a taxonomy node")
        if class_names is not None:
            if isinstance(class_names, dict):
                if set(class_names) != set(range(len(entries))):
                    raise ValueError("Detector class names must use contiguous integer IDs")
                class_names = [class_names[index] for index in range(len(entries))]
            if tuple(class_names) != self.class_names:
                raise ValueError("Taxonomy classes do not match the detector's ordered class names")
        count = len(entries)
        self.compatible_class_mask = np.array(
            [
                [target in self.paths[prediction] for prediction in self.class_node_ids]
                for target in self.class_node_ids
            ],
            dtype=bool,
        )
        heights = {identifier: 0 for identifier in self.nodes}
        for path in self.paths.values():
            for depth, identifier in enumerate(path):
                heights[identifier] = max(heights[identifier], len(path) - depth - 1)
        self.distance_matrix = np.zeros((count, count), dtype=np.float64)
        for i, left in enumerate(self.class_node_ids):
            for j, right in enumerate(self.class_node_ids[:i]):
                left_path, right_path = self.paths[left], self.paths[right]
                common = next(
                    (depth for depth, pair in enumerate(zip(left_path, right_path)) if pair[0] != pair[1]),
                    min(len(left_path), len(right_path)),
                )
                lca = left_path[common - 1]
                distance = len(left_path) + len(right_path) - 2 * common
                if self.distance_mode == "lca_height":
                    distance = heights[lca]
                self.distance_matrix[i, j] = self.distance_matrix[j, i] = distance
        ranks = sorted({node["rank"] for node in self.nodes.values() if node.get("rank")})
        self.ancestors_by_rank = {
            rank: np.asarray(
                [
                    next((key for key in self.paths[node] if self.nodes[key].get("rank") == rank), None)
                    for node in self.class_node_ids
                ],
                dtype=object,
            )
            for rank in ranks
        }

    def metadata(self):
        """Return the distance convention and class order for experiment records"""
        return {
            "schema_version": 1,
            "distance_mode": self.distance_mode,
            "num_nodes": len(self.nodes),
            "num_classes": len(self.class_names),
            "class_names": list(self.class_names),
            "class_node_ids": list(self.class_node_ids),
        }
