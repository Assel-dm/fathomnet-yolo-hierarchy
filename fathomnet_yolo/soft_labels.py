"""Taxonomic soft-label cross-entropy for YOLO foreground assignments."""

import math
from numbers import Real

import torch
from fathomnet_yolo.taxonomy import Taxonomy
from torch import nn
from torch.nn import functional as F


class HierarchicalSoftLabelLoss(nn.Module):
    """Compare foreground logits with fixed taxonomic distributions.

    beta=None selects the hard-target auxiliary CE control.
    Distances are normalized by their global maximum.
    """

    def __init__(
        self,
        taxonomy_path,
        class_names,
        beta=10.0,
        distance_mode="edge_count",
    ):
        super().__init__()

        if beta is not None and (
            isinstance(beta, bool)
            or not isinstance(beta, Real)
            or not math.isfinite(beta)
            or beta < 0
        ):
            raise ValueError(
                "soft_beta must be finite and nonnegative; "
                "use beta=None for hard auxiliary CE"
            )

        if distance_mode not in {"edge_count", "lca_height"}:
            raise ValueError(
                "soft_distance_mode must be edge_count or lca_height"
            )

        taxonomy = Taxonomy.from_json(
            taxonomy_path,
            class_names=class_names,
            distance_mode=distance_mode,
        )

        self.class_names = taxonomy.class_names
        self.num_classes = len(self.class_names)
        self.beta = None if beta is None else float(beta)
        self.distance_mode = taxonomy.distance_mode
        self.distance_scale = float(taxonomy.distance_matrix.max())

        if beta is None:
            matrix = torch.eye(
                self.num_classes,
                dtype=torch.float32,
            )
        else:
            # Float64 avoids overflow for large finite beta values.
            distances = torch.as_tensor(
                taxonomy.distance_matrix,
                dtype=torch.float64,
            )
            distances = distances / (self.distance_scale or 1.0)

            matrix = (
                -self.beta * distances
            ).softmax(dim=1).float()

        entropy = -torch.special.xlogy(
            matrix,
            matrix,
        ).sum(dim=1)

        self.register_buffer("soft_label_matrix", matrix)
        self.register_buffer("target_entropies", entropy)

        diagonal = matrix.diag()

        self.target_statistics = {
            "mode": "hard" if beta is None else "soft",
            "beta": self.beta,
            "distance_mode": self.distance_mode,
            "distance_normalization": (
                "global_max_over_detector_classes"
            ),
            "distance_scale": self.distance_scale,
            "num_classes": self.num_classes,
            "true_class_mass_mean": diagonal.mean().item(),
            "true_class_mass_min": diagonal.min().item(),
            "true_class_mass_max": diagonal.max().item(),
            "target_entropy_mean": entropy.mean().item(),
        }

        self.last_components = {}

    def forward(
        self,
        logits,
        targets,
        quality_weights,
        denominator,
    ):
        """Return quality-weighted CE using YOLO's normalization.

        logits: foreground logits, shape (N, C).
        targets: assigned class IDs, int64, shape (N,).
        quality_weights: assignment quality, shape (N,).
        denominator: sum of native YOLO target scores.
        """
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected foreground logits (N, {self.num_classes}), "
                f"got {tuple(logits.shape)}"
            )

        count = logits.shape[0]

        if targets.shape != (count,) or targets.dtype != torch.int64:
            raise ValueError(
                "Assigned foreground targets must be int64 "
                "with shape (N,)"
            )

        if quality_weights.shape != (count,):
            raise ValueError(
                "Foreground quality weights must have shape (N,)"
            )

        normalizer = torch.as_tensor(
            denominator,
            dtype=torch.float32,
            device=logits.device,
        )

        if normalizer.numel() != 1:
            raise ValueError(
                "The assignment denominator must be a scalar"
            )

        normalizer = (
            normalizer.detach()
            .reshape(())
            .clamp_min(1.0)
        )

        if not count:
            zero = logits.float().sum() * 0.0

            self.last_components = {
                name: zero.detach()
                for name in ("ce", "kl", "entropy")
            }

            return zero

        # Keep the auxiliary calculation in float32 under AMP.
        log_probabilities = F.log_softmax(
            logits.float(),
            dim=1,
        )

        target_probabilities = (
            self.soft_label_matrix
            .index_select(0, targets)
            .float()
        )

        per_assignment_ce = -(
            target_probabilities * log_probabilities
        ).sum(dim=1)

        quality = quality_weights.detach().float()

        ce = (
            quality * per_assignment_ce
        ).sum() / normalizer

        entropy = (
            quality
            * self.target_entropies.index_select(
                0,
                targets,
            ).float()
        ).sum() / normalizer

        self.last_components = {
            "ce": ce.detach(),
            "kl": (ce.detach() - entropy).clamp_min(0.0),
            "entropy": entropy,
        }

        return ce
