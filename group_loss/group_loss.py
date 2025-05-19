import torch
import torch.nn as nn
from typing import List, Union, Dict, Any

# External project imports (assumed to exist in the repo)
from group_loss.base_classes import (
    HierarchicalGroupWrapper,
    HierarchicalGroup,
    HierarchicalModule,
)

__all__ = ["HierarchicalRegularizer"]


class HierarchicalRegularizer(nn.Module):
    """Regularisation engine that follows the *v2 DSL* agreed in chat.

    **Supported keys in a config node**
    ----------------------------------
    ``type``         - "global" | "layerwise" | "hierarchical"
    ``groups``       - None | "base_level" | "layer" | "blocks" | …
    ``norm``         - outer norm between groups: "sum" | "L1" | "L2"
    ``inner_norm``   - inner norm inside a group:  "L1" | "L2" | "none"
    ``lambda``       - scaling coefficient (float, default 0.)
    ``children``     - list[Dict] (only for type="hierarchical")

    **Key semantic decisions**
    --------------------------
    * *hierarchical* acts **only** as a container; its own ``lambda`` & norms
      are ignored.  Real penalties are expressed via concrete *child* nodes.
    * For ``layerwise`` + ``groups == "base_level"`` the class implements
      **group-Lasso** when ``norm == "L1"`` and "smooth" decay when
      ``norm == "L2"``.
    * If ``inner_norm`` is omitted, defaults to "L2" (standard practice).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(self, model: HierarchicalModule) -> torch.Tensor:  # noqa: D401
        """Compute regularisation loss for *model* in a single forward pass."""
        root_groups = model.get_param_groups()
        return self._apply_node(self.config, root_groups)

    # ------------------------------------------------------------------
    # Core recursion
    # ------------------------------------------------------------------
    def _apply_node(
        self,
        node_cfg: Dict[str, Any],
        search_space: List[HierarchicalGroup],
    ) -> torch.Tensor:
        node_type: str = node_cfg["type"].lower()
        tag: Union[str, List[str], None] = node_cfg.get("groups", None)

        # Container nodes (hierarchical) _never_ impose a penalty themselves
        if node_type == "hierarchical":
            loss = torch.tensor(0.0, device=self._infer_device(search_space))
            for child in node_cfg.get("children", []):
                loss = loss + self._apply_node(child, self._filter_by_tag(search_space, tag))
            return loss

        # The rest (global / layerwise) do impose a penalty
        lam: float = float(node_cfg.get("lambda", 0.0))
        if lam == 0.0:
            # Fast-exit when λ == 0 → nothing to add, but recurse (if any)
            base_loss = torch.tensor(0.0, device=self._infer_device(search_space))
        else:
            outer_norm: str = node_cfg.get("norm", "L2").upper()
            inner_norm: str = node_cfg.get("inner_norm", "L2").upper()
            active_groups = self._filter_by_tag(search_space, tag)
            base_loss = self._compute_penalty(node_type, active_groups, lam, outer_norm, inner_norm)

        # Recurse even for global / layerwise (children allowed)
        for child_cfg in node_cfg.get("children", []):
            base_loss = base_loss + self._apply_node(child_cfg, self._filter_by_tag(search_space, tag))
        return base_loss

    # ------------------------------------------------------------------
    # Penalty calculators
    # ------------------------------------------------------------------
    def _compute_penalty(
        self,
        node_type: str,
        groups: List[HierarchicalGroup],
        lam: float,
        outer_norm: str,
        inner_norm: str,
    ) -> torch.Tensor:
        """Return λ · penalty for *this* node (no recursion inside)."""
        device = self._infer_device(groups)
        loss = torch.tensor(0.0, device=device)

        if node_type == "global":
            vec = self._collect_weights(groups, flatten=True)
            if vec.numel():
                loss = lam * self._apply_outer(vec.view(1, -1), outer_norm, inner_norm)

        elif node_type == "layerwise":
            # Distinguish the special *base_level* case (filters / neurons)
            all_group_norms: List[torch.Tensor] = []
            for grp in groups:
                if not isinstance(grp.module, HierarchicalGroupWrapper):
                    continue  # safety guard

                weights = grp.module.get_weights()  # shape [units, ...]
                if weights.numel() == 0:
                    continue

                if grp.groups and "base_level" in grp.groups:
                    # Treat each unit (filter / neuron) separately
                    unit_mat = weights.view(weights.size(0), -1)  # [N_units, dim]
                    norms = self._apply_inner(unit_mat, inner_norm)  # [N_units]
                    all_group_norms.append(norms)
                else:
                    # Whole layer = single group
                    norms = self._apply_inner(weights.view(1, -1), inner_norm)  # [1]
                    all_group_norms.append(norms)

            if all_group_norms:
                stacked = torch.cat(all_group_norms)  # [num_groups]
                loss = lam * self._apply_outer_on_vector(stacked, outer_norm)

        else:
            raise ValueError(f"Unknown node type: {node_type}")

        return loss

    # ---------------- helpers for norms -----------------
    @staticmethod
    def _apply_inner(mat: torch.Tensor, inner: str) -> torch.Tensor:
        """Compute inner norm per *row* of *mat* → returns 1-D tensor."""
        if inner == "L1":
            return torch.sum(torch.abs(mat), dim=1)
        elif inner == "L2":
            return torch.norm(mat, p=2, dim=1)
        elif inner == "NONE":
            return mat.mean(dim=1) * 0  # effectively disables contribution
        else:
            raise ValueError(f"Unsupported inner_norm: {inner}")

    @staticmethod
    def _apply_outer_on_vector(vec: torch.Tensor, outer: str) -> torch.Tensor:
        if outer in ("SUM", "NONE"):
            return torch.sum(vec)
        elif outer == "L1":
            return torch.sum(torch.abs(vec))
        elif outer == "L2":
            return torch.sum(vec.pow(2))
        else:
            raise ValueError(f"Unsupported outer norm: {outer}")

    def _apply_outer(self, mat: torch.Tensor, outer: str, inner: str) -> torch.Tensor:
        """Convenience: inner+outer in one go for *global* nodes."""
        inner_vec = self._apply_inner(mat, inner)  # [1] because mat has 1 row
        return self._apply_outer_on_vector(inner_vec, outer)

    # ---------------- utility -----------------
    def _filter_by_tag(
        self, groups: List[HierarchicalGroup], tag: Union[str, List[str], None]
    ) -> List[HierarchicalGroup]:
        if tag is None:
            return groups
        if isinstance(tag, str):
            tag = [tag]
        matched: List[HierarchicalGroup] = []
        for g in groups:
            if any(t in g.groups for t in tag):
                matched.append(g)
            if isinstance(g.module, list):
                for item in g.module:
                    if isinstance(item, HierarchicalGroup):
                        matched.extend(self._filter_by_tag([item], tag))
        return matched

    def _collect_weights(self, groups: List[HierarchicalGroup], *, flatten: bool) -> torch.Tensor:
        tensors: List[torch.Tensor] = []
        for g in groups:
            if isinstance(g.module, HierarchicalGroupWrapper):
                w = g.module.get_weights()
                tensors.append(w.view(-1) if flatten else w)
            elif isinstance(g.module, list):
                for item in g.module:
                    if isinstance(item, HierarchicalGroupWrapper):
                        w = item.get_weights()
                        tensors.append(w.view(-1) if flatten else w)
                    elif isinstance(item, HierarchicalGroup):
                        tensors.append(self._collect_weights([item], flatten=True))
        return torch.cat(tensors) if tensors else torch.empty(0, device=self._infer_device(groups))

    @staticmethod
    def _infer_device(groups: List[HierarchicalGroup]) -> torch.device:
        for g in groups:
            if isinstance(g.module, HierarchicalGroupWrapper):
                return g.module.get_weights().device
            if isinstance(g.module, list):
                for item in g.module:
                    if isinstance(item, HierarchicalGroupWrapper):
                        return item.get_weights().device
        return torch.device("cpu")
