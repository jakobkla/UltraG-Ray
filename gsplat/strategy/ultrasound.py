from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union, Optional

import torch

from .base import Strategy
from .ops import duplicate, remove, split_ultrasound


@dataclass
class Ultrasound3DStrategy(Strategy):
    """
    3D ultrasound-specific strategy:
    - Uses long-term 3D grad norms of `means` to decide where to refine.
    - Duplicates small, high-grad GSs.
    - Splits large, high-grad GSs.
    - Prunes GSs based on 3D scale (and optionally transmittance).
    """

    grow_grad3d: float = 5e-6  # threshold on avg 3D grad norm
    grow_scale3d: float = 0.01  # small vs large scale boundary (× scene_scale)
    prune_scale3d: float = 0.1  # too large → prune (× scene_scale)
    prune_scale3d_min: float = 1e-5  # too small → prune (× scene_scale)

    refine_start_iter: int = 1000
    refine_stop_iter: int = 20_000
    refine_every: int = 2500

    # maximum number of GSs
    max_gaussians: Optional[int] = None

    verbose: bool = False

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        # Running stats for 3D gradient norms
        return {
            "grad3d": None,  # [N]
            "count": None,  # [N]
            "scene_scale": scene_scale,
        }

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        # super().check_sanity(params, optimizers)
        for key in ["means", "scales", "quats"]:  # transmittances
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        # We don't need anything special pre-backward: we will use params["means"].grad
        # in step_post_backward.
        return

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        if step >= self.refine_stop_iter:
            return

        self._update_state(params, state)

        if step > self.refine_start_iter and step % self.refine_every == 0:
            n_dupli, n_split = self._grow_gs(params, optimizers, state)
            n_prune = self._prune_gs(params, optimizers, state)

            if self.verbose:
                print(
                    f"[Ultrasound3DStrategy] Step {step}: "
                    f"{n_dupli} GSs duplicated, {n_split} GSs split, {n_prune} GSs pruned. "
                    f"Now {len(params['means'])} GSs."
                )

            # reset stats
            state["grad3d"].zero_()
            state["count"].zero_()
            torch.cuda.empty_cache()

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
    ):
        means = params["means"]
        means_grad = means.grad

        if means_grad is None:
            return

        n_gaussian = means.shape[0]
        device = means.device

        # Initialize on first call
        if state["grad3d"] is None:
            state["grad3d"] = torch.zeros(n_gaussian, device=device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=device)

        # Accumulate grad norms
        grad_norm = means_grad.norm(dim=-1)  # [N]
        state["grad3d"][:n_gaussian] += grad_norm
        state["count"][:n_gaussian] += 1.0

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
    ) -> Tuple[int, int]:
        grad3d = state["grad3d"]
        count = state["count"]
        scene_scale = state["scene_scale"]

        device = grad3d.device
        grad_avg = grad3d / count.clamp_min(1.0)

        is_grad_high = grad_avg > self.grow_grad3d

        scales = torch.exp(params["scales"])  # [N, 3]
        max_scales = scales.max(dim=-1).values  # [N]

        is_small = max_scales <= self.grow_scale3d * scene_scale
        is_large = ~is_small

        is_dupli = is_grad_high & is_small
        is_split = is_grad_high & is_large

        n_dupli = is_dupli.sum().item()
        n_split = is_split.sum().item()

        max_gs = getattr(self, "max_gaussians", None)
        if max_gs is not None and max_gs > 0:
            cur_n = params["means"].shape[0]

            # each duplicate/split adds one GS
            to_add = n_dupli + n_split
            if cur_n >= max_gs or to_add == 0:
                return 0, 0

            allowed = max_gs - cur_n
            if to_add > allowed:
                # choose the "best" candidates (highest grad) up to capacity
                add_mask = is_dupli | is_split
                idx = torch.nonzero(add_mask, as_tuple=False).squeeze(-1)
                if idx.numel() > 0:
                    _, order = torch.sort(grad_avg[idx], descending=True)
                    keep = idx[order[:allowed]]

                    new_is_dupli = torch.zeros_like(is_dupli)
                    new_is_split = torch.zeros_like(is_split)
                    new_is_dupli[keep] = is_dupli[keep]
                    new_is_split[keep] = is_split[keep]

                    is_dupli = new_is_dupli
                    is_split = new_is_split
                    n_dupli = is_dupli.sum().item()
                    n_split = is_split.sum().item()

        # First duplicate
        if n_dupli > 0:
            duplicate(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_dupli,
            )

        # New GSs added by duplication should not be split immediately:
        if n_split > 0:
            # append zeros for dup entries in the mask
            is_split = torch.cat(
                [
                    is_split,
                    torch.zeros(n_dupli, dtype=torch.bool, device=device),
                ],
                dim=0,
            )
            split_ultrasound(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
            )

        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
    ) -> int:
        scene_scale = state["scene_scale"]

        scales = torch.exp(params["scales"])  # [N, 3]
        max_scales = scales.max(dim=-1).values  # [N]

        # Scale-based pruning
        is_too_big = max_scales > self.prune_scale3d * scene_scale
        is_too_small = max_scales < self.prune_scale3d_min * scene_scale

        if self.verbose:
            print(f"Pruning {is_too_big.sum().item()} GSs because they are too big.")
            print(
                f"Pruning {is_too_small.sum().item()} GSs because they are too small."
            )

        is_prune = is_too_big | is_too_small

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_prune,
            )
        return n_prune
