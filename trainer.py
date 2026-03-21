import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from util.dataset_loader import Dataset, Parser
from fused_ssim import fused_ssim
from util.gmsd import gmsd
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal
from pytorch_msssim import ms_ssim as ms_ssim_torch

from gsplat import export_splats
from gsplat.rendering import ultrasound_rasterization
from gsplat.strategy import Ultrasound3DStrategy
from util.visor_viewer import (
    UltrasoundVisorViewer,
    VisorRenderTabState,
)
from nerfview import CameraState, RenderTabState


@dataclass
class Config:
    config_file: Optional[str] = None

    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Path to images.npy file
    images_file_train: str = "data/pig_shoulder_v2/images_train.npy"
    # Path to poses.npy file
    poses_file_train: str = "data/pig_shoulder_v2/poses_train.npy"
    # Path to images.npy file
    images_file_val: str = "data/pig_shoulder_v2/images_val.npy"
    # Path to poses.npy file
    poses_file_val: str = "data/pig_shoulder_v2/poses_val.npy"

    # Directory to save results
    result_dir: str = "results/pig_shoulder_v2"
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0

    # Port for the viewer server
    port: int = 8082

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 8
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(
        default_factory=lambda: [1_000, 10_000, 20_000, 25_000, 30_000]
    )
    # Steps to save the model to resume training (ckpt)
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = True
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Initial number of GSs
    init_num_pts: int = 500_000
    # Initial extent of GSs as a multiple of the camera extent
    init_extent: float = 0.8
    # Degree of spherical harmonics
    sh_degree: int = 1
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial transmittance of GS
    init_transmittance: float = 0.99
    # Initial scale of GS
    init_scale: float = 0.05
    # Weight for SSIM loss
    ssim_lambda: float = 0.5

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Depth of top of image
    ultrasound_near_plane: float = 0.0
    # Depth of bottom of image
    ultrasound_far_plane: float = 5.0
    # Width of linear transducer
    ultrasound_opening_width: float = 5.13

    # Strategy for GS densification
    strategy: Ultrasound3DStrategy = field(
        default_factory=lambda: Ultrasound3DStrategy(
            max_gaussians=500_000, verbose=True
        )
    )

    # Noise in cm (x-axis in camera space) to appy to training views to get even coverage
    pose_sideways_noise: float = 0.0
    # Noise in cm (y-axis in camera space) to appy to training views to get even coverage
    pose_frontback_noise: float = 0.2
    # Noise in cm (z-axis in camera space) to appy to training views to get even coverage
    pose_updown_noise: float = 0.0

    # LR for 3D point positions
    means_lr: float = 1e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for transmittances
    transmittances_lr: float = 5e-4
    # LR for orientation (quaternions)
    quats_lr: float = 5e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 1e-5

    # Scale regularization
    scale_reg: float = 0.01

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # Wheter to run evaluation on training set as well
    eval_trainset: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
        strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
        strategy.refine_every = int(strategy.refine_every * factor)


def create_splats_with_optimizers(
    init_num_pts: int,
    init_extent: float,
    init_transmittance: float,
    init_scale: float,
    means_lr: float,
    scales_lr: float,
    transmittances_lr: float,
    quats_lr: float,
    sh0_lr: float,
    shN_lr: float,
    scene_scale: float,
    sh_degree: int,
    batch_size: int,
    device: str,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
    rgbs = torch.ones((init_num_pts, 1)) * 0.1

    N = points.shape[0]
    scales = torch.log(torch.full((N, 3), init_scale))  # [N, 3]

    quats = torch.rand((N, 4))  # [N, 4]
    transmittances = torch.full((N,), init_transmittance)  # [N,]
    transmittances = torch.logit(transmittances, eps=1e-10)  # [N,]

    colors = torch.zeros((N, (sh_degree + 1) ** 2, 1))  # [N, K, 1]

    # RGB to SH
    C0 = 0.28209479177387814
    colors[:, 0, :] = (rgbs - 0.5) / C0

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("transmittances", torch.nn.Parameter(transmittances), transmittances_lr),
        ("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr),
        ("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr),
    ]

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size
    optimizers = {
        name: torch.optim.Adam(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(self, cfg: Config) -> None:
        self.start_time = time.time()

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        self.cfg = cfg
        self.device = f"cuda:{0}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        print(f"[Runner] Loading ultrasound dataset")
        print(f"  Images: {cfg.images_file_train}")
        print(f"  Poses: {cfg.poses_file_train}")
        self.parser = Parser(
            images_train=cfg.images_file_train,
            images_val=cfg.images_file_val,
            poses_train=cfg.poses_file_train,
            poses_val=cfg.poses_file_val,
            near_plane=cfg.ultrasound_near_plane,
            far_plane=cfg.ultrasound_far_plane,
            opening_width=cfg.ultrasound_opening_width,
        )
        self.trainset = Dataset(self.parser, split="train")
        self.valset = Dataset(self.parser, split="val")

        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        self.splats, self.optimizers = create_splats_with_optimizers(
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_transmittance=cfg.init_transmittance,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            transmittances_lr=cfg.transmittances_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            batch_size=cfg.batch_size,
            device=self.device,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.cfg.strategy.initialize_state(
            scene_scale=self.scene_scale
        )

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = UltrasoundVisorViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
                parser=self.parser,
            )

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        opening_width: float,
        near_plane: float,
        far_plane: float,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        transmittances = torch.sigmoid(self.splats["transmittances"])

        colors = torch.cat(
            [self.splats["sh0"], self.splats["shN"]], 1
        )  # [N, K, 1] monochrome

        (
            render_ultrasound,
            render_echo_alphas,
            render_transmittances,
            render_echoes,
            info,
        ) = ultrasound_rasterization(
            means=means,
            quats=quats,
            scales=scales,
            transmittances=transmittances,
            intensities=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            width=width,
            height=height,
            near_plane=near_plane,
            far_plane=far_plane,
            opening_angle=None,
            opening_width=opening_width,
            **kwargs,
        )
        if masks is not None:
            render_ultrasound[~masks] = 0
        return (
            render_ultrasound,
            render_echo_alphas,
            render_transmittances,
            render_echoes,
            info,
        )

    def compute_metrics(
        self,
        gt_img: torch.Tensor,
        eval_img: torch.Tensor,
        data_range: float = 1.0,
    ) -> Tuple[float, float, float, float, float, float]:
        if gt_img.shape != eval_img.shape:
            raise ValueError(f"Shape mismatch: {gt_img.shape} vs {eval_img.shape}")

        device = gt_img.device

        # --- MSE & PSNR (spatial, unmasked) ---
        diff = gt_img - eval_img
        mse_val_t = (diff**2).mean()
        mse_val = float(mse_val_t.item())

        if mse_val == 0.0:
            psnr_val = float("inf")
        else:
            psnr_val = float(
                10.0
                * torch.log10(
                    torch.tensor((data_range**2) / (mse_val + 1e-8), device=device)
                ).item()
            )

        # --- MSSIM (spatial) ---
        gt_in = gt_img.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        eval_in = eval_img.unsqueeze(0).unsqueeze(0)

        ssim_val_t = ms_ssim_torch(
            gt_in,
            eval_in,
            data_range=data_range,
            size_average=True,
        )
        ssim_val = float(ssim_val_t.item())

        # --- Fourier-based SSIM (FSSIM) ---
        # 1. FFT of images
        fft_gt = torch.fft.fft2(gt_img)
        fft_eval = torch.fft.fft2(eval_img)

        mag_gt = torch.abs(fft_gt)
        mag_eval = torch.abs(fft_eval)

        # 2. Log magnitude
        spec_gt = torch.log1p(mag_gt)
        spec_eval = torch.log1p(mag_eval)

        # 3. Joint normalization to [0, 1]
        min_val = torch.min(torch.min(spec_gt), torch.min(spec_eval))
        spec_gt = spec_gt - min_val
        spec_eval = spec_eval - min_val

        max_val = torch.max(torch.max(spec_gt), torch.max(spec_eval))
        spec_gt = spec_gt / (max_val + 1e-8)
        spec_eval = spec_eval / (max_val + 1e-8)

        spec_gt_in = spec_gt.unsqueeze(0).unsqueeze(0)
        spec_eval_in = spec_eval.unsqueeze(0).unsqueeze(0)

        fssim_val_t = ms_ssim_torch(
            spec_gt_in,
            spec_eval_in,
            data_range=1.0,
            size_average=True,
        )
        fssim_val = float(fssim_val_t.item())

        # --- Gradient Magnitude Similarity (GMS) ---
        # piq.gms expects (N,C,H,W); HIGH = GOOD
        gmsd_val_t, gms_val_t = gmsd(
            eval_in,  # prediction
            gt_in,  # reference
            data_range=data_range,
            reduction="mean",  # average over image
        )
        gmsd_val = float(gmsd_val_t.item())
        gms_val_t = float(gms_val_t.item())

        return mse_val, psnr_val, ssim_val, fssim_val, gmsd_val, gms_val_t

    def compute_gaussian_stats(self) -> Dict[str, float]:
        with torch.no_grad():
            scales = torch.exp(self.splats["scales"])
            scale_norms = scales.norm(dim=-1)

            transmittances = torch.sigmoid(self.splats["transmittances"])

            colors = self.splats["sh0"]
            if colors.ndim == 3:
                colors = colors[:, 0, :]
            color_magnitudes = colors.norm(dim=-1)

            stats = {
                "scale_mean": float(scale_norms.mean().item()),
                "scale_median": float(torch.quantile(scale_norms, 0.5).item()),
                "scale_std": float(scale_norms.std().item()),
                "scale_p10": float(torch.quantile(scale_norms, 0.1).item()),
                "scale_p90": float(torch.quantile(scale_norms, 0.9).item()),
                "transmittance_mean": float(transmittances.mean().item()),
                "transmittance_median": float(
                    torch.quantile(transmittances, 0.5).item()
                ),
                "transmittance_std": float(transmittances.std().item()),
                "transmittance_p10": float(torch.quantile(transmittances, 0.1).item()),
                "transmittance_p90": float(torch.quantile(transmittances, 0.9).item()),
                "color_mean": float(color_magnitudes.mean().item()),
                "color_median": float(torch.quantile(color_magnitudes, 0.5).item()),
                "color_std": float(color_magnitudes.std().item()),
                "color_p10": float(torch.quantile(color_magnitudes, 0.1).item()),
                "color_p90": float(torch.quantile(color_magnitudes, 0.9).item()),
            }

        return stats

    def sample_cosine_offset(self, magnitude: float, device: torch.device):
        max_trials = 100
        for _ in range(max_trials):
            x = torch.rand(1, device=device) - 0.5
            accept_prob = torch.pow((torch.cos(math.pi * x)), 2.0)
            u = torch.rand(1, device=device)
            if u <= accept_prob:
                return x * magnitude

        assert False

    def train(self):
        cfg = self.cfg
        device = self.device

        # Dump cfg.
        with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
            yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        lr_final_factor = 0.1  # final LR = 1% of initial LR
        gamma = lr_final_factor ** (1.0 / max_steps)

        # One scheduler per GS optimizer
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
            for _, opt in self.optimizers.items()
        ]

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        self.global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]

            offset = (torch.rand(1, device=device) - 0.5) * cfg.pose_sideways_noise
            sideways_direction = camtoworlds[:, :3, 0]
            translation = offset * sideways_direction
            camtoworlds[:, :3, 3] = camtoworlds[:, :3, 3] + translation

            offset = self.sample_cosine_offset(cfg.pose_frontback_noise, device)
            frontback_direction = camtoworlds[:, :3, 1]
            translation = offset * frontback_direction
            camtoworlds[:, :3, 3] = camtoworlds[:, :3, 3] + translation

            offset = (torch.rand(1, device=device) - 0.5) * cfg.pose_updown_noise
            updown_direction = camtoworlds[:, :3, 2]
            translation = offset * updown_direction
            camtoworlds[:, :3, 3] = camtoworlds[:, :3, 3] + translation

            pixels = data["image"].to(
                device
            )  # [1, H, W, 1] monochrome, already in [0, 1]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]

            height, width = pixels.shape[1:3]

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            (
                renders,
                alphas,
                render_transmittances,
                render_echoes,
                info,
            ) = self.rasterize_splats(
                camtoworlds=camtoworlds,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.ultrasound_near_plane,
                far_plane=cfg.ultrasound_far_plane,
                opening_width=cfg.ultrasound_opening_width,
                masks=masks,
            )
            if renders.shape[-1] == 2:
                colors, depths = renders[..., 0:1], renders[..., 1:2]
            else:
                colors, depths = renders, None

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            if cfg.scale_reg > 0.0:
                loss += cfg.scale_reg * (torch.exp(self.splats["scales"])).mean()

            loss.backward()

            if step % 1000 == 0:
                if self.splats["transmittances"].grad is not None:
                    grad_norm = self.splats["transmittances"].grad.norm().item()
                    grad_max = self.splats["transmittances"].grad.abs().max().item()
                    print(
                        f"Step {step}: transmittances grad norm={grad_norm:.6f}, max={grad_max:.6f}"
                    )
                else:
                    print(f"Step {step}: transmittances grad is None!")

            # Calculate SSIM value (not loss) for display
            ssim_value = 1.0 - ssimloss.item()

            desc = (
                f"loss={loss.item():.3f}| "
                f"ssim={ssim_value:.4f}| "
                f"sh degree={sh_degree_to_use}| "
            )

            pbar.set_description(desc)

            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - self.global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {
                    "step": step,
                    "global_tic": self.global_tic,
                    "splats": self.splats.state_dict(),
                }
                torch.save(data, f"{self.ckpt_dir}/ckpt_{step}.pt")
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:
                sh0 = self.splats["sh0"]
                shN = self.splats["shN"]

                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=torch.ones(means.shape[0], device=means.device),
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )

            self.cfg.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            eval_at_this_step = step + 1 in cfg.eval_steps
            if eval_at_this_step:
                self.eval(step, stage="val", eval_dataset=self.valset)
                if cfg.eval_trainset:
                    self.eval(step, stage="train", eval_dataset=self.trainset)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str, eval_dataset: Dataset):
        """Entry for evaluation."""
        print("Running evaluation...")
        # remove eval time from global_tic
        eval_time_tic = time.time()
        cfg = self.cfg
        device = self.device

        evalloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time_total = 0.0
        num_eval_images = 0
        metrics = defaultdict(list)
        for i, data in enumerate(evalloader):
            camtoworlds = data["camtoworld"].to(device)
            pixels = data["image"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, render_transmittances, render_echoes, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.ultrasound_near_plane,
                far_plane=cfg.ultrasound_far_plane,
                opening_width=cfg.ultrasound_opening_width,
                masks=masks,
            )  # [1, H, W, 1] monochrome
            torch.cuda.synchronize()
            ellipse_time_total += max(time.time() - tic, 1e-10)
            num_eval_images += 1

            colors = torch.clamp(colors, 0.0, 1.0)

            # Expand mono to RGB for visualization canvas
            pixels_rgb = pixels.repeat(1, 1, 1, 3)  # [B, H, W, 3]
            colors_rgb = colors.repeat(1, 1, 1, 3)  # [B, H, W, 3]
            canvas_list = [pixels_rgb, colors_rgb]

            # target_trans_rgb = target_transmittance.repeat(1, 1, 1, 3)  # [B, H, W, 3]
            render_trans_rgb = render_transmittances.repeat(1, 1, 1, 3)  # [B, H, W, 3]
            canvas_list.extend([render_trans_rgb])

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)

            # Also save GT and prediction (and transmittance) PNGs per frame
            step_dir = os.path.join(self.render_dir, f"{stage}_step{step}")
            os.makedirs(os.path.join(step_dir, "gt"), exist_ok=True)
            os.makedirs(os.path.join(step_dir, "ultrasound"), exist_ok=True)
            os.makedirs(os.path.join(step_dir, "trans"), exist_ok=True)
            os.makedirs(os.path.join(step_dir, "echo"), exist_ok=True)
            # Squeeze mono channel for grayscale PNG output
            gt_img = (pixels.squeeze(0).squeeze(-1).cpu().numpy() * 255).astype(
                np.uint8
            )  # [H, W]
            us_img = (colors.squeeze(0).squeeze(-1).cpu().numpy() * 255).astype(
                np.uint8
            )  # [H, W]
            trans_img = (render_trans_rgb.squeeze(0).cpu().numpy() * 255).astype(
                np.uint8
            )
            echo_img = (
                render_echoes.squeeze(0).squeeze(-1).cpu().numpy() * 255
            ).astype(
                np.uint8
            )  # [H, W]
            imageio.imwrite(os.path.join(step_dir, "gt", f"{i:04d}.png"), gt_img)
            imageio.imwrite(
                os.path.join(step_dir, "ultrasound", f"{i:04d}.png"), us_img
            )
            imageio.imwrite(os.path.join(step_dir, "trans", f"{i:04d}.png"), trans_img)
            imageio.imwrite(os.path.join(step_dir, "echo", f"{i:04d}.png"), echo_img)

            gt_gray = pixels[0, :, :, 0]
            us_gray = colors[0, :, :, 0]

            (
                mse_val,
                psnr_val,
                ssim_val,
                fssim_val,
                gmsd_val,
                gms_val,
            ) = self.compute_metrics(gt_gray, us_gray, data_range=1.0)

            metrics["mse"].append(mse_val)
            metrics["psnr"].append(psnr_val)
            metrics["ssim"].append(ssim_val)
            metrics["fssim"].append(fssim_val)
            metrics["gmsd"].append(gmsd_val)
            metrics["gms"].append(gms_val)

        eval_wall_time = time.time() - eval_time_tic
        self.global_tic += eval_wall_time
        ellipse_time_per_image = ellipse_time_total / max(num_eval_images, 1)
        train_elapsed_excluding_eval = time.time() - self.global_tic

        stats = {}
        for k, v in metrics.items():
            arr = np.array(v)
            stats[f"{k}_mean"] = float(np.mean(arr))
            stats[f"{k}_std"] = float(np.std(arr))
            stats[f"{k}_min"] = float(np.min(arr))
            stats[f"{k}_max"] = float(np.max(arr))
            stats[k] = float(np.mean(arr))

        stats.update(
            {
                "ellipse_time_per_image": ellipse_time_per_image,
                "ellipse_time_total": ellipse_time_total,
                "eval_wall_time": eval_wall_time,
                "num_eval_images": num_eval_images,
                "train_elapsed_excluding_eval": train_elapsed_excluding_eval,
                "num_GS": len(self.splats["means"]),
            }
        )

        # Add Gaussian statistics
        gaussian_stats = self.compute_gaussian_stats()
        stats.update(gaussian_stats)
        print(
            f"MSE: {stats['mse']:.6f} ± {stats['mse_std']:.6f}, "
            f"PSNR: {stats['psnr']:.3f} ± {stats['psnr_std']:.3f} (min: {stats['psnr_min']:.3f}, max: {stats['psnr_max']:.3f}), "
            f"SSIM: {stats['ssim']:.4f} ± {stats['ssim_std']:.4f} (min: {stats['ssim_min']:.4f}, max: {stats['ssim_max']:.4f}), "
            f"FSSIM: {stats['fssim']:.4f} ± {stats['fssim_std']:.4f}, "
            f"GMS: {stats['gms']:.4f} ± {stats['gms_std']:.4f}, "
            f"GMSD: {stats['gmsd']:.4f} ± {stats['gmsd_std']:.4f}, "
            f"Render Time: {stats['ellipse_time_per_image']:.3f}s/image "
            f"(total {stats['ellipse_time_total']:.3f}s over {stats['num_eval_images']} images), "
            f"Eval Wall Time: {stats['eval_wall_time']:.3f}s, "
            f"Number of GS: {stats['num_GS']} "
            f"Training Time (excluding eval): {stats['train_elapsed_excluding_eval']:.2f}s"
        )

        # Save stats to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"{stage}/{k}", v, step)
        self.writer.flush()

        # Save stats to CSV file
        # Preserve old stats by unique trainer start time
        csv_filename = f"{stage}_{self.start_time}.csv"
        csv_path = os.path.join(self.stats_dir, csv_filename)
        file_exists = os.path.isfile(csv_path)

        csv_row = {
            "step": step,
            "num_GS": stats["num_GS"],
            "num_eval_images": stats.get("num_eval_images", 0),
            "ellipse_time_per_image": stats.get("ellipse_time_per_image", 0),
            "ellipse_time_total": stats.get("ellipse_time_total", 0),
            "eval_wall_time": stats.get("eval_wall_time", 0),
            "train_elapsed_excluding_eval": stats.get(
                "train_elapsed_excluding_eval", 0
            ),
            # Image quality metrics
            "mse_mean": stats.get("mse_mean", 0),
            "mse_std": stats.get("mse_std", 0),
            "mse_min": stats.get("mse_min", 0),
            "mse_max": stats.get("mse_max", 0),
            "psnr_mean": stats.get("psnr_mean", 0),
            "psnr_std": stats.get("psnr_std", 0),
            "psnr_min": stats.get("psnr_min", 0),
            "psnr_max": stats.get("psnr_max", 0),
            "ssim_mean": stats.get("ssim_mean", 0),
            "ssim_std": stats.get("ssim_std", 0),
            "ssim_min": stats.get("ssim_min", 0),
            "ssim_max": stats.get("ssim_max", 0),
            "fssim_mean": stats.get("fssim_mean", 0),
            "fssim_std": stats.get("fssim_std", 0),
            "fssim_min": stats.get("fssim_min", 0),
            "fssim_max": stats.get("fssim_max", 0),
            "gms_mean": stats.get("gms_mean", 0),
            "gms_std": stats.get("gms_std", 0),
            "gms_min": stats.get("gms_min", 0),
            "gms_max": stats.get("gms_max", 0),
            "gmsd_mean": stats.get("gmsd_mean", 0),
            "gmsd_std": stats.get("gmsd_std", 0),
            "gmsd_min": stats.get("gmsd_min", 0),
            "gmsd_max": stats.get("gmsd_max", 0),
            # Gaussian statistics
            "scale_mean": stats.get("scale_mean", 0),
            "scale_median": stats.get("scale_median", 0),
            "scale_std": stats.get("scale_std", 0),
            "scale_p10": stats.get("scale_p10", 0),
            "scale_p90": stats.get("scale_p90", 0),
            "transmittance_mean": stats.get("transmittance_mean", 0),
            "transmittance_median": stats.get("transmittance_median", 0),
            "transmittance_std": stats.get("transmittance_std", 0),
            "transmittance_p10": stats.get("transmittance_p10", 0),
            "transmittance_p90": stats.get("transmittance_p90", 0),
            "color_mean": stats.get("color_mean", 0),
            "color_median": stats.get("color_median", 0),
            "color_std": stats.get("color_std", 0),
            "color_p10": stats.get("color_p10", 0),
            "color_p90": stats.get("color_p90", 0),
        }

        with open(csv_path, "a", newline="") as csvfile:
            fieldnames = list(csv_row.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(csv_row)

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, VisorRenderTabState)

        # Get splat data
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        transmittances = torch.sigmoid(self.splats["transmittances"])
        sh0 = self.splats["sh0"]
        shN = self.splats["shN"]
        intensities = torch.cat([sh0, shN], 1)

        sh_degree = min(render_tab_state.max_sh_degree, self.cfg.sh_degree)

        # Use shared render method
        return UltrasoundVisorViewer.render_splats(
            means=means,
            quats=quats,
            scales=scales,
            transmittances=transmittances,
            intensities=intensities,
            camera_state=camera_state,
            render_tab_state=render_tab_state,
            device=self.device,
            sh_degree=sh_degree,
        )


def main(cfg: Config):
    runner = Runner(cfg)

    if cfg.ckpt is not None:
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.global_tic = ckpts[0]["global_tic"]
        runner.eval(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # With config file
    python trainer.py --config_file path/to/config.json

    # Override config file values with CLI arguments
    python trainer.py --config_file config.json --max_gaussians 50000

    """

    # First, check if there's a config file argument
    config_file_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--config_file" and i + 1 < len(sys.argv):
            config_file_path = sys.argv[i + 1]
            break

    # Load JSON config if provided
    json_config = {}
    if config_file_path and os.path.exists(config_file_path):
        print(f"[Config] Loading from JSON file: {config_file_path}")
        with open(config_file_path, "r") as f:
            json_config = json.load(f)
        # Filter out comment fields (keys starting with '_') and None values
        json_config = {
            k: v
            for k, v in json_config.items()
            if not k.startswith("_") and v is not None
        }

    # Create default config and apply JSON values
    default_cfg = Config()
    if json_config:
        for key, value in json_config.items():
            if hasattr(default_cfg, key):
                try:
                    setattr(default_cfg, key, value)
                except (TypeError, ValueError) as e:
                    print(f"[Config] Warning: Could not set {key}={value}: {e}")

    # Use tyro to parse CLI arguments, using our (possibly JSON-modified) config as default
    cfg = tyro.cli(Config, default=default_cfg)
    cfg.adjust_steps(cfg.steps_scaler)

    main(cfg)
