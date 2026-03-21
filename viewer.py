import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import viser
from torch.utils.data import DataLoader

from gsplat.rendering import ultrasound_rasterization
from nerfview import CameraState, RenderTabState
from util.dataset_loader import Dataset, Parser
from util.visor_viewer import UltrasoundVisorViewer


def validate_benchmark_config(args):
    """Validate that the benchmark config file is set and contains all required values."""
    if not hasattr(args, "benchmark_config") or args.benchmark_config is None:
        raise ValueError("Benchmark requires --config_file to be set")

    cfg = args.benchmark_config
    if not cfg:
        raise ValueError("Benchmark config file is empty or could not be loaded")

    required_keys = [
        "images_file_train",
        "images_file_val",
        "poses_file_train",
        "poses_file_val",
        "ultrasound_near_plane",
        "ultrasound_far_plane",
        "ultrasound_opening_width",
    ]

    missing_keys = [key for key in required_keys if key not in cfg]
    if missing_keys:
        raise ValueError(f"Benchmark config is missing required keys: {missing_keys}")

    return cfg


def run_benchmark(
    means,
    quats,
    scales,
    transmittances,
    intensities,
    sh_degree,
    cfg,
    args,
    device,
):
    """Run benchmark over validation poses of a given dataset."""
    parser = Parser(
        images_train=cfg["images_file_train"],
        images_val=cfg["images_file_val"],
        poses_train=cfg["poses_file_train"],
        poses_val=cfg["poses_file_val"],
        near_plane=cfg["ultrasound_near_plane"],
        far_plane=cfg["ultrasound_far_plane"],
        opening_width=cfg["ultrasound_opening_width"],
    )
    valset = Dataset(parser, split="val")

    loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)

    # Move to device
    all_camtoworlds = []
    for batch in loader:
        camtoworld = batch["camtoworld"].to(device)  # [1, 4, 4]
        all_camtoworlds.append(camtoworld)

    # Run multiple measurement iterations
    for _ in range(5):
        tic = time.time()

        for _ in range(10):
            for camtoworld in all_camtoworlds:
                viewmats = torch.linalg.inv(camtoworld)  # [1, 4, 4]

                ultrasound_rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    transmittances=transmittances,
                    intensities=intensities,
                    viewmats=viewmats,
                    width=args.benchmark_res_width,
                    height=args.benchmark_res_height,
                    near_plane=cfg["ultrasound_near_plane"],
                    far_plane=cfg["ultrasound_far_plane"],
                    opening_angle=None,
                    opening_width=cfg["ultrasound_opening_width"],
                    sh_degree=sh_degree,
                )

        elapsed = time.time() - tic
        num_frames = 10 * len(all_camtoworlds)
        fps = num_frames / elapsed
        print(
            f"[Benchmark] frames={num_frames}, time={elapsed:.3f}s, " f"FPS={fps:.2f}"
        )


def main(args):
    torch.manual_seed(42)
    device = torch.device("cuda", 0)

    means, quats, scales, transmittances, sh0, shN = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for ckpt_path in args.ckpt:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)["splats"]
        means.append(ckpt["means"])
        quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
        scales.append(torch.exp(ckpt["scales"]))
        transmittances.append(torch.sigmoid(ckpt["transmittances"]))
        sh0.append(ckpt["sh0"])
        shN.append(ckpt["shN"])
    means = torch.cat(means, dim=0)
    quats = torch.cat(quats, dim=0)
    scales = torch.cat(scales, dim=0)
    transmittances = torch.cat(transmittances, dim=0)
    sh0 = torch.cat(sh0, dim=0)
    shN = torch.cat(shN, dim=0)
    intensities = torch.cat([sh0, shN], dim=-2)
    sh_degree = int(math.sqrt(intensities.shape[-2]) - 1)
    print("Number of Gaussians:", len(means))

    if args.benchmark:
        cfg = validate_benchmark_config(args)
        run_benchmark(
            means,
            quats,
            scales,
            transmittances,
            intensities,
            sh_degree,
            cfg,
            args,
            device,
        )

    def _viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        return UltrasoundVisorViewer.render_splats(
            means=means,
            quats=quats,
            scales=scales,
            transmittances=transmittances,
            intensities=intensities,
            camera_state=camera_state,
            render_tab_state=render_tab_state,
            device=device,
            sh_degree=sh_degree,
        )

    server = viser.ViserServer(port=args.port, verbose=False)
    _ = UltrasoundVisorViewer(
        server=server,
        render_fn=_viewer_render_fn,
        output_dir=Path(args.output_dir),
        mode="rendering",
    )

    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


def load_config(config_path: str) -> dict:
    """Load and return JSON config from file path."""
    if not config_path:
        return {}
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"[Config] Loading from JSON file: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Viewer for Gaussian splatting models")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Directory to save viewer outputs (default: results/)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to checkpoint .pt file(s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the viewer server (default: 8080)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark over validation poses before starting viewer",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to JSON config file (required for benchmarking)",
    )
    parser.add_argument(
        "--benchmark_res_width",
        type=int,
        default=512,
        help="Render width for benchmark (default: 512)",
    )
    parser.add_argument(
        "--benchmark_res_height",
        type=int,
        default=512,
        help="Render height for benchmark (default: 512)",
    )
    args = parser.parse_args()

    args.benchmark_config = load_config(args.config_file) if args.config_file else None

    main(args)
