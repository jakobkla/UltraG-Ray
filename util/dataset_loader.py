import json
import os
from typing import Any, Dict

import numpy as np
import torch


class Parser:
    """Parser for ultrasound dataset with images and poses from .npy files."""

    def __init__(
        self,
        images_train: str,
        images_val: str,
        poses_train: str,
        poses_val: str,
        near_plane: float,
        far_plane: float,
        opening_width: float,
        visualize: bool = False,
    ):
        """
        Initialize the parser for ultrasound dataset.

        Args:
            images_train: Path to .npy file containing training grayscale images.
                Expected shape: (N, H, W) with values in [0, 255]
            images_val: Path to .npy file containing validation grayscale images.
                Expected shape: (M, H, W) with values in [0, 255]
            poses_train: Path to .npy file containing training camera poses.
                Expected shape: (N, 4, 4) or (N, 3, 4) - camera-to-world matrices
            poses_val: Path to .npy file containing validation camera poses.
                Expected shape: (M, 4, 4) or (M, 3, 4) - camera-to-world matrices
            near_plane: Near bound for ultrasound rendering (in cm)
            far_plane: Far bound for ultrasound rendering (in cm)
            opening_width: Opening width of the ultrasound probe (in cm)
            visualize: Whether to render a 3D visualization of the setup
        """
        self.near_plane = float(near_plane)
        self.far_plane = float(far_plane)
        self.opening_width = float(opening_width)
        self.visualize = visualize

        # Load training images
        print(f"[Parser] Loading training images from {images_train}")
        images_train_data = np.load(images_train)
        images_train_data = self._process_images(images_train_data)
        n_train, h, w = images_train_data.shape
        print(f"[Parser] Loaded {n_train} training grayscale images of size {h}x{w}")

        # Load validation images
        print(f"[Parser] Loading validation images from {images_val}")
        images_val_data = np.load(images_val)
        images_val_data = self._process_images(images_val_data)
        n_val, h_val, w_val = images_val_data.shape
        print(
            f"[Parser] Loaded {n_val} validation grayscale images of size {h_val}x{w_val}"
        )

        if h != h_val or w != w_val:
            raise ValueError(
                f"Train and val images must have the same dimensions. "
                f"Train: {h}x{w}, Val: {h_val}x{w_val}"
            )

        # Load training poses
        print(f"[Parser] Loading training poses from {poses_train}")
        camtoworlds_train = self._load_poses(poses_train)

        if len(camtoworlds_train) != n_train:
            raise ValueError(
                f"Number of training poses ({len(camtoworlds_train)}) does not match "
                f"number of training images ({n_train})"
            )

        # Load validation poses
        print(f"[Parser] Loading validation poses from {poses_val}")
        camtoworlds_val = self._load_poses(poses_val)

        if len(camtoworlds_val) != n_val:
            raise ValueError(
                f"Number of validation poses ({len(camtoworlds_val)}) does not match "
                f"number of validation images ({n_val})"
            )

        # Store train and val data separately
        self.images_train = images_train_data
        self.images_val = images_val_data
        self.camtoworlds_train = camtoworlds_train
        self.camtoworlds_val = camtoworlds_val

        self.image_names_train = [f"train_{i:04d}" for i in range(n_train)]
        self.image_names_val = [f"val_{i:04d}" for i in range(n_val)]

        print(f"[Parser] Using ultrasound intrinsics:")
        print(f"  - Opening width: {self.opening_width}")
        print(f"  - Near plane: {self.near_plane}")
        print(f"  - Far plane: {self.far_plane}")

        all_camtoworlds = np.concatenate([camtoworlds_train, camtoworlds_val], axis=0)
        self.transform = np.eye(4)

        # Calculate scene center/scale based on oriented ultrasound frames
        # Use each pose's local X (lateral) and Y (depth) axes and ultrasound near/far/width
        cam_R = all_camtoworlds[:, :3, :3]  # [N, 3, 3]
        cam_t = all_camtoworlds[:, :3, 3]  # [N, 3]
        x_dirs = cam_R[:, :, 0]  # [N, 3]
        z_dirs = cam_R[:, :, 2]  # [N, 3]

        half_width = float(self.opening_width) / 2.0
        near = float(self.near_plane)
        far = float(self.far_plane)
        half_depth = (far - near) / 2.0
        depth_center = near + half_depth

        # For each pose, compute the 4 rectangle corners in world space
        # Corners are centered at depth_center along +Z (ultrasound depth),
        # with +/- half_width along X (lateral) and +/- half_depth along Z (depth)
        all_corners = []
        for i in range(self.camtoworlds_train.shape[0]):
            t = cam_t[i]
            x_dir = x_dirs[i]
            z_dir = z_dirs[i]
            center_i = t + z_dir * depth_center
            lt = center_i + (-half_width) * x_dir + (+half_depth) * z_dir
            rt = center_i + (+half_width) * x_dir + (+half_depth) * z_dir
            rb = center_i + (+half_width) * x_dir + (-half_depth) * z_dir
            lb = center_i + (-half_width) * x_dir + (-half_depth) * z_dir
            all_corners.extend([lt, rt, rb, lb])

        all_corners = np.stack(all_corners, axis=0)  # [4N, 3]
        scene_center = np.mean(all_corners, axis=0)
        dists = np.linalg.norm(all_corners - scene_center, axis=1)
        self.scene_scale = float(np.max(dists))
        self.scene_center = scene_center

        # Center both train and val poses
        self.camtoworlds_train[:, :3, 3] = (
            self.camtoworlds_train[:, :3, 3] - self.scene_center
        )
        self.camtoworlds_val[:, :3, 3] = (
            self.camtoworlds_val[:, :3, 3] - self.scene_center
        )
        self.scene_center = np.zeros(3)

        # Store bounds for ultrasound
        self.bounds = np.array([self.near_plane, self.far_plane])

        # Store image size and additional metadata
        self.width = w
        self.height = h
        self.imsize_dict = {0: (w, h)}
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }

        print(
            f"[Parser] Loaded {n_train} training images and {n_val} validation images with poses"
        )
        print(f"[Parser] Scene scale: {self.scene_scale:.3f}")

        if self.visualize:
            self._visualize_scene()

    def _process_images(self, images: np.ndarray) -> np.ndarray:
        """Process images to ensure they are in the correct format."""
        # If images is RGB, reduce to grayscale
        if images.ndim == 4:
            images = np.mean(images, axis=-1)

        if images.ndim != 3:
            raise ValueError(f"Expected images shape (N, H, W), got {images.shape}")

        return images

    def _load_poses(self, poses_file: str) -> np.ndarray:
        """Load camera poses from file."""
        ext = os.path.splitext(poses_file)[1].lower()

        if ext == ".npy":
            poses = np.load(poses_file)
        elif ext == ".json":
            with open(poses_file, "r") as f:
                poses_list = json.load(f)
            poses = np.array(poses_list, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported pose file format: {ext}")

        # Convert to (N, 4, 4) format if needed
        if poses.shape[-2:] == (3, 4):
            # Add bottom row [0, 0, 0, 1]
            bottom = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))
            poses = np.concatenate([poses, bottom], axis=1)

        assert poses.shape[-2:] == (
            4,
            4,
        ), f"Poses must be (N, 4, 4) or (N, 3, 4), got {poses.shape}"

        # Convert positions from mm to cm by dividing by 10
        poses[:, :3, 3] *= 0.1

        return poses.astype(np.float32)


class Dataset:
    """A simple dataset class for ultrasound data."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
    ):
        """
        Initialize the dataset.

        Args:
            parser: Parser object containing loaded data
            split: "train" or "test" (test uses validation data)
        """
        self.parser = parser
        self.split = split

        if split == "train":
            self.images = self.parser.images_train
            self.camtoworlds = self.parser.camtoworlds_train
            self.image_names = self.parser.image_names_train
        else:  # test or val
            self.images = self.parser.images_val
            self.camtoworlds = self.parser.camtoworlds_val
            self.image_names = self.parser.image_names_val

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        # Load grayscale image from numpy array and normalize to [0, 1]
        image = self.images[item].astype(np.float32) / 255.0  # (H, W)

        # Keep grayscale as 1-channel
        image = image[..., np.newaxis]  # (H, W, 1)

        # Get camera pose
        camtoworld = self.camtoworlds[item]

        data = {
            "camtoworld": torch.from_numpy(camtoworld).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,
        }

        return data
