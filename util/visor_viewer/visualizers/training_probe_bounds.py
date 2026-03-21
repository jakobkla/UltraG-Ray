from typing import Callable

import numpy as np

from .base_visualizer import BaseVisualizer


class TrainingProbeBoundsVisualizer(BaseVisualizer):
    """
    Visualizer for training ultrasound probe bounds.

    This class manages the visualization of rectangular bounds representing
    the training probe positions on 3D clients.
    """

    def create_bounds(
        self,
        parser,
        get_client_mode: Callable[[int], str],
    ) -> None:
        """
        Create visualization of training ultrasound probe bounds on all 3D clients.

        Args:
            parser: Dataset parser containing probe configuration and camera poses.
            get_client_mode: Callback to determine if a client is in "3d" mode.
        """
        if parser is None:
            return

        self.clear_all()

        half_width = float(parser.opening_width) / 2.0
        near = float(parser.near_plane)
        far = float(parser.far_plane)
        half_depth = (far - near) / 2.0
        depth_center = near + half_depth

        camtoworlds = parser.camtoworlds_train
        cam_R = camtoworlds[:, :3, :3]
        cam_t = camtoworlds[:, :3, 3]
        x_dirs = cam_R[:, :, 0]
        z_dirs = cam_R[:, :, 2]

        for client_id, client in self.server.get_clients().items():
            if get_client_mode(client_id) != "3d":
                continue

            for i in range(camtoworlds.shape[0]):
                t = cam_t[i]
                x_dir = x_dirs[i]
                z_dir = z_dirs[i]
                center_i = t + z_dir * depth_center

                lt = center_i + (-half_width) * x_dir + (+half_depth) * z_dir
                rt = center_i + (+half_width) * x_dir + (+half_depth) * z_dir
                rb = center_i + (+half_width) * x_dir + (-half_depth) * z_dir
                lb = center_i + (-half_width) * x_dir + (-half_depth) * z_dir

                lines = np.array(
                    [
                        [lb, rb],
                        [rb, rt],
                        [rt, lt],
                        [lt, lb],
                    ]
                )

                handle = client.scene.add_line_segments(
                    name=f"/training_probe_bounds_{i}",
                    points=lines,
                    colors=(0.0, 0.8, 0.0),
                    line_width=0.5,
                )
                self._store_handle(client_id, i, handle)
