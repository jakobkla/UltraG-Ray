from typing import Dict

import numpy as np
import viser


class BaseVisualizer:
    """
    Base class for viser scene visualizers.

    Provides common functionality for managing per-client scene handles,
    clearing visualizations, and coordinate transformations.
    """

    def __init__(self, server: viser.ViserServer):
        """
        Initialize the base visualizer.

        Args:
            server: Viser server instance.
        """
        self.server = server
        self._client_handles: Dict[int, Dict[int, viser.SceneNodeHandle]] = {}

    def clear_all(self) -> None:
        """Remove all visualizations from all clients."""
        for client_id in list(self._client_handles.keys()):
            self.clear_for_client(client_id)
        self._client_handles.clear()

    def clear_for_client(self, client_id: int) -> None:
        """
        Clear all visualizations for a specific client.

        Args:
            client_id: The client ID to clear visualizations for.
        """
        if client_id not in self._client_handles:
            return
        for handle in self._client_handles[client_id].values():
            try:
                handle.remove()
            except Exception:
                pass
        self._client_handles[client_id].clear()

    def clear_handle(self, client_id: int, handle_id: int) -> None:
        """
        Clear a specific handle for a client.

        Args:
            client_id: The client ID.
            handle_id: The handle ID to clear.
        """
        if client_id not in self._client_handles:
            return
        handle = self._client_handles[client_id].get(handle_id)
        if handle is not None:
            try:
                handle.remove()
            except Exception:
                pass
            self._client_handles[client_id].pop(handle_id, None)

    def _ensure_client_dict(self, client_id: int) -> None:
        """Ensure the client has a handles dictionary."""
        if client_id not in self._client_handles:
            self._client_handles[client_id] = {}

    def _store_handle(
        self, client_id: int, handle_id: int, handle: viser.SceneNodeHandle
    ) -> None:
        """
        Store a handle for a client.

        Args:
            client_id: The client ID.
            handle_id: The handle ID.
            handle: The scene node handle to store.
        """
        self._ensure_client_dict(client_id)
        self._client_handles[client_id][handle_id] = handle

    @staticmethod
    def transform_to_world(
        points_local: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray,
    ) -> np.ndarray:
        """
        Transform points from local space to world space.

        Args:
            points_local: Points in local space [N, 3].
            position: Position in world space [3].
            rotation: Rotation matrix (3x3).

        Returns:
            Points transformed to world space [N, 3].
        """
        points_rotated = points_local @ rotation.T
        points_world = points_rotated + position
        return points_world

    def remove_client(self, client_id: int) -> None:
        """
        Clean up when a client disconnects.

        Args:
            client_id: The client ID that disconnected.
        """
        self.clear_for_client(client_id)
        self._client_handles.pop(client_id, None)
