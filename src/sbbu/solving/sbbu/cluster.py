"""
Cluster and disjoint-set primitives for SBBU reflections.
"""

import numpy as np
from numpy.typing import NDArray

from ...geometry import normalize_vector


class ReflectionCluster:
    """
    Reflection-plane manager for coordinate mirroring in branch-and-bound.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes in the current problem instance.
    coordinates : NDArray[np.float64]
        Shared ``(num_nodes, 3)`` coordinate array updated in place.
    """

    def __init__(self, num_nodes: int, coordinates: NDArray[np.float64]) -> None:
        """Initialize reflection-plane storage for one cluster.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the problem instance.
        coordinates : NDArray[np.float64]
            Shared ``(num_nodes, 3)`` coordinate array.
        """
        self.num_nodes = num_nodes
        self.coordinates = coordinates  # Shared reference

        # Cluster bounds
        self.start_node = 0
        self.end_node = 0

        # Reflection planes (stored as Nx3 arrays)
        self.plane_points = np.zeros((num_nodes, 3), dtype=np.float64)
        self.plane_normals = np.zeros((num_nodes, 3), dtype=np.float64)
        self.plane_distances = np.zeros(num_nodes, dtype=np.float64)
        self.num_planes = 0

        # Target position for reflections
        self.target_position = np.zeros(3, dtype=np.float64)

        # Tolerance for numerical operations
        self.tolerance = 1e-7

    def set_target_position(self, node: int) -> None:
        """
        Cache the current coordinates of a node as the reflection target.

        Parameters
        ----------
        node : int
            Index of the node whose position is copied from ``self.coordinates``.
        """
        self.target_position[:] = self.coordinates[node]

    def create_reflection_planes(self, node: int, cluster_indices: list[int]) -> None:
        """
        Build reflection planes for the active cluster decisions.

        Parameters
        ----------
        node : int
            End node index for the current cluster.
        cluster_indices : list[int]
            Ordered decision-node indices. Each valid index contributes one
            reflection plane generated from three consecutive coordinates.

        Raises
        ------
        ValueError
            If ``node`` or any ``cluster_indices`` entry is out of range, or if
            plane count exceeds allocated storage.
        """
        if node < 0 or node >= self.num_nodes:
            raise ValueError(
                f"node {node} out of range [0, {self.num_nodes - 1}] for reflection planes"
            )

        self.end_node = node
        self.set_target_position(node)
        self.num_planes = 0

        for k in range(len(cluster_indices)):
            # Get cluster index - this corresponds to a node index in this case
            cluster_idx = cluster_indices[k]
            if cluster_idx < 0 or cluster_idx >= self.num_nodes:
                raise ValueError(
                    "cluster index "
                    f"{cluster_idx} out of range [0, {self.num_nodes - 1}]"
                )
            if cluster_idx < 3:
                continue

            # Create plane from three consecutive points
            point_a = self.coordinates[cluster_idx - 1]
            point_b = self.coordinates[cluster_idx - 2]
            point_c = self.coordinates[cluster_idx - 3]

            self._add_reflection_plane(point_a, point_b, point_c)

    def _add_reflection_plane(
        self,
        point_a: NDArray[np.float64],
        point_b: NDArray[np.float64],
        point_c: NDArray[np.float64],
    ) -> None:
        """Add a reflection plane defined by three points."""
        if self.num_planes >= self.num_nodes:
            raise ValueError(
                f"Too many reflection planes ({self.num_planes + 1}) for {self.num_nodes} nodes"
            )

        # Calculate vectors from point_a to the other points
        u = point_b - point_a
        v = point_c - point_a

        # Calculate normal vector via cross product
        normal = np.cross(u, v)
        normal = normalize_vector(normal)

        # Store plane point and normal
        plane_idx = self.num_planes
        self.plane_points[plane_idx] = point_a
        self.plane_normals[plane_idx] = normal

        # Calculate distance from origin to plane
        self.plane_distances[plane_idx] = float(np.dot(point_a, normal))

        self.num_planes += 1

    def reflect_single_node(self, reflection_flags: list[bool]) -> None:
        """
        Apply selected reflections to the current target node.

        Parameters
        ----------
        reflection_flags : list[bool]
            Per-plane reflection decisions where ``True`` mirrors across the
            corresponding reflection plane.

        Raises
        ------
        ValueError
            If ``reflection_flags`` is shorter than the active plane count.
        """
        if len(reflection_flags) < self.num_planes:
            raise ValueError(
                "reflection_flags length "
                f"{len(reflection_flags)} is shorter than plane count {self.num_planes}"
            )

        # Reset target node to original position
        node_idx = self.end_node
        self.coordinates[node_idx] = self.target_position

        # Apply reflections in reverse order
        for k in range(self.num_planes - 1, -1, -1):
            if reflection_flags[k]:
                self._mirror_node(k, node_idx)

    def reflect_all_nodes(
        self, cluster_indices: list[int], reflection_flags: list[bool]
    ) -> None:
        """
        Apply selected reflections to all nodes affected by a cluster.

        Parameters
        ----------
        cluster_indices : list[int]
            Decision-node indices aligned with ``reflection_flags``.
        reflection_flags : list[bool]
            Per-plane reflection decisions where ``True`` mirrors each affected
            node.

        Raises
        ------
        ValueError
            If ``reflection_flags`` is shorter than the active plane count, or
            if cluster indices are invalid/incompatible with active planes.
        """
        if len(reflection_flags) < self.num_planes:
            raise ValueError(
                "reflection_flags length "
                f"{len(reflection_flags)} is shorter than plane count {self.num_planes}"
            )

        active_cluster_indices: list[int] = []
        for cluster_idx in cluster_indices:
            if cluster_idx < 0 or cluster_idx >= self.num_nodes:
                raise ValueError(
                    "cluster index "
                    f"{cluster_idx} out of range [0, {self.num_nodes - 1}]"
                )
            if cluster_idx >= 3:
                active_cluster_indices.append(cluster_idx)

        if len(active_cluster_indices) != self.num_planes:
            raise ValueError(
                "active cluster index count "
                f"{len(active_cluster_indices)} must match plane count {self.num_planes}"
            )

        # Reset target node position
        node_idx = self.end_node
        self.coordinates[node_idx] = self.target_position

        # Apply reflections to all affected nodes
        for k in range(self.num_planes - 1, -1, -1):
            if reflection_flags[k]:
                # Reflect nodes from cluster_indices[k] to end_node
                cluster_start = active_cluster_indices[k]
                if cluster_start > self.end_node:
                    raise ValueError(
                        f"cluster start {cluster_start} must not exceed end_node {self.end_node}"
                    )
                for i in range(cluster_start, self.end_node + 1):
                    self._mirror_node(k, i)

    def _mirror_node(self, plane_idx: int, node_idx: int) -> None:
        """Mirror a node across a reflection plane."""
        # Get plane normal and distance
        normal = self.plane_normals[plane_idx]
        plane_dist = self.plane_distances[plane_idx]

        # Get node position
        node_pos = self.coordinates[node_idx]

        # Calculate reflection
        # Distance from point to plane: dot(point, normal) - plane_distance
        point_to_plane_dist = float(np.dot(node_pos, normal)) - plane_dist

        # Reflect: new_point = point - 2 * distance * normal
        reflection_offset = 2.0 * point_to_plane_dist
        self.coordinates[node_idx] -= reflection_offset * normal


class UnionFind:
    """
    Disjoint-set structure used to track merged reflection clusters.

    Parameters
    ----------
    size : int
        Number of elements managed by the disjoint-set structure.
    """

    def __init__(self, size: int) -> None:
        """Initialize with given size."""
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        self.size = size
        self.parent = list(range(size))

    def _validate_index(self, x: int) -> None:
        """Validate that x is in [0, size)."""
        if x < 0 or x >= self.size:
            raise ValueError(f"index {x} out of range [0, {self.size - 1}]")

    def find(self, x: int) -> int:
        """
        Return the representative root for an element.

        Parameters
        ----------
        x : int
            Element index to query.

        Returns
        -------
        int
            Root index for ``x`` after path compression.
        """
        self._validate_index(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]

        # Path compression in a second pass.
        while self.parent[x] != x:
            parent = self.parent[x]
            self.parent[x] = root
            x = parent

        return root

    def union(self, x: int, y: int) -> None:
        """
        Merge the set containing ``x`` into the set containing ``y``.

        Parameters
        ----------
        x : int
            Element whose root will be attached.
        y : int
            Element whose root becomes the merged-set root.
        """
        self._validate_index(x)
        self._validate_index(y)
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            self.parent[root_x] = root_y
