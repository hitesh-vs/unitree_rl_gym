"""G1 robot environment with graph-based observation encoding.

Extends the base :class:`~legged_gym.envs.g1.g1_env.G1Robot` environment to
include graph data (node features and normalised adjacency matrix) in the
observation tensors, enabling the Transformer+GCN policy.
"""

import os

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.g1.g1_env import G1Robot
from graphs.parser import build_graph_tensors


class G1GraphRobot(G1Robot):
    """G1 robot environment augmented with kinematic graph information.

    On top of the standard G1Robot observations, this class:

    1. Parses the robot URDF to build a kinematic graph.
    2. Computes node features (one-hot or topological encodings).
    3. Precomputes the symmetrically-normalised adjacency matrix
       ``D^{-1/2} A D^{-1/2}`` once at startup.
    4. Exposes the graph tensors via :meth:`get_graph_tensors` for use by
       :class:`~legged_gym.algos.ppo.graph_obs_adapter.GraphObsAdapter`.

    All graph data is static (robot topology does not change) and is stored
    as GPU tensors to avoid repeated host-device transfers.

    The graph encoding mode is controlled by ``cfg.graph_mode``, which
    defaults to ``'onehot'``.  Valid values are ``'none'``,
    ``'onehot'``, and ``'topological'``.
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # Read graph mode before super().__init__ so it is available during
        # _init_buffers → _init_graph_data which is called from the parent.
        self.graph_mode = getattr(cfg, 'graph_mode', 'onehot')
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    # ------------------------------------------------------------------
    # Buffer / graph initialisation
    # ------------------------------------------------------------------

    def _init_buffers(self):
        super()._init_buffers()
        self._init_graph_data()

    def _init_graph_data(self):
        """Build graph tensors from the robot URDF.

        Parses the URDF configured in ``self.cfg.asset.file``, computes node
        features according to :attr:`graph_mode`, and precomputes the
        normalised adjacency matrix.  All tensors are moved to
        ``self.device``.

        If the URDF file is not found, a warning is printed and
        :attr:`graph_tensors` is set to ``None`` so that the environment
        still functions without graph support.
        """
        urdf_path = self.cfg.asset.file.format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR
        )

        if not os.path.isfile(urdf_path):
            print(
                f"[G1GraphRobot] Warning: URDF not found at '{urdf_path}'. "
                "Graph data will be unavailable."
            )
            self.graph_tensors = None
            return

        print(
            f"[G1GraphRobot] Building graph from URDF: {urdf_path} "
            f"(mode='{self.graph_mode}')"
        )
        self.graph_tensors = build_graph_tensors(
            urdf_path=urdf_path,
            mode=self.graph_mode,
            device=self.device,
        )
        print(
            f"[G1GraphRobot] Graph ready: {self.graph_tensors['num_nodes']} nodes, "
            f"feature_dim={self.graph_tensors['feature_dim']}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_graph_tensors(self):
        """Return the precomputed graph tensors.

        Returns:
            dict or None: A dict with keys ``'node_features'``,
                ``'adj_normalized'``, ``'num_nodes'``, and ``'feature_dim'``
                (all on ``self.device``), or ``None`` if the URDF was not
                found during initialisation.
        """
        return self.graph_tensors
