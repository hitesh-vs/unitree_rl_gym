"""G1 robot environment with graph-based observation encoding.

Extends the base :class:`~legged_gym.envs.g1.g1_env.G1Robot` environment to
include:

1. **Morphological context** – 7 static per-DOF features (position limits,
   velocity limit, torque limit, PD stiffness, PD damping, default joint
   position) appended to the proprioceptive observation for every step.
2. **Graph data** – kinematic node features and normalised adjacency matrix
   parsed from the robot URDF, enabling the Transformer+GCN policy to reason
   over the robot's topology and physical structure.
"""

import math
import os

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.g1.g1_env import G1Robot
from graphs.parser import build_graph_tensors


class G1GraphRobot(G1Robot):
    """G1 robot environment augmented with kinematic graph and morphological
    context information.

    On top of the standard G1Robot observations, this class:

    1. Appends a static per-DOF morphological observation (position limits,
       velocity limit, torque limit, PD stiffness, PD damping, default
       position) to every proprioceptive observation vector, giving the
       policy direct access to the robot's physical properties.
    2. Parses the robot URDF to build a kinematic graph.
    3. Computes topology node features (one-hot or topological encodings)
       augmented with per-link inertial properties (mass, diagonal inertia,
       centre-of-mass position).
    4. Precomputes the symmetrically-normalised adjacency matrix
       ``D^{-1/2} A D^{-1/2}`` once at startup.
    5. Exposes graph tensors via :meth:`get_graph_tensors` and the morpho
       tensor via :meth:`get_morpho_tensors` for use by downstream adapters
       and training scripts.

    All static data (morphological context and graph tensors) is stored as
    GPU tensors to avoid repeated host-device transfers.

    Observation layout (total: ``num_observations = 47 + num_actions × 7``):

    .. code-block:: none

        [0:3]      base angular velocity  (scaled)
        [3:6]      projected gravity
        [6:9]      velocity commands (scaled)
        [9:21]     joint positions − default_pos (scaled)
        [21:33]    joint velocities (scaled)
        [33:45]    previous actions
        [45]       sin(2π · phase)
        [46]       cos(2π · phase)
        ── per-DOF morphological context (7 features × 12 DOFs = 84) ──
        [47:131]   [lower_lim, upper_lim, vel_lim, torque_lim, kp, kd, default_pos]
                   repeated for each actuated DOF, all normalised to [-1, 1]

    The graph encoding mode is controlled by the ``graph_mode`` attribute on
    the environment config, defaulting to ``'topological'``.  Valid values
    are ``'none'``, ``'onehot'``, and ``'topological'``.
    """

    #: Number of static morphological features appended per actuated DOF:
    #: lower pos limit, upper pos limit, velocity limit, torque limit,
    #: PD stiffness (kp), PD damping (kd), default joint position.
    _MORPHO_FEATURES_PER_DOF = 7

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # Read graph mode before super().__init__ so it is available during
        # _init_buffers → _init_graph_data which is called from the parent.
        self.graph_mode = getattr(cfg, 'graph_mode', 'topological')

        # Expand the observation space to accommodate per-DOF morphological
        # context.  This must happen before super().__init__() allocates
        # self.obs_buf, which uses cfg.env.num_observations.
        cfg.env.num_observations += cfg.env.num_actions * self._MORPHO_FEATURES_PER_DOF

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    # ------------------------------------------------------------------
    # Buffer / graph / morphology initialisation
    # ------------------------------------------------------------------

    def _init_buffers(self):
        super()._init_buffers()
        self._init_graph_data()
        self._init_morpho_obs()

    def _init_graph_data(self):
        """Build graph tensors from the robot URDF.

        Parses the URDF configured in ``self.cfg.asset.file``, computes
        topology node features according to :attr:`graph_mode`, augments
        them with per-link inertial properties (mass, diagonal inertia,
        centre-of-mass), and precomputes the normalised adjacency matrix.
        All tensors are moved to ``self.device``.

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
            f"feature_dim={self.graph_tensors['feature_dim']} "
            f"(topology + 7 inertial features per node)"
        )

    def _init_morpho_obs(self):
        """Build a static per-DOF morphological observation tensor.

        Collects 7 physical properties for each of the ``num_actions``
        actuated DOFs from the IsaacGym simulator buffers (already set by
        the parent ``_init_buffers`` / ``_process_dof_props`` calls) and
        normalises each feature:

        .. code-block:: none

            col 0 – lower position limit   (÷ π, clipped to [−1, 1])
            col 1 – upper position limit   (÷ π, clipped to [−1, 1])
            col 2 – velocity limit          (÷ max velocity limit)
            col 3 – torque limit            (÷ max torque limit)
            col 4 – PD stiffness (kp)       (÷ max kp)
            col 5 – PD damping  (kd)        (÷ max kd)
            col 6 – default joint position  (÷ π, clipped to [−1, 1])

        The resulting ``(1, num_actions × 7)`` tensor is stored as
        ``self.morpho_obs`` and broadcast across all parallel environments
        inside :meth:`compute_observations`.

        This method also truncates ``self.noise_scale_vec`` to the
        proprioceptive observation length.  The parent
        :meth:`~legged_gym.envs.g1.g1_env.G1Robot.compute_observations`
        constructs a ``(num_envs, 47)`` obs tensor and then applies noise
        using ``self.noise_scale_vec``.  Because we expanded
        ``num_observations`` to 131, the pre-built noise vector has 131
        elements; truncating it to 47 keeps broadcasting correct while
        ensuring no noise is added to the static morphological context.
        """
        n = self.num_actions  # number of actuated DOFs

        lower       = self.dof_pos_limits[:n, 0]
        upper       = self.dof_pos_limits[:n, 1]
        vel_lim     = self.dof_vel_limits[:n]
        torque_lim  = self.torque_limits[:n]
        kp          = self.p_gains          # shape: (num_actions,)
        kd          = self.d_gains          # shape: (num_actions,)
        default_pos = self.default_dof_pos.squeeze(0)[:n]

        def _safe_norm(x: torch.Tensor) -> torch.Tensor:
            """Divide by max absolute value; return as-is if all-zero."""
            x_max = x.abs().max()
            return x / x_max if x_max > 0 else x

        lower_n      = (lower       / math.pi).clamp(-1.0, 1.0)
        upper_n      = (upper       / math.pi).clamp(-1.0, 1.0)
        vel_lim_n    = _safe_norm(vel_lim)
        torque_lim_n = _safe_norm(torque_lim)
        kp_n         = _safe_norm(kp)
        kd_n         = _safe_norm(kd)
        default_n    = (default_pos / math.pi).clamp(-1.0, 1.0)

        # Stack → (num_actions, 7), then flatten → (1, num_actions * 7)
        morpho = torch.stack(
            [lower_n, upper_n, vel_lim_n, torque_lim_n, kp_n, kd_n, default_n],
            dim=-1,
        )
        self.morpho_obs = morpho.view(1, -1)  # (1, num_actions * 7)

        # Truncate noise_scale_vec to the proprioceptive dimension.
        # The parent's compute_observations() creates self.obs_buf with
        # only the proprioceptive entries (47 dims) before applying noise:
        #   self.obs_buf += (2 * rand - 1) * self.noise_scale_vec
        # After the expansion, noise_scale_vec has 131 elements which would
        # cause a shape mismatch.  Indices 47-130 are already zero (morpho
        # context should not be noised), so a simple truncation is safe.
        #
        # NOTE: self.num_obs is already the expanded value (131) because
        # __init__ mutates cfg.env.num_observations *before* calling
        # super().__init__(), which is where BaseTask sets self.num_obs.
        proprio_dim = self.num_obs - n * self._MORPHO_FEATURES_PER_DOF
        self.noise_scale_vec = self.noise_scale_vec[:proprio_dim]

    # ------------------------------------------------------------------
    # Observation computation
    # ------------------------------------------------------------------

    def compute_observations(self):
        """Compute observations including static per-DOF morphological context.

        Delegates to the parent method to fill ``self.obs_buf`` with the
        standard 47-dim proprioceptive observation (with noise if enabled),
        then appends the pre-computed static morphological context to
        produce a ``(num_envs, num_observations)`` tensor.
        """
        # Parent fills self.obs_buf with shape (num_envs, 47) and applies noise
        super().compute_observations()

        # Broadcast morpho_obs (1, 84) → (num_envs, 84) and concatenate
        morpho = self.morpho_obs.expand(self.num_envs, -1)
        self.obs_buf = torch.cat([self.obs_buf, morpho], dim=-1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_graph_tensors(self):
        """Return the precomputed graph tensors.

        Returns:
            dict or None: A dict with keys ``'node_features'``,
                ``'adj_normalized'``, ``'num_nodes'``, and ``'feature_dim'``
                (all on ``self.device``), or ``None`` if the URDF was not
                found during initialisation.  Node features combine topology
                encodings with 7 per-link inertial features (mass, diagonal
                inertia, COM position).
        """
        return self.graph_tensors

    def get_morpho_tensors(self):
        """Return the static per-DOF morphological observation tensor.

        Returns:
            torch.Tensor or None: Tensor of shape
                ``(1, num_actions × 7)`` ready for broadcasting across all
                environments, or ``None`` if :meth:`_init_morpho_obs` has
                not yet been called.
        """
        return getattr(self, 'morpho_obs', None)
