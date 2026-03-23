"""
legged_gym/envs/g1/multi_variant_env.py

MultiVariantG1Robot — loads N robot variants into one Isaac Gym sim.
Envs are distributed evenly across variants.
Exposes env_variant_ids (N,) tensor so runner/obs_builder can select
per-variant data at step time.
"""

import os
import json
import numpy as np
import torch

from isaacgym import gymapi
from isaacgym.torch_utils import to_torch, torch_rand_float

from legged_gym.envs.g1.g1_env import G1Robot
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR


class MultiVariantG1Robot(G1Robot):
    """
    Identical to G1Robot except _create_envs loads multiple URDF assets
    and distributes envs evenly across them.

    Extra attributes set after __init__:
        self.env_variant_ids   : (num_envs,) long tensor — variant index per env
        self.variant_names     : list[str]
        self.variant_meta      : list[dict]  — metadata per variant
        self.num_variants      : int
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device,
                 headless, variants_metadata_path):
        """
        variants_metadata_path: path to variants_metadata.json produced by
                                 generate_variant_urdfs.py
        """
        # Load metadata before super().__init__ so _create_envs can use it
        with open(variants_metadata_path) as f:
            meta = json.load(f)

        self.variant_names = list(meta.keys())
        self.variant_meta  = list(meta.values())
        self.num_variants  = len(self.variant_names)

        print(f"[MultiVariantG1Robot] Loading {self.num_variants} variants:")
        for i, (name, m) in enumerate(zip(self.variant_names, self.variant_meta)):
            print(f"  [{i}] {name}  base_height_target={m['base_height_target']}")

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _create_envs(self):
        """
        Override: loads one asset per variant, distributes envs evenly,
        stores env_variant_ids.
        """
        # ── Asset options (same as base) ──────────────────────────────────
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints  = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = \
            self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link           = self.cfg.asset.fix_base_link
        asset_options.density                 = self.cfg.asset.density
        asset_options.angular_damping         = self.cfg.asset.angular_damping
        asset_options.linear_damping          = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity    = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity     = self.cfg.asset.max_linear_velocity
        asset_options.armature                = self.cfg.asset.armature
        asset_options.thickness               = self.cfg.asset.thickness
        asset_options.disable_gravity         = self.cfg.asset.disable_gravity

        # ── Load one asset per variant ────────────────────────────────────
        robot_assets = []
        for i, meta in enumerate(self.variant_meta):
            urdf_path  = meta["urdf"]
            asset_root = os.path.dirname(urdf_path)
            asset_file = os.path.basename(urdf_path)
            asset = self.gym.load_asset(
                self.sim, asset_root, asset_file, asset_options)
            robot_assets.append(asset)
            print(f"[MultiVariantG1Robot] Loaded asset [{i}]: {asset_file}")

        # Use first asset to get shared properties
        ref_asset = robot_assets[0]
        self.num_dof    = self.gym.get_asset_dof_count(ref_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(ref_asset)
        dof_props_asset         = self.gym.get_asset_dof_properties(ref_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(ref_asset)

        body_names = self.gym.get_asset_rigid_body_names(ref_asset)
        self.dof_names  = self.gym.get_asset_dof_names(ref_asset)
        self.num_bodies = len(body_names)
        self.num_dofs   = len(self.dof_names)

        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = (self.cfg.init_state.pos +
                                self.cfg.init_state.rot +
                                self.cfg.init_state.lin_vel +
                                self.cfg.init_state.ang_vel)
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False)

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs          = []
        env_variant_ids    = []

        # ── Distribute envs evenly across variants ────────────────────────
        # e.g. 512 envs, 6 variants → 85 or 86 envs per variant
        base_per   = self.num_envs // self.num_variants
        remainder  = self.num_envs  % self.num_variants
        variant_counts = [
            base_per + (1 if i < remainder else 0)
            for i in range(self.num_variants)
        ]
        # Build flat assignment: env i → variant id
        variant_assignment = []
        for vid, count in enumerate(variant_counts):
            variant_assignment.extend([vid] * count)

        print(f"[MultiVariantG1Robot] Env distribution: {variant_counts}")

        for i in range(self.num_envs):
            vid          = variant_assignment[i]
            robot_asset  = robot_assets[vid]
            env_variant_ids.append(vid)

            start_pose = gymapi.Transform()
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(
                -1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper,
                int(np.sqrt(self.num_envs)))

            rigid_shape_props = self._process_rigid_shape_props(
                rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(
                robot_asset, rigid_shape_props)

            actor_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose,
                self.cfg.asset.name, i,
                self.cfg.asset.self_collisions, 0)

            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(
                env_handle, actor_handle, dof_props)

            body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, actor_handle, body_props, recomputeInertia=True)

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        # ── Store variant assignment ───────────────────────────────────────
        self.env_variant_ids = torch.tensor(
            env_variant_ids, dtype=torch.long, device=self.device)

        # ── Foot / contact indices (same for all variants — same topology) ─
        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long,
            device=self.device, requires_grad=False)
        for i, fn in enumerate(feet_names):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], fn)

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names), dtype=torch.long,
            device=self.device, requires_grad=False)
        for i, cn in enumerate(penalized_contact_names):
            self.penalised_contact_indices[i] = \
                self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], cn)

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names), dtype=torch.long,
            device=self.device, requires_grad=False)
        for i, cn in enumerate(termination_contact_names):
            self.termination_contact_indices[i] = \
                self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], cn)

        print(f"[MultiVariantG1Robot] Created {self.num_envs} envs "
              f"across {self.num_variants} variants.")