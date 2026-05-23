import os
from dataclasses import field

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


TACINSERT_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"


def contact_force_cfg(
    *,
    enabled: bool = False,
    force_source: str = "peg_hole",
    use_as_obs: bool = False,
    use_as_state: bool = False,
    use_as_reward: bool = False,
    log_contact_force: bool = False,
    ema_alpha: float = 0.25,
) -> dict:
    return {
        "enabled": enabled,
        "force_source": force_source,
        "use_as_obs": use_as_obs,
        "use_as_state": use_as_state,
        "use_as_reward": use_as_reward,
        "log_contact_force": log_contact_force,
        "ema_alpha": ema_alpha,
    }


@configclass
class FixedAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0
    height: float = 0.0
    base_height: float = 0.0
    friction: float = 0.75
    mass: float = 0.05


@configclass
class HeldAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0
    height: float = 0.0
    friction: float = 0.75
    mass: float = 0.05


@configclass
class RobotCfg:
    robot_usd: str = f"{ASSET_DIR}/franka_mimic.usd"
    franka_fingerpad_length: float = 0.017608
    friction: float = 0.75


@configclass
class TacInsertTask:
    robot_cfg: RobotCfg = RobotCfg()
    name: str = ""
    duration_s: float = 5.0

    fixed_asset_cfg: FixedAssetCfg = FixedAssetCfg()
    held_asset_cfg: HeldAssetCfg = HeldAssetCfg()
    asset_size: float = 0.0

    hand_init_pos: list = [0.0, 0.0, 0.047]
    hand_init_pos_range: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0.0, 0.0]
    hand_init_orn_range: list = [0.0, 0.0, 0.0]

    fixed_asset_init_pos_range: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 0.0
    grasp_yaw_comp_deg: float = 0.0

    held_asset_pos_range: list = [0.003, 0.0, 0.003]
    held_asset_rot_init: float = 0.0

    dr_randomize_dynamics: bool = True
    dr_friction_range: list = [0.3, 0.6]
    dr_gains_trans_range: list = [100.0, 300.0]
    dr_gains_rot_range: list = [30.0, 50.0]
    dr_dead_zone_trans_range: list = [0.0, 0.2]
    dr_dead_zone_rot_range: list = [0.0, 0.04]

    action_penalty_ee_scale: float = 0.008
    action_grad_penalty_scale: float = 0.05

    num_keypoints: int = 4
    keypoint_scale: float = 0.15
    keypoint_coef_baseline: list = [5, 4]
    kp_baseline_scale: float = 1.0
    keypoint_coef_coarse: list = [50, 2]
    kp_coarse_scale: float = 1.0
    keypoint_coef_fine: list = [100, 0]
    kp_fine_scale: float = 1.0

    success_threshold: float = 0.04
    success_threshold_scale: float = 1.0
    engage_threshold: float = 0.9
    engage_threshold_scale: float = 1.0
    engage_half_threshold: float = 0.55
    engage_half_threshold_scale: float = 1.0

    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.0
    z_dist_coef: list = [20, 4]
    z_dist_reward_scale: float = 2.5
    z_reward_gate_sharpness: float = 100.0

    orientation_reward_scale: float = 3.0
    yaw_success_threshold: float = 0.05
    symmetry_angles_deg: list = []
    orientation_coef: list = [5, 4]
    orientation_reward_threshold: float = 1.0

    requires_orientation_logic: bool = False
    use_decoupled_reward: bool = False
    show_visual_markers: bool = True

    contact_force: dict = contact_force_cfg()
    contact_force_penalty_attempt_scale: float = 0.01
    contact_force_penalty_insertion_scale: float = 0.03
    insert_depth_margin: float = 0.002

    is_multi_hole_task: bool = False
    multi_hole_shape: str = ""
    multi_hole_sample_weights: list = []
    hole_sampler: dict = {
        "enabled": False,
        "sampling_mode": "fixed",
        "shape": "",
        "num_classes": 0,
        "weights": None,
        "adaptive_min_prob": 0.10,
        "adaptive_warmup_episodes": 300,
        "adaptive_min_samples_per_tol": 200,
        "adaptive_update_interval_episodes": 20,
        "adaptive_alpha": 1.5,
        "adaptive_beta": 0.2,
        "adaptive_eps": 1e-6,
        "asset_usd_paths": [],
        "hole_pose_table_cm_deg": [],
        "anchor_pos": (0.0, 0.0, 0.0),
        "park_pos": (0.0, 0.0, -2.0),
        "obs_key": "tolerance_onehot",
        "inject_onehot_obs": False,
        "inject_onehot_state": False,
    }

    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "fingertip_quat_rel_fixed",
        "ee_linvel",
        "ee_angvel",
    ]

    fixed_asset: ArticulationCfg = field(default=None, init=False)
    held_asset: ArticulationCfg = field(default=None, init=False)
    fixed_assets_multi: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self.fixed_asset = self._make_fixed_asset_cfg(
            prim_path="/World/envs/env_.*/FixedAsset",
            usd_path=self.fixed_asset_cfg.usd_path,
            pos=(0.6, 0.0, 0.05),
        )
        self.held_asset = self._make_held_asset_cfg()

        if self.hole_sampler.get("enabled", False):
            sampler_cfg = self.hole_sampler
            asset_usd_paths = sampler_cfg["asset_usd_paths"]
            if len(asset_usd_paths) != sampler_cfg["num_classes"]:
                raise ValueError(
                    "hole_sampler['asset_usd_paths'] length mismatch: "
                    f"{len(asset_usd_paths)} vs num_classes={sampler_cfg['num_classes']}"
                )
            self.fixed_assets_multi = [
                self._make_fixed_asset_cfg(
                    prim_path=f"/World/envs/env_.*/FixedAssetTol{i}",
                    usd_path=usd_path,
                    pos=sampler_cfg["park_pos"],
                )
                for i, usd_path in enumerate(asset_usd_paths)
            ]
        else:
            self.fixed_assets_multi = []

    def _make_fixed_asset_cfg(
        self, prim_path: str, usd_path: str, pos: tuple[float, float, float]
    ) -> ArticulationCfg:
        return ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=192,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=self.fixed_asset_cfg.mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.0005, rest_offset=0.0
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=pos,
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={},
                joint_vel={},
            ),
            actuators={},
        )

    def _make_held_asset_cfg(self) -> ArticulationCfg:
        return ArticulationCfg(
            prim_path="/World/envs/env_.*/HeldAsset",
            spawn=sim_utils.UsdFileCfg(
                usd_path=self.held_asset_cfg.usd_path,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=192,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=self.held_asset_cfg.mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.0005, rest_offset=0.0
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.4, 0.1),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={},
                joint_vel={},
            ),
            actuators={},
        )


@configclass
class CircleHole(FixedAssetCfg):
    usd_path: str = f"{TACINSERT_ASSETS_DIR}/circle/circle_hole.usd"
    diameter: float = 0.010
    height: float = 0.025
    base_height: float = 0.0


@configclass
class CirclePeg_I(HeldAssetCfg):
    usd_path: str = f"{TACINSERT_ASSETS_DIR}/circle/circle_peg_I.usd"
    diameter: float = 0.008
    height: float = 0.050
    mass: float = 0.009


@configclass
class TacInsertCircleHole_I(TacInsertTask):
    name: str = "tacinsert_circle_I"
    fixed_asset_cfg: FixedAssetCfg = CircleHole()
    held_asset_cfg: HeldAssetCfg = CirclePeg_I()
    asset_size: float = held_asset_cfg.diameter
    duration_s: float = 10.0

    hand_init_orn_range: list = [0.0, 0.0, 0.2]
    fixed_asset_init_orn_range_deg: float = 30.0
    z_reward_gate_sharpness: float = 100.0
    action_penalty_ee_scale: float = 0.3
    action_grad_penalty_scale: float = 0.6

    use_decoupled_reward: bool = True
    requires_orientation_logic: bool = True
    symmetry_angles_deg: list = [0.0, 90.0, 180.0, 270.0]
    yaw_success_threshold: float = 0.5
    orientation_reward_scale: float = 3.0
    orientation_coef: list = [5, 4]
    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "fingertip_quat_rel_fixed",
        "ee_linvel",
        "ee_angvel",
    ]


@configclass
class SquareHole(FixedAssetCfg):
    usd_path: str = f"{TACINSERT_ASSETS_DIR}/square/square_hole.usd"
    diameter: float = 0.010
    height: float = 0.025
    base_height: float = 0.0


@configclass
class SquarePeg_II(HeldAssetCfg):
    usd_path: str = f"{TACINSERT_ASSETS_DIR}/square/square_peg_II.usd"
    diameter: float = 0.0095
    height: float = 0.050
    mass: float = 0.024


@configclass
class TacInsertSquareHole_II(TacInsertTask):
    name: str = "tacinsert_square_II"
    fixed_asset_cfg: FixedAssetCfg = SquareHole()
    held_asset_cfg: HeldAssetCfg = SquarePeg_II()
    asset_size: float = held_asset_cfg.diameter
    duration_s: float = 15.0

    hand_init_orn_range: list = [0.0, 0.0, 0.785]
    fixed_asset_init_orn_range_deg: float = 90.0
    action_penalty_ee_scale: float = 0.03
    action_grad_penalty_scale: float = 0.1

    orientation_reward_scale: float = 3.0
    yaw_success_threshold: float = 0.05
    symmetry_angles_deg: list = [0.0, 90.0, 180.0, 270.0]
    orientation_coef: list = [5, 4]

    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.0
    z_dist_coef: list = [20, 4]
    z_dist_reward_scale: float = 2.0
    z_reward_gate_sharpness: float = 100.0

    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True


@configclass
class LHole(FixedAssetCfg):
    usd_path: str = f"{TACINSERT_ASSETS_DIR}/L_hole/L_hole.usd"
    diameter: float = 0.015
    height: float = 0.025
    base_height: float = 0.0


@configclass
class LPeg_III(HeldAssetCfg):
    usd_path: str = f"{TACINSERT_ASSETS_DIR}/L_hole/L_peg_III.usd"
    diameter: float = 0.0149
    height: float = 0.050
    mass: float = 0.022


@configclass
class TacInsertLHole_III(TacInsertTask):
    name: str = "tacinsert_L_III"
    fixed_asset_cfg: FixedAssetCfg = LHole()
    held_asset_cfg: HeldAssetCfg = LPeg_III()
    asset_size: float = held_asset_cfg.diameter
    duration_s: float = 20.0

    hand_init_orn_range: list = [0.0, 0.0, 0.2]
    fixed_asset_init_orn_deg: float = 180.0
    fixed_asset_init_orn_range_deg: float = 30.0
    action_penalty_ee_scale: float = 0.05
    action_grad_penalty_scale: float = 0.1

    orientation_reward_scale: float = 2.5
    yaw_success_threshold: float = 0.1
    orientation_reward_threshold: float = 5.0
    symmetry_angles_deg: list = [0.0]
    orientation_coef: list = [50, 2]

    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.5
    z_dist_coef: list = [50, 1]
    z_dist_reward_scale: float = 2.5
    z_reward_gate_sharpness: float = 100.0

    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True
    contact_force: dict = contact_force_cfg(
        enabled=True,
        force_source="gripper_peg",
        use_as_obs=True,
        use_as_state=True,
        use_as_reward=True,
        log_contact_force=True,
    )


@configclass
class TriangleHole(FixedAssetCfg):
    usd_path: str = f"{TACINSERT_ASSETS_DIR}/triangle/triangle_hole.usd"
    diameter: float = 0.012
    height: float = 0.025
    base_height: float = 0.0


@configclass
class TrianglePeg_IV(HeldAssetCfg):
    usd_path: str = f"{TACINSERT_ASSETS_DIR}/triangle/triangle_peg_IV.usd"
    diameter: float = 0.01196
    height: float = 0.050
    mass: float = 0.019


@configclass
class TacInsertTriangleHole_IV(TacInsertTask):
    name: str = "tacinsert_triangle_IV"
    fixed_asset_cfg: FixedAssetCfg = TriangleHole()
    held_asset_cfg: HeldAssetCfg = TrianglePeg_IV()
    asset_size: float = held_asset_cfg.diameter
    duration_s: float = 20.0

    hand_init_orn_range: list = [0.0, 0.0, 0.2]
    fixed_asset_init_orn_deg: float = 180.0
    fixed_asset_init_orn_range_deg: float = 30.0
    action_penalty_ee_scale: float = 0.03
    action_grad_penalty_scale: float = 0.1

    orientation_reward_scale: float = 3.0
    yaw_success_threshold: float = 0.005
    symmetry_angles_deg: list = [0.0, 120.0, 240.0]
    orientation_coef: list = [5, 4]

    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.0
    z_dist_coef: list = [20, 4]
    z_dist_reward_scale: float = 2.5
    z_reward_gate_sharpness: float = 100.0

    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True


@configclass
class HexagonHole(FixedAssetCfg):
    usd_path: str = f"{TACINSERT_ASSETS_DIR}/hexagon/hexagon_hole.usd"
    diameter: float = 0.012
    height: float = 0.025
    base_height: float = 0.0


@configclass
class HexagonPeg_IV(HeldAssetCfg):
    usd_path: str = f"{TACINSERT_ASSETS_DIR}/hexagon/hexagon_peg_IV.usd"
    diameter: float = 0.011976
    height: float = 0.050
    mass: float = 0.019


@configclass
class TacInsertHexagonHole_IV(TacInsertTask):
    name: str = "tacinsert_hexagon_IV"
    fixed_asset_cfg: FixedAssetCfg = HexagonHole()
    held_asset_cfg: HeldAssetCfg = HexagonPeg_IV()
    asset_size: float = held_asset_cfg.diameter
    duration_s: float = 20.0

    hand_init_orn_range: list = [0.0, 0.0, 0.2]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 30.0
    action_penalty_ee_scale: float = 0.03
    action_grad_penalty_scale: float = 0.1

    orientation_reward_scale: float = 3.0
    yaw_success_threshold: float = 0.005
    symmetry_angles_deg: list = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    orientation_coef: list = [5, 4]

    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.0
    z_dist_coef: list = [20, 4]
    z_dist_reward_scale: float = 2.5
    z_reward_gate_sharpness: float = 100.0

    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True


@configclass
class ManipulationSquarePeg(HeldAssetCfg):
    usd_path: str = (
        f"{TACINSERT_ASSETS_DIR}/manipulation-net/atb_m1_rectangular_peg.usd"
    )
    diameter: float = 0.015
    height: float = 0.050
    mass: float = 0.03


@configclass
class ManipulationCirclePeg(HeldAssetCfg):
    usd_path: str = f"{TACINSERT_ASSETS_DIR}/manipulation-net/atb_m1_cylinder_peg.usd"
    diameter: float = 0.0118
    height: float = 0.050
    mass: float = 0.009


@configclass
class ManipulationRectangleHole(FixedAssetCfg):
    usd_path: str = (
        f"{TACINSERT_ASSETS_DIR}/manipulation-net/rectangle/rectangular_hole_I.usd"
    )
    diameter: float = 0.015
    height: float = 0.022
    base_height: float = 0.0
    mass: float = 1.0
    friction: float = 1.0


@configclass
class ManipulationCircleHole(FixedAssetCfg):
    usd_path: str = f"{TACINSERT_ASSETS_DIR}/manipulation-net/circle/circle_hole_I.usd"
    diameter: float = 0.015
    height: float = 0.022
    base_height: float = 0.0
    mass: float = 1.0
    friction: float = 1.0


@configclass
class TacInsertManipulationSquareSingleHole(TacInsertTask):
    name: str = "tacinsert_manipulation_square_single_hole"
    fixed_asset_cfg: FixedAssetCfg = ManipulationRectangleHole()
    held_asset_cfg: HeldAssetCfg = ManipulationSquarePeg()
    asset_size: float = held_asset_cfg.diameter
    duration_s: float =20.0

    is_multi_hole_task: bool = True
    multi_hole_shape: str = "rectangle"
    multi_hole_sample_weights: list = [0.0, 0.0, 0.0, 1.0]
    hole_sampler: dict = {
        "enabled": True,
        "sampling_mode": "fixed",
        "shape": "rectangle",
        "num_classes": 4,
        "weights": None,
        "adaptive_min_prob": 0.10,
        "adaptive_warmup_episodes": 120,
        "adaptive_min_samples_per_tol": 300,
        "adaptive_update_interval_episodes": 10,
        "adaptive_alpha": 2.0,
        "adaptive_beta": 0.3,
        "adaptive_eps": 1e-6,
        "asset_usd_paths": [
            f"{TACINSERT_ASSETS_DIR}/manipulation-net/rectangle/rectangular_hole_I.usd",
            f"{TACINSERT_ASSETS_DIR}/manipulation-net/rectangle/rectangular_hole_II.usd",
            f"{TACINSERT_ASSETS_DIR}/manipulation-net/rectangle/rectangular_hole_III.usd",
            f"{TACINSERT_ASSETS_DIR}/manipulation-net/rectangle/rectangular_hole_IV.usd",
        ],
        "hole_pose_table_cm_deg": [
            [22.091064, 2.996730, 90.0],
            [22.091064, 8.634977, 0.0],
            [22.091064, 13.752822, 30.0],
            [22.091064, 19.000000, -30.0],
        ],
        "anchor_pos": (0.6, -0.16, 0.05),
        "park_pos": (0.0, 0.0, -2.0),
        "obs_key": "tolerance_onehot",
        "inject_onehot_obs": True,
        "inject_onehot_state": True,
    }

    fixed_asset_init_orn_deg: float = 180.0
    fixed_asset_init_orn_range_deg: float = 0.0
    hand_init_orn_range: list = [0.0, 0.0, 0.2]

    action_penalty_ee_scale: float = 0.05
    action_grad_penalty_scale: float = 0.1
    orientation_reward_scale: float = 2.5
    yaw_success_threshold: float = 0.1
    orientation_reward_threshold: float = 5.0
    symmetry_angles_deg: list = [0.0]
    orientation_coef: list = [50, 2]

    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.5
    z_dist_coef: list = [50, 1]
    z_dist_reward_scale: float = 2.5
    z_reward_gate_sharpness: float = 100.0

    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True
    contact_force: dict = contact_force_cfg(
        enabled=True,
        force_source="gripper_peg",
        use_as_obs=True,
        use_as_state=True,
        use_as_reward=True,
        log_contact_force=True,
    )
    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat_rel_fixed",
        "ee_linvel",
        "ee_angvel",
    ]

    def __post_init__(self):
        super().__post_init__()
        self.fixed_asset.init_state.pos = self.hole_sampler["anchor_pos"]


@configclass
class TacInsertManipulationCircleSingleHole(TacInsertTask):
    name: str = "tacinsert_manipulation_circle_single_hole"
    fixed_asset_cfg: FixedAssetCfg = ManipulationCircleHole()
    held_asset_cfg: HeldAssetCfg = ManipulationCirclePeg()
    asset_size: float = held_asset_cfg.diameter
    duration_s: float = 20.0

    is_multi_hole_task: bool = True
    multi_hole_shape: str = "circle"
    multi_hole_sample_weights: list = [0.0, 1.0, 0.0, 0.0]
    hole_sampler: dict = {
        "enabled": True,
        "sampling_mode": "fixed",
        "shape": "circle",
        "num_classes": 4,
        "weights": None,
        "adaptive_min_prob": 0.10,
        "adaptive_warmup_episodes": 120,
        "adaptive_min_samples_per_tol": 300,
        "adaptive_update_interval_episodes": 10,
        "adaptive_alpha": 2.0,
        "adaptive_beta": 0.3,
        "adaptive_eps": 1e-6,
        "asset_usd_paths": [
            f"{TACINSERT_ASSETS_DIR}/manipulation-net/circle/circle_hole_I.usd",
            f"{TACINSERT_ASSETS_DIR}/manipulation-net/circle/circle_hole_II.usd",
            f"{TACINSERT_ASSETS_DIR}/manipulation-net/circle/circle_hole_III.usd",
            f"{TACINSERT_ASSETS_DIR}/manipulation-net/circle/circle_hole_IV.usd",
        ],
        "hole_pose_table_cm_deg": [
            [3.21623, 2.996673, 0.0],
            [3.21623, 8.634977, 0.0],
            [3.21623, 13.752822, 0.0],
            [3.21623, 19.000000, 0.0],
        ],
        "anchor_pos": (0.6, -0.16, 0.05),
        "park_pos": (0.0, 0.0, -2.0),
        "obs_key": "tolerance_onehot",
        "inject_onehot_obs": True,
        "inject_onehot_state": True,
    }

    fixed_asset_init_orn_deg: float = 180.0
    fixed_asset_init_orn_range_deg: float = 0.0
    hand_init_orn_range: list = [0.0, 0.0, 0.2]

    action_penalty_ee_scale: float = 0.05
    action_grad_penalty_scale: float = 0.1
    orientation_reward_scale: float = 2.5
    yaw_success_threshold: float = 0.1
    orientation_reward_threshold: float = 5.0
    symmetry_angles_deg: list = [0.0]
    orientation_coef: list = [50, 2]

    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.5
    z_dist_coef: list = [50, 1]
    z_dist_reward_scale: float = 2.5
    z_reward_gate_sharpness: float = 100.0

    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True
    contact_force: dict = contact_force_cfg(enabled=False)
    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat_rel_fixed",
        "ee_linvel",
        "ee_angvel",
    ]

    def __post_init__(self):
        super().__post_init__()
        self.fixed_asset.init_state.pos = self.hole_sampler["anchor_pos"]
