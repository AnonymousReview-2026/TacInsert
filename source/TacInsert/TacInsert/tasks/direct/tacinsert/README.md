# TacInsert Simulation Experiments

This directory contains the Isaac Lab direct RL implementation for TacInsert peg-in-hole simulation experiments. It defines the task assets, Gym registrations, environment logic, contact-force options, RL-Games training configurations, and utility code used for simulation-side policy training and evaluation.

Environment setup and package installation are intentionally documented in the repository-level README. This file focuses only on simulation task definitions, training usage, algorithmic mechanisms, and extension points.

## 1. Task Definitions and Benchmarks

TacInsert focuses on the insertion phase of precision assembly: the peg is already held by the Franka gripper, and the policy controls the final contact-rich motion into a target hole. The actor observes deployable quantities such as relative fingertip-hole pose, end-effector velocity, previous action, optional contact force, and optional tolerance one-hot encoding. The critic may receive privileged simulation state during training.

### 1.1 Registered Gym Tasks

The current open-source simulation set registers seven tasks:

| Gym ID | Environment config | Task config | Agent config |
| --- | --- | --- | --- |
| `TacInsert-CircleHole-I-Direct-v0` | `TacInsertCircleHole_I_Cfg` | `TacInsertCircleHole_I` | `rl_games_ppo_circle_cfg.yaml` |
| `TacInsert-SquareHole-II-Direct-v0` | `TacInsertSquareHole_II_Cfg` | `TacInsertSquareHole_II` | `rl_games_ppo_square_cfg.yaml` |
| `TacInsert-LHole-III-Direct-v0` | `TacInsertLHole_III_Cfg` | `TacInsertLHole_III` | `rl_games_ppo_L_cfg.yaml` |
| `TacInsert-TriangleHole-IV-Direct-v0` | `TacInsertTriangleHole_IV_Cfg` | `TacInsertTriangleHole_IV` | `rl_games_ppo_triangle_cfg.yaml` |
| `TacInsert-HexagonHole-IV-Direct-v0` | `TacInsertHexagonHole_IV_Cfg` | `TacInsertHexagonHole_IV` | `rl_games_ppo_hexagon_cfg.yaml` |
| `TacInsert-Manipulation-Square-SingleHole-Direct-v0` | `TacInsertManipulationSquareSingleHole_Cfg` | `TacInsertManipulationSquareSingleHole` | `rl_games_ppo_square_cfg.yaml` |
| `TacInsert-Manipulation-Circle-SingleHole-Direct-v0` | `TacInsertManipulationCircleSingleHole_Cfg` | `TacInsertManipulationCircleSingleHole` | `rl_games_ppo_circle_cfg.yaml` |

You can list the registered TacInsert environments from the repository root:

```bash
python scripts/list_envs.py
```

### 1.2 Single-Hole Benchmark

The single-hole benchmark uses one active hole asset and one held peg asset. It covers five hole shapes, with representative tolerance levels selected for this release:

| Task | Shape | Tolerance level | Nominal clearance convention |
| --- | --- | --- | --- |
| `TacInsertCircleHole_I` | Circle | Tol. I | 2 mm |
| `TacInsertSquareHole_II` | Square | Tol. II | 0.5 mm |
| `TacInsertLHole_III` | L-shape | Tol. III | 0.1 mm |
| `TacInsertTriangleHole_IV` | Triangle | Tol. IV | 0.02 mm |
| `TacInsertHexagonHole_IV` | Hexagon | Tol. IV | 0.02 mm |

Each task sets its own geometry, symmetry, yaw tolerance, reward weights, action penalty weights, initial pose randomization, and optional contact-force pathway in `tacinsert_tasks_cfg.py`.

### 1.3 ManipulationNet-Style Single-Hole Sampler Tasks

The ManipulationNet-style tasks reproduce a board-like benchmark protocol while keeping the TacInsert actor interface consistent with the single-hole setting. The board contains multiple hole shapes and tolerance levels. TacInsert maps one shape at a time into an Isaac Lab single-hole sampler:

1. A tolerance index is sampled at reset.
2. The corresponding tolerance-specific hole asset is placed at the active workspace pose.
3. The remaining tolerance-specific assets are parked away from the workspace.
4. The actor receives a tolerance one-hot code, but not the absolute board pose.

This lets one shape-specific policy train across several tolerance levels while preserving the deployable observation interface based on relative fingertip-hole pose, velocity, previous action, and optional contact force.

![ManipulationNet-style sampler](figures/manipulationnet_sampler.png)

The current release includes:

| Task | Shape | Sampler classes | Default sampling |
| --- | --- | --- | --- |
| `TacInsertManipulationSquareSingleHole` | Rectangle/square-style ManipulationNet hole | Tol. I-IV | Fixed weights from `multi_hole_sample_weights` |
| `TacInsertManipulationCircleSingleHole` | Circle ManipulationNet hole | Tol. I-IV | Fixed weights from `multi_hole_sample_weights` |

The sampler is configured through each task's `hole_sampler` dictionary. To train a mixed-tolerance policy, set non-zero weights for all desired tolerance classes, or switch `sampling_mode` to `"adaptive"` as described below.

### 1.4 Tolerance Conventions

TacInsert uses `Tol. I` through `Tol. IV` for increasingly tight insertion settings. The single-hole benchmark and the ManipulationNet-style benchmark share the same notation, but their first two tolerance levels follow different asset conventions:

| Benchmark family | Tol. I | Tol. II | Tol. III | Tol. IV |
| --- | --- | --- | --- | --- |
| Single-hole benchmark | 2 mm | 0.5 mm | 0.1 mm | 0.02 mm |
| ManipulationNet-style benchmark | 3 mm | 1 mm | 0.1 mm | 0.02 mm |

For sampler-enabled multi-clearance tasks, the tolerance one-hot code is ordered as `[Tol. I, Tol. II, Tol. III, Tol. IV]`. The one-hot code identifies the sampled tolerance class only; it does not encode the absolute board position.

### 1.5 `TacInsertTask` Configuration

`TacInsertTask` is the base task configuration class in `tacinsert_tasks_cfg.py`. Concrete task classes override its fields to define a benchmark instance.

Important asset and task fields:

| Field | Purpose |
| --- | --- |
| `name` | Internal task name used in logging and environment metadata. |
| `duration_s` | Episode duration in seconds. |
| `fixed_asset_cfg` | Hole/base asset USD, geometry scale, mass, and friction. |
| `held_asset_cfg` | Peg asset USD, geometry scale, mass, and friction. |
| `asset_size` | Characteristic peg size used by success and reset logic. |
| `robot_cfg.robot_usd` | Robot USD. The default uses the Isaac Lab Nucleus Factory Franka mimic asset. |
| `obs_order` | Actor observation fields before automatic contact-force or sampler injection. |

Episode-level pose randomization:

| Field | Effect |
| --- | --- |
| `hand_init_pos`, `hand_init_pos_range` | Nominal and randomized end-effector starting position relative to the target. |
| `hand_init_orn`, `hand_init_orn_range` | Nominal and randomized end-effector orientation. |
| `fixed_asset_init_pos_range` | Randomized hole/base placement range. |
| `fixed_asset_init_orn_deg`, `fixed_asset_init_orn_range_deg` | Nominal and randomized in-plane hole yaw. |
| `held_asset_pos_range` | Randomized peg offset in the gripper. |
| `held_asset_rot_init` | Initial held-asset rotation offset. |

Dynamics and control randomization:

| Field | Effect |
| --- | --- |
| `dr_randomize_dynamics` | Enables per-reset dynamics and controller randomization. |
| `dr_friction_range` | Randomizes relevant contact friction. |
| `dr_gains_trans_range`, `dr_gains_rot_range` | Randomizes task-space controller gains. |
| `dr_dead_zone_trans_range`, `dr_dead_zone_rot_range` | Randomizes action dead-zone thresholds. |

Reward-related fields:

| Field | Effect |
| --- | --- |
| `use_decoupled_reward` | Uses the decoupled alignment-insertion reward instead of the keypoint reward. |
| `xy_dist_coef`, `xy_dist_reward_scale` | Planar alignment shaping and scale. |
| `z_dist_coef`, `z_dist_reward_scale` | Vertical insertion shaping and scale. |
| `z_reward_gate_sharpness` | Controls how strongly XY alignment gates the insertion reward. |
| `requires_orientation_logic` | Enables symmetry-aware orientation reward and success checks. |
| `symmetry_angles_deg` | Rotational symmetry set for yaw-error folding. |
| `orientation_coef`, `orientation_reward_scale` | Orientation reward shaping and scale. |
| `yaw_success_threshold` | Orientation success threshold. |
| `action_penalty_ee_scale`, `action_grad_penalty_scale` | Penalize large actions and rapid action changes. |
| `success_threshold`, `engage_threshold`, `engage_half_threshold` | Success and engagement thresholds used by reward and logging. |

Contact-force fields:

| Field | Effect |
| --- | --- |
| `contact_force["enabled"]` | Creates the Isaac Lab `ContactSensor` path for this task. |
| `contact_force["force_source"]` | `"peg_hole"` measures peg-hole interaction; `"gripper_peg"` measures gripper-peg interaction. |
| `contact_force["use_as_obs"]` | Appends processed contact force to actor observations. |
| `contact_force["use_as_state"]` | Appends processed contact force to critic state. |
| `contact_force["use_as_reward"]` | Enables the force penalty in the reward. |
| `contact_force["log_contact_force"]` | Enables CSV logging in single-environment evaluation mode. |
| `contact_force["ema_alpha"]` | Exponential moving average factor for force smoothing. |
| `contact_force_penalty_attempt_scale` | Penalizes total force before insertion. |
| `contact_force_penalty_insertion_scale` | Penalizes lateral force during insertion. |
| `insert_depth_margin` | Separates approach and insertion stages for force reward. |

Single-hole sampler fields:

| Field | Effect |
| --- | --- |
| `hole_sampler["enabled"]` | Enables tolerance-class sampling and multi-asset placement. |
| `hole_sampler["sampling_mode"]` | `"fixed"` uses fixed weights; `"adaptive"` updates probabilities from per-tolerance success rates. |
| `hole_sampler["num_classes"]` | Number of tolerance classes, normally 4. |
| `hole_sampler["weights"]` | Optional base sampling weights. If `None`, `multi_hole_sample_weights` is used. |
| `hole_sampler["asset_usd_paths"]` | One USD hole asset per tolerance class. |
| `hole_sampler["hole_pose_table_cm_deg"]` | Board-frame hole centers and in-plane yaw in centimeters/degrees. |
| `hole_sampler["anchor_pos"]` | Active workspace anchor for the sampled hole. |
| `hole_sampler["park_pos"]` | Off-workspace parking position for inactive hole assets. |
| `hole_sampler["inject_onehot_obs"]` | Appends tolerance one-hot code to actor observations. |
| `hole_sampler["inject_onehot_state"]` | Appends tolerance one-hot code to critic state. |

Adaptive sampler parameters:

| Field | Effect |
| --- | --- |
| `adaptive_min_prob` | Minimum probability floor for each tolerance. |
| `adaptive_warmup_episodes` | Number of sampler episodes before adaptive updates start. |
| `adaptive_min_samples_per_tol` | Minimum cumulative samples required per tolerance before updates. |
| `adaptive_update_interval_episodes` | Update interval for sampling probabilities. |
| `adaptive_alpha` | Difficulty exponent applied to `(1 - success_rate)`. |
| `adaptive_beta` | Smoothing factor for probability updates. |
| `adaptive_eps` | Numerical stability constant. |

Example sampler change inside an existing concrete task class:

```python
class TacInsertManipulationSquareSingleHole(TacInsertTask):
    multi_hole_sample_weights = [0.25, 0.25, 0.25, 0.25]
    hole_sampler = {
        "enabled": True,
        "sampling_mode": "adaptive",
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
        "asset_usd_paths": [...],
        "hole_pose_table_cm_deg": [...],
        "anchor_pos": (0.6, -0.16, 0.05),
        "park_pos": (0.0, 0.0, -2.0),
        "obs_key": "tolerance_onehot",
        "inject_onehot_obs": True,
        "inject_onehot_state": True,
    }
```

When editing task dictionaries, prefer changing the concrete task class directly rather than mutating the base `TacInsertTask`.

### 1.6 `ObsRandCfg` Observation Noise Configuration

`ObsRandCfg` is defined in `tacinsert_env_cfg.py` and applies globally through `TacInsertEnvCfg.obs_rand`. It separates episode-static bias from step-wise dynamic noise.

Episode-static observation bias:

| Field | Effect |
| --- | --- |
| `fixed_asset_pos_static_error` | A sampled hole-position bias held constant for the entire episode. This emulates calibration or target-pose error. |
| `fingertip_quat_static_error` | A sampled end-effector orientation bias held constant for the episode. |

Step-wise dynamic noise:

| Field | Effect |
| --- | --- |
| `use_all_noise` | Enables dynamic Gaussian noise injection. Static bias remains active independently. |
| `fixed_asset_pos` | Dynamic target-position noise. |
| `fingertip_pos` | Dynamic fingertip-position noise. |
| `fingertip_quat` | Dynamic fingertip-orientation noise. |
| `ee_linvel` | Dynamic end-effector linear-velocity noise. |
| `ee_angvel` | Dynamic end-effector angular-velocity noise. |
| `contact_force` | Dynamic contact-force noise when contact force is enabled. |

The noise is injected while constructing observations and states. It does not move the actual simulated objects. This distinction is important: task-level randomization changes the physical scene, while `ObsRandCfg` changes what the policy observes.

### 1.7 Actor Observation Variants

The actor observation is assembled from deployable signals plus the previous smoothed action. Depending on whether contact force and tolerance encoding are enabled, common dimensions are:

| Setting | Components | Actor dimension |
| --- | --- | --- |
| Pose-only single-hole | Relative pose, end-effector velocity, previous action | 19 |
| Force-enabled single-hole | Pose-only fields plus 3D contact force | 22 or 23, depending on the task-specific quaternion fields in `obs_order` |
| Multi-clearance without force | Pose-only fields plus tolerance one-hot | 23 |
| Full force-enabled multi-clearance | Tolerance one-hot, relative pose, velocity, contact force, previous action | 26 |

The exact dimension is computed at runtime from `OBS_DIM_CFG`, `obs_order`, automatic contact-force injection, automatic sampler one-hot injection, and the 6D previous action. Always verify the printed observation space before loading an older checkpoint, because a checkpoint trained with contact force will not match a force-disabled observation space.

## 3. Training Guide

### 3.1 Basic PPO Training

Run training from the repository root:

```bash
python scripts/rl_games/train.py --task TacInsert-CircleHole-I-Direct-v0 --num_envs 128 --headless
```

Useful command-line arguments:

| Argument | Purpose |
| --- | --- |
| `--task` | Gym ID of the TacInsert task. |
| `--num_envs` | Number of parallel simulation environments. |
| `--headless` | Runs without rendering for faster training. |
| `--checkpoint` | Resumes from a specific checkpoint. |
| `--seed` | Sets the training seed. |

Training logs and checkpoints are written under:

```text
logs/rl_games/TacInsert/<experiment_name>/
```

The experiment name comes from the selected YAML file in `agents/`, for example `Circle_I`, `Square_II`, or `Hexagon_IV`.

### 3.2 Monitoring Training with TensorBoard

RL-Games writes TensorBoard summaries under each experiment directory. After starting training, open a second terminal from the repository root and point TensorBoard to the experiment's `summaries/` directory:

```bash
python -m tensorboard.main --logdir logs/rl_games/TacInsert/Circle_I/summaries
```

Then open:

```text
http://localhost:6006/
```

You can also monitor all TacInsert runs at once:

```bash
python -m tensorboard.main --logdir logs/rl_games/TacInsert
```

Useful scalar groups include:

| Scalar prefix | Meaning |
| --- | --- |
| `rewards/` or RL-Games reward scalars | PPO return and reward statistics. |
| `losses/` | Actor, critic, entropy, and optimization losses. |
| `logs_rew_*` | TacInsert reward components exported by the environment. |
| `success_rate` and `success_times` | Episode-level insertion success metrics when available. |
| `logs_sampler/*` | Per-tolerance sampler probabilities, sample counts, and success rates for sampler-enabled tasks. |

If TensorBoard is not installed in the active environment, install it in the same Python environment used for Isaac Lab training.

### 3.3 Training Other Registered Tasks

Examples:

```bash
python scripts/rl_games/train.py --task TacInsert-SquareHole-II-Direct-v0 --num_envs 128 --headless
python scripts/rl_games/train.py --task TacInsert-LHole-III-Direct-v0 --num_envs 128 --headless
python scripts/rl_games/train.py --task TacInsert-TriangleHole-IV-Direct-v0 --num_envs 128 --headless
python scripts/rl_games/train.py --task TacInsert-HexagonHole-IV-Direct-v0 --num_envs 128 --headless
python scripts/rl_games/train.py --task TacInsert-Manipulation-Square-SingleHole-Direct-v0 --num_envs 128 --headless
python scripts/rl_games/train.py --task TacInsert-Manipulation-Circle-SingleHole-Direct-v0 --num_envs 128 --headless
```

### 3.4 Evaluation and Visualization

Run policy playback:

```bash
python scripts/rl_games/play.py \
    --task TacInsert-CircleHole-I-Direct-v0 \
    --num_envs 16 \
    --checkpoint logs/rl_games/TacInsert/Circle_I/nn/TacInsert.pth
```

For single-environment evaluation with terminal success printouts and optional force logging:

```bash
python scripts/rl_games/play.py \
    --task TacInsert-LHole-III-Direct-v0 \
    --num_envs 1 \
    --checkpoint logs/rl_games/TacInsert/LHole_III/nn/TacInsert.pth
```

`play.py` sets the environment into evaluation mode. When a task enables contact-force logging and `num_envs == 1`, CSV files are written to:

```text
source/TacInsert/TacInsert/tasks/direct/tacinsert/contact_force_logs/
```

In evaluation mode, `TacInsertEnv` emits an episode summary through `infos["eval_printout"]`, and `play.py` prints it to the terminal. With `--num_envs 1`, this gives one printout per completed episode. With vectorized evaluation, the printout is an aggregate over the current environment batch:

```text
--- Episode 1 Finished ---
    Success Rate (insert_success): 87.50% (14/16)
    Average Success Time: 2.43 seconds
```

### 3.5 Optional Checkpoint Smoke Tests

Optional local checkpoints can be placed under:

```text
source/TacInsert/TacInsert/tasks/direct/tacinsert/checkpoints/
```

The checkpoint binaries are intentionally ignored by git because each file is large. For anonymous review, smoke-test checkpoints should be distributed through the paper's anonymous supplementary material or an anonymous model-hosting repository.

If an anonymous checkpoint host is provided, download them into the local checkpoint directory:

```powershell
hf download <ANONYMOUS_CHECKPOINT_REPO> TacInsert-LHole-III-Direct-v0.pth `
  --repo-type model `
  --local-dir source/TacInsert/TacInsert/tasks/direct/tacinsert/checkpoints

hf download <ANONYMOUS_CHECKPOINT_REPO> TacInsert-Manipulation-Square-SingleHole-Direct-v0.pth `
  --repo-type model `
  --local-dir source/TacInsert/TacInsert/tasks/direct/tacinsert/checkpoints
```

The checkpoint directory README documents the expected filenames, SHA256 hashes, task IDs, observation dimensions, and playback commands.

If the two validation checkpoints are present locally, run:

```powershell
python scripts/rl_games/play.py `
  --task TacInsert-LHole-III-Direct-v0 `
  --num_envs 128 `
  --headless `
  --checkpoint source/TacInsert/TacInsert/tasks/direct/tacinsert/checkpoints/TacInsert-LHole-III-Direct-v0.pth
```

Local validation of this checkpoint produced the following 128-env headless evaluation summaries:

This checkpoint matches the current `TacInsertLHole_III` configuration with contact-force observation and `rl_games_ppo_L_cfg.yaml` using `fixed_sigma: True`.

| Episode | Success rate | Average success time |
| --- | ---: | ---: |
| 1 | 97.66 percent | 5.95 s |
| 2 | 96.88 percent | 6.65 s |
| 3 | 94.53 percent | 6.08 s |

For the ManipulationNet-style square single-hole checkpoint:

```powershell
python scripts/rl_games/play.py `
  --task TacInsert-Manipulation-Square-SingleHole-Direct-v0 `
  --num_envs 128 `
  --headless `
  --checkpoint source/TacInsert/TacInsert/tasks/direct/tacinsert/checkpoints/TacInsert-Manipulation-Square-SingleHole-Direct-v0.pth
```

This checkpoint matches the current `TacInsertManipulationSquareSingleHole` configuration with contact-force observation, tolerance one-hot observation, observation dimension 26, and `rl_games_ppo_square_cfg.yaml` using `fixed_sigma: False`.

With the default fixed Tol. I sampler, local validation reached approximately 98-99 percent success over three 128-env episodes. When the same task was configured with a fixed Tol. IV sampler by setting `multi_hole_sample_weights = [0.0, 0.0, 0.0, 1.0]`, the same checkpoint reached approximately 61-63 percent success over three 128-env episodes.

For contact-force CSV logging, use `--num_envs 1` on a task with `contact_force["log_contact_force"] = True`, such as the L-hole or ManipulationNet-style square single-hole task. Logs are written to `contact_force_logs/`, which is ignored by git.

### 3.6 Training Configuration Files

RL-Games hyperparameters live in:

```text
source/TacInsert/TacInsert/tasks/direct/tacinsert/agents/
```

The current task-specific files are:

| File | Intended task family |
| --- | --- |
| `rl_games_ppo_circle_cfg.yaml` | Circle and ManipulationNet circle tasks |
| `rl_games_ppo_square_cfg.yaml` | Square and ManipulationNet square tasks |
| `rl_games_ppo_L_cfg.yaml` | L-shape task |
| `rl_games_ppo_triangle_cfg.yaml` | Triangle task |
| `rl_games_ppo_hexagon_cfg.yaml` | Hexagon task |

Common fields to adjust:

| YAML field | Purpose |
| --- | --- |
| `params.config.name` | Log root group, currently `TacInsert`. |
| `params.config.full_experiment_name` | Experiment folder name. |
| `params.config.max_epochs` | Total PPO training epochs. |
| `params.config.minibatch_size` | PPO minibatch size. |
| `params.config.horizon_length` | Rollout horizon. |
| `params.config.learning_rate` | PPO optimizer learning rate. |
| `params.config.normalize_input` | Observation normalization. |
| `params.network` | Actor-critic network architecture. |
| `params.network.space.continuous.fixed_sigma` | Enables globally shared policy variance when set to `True`. |

The training script also supports adaptive sampler checkpoint selection for sampler-enabled tasks. During training, it can save a balanced checkpoint based on per-tolerance success metrics in addition to standard RL-Games checkpoints.

### 3.7 Training Budget and Evaluation Protocol

The released YAML files default to `num_actors: 128`, `horizon_length: 128`, and `seq_length: 128`. Some paper experiments used larger batches, such as 256 or 512 parallel environments, for faster wall-clock iteration. You can reproduce that style by increasing `--num_envs` if GPU memory and contact-sensor cost permit:

```bash
python scripts/rl_games/train.py --task TacInsert-LHole-III-Direct-v0 --num_envs 256 --headless
```

For controlled simulation evaluation, use the same task and policy settings as training, then run several full episodes. A practical protocol is:

```bash
python scripts/rl_games/play.py \
    --task TacInsert-LHole-III-Direct-v0 \
    --num_envs 128 \
    --headless \
    --checkpoint logs/rl_games/TacInsert/LHole_III/nn/TacInsert.pth
```

For sampler-enabled multi-clearance tasks, evaluate each tolerance separately by setting a one-hot fixed sampling distribution in the concrete task config. This avoids mixing Tol. I-IV in one aggregate success rate and makes per-tolerance comparisons easier to interpret.

When `sampling_mode: "adaptive"` is enabled, `train.py` also maintains an optional balanced checkpoint:

```text
logs/rl_games/TacInsert/<experiment_name>/nn/best_balanced.pth
logs/rl_games/TacInsert/<experiment_name>/nn/best_balanced_meta.json
```

This checkpoint is selected from per-tolerance sampler statistics rather than only from the aggregate RL-Games score. It is useful when the mean score is high but one tight tolerance still performs poorly.

### 3.8 Enabling Globally Shared Policy Variance

TacInsert uses RL-Games continuous Gaussian policies. The policy predicts the action mean, while the standard deviation can either be state-dependent or globally shared across observations. In force-conditioned insertion, abrupt contact-force transients can make state-dependent exploration noisy. Setting `fixed_sigma: True` uses a learned state-independent sigma vector shared by all observations, which keeps contact force in the mean-action pathway while preventing contact state from directly changing the exploration scale.

Enable globally shared variance in the selected YAML file:

```yaml
params:
  network:
    space:
      continuous:
        sigma_activation: None
        sigma_init:
          name: const_initializer
          val: 0
        fixed_sigma: True
```

Use `fixed_sigma: True` for force-conditioned or tight-tolerance tasks where training stability matters, especially when `contact_force["use_as_obs"]` is enabled. Use `fixed_sigma: False` only when intentionally testing a state-dependent exploration baseline.

## 4. Core Algorithmic Mechanisms

### 4.1 Action Interface and Control

The action space is six-dimensional:

```text
[dx, dy, dz, droll, dpitch, dyaw]
```

Actions are interpreted as bounded end-effector pose deltas. The raw policy action is first smoothed with an exponential moving average:

```text
smoothed_action = ema_factor * raw_action + (1 - ema_factor) * previous_smoothed_action
```

The default `CtrlCfg.ema_factor` is `0.2`. The smoothed translational channels are scaled by `CtrlCfg.pos_action_threshold`, and the rotational channels are scaled by `CtrlCfg.rot_action_threshold`. The resulting fingertip target is clipped to the workspace box defined by `CtrlCfg.pos_action_bounds`. Roll and pitch are reset to the vertical insertion orientation, while yaw remains controllable.

Important action and controller fields:

| Field | Default | Purpose |
| --- | --- | --- |
| `CtrlCfg.ema_factor` | `0.2` | Smooths policy actions before pose integration. |
| `CtrlCfg.pos_action_threshold` | `[0.02, 0.02, 0.02]` m | Per-step translational action scale. |
| `CtrlCfg.rot_action_threshold` | `[0.097, 0.097, 0.097]` rad | Per-step rotational action scale. |
| `CtrlCfg.pos_action_bounds` | `[0.05, 0.05, 0.05]` m | Workspace clipping bound around the perceived hole position. |
| `CtrlCfg.default_task_prop_gains` | `[100, 100, 100, 30, 30, 30]` | Default task-space controller proportional gains. |

`tacinsert_control.py` implements the low-level control utilities used by the environment, including differential inverse kinematics and operational-space control. `CtrlCfg` also defines reset joint targets, reset gains, and null-space gains.

### 4.2 Asymmetric Actor-Critic Observations

TacInsert uses an asymmetric actor-critic interface:

- The actor receives deployable observations from `obs_order`, plus optional `contact_force` and optional `tolerance_onehot`.
- The critic receives a richer state defined by `TacInsertEnvCfg.state_order`, including privileged simulation quantities such as absolute object poses, joint positions, randomized control parameters, and optional contact force.

The observation dimensions are computed from `OBS_DIM_CFG` and `STATE_DIM_CFG`, so adding a new observation key requires updating both the dimension dictionary and the logic that fills `obs_dict` or `state_dict` in `tacinsert_env.py`.

### 4.3 Decoupled Alignment-Insertion Reward

Most TacInsert tasks enable `use_decoupled_reward=True`. This reward separates planar alignment from vertical insertion:

- `xy_align` rewards planar alignment between the peg base and the target insertion pose.
- `z_insert` rewards insertion depth.
- `z_insert` is gated by planar alignment using `exp(-z_reward_gate_sharpness * xy_dist)`.
- If orientation logic is enabled, insertion reward is additionally gated by a yaw-orientation threshold.
- `orientation` uses task-specific rotational symmetries, such as 90 degrees for square holes or 60 degrees for hexagonal holes.
- `curr_engaged`, `curr_engaged_half`, and `curr_success` provide sparse progress and success rewards.
- Action magnitude and action change penalties discourage abrupt motions.

This decomposition discourages the policy from exploiting vertical motion before the peg is sufficiently aligned with the hole.

### 4.4 Baseline Keypoint Reward

The original keypoint reward path is still available by setting `use_decoupled_reward=False`. It computes distances between corresponding keypoints on the held peg and target pose and applies coarse/fine shaping terms. This path is useful for comparisons or for tasks where keypoint-distance shaping is sufficient.

### 4.5 Contact Force Processing

Contact force is optional and task-controlled. When enabled, the environment:

1. Creates a `ContactSensor` in `TacInsertEnvCfg.__post_init__`.
2. Selects a force source through `contact_force["force_source"]`.
3. Applies a first-step tare to remove initial bias.
4. Smooths the signal with EMA using `contact_force["ema_alpha"]`.
5. Optionally injects it into actor observation and critic state.
6. Optionally uses it as a force penalty in the reward.
7. Optionally logs it to CSV during single-environment evaluation.

The reward penalty separates approach and insertion:

- Before the peg reaches the insertion-depth margin, total force is penalized.
- After the peg enters the insertion region, lateral force is penalized more directly.

This encourages compliant approach behavior while discouraging high lateral load during insertion.

### 4.6 Globally Shared Policy Variance

For continuous actions, RL-Games represents the actor as a diagonal Gaussian policy. With `fixed_sigma: False`, the exploration scale may depend on the current observation. This can be problematic for contact-rich insertion because force observations can change abruptly during contact transitions, causing the policy's exploration variance to fluctuate with contact state.

TacInsert's stable force-conditioned setting uses:

```yaml
params.network.space.continuous.fixed_sigma: True
```

Under this setting, the actor still conditions the action mean on the full observation, including smoothed contact force when enabled, but the policy variance is a learned global vector shared across all observations. This separates contact-conditioned correction from contact-conditioned exploration scale. In practice, this is most important when contact force is part of the actor observation and the task is close to the tight-tolerance regime.

The related force-stabilization settings are:

| Mechanism | Configuration |
| --- | --- |
| Temporal force smoothing | `contact_force["ema_alpha"]` in `TacInsertTask` |
| Globally shared variance | `params.network.space.continuous.fixed_sigma: True` in the RL-Games YAML |
| Force observation injection | `contact_force["use_as_obs"]` in `TacInsertTask` |

### 4.7 Tolerance-Adaptive Sampler

Sampler-enabled tasks use `hole_sampler` to choose which tolerance asset is active at reset. In fixed mode, probabilities come from `hole_sampler["weights"]` or `multi_hole_sample_weights`.

In adaptive mode, TacInsert updates the sampling probabilities from observed per-tolerance success rates:

1. Warm up until `adaptive_warmup_episodes` is reached.
2. Wait until every tolerance has at least `adaptive_min_samples_per_tol` samples.
3. Compute difficulty as `1 - success_rate`.
4. Reweight the base distribution by `difficulty ** adaptive_alpha`.
5. Smooth updates with `adaptive_beta`.
6. Project probabilities to keep every class above `adaptive_min_prob`.

The intent is to keep all tolerance levels represented while reallocating more simulation experience to harder tolerances.

### 4.8 Stability Metrics Used in Paper-Style Ablations

For force-processing and variance ablations, the paper summarizes training stability with several scalar metrics. These are not required to run the environment, but they are useful when reproducing experiments:

| Metric | Meaning |
| --- | --- |
| `S_max` | Best success rate reached during training. |
| `S_final` | Final-window success rate near the end of training. |
| `T_50` | First training progress point where success reaches 50 percent. |
| `AUC` | Mean success over the training curve, useful for combining sample efficiency and late-stage retention. |

In practice, these metrics can be computed from TensorBoard scalar exports or from logged evaluation success-rate curves. For adaptive sampler tasks, inspect both aggregate success and per-tolerance `logs_sampler/success_rate_tol_*` curves.

## 6. Code Framework and Extension

### 6.1 File Layout

| File or directory | Role |
| --- | --- |
| `__init__.py` | Registers the seven Gym environments and maps them to RL-Games YAML files. |
| `tacinsert_env.py` | Main `TacInsertEnv` implementation: scene setup, stepping, observations, rewards, resets, sampler logic, contact-force processing, and evaluation logs. |
| `tacinsert_env_cfg.py` | Isaac Lab environment configuration, simulation settings, robot configuration, contact sensor configuration, `ObsRandCfg`, and concrete env config classes. |
| `tacinsert_tasks_cfg.py` | Task definitions, asset paths, reward parameters, domain randomization parameters, contact-force options, and sampler configuration. |
| `tacinsert_control.py` | Task-space control, torque computation, and differential IK helpers. |
| `tacinsert_utils.py` | Pose utilities, reward shaping helpers, symmetry-aware yaw handling, and observation packing. |
| `tactile_datalogger.py` | Generic episode-scoped CSV logger used for contact-force logging. |
| `agents/` | RL-Games PPO configuration files. |
| `assets/` | Open-source USD assets for TacInsert simulation tasks. |
| `figures/` | Images used by this simulation README. |

### 6.2 Adding a New Single-Hole Task

1. Add the hole and peg USD files under `assets/<shape>/`.
2. Define a `FixedAssetCfg` subclass for the hole.
3. Define a `HeldAssetCfg` subclass for the peg.
4. Define a `TacInsertTask` subclass that sets assets, geometry size, episode duration, randomization ranges, reward parameters, and symmetry angles.
5. Define a `TacInsert..._Cfg` subclass in `tacinsert_env_cfg.py`.
6. Register the Gym ID in `__init__.py`.
7. Add or reuse an RL-Games YAML file in `agents/`.

Minimum task skeleton:

```python
@configclass
class MyHole(FixedAssetCfg):
    usd_path = f"{TACINSERT_ASSETS_DIR}/my_shape/my_hole.usd"
    diameter = 0.010
    height = 0.025


@configclass
class MyPeg(HeldAssetCfg):
    usd_path = f"{TACINSERT_ASSETS_DIR}/my_shape/my_peg.usd"
    diameter = 0.0095
    height = 0.050


@configclass
class TacInsertMyHole_II(TacInsertTask):
    name = "tacinsert_my_hole_II"
    fixed_asset_cfg = MyHole()
    held_asset_cfg = MyPeg()
    asset_size = held_asset_cfg.diameter
    duration_s = 15.0
    use_decoupled_reward = True
    requires_orientation_logic = True
    symmetry_angles_deg = [0.0, 180.0]
```

### 6.3 Adding a New ManipulationNet-Style Sampler Task

1. Add one hole USD per tolerance level.
2. Set `is_multi_hole_task=True`.
3. Set `hole_sampler["enabled"]=True`.
4. Provide `asset_usd_paths` in tolerance order.
5. Provide a board-frame pose table through `hole_pose_table_cm_deg`.
6. Enable `inject_onehot_obs` and usually `inject_onehot_state`.
7. Choose fixed or adaptive sampling.

For a multi-tolerance training run, set the concrete task fields to non-zero weights for all tolerance classes while keeping the task's existing asset paths and pose table:

```diff
-    multi_hole_sample_weights = [1.0, 0.0, 0.0, 0.0]
+    multi_hole_sample_weights = [0.25, 0.25, 0.25, 0.25]
     hole_sampler = {
         "enabled": True,
-        "sampling_mode": "fixed",
+        "sampling_mode": "adaptive",
         ...
     }
```

For per-tolerance evaluation, set a one-hot fixed distribution in the concrete task:

```diff
+    multi_hole_sample_weights = [0.0, 0.0, 0.0, 1.0]
     hole_sampler = {
         "enabled": True,
+        "sampling_mode": "fixed",
         ...
     }
```

### 6.4 Common Validation Checklist

After adding or modifying a task:

```bash
python scripts/list_envs.py
python -m py_compile source/TacInsert/TacInsert/tasks/direct/tacinsert/*.py
```

Then run a small visual or headless smoke test:

```bash
python scripts/rl_games/play.py --task TacInsert-CircleHole-I-Direct-v0 --num_envs 1
```

For training changes, start with a short run and inspect:

- observation and state dimensions
- reward component logs
- success rate
- sampler probability logs, if sampler is enabled
- contact-force magnitude and sign, if contact force is enabled
