# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from distutils.util import strtobool

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="rl_games_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="Run training with multiple GPUs or nodes.",
)
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Path to model checkpoint."
)
parser.add_argument(
    "--sigma", type=str, default=None, help="The policy's initial standard deviation."
)
parser.add_argument(
    "--max_iterations", type=int, default=None, help="RL Policy training iterations."
)
parser.add_argument(
    "--wandb-project-name", type=str, default=None, help="the wandb's project name"
)
parser.add_argument(
    "--wandb-entity",
    type=str,
    default=None,
    help="the entity (team) of wandb's project",
)
parser.add_argument(
    "--wandb-name", type=str, default=None, help="the name of wandb's run"
)
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="if toggled, this experiment will be tracked with Weights and Biases",
)
parser.add_argument(
    "--export_io_descriptors",
    action="store_true",
    default=False,
    help="Export IO descriptors.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import math
import os
import random
from datetime import datetime

import omni
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rl_games import (
    MultiObserver,
    PbtAlgoObserver,
    RlGamesGpuEnv,
    RlGamesVecEnvWrapper,
)

import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config

import TacInsert.tasks as tacinsert_tasks

_REGISTERED_TASK_MODULES = (isaaclab_tasks, tacinsert_tasks)


class AdaptiveBalancedCheckpointObserver(IsaacAlgoObserver):
    """Save an extra balanced checkpoint for adaptive tolerance-sampler tasks."""

    def __init__(
        self,
        enabled: bool,
        min_samples_per_tol: int = 8,
        score_weights: tuple[float, float] = (0.7, 0.3),
        score_margin: float = 0.002,
        ema_alpha: float = 0.2,
        use_ema: bool = True,
        ckpt_stem: str = "best_balanced",
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.min_samples_per_tol = int(min_samples_per_tol)
        self.w_bal = float(score_weights[0])
        self.w_min = float(score_weights[1])
        self.score_margin = float(score_margin)
        self.ema_alpha = float(ema_alpha)
        self.use_ema = bool(use_ema)
        self.ckpt_stem = ckpt_stem

        self._last_sr = [None, None, None, None]
        self._last_count = [None, None, None, None]
        self._ema_sr = None

        self._best_score = float("-inf")
        self._best_s_bal = float("-inf")
        self._best_s_min = float("-inf")
        self._best_epoch = -1
        self._best_frame = -1

    @staticmethod
    def _to_float(value):
        if value is None:
            return None
        try:
            if hasattr(value, "item"):
                return float(value.item())
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _extract_value(infos: dict, key: str):
        if key in infos:
            return infos[key]
        episode = infos.get("episode", None)
        if isinstance(episode, dict) and key in episode:
            return episode[key]
        return None

    def process_infos(self, infos, done_indices):
        super().process_infos(infos, done_indices)

        if not self.enabled or not isinstance(infos, dict):
            return

        for i in range(4):
            sr_val = self._to_float(
                self._extract_value(infos, f"logs_sampler/success_rate_tol_{i}")
            )
            cnt_val = self._to_float(
                self._extract_value(infos, f"logs_sampler/sample_count_tol_{i}")
            )

            if sr_val is not None:
                self._last_sr[i] = sr_val
            if cnt_val is not None:
                self._last_count[i] = cnt_val

    def _ready_for_balanced_update(self) -> bool:
        if not self.enabled:
            return False
        if any(v is None for v in self._last_sr):
            return False
        if any(v is None for v in self._last_count):
            return False
        return all(v >= self.min_samples_per_tol for v in self._last_count)

    def _compute_score(self):
        sr_raw = [float(v) for v in self._last_sr]
        if self.use_ema:
            if self._ema_sr is None:
                self._ema_sr = sr_raw[:]
            else:
                self._ema_sr = [
                    (1.0 - self.ema_alpha) * self._ema_sr[i]
                    + self.ema_alpha * sr_raw[i]
                    for i in range(4)
                ]
            sr_eval = self._ema_sr
        else:
            sr_eval = sr_raw

        s_bal = sum(sr_eval) / 4.0
        s_min = min(sr_eval)
        score = self.w_bal * s_bal + self.w_min * s_min
        return sr_raw, sr_eval, s_bal, s_min, score

    def _is_improved(self, s_bal: float, s_min: float, score: float) -> bool:
        if self._best_epoch < 0:
            return True

        eps = 1e-12
        if s_min > self._best_s_min + eps:
            return True
        if s_min < self._best_s_min - eps:
            return False
        if score > self._best_score + self.score_margin:
            return True
        if score < self._best_score - eps:
            return False
        return s_bal > self._best_s_bal + eps

    def _save_balanced_checkpoint(
        self,
        epoch_num: int,
        frame: int,
        sr_raw,
        sr_eval,
        s_bal: float,
        s_min: float,
        score: float,
    ):
        if not hasattr(self, "algo"):
            return
        nn_dir = getattr(self.algo, "nn_dir", None)
        if not nn_dir:
            return
        os.makedirs(nn_dir, exist_ok=True)

        self.algo.save(os.path.join(nn_dir, self.ckpt_stem))

        meta = {
            "epoch": int(epoch_num),
            "frame": int(frame),
            "sr_raw": [float(x) for x in sr_raw],
            "sr_eval": [float(x) for x in sr_eval],
            "sample_count": [float(x) for x in self._last_count],
            "s_bal": float(s_bal),
            "s_min": float(s_min),
            "score": float(score),
            "min_samples_per_tol": int(self.min_samples_per_tol),
            "use_ema": bool(self.use_ema),
            "ema_alpha": float(self.ema_alpha),
            "score_weights": [self.w_bal, self.w_min],
            "score_margin": float(self.score_margin),
        }
        with open(
            os.path.join(nn_dir, "best_balanced_meta.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(meta, f, indent=2)

        self._best_score = score
        self._best_s_bal = s_bal
        self._best_s_min = s_min
        self._best_epoch = int(epoch_num)
        self._best_frame = int(frame)
        print(
            f"[ADAPTIVE-BALANCED] Updated best_balanced at epoch={epoch_num}, frame={frame}, "
            f"S_min={s_min:.4f}, S_bal={s_bal:.4f}, Score={score:.4f}"
        )

    def after_print_stats(self, frame, epoch_num, total_time):
        super().after_print_stats(frame, epoch_num, total_time)

        if not self._ready_for_balanced_update():
            return

        sr_raw, sr_eval, s_bal, s_min, score = self._compute_score()
        if self._is_improved(s_bal, s_min, score):
            self._save_balanced_checkpoint(
                epoch_num, frame, sr_raw, sr_eval, s_bal, s_min, score
            )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict
):
    """Train with RL-Games agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = (
        args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    )
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations
        if args_cli.max_iterations is not None
        else agent_cfg["params"]["config"]["max_epochs"]
    )
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(
            f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}"
        )
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    # multi-gpu training config
    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    config_name = agent_cfg["params"]["config"]["name"]
    log_root_path = os.path.join("logs", "rl_games", config_name)
    if "pbt" in agent_cfg:
        if agent_cfg["pbt"]["directory"] == ".":
            log_root_path = os.path.abspath(log_root_path)
        else:
            log_root_path = os.path.join(agent_cfg["pbt"]["directory"], log_root_path)

    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = agent_cfg["params"]["config"].get(
        "full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
    wandb_project = (
        config_name
        if args_cli.wandb_project_name is None
        else args_cli.wandb_project_name
    )
    experiment_name = log_dir if args_cli.wandb_name is None else args_cli.wandb_name

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(
        env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups
    )

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(
            config_name, num_actors, **kwargs
        ),
    )
    env_configurations.register(
        "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env}
    )

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games

    sampler_cfg = getattr(getattr(env_cfg, "task", None), "hole_sampler", {})
    adaptive_sampler_enabled = bool(sampler_cfg.get("enabled", False)) and (
        str(sampler_cfg.get("sampling_mode", "fixed")).lower() == "adaptive"
    )
    if args_cli.distributed and getattr(app_launcher, "global_rank", 0) != 0:
        adaptive_sampler_enabled = False

    balanced_observer = AdaptiveBalancedCheckpointObserver(
        enabled=adaptive_sampler_enabled,
        min_samples_per_tol=max(8, int(env_cfg.scene.num_envs * 0.03)),
        score_weights=(0.7, 0.3),
        score_margin=0.002,
        ema_alpha=0.2,
        use_ema=True,
        ckpt_stem="best_balanced",
    )

    if "pbt" in agent_cfg and agent_cfg["pbt"]["enabled"]:
        observers = MultiObserver(
            [balanced_observer, PbtAlgoObserver(agent_cfg, args_cli)]
        )
        runner = Runner(observers)
    else:
        runner = Runner(balanced_observer)

    runner.load(agent_cfg)

    # reset the agent and env
    runner.reset()
    # train the agent

    global_rank = int(os.getenv("RANK", "0"))
    if args_cli.track and global_rank == 0:
        if args_cli.wandb_entity is None:
            raise ValueError(
                "Weights and Biases entity must be specified for tracking."
            )
        import wandb

        wandb.init(
            project=wandb_project,
            entity=args_cli.wandb_entity,
            name=experiment_name,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        if not wandb.run.resumed:
            wandb.config.update({"env_cfg": env_cfg.to_dict()})
            wandb.config.update({"agent_cfg": agent_cfg})

    if args_cli.checkpoint is not None:
        runner.run(
            {
                "train": True,
                "play": False,
                "sigma": train_sigma,
                "checkpoint": resume_path,
            }
        )
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
