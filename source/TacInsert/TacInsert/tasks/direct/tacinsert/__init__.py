import gymnasium as gym

from . import agents
from .tacinsert_env_cfg import (
    TacInsertCircleHole_I_Cfg,
    TacInsertHexagonHole_IV_Cfg,
    TacInsertLHole_III_Cfg,
    TacInsertManipulationCircleSingleHole_Cfg,
    TacInsertManipulationSquareSingleHole_Cfg,
    TacInsertSquareHole_II_Cfg,
    TacInsertTriangleHole_IV_Cfg,
)


def _register(task_id: str, env_cfg_entry_point, rl_games_cfg: str):
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.tacinsert_env:TacInsertEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg_entry_point,
            "rl_games_cfg_entry_point": f"{agents.__name__}:{rl_games_cfg}",
        },
    )


_register(
    "TacInsert-CircleHole-I-Direct-v0",
    TacInsertCircleHole_I_Cfg,
    "rl_games_ppo_circle_cfg.yaml",
)
_register(
    "TacInsert-SquareHole-II-Direct-v0",
    TacInsertSquareHole_II_Cfg,
    "rl_games_ppo_square_cfg.yaml",
)
_register(
    "TacInsert-LHole-III-Direct-v0",
    TacInsertLHole_III_Cfg,
    "rl_games_ppo_L_cfg.yaml",
)
_register(
    "TacInsert-TriangleHole-IV-Direct-v0",
    TacInsertTriangleHole_IV_Cfg,
    "rl_games_ppo_triangle_cfg.yaml",
)
_register(
    "TacInsert-HexagonHole-IV-Direct-v0",
    TacInsertHexagonHole_IV_Cfg,
    "rl_games_ppo_hexagon_cfg.yaml",
)
_register(
    "TacInsert-Manipulation-Square-SingleHole-Direct-v0",
    TacInsertManipulationSquareSingleHole_Cfg,
    "rl_games_ppo_square_cfg.yaml",
)
_register(
    "TacInsert-Manipulation-Circle-SingleHole-Direct-v0",
    TacInsertManipulationCircleSingleHole_Cfg,
    "rl_games_ppo_circle_cfg.yaml",
)
