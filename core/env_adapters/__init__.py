"""
Environment Adapters for different simulation backends and real-world robots.

This module provides a unified interface for:
- CALVIN (PyBullet)
- LIBERO/LIBERO-PRO (MuJoCo/Robosuite)
- Manipulator custom MuJoCo
- ManiSkill (SAPIEN/PhysX)
- Real-world robots

Usage:
    from core.env_adapters import create_adapter
    
    adapter = create_adapter("calvin", env_config)
    # or
    adapter = create_adapter("libero", env_config)
    # or
    adapter = create_adapter("mujoco", env_config)
"""

from .base_adapter import BaseEnvAdapter, Pose3D, CameraParams, TrackedObject, InteractableObject
from .calvin_adapter import CalvinAdapter
from .libero_adapter import LiberoAdapter
from .manipulator_mujoco_adapter import ManipulatorMujocoAdapter


def create_calvin_env(env_config: dict):
    from calvin_env.envs.play_table_env import PlayTableSimEnv
    from omegaconf import OmegaConf
    
    # Convert dict to OmegaConf for Hydra instantiation
    cfg = OmegaConf.create(env_config)
    
    env = PlayTableSimEnv(
        robot_cfg=cfg.get('robot_cfg', None),
        scene_cfg=cfg.get('scene_cfg', None),
        cameras=cfg.get('cameras', {}),
        seed=cfg.get('seed', 0),
        use_vr=cfg.get('use_vr', False),
        bullet_time_step=cfg.get('bullet_time_step', 240),
        show_gui=cfg.get('show_gui', False),
        use_scene_info=cfg.get('use_scene_info', True),
        use_egl=cfg.get('use_egl', True),
        cubes_table_only=cfg.get('cubes_table_only', False),
        control_freq=cfg.get('control_freq', 30),
    )
    return env

# Note: LiberoAdapter creates its own environments internally via create_libero_envs()

def create_manipulator_env(env_config: dict):
    from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv
    return UR5GraspEnv(env_config)


def create_adapter(backend: str, env_config: dict, **kwargs) -> BaseEnvAdapter:
    """
    Factory function to create the appropriate adapter.
    
    Args:
        backend: One of "calvin", "libero", "mujoco"
        env_config: Environment configuration dict
        **kwargs: Additional arguments for the adapter
    
    Returns:
        BaseEnvAdapter instance
    """
    backend = backend.lower()

    if backend == "calvin":
        env = create_calvin_env(env_config)
        return CalvinAdapter(env, env_config, **kwargs)
    elif backend == "libero":
        # LiberoAdapter creates environments internally
        return LiberoAdapter(None, env_config, **kwargs)
    elif backend in ("mujoco", "manipulator"):
        env = create_manipulator_env(env_config)
        return ManipulatorMujocoAdapter(env, env_config, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Supported: calvin, libero, mujoco")


__all__ = [
    "BaseEnvAdapter",
    "Pose3D",
    "CameraParams",
    "TrackedObject",
    "InteractableObject",
    "CalvinAdapter",
    "LiberoAdapter",
    "ManipulatorMujocoAdapter",
    "create_adapter",
]
