import os
from typing import Any, Optional

import mujoco
import mujoco.viewer
import numpy as np
import spatialmath as sm
from scipy.spatial.transform import Rotation as R

from manipulator_grasp.arm.robot import Robot, UR5e
from manipulator_grasp.utils import mj


class UR5GraspEnv:
    def __init__(self, config: Optional[dict[str, Any]] = None):
        cfg = config or {}

        self.sim_hz = int(cfg.get("sim_hz", 500))
        self.control_hz = int(cfg.get("control_hz", 50))
        self.frame_skip = max(1, self.sim_hz // max(1, self.control_hz))

        self.height = int(cfg.get("observation_height", 256))
        self.width = int(cfg.get("observation_width", 256))
        self.visualization_height = int(cfg.get("visualization_height", 640))
        self.visualization_width = int(cfg.get("visualization_width", 640))
        self.fovy = float(cfg.get("camera_fovy", np.pi / 4.0))

        self.camera_name = cfg.get("camera_name", "cam")
        self.render_viewer = bool(cfg.get("render", False))
        self.max_episode_steps = int(cfg.get("max_episode_steps", 300))

        self.task_name = cfg.get("task_name", "pick_and_place_cylinder")
        self.target_object = cfg.get("target_object", "Cylinder")
        self.target_zone = cfg.get("target_zone", "zone_drop")
        self.success_distance = float(cfg.get("success_distance", 0.08))

        self.scene_path = cfg.get(
            "scene_path",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "scenes", "scene.xml"),
        )

        self.mj_model: Optional[mujoco.MjModel] = None
        self.mj_data: Optional[mujoco.MjData] = None
        self.robot: Optional[Robot] = None
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        self.mj_renderer: Optional[mujoco.Renderer] = None
        self.mj_depth_renderer: Optional[mujoco.Renderer] = None
        self.mj_seg_renderer: Optional[mujoco.Renderer] = None
        self.mj_viewer: Optional[mujoco.viewer.Handle] = None

        self.step_num = 0

    @property
    def action_dim(self) -> int:
        return 7

    def _ensure_initialized(self):
        if self.mj_model is None or self.mj_data is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

    def _get_body_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_initialized()
        body = self.mj_data.body(body_name)
        return body.xpos.copy(), body.xquat.copy()

    def _get_joint_positions(self) -> np.ndarray:
        self._ensure_initialized()
        return np.array([float(mj.get_joint_q(self.mj_model, self.mj_data, jn)) for jn in self.joint_names], dtype=np.float32)

    def _set_joint_positions(self, q: np.ndarray):
        self._ensure_initialized()
        for i, jn in enumerate(self.joint_names):
            mj.set_joint_q(self.mj_model, self.mj_data, jn, float(q[i]))

    def _build_state_vector(self) -> np.ndarray:
        q = self._get_joint_positions()
        ee_pos, ee_quat = self._get_body_pose("flange")
        target_pos, _ = self._get_body_pose(self.target_object)
        zone_pos, _ = self._get_body_pose(self.target_zone)
        box_pos, _ = self._get_body_pose("Box")
        t_pos, _ = self._get_body_pose("T_block")
        grip = float(np.clip(self.mj_data.ctrl[6] / 255.0, 0.0, 1.0))

        state = np.zeros(32, dtype=np.float32)
        cursor = 0
        state[cursor:cursor + 6] = q
        cursor += 6
        state[cursor:cursor + 3] = ee_pos
        cursor += 3
        state[cursor:cursor + 4] = ee_quat
        cursor += 4
        state[cursor:cursor + 3] = target_pos
        cursor += 3
        state[cursor:cursor + 3] = zone_pos
        cursor += 3
        state[cursor:cursor + 3] = box_pos
        cursor += 3
        state[cursor:cursor + 3] = t_pos
        cursor += 3
        state[cursor] = grip
        return state

    def _check_success(self) -> tuple[bool, str]:
        obj_pos, _ = self._get_body_pose(self.target_object)
        zone_pos, _ = self._get_body_pose(self.target_zone)
        dist = float(np.linalg.norm(obj_pos - zone_pos))
        if dist <= self.success_distance:
            return True, f"{self.target_object.lower()}_to_{self.target_zone.lower()}"
        return False, "no_behavior"

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)

        if self.mj_viewer is not None:
            self.mj_viewer.close()
            self.mj_viewer = None

        self.mj_model = mujoco.MjModel.from_xml_path(self.scene_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.robot = UR5e()
        self.robot.set_base(mj.get_body_pose(self.mj_model, self.mj_data, "ur5e_base").t)
        self.robot.set_tool(sm.SE3.Trans(0.0, 0.0, 0.13) * sm.SE3.RPY(-np.pi / 2, -np.pi / 2, 0.0))

        q0 = np.array([0.0, -0.6, 1.2, -1.2, -1.57, 0.0], dtype=np.float32)
        self._set_joint_positions(q0)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        mj.attach(self.mj_model, self.mj_data, "attach", "2f85", self.robot.fkine(q0))
        self.mj_data.ctrl[:6] = q0
        self.mj_data.ctrl[6] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.mj_renderer = mujoco.Renderer(self.mj_model, height=self.visualization_height, width=self.visualization_width)
        self.mj_depth_renderer = mujoco.Renderer(self.mj_model, height=self.visualization_height, width=self.visualization_width)
        self.mj_depth_renderer.enable_depth_rendering()

        self.mj_seg_renderer = mujoco.Renderer(self.mj_model, height=self.visualization_height, width=self.visualization_width)
        self.mj_seg_renderer.enable_segmentation_rendering()

        if self.render_viewer:
            self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        self.step_num = 0
        obs = self.get_obs()
        info = {
            "success": False,
            "behavior_name": "no_behavior",
            "task_description": self.task_name,
        }
        return obs, info

    def close(self):
        if self.mj_viewer is not None:
            self.mj_viewer.close()
            self.mj_viewer = None
        if self.mj_renderer is not None:
            self.mj_renderer.close()
            self.mj_renderer = None
        if self.mj_depth_renderer is not None:
            self.mj_depth_renderer.close()
            self.mj_depth_renderer = None
        if self.mj_seg_renderer is not None:
            self.mj_seg_renderer.close()
            self.mj_seg_renderer = None

    def step(self, action: Optional[np.ndarray] = None):
        self._ensure_initialized()

        if action is not None:
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if action.size < 7:
                padded = np.zeros(7, dtype=np.float32)
                padded[:action.size] = action
                action = padded
            elif action.size > 7:
                action = action[:7]

            self.mj_data.ctrl[:6] = action[:6]
            self.mj_data.ctrl[6] = float(np.clip(action[6], 0.0, 255.0))

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.mj_model, self.mj_data)

        if self.mj_viewer is not None:
            self.mj_viewer.sync()

        self.step_num += 1
        obs = self.get_obs()
        success, behavior_name = self._check_success()
        terminated = success
        truncated = self.step_num >= self.max_episode_steps
        reward = 1.0 if success else 0.0

        info = {
            "success": success,
            "behavior_name": behavior_name,
            "task_description": self.task_name,
        }
        return obs, reward, terminated, truncated, info

    def render(
        self,
        camera_name: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        include_segmentation: bool = False,
    ) -> dict[str, np.ndarray]:
        self._ensure_initialized()

        cam = camera_name or self.camera_name
        w = int(width or self.visualization_width)
        h = int(height or self.visualization_height)

        if self.mj_renderer is None or self.mj_renderer.width != w or self.mj_renderer.height != h:
            if self.mj_renderer is not None:
                self.mj_renderer.close()
            self.mj_renderer = mujoco.Renderer(self.mj_model, height=h, width=w)

        if self.mj_depth_renderer is None or self.mj_depth_renderer.width != w or self.mj_depth_renderer.height != h:
            if self.mj_depth_renderer is not None:
                self.mj_depth_renderer.close()
            self.mj_depth_renderer = mujoco.Renderer(self.mj_model, height=h, width=w)
            self.mj_depth_renderer.enable_depth_rendering()

        self.mj_renderer.update_scene(self.mj_data, camera=cam)
        self.mj_depth_renderer.update_scene(self.mj_data, camera=cam)

        out = {
            "img": self.mj_renderer.render(),
            "depth": self.mj_depth_renderer.render().astype(np.float32),
        }

        if include_segmentation:
            if self.mj_seg_renderer is None or self.mj_seg_renderer.width != w or self.mj_seg_renderer.height != h:
                if self.mj_seg_renderer is not None:
                    self.mj_seg_renderer.close()
                self.mj_seg_renderer = mujoco.Renderer(self.mj_model, height=h, width=w)
                self.mj_seg_renderer.enable_segmentation_rendering()

            self.mj_seg_renderer.update_scene(self.mj_data, camera=cam)
            out["segmentation"] = self.mj_seg_renderer.render().copy()

        return out

    def get_obs(self) -> dict[str, Any]:
        self._ensure_initialized()

        rgb = self.render(width=self.width, height=self.height)["img"]
        ee_pos, ee_quat = self._get_body_pose("flange")
        ee_euler = R.from_quat(np.array([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])).as_euler("xyz")
        joints = self._get_joint_positions()
        grip = float(np.clip(self.mj_data.ctrl[6] / 255.0, 0.0, 1.0))

        robot_obs = np.concatenate([ee_pos, ee_euler, [grip], joints, [grip]]).astype(np.float32)

        return {
            "robot_obs": robot_obs,
            "rgb_obs": {
                "rgb_static": rgb,
            },
            "state_vector": self._build_state_vector(),
            "task": self.task_name,
        }


if __name__ == "__main__":
    env = UR5GraspEnv({"render": True})
    env.reset()
    for _ in range(2000):
        env.step()
    print(env.render(include_segmentation=True).keys())
    env.close()
