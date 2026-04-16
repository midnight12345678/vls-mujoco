"""
Manipulator MuJoCo Environment Adapter.

Adapter for manipulator_grasp MuJoCo environments.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import mujoco
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv

from .base_adapter import BaseEnvAdapter, CameraParams, InteractableObject, Pose3D, TrackedObject


class ManipulatorMujocoAdapter(BaseEnvAdapter):
    ACTION_SCALE_POS = 0.01

    def __init__(self, env: UR5GraspEnv, env_config: dict, device: str = "cuda"):
        super().__init__(env, env_config, device)
        self.vlm_camera = env_config.get("vlm_camera", env_config.get("camera_name", "cam"))
        self.target_object = env_config.get("target_object", "Cylinder")
        self.target_zone = env_config.get("target_zone", "zone_drop")
        self.task_name = env_config.get("task_name", "pick_and_place_cylinder")
        self._behavior_dict = {"no_behavior": 0, f"{self.target_object.lower()}_to_{self.target_zone.lower()}": 0}

    def _torch_device(self) -> torch.device:
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            return torch.device(self.device)
        return torch.device("cpu")

    def _get_body_pose(self, name: str) -> Pose3D:
        pos = self.env.mj_data.body(name).xpos.copy()
        quat = self.env.mj_data.body(name).xquat.copy()  # wxyz
        return Pose3D(position=pos, quaternion=quat)

    # ==================== Core Robot State ====================
    def get_ee_pose(self) -> Pose3D:
        return self._get_body_pose("flange")

    def get_ee_pose_world(self) -> Pose3D:
        return self.get_ee_pose()

    def get_robot_base_pose(self) -> Pose3D:
        return self._get_body_pose("ur5e_base")

    def get_joint_positions(self) -> np.ndarray:
        return np.array([float(self.env.mj_data.joint(jn).qpos) for jn in self.env.joint_names], dtype=np.float32)

    def get_gripper_state(self) -> float:
        return float(np.clip(self.env.mj_data.ctrl[6] / 255.0, 0.0, 1.0))

    # ==================== Action Processing ====================
    def delta_actions_to_ee_trajectory(self, action_sequence: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(action_sequence, np.ndarray):
            action_sequence = torch.from_numpy(action_sequence).float()

        device = action_sequence.device
        dtype = action_sequence.dtype
        start_pos = torch.tensor(self.get_ee_pose_world().position, device=device, dtype=dtype)

        delta_positions = action_sequence[:, :3] * self.ACTION_SCALE_POS
        cumsum_deltas = torch.cumsum(delta_positions, dim=0)
        return torch.cat([start_pos.unsqueeze(0), start_pos + cumsum_deltas], dim=0)

    def get_action_space_info(self) -> Dict[str, Any]:
        return {
            "dim": 7,
            "type": "joint_target_plus_gripper",
            "bounds": (
                np.array([-2 * np.pi] * 6 + [0.0], dtype=np.float32),
                np.array([2 * np.pi] * 6 + [255.0], dtype=np.float32),
            ),
            "normalized": False,
        }

    # ==================== Camera & Perception ====================
    def get_vlm_image(self) -> np.ndarray:
        out = self.env.render(camera_name=self.vlm_camera, width=self._env_config.get("visualization_width", 640), height=self._env_config.get("visualization_height", 640))
        return out["img"]

    def get_camera_params(self, camera_name: str) -> CameraParams:
        model = self.env.mj_model
        data = self.env.mj_data

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise ValueError(f"Camera '{camera_name}' not found")

        width = self._env_config.get("visualization_width", 640)
        height = self._env_config.get("visualization_height", 640)

        fovy_deg = float(model.cam_fovy[cam_id])
        fovy_rad = np.radians(fovy_deg)
        focal = (height / 2.0) / np.tan(fovy_rad / 2.0)

        intrinsic = np.array(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        cam_pos = data.cam(camera_name).xpos.copy()
        cam_rot = data.cam(camera_name).xmat.reshape(3, 3).copy()
        cam_rot[:, 2] = -cam_rot[:, 2]

        cam2world = np.eye(4, dtype=np.float32)
        cam2world[:3, :3] = cam_rot
        cam2world[:3, 3] = cam_pos
        world2cam = np.linalg.inv(cam2world)

        return CameraParams(intrinsic=intrinsic, extrinsic=world2cam, width=width, height=height)

    def get_camera_names(self) -> List[str]:
        return [self.vlm_camera]

    # ==================== Scene Objects ====================
    def get_interactable_objects(self) -> List[InteractableObject]:
        names = self._env_config.get("interactable_objects", ["Cylinder", "Box", "T_block"])
        objects: List[InteractableObject] = []
        seg_idx = 1
        for name in names:
            body_id = mujoco.mj_name2id(self.env.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                objects.append(
                    InteractableObject(name=name, object_id=body_id, link_index=-1, segment_index=seg_idx)
                )
                seg_idx += 1
        return objects

    def get_scene_objects(self, exclude_names: Optional[List[str]] = None) -> List[TrackedObject]:
        exclude = set(exclude_names or [])
        out: List[TrackedObject] = []
        for obj in self.get_interactable_objects():
            if obj.name in exclude:
                continue
            out.append(TrackedObject(name=obj.name, pose=self._get_body_pose(obj.name), obj_ref=obj))
        return out

    def get_object_pose(self, object_name: str) -> Optional[Pose3D]:
        body_id = mujoco.mj_name2id(self.env.mj_model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        if body_id < 0:
            return None
        return self._get_body_pose(object_name)

    def get_object_pose_by_segment(self, segment_index: int) -> Optional[Pose3D]:
        for obj in self.get_interactable_objects():
            if obj.segment_index == segment_index:
                return self.get_object_pose(obj.name)
        return None

    # ==================== Segmentation Processing ====================
    def process_segmentation(
        self,
        seg_image: np.ndarray,
        camera_name: str = "cam",
    ) -> Tuple[np.ndarray, List[InteractableObject], Dict[int, str]]:
        if seg_image.ndim == 3 and seg_image.shape[2] > 1:
            seg_ids = seg_image[:, :, 1]
        elif seg_image.ndim == 3:
            seg_ids = seg_image[:, :, 0]
        else:
            seg_ids = seg_image

        interactables = self.get_interactable_objects()
        body_to_obj = {o.object_id: o for o in interactables}

        processed = np.zeros_like(seg_ids, dtype=np.int32)
        id_to_name = {0: "background"}
        found: List[InteractableObject] = []

        unique_ids = np.unique(seg_ids)
        for geom_id in unique_ids:
            if geom_id <= 0:
                continue
            if geom_id >= self.env.mj_model.ngeom:
                continue
            body_id = int(self.env.mj_model.geom_bodyid[int(geom_id)])
            obj = body_to_obj.get(body_id)
            if obj is None:
                continue
            mask = seg_ids == geom_id
            processed[mask] = obj.segment_index
            if obj not in found:
                found.append(obj)
                id_to_name[obj.segment_index] = obj.name

        return processed, found, id_to_name

    def _depth_to_pointcloud(self, depth: np.ndarray, camera_params: CameraParams) -> np.ndarray:
        H, W = depth.shape
        fx, fy = camera_params.intrinsic[0, 0], camera_params.intrinsic[1, 1]
        cx, cy = camera_params.intrinsic[0, 2], camera_params.intrinsic[1, 2]

        u, v = np.meshgrid(np.arange(W), np.arange(H))
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points_cam = np.stack([x, y, z], axis=-1)
        valid = (z > 0) & np.isfinite(z)

        cam2world = np.linalg.inv(camera_params.extrinsic)
        points_h = np.concatenate([points_cam.reshape(-1, 3), np.ones((H * W, 1))], axis=-1)
        points_world = (points_h @ cam2world.T)[:, :3].reshape(H, W, 3)

        out = np.zeros((H, W, 3), dtype=np.float32)
        out[valid] = points_world[valid]
        return out

    def get_keypoint_detection_inputs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
        rendered = self.env.render(
            camera_name=self.vlm_camera,
            width=self._env_config.get("visualization_width", 640),
            height=self._env_config.get("visualization_height", 640),
            include_segmentation=True,
        )
        rgb = rendered["img"]
        depth = rendered["depth"].astype(np.float32)
        seg_raw = rendered["segmentation"]

        segmentation, _, segment_id_to_name = self.process_segmentation(seg_raw)
        camera_params = self.get_camera_params(self.vlm_camera)
        points = self._depth_to_pointcloud(depth, camera_params)
        return rgb, depth, points, segmentation, segment_id_to_name

    def get_policy_observation(self, sample_num: int = 1) -> Dict[str, Any]:
        obs = self.get_obs()
        state = obs.get("state_vector", np.zeros(32, dtype=np.float32))
        rgb = obs.get("rgb_obs", {}).get("rgb_static")

        if rgb is None:
            rgb = np.zeros((224, 224, 3), dtype=np.uint8)

        rgb = Image.fromarray(rgb.astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        rgb = np.asarray(rgb).astype(np.float32) / 255.0

        device = self._torch_device()
        state_t = torch.from_numpy(state).float().to(device).unsqueeze(0)
        img_t = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device).unsqueeze(0)

        out = {
            "observation.state": state_t,
            "observation.images.base_0_rgb": img_t,
            "observation.images.left_wrist_0_rgb": img_t,
            "observation.images.right_wrist_0_rgb": img_t,
            "observation.images.image": img_t,
            "observation.images.image2": img_t,
            "task": [self.get_instruction()],
        }
        out = {
            k: (v.expand(sample_num, *v.shape[1:]) if isinstance(v, torch.Tensor) else [v[0]] * sample_num)
            for k, v in out.items()
        }
        return out

    # ==================== Behavior Analysis ====================
    def check_success(self) -> Tuple[bool, str]:
        obj_pose = self.get_object_pose(self.target_object)
        zone_pose = self.get_object_pose(self.target_zone)
        if obj_pose is None or zone_pose is None:
            return False, "no_behavior"

        dist = np.linalg.norm(obj_pose.position - zone_pose.position)
        success_name = f"{self.target_object.lower()}_to_{self.target_zone.lower()}"
        if dist <= float(self._env_config.get("success_distance", 0.08)):
            return True, success_name
        return False, "no_behavior"

    def get_behavior_static(self) -> Dict[str, int]:
        return self._behavior_dict

    # ==================== Environment Interface ====================
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.size < 7:
            padded = np.zeros(7, dtype=np.float32)
            padded[: action.size] = action
            action = padded
        elif action.size > 7:
            action = action[:7]

        obs, reward, terminated, truncated, info = self.env.step(action)
        success, behavior_name = self.check_success()
        info["success"] = success
        info["behavior_name"] = behavior_name
        if success:
            self._behavior_dict[behavior_name] = self._behavior_dict.get(behavior_name, 0) + 1
            terminated = True
        elif truncated:
            self._behavior_dict["no_behavior"] += 1
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        return self.env.reset(**kwargs)

    def get_obs(self) -> Dict:
        return self.env.get_obs()

    # ==================== Task-Specific Guidance ====================
    def get_task_info(self) -> Dict[str, Any]:
        return {
            "instruction": self.get_instruction(),
            "recommended_guide_scale": self._env_config.get("recommended_guide_scale", 80.0),
            "task_type": "manipulation",
            "requires_precision": True,
            "task_name": self.task_name,
        }

    def get_instruction(self) -> str:
        return self._env_config.get("instruction", self.task_name)

    def get_task_description(self) -> str:
        return self.get_instruction()
