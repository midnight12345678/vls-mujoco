"""
VLS Main Entry Point - Supports multiple backends (CALVIN, LIBERO, RealWorld)

Uses Hydra for configuration management and EnvAdapter abstraction layer.

Usage:
    python main.py                                    # Default: calvin + drawer_open
    python main.py env=calvin task=drawer_open        # Explicit (auto-loads task/calvin/drawer_open.yaml)
    python main.py env=libero task=goal               # Switch to LIBERO goal task
    python main.py main.episode_num=50                # Override parameters
    python main.py +experiment=debug                  # Use experiment config
"""

import os
import shutil
# Set tokenizers parallelism to false to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Default to EGL for headless Linux rendering to avoid corrupted offscreen frames.
if os.name != "nt" and "MUJOCO_GL" not in os.environ:
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ["MUJOCO_GL"] = "egl"

from typing import Any, Callable, List, Optional
import gymnasium as gym
import numpy as np
import torch
from collections import deque
from PIL import Image
import cv2
import imageio
from functools import partial
import matplotlib.pyplot as plt
import h5py
import random
from dataclasses import dataclass, field
import tqdm
import time
import sys
import json
import re
import warnings

# Hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf

warnings.filterwarnings('ignore', category=FutureWarning, message='.*pynvml.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*pkg_resources.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*xFormers.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*env.get_obs.*')

# Add project root to path (for local modules like steer_utils, utils, etc.)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Monkey-patch calvin_env Light class with our modified version (supports initial_state config)
from patches.light import Light as PatchedLight, LightState as PatchedLightState
sys.modules['calvin_env.scene.objects.light'] = type(sys)('calvin_env.scene.objects.light')
sys.modules['calvin_env.scene.objects.light'].Light = PatchedLight
sys.modules['calvin_env.scene.objects.light'].LightState = PatchedLightState

# Import adapters
from core.env_adapters import create_adapter, BaseEnvAdapter
from core.keypoint_tracker import KeypointTracker
from core.keypoint_detector import KeypointDetector
from core.sam3_segmenter import create_segmenter as create_sam3_segmenter
from core.gemini_grounder import create_gemini_grounder, create_gemini_stage_recognizer
from vlm_query.vlm_agent import VLMAgent
from utils.vis_utils import TrajectoryVideoRecorder, add_text_to_image

from core.diffusion_policy_steer import DiffusionPolicySteer
from core.pi05_steer import PI05PolicySteer
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.factory import make_env_pre_post_processors

# Import logging utility
from utils.logging_utils import SteerLogger

# Create logger instance
log = SteerLogger("Main")


class Main:
    def __init__(self, cfg: DictConfig):
        """
        Initialize with Hydra DictConfig.
        
        Args:
            cfg: Hydra configuration (OmegaConf DictConfig)
        """
        self.cfg = cfg
        self.config = cfg.main  # Shortcut to main config section
        
        # Log the resolved configuration
        log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)[:500]}...")

        # Set random seed
        seed = cfg.get('seed', 0)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        # Get backend from config
        self.backend = cfg.backend.get('backend', 'calvin')
        log.info(f"Using backend: {self.backend}")

        self.sample_batch_size = int(self.config.get('sample_batch_size', 1))
        if self.backend == 'mujoco' and self.config.get('use_guidance', False) and self.sample_batch_size <= 1:
            self.sample_batch_size = 8
            log.info("MuJoCo guidance is enabled with sample_batch_size=1; using effective sample_batch_size=8 to activate diversity/FKD steering.")
        
        # Get backend-specific config
        env_config = OmegaConf.to_container(cfg.backend.get(self.backend, {}), resolve=True)
        
        # Add main.episode_num to env_config for adapters that need it (e.g., LiberoAdapter)
        env_config['episode_num'] = self.config.get('episode_num', 10)

        # Create adapter
        self.adapter = create_adapter(self.backend, env_config)
        
        # Get task info from adapter for guidance adjustment
        task_info = self.adapter.get_task_info()
        instruction = task_info.get('instruction', '')
        recommended_scale = task_info.get('recommended_guide_scale')
        base_guide_scale = self.config.get('guide_scale', 80.0)
        
        # Override with recommended scale if available
        if recommended_scale is not None:
            self.current_guide_scale = recommended_scale
            if recommended_scale != base_guide_scale:
                log.info(f"Task '{instruction}' uses custom guide_scale={recommended_scale} (base: {base_guide_scale})")
            else:
                log.info(f"Using guide_scale: {recommended_scale} for task: '{instruction}'")
        else:
            self.current_guide_scale = base_guide_scale
            log.info(f"Using default guide_scale: {base_guide_scale}")
        
        # Initialize policy
        policy_config = cfg.get('policy', {})
        policy_type = policy_config.get('type', 'diffusion')
        log.info(f"Policy config type: {policy_type}")
        self.device = str(cfg.get('device', 'cuda'))
        
        # Get pretrained_path from the specific policy type config
        type_config = policy_config.get(policy_type, {})
        pretrained_path = type_config.get('pretrained_path', 'Vision-Language-Steering/vls_calvin_base')
        log.info(f"Loading {policy_type} from: {pretrained_path}")
        
        if policy_type == 'diffusion':
            self.policy = DiffusionPolicySteer.from_pretrained(pretrained_path)
        elif policy_type == 'pi05':
            pi05_config = PreTrainedConfig.from_pretrained(pretrained_path)
            pi05_config.device = self.device
            self.policy = PI05PolicySteer.from_pretrained(
                pretrained_path,
                config=pi05_config,
                strict=False,
            )
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
        
        self.policy.to(self.device)

        preprocessor_overrides = {
            "device_processor": {"device": self.device},
        }

        self.policy_preprocessor, self.policy_postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config, 
            pretrained_path=pretrained_path, 
            preprocessor_overrides=preprocessor_overrides,
        )

        self.policy.post_init(
            adapter=self.adapter,
            postprocessor=self.policy_postprocessor,
            sample_batch_size=self.sample_batch_size,
            policy_config=policy_config[policy_type],
        )

        self.policy.eval()
        log.info(f"Loaded {policy_type} policy from {pretrained_path}")
        
        # Output directory (use Hydra's output directory directly)
        self.output_dir = self.config.get('output_dir', 'results/')
        # Ensure it ends with /
        if not self.output_dir.endswith('/'):
            self.output_dir += '/'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self._init_components(cfg)
        
        # Reset environment and policy
        # self.adapter.reset()
        self.policy.reset()
    
    def _init_components(self, cfg: DictConfig):
        """Initialize components."""
        # Get perception config (loaded from perception.yaml with @package perception)
        perception_cfg = cfg.get('perception', {})
        guidance_enabled = bool(self.config.get("use_guidance", True))

        if guidance_enabled:
            # Initialize keypoint detector
            kp_config = OmegaConf.to_container(perception_cfg.get('keypoint_detector', {}), resolve=True)
            self.keypoint_detector = KeypointDetector(config=kp_config)

            # Initialize SAM3 segmenter if enabled
            sam3_config = perception_cfg.get('sam3', {})
            self.use_sam3 = sam3_config.get('enabled', False) if sam3_config else False
            if self.use_sam3:
                log.info("Initializing SAM3 segmenter for text-prompted segmentation")
                sam3_dict = OmegaConf.to_container(sam3_config, resolve=True)
                self.sam3_segmenter = create_sam3_segmenter(sam3_dict)
                self.sam3_default_objects = sam3_config.get('default_objects', [])
            else:
                self.sam3_segmenter = None

            # Initialize Gemini grounder if enabled
            gemini_config = perception_cfg.get('gemini_grounding', {})
            self.use_gemini = gemini_config.get('enabled', False) if gemini_config else False
            if self.use_gemini:
                api_key = gemini_config.get('api_key') or os.environ.get('GOOGLE_API_KEY')
                if api_key:
                    gemini_dict = OmegaConf.to_container(gemini_config, resolve=True)
                    gemini_dict['api_key'] = api_key
                    log.info("Initializing Gemini grounder for visual grounding")
                    self.gemini_grounder = create_gemini_grounder(gemini_dict)
                    self.gemini_default_objects = list(gemini_config.get('default_objects', []))
                else:
                    log.warning("Gemini API key not found - disabling Gemini grounding")
                    self.use_gemini = False
                    self.gemini_grounder = None
            else:
                self.gemini_grounder = None
        else:
            self.keypoint_detector = None
            self.use_sam3 = False
            self.sam3_segmenter = None
            self.sam3_default_objects = []
            self.use_gemini = False
            self.gemini_grounder = None
            self.gemini_default_objects = []
            log.info("Guidance disabled: skipping perception/VLM guidance components.")

        # Initialize keypoint tracker
        self.keypoint_tracker = KeypointTracker(self.adapter)

        if guidance_enabled:
            # Initialize VLM agent (OpenAI - for generating guidance functions)
            vlm_config = OmegaConf.to_container(perception_cfg.get('vlm_agent', {}), resolve=True)
            # Ensure query_template_dir is absolute path
            if vlm_config.get('query_template_dir') and not os.path.isabs(vlm_config['query_template_dir']):
                vlm_config['query_template_dir'] = os.path.join(PROJECT_ROOT, vlm_config['query_template_dir'])
            self.vlm_agent = VLMAgent(
                config=vlm_config,
                base_dir=self.output_dir,
                env_type=self.backend,
            )

            # Initialize Gemini stage recognizer (for real-time stage recognition)
            gemini_config = OmegaConf.to_container(perception_cfg.get('gemini', {}), resolve=True)
            if self.config.get("use_vlm_stage_recognition", True):
                try:
                    self.gemini_stage_recognizer = create_gemini_stage_recognizer(gemini_config)
                    log.info("Gemini stage recognizer initialized")
                except Exception as e:
                    log.warning(f"Failed to initialize Gemini stage recognizer: {e}")
                    self.gemini_stage_recognizer = None
            else:
                self.gemini_stage_recognizer = None
        else:
            self.vlm_agent = None
            self.gemini_stage_recognizer = None

        # Initialize video recorder
        self.video_recorder = TrajectoryVideoRecorder(output_dir=self.output_dir)

        self.cached_functions_dir = self.config.get('cached_functions_dir', None)

    def _get_policy_observation(self) -> dict:
        """Get observation in policy expected format (backend-agnostic)."""
        sample_num = self.sample_batch_size
        observation = self.adapter.get_policy_observation(sample_num=sample_num)
        
        processed_observation = self.policy_preprocessor(observation)
        return processed_observation

    def _build_mujoco_fallback_keypoints(self) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
        """Build minimal keypoints for MuJoCo steering when perception returns no keypoints."""
        target_object = getattr(self.adapter, "target_object", "Cylinder")
        target_zone = getattr(self.adapter, "target_zone", "zone_drop")

        interactables = self.adapter.get_interactable_objects()
        obj_seg = 1
        for obj in interactables:
            if obj.name == target_object:
                obj_seg = obj.segment_index
                break

        keypoints: list[np.ndarray] = []
        mask_ids: list[int] = []
        segment_id_to_name: dict[int, str] = {obj_seg: target_object, 0: target_zone}

        obj_pose = self.adapter.get_object_pose(target_object)
        if obj_pose is not None:
            keypoints.append(np.asarray(obj_pose.position, dtype=np.float32))
            mask_ids.append(obj_seg)

        zone_pose = self.adapter.get_object_pose(target_zone)
        if zone_pose is not None:
            keypoints.append(np.asarray(zone_pose.position, dtype=np.float32))
            # Use segment 0 to keep it static in keypoint tracker.
            mask_ids.append(0)

        ee_pose = self.adapter.get_ee_pose_world()
        keypoints.append(np.asarray(ee_pose.position, dtype=np.float32))
        mask_ids.append(0)

        return (
            np.asarray(keypoints, dtype=np.float32),
            np.asarray(mask_ids, dtype=np.int32),
            segment_id_to_name,
        )

    def _augment_mujoco_keypoints_with_zone(
        self,
        key_points: np.ndarray,
        mask_ids: np.ndarray,
        segment_id_to_name: dict[int, str],
    ) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
        """Ensure MuJoCo keypoints include an explicit target-zone keypoint for placement stage guidance."""
        if self.backend != "mujoco":
            return key_points, mask_ids, segment_id_to_name

        target_zone = getattr(self.adapter, "target_zone", "zone_drop")
        zone_pose = self.adapter.get_object_pose(target_zone)
        if zone_pose is None:
            return key_points, mask_ids, segment_id_to_name

        key_points = np.asarray(key_points, dtype=np.float32).reshape(-1, 3)
        mask_ids = np.asarray(mask_ids, dtype=np.int32).reshape(-1)
        segment_id_to_name = dict(segment_id_to_name or {})

        # Reserve segment 0 for static reference points in tracker; map it to target zone for MuJoCo.
        segment_id_to_name[0] = target_zone

        # Avoid duplicate static-zone keypoints.
        if np.any(mask_ids == 0):
            return key_points, mask_ids, segment_id_to_name

        zone_point = np.asarray(zone_pose.position, dtype=np.float32).reshape(1, 3)
        key_points = np.concatenate([key_points, zone_point], axis=0)
        mask_ids = np.concatenate([mask_ids, np.array([0], dtype=np.int32)], axis=0)
        log.debug("Added explicit MuJoCo zone keypoint for target placement guidance.")
        return key_points, mask_ids, segment_id_to_name

    def _build_keypoint_object_map(
        self,
        mask_ids: np.ndarray,
        segment_id_to_name: dict[int, str],
    ) -> dict[int, str]:
        mask_ids = np.asarray(mask_ids, dtype=np.int32).reshape(-1)
        return {
            idx: segment_id_to_name.get(int(mask_id), f"unknown_{int(mask_id)}")
            for idx, mask_id in enumerate(mask_ids)
        }

    def _estimate_mujoco_object_anchor(self, object_name: str, object_center: np.ndarray) -> np.ndarray:
        name = str(object_name).lower()
        if name == "box":
            # Box half-height in scene.xml is 0.025m; use top-center as grasp anchor.
            return object_center + np.array([0.0, 0.0, 0.025], dtype=np.float32)
        if name == "cylinder":
            # Cylinder half-height in scene.xml is 0.1m; pull toward top-center.
            return object_center + np.array([0.0, 0.0, 0.1], dtype=np.float32)
        return object_center

    def _log_mujoco_keypoint_mapping_errors(
        self,
        key_points: np.ndarray,
        key_points_objects_map: dict[int, str],
        tag: str,
    ):
        if self.backend != "mujoco":
            return

        key_points = np.asarray(key_points, dtype=np.float32).reshape(-1, 3)
        log.info(f"[MuJoCoKeypointCheck/{tag}] mapping residuals (keypoint -> object center)")
        for idx in sorted(key_points_objects_map.keys()):
            if idx < 0 or idx >= len(key_points):
                continue
            object_name = key_points_objects_map[idx]
            pose = self.adapter.get_object_pose(object_name)
            if pose is None:
                continue
            delta = key_points[idx] - np.asarray(pose.position, dtype=np.float32)
            err = float(np.linalg.norm(delta))
            log.info(
                f"  kp[{idx}] {object_name}: err={err:.4f}m "
                f"(dx={delta[0]:+.4f}, dy={delta[1]:+.4f}, dz={delta[2]:+.4f})"
            )

    def _refine_mujoco_target_keypoint(
        self,
        key_points: np.ndarray,
        key_points_objects_map: dict[int, str],
    ) -> np.ndarray:
        if self.backend != "mujoco":
            return key_points

        target_object = getattr(self.adapter, "target_object", None)
        if target_object is None:
            return key_points

        target_idx = next(
            (idx for idx, name in key_points_objects_map.items() if str(name) == str(target_object)),
            None,
        )
        if target_idx is None:
            return key_points

        key_points = np.asarray(key_points, dtype=np.float32).reshape(-1, 3)
        if target_idx < 0 or target_idx >= len(key_points):
            return key_points

        target_pose = self.adapter.get_object_pose(target_object)
        if target_pose is None:
            return key_points

        object_center = np.asarray(target_pose.position, dtype=np.float32)
        target_anchor = self._estimate_mujoco_object_anchor(target_object, object_center)
        raw_err = float(np.linalg.norm(key_points[target_idx] - target_anchor))

        snap_threshold = float(self.cfg.backend.get("mujoco", {}).get("target_keypoint_snap_threshold", 0.03))
        if raw_err > snap_threshold:
            log.warning(
                f"[MuJoCoKeypointCheck] Target keypoint drift too large for {target_object} "
                f"(err={raw_err:.4f}m > {snap_threshold:.4f}m). "
                "Snapping to physics anchor."
            )
            key_points[target_idx] = target_anchor
        return key_points

    def _refine_live_mujoco_guidance_keypoints(
        self,
        key_points: np.ndarray,
        key_points_objects_map: dict[int, str],
    ) -> np.ndarray:
        """
        Keep MuJoCo guidance keypoints locked to physics anchors during rollout.

        The detector-based keypoint registration is useful for VLM indexing, but guidance
        should always use stable physics anchors for target/zone to avoid drift-induced
        steering jitter.
        """
        if self.backend != "mujoco":
            return key_points

        key_points = np.asarray(key_points, dtype=np.float32).reshape(-1, 3).copy()
        if key_points.shape[0] == 0:
            return key_points

        target_object = str(getattr(self.adapter, "target_object", ""))
        target_zone = str(getattr(self.adapter, "target_zone", ""))

        for idx, name in key_points_objects_map.items():
            if idx < 0 or idx >= key_points.shape[0]:
                continue

            obj_name = str(name)
            if obj_name == target_zone:
                zone_pose = self.adapter.get_object_pose(target_zone)
                if zone_pose is not None:
                    key_points[idx] = np.asarray(zone_pose.position, dtype=np.float32)
                continue

            if obj_name == target_object:
                target_pose = self.adapter.get_object_pose(target_object)
                if target_pose is None:
                    continue
                object_center = np.asarray(target_pose.position, dtype=np.float32)
                key_points[idx] = self._estimate_mujoco_object_anchor(target_object, object_center)

        return key_points

    def _project_world_keypoints_to_pixels(
        self,
        key_points: np.ndarray,
        camera_name: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        key_points = np.asarray(key_points, dtype=np.float32).reshape(-1, 3)
        cam_name = camera_name or getattr(self.adapter, "vlm_camera", None)
        if cam_name is None:
            raise ValueError("No camera name available for keypoint projection")

        cam = self.adapter.get_camera_params(cam_name)
        intrinsic = np.asarray(cam.intrinsic, dtype=np.float32)
        world2cam = np.asarray(cam.extrinsic, dtype=np.float32)

        points_h = np.concatenate(
            [key_points, np.ones((key_points.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        points_cam = (points_h @ world2cam.T)[:, :3]

        z = points_cam[:, 2]
        valid_depth = np.isfinite(z) & (z > 1e-6)

        u = intrinsic[0, 0] * (points_cam[:, 0] / np.maximum(z, 1e-6)) + intrinsic[0, 2]
        v = intrinsic[1, 1] * (points_cam[:, 1] / np.maximum(z, 1e-6)) + intrinsic[1, 2]

        pixels = np.stack([v, u], axis=1)
        in_bounds = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < float(cam.height))
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < float(cam.width))
        )
        valid = valid_depth & in_bounds
        return pixels, valid

    def _refresh_projected_keypoints_image(
        self,
        rgb: np.ndarray,
        key_points: np.ndarray,
        mask_ids: np.ndarray,
        fallback_projected: np.ndarray,
        key_points_objects_map: Optional[dict[int, str]] = None,
    ) -> np.ndarray:
        if self.backend != "mujoco":
            return fallback_projected

        try:
            pixels, valid = self._project_world_keypoints_to_pixels(key_points)
        except Exception as e:
            log.warning(f"Failed to project refined MuJoCo keypoints ({e}); keeping detector overlay.")
            return fallback_projected

        projected = np.ascontiguousarray(rgb.copy(), dtype=np.uint8)

        def _draw_index_box(img: np.ndarray, row: int, col: int, index: int):
            text = str(index)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            box_w = text_w + 8
            box_h = text_h + baseline + 6
            x1 = max(0, col - box_w // 2)
            y1 = max(0, row - box_h // 2)
            x2 = min(img.shape[1] - 1, x1 + box_w)
            y2 = min(img.shape[0] - 1, y1 + box_h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 120, 255), 2)
            text_x = x1 + (box_w - text_w) // 2
            text_y = y1 + (box_h + text_h) // 2 - baseline
            cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)

        visible_count = 0
        for idx, px in enumerate(np.asarray(pixels, dtype=np.float32)):
            if idx >= len(valid) or not bool(valid[idx]):
                continue
            row = int(round(float(px[0])))
            col = int(round(float(px[1])))
            _draw_index_box(projected, row, col, idx)
            visible_count += 1

        invalid_count = int((~valid).sum())
        if key_points_objects_map:
            target_object = str(getattr(self.adapter, "target_object", ""))
            target_idx = next(
                (idx for idx, name in key_points_objects_map.items() if str(name) == target_object),
                None,
            )
            if target_idx is not None and 0 <= target_idx < len(valid) and not bool(valid[target_idx]):
                log.warning(
                    "Refined target keypoint is outside camera view; "
                    "keeping detector overlay to preserve keypoint-object correspondence."
                )
                return fallback_projected

        if invalid_count > 0:
            log.warning(
                f"{invalid_count} refined keypoints are outside camera view; drawing visible refined keypoints only."
            )
        if visible_count == 0:
            log.warning("No refined keypoints are visible in camera; keeping detector overlay.")
            return fallback_projected

        return projected

    def _mujoco_guidance_has_hardcoded_world_target(self, txt_path: str) -> bool:
        """Detect common failure mode where VLM emits fixed world-coordinate targets instead of keypoint-based goals."""
        if self.backend != "mujoco" or not os.path.exists(txt_path):
            return False

        with open(txt_path, 'r') as f:
            code = f.read()

        # Match literal 3D coordinate tensors like torch.tensor([1.6, 0.5, 0.0]).
        coord_re = re.compile(
            r"torch\.tensor\(\s*\[\s*-?\d+(?:\.\d+)?(?:e[+-]?\d+)?\s*,\s*"
            r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?\s*,\s*"
            r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?\s*\]"
        )

        for line in code.splitlines():
            stripped = line.strip()
            if "dtype=torch.long" in stripped:
                continue
            if coord_re.search(stripped):
                return True

        return False

    def _is_grasp_only_instruction(self, instruction: str) -> bool:
        text = str(instruction).lower()
        has_grasp = ("grasp" in text) or ("pick up" in text)
        has_place = any(tok in text for tok in ("place", "drop", "put into", "put in", "into", "onto"))
        return has_grasp and not has_place

    def _mujoco_requires_placement(self) -> bool:
        """
        Determine whether the current MuJoCo task requires a placement stage.

        For MuJoCo, relying only on instruction text is brittle (e.g. "grasp ...")
        while success is still defined as object-to-zone placement.
        """
        if self.backend != "mujoco":
            return False

        task_info = {}
        try:
            task_info = self.adapter.get_task_info() if hasattr(self.adapter, "get_task_info") else {}
        except Exception:
            task_info = {}

        # Explicit config from adapter/task definition takes precedence.
        if "requires_placement" in task_info:
            return bool(task_info["requires_placement"])

        # Fallback: if a target zone is configured, treat as placement task.
        target_zone = str(getattr(self.adapter, "target_zone", "") or "").strip()
        return target_zone != ""

    def _setup_builtin_mujoco_guidance(self, key_points_objects_map: dict[int, str]):
        """Fallback stage guidance for MuJoCo when VLM generation is unavailable."""
        target_object = getattr(self.adapter, "target_object", "Cylinder")
        target_zone = getattr(self.adapter, "target_zone", "zone_drop")
        grasp_only = self._is_grasp_only_instruction(self.adapter.get_instruction()) and not self._mujoco_requires_placement()

        obj_idx = next((idx for idx, name in key_points_objects_map.items() if name == target_object), 0)
        zone_idx = next((idx for idx, name in key_points_objects_map.items() if name == target_zone), obj_idx)

        def _weights(action_sequence: torch.Tensor, final_boost: float = 3.0):
            T = action_sequence.shape[1]
            ts = torch.arange(T, device=action_sequence.device)
            final_start = int(0.7 * T)
            w = torch.where(
                ts >= final_start,
                torch.tensor(final_boost, dtype=action_sequence.dtype, device=action_sequence.device),
                torch.tensor(1.0, dtype=action_sequence.dtype, device=action_sequence.device),
            )
            return w / w.sum()

        def stage1_guidance(keypoints: torch.Tensor, action_sequence: torch.Tensor):
            obj_pos = keypoints[obj_idx]
            w = _weights(action_sequence, final_boost=3.0)
            d = torch.norm(action_sequence - obj_pos.view(1, 1, 3), dim=-1)
            reward = -((d ** 2) * w).sum(dim=1)
            return reward.mean()

        def stage2_guidance(keypoints: torch.Tensor, action_sequence: torch.Tensor):
            zone_pos = keypoints[zone_idx]
            obj_pos = keypoints[obj_idx]
            w = _weights(action_sequence, final_boost=4.0)
            d_zone = torch.norm(action_sequence - zone_pos.view(1, 1, 3), dim=-1)
            d_obj = torch.norm(action_sequence - obj_pos.view(1, 1, 3), dim=-1)
            reward = -((d_zone ** 2) * w).sum(dim=1) + 0.15 * (d_obj * w).sum(dim=1)
            return reward.mean()

        if grasp_only:
            self.program_info = {
                "num_stages": 1,
                "stage_names": ["approach_and_grasp_object"],
            }
            self.guidance_fns = {
                1: [stage1_guidance],
            }
            self.stage_descriptions = "Stage 1: approach and grasp target object"
        else:
            self.program_info = {
                "num_stages": 2,
                "stage_names": ["approach_object", "move_to_target_zone"],
            }
            self.guidance_fns = {
                1: [stage1_guidance],
                2: [stage2_guidance],
            }
            self.stage_descriptions = (
                "Stage 1: approach and grasp target object\n"
                "Stage 2: move grasped object to target zone"
            )
        log.warning("Using built-in MuJoCo steering guidance (VLM guidance unavailable)")
    
    def perform_task_for_episode(self, episode_dir: str):
        """Prepare for each episode (keypoint detection, guidance generation, etc.)"""
        from utils.guidance_utils import load_functions_from_txt

        # Get keypoint detection inputs from adapter (includes segmentation if available)
        rgb, depth, points, segmentation, segment_id_to_name = self.adapter.get_keypoint_detection_inputs()
        
        # Get instruction from adapter (backend-specific)
        instruction = self.adapter.get_instruction()
        log.info(f"Task instruction: {instruction}")

        # segmentation = None

        # Check if adapter provided valid segmentation
        has_valid_segmentation = (
            segmentation is not None and 
            segment_id_to_name is not None and 
            len(segment_id_to_name) > 0
        )
        
        if has_valid_segmentation:
            log.info(f"Using adapter-provided segmentation with {len(segment_id_to_name)} objects: {list(segment_id_to_name.values())}")
        else:
            log.warning("No valid segmentation from adapter, will use Gemini/SAM3")
            
            # Get interactable objects from adapter for grounding
            interactable_objects = self.adapter.get_interactable_objects()
            object_names = [obj.name for obj in interactable_objects]
            log.info(f"Interactable objects from adapter: {object_names}")
            
            # Use Gemini grounding for segmentation (preferred)
            if self.use_gemini and self.gemini_grounder is not None and len(object_names) > 0:
                log.info("Using Gemini for visual grounding")
                log.info(f"Gemini detecting: {object_names}")
                segmentation, segment_id_to_name = self.gemini_grounder.detect_to_segmentation(
                    rgb, object_names
                )

            # Fallback to SAM3
            elif self.use_sam3 and self.sam3_segmenter is not None:
                log.info("Using SAM3 for text-prompted segmentation")
                
                # Use VLM to plan which objects to segment based on instruction
                planned_objects = self.vlm_agent.plan_segmentation(rgb, instruction)

                if planned_objects and len(planned_objects) > 0:
                    object_names = planned_objects
                    log.info(f"VLM planned segmentation: {object_names}")
                elif len(object_names) == 0:
                    # Last resort: use default objects from config
                    object_names = self.sam3_default_objects
                    log.info(f"Using config default objects: {object_names}")

                segmentation, segment_id_to_name = self.sam3_segmenter.segment(
                    rgb, object_names, return_all_masks=False
                )
            else:
                log.error("No segmentation method available and adapter didn't provide segmentation!")
                raise RuntimeError("Cannot proceed without segmentation")

        # Detect keypoints
        key_points, projected_img, mask_ids = self.keypoint_detector.get_keypoints(
            rgb=rgb,
            points=points,
            segmentation=segmentation,
            segment_id_to_name=segment_id_to_name
        )

        if key_points is None or len(key_points) == 0:
            if self.backend == "mujoco":
                log.warning("No keypoints detected for MuJoCo task. Using fallback object/zone keypoints.")
                key_points, mask_ids, segment_id_to_name = self._build_mujoco_fallback_keypoints()
                projected_img = rgb.copy()
            else:
                raise RuntimeError("No keypoints detected and no fallback available for this backend")

        # For MuJoCo placement tasks, append an explicit zone keypoint to prevent VLM from inventing world coords.
        key_points, mask_ids, segment_id_to_name = self._augment_mujoco_keypoints_with_zone(
            key_points,
            mask_ids,
            segment_id_to_name,
        )

        key_points_objects_map = self._build_keypoint_object_map(mask_ids, segment_id_to_name)
        self._log_mujoco_keypoint_mapping_errors(key_points, key_points_objects_map, tag="before_refine")
        key_points = self._refine_mujoco_target_keypoint(key_points, key_points_objects_map)
        self._log_mujoco_keypoint_mapping_errors(key_points, key_points_objects_map, tag="after_refine")
        projected_img = self._refresh_projected_keypoints_image(
            rgb,
            key_points,
            mask_ids,
            key_points_objects_map=key_points_objects_map,
            fallback_projected=projected_img,
        )
        
        # Register keypoints
        key_points_objects_map = self.keypoint_tracker.register_keypoints(
            key_points,
            mask_ids=mask_ids,
            segment_id_to_name=segment_id_to_name
        )
        
        # Save for stage recognition (used by Gemini)
        self.init_img_with_keypoints = projected_img  # Initial image with keypoint annotations
        self.keypoint_id_to_object = key_points_objects_map  # Keypoint ID -> object name mapping
        
        self.vlm_agent.task_dir = os.path.join(episode_dir, 'vlm_agent')     
        os.makedirs(self.vlm_agent.task_dir, exist_ok=True)
        
        # Load or generate guidance functions
        if self.cached_functions_dir is not None:
            guidance_functions_dir = self.cached_functions_dir
            if not os.path.exists(guidance_functions_dir):
                raise ValueError(f"cached_functions_dir does not exist: {guidance_functions_dir}")
            log.info(f"Using cached guidance functions from: {guidance_functions_dir}")
        else:
            # Use instruction from adapter (already retrieved above)
            metadata = {
                'init_keypoint_positions': key_points, 
                'num_keypoints': len(key_points), 
                'key_points_objects_map': key_points_objects_map
            }
            try:
                guidance_functions_dir = self.vlm_agent.generate_guidance(
                    projected_img, instruction, metadata
                )
            except Exception as e:
                if self.backend == "mujoco":
                    log.warning(f"VLM guidance generation failed for MuJoCo ({e}), switching to built-in steering guidance.")
                    self._setup_builtin_mujoco_guidance(key_points_objects_map)
                    return
                raise
        
        try:
            # Load guidance functions
            with open(os.path.join(guidance_functions_dir, 'metadata.json'), 'r') as f:
                self.program_info = json.load(f)

            self.guidance_fns = dict()
            has_hardcoded_world_target = False
            for stage in range(1, self.program_info['num_stages'] + 1):
                load_path = os.path.join(guidance_functions_dir, f'stage{stage}_guidance.txt')
                self.guidance_fns[stage] = load_functions_from_txt(load_path) if os.path.exists(load_path) else []
                if self._mujoco_guidance_has_hardcoded_world_target(load_path):
                    has_hardcoded_world_target = True

            if self.backend == "mujoco" and has_hardcoded_world_target:
                log.warning("Detected hardcoded world-coordinate target in MuJoCo VLM guidance; switching to built-in keypoint-based guidance.")
                self._setup_builtin_mujoco_guidance(key_points_objects_map)
                return

            output_raw_path = os.path.join(guidance_functions_dir, 'output_raw.txt')
            self.stage_descriptions = self.vlm_agent._extract_stage_descriptions_from_output(output_raw_path)

            if self.backend == "mujoco" and self._is_grasp_only_instruction(instruction) and not self._mujoco_requires_placement():
                if self.program_info.get('num_stages', 1) > 1:
                    log.info("MuJoCo instruction is grasp-only; enforcing single-stage guidance.")
                stage_names = self.program_info.get('stage_names') or []
                first_stage_name = stage_names[0] if len(stage_names) > 0 else "approach_and_grasp_object"
                self.program_info['num_stages'] = 1
                self.program_info['stage_names'] = [first_stage_name]
                self.guidance_fns = {1: self.guidance_fns.get(1, [])}
                if isinstance(self.stage_descriptions, str):
                    lines = [ln.strip() for ln in self.stage_descriptions.splitlines() if ln.strip()]
                    self.stage_descriptions = lines[0] if len(lines) > 0 else "Stage 1: approach and grasp target object"
        except Exception as e:
            if self.backend == "mujoco":
                log.warning(f"Failed to load VLM guidance files for MuJoCo ({e}), switching to built-in steering guidance.")
                self._setup_builtin_mujoco_guidance(key_points_objects_map)
                return
            raise
        
        for stage, fns in self.guidance_fns.items():
            log.info(f"Stage {stage}: {len(fns)} guidance functions loaded")
        log.info(f"Loaded {len(self.guidance_fns)} stages total")
        log.debug(f"Stage descriptions:\n{self.stage_descriptions}")
    
    def _get_gripper_value(self, action_chunk, action_idx: int) -> Optional[float]:
        """Extract gripper value from action chunk."""
        if action_chunk is None:
            return None
        val = action_chunk[0][action_idx][-1]
        return val.item() if hasattr(val, 'item') else val
    
    def _update_stage(self, state: dict, gripper_val: Optional[float], 
                      upper_th: float, lower_th: float) -> dict:
        """
        Update stage recognition state based on reward and gripper triggers.
        Returns updated state dict.
        """
        curr_reward = self.policy.get_normalized_reward()
        prev_reward = state['prev_norm_reward']
        prev_gripper = state['prev_gripper_open']
        
        # Detect gripper change (action < 0 = OPEN, action > 0 = CLOSE)
        curr_gripper_open = (gripper_val < 0) if gripper_val is not None else None
        gripper_changed = (prev_gripper is not None and curr_gripper_open is not None 
                          and curr_gripper_open != prev_gripper)
        
        # Detect gripper state changes
        gripper_just_closed = gripper_changed and not curr_gripper_open
        gripper_just_opened = gripper_changed and curr_gripper_open
        
        # Schmitt trigger on reward (only relevant when guidance is active)
        reward_rising = state['use_guidance'] and (prev_reward < upper_th and curr_reward >= upper_th)
        reward_falling = state['use_guidance'] and (prev_reward > lower_th and curr_reward <= lower_th)
        
        # Build trigger reason (priority: gripper > reward > chunk interval > periodic)
        trigger_reason = None
        if gripper_just_closed:
            trigger_reason = "gripper closed"
        elif gripper_just_opened:
            trigger_reason = "gripper opened"
        elif reward_rising:
            trigger_reason = f"reward rose above {upper_th:.0%}"
        elif reward_falling:
            trigger_reason = f"reward dropped below {lower_th:.0%}"
        
        # Query VLM if triggered (with limit check)
        vlm_query_limit = self.config.get("vlm_query_limit", 10)
        vlm_query_count = state.get('vlm_query_count', 0)

        if self.gemini_stage_recognizer is not None and getattr(self.gemini_stage_recognizer, 'is_disabled', False):
            log.warning("Gemini stage recognizer disabled; switching to local stage fallback.")
            self.gemini_stage_recognizer = None

        # Local fallback stage transition when external stage recognizer is disabled.
        if self.gemini_stage_recognizer is None and state.get('use_guidance', False):
            max_stage = len(getattr(self, 'guidance_fns', {}))
            if max_stage > 1:
                should_advance = (state['current_stage'] == 1) and (gripper_just_closed or reward_rising)
                if should_advance:
                    next_stage = min(state['current_stage'] + 1, max_stage)
                    if next_stage != state['current_stage']:
                        log.info(f"[LocalStage] Stage: {state['current_stage']} -> {next_stage} (trigger: {trigger_reason or 'reward/gripper'})")
                        self.policy.reset_stage()
                        curr_reward = 0.0
                        state['current_stage'] = next_stage
        
        if trigger_reason and self.gemini_stage_recognizer is not None and vlm_query_count < vlm_query_limit:
            log.info(f"[Trigger] {trigger_reason} (query {vlm_query_count + 1}/{vlm_query_limit})")
            
            new_stage, need_guidance = self.gemini_stage_recognizer.identify_stage_and_guidance(
                current_rgb=np.array(self.adapter.get_vlm_image()),
                instruction=self.adapter.get_task_description(),
                stage_descriptions=self.stage_descriptions,
                init_img_with_keypoints=self.init_img_with_keypoints,
                keypoint_id_to_object=self.keypoint_id_to_object,
                num_stages=len(self.guidance_fns),
                trigger_reason=trigger_reason,
            )
            
            state['vlm_query_count'] = vlm_query_count + 1
            
            if new_stage != state['current_stage'] or need_guidance != state['use_guidance']:
                log.info(f"[VLM] Stage: {state['current_stage']} → {new_stage}, Guidance: {state['use_guidance']} → {need_guidance}")
                if new_stage != state['current_stage']:
                    self.policy.reset_stage()
                    curr_reward = 0.0  # Reset for new stage
            
            state.update({
                'current_stage': new_stage,
                'use_guidance': need_guidance,
            })
        elif trigger_reason and vlm_query_count >= vlm_query_limit:
            log.debug(f"[Trigger] {trigger_reason} (skipped, limit {vlm_query_limit} reached)")
        
        # Update tracking state
        state['prev_norm_reward'] = curr_reward
        state['prev_gripper_open'] = curr_gripper_open
        return state
    
    def run(self):
        """Main running loop."""
        self.success_count = 0
        base_output_dir = self.output_dir
        episode_num = self.config.get('episode_num', 10)

        if not self.config.get("use_guidance", True):
            log.warning("Guidance is disabled (main.use_guidance=false). Running policy-only rollout.")
        
        for episode in range(episode_num):
            episode_seed = torch.randint(0, 1000, (1,)).item()
            self.policy.reset()
            self.video_recorder.clear()
            episode_dir = os.path.join(base_output_dir, f'episode_{episode+1}')
            os.makedirs(episode_dir, exist_ok=True)
            
            # Reset environment
            self.adapter.reset(seed=episode_seed)
            
            # Update guide_scale for current task (may change when switching tasks)
            task_info = self.adapter.get_task_info()
            recommended_scale = task_info.get('recommended_guide_scale')
            if recommended_scale is not None:
                self.current_guide_scale = recommended_scale
            task_label = task_info.get('task_id') or task_info.get('task_name') or task_info.get('instruction') or '?'
            log.info(f"Task {task_label}: guide_scale={self.current_guide_scale}")
            
            # Perform task preparation with error handling
            episode_error = False
            if self.config.get("use_guidance", True):
                try:
                    self.perform_task_for_episode(episode_dir)
                except Exception as e:
                    log.error(f"Episode {episode+1} preparation failed: {e}")
                    episode_error = True
                    # Save error info
                    error_file = os.path.join(episode_dir, 'error.txt')
                    with open(error_file, 'w') as f:
                        import traceback
                        f.write(f"Error during episode preparation:\n{traceback.format_exc()}")
                # Reset stage manager
            
            # Skip this episode if preparation failed
            if episode_error:
                log.warning(f"Skipping episode {episode+1} due to preparation error")
                # Save a placeholder fail video/marker
                fail_marker = os.path.join(episode_dir, f'episode_{episode+1}_fail_error.txt')
                with open(fail_marker, 'w') as f:
                    f.write("Episode failed due to guidance function error\n")
                continue
            
            # Wrap entire episode execution in try-except
            try:
                self._run_episode(episode, episode_dir)
            except Exception as e:
                log.error(f"Episode {episode+1} execution failed: {e}")
                import traceback
                error_file = os.path.join(episode_dir, 'error.txt')
                with open(error_file, 'w') as f:
                    f.write(f"Error during episode execution:\n{traceback.format_exc()}")
                # Save a placeholder fail marker
                fail_marker = os.path.join(episode_dir, f'episode_{episode+1}_fail_error.txt')
                with open(fail_marker, 'w') as f:
                    f.write(f"Episode failed due to execution error: {e}\n")
                log.warning(f"Skipping episode {episode+1} due to execution error")
                continue
        
        # Final statistics
        success_rate = self.success_count / episode_num * 100
        log.info(f"Tested {episode_num} episodes, success rate: {success_rate:.2f}%")
        
        # Save results
        log_file = os.path.join(self.output_dir, 'results.txt')
        with open(log_file, 'a') as f:
            f.write(f"Backend: {self.backend}\n")
            f.write(f"Success count: {self.success_count}/{episode_num}\n")
            f.write(f"Success rate: {success_rate:.2f}%\n")


    def _save_episode_compat_video(self, video_base_path: str):
        """Save compatibility video without camera suffix: episode_{n}_{success|fail}.mp4."""
        cam_name = getattr(self.adapter, "vlm_camera", None)
        if not cam_name:
            return

        src = f"{video_base_path}_{cam_name}.mp4"
        dst = f"{video_base_path}.mp4"
        if not os.path.exists(src):
            return
        shutil.copyfile(src, dst)
        log.info(f"Saved compatibility episode video to {dst}")

    def _save_keypoints_tracking_video(self, episode_dir: str, frames: list[np.ndarray]):
        """Save keypoint tracking video for the episode."""
        if len(frames) == 0:
            return

        max_h = max(frame.shape[0] for frame in frames)
        max_w = max(frame.shape[1] for frame in frames)
        target_shape = (max_h, max_w, 3)

        normalized_frames: list[np.ndarray] = []
        for frame in frames:
            if frame.shape != target_shape:
                padded = np.zeros(target_shape, dtype=np.uint8)
                padded[:frame.shape[0], :frame.shape[1], :] = frame
                frame = padded
            normalized_frames.append(frame)

        video_path = os.path.join(episode_dir, "keypoints_tracking.mp4")
        imageio.mimsave(
            video_path,
            normalized_frames,
            fps=self.video_recorder.fps,
            codec='libx264',
            quality=8
        )
        log.info(f"Saved keypoint tracking video to {video_path} ({len(normalized_frames)} frames)")

    def _run_episode(self, episode: int, episode_dir: str):
        """Run a single episode with the current configuration."""
        from utils.vis_utils import draw_keypoints_on_image

        observation = self._get_policy_observation()
        
        # Evaluation loop
        done = False
        global_steps = 0
        keypoints = None
        mask_ids = None
        generate_new_chunk = False
        action_executed = 0
        use_guidance = False
        action_horizon = self.policy._action_chunk_horizon
        current_stage = 1
        current_guidance_fns = None
        trajectory_frame_idx = 0
        keypoint_tracking_frames: list[np.ndarray] = []
        
        # Stage recognition thresholds
        UPPER_THRESHOLD = self.config.get("schmitt_upper", 0.8)
        LOWER_THRESHOLD = self.config.get("schmitt_lower", 0.6)
        action_chunk = None
        
        log.info(f"Task description: {self.adapter.get_task_description()}")
        
        # Default: start with stage 1 and guidance ON
        current_stage = 1
        use_guidance = self.config.get("use_guidance", True) and hasattr(self, 'guidance_fns')
        current_guidance_fns = self.guidance_fns.get(current_stage, []) if use_guidance else None
        
        # Check if guidance functions are available
        if use_guidance and not current_guidance_fns:
            log.warning(f"No guidance functions for initial stage {current_stage}")
        else:
            log.info(f"Initial guidance: stage={current_stage}, use_guidance={use_guidance}, "
                    f"num_fns={len(current_guidance_fns) if current_guidance_fns else 0}")
        
        # Stage recognition state
        stage_state = {
            'prev_norm_reward': 0.0,
            'prev_gripper_open': None,
            'current_stage': current_stage,
            'use_guidance': use_guidance,
            'vlm_query_count': 0,
        }
        
        while not done:
            generate_new_chunk = (action_executed == 0)

            if generate_new_chunk and self.config.get("use_guidance", True) and hasattr(self, 'guidance_fns'):
                keypoints = self.keypoint_tracker.get_keypoint_positions()
                keypoints = self._refine_live_mujoco_guidance_keypoints(
                    keypoints,
                    getattr(self, "keypoint_id_to_object", {}),
                )
                mask_ids = self.keypoint_tracker.get_mask_ids()
                
                # Update stage recognition
                gripper_val = self._get_gripper_value(action_chunk, action_executed)
                stage_state = self._update_stage(stage_state, gripper_val, UPPER_THRESHOLD, LOWER_THRESHOLD)
                
                current_stage = stage_state['current_stage']
                use_guidance = stage_state['use_guidance']
                current_guidance_fns = self.guidance_fns.get(current_stage, []) if use_guidance else None
                
                if use_guidance and not current_guidance_fns:
                    log.warning(f"No guidance functions for stage {current_stage}, disabling")
                    use_guidance = False
                    current_guidance_fns = None
            elif generate_new_chunk:
                use_guidance = False
                keypoints = None
                current_guidance_fns = None
                
            # Get parameters from config
            guide_scale = getattr(self, 'current_guide_scale', self.config.get("guide_scale", 80.0))
            sigmoid_k = self.config.get("sigmoid_k", 12.0)
            sigmoid_x0 = self.config.get("sigmoid_x0", 0.7)
            
            action_chunk = self.policy.select_action(
                observation,
                generate_new_chunk=generate_new_chunk,
                use_guidance=use_guidance,
                keypoints=keypoints,
                guidance_fns=current_guidance_fns,
                guide_scale=guide_scale,
                sigmoid_k=sigmoid_k,
                sigmoid_x0=sigmoid_x0,
                start_ratio=self.config.get("start_ratio", None),
                use_diversity=self.config.get("use_diversity", True),
                diversity_scale=self.config.get("diversity_scale", 10.0),
                MCMC_steps=self.config.get("MCMC_steps", 4),
                verbose=True,
                use_fkd=self.config.get("use_fkd", False),
                fkd_config=OmegaConf.to_container(self.config.get("fkd", {}), resolve=True) if self.config.get("fkd") else None,
                global_step=global_steps,
                current_stage=current_stage,
            )

            # Get image and add status overlay
            if self.config.get("debug_draw_trajectory", False):
                from utils.vis_utils import draw_action_trajectory_on_vlm_image
                image = draw_action_trajectory_on_vlm_image(
                    adapter=self.adapter,
                    action_chunk=action_chunk[:, action_executed:],
                    num_steps=action_horizon,
                    global_step=global_steps,
                    action_executed=action_executed,
                )
            else:
                image = np.array(self.adapter.get_vlm_image())
            
            # Draw keypoints on image for debugging (disabled by default)
            if self.config.get("debug_draw_keypoints", False) and keypoints is not None:
                image = draw_keypoints_on_image(
                    adapter=self.adapter,
                    image=image,
                    keypoints=keypoints,
                    mask_ids=mask_ids
                )

            # Build dedicated keypoint-tracking frame (separate from trajectory/status debug video).
            keypoint_frame = np.array(self.adapter.get_vlm_image())
            tracker_mask_ids = self.keypoint_tracker.get_mask_ids()
            if tracker_mask_ids.size > 0:
                tracker_keypoints = self.keypoint_tracker.get_keypoint_positions()
                tracker_keypoints = self._refine_live_mujoco_guidance_keypoints(
                    tracker_keypoints,
                    getattr(self, "keypoint_id_to_object", {}),
                )
                keypoint_frame = draw_keypoints_on_image(
                    adapter=self.adapter,
                    image=keypoint_frame,
                    keypoints=tracker_keypoints,
                    mask_ids=tracker_mask_ids,
                )
            keypoint_frame = add_text_to_image(
                keypoint_frame,
                [f"Step:{global_steps} Stage:{current_stage}", "Keypoints:TRACK"],
            )
            keypoint_tracking_frames.append(keypoint_frame)
            
            # Add guidance status overlay
            gripper_val = self._get_gripper_value(action_chunk, action_executed)
            gripper_str = f"Grip:{'O' if gripper_val and gripper_val < 0 else 'C'}({gripper_val:.2f})" if gripper_val else "Grip:-"
            
            status_text = [
                f"Step:{global_steps} Stage:{current_stage}",
                f"Guide:{'ON' if use_guidance else 'OFF'} {gripper_str}",
            ]
            
            # Add reward-based guidance info
            if hasattr(self.policy, 'get_normalized_reward') and use_guidance:
                norm_r = self.policy.get_normalized_reward()
                scale = self.policy.get_last_scale()
                
                # Use config values for consistent display
                k = self.config.get("sigmoid_k", 12.0)
                x0 = self.config.get("sigmoid_x0", 0.8)
                sig_strength = 1.0 / (1.0 + np.exp(k * (norm_r - x0)))
                
                # Show scale as "-" if not yet computed (first chunk before guidance runs)
                scale_str = f"{scale:.1f}" if scale > 0 else "-"
                status_text.extend([
                    f"Norm_R: {norm_r:.2f}",
                    f"Sig_Str: {sig_strength:.1%}",
                    f"Scale: {scale_str}",
                ])
            
            image_with_status = add_text_to_image(image, status_text)
            self.video_recorder.add_frame(self.adapter.vlm_camera, image_with_status)
            obs, reward, terminated, truncated, info = self.adapter.step(action_chunk[0][action_executed])

            action_executed += 1
            if action_executed == action_horizon:
                action_executed = 0
            
            observation = self._get_policy_observation()
            global_steps += 1
            
            if terminated or truncated:
                is_success = info.get('success', False)
                if is_success:
                    done = True
                    self.success_count += 1
                
                behavior_name = info.get("behavior_name", "unknown")
                video_path = os.path.join(episode_dir, f'episode_{episode+1}_{"success" if is_success else "fail"}')
                self.video_recorder.save_video(save_path=video_path, success=is_success, behavior_name=behavior_name)
                self._save_episode_compat_video(video_path)
                self._save_keypoints_tracking_video(episode_dir, keypoint_tracking_frames)
                break
        
        log.info(f"Episode {episode+1} finished, success: {info.get('success', False)}, steps: {global_steps}")
    
    def run_test(self):
        """Test segmentation and keypoint detection."""
        self.adapter.reset()
        rgb, depth, points, segmentation, segment_id_to_name = self.adapter.get_keypoint_detection_inputs()
        
        # Save images
        rgb_path = os.path.join(self.output_dir, 'rgb_static.png')
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        log.debug(f"Saved RGB image to: {rgb_path}")
        
        # Visualize segmentation
        unique_ids = np.unique(segmentation)
        seg_colored = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
        np.random.seed(42)
        for seg_idx in unique_ids:
            if seg_idx == 0:
                continue
            color = np.random.randint(50, 255, 3).tolist()
            seg_colored[segmentation == seg_idx] = color
        
        seg_path = os.path.join(self.output_dir, 'segmentation.png')
        cv2.imwrite(seg_path, seg_colored)
        log.debug(f"Saved segmentation image to: {seg_path}")
        
        log.info("Interactable objects:")
        for seg_idx, name in segment_id_to_name.items():
            if seg_idx > 0:
                pixel_count = np.sum(segmentation == seg_idx)
                if pixel_count > 0:
                    log.debug(f"  [{seg_idx}] {name}: {pixel_count} pixels")
        
        # Detect keypoints
        log.info("Detecting keypoints...")
        keypoints, projected_img, mask_ids, dino_vis = self.keypoint_detector.get_keypoints_with_visualization(
            rgb=rgb,
            points=points,
            segmentation=segmentation,
            segment_id_to_name=segment_id_to_name,
            save_dir=self.output_dir
        )
        
        projected_path = os.path.join(self.output_dir, 'keypoints_projected.png')
        cv2.imwrite(projected_path, cv2.cvtColor(projected_img, cv2.COLOR_RGB2BGR))
        log.info(f"Saved keypoint projection to: {projected_path}")
        
        log.info(f"Detected {len(keypoints)} keypoints:")
        for i, (kp, mid) in enumerate(zip(keypoints, mask_ids)):
            obj_name = segment_id_to_name.get(mid, f"unknown_{mid}")
            log.info(f"  [{i}] {obj_name}: position=[{kp[0]:.4f}, {kp[1]:.4f}, {kp[2]:.4f}]")


def _normalize_hydra_cli_args(argv: List[str]) -> List[str]:
    """Normalize legacy CLI flags to Hydra-compatible flags.

    Supports:
        --config config.yaml
        --config=config.yaml
        --config path/to/config.yaml
    """
    normalized = [argv[0]]
    idx = 1

    while idx < len(argv):
        arg = argv[idx]

        if arg == "--config":
            if idx + 1 >= len(argv):
                raise ValueError("--config requires a value, e.g. --config config.yaml")
            config_value = argv[idx + 1]
            idx += 2
        elif arg.startswith("--config="):
            config_value = arg.split("=", 1)[1]
            idx += 1
        else:
            normalized.append(arg)
            idx += 1
            continue

        if not config_value:
            raise ValueError("--config value cannot be empty")

        config_path, config_file = os.path.split(config_value)
        config_name, ext = os.path.splitext(config_file)

        if ext in {".yaml", ".yml"}:
            target_name = config_name
        else:
            target_name = config_file

        if not target_name:
            raise ValueError(f"Invalid --config value: {config_value}")

        if config_path:
            normalized.extend(["--config-path", config_path])
        normalized.extend(["--config-name", target_name])

    return normalized


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point with Hydra configuration.
    
    Usage:
        # CALVIN (uses task configs)
        python main.py env=calvin task=drawer_open
        python main.py env=calvin task=button_on
        
        # LIBERO (uses suite_name directly, no task configs)
        python main.py env=libero env.libero.suite_name=libero_goal
        python main.py env=libero env.libero.suite_name=libero_spatial
        
        # Override parameters
        python main.py main.episode_num=50 main.guide_scale=120
    """
    # Print resolved config
    log.info(f"Working directory: {os.getcwd()}")
    log.info(f"Output directory: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    
    env_backend = cfg.backend.backend
    log.info(f"Using environment: {env_backend}")
    
    # Backend-specific logging
    if env_backend == "calvin":
        target_behavior = cfg.backend.calvin.get("target_behavior")
        log.info(f"Target behavior: {target_behavior or 'any'}")
    elif env_backend == "libero":
        log.info(f"Suite: {cfg.backend.libero.suite_name}")
    elif env_backend == "mujoco":
        log.info(f"Task: {cfg.backend.mujoco.get('task_name', 'unknown')}")
        log.info(f"Target object: {cfg.backend.mujoco.get('target_object', 'unknown')}")
    
    # Initialize and run
    runner = Main(cfg)
    runner.run()


if __name__ == "__main__":
    try:
        sys.argv = _normalize_hydra_cli_args(sys.argv)
    except ValueError as exc:
        print(f"CLI argument error: {exc}", file=sys.stderr)
        sys.exit(2)
    main()
