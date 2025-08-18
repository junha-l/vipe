# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import logging
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import imageio
import Imath
import numpy as np
import OpenEXR
import torch

from vipe.ext.lietorch import SE3
from vipe.slam.system import SLAMOutput
from vipe.streams.base import FrameAttribute, VideoFrame, VideoStream
from vipe.utils.cameras import CameraType
from vipe.utils.geometry import se3_matrix_to_se3
from vipe.utils.visualization import VideoWriter

logger = logging.getLogger(__name__)


def _convert_pose_to_z_up_axis(pose_matrix: np.ndarray) -> np.ndarray:
    transform = np.array([
        [1,  0,  0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=np.float32)
    
    converted_pose = pose_matrix.copy()
    
    # Transform rotation: R' = T * R
    converted_pose[:3, :3] = transform @ pose_matrix[:3, :3]
    
    # Transform translation: t' = T * t  
    converted_pose[:3, 3] = transform @ pose_matrix[:3, 3]
    
    return converted_pose


def _convert_points_to_z_up_axis(points: np.ndarray) -> np.ndarray:
    transform = np.array([
        [1,  0,  0],
        [0,  0, 1],
        [0,  -1,  0]
    ], dtype=np.float32)
    
    return points @ transform.T



@dataclass
class ArtifactPath:
    base_path: Path
    artifact_name: str
    
    @property
    def pcd_path(self) -> Path:
        return self.base_path / f"{self.artifact_name}" / "pcd.npz"

    @property
    def rgbd_tar_path(self) -> Path:
        return self.base_path / f"{self.artifact_name}" / "rgbd.tar"

    @property
    def rgb_path(self) -> Path:
        return self.base_path / "rgb" / f"{self.artifact_name}.mp4"

    @property
    def pose_path(self) -> Path:
        return self.base_path / "pose" / f"{self.artifact_name}.npz"

    @property
    def depth_path(self) -> Path:
        return self.base_path / "depth" / f"{self.artifact_name}.zip"

    @property
    def intrinsics_path(self) -> Path:
        return self.base_path / "intrinsics" / f"{self.artifact_name}.npz"

    @property
    def camera_type_path(self) -> Path:
        return self.base_path / "intrinsics" / f"{self.artifact_name}_camera.txt"

    @property
    def flow_path(self) -> Path:
        return self.base_path / "flow" / f"{self.artifact_name}.zip"

    @property
    def mask_path(self) -> Path:
        return self.base_path / "mask" / f"{self.artifact_name}.zip"

    @property
    def mask_phrase_path(self) -> Path:
        return self.base_path / "mask" / f"{self.artifact_name}.txt"

    @property
    def meta_info_path(self) -> Path:
        return self.base_path / "vipe" / f"{self.artifact_name}_info.pkl"

    @classmethod
    def glob_artifacts(cls, base_path: Path, use_video: bool = False) -> Iterator["ArtifactPath"]:
        if use_video:
            for artifact_path in (base_path / "rgb").glob("*.mp4"):
                artifact_name = artifact_path.stem
                yield cls(base_path, artifact_name)
        else:
            for artifact_path in (base_path / "vipe").glob("*_info.pkl"):
                artifact_name = artifact_path.stem.replace("_info", "")
                yield cls(base_path, artifact_name)

    @property
    def meta_vis_path(self) -> Path:
        return self.base_path / "vipe" / f"{self.artifact_name}_vis.mp4"

    @property
    def essential_paths(self) -> list[Path]:
        return [
            self.rgb_path,
            self.pose_path,
            self.depth_path,
            self.intrinsics_path,
            self.flow_path,
            self.mask_path,
            self.mask_phrase_path,
            self.meta_info_path,
            self.meta_vis_path,
        ]

    @property
    def eval_metrics_path(self) -> Path:
        return self.base_path / "eval" / f"{self.artifact_name}_metrics.pkl"

    @property
    def eval_traj_vis_path(self) -> Path:
        return self.base_path / "eval" / f"{self.artifact_name}_trajectory_vis.png"

    @property
    def eval_gt_pose_path(self) -> Path:
        return self.base_path / "eval" / f"{self.artifact_name}_pose_gt.npz"

    @property
    def eval_gt_intrinsics_path(self) -> Path:
        return self.base_path / "eval" / f"{self.artifact_name}_intrinsics_gt.npz"

    @property
    def eval_gt_camera_type_path(self) -> Path:
        return self.base_path / "eval" / f"{self.artifact_name}_camera_gt.txt"

    @property
    def eval_gt_depth_path(self) -> Path:
        return self.base_path / "eval" / f"{self.artifact_name}_depth_gt.zip"

    @property
    def aux_vis_plot_path(self) -> Path:
        return self.base_path / "vipe_aux_vis" / f"{self.artifact_name}_plot.png"

    @property
    def aux_vis_traj_path(self) -> Path:
        return self.base_path / "vipe_aux_vis" / f"{self.artifact_name}_traj.mp4"


def save_pose_artifacts(out_path: ArtifactPath, cached_final_stream: VideoStream, gt: bool = False) -> None:
    # Save OpenCV cam2world matrices as 4x4 matrix in npz file
    if gt:
        pose_list = cached_final_stream.get_gt_stream_attribute(FrameAttribute.POSE)
        path = out_path.eval_gt_pose_path
    else:
        pose_list = cached_final_stream.get_stream_attribute(FrameAttribute.POSE)
        path = out_path.pose_path

    pose_list = [
        (frame_idx, pose_data.matrix().cpu().numpy())
        for frame_idx, pose_data in enumerate(pose_list)
        if pose_data is not None
    ]
    if len(pose_list) > 0:
        pose_data = np.stack([pose for _, pose in pose_list], axis=0)
        pose_inds = np.array([frame_idx for frame_idx, _ in pose_list])
        path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(path, data=pose_data, inds=pose_inds)


def read_pose_artifacts(npz_file_path: Path) -> tuple[np.ndarray, SE3]:
    data = np.load(npz_file_path)
    return data["inds"], se3_matrix_to_se3(data["data"])


def read_pose_artifacts_benchmark(npz_file_path: Path) -> dict:
    data = np.load(npz_file_path)
    return dict(
        ids=data["inds"],
        trajectory=se3_matrix_to_se3(data["data"]),
        runtime=data.get("runtime", None),
        keyframe_ids=data.get("keyframe_ids", None),
        frame_num=len(data["inds"]),
    )


def save_intrinsics_artifacts(out_path: ArtifactPath, cached_final_stream: VideoStream, gt: bool = False) -> None:
    # Save intrinsics as [fx, fy, cx, cy] in npz file
    if gt:
        intrinsics_list = cached_final_stream.get_gt_stream_attribute(FrameAttribute.INTRINSICS)
        camera_type_list = cached_final_stream.get_gt_stream_attribute(FrameAttribute.CAMERA_TYPE)
        intr_path = out_path.eval_gt_intrinsics_path
        camera_type_path = out_path.eval_gt_camera_type_path
    else:
        intrinsics_list = cached_final_stream.get_stream_attribute(FrameAttribute.INTRINSICS)
        camera_type_list = cached_final_stream.get_stream_attribute(FrameAttribute.CAMERA_TYPE)
        intr_path = out_path.intrinsics_path
        camera_type_path = out_path.camera_type_path

    intrinsics_list = [
        (frame_idx, intr_data.cpu().numpy())
        for frame_idx, intr_data in enumerate(intrinsics_list)
        if intr_data is not None
    ]
    if len(intrinsics_list) > 0:
        intrinsics_data = np.stack([intrinsics for _, intrinsics in intrinsics_list], axis=0)
        intrinsics_inds = np.array([frame_idx for frame_idx, _ in intrinsics_list])
        intr_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(intr_path, data=intrinsics_data, inds=intrinsics_inds)

    camera_type_list = [
        (frame_idx, camera_type_data)
        for frame_idx, camera_type_data in enumerate(camera_type_list)
        if camera_type_data is not None
    ]
    if len(camera_type_list) > 0:
        camera_type_path.parent.mkdir(exist_ok=True, parents=True)
        with camera_type_path.open("w") as f:
            for frame_idx, camera_type_data in camera_type_list:
                f.write(f"{frame_idx}: {camera_type_data.name}\n")


def read_intrinsics_artifacts(
    intr_file_path: Path, camera_file_path: Path | None = None
) -> tuple[np.ndarray, torch.Tensor, list[CameraType]]:
    data = np.load(intr_file_path)
    inds, intrinsics = data["inds"], torch.from_numpy(data["data"])
    if camera_file_path is None or not camera_file_path.exists():
        assert intrinsics.shape[1] == 4
        camera_types = [CameraType.PINHOLE] * intrinsics.shape[0]

    else:
        with camera_file_path.open("r") as f:
            camera_types = [CameraType[line.split(":")[1].strip()] for line in f.readlines()]

    return inds, intrinsics, camera_types


def save_rgb_artifacts(out_path: ArtifactPath, cached_final_stream: VideoStream) -> None:
    # Save original RGB as H264-encoded video.
    with VideoWriter(out_path.rgb_path, cached_final_stream.fps()) as rgb_writer:
        for frame_data in cached_final_stream:
            rgb_writer.write((frame_data.rgb.cpu().numpy() * 255).astype(np.uint8))


def read_rgb_artifacts(rgb_file_path: Path) -> Iterator[tuple[int, torch.Tensor]]:
    """
    Read RGB from H264-encoded video.
    """
    reader = imageio.get_reader(rgb_file_path, "ffmpeg")
    for frame_idx, rgb in enumerate(reader):
        rgb = torch.from_numpy(rgb) / 255.0
        yield frame_idx, rgb


def save_depth_artifacts(out_path: ArtifactPath, cached_final_stream: VideoStream, gt: bool = False) -> None:
    # Save metric depth as zipped exr files.
    if gt:
        metric_depth_list = cached_final_stream.get_gt_stream_attribute(FrameAttribute.METRIC_DEPTH)
        path = out_path.eval_gt_depth_path
    else:
        metric_depth_list = cached_final_stream.get_stream_attribute(FrameAttribute.METRIC_DEPTH)
        path = out_path.depth_path

    metric_depth_list = [
        (frame_idx, depth_data.cpu().numpy())
        for frame_idx, depth_data in enumerate(metric_depth_list)
        if depth_data is not None
    ]
    if len(metric_depth_list) > 0:
        path.parent.mkdir(exist_ok=True, parents=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for frame_idx, metric_depth in metric_depth_list:
                height, width = metric_depth.shape
                header = OpenEXR.Header(width, height)
                header["channels"] = {"Z": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))}
                with tempfile.NamedTemporaryFile(suffix=".exr") as f:
                    exr = OpenEXR.OutputFile(f.name, header)
                    exr.writePixels({"Z": metric_depth.astype(np.float16).tobytes()})
                    exr.close()
                    z.write(f.name, f"{frame_idx:05d}.exr")


def read_depth_artifacts(zip_file_path: Path) -> Iterator[tuple[int, torch.Tensor]]:
    """
    Read metric depth from zipped exr files.
    """
    valid_width, valid_height = 0, 0
    with zipfile.ZipFile(zip_file_path, "r") as z:
        for file_name in sorted(z.namelist()):
            frame_idx = int(file_name.split(".")[0])
            with z.open(file_name) as f:
                try:
                    exr = OpenEXR.InputFile(f)
                except OSError:
                    # Sometimes EXR loader might fail, we return all nan maps.
                    logger.warning(f"Failed to load EXR file {zip_file_path}-{file_name}. Returning all nan maps.")
                    assert valid_width > 0 and valid_height > 0
                    yield (
                        frame_idx,
                        torch.full(
                            (valid_height, valid_width),
                            float("nan"),
                            dtype=torch.float32,
                        ),
                    )
                    continue
                header = exr.header()
                dw = header["dataWindow"]
                valid_width = width = dw.max.x - dw.min.x + 1
                valid_height = height = dw.max.y - dw.min.y + 1
                channels = exr.channels(["Z"])
                depth_data = np.frombuffer(channels[0], dtype=np.float16).reshape((height, width))
                yield frame_idx, torch.from_numpy(depth_data.copy()).float()


def read_instance_artifacts(
    zip_file_path: Path,
) -> Iterator[tuple[int, torch.Tensor]]:
    """
    Read instance mask from zipped PNG files.
    """
    with zipfile.ZipFile(zip_file_path, "r") as z:
        for file_name in sorted(z.namelist()):
            frame_idx = int(file_name.split(".")[0])
            with z.open(file_name) as f:
                mask_buffer = np.frombuffer(f.read(), dtype=np.uint8)
                mask = cv2.imdecode(mask_buffer, cv2.IMREAD_UNCHANGED)
                yield frame_idx, torch.from_numpy(mask.copy()).byte()


def read_instance_phrases(instance_phrase_path: Path) -> dict[int, str]:
    """
    Read instance phrases from txt file.
    """
    instance_phrases = {}
    with instance_phrase_path.open("r") as f:
        for line in f.readlines():
            idx, phrase = line.split(":")
            instance_phrases[int(idx)] = phrase.strip()
    return instance_phrases


def save_pcd_artifacts(out_path: ArtifactPath, slam_output: SLAMOutput, z_up: bool = False) -> None:
    """
    Save point cloud as npz file.
    
    Args:
        out_path: ArtifactPath for saving
        slam_output: SLAMOutput containing point cloud data
        coordinate_frame: "opencv" (default, -Y up) or "standard" (+Z up)
    """
    # Save point cloud as npz file.
    pcd_xyz = slam_output.slam_map.dense_disp_xyz.cpu().numpy()
    pcd_rgb = slam_output.slam_map.dense_disp_rgb.cpu().numpy()
    pcd_rgb = pcd_rgb.reshape(-1, 3)
    pcd_xyz = pcd_xyz.astype(np.float32)
    
    # Apply coordinate frame conversion if requested
    if z_up:
        pcd_xyz = _convert_points_to_z_up_axis(pcd_xyz)
    
    out_path.pcd_path.parent.mkdir(exist_ok=True, parents=True)
    np.savez(out_path.pcd_path, coord=pcd_xyz, color=(pcd_rgb * 255).astype(np.uint8))

def save_artifacts_as_tar(out_path: ArtifactPath, cached_final_stream: VideoStream, slam_output: SLAMOutput, z_up: bool = False) -> None:
    """
    Save RGB, depth, pose, and intrinsics in a single tar file and PCD separately.
    
    Args:
        out_path: ArtifactPath for saving
        cached_final_stream: VideoStream containing frame data
        slam_output: SLAMOutput containing point cloud data
    """
    scene_name = out_path.artifact_name
    
    # Save point cloud as npz file (separate from tar)
    save_pcd_artifacts(out_path, slam_output, z_up=z_up)
    
    # Create the rgbd.tar file
    out_path.rgbd_tar_path.parent.mkdir(exist_ok=True, parents=True)
    
    with tarfile.open(out_path.rgbd_tar_path, "w") as tar:
        # Process each frame in the stream
        for frame_idx, frame_data in enumerate(cached_final_stream):
            frame_id_str = f"{frame_idx:05d}"
            
            # Save RGB as JPG
            if frame_data.rgb is not None:
                rgb_data = (frame_data.rgb.cpu().numpy() * 255).astype(np.uint8)
                _, rgb_buffer = cv2.imencode('.jpg', cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR))
                rgb_info = tarfile.TarInfo(name=f"{scene_name}/{frame_id_str}.rgb.jpg")
                rgb_info.size = len(rgb_buffer)
                tar.addfile(rgb_info, io.BytesIO(rgb_buffer.tobytes()))
            
            # Save depth as PNG (scaled by 1000.0, uint16)
            depth_data = cached_final_stream.get_stream_attribute(FrameAttribute.METRIC_DEPTH)[frame_idx]
            if depth_data is not None:
                depth_scaled = (depth_data.cpu().numpy() * 1000.0).astype(np.uint16)
                _, depth_buffer = cv2.imencode('.png', depth_scaled)
                depth_info = tarfile.TarInfo(name=f"{scene_name}/{frame_id_str}.depth.png")
                depth_info.size = len(depth_buffer)
                tar.addfile(depth_info, io.BytesIO(depth_buffer.tobytes()))
            
            # Save pose as TXT (4x4 matrix)
            pose_data = cached_final_stream.get_stream_attribute(FrameAttribute.POSE)[frame_idx]
            if pose_data is not None:
                pose_matrix = pose_data.matrix().cpu().numpy()
                
                # Apply coordinate frame conversion if requested
                if z_up:
                    pose_matrix = _convert_pose_to_z_up_axis(pose_matrix)
                
                pose_str = io.StringIO()
                np.savetxt(pose_str, pose_matrix, fmt="%.6f")
                pose_bytes = pose_str.getvalue().encode('utf-8')
                pose_info = tarfile.TarInfo(name=f"{scene_name}/{frame_id_str}.pose.txt")
                pose_info.size = len(pose_bytes)
                tar.addfile(pose_info, io.BytesIO(pose_bytes))
            
            # Save intrinsics as TXT (3x3 matrix)
            intrinsics_data = cached_final_stream.get_stream_attribute(FrameAttribute.INTRINSICS)[frame_idx]
            if intrinsics_data is not None:
                # Convert from [fx, fy, cx, cy] to 3x3 matrix
                fx, fy, cx, cy = intrinsics_data.cpu().numpy()
                intrinsics_matrix = np.array([
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0]
                ])
                intr_str = io.StringIO()
                np.savetxt(intr_str, intrinsics_matrix, fmt="%.6f")
                intr_bytes = intr_str.getvalue().encode('utf-8')
                intr_info = tarfile.TarInfo(name=f"{scene_name}/{frame_id_str}.intr.txt")
                intr_info.size = len(intr_bytes)
                tar.addfile(intr_info, io.BytesIO(intr_bytes))

    # Save Instance mask as zipped PNG files.
    instance_list = [
        (frame_idx, frame_data.instance)
        for frame_idx, frame_data in enumerate(cached_final_stream)
        if frame_data.instance is not None
    ]
    if len(instance_list) > 0:
        out_path.mask_path.parent.mkdir(exist_ok=True, parents=True)
        with zipfile.ZipFile(out_path.mask_path, "w", zipfile.ZIP_DEFLATED) as z:
            for frame_idx, instance in instance_list:
                _, mask_buffer = cv2.imencode(".png", instance.cpu().numpy().astype(np.uint8))
                z.writestr(f"{frame_idx:05d}.png", mask_buffer.tobytes())

    # Save Instance phrases as txt file.
    instance_phrases_combined = {}
    for frame_data in cached_final_stream:
        assert isinstance(frame_data, VideoFrame)
        if frame_data.instance_phrases is not None:
            instance_phrases_combined.update(frame_data.instance_phrases)
    if len(instance_phrases_combined) > 0:
        out_path.mask_phrase_path.parent.mkdir(exist_ok=True, parents=True)
        with out_path.mask_phrase_path.open("w") as f:
            for idx, phrase in instance_phrases_combined.items():
                f.write(f"{idx}: {phrase}\n")


def save_artifacts(out_path: ArtifactPath, cached_final_stream: VideoStream) -> None:
    """
    Save each attribute independently.
    """

    # Save OpenCV cam2world matrices as 4x4 matrix in npz file
    save_pose_artifacts(out_path, cached_final_stream)

    # Save intrinsics as [fx, fy, cx, cy] in npz file
    save_intrinsics_artifacts(out_path, cached_final_stream)

    # Save original RGB as H264-encoded video.
    save_rgb_artifacts(out_path, cached_final_stream)

    # Save metric depth as zipped exr files.
    save_depth_artifacts(out_path, cached_final_stream)

    # Save Instance mask as zipped PNG files.
    instance_list = [
        (frame_idx, frame_data.instance)
        for frame_idx, frame_data in enumerate(cached_final_stream)
        if frame_data.instance is not None
    ]
    if len(instance_list) > 0:
        out_path.mask_path.parent.mkdir(exist_ok=True, parents=True)
        with zipfile.ZipFile(out_path.mask_path, "w", zipfile.ZIP_DEFLATED) as z:
            for frame_idx, instance in instance_list:
                _, mask_buffer = cv2.imencode(".png", instance.cpu().numpy().astype(np.uint8))
                z.writestr(f"{frame_idx:05d}.png", mask_buffer.tobytes())

    # Save Instance phrases as txt file.
    instance_phrases_combined = {}
    for frame_data in cached_final_stream:
        assert isinstance(frame_data, VideoFrame)
        if frame_data.instance_phrases is not None:
            instance_phrases_combined.update(frame_data.instance_phrases)
    if len(instance_phrases_combined) > 0:
        out_path.mask_phrase_path.parent.mkdir(exist_ok=True, parents=True)
        with out_path.mask_phrase_path.open("w") as f:
            for idx, phrase in instance_phrases_combined.items():
                f.write(f"{idx}: {phrase}\n")
