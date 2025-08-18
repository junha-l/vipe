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

from pathlib import Path

import cv2
import torch
from natsort import natsorted

from vipe.streams.base import ProcessedVideoStream, StreamList, VideoFrame, VideoStream


class DirectoryFramesStream(VideoStream):
    """
    A video stream from extracted frames in a directory.
    Supports common image formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif
    """

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    def __init__(
        self, 
        path: Path, 
        fps: float = 30.0, 
        seek_range: range | None = None, 
        name: str | None = None,
    ) -> None:
        """
        Initialize DirectoryFramesStream.
        
        Args:
            path: Path to directory containing image frames
            fps: Frames per second (default: 30.0)
            seek_range: Range of frame indices to process
            name: Optional name for the stream
        """
        super().__init__()
        if seek_range is None:
            seek_range = range(-1)

        self.path = Path(path)
        self._name = name if name is not None else self.path.name
        self._fps = fps
        
        if not self.path.is_dir():
            raise ValueError(f"Path {path} is not a directory")

        # Get all supported image files
        self.frame_files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            self.frame_files.extend(self.path.glob(f"*{ext}"))
            self.frame_files.extend(self.path.glob(f"*{ext.upper()}"))

        if not self.frame_files:
            raise ValueError(f"No supported image files found in {path}")

        # Sort frames based on specified method
        self.frame_files = natsorted(self.frame_files)

        # Read first frame to get dimensions
        first_frame = cv2.imread(str(self.frame_files[0]))
        if first_frame is None:
            raise ValueError(f"Could not read first frame: {self.frame_files[0]}")
        
        self._height, self._width = first_frame.shape[:2]

        # Set up frame range
        _n_frames = len(self.frame_files)
        self.start = seek_range.start
        self.end = seek_range.stop if seek_range.stop != -1 else _n_frames
        self.end = min(self.end, _n_frames)
        self.step = seek_range.step

    def frame_size(self) -> tuple[int, int]:
        return (self._height, self._width)

    def fps(self) -> float:
        return self._fps

    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(range(self.start, self.end, self.step))

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> VideoFrame:
        if self.current_idx >= len(self):
            raise StopIteration

        # Calculate actual frame index based on range and step
        actual_frame_idx = self.start + self.current_idx * self.step
        
        if actual_frame_idx >= len(self.frame_files):
            raise StopIteration

        # Read the frame
        frame_path = self.frame_files[actual_frame_idx]
        frame = cv2.imread(str(frame_path))
        
        if frame is None:
            raise RuntimeError(f"Could not read frame: {frame_path}")

        # Convert BGR to RGB and normalize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = torch.as_tensor(frame).float() / 255.0
        frame_rgb = frame_rgb.cuda()

        self.current_idx += 1

        return VideoFrame(raw_frame_idx=actual_frame_idx, rgb=frame_rgb)


class DirectoryFramesStreamList(StreamList):
    """
    Stream list for handling multiple directories of extracted frames.
    """
    
    def __init__(
        self, 
        base_path: str, 
        fps: float = 30.0,
        frame_start: int = 0, 
        frame_end: int = -1, 
        frame_skip: int = 1, 
        cached: bool = False,
    ) -> None:
        super().__init__()
        
        base_path = Path(base_path)
        
        if base_path.is_dir():
            # Check if it's a single directory with frames
            has_images = any(
                base_path.glob(f"*{ext}") or base_path.glob(f"*{ext.upper()}")
                for ext in DirectoryFramesStream.SUPPORTED_EXTENSIONS
            )
            
            if has_images:
                # Single directory with frames
                self.frame_directories = [base_path]
            else:
                # Directory containing subdirectories with frames
                self.frame_directories = [
                    d for d in base_path.iterdir() 
                    if d.is_dir() and any(
                        d.glob(f"*{ext}") or d.glob(f"*{ext.upper()}")
                        for ext in DirectoryFramesStream.SUPPORTED_EXTENSIONS
                    )
                ]
                self.frame_directories.sort()
        else:
            raise ValueError(f"Path {base_path} does not exist or is not a directory")

        if not self.frame_directories:
            raise ValueError(f"No directories with supported image files found in {base_path}")

        self.fps = fps
        self.frame_range = range(frame_start, frame_end, frame_skip)
        self.cached = cached

    def __len__(self) -> int:
        return len(self.frame_directories)

    def __getitem__(self, index: int) -> VideoStream:
        stream: VideoStream = DirectoryFramesStream(
            self.frame_directories[index], 
            fps=self.fps,
            seek_range=self.frame_range,
        )
        if self.cached:
            stream = ProcessedVideoStream(stream, []).cache(desc="Loading frames", online=False)
        return stream

    def stream_name(self, index: int) -> str:
        return self.frame_directories[index].name