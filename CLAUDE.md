# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ViPE (Video Pose Engine) is NVIDIA's spatial AI tool for annotating camera poses and dense depth maps from raw videos. It processes unconstrained videos to estimate camera intrinsics, motion, and near-metric depth maps, supporting various camera models including pinhole, wide-angle, and 360° panoramas.

## Architecture

ViPE follows a modular pipeline architecture:

- **Pipeline System**: `vipe/pipeline/` contains the core processing pipeline (`DefaultAnnotationPipeline`) that orchestrates video processing through configurable stages
- **SLAM System**: `vipe/slam/` implements visual-inertial SLAM with bundle adjustment, factor graphs, and motion filtering
- **Priors**: `vipe/priors/` contains depth estimation models (UniDepth, Metric3D, VideoDepthAnything, etc.), geometric calibration, and object tracking
- **Streams**: `vipe/streams/` handles video input/output processing and frame management
- **Extensions**: `vipe/ext/` provides optimized CUDA kernels for correlation, Lie group operations, and other performance-critical operations

The system uses Hydra for configuration management with YAML configs in `configs/`.

## Installation & Build Commands

```bash
# Create conda environment and install dependencies
conda env create -f envs/base.yml
conda activate vipe
pip install -r envs/requirements.txt

# Build and install the package (with CUDA extensions)
pip install --no-build-isolation -e .
```

The build process compiles CUDA extensions defined in `csrc/` including correlation samplers, Lie algebra operations, and SLAM kernels.

## Running ViPE

### CLI Interface
```bash
# Basic inference on a video
vipe infer YOUR_VIDEO.mp4

# With custom output directory and visualization
vipe infer YOUR_VIDEO.mp4 --output custom_results --visualize

# Use different pipeline (no_vda for less GPU memory usage)
vipe infer YOUR_VIDEO.mp4 --pipeline no_vda

# Visualize results
vipe visualize vipe_results/
```

### Run Script (Advanced)
```bash
# Full pipeline with hydra config overrides
python run.py pipeline=default streams=raw_mp4_stream streams.base_path=YOUR_VIDEO_OR_DIR_PATH

# Pose-only pipeline without depth estimation
python run.py pipeline=default streams=raw_mp4_stream streams.base_path=YOUR_VIDEO_OR_DIR_PATH pipeline.post.depth_align_model=null
```

## Configuration System

- Main config: `configs/default.yaml`
- Pipeline configs: `configs/pipeline/` (default, no_vda, static_vda)
- SLAM configs: `configs/slam/default.yaml`
- Stream configs: `configs/streams/`

Pipeline configurations control:
- Camera type and intrinsics estimation method
- SLAM settings and depth models
- Post-processing depth alignment
- Output paths and visualization options

## Code Style & Tools

The project uses:
- **Ruff** for linting and formatting (configured in `pyproject.toml`)
- **Black-compatible** formatting with 120 character line length
- **Type hints** throughout the codebase
- **SPDX license headers** on all source files

To run formatting:
```bash
ruff format .
ruff check .
```

## Key Extension Points

- **Depth Models**: Add new depth estimation models in `vipe/priors/depth/`
- **Pipeline Processors**: Extend `vipe/pipeline/processors.py` for new processing stages
- **Video Streams**: Add new input formats by extending `vipe/streams/base.py`
- **SLAM Components**: Modify SLAM behavior in `vipe/slam/system.py`

## Output Structure

Results are saved to `vipe_results/` (configurable) containing:
- `rgb/`: Input video frames
- `depth/`: Depth maps (.exr format)
- `pose/`: Camera poses (.npz)
- `intrinsics/`: Camera parameters
- `vipe/`: Processing metadata and visualization videos