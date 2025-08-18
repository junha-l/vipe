import logging
from pathlib import Path

from natsort import natsorted

from vipe.streams.base import StreamList, VideoStream
from vipe.streams.directory_frames_stream import DirectoryFramesStream

logger = logging.getLogger(__name__)

class DL3DVStreamList(StreamList):
    """
    Stream list for DL3DV dataset.
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
        subsets = [f"{i}K" for i in range(1, 11)]

        frame_directories = []
        for subset in subsets:
            subset_dir = base_path / subset
            if not subset_dir.is_dir():
                logger.warning(f"Subset {subset} not found in {base_path}")
                continue
            scene_dirs = [d for d in subset_dir.iterdir() if d.is_dir()]
            frame_directories.extend(scene_dirs)
        
        self.frame_directories = natsorted(frame_directories)

        if not self.frame_directories:
            raise ValueError(f"No directories with supported image files found in {base_path}")

        self.fps = fps
        self.frame_range = range(frame_start, frame_end, frame_skip)
        self.cached = cached

    def __len__(self) -> int:
        return len(self.frame_directories)

    def __getitem__(self, index: int) -> VideoStream:
        stream: VideoStream = DirectoryFramesStream(
            self.frame_directories[index] / "images_8", 
            name=self.stream_name(index),
            fps=self.fps,
            seek_range=self.frame_range,
        )
        if self.cached:
            stream = ProcessedVideoStream(stream, []).cache(desc="Loading frames", online=False)
        return stream

    def stream_name(self, index: int) -> str:
        return "@".join(self.frame_directories[index].parts[-2:])
