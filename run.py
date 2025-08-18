import os
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(args: DictConfig) -> None:
    from vipe.streams.base import StreamList

    # Gather all video streams
    stream_list = StreamList.make(args.streams)

    from vipe.pipeline import make_pipeline
    from vipe.utils.logging import configure_logging

    # Process each video stream with distributed processing support
    logger = configure_logging()
    
    # Slice stream list for distributed processing
    world_size = args.get("world_size", 1)
    local_rank = args.get("local_rank", 0)

    slurm_id = os.environ["SLURM_JOB_ID"]
    log_dir = f"logs/{slurm_id}"
    os.makedirs(log_dir, exist_ok=True)
    status_file_path = os.path.join(log_dir, "status.txt")
    if not os.path.exists(status_file_path):
        with open(status_file_path, "w") as f:
            f.write("RUNNING")
    
    if world_size > 1:
        # Distribute streams across workers
        total_streams = len(stream_list)
        streams_per_worker = (total_streams + world_size - 1) // world_size  # Ceiling division
        start_idx = local_rank * streams_per_worker
        end_idx = min(start_idx + streams_per_worker, total_streams)
        
        if start_idx >= total_streams:
            logger.info(f"Worker {local_rank}/{world_size} has no streams to process")
            return
            
        logger.info(f"Worker {local_rank}/{world_size} processing streams {start_idx}-{end_idx-1} of {total_streams}")
        stream_indices = list(range(start_idx, end_idx))
    else:
        # Single worker processes all streams
        stream_indices = list(range(len(stream_list)))
    
    for i, stream_idx in enumerate(stream_indices):
        try:
            video_stream = stream_list[stream_idx]
        except Exception as e:
            print(e)
            continue
        logger.info(
            f"Worker {local_rank}: Processing {video_stream.name()} ({i + 1} / {len(stream_indices)}) [Global: {stream_idx + 1} / {len(stream_list)}]"
        )
        pipeline = make_pipeline(args.pipeline)
        pipeline.run(video_stream)
        logger.info(f"Worker {local_rank}: Finished processing {video_stream.name()}")

    with open(status_file_path,"w") as f:
        f.write("FINISHED")

if __name__ == "__main__":
    run()
