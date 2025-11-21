from teleai_pipe.pipelines.rife_pipeline import RIFEVFIPipeline
from teleai_pipe.common.model_manager import ModelManager
import torch
import time
import os
from teleai_pipe.utils.video import VideoData, save_video
from teleai_pipe.utils.utils import get_example_video_name

PPL_CONFIG = dict(
    name="RIFE",
    model_path="/nvfile-heatstorage/model_zoo/modelscope/RIFEv4.26_0921/flownet.pkl",
    target_fps=30,
)


def get_pipeline():
    model_manager = ModelManager(device="cpu")
    # vae
    model_manager.load_model(
        PPL_CONFIG["model_path"], torch_dtype=torch.bfloat16, device="cuda"
    )
    pipeline = RIFEVFIPipeline.from_model_manager(
        model_manager, torch_dtype=torch.bfloat16, device="cuda"
    )
    return pipeline


def run(pipeline: RIFEVFIPipeline, input_video, base_fps, target_fps):
    result_video = pipeline(
        input_video=input_video, base_fps=base_fps, target_fps=target_fps
    )
    return result_video


if __name__ == "__main__":
    pipeline = get_pipeline()
    video_path = f"{os.path.dirname(__file__)}/../data/dag.mp4"
    src_video_data = VideoData(
        video_path,
    )
    start = time.time()
    target_fps = PPL_CONFIG["target_fps"]
    result_video = run(pipeline, src_video_data, 16, target_fps)
    print(f"interplace cost {time.time() - start} s")
    save_video(
        result_video,
        os.path.join(
            os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./"),
            get_example_video_name(__file__),
        ),
        quality=6,
        fps=target_fps,
    )
