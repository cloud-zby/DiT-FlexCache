import os
from mpire import WorkerPool
import click
from loguru import logger
from datetime import datetime
from teleai_pipe.utils.cuda import filter_gpus_by_memory


SINGLE_EXAMPLE_LIST = [
    ("examples/teleai_video_v1/i2v_80G.py", 1),
]

PARALLEL_EXAMPLE_LIST = [
    (
        "examples/teleai_video_v1_2/wan_14b_fl2v_prompt_enchance_h100.py",
        2,
    ),
    (
        "examples/teleai_video_v1_2/wan_14b_image_to_video_distill_lora_h100.py",
        2,
    ),
    (
        "examples/teleai_video_v1_2/wan_14b_image_to_video_prompt_enhance_h100.py",
        2,
    ),
    (
        "examples/teleai_video_v1_2/wan_14b_text_to_video_prompt_enhance_h100.py",
        2,
    ),
    (
        "examples/teleai_video_v1_3/wan22_14b_image_to_video_prompt_enhance_distill_lora_h100.py",
        2,
    ),
    ("examples/teleai_video_v1_3/wan22_14b_image_to_video_prompt_enhance_h100.py", 2),
    (
        "examples/teleai_video_v1_3/wan22_14b_text_to_video_prompt_enhance_distill_lora_h100.py",
        2,
    ),
    ("examples/teleai_video_v1_3/wan22_14b_text_to_video_prompt_enhance_h100.py", 2),
    ("examples/teleai_video_v1_5/teleai_video_10b_image_to_hr_video_h100.py", 2),
    (
        "examples/teleai_video_v1_5/wan22_14b_text_to_hr_video_prompt_enhance_distill_lora_h100.py",
        2,
    ),
    ("examples/teleai_video_v1_5/teleai_video_1.3b_sr_h100.py", 2),
]


def run_example(resut_dir, example_file, device_ids):
    cmd = f"CUDA_VISIBLE_DEVICES={device_ids} TELEAI_EXAMPLE_OUTPUT_DIR={resut_dir} python {example_file}"
    logger.info(f"cmd is {cmd}")
    os.system(cmd)


@click.command()
@click.option(
    "--device_ids", "-d", type=str, default="auto", help="device id like: 0-7; 0; 0,1;"
)
@click.option(
    "--test_parallel", "-tp", is_flag=True, default=False, help="test parallel models"
)
@click.option(
    "--test_single", "-ts", is_flag=True, default=False, help="test single models"
)
def main(device_ids, test_single, test_parallel):
    if device_ids == "auto":
        device_ids = filter_gpus_by_memory(min_free_memory=60000)
        if not device_ids:
            logger.warning("no free gpu can be used")
            return
    elif "-" in device_ids:
        start_id, end_id = [int(id) for id in device_ids.split("-")]
        assert (
            start_id >= 0 and end_id < 8
        ), f"device id should be in 0-7, not {device_ids}"
        device_ids = list(range(start_id, end_id))
    else:
        device_ids = [int(id) for id in device_ids.split(",")]
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = f"work_dirs/example_result/{current_time}"
    os.makedirs(result_dir, exist_ok=True)
    if test_single:
        logger.info(
            f"run {len(SINGLE_EXAMPLE_LIST)} single examples on device {','.join([str(i) for i in device_ids])}"
        )
        tasks = []
        for i, example_info in enumerate(SINGLE_EXAMPLE_LIST):
            tasks.append((result_dir, example_info[0], device_ids[i % len(device_ids)]))
        with WorkerPool(n_jobs=len(device_ids), start_method="fork") as pool:
            pool.map(run_example, tasks, progress_bar=True)
    if test_parallel:
        logger.info(
            f"run {len(PARALLEL_EXAMPLE_LIST)} parallel examples on device {','.join([str(i) for i in device_ids])}"
        )
        for i, example_info in enumerate(PARALLEL_EXAMPLE_LIST):
            run_example(
                result_dir, example_info[0], ",".join([str(i) for i in device_ids])
            )


if __name__ == "__main__":
    main()
