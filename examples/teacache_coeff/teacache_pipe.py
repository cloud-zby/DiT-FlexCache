from teleai_pipe.utils.video import save_video
from teleai_pipe.utils.cuda import filter_gpus_by_memory
from teleai_pipe.utils.utils import import_function_from_file, split_list
from mpire import WorkerPool
from PIL import Image
import json
import os
import click
import itertools
from loguru import logger
from datetime import datetime
from tqdm import tqdm
import time


def run_i2v_pipeline(
    pipe_file, tasks, device_ids, result_dir, prompt_type, output_dir="work_dirs"
):
    device_ids = ",".join([str(i) for i in device_ids])
    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
    get_pipeline = import_function_from_file(pipe_file, "get_pipeline")
    run = import_function_from_file(pipe_file, "run")
    pipe = get_pipeline(output_dir=output_dir)
    for task_idx, task in enumerate(tqdm(tasks)):
        condition = task["condition"]
        image_path = condition["first_image"]
        video_name = (
            f"{os.path.splitext(os.path.basename(condition['first_image']))[0]}.mp4"
        )
        video_path = os.path.abspath(os.path.join(result_dir, video_name))
        logger.info(f"save video to {video_path}")
        if os.path.exists(video_path):
            logger.info(f"find existing result {video_path}")
            continue
        image = Image.open(image_path).convert("RGB")
        prompt = condition[f"prompt_{prompt_type}"]
        assert prompt, "prompt can not be empty"
        start = time.time()
        video = run(
            pipe,
            image,
            prompt,
            condition["height"],
            condition["width"],
            prompt_id=task_idx,
            output_dir=output_dir,
        )
        task["generate_clips"] = [video_path]
        task["metric_result"] = {"time_cost": time.time() - start}
        save_video(
            video,
            video_path,
            fps=16,
            quality=6,
        )
    return tasks


def run_fl2v_pipeline(
    pipe_file, tasks, device_ids, result_dir, prompt_type, output_dir="work_dirs"
):
    device_ids = ",".join([str(i) for i in device_ids])
    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
    get_pipeline = import_function_from_file(pipe_file, "get_pipeline")
    run = import_function_from_file(pipe_file, "run")
    pipe = get_pipeline(output_dir=output_dir)
    for task_idx, task in enumerate(tqdm(tasks)):
        condition = task["condition"]
        video_name = (
            f"{os.path.splitext(os.path.basename(condition['first_image']))[0]}.mp4"
        )
        video_path = os.path.abspath(os.path.join(result_dir, video_name))
        if os.path.exists(video_path):
            logger.info(f"find existing result {video_path}")
            continue
        logger.info(f"save video to {video_path}")
        first_image_path = condition["first_image"]
        first_image = Image.open(first_image_path).convert("RGB")
        last_image_path = condition["last_image"]
        last_image = Image.open(last_image_path).convert("RGB")
        prompt = condition[f"prompt_{prompt_type}"]
        start = time.time()
        video = run(
            pipe,
            first_image,
            last_image,
            prompt,
            condition["height"],
            condition["width"],
            prompt_id=task_idx,
            output_dir=output_dir,
        )
        task["generate_clips"] = [video_path]
        task["metric_result"] = {"time_cost": time.time() - start}
        save_video(
            video,
            video_path,
            fps=16,
            quality=6,
        )
    return tasks


def run_t2v_pipeline(
    pipe_file, tasks, device_ids, result_dir, prompt_type, output_dir="work_dirs"
):
    device_ids = ",".join([str(i) for i in device_ids])
    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
    get_pipeline = import_function_from_file(pipe_file, "get_pipeline")
    run = import_function_from_file(pipe_file, "run")
    pipe = get_pipeline(output_dir=output_dir)
    for task_idx, task in enumerate(tqdm(tasks)):
        condition = task["condition"]
        prompt = condition[f"prompt_{prompt_type}"]
        video_path = os.path.abspath(
            os.path.join(result_dir, f"{prompt.split('.')[0][:20]}.mp4")
        )
        if os.path.exists(video_path):
            logger.info(f"find existing result {video_path}")
            continue
        assert prompt, "prompt can not be empty"
        start = time.time()
        video = run(
            pipe,
            prompt,
            condition["height"],
            condition["width"],
            prompt_id=task_idx,
            output_dir=output_dir,
        )
        task["generate_clips"] = [video_path]
        task["metric_result"] = {"time_cost": time.time() - start}
        save_video(
            video,
            video_path,
            fps=16,
            quality=6,
        )
    return tasks


RUNNER_MAP = {
    "i2v": run_i2v_pipeline,
    "fl2v": run_fl2v_pipeline,
    "t2v": run_t2v_pipeline,
}


@click.command()
@click.option(
    "--device_ids", "-d", type=str, default="auto", help="device id like: 0-7; 0; 0,1;"
)
@click.option(
    "--pipeline_file",
    "-pf",
    type=str,
    default="",
    help="python file which has get_pipeline func",
)
@click.option("--benchmark_path", "-b", type=str, default="", help="benchmark dir path")
@click.option(
    "--test_parallel", "-tp", is_flag=True, default=False, help="test parallel models"
)
@click.option(
    "--task", "-t", type=str, default="i2v", help="tasks including t2v, i2v and fl2v"
)
@click.option(
    "--output-dir", "-o", type=str, default="work_dirs", help="output directory"
)
@click.option(
    "--prompt_type",
    "-pt",
    type=str,
    default="en",
    help="en, zh, en_extend or zh_extend",
)
@click.option(
    "--json_name",
    "-jn",
    type=str,
    default="result.json",
    help="name of the result  json ",
)
@click.option(
    "--resume", "-r", type=str, default="", help="result directory path for resuming"
)
def main(
    device_ids,
    pipeline_file,
    test_parallel,
    benchmark_path,
    task,
    output_dir,
    prompt_type,
    resume,
    json_name,
):
    assert prompt_type in [
        "en",
        "zh",
        "en_extend",
        "zh_extend",
    ], f"prompt type {prompt_type} is not supported"
    benchmark_path = os.path.abspath(benchmark_path)
    logger.info(f"start test {pipeline_file} on {benchmark_path}")
    pipeline_cfg = import_function_from_file(pipeline_file, "PPL_CONFIG")
    pipeline_cfg["pipeline_path"] = os.path.abspath(pipeline_file)
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
        device_ids = list(range(start_id, end_id + 1))
    else:
        device_ids = [int(id) for id in device_ids.split(",")]
    logger.info(f"test {task} on {','.join([str(i) for i in device_ids])}")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pipeline_name = os.path.splitext(os.path.basename(pipeline_file))[0]
    benchmark_name = os.path.splitext(os.path.basename(benchmark_path))[0]
    output_dir = os.path.abspath(output_dir)
    result_dir = f"{output_dir}/{task}/{pipeline_name}/{benchmark_name}_{prompt_type}/{current_time}"
    if resume:
        assert os.path.exists(resume), f"resume dir {resume} is not exist"
        result_dir = resume
    os.makedirs(result_dir, exist_ok=True)
    with open(benchmark_path) as f:
        bmk = json.load(f)
    tasks = bmk["items"]
    run_pipeline = RUNNER_MAP[task]
    device_num = len(device_ids)
    if device_num == 1:
        tasks = run_pipeline(pipeline_file, tasks, device_ids, result_dir, prompt_type)
    else:
        worker_num = min(device_num, len(tasks))
        tasks_list = split_list(tasks, worker_num)
        param_list = [
            (pipeline_file, sub_tasks, [device_id], result_dir, prompt_type)
            for sub_tasks, device_id in zip(tasks_list, device_ids)
        ]
        with WorkerPool(n_jobs=worker_num, start_method="spawn") as pool:
            tasks = pool.map(run_pipeline, param_list, progress_bar=True)
            tasks = list(itertools.chain.from_iterable(tasks))
    bmk["items"] = tasks
    bmk["algo_info"] = {
        "pipeline_path": pipeline_file,
        "prompt_type": prompt_type,
        "task": task,
        "name": pipeline_cfg["name"],
        "meta": f"{json.dumps(pipeline_cfg, indent=4)}",
    }
    json_result_path = os.path.join(result_dir, json_name)
    logger.info(f"save video result to {json_result_path}")
    with open(json_result_path, "w") as f:
        json.dump(bmk, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
