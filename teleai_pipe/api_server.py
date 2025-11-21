import click
from teleai_pipe.server.main import run_server


@click.command()
@click.option(
    "--pipe_path", "-pp", required=True, type=str, help="Path to pipeline file"
)
@click.option("--task", "-t", default="i2v", type=str, help="Task type (i2v, etc.)")
@click.option("--port", "-p", default=8000, type=int, help="Server port")
@click.option("--host", default="127.0.0.1", type=str, help="Server host")
@click.option(
    "--cache-dir",
    "-c",
    default="work_dirs/server_cache",
    type=str,
    help="cache dir for input and output",
)
@click.option("--gpu-num", "-g", type=int, help="gpu number for parallel infer")
def main(pipe_path, task, port, host, cache_dir, gpu_num):
    run_server(pipe_path, task, port, host, cache_dir, gpu_num)


if __name__ == "__main__":
    main()
