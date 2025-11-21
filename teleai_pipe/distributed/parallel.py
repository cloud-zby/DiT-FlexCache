import os
import copy
import math
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, CPUOffload
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel.style import ParallelStyle
from torch.distributed.tensor.parallel._utils import _validate_tp_mesh_dim
from contextlib import contextmanager
from datetime import timedelta
from functools import partial
from typing import Callable, Dict, List, Union, Optional
from queue import Empty
from typing import Optional
from loguru import logger


class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instance


class ProcessGroupSingleton(Singleton):
    def __init__(self):
        self.CFG_GROUP: Optional[dist.ProcessGroup] = None
        self.SP_GROUP: Optional[dist.ProcessGroup] = None
        self.TP_GROUP: Optional[dist.ProcessGroup] = None

        self.CFG_RANKS: List[int] = []
        self.SP_RANKS: List[int] = []
        self.TP_RANKS: List[int] = []


PROCESS_GROUP = ProcessGroupSingleton()


def get_cfg_group():
    return PROCESS_GROUP.CFG_GROUP


def get_cfg_world_size():
    return PROCESS_GROUP.CFG_GROUP.size()


def get_cfg_rank():
    return PROCESS_GROUP.CFG_GROUP.rank()


def get_cfg_ranks():
    return PROCESS_GROUP.CFG_RANKS


def get_sp_group():
    return PROCESS_GROUP.SP_GROUP


def get_sp_world_size():
    return PROCESS_GROUP.SP_GROUP.size()


def get_sp_rank():
    return PROCESS_GROUP.SP_GROUP.rank()


def get_sp_ranks():
    return PROCESS_GROUP.SP_RANKS


def get_tp_group():
    return PROCESS_GROUP.TP_GROUP


def get_tp_world_size():
    return PROCESS_GROUP.TP_GROUP.size()


def get_tp_rank():
    return PROCESS_GROUP.TP_GROUP.rank()


def get_tp_ranks():
    return PROCESS_GROUP.TP_RANKS


def init_parallel_pgs(
    cfg_degree: int = 1,
    sp_ulysses_degree: int = 1,
    sp_ring_degree: int = 1,
    tp_degree: int = 1,
    rank: int = 0,
    world_size: int = 1,
):
    try:
        from tele_fuser.yunchang.globals import set_seq_parallel_pg
    except ImportError:
        logger.warning("failed to import yuanchang, please install TeleFuser")

    sp_degree = sp_ulysses_degree * sp_ring_degree

    assert (
        sp_degree == 1 or tp_degree == 1
    ), "not allowed to enable sequence parallel and tensor parallel together"
    assert (
        world_size == cfg_degree * sp_degree * tp_degree
    ), f"world_size ({world_size}) must be equal to cfg_degree ({cfg_degree}) * sp_degree ({sp_degree}) * tp_degree ({tp_degree})"
    logger.info(
        f"init parallel gps with cfg {cfg_degree}, sp_ulysses {sp_ulysses_degree}, sp_ring {sp_ring_degree}, tp {tp_degree}"
    )

    def make_parallel_groups(blocks: List[List[int]], degree: int):
        groups, chunks = [], []
        for block in blocks:
            size = len(block) // degree
            chunk = [block[i * size : (i + 1) * size] for i in range(degree)]
            chunks.extend(chunk)
            groups.extend(list(zip(*chunk)))
        return groups, chunks

    blocks = [list(range(world_size))]
    cfg_groups, cfg_blocks = make_parallel_groups(blocks, cfg_degree)
    for cfg_ranks in cfg_groups:
        cfg_group = dist.new_group(cfg_ranks)
        if rank in cfg_ranks:
            PROCESS_GROUP.CFG_GROUP = cfg_group
            PROCESS_GROUP.CFG_RANKS = cfg_ranks

    sp_groups, sp_blocks = make_parallel_groups(cfg_blocks, sp_degree)
    for sp_ranks in sp_groups:
        group = dist.new_group(sp_ranks)
        if rank in sp_ranks:
            PROCESS_GROUP.SP_GROUP = group
            PROCESS_GROUP.SP_RANKS = sp_ranks

    tp_groups, _ = make_parallel_groups(sp_blocks, tp_degree)
    for tp_ranks in tp_groups:
        group = dist.new_group(tp_ranks)
        if rank in tp_ranks:
            PROCESS_GROUP.TP_GROUP = group
            PROCESS_GROUP.TP_RANKS = tp_ranks
    if sp_ulysses_degree > 1 or sp_ring_degree > 1:
        set_seq_parallel_pg(sp_ulysses_degree, sp_ring_degree, rank, world_size)
        
    print("Init parallel pgs finished.")


def clone(data):
    if isinstance(data, dict):
        return {k: clone(v) for k, v in data.items()}
    if isinstance(data, tuple) or isinstance(data, list):
        return [clone(t) for t in data]
    elif isinstance(data, torch.Tensor):
        return data.clone()
    else:
        return copy.deepcopy(data)


def to_device(data, device):
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    if isinstance(data, tuple) or isinstance(data, list):
        return [to_device(t, device) for t in data]
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


def split_and_get(data, num, dim, index):
    if isinstance(data, dict):
        return {k: split_and_get(v, num, dim, index) for k, v in data.items()}
    if isinstance(data, tuple) or isinstance(data, list):
        return [split_and_get(t, num, dim, index) for t in data]
    if isinstance(data, torch.Tensor):
        if data.shape[dim] < num:
            raise ValueError(
                f"data.shape[{dim}] ({data.shape[dim]}) < num ({num}), split failed"
            )
        return torch.split(data, data.shape[dim] // num, dim)[index]
    return data


def shard_model(
    module: nn.Module,
    device_id: int,
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
    wrap_module_names: Optional[List[str]] = None,
):
    wrap_module_names = wrap_module_names or []

    def wrap_fn(m):
        for name in wrap_module_names:
            submodule = getattr(module, name)
            if isinstance(submodule, nn.ModuleList) and m in submodule:
                return True
            elif not isinstance(submodule, nn.ModuleList) and m is submodule:
                return True
        return False

    return FSDP(
        module,
        device_id=device_id,
        sharding_strategy=sharding_strategy,
        # use_orig_params=True,
        auto_wrap_policy=partial(lambda_auto_wrap_policy, lambda_fn=wrap_fn),
    )


def shard_modelv2(
    module,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=False,
    use_lora=False,
    cpu_offload=True,
    wrap_module_names: Optional[List[str]] = None,
):
    wrap_module_names = wrap_module_names or []

    def wrap_fn(m):
        for name in wrap_module_names:
            submodule = getattr(module, name)
            if isinstance(submodule, nn.ModuleList) and m in submodule:
                return True
            elif not isinstance(submodule, nn.ModuleList) and m is submodule:
                return True
        return False

    model = module
    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(lambda_auto_wrap_policy, lambda_fn=wrap_fn),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        ),
        device_id=device_id,
        sync_module_states=sync_module_states,
        use_orig_params=True if use_lora else False,
        cpu_offload=CPUOffload(offload_params=True) if cpu_offload else None,
    )
    return model


def parallelize_module(
    module: nn.Module,
    device_mesh: DeviceMesh,
    parallelize_plan: Optional[Union[ParallelStyle, Dict[str, ParallelStyle]]] = None,
):
    _validate_tp_mesh_dim(device_mesh)
    if parallelize_plan is None:
        return module
    if isinstance(parallelize_plan, ParallelStyle):
        return parallelize_plan._apply(module, device_mesh)
    for module_path, parallelize_style in parallelize_plan.items():
        if module_path.strip() == "":
            raise ValueError(
                "Expect module path to be non-empty, but got empty string!"
            )
        try:
            submodule = module.get_submodule(module_path)
            parallelize_style._apply(submodule, device_mesh)
        except AttributeError:
            continue
    return module


NCCL_TIMEOUT_SEC = int(os.environ.get("NCCL_TIMEOUT_SEC", 600))
PARALLEL_FWD_TIMEOUT_SEC = int(os.environ.get("PARALLEL_FWD_TIMEOUT_SEC", 3000))


def _worker_loop(
    rank: int,
    world_size: int,
    queue_in: List[mp.Queue],
    queue_out: mp.Queue,
    module: nn.Module,
    cfg_degree: int,
    sp_ulysses_degree: int,
    sp_ring_degree: int,
    tp_degree: int,
    shard_fn: Optional[Callable] = None,
    master_port: int = 29500,
    device_type: str = "cuda",
    compile: bool = False,
    model_name="",
    device_ids=None,
):
    """
    https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
    """
    try:
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        device_id = rank
        if device_ids is not None:
            device_id = device_ids[rank]
        device = torch.device(type=device_type, index=device_id)
        torch.cuda.set_device(device)

        timeout = timedelta(seconds=NCCL_TIMEOUT_SEC)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timeout,
            world_size=world_size,
            rank=rank,
            device_id=device,
        )
        init_parallel_pgs(
            cfg_degree=cfg_degree,
            sp_ulysses_degree=sp_ulysses_degree,
            sp_ring_degree=sp_ring_degree,
            tp_degree=tp_degree,
            rank=rank,
            world_size=world_size,
        )

        if tp_degree > 1:
            module = parallelize_module(
                module=module,
                device_mesh=DeviceMesh(device_type, torch.tensor(get_tp_ranks())),
                parallelize_plan=module.get_tp_plan(),
            ).to(device_type)
        elif shard_fn:
            module = shard_fn(module=module, device_id=device_id)
        if compile:
            module.compile()

        while True:
            data = queue_in[rank].get()
            name, args, kwargs = data
            del data
            if name == "cpu":
                module.cpu()
                torch.cuda.empty_cache()
                if rank == 0:
                    queue_out.put("offload to cpu success")
                continue
            if name == "exit":
                logger.info(f"parallel model {model_name} on rank {rank} exits")
                break
            if not hasattr(module, name):
                raise AttributeError(
                    f'{module.__class__.__name__} has no attribute "{name}"'
                )
            local_args = split_and_get(args, get_cfg_world_size(), 0, get_cfg_rank())
            local_kwargs = split_and_get(
                kwargs, get_cfg_world_size(), 0, get_cfg_rank()
            )
            del args, kwargs
            local_kwargs = to_device(local_kwargs, device)
            local_args = to_device(local_args, device)
            with torch.no_grad():
                y = getattr(module, name)(*local_args, **local_kwargs)
            del local_args, local_kwargs
            torch.cuda.empty_cache()
            is_rank_zero_in_cfg_group = (
                get_cfg_world_size() > 1 and get_sp_rank() == 0 and get_tp_rank() == 0
            )
            if y is not None and is_rank_zero_in_cfg_group:
                gathered = torch.zeros(
                    (get_cfg_world_size(), *y.shape[1:]), dtype=y.dtype, device=y.device
                )
                dist.all_gather_into_tensor(gathered, y, group=get_cfg_group())
                y = gathered

            if rank == 0:
                queue_out.put(y)
            dist.barrier()
    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error(f"Error in worker loop (rank {rank}): {e}")
        queue_out.put(
            e
        )  # any exception caught in the worker will be raised to the main process
    finally:
        del module, args, kwargs
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        dist.destroy_process_group()


class ParallelModel(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        cfg_degree: int,
        sp_ulysses_degree: int,
        sp_ring_degree: int,
        tp_degree: int,
        shard_fn: Optional[Callable] = None,
        master_port: int = 29500,
        device: str = "cuda",
        compile=False,
        name="",
        device_ids: Optional[List] = None,
    ):
        current_method = mp.get_start_method(allow_none=True)
        if current_method is None or current_method != "spawn":
            try:
                mp.set_start_method("spawn")
            except RuntimeError as e:
                raise RuntimeError("Failed to set start method to spawn:", e)

        super().__init__()
        self.world_size = cfg_degree * sp_ulysses_degree * sp_ring_degree * tp_degree
        if device_ids is not None:
            assert (
                len(device_ids) == self.world_size
            ), f"{device_ids} is not equal world size {self.world_size}"
        spawn_ctx = mp.get_context("spawn")
        self.queue_in = [spawn_ctx.Queue() for _ in range(self.world_size)]
        self.queue_out = spawn_ctx.Queue()
        self.name = name
        self.original_model = module
        self._original_attrs = dict()
        try:
            for attr_name in dir(module):
                if not attr_name.startswith("_"):  # 跳过私有属性
                    try:
                        attr_value = getattr(module, attr_name)
                        self._original_attrs[attr_name] = callable(attr_value)
                    except Exception:
                        continue
        except Exception as e:
            logger.error(f"get attr error {e}")
        self.ctx = mp.spawn(
            _worker_loop,
            args=(
                self.world_size,
                self.queue_in,
                self.queue_out,
                module,
                cfg_degree,
                sp_ulysses_degree,
                sp_ring_degree,
                tp_degree,
                shard_fn,
                master_port,
                device,
                compile,
                self.name,
                device_ids,
            ),
            nprocs=self.world_size,
            join=False,
        )
        print(f"after start ctx module type is {type(module)}")

    def put_data(self, data):
        data = to_device(data, "cpu")
        for q in self.queue_in:
            q.put(data)

    def forward(self, *args, **kwargs):
        data = ["forward", args, kwargs]
        self.put_data(data)
        try:
            res = self.queue_out.get(timeout=PARALLEL_FWD_TIMEOUT_SEC)
            if isinstance(res, Exception):
                raise res
        except Empty:
            logger.error("ParallelModel forward timeout")
            raise RuntimeError("ParallelModel forward timeout")
        except Exception as e:
            logger.error(f"ParallelModel forward error: {e}")
            raise RuntimeError(f"ParallelModel forward error: {e}")
        return res

    def __getattr__(self, name):
        if name in self._original_attrs:
            if not self._original_attrs[name]:
                attr = getattr(self.original_model, name)
                return attr
            else:

                def wrapped_func(*args, **kwargs):
                    data = [name, args, kwargs]
                    self.put_data(data)
                    try:
                        res = self.queue_out.get(timeout=PARALLEL_FWD_TIMEOUT_SEC)
                        if isinstance(res, Exception):
                            raise res
                    except Empty:
                        logger.error(f"ParallelModel:{self.name} {name} timeout")
                        raise RuntimeError(f"ParallelModel:{self.name} {name} timeout")
                    except Exception as e:
                        logger.error(f"ParallelModel:{self.name} {name} error: {e}")
                        raise RuntimeError(
                            f"ParallelModel:{self.name} {name} error: {e}"
                        )
                    logger.info(f"ParallelModel:{self.name} {name} done")
                    return res

                return wrapped_func
        else:
            return super().__getattr__(name)

    def __del__(self):
        if hasattr(self, "ctx"):
            self.put_data(["exit", None, None])
            for p in self.ctx.processes:
                p.join(timeout=10)
                if p.is_alive():
                    p.kill()
            for q in self.queue_in:
                q.close()
            self.queue_out.close()


_sequence_parallel_enabled = False


@contextmanager
def sequence_parallel(
    tensors: Optional[List[torch.Tensor]] = None,
    seq_dims: Optional[List[int]] = None,
    enabled: bool = False,
):
    if not enabled:
        yield
        return

    tensors = [] if tensors is None else tensors
    seq_dims = [] if seq_dims is None else seq_dims
    assert len(tensors) == len(
        seq_dims
    ), "tensors and seq_dims must have the same number of elements"

    for tensor, seq_dim in zip(tensors, seq_dims):
        # pad seq_len to multiple of sp_world_size
        # TODO: long_context_attention does not support attn_mask, may cause loss of numerical precision with padding
        seq_len = tensor.size(seq_dim)
        pad_len = (
            math.ceil(seq_len / get_sp_world_size()) * get_sp_world_size() - seq_len
        )
        padding = [0] * (2 * tensor.ndim)
        padding[-2 * seq_dim - 1] = pad_len
        padded_tensor = F.pad(tensor, padding)

        chunks = torch.chunk(padded_tensor, get_sp_world_size(), dim=seq_dim)
        chunk = chunks[get_sp_rank()]
        tensor.resize_(chunk.shape)
        tensor.copy_(chunk)

    global _sequence_parallel_enabled
    _sequence_parallel_enabled = True
    yield
    _sequence_parallel_enabled = False


def sequence_parallel_unshard(
    tensors: List[torch.Tensor],
    seq_dims: List[int],
    seq_lens: List[int],
) -> List[torch.Tensor]:
    if not _sequence_parallel_enabled:
        return tensors

    assert len(tensors) == len(
        seq_dims
    ), "tensors and seq_dims must have the same number of elements"
    assert len(tensors) == len(
        seq_lens
    ), "tensors and seq_lens must have the same number of elements"
    unshard_tensors = []
    for tensor, seq_dim, seq_len in zip(tensors, seq_dims, seq_lens):
        unshard = [torch.zeros_like(tensor) for _ in range(get_sp_world_size())]
        dist.all_gather(unshard, tensor, group=get_sp_group())
        unshard = torch.cat(unshard, dim=seq_dim).narrow(
            dim=seq_dim, start=0, length=seq_len
        )
        unshard_tensors.append(unshard)
    return unshard_tensors


__all__ = ["ParallelModel", "sequence_parallel", "sequence_parallel_unshard"]
