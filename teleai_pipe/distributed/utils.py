import torch
import torch.distributed as dist


def sync_and_check_flag(flag: bool, world_size) -> bool:
    """
    同步所有进程的bool标志，并检查是否所有标志都为True

    参数:
        flag: 当前进程的标志

    返回:
        所有进程的标志是否都为True
    """
    flag_tensor = torch.tensor(int(flag), dtype=torch.int, device="cuda")
    dist.all_reduce(flag_tensor, op=dist.ReduceOp.SUM)
    return flag_tensor.item() == world_size
