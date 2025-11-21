from teleai_pipe.utils.utils import load_state_dict_from_safetensors

original_lora = load_state_dict_from_safetensors(
    "/nvfile-heatstorage/model_zoo/huggingface/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v/loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"
)
rm_ids = [39, 38, 36, 31, 35, 30, 3, 33, 2, 29]


def rename_and_filter_blocks(state_dict, ids_to_remove):
    """
    删除指定 id 的 blocks，并将剩余的 blocks 按照新的顺序重命名。

    参数:
        state_dict (dict): 原始模型的权重字典。
        ids_to_remove (list): 需要删除的 block id 列表。

    返回:
        dict: 处理后的新权重字典，blocks 已按 0 到 m-1 重新编号。
    """
    # 获取所有 block id
    all_block_ids = set()
    prefix = "diffusion_model.blocks."
    for key in state_dict.keys():
        if key.startswith(prefix):
            # 提取 id 部分
            parts = key[len(prefix) :].split(".", 1)
            block_id = int(parts[0])
            all_block_ids.add(block_id)

    # 确定保留的 block id 并排序
    remaining_ids = sorted(list(all_block_ids - set(ids_to_remove)))

    # 构建新旧 id 映射关系：从 0 开始连续编号
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(remaining_ids)}

    # 创建新的 state_dict
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            parts = key[len(prefix) :].split(".", 1)
            old_block_id = int(parts[0])
            if old_block_id in id_mapping:
                new_block_id = id_mapping[old_block_id]
                # 替换旧 id 为新 id
                new_key = f"{prefix}{new_block_id}.{parts[1]}"
                new_state_dict[new_key] = value
        else:
            # 非 block 权重直接保留
            new_state_dict[key] = value

    return new_state_dict


new_state_dict = rename_and_filter_blocks(original_lora, rm_ids)
print(new_state_dict)
import torch
import safetensors.torch as st

st.save_file(
    new_state_dict,
    "Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64_30blocks.safetensors",
)
