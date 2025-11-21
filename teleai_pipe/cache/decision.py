from __future__ import annotations

from typing import Iterable, List


def determine_skip_step(similarity_score: float, saved_steps: Iterable[int]) -> int:
    """根据相似度与可用缓存步数，计算应跳过到的起始步。

    规则与现有 FlexCache 行为保持一致：
    - sim>=0.90 且存在 step 25 → 跳至 20
    - sim>=0.85 且存在 step 20 → 跳至 15
    - sim>=0.80 且存在 step 15 → 跳至 10
    - sim>=0.75 且存在 step 10 → 跳至 5
    - 否则，若存在 step 5 → 跳至 5
    - 上述均不满足 → 0（不跳过）

    参数:
        similarity_score: 相似度 [0,1]
        saved_steps: 已缓存可用的关键步列表/集合

    返回:
        跳过的起始步（int），0 表示不跳过。
    """

    steps = set(int(s) for s in saved_steps)

    if similarity_score >= 0.90 and 0 in steps:
        return 1
    elif similarity_score >= 0.85 and 20 in steps:
        return 15
    elif similarity_score >= 0.80 and 15 in steps:
        return 10
    elif similarity_score >= 0.75 and 10 in steps:
        return 5
    elif 5 in steps:
        return 5
    else:
        return 0

