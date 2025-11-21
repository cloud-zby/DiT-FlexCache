from __future__ import annotations

from loguru import logger as _logger
from typing import Any


class _CacheLogger:
    """缓存专用日志器（带前缀 + 内部容错）。

    - 所有方法都在内部捕获异常，确保日志系统异常不影响主流程；
    - 支持惰性格式化：透传 *args/**kwargs 到 loguru，避免在调用处进行 f-string 预格式化；
    - 兼容旧调用：`msg` 可为任意对象，将在内部转成字符串。
    """

    def __init__(self, prefix: str = "[CACHE]") -> None:
        self._prefix = prefix

    def _fmt(self, msg: Any) -> str:
        try:
            return f"{self._prefix} {msg}"
        except Exception:
            # 最坏情况下也不要抛出
            try:
                return f"{self._prefix} <unprintable message>"
            except Exception:
                return "[CACHE] <log error>"

    def _safe_call(self, level: str, msg: Any, *args, **kwargs) -> None:
        try:
            method = getattr(_logger, level, _logger.info)
            # 透传 args/kwargs 以利用 loguru 的惰性格式化
            if args or kwargs:
                method(self._fmt(msg), *args, **kwargs)
            else:
                method(self._fmt(msg))
        except Exception:
            # 降级到最轻量的 debug，且吞掉所有异常
            try:
                _logger.debug("[CACHE] <logging failed>")
            except Exception:
                pass

    def debug(self, msg: Any, *args, **kwargs) -> None:
        self._safe_call("debug", msg, *args, **kwargs)

    def info(self, msg: Any, *args, **kwargs) -> None:
        self._safe_call("info", msg, *args, **kwargs)

    def warning(self, msg: Any, *args, **kwargs) -> None:
        self._safe_call("warning", msg, *args, **kwargs)

    def error(self, msg: Any, *args, **kwargs) -> None:
        self._safe_call("error", msg, *args, **kwargs)


cache_logger = _CacheLogger()

__all__ = ["cache_logger"]
