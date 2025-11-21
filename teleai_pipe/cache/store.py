from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence


@dataclass
class VectorPoint:
    id: str
    vector: Sequence[float]
    payload: dict


class VectorStore:

    def upsert(self, collection: str, points: List[VectorPoint]) -> None:   
        pass

    def search(self, collection: str, query_vector: Sequence[float], limit: int = 1):   
        return []

    def delete(self, collection: str, ids: Iterable[str]) -> None:   
        pass

    def get_vector_size(self, collection: str) -> Optional[int]:   
        return None


class KVStore:
    """键值存储最小接口。"""

    def get(self, key: str) -> Optional[bytes]:   
        return None

    def put(self, key: str, value: bytes) -> None:   
        pass

    def remove(self, key: str) -> None:   
        pass

    def get_size(self, key: str) -> int:   
        return 0

    def is_exist(self, key: str) -> bool:   
        return False


class InMemoryKVStore(KVStore):

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    def get(self, key: str) -> Optional[bytes]:
        return self._data.get(key)

    def put(self, key: str, value: bytes) -> None:
        self._data[key] = value

    def remove(self, key: str) -> None:
        self._data.pop(key, None)

    def get_size(self, key: str) -> int:
        v = self._data.get(key)
        return len(v) if v is not None else 0

    def is_exist(self, key: str) -> bool:
        return key in self._data
