from dataclasses import dataclass
from typing import Any


@dataclass
class VectorRecord:
    id: str
    text: str
    metadata: dict[str, Any]
    embedding: list[float]