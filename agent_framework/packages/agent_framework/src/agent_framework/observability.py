from _contextvars import ContextVar
from contextlib import contextmanager
from typing import Iterator
from uuid import uuid4

request_id_var: ContextVar[str] = ContextVar("request_id", default="")

def new_request_id() -> str:

    return  "req_" + uuid4().hex[:12]

def current_request_id() -> str:
    """
    读不到就返回 "-"? 不。见下
    """

    return request_id_var.get("request_id") is None or "-"

@contextmanager
def request_scope(request_id: str | None = None) -> Iterator[str]:
    """设置并复位 token；异常也要复位，否则线程池复用会串号。"""