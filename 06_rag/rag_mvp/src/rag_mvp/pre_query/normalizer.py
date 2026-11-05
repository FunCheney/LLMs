import unicodedata


class InvalidQueryError(ValueError):
    """用户查询不符合输入要求。"""


def normalize_query(query: str, * , max_length: int = 4000) -> str:
    if not isinstance(query, str):
        raise InvalidQueryError("query must be a string")

    normalized = unicodedata.normalize("NFKC", query)
    normalized = " ".join(normalized.split())

    if not normalized:
        raise InvalidQueryError("query must not be empty")

    if len(normalized) > max_length:
        raise InvalidQueryError("query must not be longer than {} characters".format(max_length))
    return normalized
