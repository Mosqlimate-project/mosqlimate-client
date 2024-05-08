def url_pagination(params: dict) -> None:
    if "page" not in params or "per_page" not in params:
        raise ValueError(
            "'page' and 'per_page' parameters are required to requests"
            " with pagination"
        )
