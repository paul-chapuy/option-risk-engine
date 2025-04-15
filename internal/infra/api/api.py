from typing import Any, Optional, Callable, Dict, Any, Tuple, Protocol, Dict, Any
from dataclasses import dataclass

import requests


@dataclass
class Response:
    data: Any
    status: int


class AuthStrategy(Protocol):

    def apply_headers(self, headers: Dict[str, Any]) -> None: ...

    def apply_params(self, params: Dict[str, Any]) -> None: ...


class HeaderApiKeyAuth:

    def __init__(self, api_key: str, header_name: str = "Authorization") -> None:
        self.api_key = api_key
        self.header_name = header_name

    def apply_headers(self, headers: Dict[str, Any]) -> None:
        headers[self.header_name] = self.api_key

    def apply_params(self, params: Dict[str, Any]) -> None:
        pass


class QueryParamAuth:

    def __init__(self, api_key: str):
        self.api_key = api_key

    def apply_headers(self, headers: Dict[str, Any]) -> None:
        pass

    def apply_params(self, params: Dict[str, Any]) -> None:
        params["api_key"] = self.api_key
        params["file_type"] = "json"


@dataclass
class Context:

    auth: AuthStrategy
    make_url: Callable[[str, str], str]

    @staticmethod
    def make(auth: AuthStrategy) -> "Context":
        return Context(
            auth=auth,
            make_url=lambda base, endpoint: f"{base}/{endpoint}",
        )


class API:

    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx

    def request(
        self,
        method: str,
        url: Tuple[str, str],
        body: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Response:
        headers = {"User-Agent": "option-risk-engine"}

        self._ctx.auth.apply_headers(headers)
        self._ctx.auth.apply_params(params)

        response = requests.request(
            method,
            url=self._ctx.make_url(*url),
            headers=headers,
            params=params,
            json=body,
            timeout=120,
        )

        return Response(
            data=response.json() if response.content else {},
            status=response.status_code,
        )
