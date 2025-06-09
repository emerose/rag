"""Authentication utilities for the RAG MCP server."""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class APIKeyAuthMiddleware:
    """Simple middleware enforcing a static API key.

    The middleware checks the ``Authorization`` header for a value of the form
    ``"Bearer <key>"``. Requests without the correct key receive ``401``.
    """

    def __init__(self, app: ASGIApp, api_key: str) -> None:
        self.app = app
        self.api_key = api_key

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Only apply auth to HTTP requests, pass through other scope types (lifespan, websocket)
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
            
        request = Request(scope, receive=receive)
        auth_header = request.headers.get("authorization")
        if auth_header != f"Bearer {self.api_key}":
            response = JSONResponse({"detail": "Unauthorized"}, status_code=401)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
