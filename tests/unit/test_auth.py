from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from rag.auth import APIKeyAuthMiddleware


def _homepage(request):
    return PlainTextResponse("ok")


def test_api_key_middleware() -> None:
    app = Starlette(routes=[Route("/", _homepage)])
    app.add_middleware(APIKeyAuthMiddleware, api_key="secret")
    client = TestClient(app)

    assert client.get("/").status_code == 401
    resp = client.get("/", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200
    assert resp.text == "ok"

