import pytest
from unittest.mock import AsyncMock, MagicMock

from rag.auth import APIKeyAuthMiddleware


async def test_api_key_middleware() -> None:
    """Test APIKeyAuthMiddleware without TestClient to avoid CI hanging."""
    # Mock ASGI app
    mock_app = AsyncMock()
    middleware = APIKeyAuthMiddleware(mock_app, api_key="secret")
    
    # Mock scope, receive, send for HTTP request without auth
    scope_no_auth = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": []
    }
    receive_mock = AsyncMock()
    send_mock = AsyncMock()
    
    # Test unauthorized request
    await middleware(scope_no_auth, receive_mock, send_mock)
    
    # Should have sent a 401 response, not called the app
    mock_app.assert_not_called()
    send_mock.assert_called()
    
    # Reset mocks
    mock_app.reset_mock()
    send_mock.reset_mock()
    
    # Mock scope with valid auth header
    scope_with_auth = {
        "type": "http", 
        "method": "GET",
        "path": "/",
        "headers": [(b"authorization", b"Bearer secret")]
    }
    
    # Test authorized request
    await middleware(scope_with_auth, receive_mock, send_mock)
    
    # Should have called the app, not sent response directly
    mock_app.assert_called_once_with(scope_with_auth, receive_mock, send_mock)
    
    # Test non-HTTP scope (should pass through)
    scope_lifespan = {"type": "lifespan"}
    mock_app.reset_mock()
    
    await middleware(scope_lifespan, receive_mock, send_mock)
    mock_app.assert_called_once_with(scope_lifespan, receive_mock, send_mock)

