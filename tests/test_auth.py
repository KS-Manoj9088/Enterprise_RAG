"""
Unit tests for backend/auth.py — JWT creation, decoding, and user resolution.
Google OAuth calls are mocked so no real tokens are needed.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException


class TestCreateJwt:
    """Tests for create_jwt()."""

    def test_returns_string(self):
        from backend.auth import create_jwt
        token = create_jwt("user-123", "user@example.com")
        assert isinstance(token, str)
        assert len(token) > 0

    def test_different_users_produce_different_tokens(self):
        from backend.auth import create_jwt
        t1 = create_jwt("user-001", "a@example.com")
        t2 = create_jwt("user-002", "b@example.com")
        assert t1 != t2


class TestDecodeJwt:
    """Tests for decode_jwt()."""

    def test_decode_valid_token(self):
        from backend.auth import create_jwt, decode_jwt
        token = create_jwt("user-123", "user@example.com")
        payload = decode_jwt(token)
        assert payload["sub"] == "user-123"
        assert payload["email"] == "user@example.com"

    def test_invalid_token_raises_401(self):
        from backend.auth import decode_jwt
        with pytest.raises(HTTPException) as exc_info:
            decode_jwt("not.a.valid.token")
        assert exc_info.value.status_code == 401

    def test_tampered_token_raises_401(self):
        from backend.auth import create_jwt, decode_jwt
        token = create_jwt("user-123", "user@example.com")
        # Tamper the signature
        tampered = token[:-5] + "XXXXX"
        with pytest.raises(HTTPException) as exc_info:
            decode_jwt(tampered)
        assert exc_info.value.status_code == 401

    def test_empty_token_raises_401(self):
        from backend.auth import decode_jwt
        with pytest.raises(HTTPException) as exc_info:
            decode_jwt("")
        assert exc_info.value.status_code == 401


class TestVerifyGoogleToken:
    """Tests for verify_google_token() — Google API is always mocked."""

    def test_valid_token_returns_user_info(self):
        from backend.auth import verify_google_token
        mock_info = {
            "email": "user@gmail.com",
            "name": "Test User",
            "picture": "https://example.com/pic.jpg",
            "iss": "accounts.google.com",
        }
        with patch("backend.auth.id_token.verify_oauth2_token", return_value=mock_info):
            result = verify_google_token("fake-google-token")
        assert result["email"] == "user@gmail.com"
        assert result["name"] == "Test User"
        assert result["picture"] == "https://example.com/pic.jpg"

    def test_invalid_token_raises_401(self):
        from backend.auth import verify_google_token
        with patch("backend.auth.id_token.verify_oauth2_token",
                   side_effect=ValueError("Invalid token")):
            with pytest.raises(HTTPException) as exc_info:
                verify_google_token("bad-token")
            assert exc_info.value.status_code == 401

    def test_wrong_issuer_raises_401(self):
        from backend.auth import verify_google_token
        mock_info = {
            "email": "user@gmail.com",
            "name": "Test",
            "picture": "",
            "iss": "evil.com",  # wrong issuer
        }
        with patch("backend.auth.id_token.verify_oauth2_token", return_value=mock_info):
            with pytest.raises(HTTPException) as exc_info:
                verify_google_token("fake-token")
            assert exc_info.value.status_code == 401


class TestGetCurrentUser:
    """Tests for get_current_user() dependency."""

    def test_valid_token_returns_user(self, db_session, sample_user):
        from backend.auth import create_jwt, get_current_user
        from fastapi.security import HTTPAuthorizationCredentials

        token = create_jwt(sample_user.id, sample_user.email)
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        user = get_current_user(credentials=credentials, db=db_session)
        assert user.id == sample_user.id
        assert user.email == sample_user.email

    def test_token_for_nonexistent_user_raises_401(self, db_session):
        from backend.auth import create_jwt, get_current_user
        from fastapi.security import HTTPAuthorizationCredentials

        token = create_jwt("nonexistent-user-id", "ghost@example.com")
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        with pytest.raises(HTTPException) as exc_info:
            get_current_user(credentials=credentials, db=db_session)
        assert exc_info.value.status_code == 401
