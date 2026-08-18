"""
Error types and their HTTP mapping.

Every failure returns the same JSON shape so the frontend can render one error
component instead of guessing at response bodies:

    {"error": "not_found", "message": "...", "context": {...}, "request_id": "..."}
"""

from __future__ import annotations

from typing import Any


class ApiError(Exception):
    """Base class for errors that map to a deliberate HTTP response."""

    status_code = 500
    error_type = "internal_error"

    def __init__(self, message: str, **context: Any):
        super().__init__(message)
        self.message = message
        self.context = {k: v for k, v in context.items() if v is not None}

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"error": self.error_type,
                                   "message": self.message}
        if self.context:
            payload["context"] = self.context
        return payload


class NotFound(ApiError):
    status_code = 404
    error_type = "not_found"


class BadRequest(ApiError):
    status_code = 400
    error_type = "bad_request"


class ServiceUnavailable(ApiError):
    status_code = 503
    error_type = "service_unavailable"


class Conflict(ApiError):
    status_code = 409
    error_type = "conflict"


class NotImplementedYet(ApiError):
    status_code = 501
    error_type = "not_implemented"
