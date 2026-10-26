"""Exception hierarchy for indbot.

Every module raises one of these, never a bare Exception, so the engine can decide
what is fatal, what is retryable and what the user needs to see in the UI.
"""

from __future__ import annotations

from typing import Any


class IndbotError(Exception):
    """Base class for every error raised inside indbot."""

    #: Shown to the user in the UI. Subclasses override with something plainer.
    user_message: str = "Something went wrong."

    def __init__(self, message: str = "", **context: Any) -> None:
        super().__init__(message or self.user_message)
        self.message = message or self.user_message
        self.context = context

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, for the snapshot's `errors` list."""
        return {
            "type": type(self).__name__,
            "message": self.message,
            "context": {k: v for k, v in self.context.items() if _jsonable(v)},
        }


def _jsonable(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None), list, dict))


class ConfigError(IndbotError):
    """A setting is missing, malformed or out of range."""

    user_message = "Your configuration is not valid."


class AuthError(IndbotError):
    """No access token, or the broker rejected the one we have."""

    user_message = "INDstocks rejected the access token. Generate a new one."


class ApiError(IndbotError):
    """The API answered with a non-2xx status."""

    user_message = "The INDstocks API returned an error."

    def __init__(
        self,
        message: str = "",
        *,
        status: int | None = None,
        payload: Any = None,
        endpoint: str = "",
        **context: Any,
    ) -> None:
        super().__init__(message, endpoint=endpoint, **context)
        self.status = status
        self.payload = payload

    def as_dict(self) -> dict[str, Any]:
        d = super().as_dict()
        d["status"] = self.status
        if _jsonable(self.payload):
            d["payload"] = self.payload
        return d


class RateLimitError(ApiError):
    """429. The caller should back off; the client retries these itself."""

    user_message = "Hit the INDstocks rate limit. Backing off."


class OrderError(IndbotError):
    """An order was rejected, or did not reach a terminal state in time.

    Raised loudly on a failed EXIT: a stranded position is the worst outcome in
    this system, so the engine escalates these rather than retrying quietly.
    """

    user_message = "The broker rejected an order."

    def __init__(self, message: str = "", *, payload: Any = None, **context: Any) -> None:
        super().__init__(message, **context)
        self.payload = payload

    def as_dict(self) -> dict[str, Any]:
        d = super().as_dict()
        if _jsonable(self.payload):
            d["payload"] = self.payload
        return d


class DataError(IndbotError):
    """Market data was missing, stale or self-contradictory."""

    user_message = "Market data for this instrument is missing or stale."


class RiskBlocked(IndbotError):
    """The risk gate refused a trade. Not a failure — the system working."""

    user_message = "The risk gate blocked this trade."

    def __init__(self, message: str = "", *, reasons: list[str] | None = None, **context: Any) -> None:
        super().__init__(message, **context)
        self.reasons = reasons or []

    def as_dict(self) -> dict[str, Any]:
        d = super().as_dict()
        d["reasons"] = self.reasons
        return d
