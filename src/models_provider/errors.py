"""Errors shared by the authentication modules."""


class AuthenticationError(RuntimeError):
    """Raised when a provider cannot authenticate a request or complete sign-in."""
