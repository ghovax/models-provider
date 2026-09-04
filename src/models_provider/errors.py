"""Errors shared by the authentication modules."""


class AuthenticationError(RuntimeError):
    """Raised when a provider cannot authenticate a request or complete sign-in."""


class ContextWindowError(RuntimeError):
    """Raised when a provider rejects an input that exceeds the model context window."""

    def __init__(self, message: str, *, model: str = "", context_window: int = 0) -> None:
        super().__init__(message)
        self.model = model
        self.context_window = context_window
